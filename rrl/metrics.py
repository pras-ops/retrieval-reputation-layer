"""
RRL retrieval and adaptation metrics.

Two families live here:

* Ranking quality (`hit_at_k`, `mrr`, `ndcg_at_k`) — measures whether the right
  evidence was retrieved. These are the primary metrics: they are observed once per
  query, so a sweep yields thousands of observations instead of the handful of
  per-seed means a task-success rate gives you.
* Adaptation (`demotion_lag`, `post_feedback_performance`) — measures how fast a
  reputation layer reacts once evidence goes bad, which a static reranker cannot do
  at all and which an end-of-run average hides completely.

Note on naming: `demotion_lag` counts *observations* until a document stops being
selected. Feedback Adaptation for RAG (arXiv:2604.06647) uses "correction lag" for
the wall-clock latency until an updated index is queryable — a systems metric, not
this one. They are complementary; keep the names distinct.
"""

from typing import Dict, List, Optional, Sequence
import math


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 1) -> float:
    """1.0 if any relevant id appears in the top k, else 0.0."""
    rel = set(relevant_ids)
    return 1.0 if any(cid in rel for cid in list(ranked_ids)[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """1/rank of the first relevant id (1-indexed); 0.0 if none is ranked."""
    rel = set(relevant_ids)
    for i, cid in enumerate(ranked_ids):
        if cid in rel:
            return 1.0 / (i + 1)
    return 0.0


def mrr(rankings: Sequence[Sequence[str]], relevants: Sequence[Sequence[str]]) -> float:
    """Mean reciprocal rank over a set of queries."""
    if not rankings:
        return 0.0
    return sum(reciprocal_rank(r, g) for r, g in zip(rankings, relevants)) / len(rankings)


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: Dict[str, float],
    k: int = 10,
) -> float:
    """
    Normalized DCG@k with graded relevance. `relevance` maps candidate id -> gain;
    ids absent from the map score 0. Returns 0.0 when no gain is achievable.
    """
    top = list(ranked_ids)[:k]
    gains = [float(relevance.get(cid, 0.0)) for cid in top]
    ideal = sorted((v for v in relevance.values() if v > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg <= 0.0:
        return 0.0
    return dcg(gains) / idcg


def demotion_lag(
    selections: Sequence[str],
    target_id: str,
    became_bad_at: int,
    consecutive: int = 3,
) -> Optional[int]:
    """
    Observations from the moment `target_id` went bad until the system stops picking it.

    Requires `consecutive` successive non-selections before declaring the document
    demoted, so a single lucky exploration step is not mistaken for adaptation.
    Returns None if it never demotes within the trace, which is itself a result and
    must be reported rather than dropped from an average.
    """
    run = 0
    for i in range(became_bad_at, len(selections)):
        if selections[i] == target_id:
            run = 0
        else:
            run += 1
            if run >= consecutive:
                return (i - consecutive + 1) - became_bad_at
    return None


def post_feedback_performance(
    outcomes: Sequence[float],
    feedback_at: int,
    window: int = 30,
) -> Dict[str, float]:
    """
    Mean outcome in the `window` observations before vs after a feedback event.
    `delta` is the adaptation effect; `n_before`/`n_after` are reported so a thin
    window is visible instead of being read as a confident zero.
    """
    lo = max(0, feedback_at - window)
    before = list(outcomes[lo:feedback_at])
    after = list(outcomes[feedback_at : feedback_at + window])
    mb = sum(before) / len(before) if before else 0.0
    ma = sum(after) / len(after) if after else 0.0
    return {
        "before": mb,
        "after": ma,
        "delta": ma - mb,
        "n_before": float(len(before)),
        "n_after": float(len(after)),
    }


def diagnosticity(
    outcomes_when_correct: Sequence[float],
    outcomes_when_wrong: Sequence[float],
) -> Dict[str, float]:
    """
    Verifier diagnosticity Delta = P(pass | correct doc) - P(pass | wrong doc), plus
    the number of observations per document needed to separate the two at 80% power.

    Delta is what decides whether outcome feedback can identify a document at all:
    the required sample size scales as 1/Delta^2, so a weak verifier is not a small
    penalty, it is a different regime. Report this next to any pass-rate claim.
    """
    c = list(outcomes_when_correct)
    w = list(outcomes_when_wrong)
    if not c or not w:
        return {"p_correct": 0.0, "p_wrong": 0.0, "delta": 0.0, "n_required": float("inf")}
    p1 = sum(c) / len(c)
    p2 = sum(w) / len(w)
    d = p1 - p2
    if abs(d) < 1e-9:
        n_req = float("inf")
    else:
        z_a, z_b = 1.959964, 0.8416212
        n_req = ((z_a + z_b) ** 2) * (p1 * (1 - p1) + p2 * (1 - p2)) / (d**2)
    return {
        "p_correct": p1,
        "p_wrong": p2,
        "delta": d,
        "n_required": n_req,
        "n_correct": float(len(c)),
        "n_wrong": float(len(w)),
    }


def observations_required(delta: float, p_bar: float = 0.5) -> float:
    """
    Observations per document needed to separate a correct doc from a distractor at
    80% power / alpha=0.05, for a verifier with diagnosticity `delta`. `p_bar` is the
    working base rate used for the variance term.
    """
    if abs(delta) < 1e-9:
        return float("inf")
    z_a, z_b = 1.959964, 0.8416212
    return ((z_a + z_b) ** 2) * (2.0 * p_bar * (1.0 - p_bar)) / (delta**2)
