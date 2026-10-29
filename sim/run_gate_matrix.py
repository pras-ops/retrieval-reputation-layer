"""
Full-outcome-table benchmark (Gate F): a reproducible, leak-free replacement for Gate E.

Why this exists — three problems with sim/run_gate_recurring.py (Gate E):
  1. Answer-key leak : the corpus held each query's OWN reference solution, so the only
                       useful document was the answer itself (cached data: 78.8% pass when
                       retrieved, often copied verbatim, vs ~30% for any other document).
  2. Not replayable  : the cache only held the (query, doc) pairs one policy happened to
                       pick, so any code change that altered a pick caused cache misses.
  3. Weak baselines  : RRL was only compared to a web-passage cross-encoder.

What this harness does instead (the "replay" methodology of Li et al., WSDM 2011, with
full information):
  * Corpus  : reference solutions to OTHER problems in the same MBPP topic families. A
              query's own solution is never in the corpus (unless --leaky, which reproduces
              the Gate E design for comparison).
  * Pool    : every policy chooses 1 document from the same fixed top-5 hybrid-retrieval
              pool per query, so the outcome of every possible choice can be pre-computed.
  * Phase 1 : build — generate + unit-test every (query, pool doc) cell, plus a
              no-retrieval cell per query, into a JSONL outcome table.
  * Phase 2 : evaluate — replay every policy offline against that table. No LLM calls,
              fully deterministic, and any new policy can be added and re-run for free.
  * Stats   : per-seed pass rates, paired differences with 95% CIs, paired t-test and
              Wilcoxon signed-rank p-values (both arms run on the same seeds/streams).

Policies (all pick 1 of the same 5 candidates per step):
  no_retrieval   generate without any reference document
  static_rrf     top-1 hybrid (vector + BM25, RRF) retrieval, no learning
  static_ce      cross-encoder rerank of the pool, no learning (the Gate E baseline)
  query_memory   per-query "reuse what worked": walk the cross-encoder order until a
                 document passes for this exact query, keep it while it keeps passing
  query_memory_robust  the same, but keeping a document that has worked through an isolated
                 failure (moving on after two in a row, three if unverified)
  memrl_q        MemRL-style (arXiv 2601.03192): (1-λ)·z(sim) + λ·z(Q), Q ← Q + α(r − Q)
  rrl            ReputationLayer, Gate E configuration (Thompson exploration, decay)
  rrl_nodecay    same, γ = 1 (no decay)
  rrl_ce         rrl_nodecay, with the cross-encoder rank blended into the similarity input
  rrl_combined   rrl_ce + feedback-source reliability (rrl/reliability.py) + change
                 detection (rrl/drift.py): the combined score
  rrl_combined_ctx  rrl_combined with counters per request, shrunk toward each document's
                 global counters (ReputationLayer cluster_id)
  rrl_combined_ctx_verified  rrl_combined_ctx that keeps a tool for a request once the
                 verifier confirmed it worked there, until a verified failure
                 (ReputationLayer stick_to_verified)
  rrl_query      query-level Beta counters shrunk toward the document's global reputation
                 (hierarchical / empirical-Bayes), Thompson sampling, reliability-weighted
                 feedback, reranker-blended similarity
  oracle         best document in the pool for each query (upper bound, not a policy)

Scenarios (--scenario; outcomes always come from the same table, only the feedback a policy
sees and the timing of a usefulness shift change, so no new LLM calls are needed):
  stationary       verifier on every step, honest feedback
  sparse_verifier  verifier on 20% of steps; otherwise an honest user and an 80%-accurate judge
  sycophantic      verifier on 20% of steps; the user accepts 50% of answers regardless
  drift            verifier on every step; halfway through, each query's useful document
                   moves to another document in its pool (the pool's outcomes are rotated)

Usage:
    python3 sim/run_gate_matrix.py --selftest                      # offline plumbing check
    python3 sim/run_gate_matrix.py --generator hf --build          # fill table, local model
    USE_REAL_GEMINI=true GCP_PROJECT_ID=... \\
        python3 sim/run_gate_matrix.py --generator gemini --build  # fill table, Gemini
    python3 sim/run_gate_matrix.py --generator hf                  # evaluate from table only
    python3 sim/run_gate_matrix.py --generator gemini --scenario all
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rrl.drift import ChangeDetector
from rrl.feedback import OutcomeSignals, calculate_outcome
from rrl.layer import ReputationLayer
from rrl.reliability import SourceReliability
from rrl.store import CandidateStore
from run_gate_recurring import entry_point, load_mbpp, run_tests, topic

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
RESULTS_DIR = os.path.join(HERE, "results")

N_QUERY = 30
N_DOCS = 60
POOL_K = 5
N_FAMILIES = 8
NO_DOC = "NONE"

# Exact Hugging Face revisions used for the recorded results.
HF_REVISIONS = {"Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775"}

POLICIES = [
    "no_retrieval",
    "static_rrf",
    "static_ce",
    "query_memory",
    "query_memory_robust",
    "memrl_q",
    "rrl",
    "rrl_nodecay",
    "rrl_ce",
    "rrl_combined",
    "rrl_combined_ctx",
    "rrl_combined_ctx_verified",
    "rrl_query",
    "oracle",
]

# Ablations of rrl_combined, run with --ablation. Each is rrl_combined with one change.
ABLATIONS = [
    "rrl_combined_globalrel",  # source reliability without per-document rates
    "rrl_combined_nocd",  # no change detection
    "rrl_combined_cdlong",  # change detection also resets the long-term counters
    "rrl_combined_cdold",  # the original detector: window 4, fixed 0.6 threshold
    "rrl_combined_conf",  # confidence-gated exploration, without per-context counters
    "rrl_combined_ctx_conf",  # per-context counters + confidence-gated exploration
    "rrl_combined_e10",  # exploration weight 0.10 (the library default)
    "rrl_combined_e0",  # exploration off
]

# Weights used by every RRL variant in the benchmarks: (sim, recent, long-term, explore).
# This is the Gate E configuration; ReputationLayer's library default is (0.55, 0.30, 0.15, 0.10).
BENCH_WEIGHTS = (0.20, 0.40, 0.10, 0.30)

# Policies replayed by run_policies(); --ablation appends ABLATIONS.
RUN_POLICIES: List[str] = list(POLICIES)


def enable_ablation() -> None:
    for name in ABLATIONS:
        if name not in RUN_POLICIES:
            RUN_POLICIES.insert(RUN_POLICIES.index("oracle"), name)


# Change-detector resets per (policy, seed), filled by the RRL policies during a replay.
DIAGNOSTICS: Dict[str, Dict[int, int]] = defaultdict(dict)

LEARNERS = [
    "query_memory",
    "memrl_q",
    "rrl",
    "rrl_nodecay",
    "rrl_ce",
    "rrl_combined",
    "rrl_query",
]

# verifier coverage, probability the user accepts regardless of the outcome, judge accuracy
SCENARIOS: Dict[str, dict] = {
    "stationary": {"coverage": 1.0, "sycophancy": 0.0, "judge_acc": 0.8, "drift": False},
    "sparse_verifier": {"coverage": 0.2, "sycophancy": 0.0, "judge_acc": 0.8, "drift": False},
    "sycophantic": {"coverage": 0.2, "sycophancy": 0.5, "judge_acc": 0.8, "drift": False},
    "drift": {"coverage": 1.0, "sycophancy": 0.0, "judge_acc": 0.8, "drift": True},
}


# ---------------------------------------------------------------- dataset


def build_dataset(seed: int, leaky: bool = False) -> Tuple[List[dict], Dict[int, dict]]:
    """
    Query problems and a corpus of reference solutions to OTHER problems from the same
    topic families. With leaky=True each query's own solution is added (the Gate E design).
    Returns (query_problems, corpus) where corpus maps task_id -> problem.
    """
    rng = random.Random(seed)
    by_family: Dict[str, List[dict]] = defaultdict(list)
    for r in load_mbpp():
        if r.get("test_list") and entry_point(r):
            by_family[topic(r["text"])].append(r)
    families = sorted(by_family, key=lambda f: (-len(by_family[f]), f))[:N_FAMILIES]
    pool = [p for f in families for p in by_family[f]]
    rng.shuffle(pool)

    queries = pool[:N_QUERY]
    corpus = {p["task_id"]: p for p in pool[N_QUERY : N_QUERY + N_DOCS]}
    if leaky:
        corpus.update({p["task_id"]: p for p in queries})
    return queries, corpus


class PoolBuilder:
    """Fixed top-K hybrid retrieval pools, plus cross-encoder order, per query."""

    def __init__(self, embedder=None, cross_encoder=None):
        self.embedder = embedder
        self.cross_encoder = cross_encoder

    def build(self, queries: List[dict], corpus: Dict[int, dict]):
        from rrl.ingest import Ingester
        from rrl.retriever import Retriever

        store = CandidateStore()
        ingester = Ingester(model=self.embedder) if self.embedder is not None else Ingester()
        for tid, p in corpus.items():
            ingester.ingest_document(store, f"mbpp{tid}", p["code"])
        # Pure similarity: no reputation terms, no exploration.
        retriever = Retriever(store, weights=(1.0, 0.0, 0.0, 0.0), model=ingester.model)

        pools: Dict[int, List[Tuple[int, float]]] = {}
        ce_order: Dict[int, List[int]] = {}
        for q in queries:
            # Long solutions are split into several chunks; keep each document once.
            res = retriever.retrieve(q["text"], top_k=POOL_K * 3, explore=False, epsilon=0.0)
            pool: List[Tuple[int, float]] = []
            for c, _score, sim in res:
                d = _doc_task_id(c.id)
                if d not in {x for x, _ in pool}:
                    pool.append((d, sim))
            pool = pool[:POOL_K]
            pools[q["task_id"]] = pool
            if self.cross_encoder is not None:
                pairs = [(q["text"], corpus[d]["code"]) for d, _ in pool]
                scores = list(self.cross_encoder.predict(pairs))
                order = sorted(range(len(pool)), key=lambda i: -scores[i])
                ce_order[q["task_id"]] = [pool[i][0] for i in order]
            else:
                ce_order[q["task_id"]] = [d for d, _ in pool]
        return pools, ce_order


def _doc_task_id(candidate_id: str) -> int:
    m = re.match(r"mbpp(\d+)", candidate_id)
    if not m:
        raise ValueError(f"unexpected candidate id {candidate_id}")
    return int(m.group(1))


# ---------------------------------------------------------------- generators


def _strip_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return text.replace("```python", "").replace("```", "").strip()


def _prompt(problem: dict, reference: Optional[str]) -> str:
    head = "You are a Python coding assistant. "
    if reference is not None:
        head += (
            "Using the reference example only if it is relevant, write a correct solution "
            "to the problem."
        )
    else:
        head += "Write a correct solution to the problem."
    body = " Return ONLY the function definition, no markdown, no commentary.\n\n"
    if reference is not None:
        body += f"Reference example:\n{reference}\n\n"
    body += f"Problem: {problem['text']}\nThe function must be named `{entry_point(problem)}`.\n"
    return head + body


class GeminiGenerator:
    tag = "gemini-2.5-flash"

    def __init__(self, workers: int = 8):
        from rrl.judge import _get_client

        self.workers = workers

        try:
            from google.genai import types
        except ImportError as e:
            raise SystemExit(f"google-genai not installed: {e}")
        self.types = types
        self.client = _get_client()
        if self.client is None:
            raise SystemExit("No Gemini client: set GCP_PROJECT_ID (Vertex) or GEMINI_API_KEY.")

    def _one(self, prompt: str) -> Optional[str]:
        import time

        for attempt in range(4):
            try:
                resp = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=0.0,
                        # A hung request otherwise blocks the whole build.
                        http_options=self.types.HttpOptions(timeout=90_000),
                    ),
                )
                return _strip_code(resp.text or "")
            except Exception as e:
                print(f"[gen warning] Gemini failed (attempt {attempt + 1}/4): {e}", flush=True)
                time.sleep(2 ** (attempt + 1))
        return None

    def generate(self, prompts: List[str]) -> List[Optional[str]]:
        """None marks a failed call: it is not recorded, so the next --build retries it."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._one, prompts))


class HFGenerator:
    """Local Hugging Face chat model, greedy decoding, batched."""

    def __init__(self, model_name: str, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tag = "hf-" + model_name.split("/")[-1]
        self.batch_size = batch_size
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Pinned so a changed upstream repository cannot silently change the benchmark.
        rev = HF_REVISIONS.get(model_name)
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left", revision=rev)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, revision=rev
        )
        self.model.to(self.device).eval()

    def generate(self, prompts: List[str]) -> List[str]:
        out: List[str] = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            texts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in chunk
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=384,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                )
            for row in gen[:, enc["input_ids"].shape[1] :]:
                out.append(_strip_code(self.tok.decode(row, skip_special_tokens=True)))
        return out


class MockGenerator:
    """Plumbing only: passes iff the reference IS the problem's own solution."""

    tag = "mock"

    def __init__(self, problems_by_id: Dict[int, dict]):
        self.by_code = {p["code"]: p for p in problems_by_id.values()}

    def generate(self, prompts: List[str]) -> List[str]:
        out = []
        for p in prompts:
            m = re.search(r"Reference example:\n(.*?)\n\nProblem:", p, re.S)
            ref = m.group(1) if m else None
            out.append(ref if ref is not None else "    pass\n")
        return out


# ---------------------------------------------------------------- outcome table


class OutcomeTable:
    """JSONL table of (task_id, doc) -> passed for one generator. Append-only."""

    def __init__(self, tag: str, path: Optional[str] = None):
        self.path = path or os.path.join(DATA_DIR, f"matrix_outcomes_{tag}.jsonl")
        self.cells: Dict[Tuple[int, str], float] = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        # Rows written before the "error" field existed stored a failed
                        # API call as an empty completion; treat those as missing.
                        if "error" not in r and not r.get("completion", "").strip():
                            continue
                        self.cells[(int(r["task_id"]), str(r["doc"]))] = float(r["passed"])

    def get(self, task_id: int, doc) -> float:
        key = (task_id, str(doc))
        if key not in self.cells:
            raise KeyError(
                f"outcome table {os.path.basename(self.path)} has no cell {key}; "
                "run with --build first"
            )
        return self.cells[key]

    def missing(self, needed: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        return [k for k in dict.fromkeys(needed) if k not in self.cells]

    def fill(self, needed, problems: Dict[int, dict], gen, chunk: int = 64) -> None:
        todo = self.missing(needed)
        print(f"[build] {len(todo)} cells to generate ({len(self.cells)} already in table)")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        for i in range(0, len(todo), chunk):
            part = todo[i : i + chunk]
            prompts = [
                _prompt(problems[t], None if d == NO_DOC else problems[int(d)]["code"])
                for t, d in part
            ]
            completions = gen.generate(prompts)
            failed = 0
            with open(self.path, "a") as f:
                for (t, d), comp in zip(part, completions):
                    if comp is None:
                        failed += 1
                        continue
                    passed = run_tests(problems[t], comp)
                    self.cells[(t, d)] = passed
                    row = {"task_id": t, "doc": d, "passed": passed, "completion": comp}
                    f.write(json.dumps({**row, "error": False}) + "\n")
            note = f" ({failed} failed calls left for the next --build)" if failed else ""
            print(f"[build] {min(i + chunk, len(todo))}/{len(todo)}{note}", flush=True)


# ---------------------------------------------------------------- policies


Pool = List[Tuple[int, float]]  # (doc task_id, normalized hybrid sim)
Policy = Tuple[
    Callable[[int, Pool, int], object], Callable[[int, object, OutcomeSignals, int], None]
]


def naive_reward(signals: OutcomeSignals) -> float:
    """
    What a learner without RRL's safeguards sees: the verifier when it ran, and otherwise the
    other signals at face value (a plain weighted mean, no cap and no per-source trust).
    """
    if signals.s_gt is not None:
        return signals.s_gt
    y = calculate_outcome(signals, use_safeguards=False)
    return 0.5 if y is None else y


def make_policy(name: str, seed: int, ce_order: Dict[int, List[int]]) -> Policy:
    """Returns (choose(task_id, pool, step) -> doc, update(task_id, doc, signals, step))."""
    noop = lambda *a: None  # noqa: E731

    if name == "no_retrieval":
        return (lambda t, pool, step: NO_DOC), noop

    if name == "static_rrf":
        return (lambda t, pool, step: pool[0][0]), noop

    if name == "static_ce":
        return (lambda t, pool, step: ce_order[t][0]), noop

    if name == "query_memory":
        idx: Dict[int, int] = defaultdict(int)

        def choose_qm(t, pool, step):
            return ce_order[t][idx[t]]

        def update_qm(t, doc, signals, step):
            # Keep a document while it works; move to the next one when it fails.
            if naive_reward(signals) <= 0.5:
                idx[t] = (idx[t] + 1) % len(ce_order[t])

        return choose_qm, update_qm

    if name == "query_memory_robust":
        # The same idea, robust to transient failures: search exactly like query_memory, but
        # once a tool has worked for this request keep it through an isolated failure, moving
        # on only after two failures in a row (three if none of them was verified).
        idx_r: Dict[int, int] = defaultdict(int)
        fails: Dict[int, int] = defaultdict(int)
        locked: Dict[int, bool] = defaultdict(bool)

        def choose_qmr(t, pool, step):
            return ce_order[t][idx_r[t]]

        def update_qmr(t, doc, signals, step):
            if naive_reward(signals) > 0.5:
                fails[t], locked[t] = 0, True
                return
            fails[t] += 1
            verified = signals.s_gt is not None
            if locked[t]:
                move = (verified and fails[t] >= 2) or fails[t] >= 3
            else:
                move = True
            if move:
                idx_r[t] = (idx_r[t] + 1) % len(ce_order[t])
                fails[t], locked[t] = 0, False

        return choose_qmr, update_qmr

    if name == "memrl_q":
        lam, alpha, q_init = 0.5, 0.3, 0.5
        q: Dict[int, float] = defaultdict(lambda: q_init)

        def z(xs):
            m = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
            return [(x - m) / sd if sd > 0 else 0.0 for x in xs]

        def choose_mq(t, pool, step):
            zs, zq = z([s for _, s in pool]), z([q[d] for d, _ in pool])
            scores = [(1 - lam) * a + lam * b for a, b in zip(zs, zq)]
            return pool[max(range(len(pool)), key=lambda i: scores[i])][0]

        def update_mq(t, doc, signals, step):
            q[doc] += alpha * (naive_reward(signals) - q[doc])

        return choose_mq, update_mq

    if (
        name in ("rrl", "rrl_nodecay", "rrl_ce", "rrl_combined", "rrl_combined_ctx")
        or name == "rrl_combined_ctx_verified"
        or name in ABLATIONS
    ):
        combined = name.startswith("rrl_combined")
        w_explore = {"rrl_combined_e10": 0.10, "rrl_combined_e0": 0.0}.get(name, BENCH_WEIGHTS[3])
        if not combined or name == "rrl_combined_nocd":
            detector = None
        elif name == "rrl_combined_cdold":
            detector = ChangeDetector(window=4, threshold=0.6)
        else:
            detector = ChangeDetector(reset_long_term=name == "rrl_combined_cdlong")
        # rrl_combined_ctx keeps counters per request (shrunk toward each tool's global ones);
        # *_conf stops exploring for a request once its leader is clear (95%); *_verified keeps
        # a tool for a request once the verifier confirmed it worked there.
        per_context = name in (
            "rrl_combined_ctx",
            "rrl_combined_ctx_verified",
            "rrl_combined_ctx_conf",
        )
        layer = ReputationLayer(
            CandidateStore(),
            weights=BENCH_WEIGHTS[:3] + (w_explore,),
            gamma=0.95 if name == "rrl" else 1.0,
            decay_unit_sec=1.0,
            reliability=(
                SourceReliability(doc_strength=None if name == "rrl_combined_globalrel" else 4.0)
                if combined
                else None
            ),
            change_detector=detector,
            confident_greedy=0.95 if name.endswith("_conf") else None,
            stick_to_verified=name == "rrl_combined_ctx_verified",
        )
        blend_ce = name != "rrl" and name != "rrl_nodecay"
        explore = w_explore > 0
        rng_state = {"state": random.Random(seed).getstate()}
        pending: Dict[int, str] = {}

        def choose_rrl(t, pool, step):
            sims = {d: s for d, s in pool}
            if blend_ce:
                # Reranker as one more input: its rank score (best = 1, worst = 0),
                # averaged with the retriever's similarity.
                order = ce_order[t]
                rank_score = {d: 1.0 - i / max(1, len(order) - 1) for i, d in enumerate(order)}
                sims = {d: 0.5 * s + 0.5 * rank_score[d] for d, s in sims.items()}
            random.setstate(rng_state["state"])
            res = layer.rescore(
                {f"mbpp{d}": s for d, s in sims.items()},
                top_k=1,
                explore=explore,
                epsilon=0.0,
                now=float(step),
                cluster_id=f"q{t}" if per_context else None,
            )
            rng_state["state"] = random.getstate()
            pending[t] = res.response_id
            return _doc_task_id(res.results[0][0])

        def update_rrl(t, doc, signals, step):
            layer.record_feedback(
                pending.pop(t),
                s_gt=signals.s_gt,
                s_behave=signals.s_behave,
                s_judge=signals.s_judge,
                now=float(step),
            )
            if detector is not None:
                DIAGNOSTICS[name][seed] = detector.resets

        return choose_rrl, update_rrl

    if name == "rrl_query":
        # Per-(query, doc) evidence, shrunk toward the doc's global success rate with
        # prior strength k: a new query starts from the doc's reputation, and its own
        # outcomes take over as they accumulate.
        k, w_sim = 2.0, 0.3
        rel = SourceReliability()
        doc_s: Dict[int, float] = defaultdict(float)
        doc_n: Dict[int, float] = defaultdict(float)
        qd_s: Dict[Tuple[int, int], float] = defaultdict(float)
        qd_n: Dict[Tuple[int, int], float] = defaultdict(float)
        rng = random.Random(seed)

        def blended(t, pool):
            order = ce_order[t]
            rank = {d: 1.0 - i / max(1, len(order) - 1) for i, d in enumerate(order)}
            return {d: 0.5 * s + 0.5 * rank[d] for d, s in pool}

        def choose_rq(t, pool, step):
            best, best_score = None, -1.0
            for d, sim in blended(t, pool).items():
                mu = (doc_s[d] + 1.0) / (doc_n[d] + 2.0)
                a = k * mu + qd_s[(t, d)]
                b = k * (1.0 - mu) + qd_n[(t, d)] - qd_s[(t, d)]
                score = w_sim * sim + (1.0 - w_sim) * rng.betavariate(a, b)
                if score > best_score:
                    best, best_score = d, score
            return best

        def update_rq(t, doc, signals, step):
            rel.observe(signals, [str(doc)])
            y = rel.combine(signals, str(doc))
            if y is None:
                return
            doc_s[doc] += y
            doc_n[doc] += 1.0
            qd_s[(t, doc)] += y
            qd_n[(t, doc)] += 1.0

        return choose_rq, update_rq

    raise ValueError(name)


# ---------------------------------------------------------------- stats


def _t975(n: int) -> float:
    try:
        from scipy import stats as st

        return float(st.t.ppf(0.975, n - 1))
    except Exception:
        return 2.045 if n >= 30 else 2.262


def mean_ci(xs: List[float]) -> Tuple[float, float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, m, m
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = _t975(n) * sd / math.sqrt(n)
    return m, m - h, m + h


def paired(a: List[float], b: List[float]) -> dict:
    """Paired comparison of b vs a (b − a) across seeds."""
    d = [y - x for x, y in zip(a, b)]
    m, lo, hi = mean_ci(d)
    out = {
        "mean_diff": m,
        "ci95": [lo, hi],
        "wins": sum(x > 0 for x in d),
        "losses": sum(x < 0 for x in d),
        "ties": sum(x == 0 for x in d),
    }
    try:
        from scipy import stats as st

        out["t_test_p"] = float(st.ttest_rel(b, a).pvalue) if any(d) else 1.0
        out["wilcoxon_p"] = float(st.wilcoxon(b, a).pvalue) if any(d) else 1.0
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- run


def make_feedback(scenario: dict, seed: int, n_steps: int):
    """
    Pre-sampled feedback noise, shared by every policy on a seed, so policies differ only
    in what they choose. Returns signals(step, true_outcome, doc) -> OutcomeSignals.

    scenario["fooled_docs"] / scenario["fooled_prob"] (optional): documents whose answers the
    user accepts with that probability regardless of the outcome — errors that depend on which
    document was used, unlike the uniform `sycophancy`.
    """
    rng = random.Random(seed * 7919 + 1)
    draws = [(rng.random(), rng.random(), rng.random()) for _ in range(n_steps)]

    fooled_docs = scenario.get("fooled_docs", frozenset())
    fooled_prob = scenario.get("fooled_prob", 0.0)

    def signals(step: int, r: float, doc=None) -> OutcomeSignals:
        u_cov, u_syc, u_judge = draws[step]
        ok = r > 0.5
        if doc in fooled_docs:
            user_accepts = True if u_syc < fooled_prob else ok
        else:
            user_accepts = True if u_syc < scenario["sycophancy"] else ok
        judge_says = ok if u_judge < scenario["judge_acc"] else not ok
        return OutcomeSignals(
            s_gt=r if u_cov < scenario["coverage"] else None,
            s_behave=0.75 if user_accepts else 0.10,
            s_judge=1.0 if judge_says else 0.0,
        )

    return signals


def run_seed(
    seed, epochs, table, pools, ce_order, scenario=SCENARIOS["stationary"]
) -> Dict[str, List[float]]:
    queries = list(pools)
    rng = random.Random(seed)
    stream: List[int] = []
    for _ in range(epochs):
        order = list(queries)
        rng.shuffle(order)
        stream.extend(order)
    shift_at = len(stream) // 2 if scenario["drift"] else len(stream)

    def outcome(t: int, doc, step: int) -> float:
        if step < shift_at or doc == NO_DOC:
            return table.get(t, doc)
        # After the shift, pool position k behaves like position k + 1 did before.
        ids = [d for d, _ in pools[t]]
        return table.get(t, ids[(ids.index(doc) + 1) % len(ids)])

    return run_policies(seed, stream, pools, ce_order, outcome, scenario)


def run_policies(
    seed: int,
    stream: List[int],
    pools: Dict[int, Pool],
    ce_order: Dict[int, List[int]],
    outcome: Callable[[int, object, int], float],
    scenario: dict,
    policies: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """
    Replays every policy over one query stream. outcome(task, doc, step) -> reward.
    scenario["signals_fn"](step, task, doc, reward) -> OutcomeSignals, if given, replaces the
    simulated feedback (for benchmarks whose feedback comes from the tools themselves).
    """
    signals = make_feedback(scenario, seed, len(stream))
    histories: Dict[str, List[float]] = {}
    for name in policies if policies is not None else RUN_POLICIES:
        if name == "oracle":
            histories[name] = [
                max(outcome(t, d, step) for d, _ in pools[t]) for step, t in enumerate(stream)
            ]
            continue
        choose, update = make_policy(name, seed, ce_order)
        h = []
        for step, t in enumerate(stream):
            doc = choose(t, pools[t], step)
            r = outcome(t, doc, step)
            sig = (
                scenario["signals_fn"](step, t, doc, r)
                if "signals_fn" in scenario
                else signals(step, r, doc)
            )
            update(t, doc, sig, step)
            h.append(r)
        histories[name] = h
    return histories


def needed_cells(pools) -> List[Tuple[int, str]]:
    cells = []
    for t, pool in pools.items():
        cells.append((t, NO_DOC))
        cells.extend((t, str(d)) for d, _ in pool)
    return cells


def load_models(offline_mock: bool):
    if offline_mock:
        return _MockEmbedder(), None
    from sentence_transformers import CrossEncoder, SentenceTransformer

    # CPU: GPU/MPS kernels are not bit-reproducible, which can flip near-tied pool members.
    return (
        SentenceTransformer("all-MiniLM-L6-v2", device="cpu"),
        CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"),
    )


def load_or_build_pools(seeds: List[int], leaky: bool, embedder, cross_encoder) -> dict:
    """
    The candidate pools are part of the benchmark definition, so they are frozen to
    data/gate_f_pools_<variant>.json the first time they are built and reused afterwards.
    Seeds missing from the file are built and added.
    """
    path = os.path.join(DATA_DIR, f"gate_f_pools_{'leaky' if leaky else 'leakfree'}.json")
    frozen: dict = {}
    if os.path.exists(path):
        with open(path) as f:
            frozen = json.load(f)
    datasets = {}
    changed = False
    for s in seeds:
        if str(s) not in frozen:
            if embedder is None:
                embedder, cross_encoder = load_models(offline_mock=False)
            q, corpus = build_dataset(s, leaky=leaky)
            pools, ce_order = PoolBuilder(embedder, cross_encoder).build(q, corpus)
            frozen[str(s)] = {
                "pools": {str(t): [[d, sim] for d, sim in p] for t, p in pools.items()},
                "ce_order": {str(t): o for t, o in ce_order.items()},
            }
            changed = True
        entry = frozen[str(s)]
        datasets[s] = (
            {int(t): [(int(d), float(sim)) for d, sim in p] for t, p in entry["pools"].items()},
            {int(t): [int(d) for d in o] for t, o in entry["ce_order"].items()},
        )
    if changed:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(frozen, f)
    return datasets


class _MockEmbedder:
    """Deterministic bag-of-words hashing embedder (self-test only; no downloads)."""

    def encode(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        vecs = []
        for t in texts:
            v = [0.0] * 64
            for w in re.findall(r"[a-z]+", t.lower()):
                v[sum(map(ord, w)) % 64] += 1.0
            vecs.append(_Vec(v))
        return vecs[0] if single else vecs


class _Vec(list):
    def tolist(self):
        return list(self)


def summarize(per_seed: Dict[str, Dict[str, List[float]]], seeds: List[int]) -> dict:
    late_frac = 0.2
    names = list(per_seed[seeds[0]])
    overall: Dict[str, List[float]] = {p: [] for p in names}
    late: Dict[str, List[float]] = {p: [] for p in names}
    for s in seeds:
        for p in names:
            h = per_seed[s][p]
            k = max(1, int(len(h) * late_frac))
            overall[p].append(sum(h) / len(h))
            late[p].append(sum(h[-k:]) / k)

    comparisons = {}
    for metric, data in (("overall", overall), ("late", late)):
        comparisons[metric] = {
            "static_ce_vs_no_retrieval": paired(data["no_retrieval"], data["static_ce"]),
            "query_memory_vs_static_ce": paired(data["static_ce"], data["query_memory"]),
            "rrl_vs_static_ce": paired(data["static_ce"], data["rrl"]),
            "rrl_nodecay_vs_rrl": paired(data["rrl"], data["rrl_nodecay"]),
            "rrl_ce_vs_rrl_nodecay": paired(data["rrl_nodecay"], data["rrl_ce"]),
            "rrl_combined_vs_rrl_ce": paired(data["rrl_ce"], data["rrl_combined"]),
            "rrl_combined_vs_static_ce": paired(data["static_ce"], data["rrl_combined"]),
            "rrl_combined_vs_query_memory": paired(data["query_memory"], data["rrl_combined"]),
            "rrl_combined_vs_memrl_q": paired(data["memrl_q"], data["rrl_combined"]),
            "rrl_query_vs_static_ce": paired(data["static_ce"], data["rrl_query"]),
            "rrl_query_vs_query_memory": paired(data["query_memory"], data["rrl_query"]),
            "rrl_query_vs_memrl_q": paired(data["memrl_q"], data["rrl_query"]),
            "rrl_query_vs_rrl_combined": paired(data["rrl_combined"], data["rrl_query"]),
            "rrl_combined_ctx_vs_rrl_combined": paired(
                data["rrl_combined"], data["rrl_combined_ctx"]
            ),
            "rrl_combined_ctx_vs_memrl_q": paired(data["memrl_q"], data["rrl_combined_ctx"]),
            "rrl_combined_ctx_vs_query_memory": paired(
                data["query_memory"], data["rrl_combined_ctx"]
            ),
            "rrl_combined_ctx_verified_vs_rrl_combined_ctx": paired(
                data["rrl_combined_ctx"], data["rrl_combined_ctx_verified"]
            ),
            "rrl_combined_ctx_verified_vs_query_memory": paired(
                data["query_memory"], data["rrl_combined_ctx_verified"]
            ),
            "rrl_combined_ctx_verified_vs_query_memory_robust": paired(
                data["query_memory_robust"], data["rrl_combined_ctx_verified"]
            ),
            "query_memory_robust_vs_query_memory": paired(
                data["query_memory"], data["query_memory_robust"]
            ),
            "rrl_combined_ctx_verified_vs_memrl_q": paired(
                data["memrl_q"], data["rrl_combined_ctx_verified"]
            ),
        }
        for a in ABLATIONS:
            if a in data:
                comparisons[metric][f"{a}_vs_rrl_combined"] = paired(data["rrl_combined"], data[a])
    resets = {
        p: mean_ci([float(DIAGNOSTICS[p].get(s, 0)) for s in seeds])
        for p in names
        if p in DIAGNOSTICS
    }
    return {
        "overall": {p: mean_ci(v) for p, v in overall.items()},
        "late": {p: mean_ci(v) for p, v in late.items()},
        "per_seed_overall": overall,
        "comparisons": comparisons,
        "change_detector_resets": resets,
    }


def print_report(summary: dict, title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'policy':<24} | {'overall pass rate':<26} | {'late (last 20%)':<26}")
    print("-" * 78)
    for p in summary["overall"]:
        o, lt = summary["overall"][p], summary["late"][p]
        print(
            f"{p:<24} | {o[0]:.3f} [{o[1]:.3f}, {o[2]:.3f}]     "
            f"| {lt[0]:.3f} [{lt[1]:.3f}, {lt[2]:.3f}]"
        )
    if summary.get("change_detector_resets"):
        print("\nChange-detector resets per run (mean over seeds):")
        for p, (m, lo, hi) in summary["change_detector_resets"].items():
            print(f"  {p:<24} {m:.1f} [{lo:.1f}, {hi:.1f}]")
    for metric in ("overall", "late"):
        print(f"\nPaired comparisons ({metric}); diff = second − first arm, 95% CI:")
        for name, c in summary["comparisons"][metric].items():
            p_t = c.get("t_test_p", float("nan"))
            p_w = c.get("wilcoxon_p", float("nan"))
            print(
                f"  {name:<40} {c['mean_diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
                f"  W/L/T {c['wins']}/{c['losses']}/{c['ties']}  t p={p_t:.3g}  wilcoxon p={p_w:.3g}"
            )


def plot(per_seed, seeds, path, title="Gate F (MBPP, leak-free, full outcome table)") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    shown = [
        "static_ce",
        "llm_picks",
        "query_memory",
        "memrl_q",
        "rrl_combined",
        "rrl_combined_ctx_verified",
        "oracle",
    ]
    plt.figure(figsize=(10, 6))
    for p in [x for x in shown if x in per_seed[seeds[0]]]:
        n = len(per_seed[seeds[0]][p])
        curve = [sum(per_seed[s][p][i] for s in seeds) / len(seeds) for i in range(n)]
        w = 30
        smooth = [
            sum(curve[max(0, i - w + 1) : i + 1]) / len(curve[max(0, i - w + 1) : i + 1])
            for i in range(n)
        ]
        style = "--" if p in ("oracle", "no_retrieval") else "-"
        plt.plot(smooth, style, label=p, linewidth=2)
    plt.xlabel("Query step")
    plt.ylabel("Unit-test pass rate (moving average, mean over seeds)")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")


def selftest() -> None:
    """Offline: mock embedder + mock generator, checks plumbing and determinism."""
    import tempfile

    print("SELF-TEST (offline): leak-free corpus, outcome table, replay determinism.")
    problems = {p["task_id"]: p for p in load_mbpp()}
    for leaky in (False, True):
        q, corpus = build_dataset(seed=42, leaky=leaky)
        own = sum(p["task_id"] in corpus for p in q)
        print(
            f"  leaky={leaky}: {len(q)} queries, {len(corpus)} docs, own solutions in corpus={own}"
        )
        assert (own == len(q)) if leaky else (own == 0)

    embedder, _ = load_models(offline_mock=True)
    with tempfile.TemporaryDirectory() as d:
        table = OutcomeTable("mock", path=os.path.join(d, "t.jsonl"))
        gen = MockGenerator(problems)
        results = []
        for _ in range(2):
            per_seed = {}
            for s in (42, 43):
                q, corpus = build_dataset(s, leaky=True)
                pools, ce = PoolBuilder(embedder=embedder).build(q, corpus)
                table.fill(needed_cells(pools), problems, gen)
                per_seed[s] = run_seed(s, 3, table, pools, ce)
                for sc in SCENARIOS.values():
                    run_seed(s, 2, table, pools, ce, sc)
            results.append(per_seed)
        assert results[0] == results[1], "replay is not deterministic"
        reloaded = OutcomeTable("mock", path=table.path)
        assert reloaded.cells == table.cells, "outcome table did not round-trip"
        summ = summarize(results[0], [42, 43])
    print(
        f"  oracle >= every policy: {all(summ['overall']['oracle'][0] >= summ['overall'][p][0] for p in summ['overall'])}"
    )
    print("SELF-TEST OK (deterministic replay, no cache misses)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--generator", choices=["gemini", "hf"], default="hf")
    ap.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batch-size", type=int, default=16, help="hf generator batch size")
    ap.add_argument("--workers", type=int, default=8, help="parallel Gemini calls")
    ap.add_argument("--build", action="store_true", help="generate missing outcome cells first")
    ap.add_argument("--leaky", action="store_true", help="Gate E design: own solution in corpus")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--scenario", default="stationary", choices=sorted(SCENARIOS) + ["all"])
    ap.add_argument("--ablation", action="store_true", help="also replay the ABLATIONS")
    ap.add_argument(
        "--timeouts-as-failures",
        action="store_true",
        help="record cells whose generation still fails after --build as failed (error=timeout)",
    )
    args = ap.parse_args()
    if args.ablation:
        enable_ablation()

    if args.selftest:
        selftest()
        return

    problems = {p["task_id"]: p for p in load_mbpp()}
    seeds = list(range(42, 42 + args.seeds))

    datasets = load_or_build_pools(seeds, args.leaky, None, None)

    if args.generator == "gemini":
        if args.build and os.getenv("USE_REAL_GEMINI", "false").lower() != "true":
            sys.exit("Gemini generation requires USE_REAL_GEMINI=true (it makes paid API calls).")
        tag = GeminiGenerator.tag
    else:
        tag = "hf-" + args.hf_model.split("/")[-1]
    table = OutcomeTable(tag)

    needed = [c for s in seeds for c in needed_cells(datasets[s][0])]
    if args.build:
        gen = (
            GeminiGenerator(workers=args.workers)
            if args.generator == "gemini"
            else HFGenerator(args.hf_model, batch_size=args.batch_size)
        )
        table.fill(needed, problems, gen)
    missing = table.missing(needed)
    if missing and args.timeouts_as_failures:
        # No answer within the time limit is a failed task, as it would be in production.
        with open(table.path, "a") as f:
            for t, d in missing:
                table.cells[(t, d)] = 0.0
                row = {"task_id": t, "doc": d, "passed": 0.0, "completion": "", "error": "timeout"}
                f.write(json.dumps(row) + "\n")
        print(f"[build] recorded {len(missing)} timed-out cells as failures")
        missing = []
    if missing:
        sys.exit(f"{len(missing)} outcome cells missing from {table.path}; run with --build.")

    variant = "leaky" if args.leaky else "leakfree"
    names = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for sc_name in names:
        per_seed = {
            s: run_seed(s, args.epochs, table, *datasets[s], scenario=SCENARIOS[sc_name])
            for s in seeds
        }
        summary = summarize(per_seed, seeds)
        title = (
            f"GATE F — MBPP {variant}, scenario={sc_name}, generator={tag}, "
            f"{len(seeds)} seeds x {args.epochs} epochs, {N_QUERY} queries, pool top-{POOL_K}"
        )
        print_report(summary, title)
        stem = f"gate_f_{variant}_{sc_name}_{tag}"
        out = os.path.join(RESULTS_DIR, stem + ".json")
        with open(out, "w") as f:
            json.dump(
                {
                    "title": title,
                    "scenario": SCENARIOS[sc_name],
                    "seeds": seeds,
                    "epochs": args.epochs,
                    **summary,
                },
                f,
                indent=2,
            )
        print(f"\n[results] {out}")
        plot(per_seed, seeds, os.path.join(RESULTS_DIR, stem + ".png"), title=f"Gate F — {sc_name}")


if __name__ == "__main__":
    main()
