# Gate G (hard) — poisoned reranker, held-out questions (current)

`python3 sim/run_gate_g.py --variant hard --scenario all` replays
`data/gate_g/outcomes_gemini-2.5-flash-t0.7-nothink_hard.jsonl` (1,920 Gemini 2.5 Flash cells).
160 questions (20 concepts × 8 salts); 40 are held out and appear once, in a final block.
Poisoned guides mirror their own concept's wording, and the cross-encoder ranks one first for
79 of the 80 questions on poisoned concepts. 30 seeds; differences are paired over seeds.

Feedback scenarios: *sycophantic* users accept 50% of answers at random; *fooled* users accept
90% of answers built from a poisoned guide and are honest otherwise (errors that depend on the
document). Both have the verifier on 20% of steps; *drift* switches zyn 1.4 → 2.0 halfway.

## Overall pass rate

| Policy | stationary | sparse verifier | sycophantic | fooled | drift | drift + sycophantic | drift + fooled |
|---|---|---|---|---|---|---|---|
| static cross-encoder | 50.8% | 50.8% | 50.8% | 50.8% | 23.5% | 23.5% | 23.5% |
| reuse what worked | 83.7% | 83.7% | 75.6% | 63.4% | 51.3% | 38.7% | 39.6% |
| MemRL-style | 96.1% | **96.3%** | 92.1% | 83.2% | **83.4%** | **70.7%** | 65.0% |
| RRL + reranker input | 96.5% | 92.7% | 90.3% | 87.5% | 62.5% | 51.1% | 52.1% |
| **RRL combined** | **96.5%** | 94.4% | 91.5% | **88.8%** | 80.6% | 59.7% | **65.2%** |

## Held-out questions (never seen before the final block)

| Policy | stationary | sparse verifier | sycophantic | fooled | drift | drift + sycophantic | drift + fooled |
|---|---|---|---|---|---|---|---|
| static cross-encoder | 50.0% | 50.0% | 50.0% | 50.0% | 0.0% | 0.0% | 0.0% |
| reuse what worked | 50.0% | 50.0% | 50.0% | 50.0% | 0.0% | 0.0% | 0.0% |
| MemRL-style | 100% | 100% | 99.8% | 96.0% | 98.1% | **90.1%** | 84.0% |
| **RRL combined** | 100% | 100% | 99.8% | **98.9%** | 99.1% | 73.6% | 84.0% |

## Ablation (overall / held-out; same seeds)

| Variant | drift | drift + sycophantic | fooled | drift + fooled |
|---|---|---|---|---|
| RRL + reranker input | 62.5 / 77.1 | 51.1 / 52.0 | 87.5 / 97.9 | 52.1 / 68.4 |
| + source reliability only | 62.5 / 77.1 | 47.7 / 41.9 | 88.8 / 98.9 | 53.5 / 67.8 |
| + change detection only | 80.6 / 99.1 | 56.5 / 62.7 | 87.5 / 97.9 | 64.2 / 88.3 |
| + both (RRL combined) | 80.6 / 99.1 | 59.7 / 73.6 | 88.8 / 98.9 | 65.2 / 84.0 |
| both, exploration off | **85.8 / 100** | 57.7 / 60.5 | 88.3 / 96.6 | **65.6** / 77.3 |
| MemRL-style | 83.4 / 98.1 | **70.7 / 90.1** | 83.2 / 96.0 | 65.0 / 84.0 |

(The ablation rows come from a one-off script over the same table, not from `run_gate_g.py`.)

## What this says

1. **Learning which documents work transfers to new questions.** On held-out questions the
   combined score beats the static reranker and the per-question memory by +49 to +99 points in
   every scenario (30/30 seeds).
2. **When users are fooled by the poisoned guides, RRL beats MemRL-style**: +5.6 points overall
   (27/30 seeds, p < 1e-6) and +2.9 on held-out questions (p = 0.0003). A learner that takes
   feedback at face value ranks the poison up. Most of this margin comes from RRL's existing
   safeguards (the verifier override, the per-document liar counter and the cap on positive
   behavior); the per-document source reliability adds about one more point.
3. **Change detection is the component that handles the library switch** (+18 points under
   drift, +12 under drift + fooled). Source reliability alone does nothing there and slightly
   hurts under random sycophancy.
4. **Exploration is a cost here.** With Thompson exploration off, the combined score beats
   MemRL-style under drift (85.8% vs 83.4%), but loses some held-out accuracy elsewhere.
5. **The open weakness is drift + random sycophancy**, where MemRL-style wins by 11 points overall
   and 16.5 on held-out questions (27–30/30 seeds). Its running average forgets quickly; RRL's
   counters plus a change detector working from noisy outcomes recover more slowly. That is a
   hypothesis from these numbers, not a tested explanation.

## Limits

- Controlled synthetic corpus; one fixed corpus; the two samples per cell mostly agree.
- MemRL-style is a reimplementation of MemRL's ranking rule on this benchmark, not MemRL itself.
- The feedback models (uniform sycophancy, poison-specific trust) are simple stand-ins for real
  user behavior.

---

# Gate G — private-API documents (current)

`sim/run_gate_g.py`, outcome table `data/gate_g/outcomes_gemini-2.5-flash-t0.7-nothink.jsonl`,
30 seeds × 4 epochs × 80 questions (20 concepts × 4 salts), Gemini 2.5 Flash (temperature 0.7,
thinking off, 2 samples per cell), real execution against `zyn` 1.4 / 2.0. Reproduce offline with
`python3 sim/run_gate_g.py --scenario all`.

## What the outcome table shows

| Document given to the model | Pass under zyn 1.4 | Pass under zyn 2.0 |
|---|---|---|
| None | 38% | 3% |
| The question's own 1.x doc | 100% | 0% |
| The question's own 2.0 doc | 0% | 100% |
| Any poisoned, sibling or other doc | 0% | 0% |

The documents decide the outcome, unlike Gate F. But the cross-encoder ranks the question's own
1.x doc first and its 2.0 doc second for all 80 questions, and the two samples per cell agree 94%
of the time. So ranking is easy here; the hard parts are the library switch and noisy feedback.

## Overall pass rate (mean over 30 seeds)

| Policy | stationary | sparse verifier | sycophantic | drift | drift + sycophantic |
|---|---|---|---|---|---|
| no retrieval | 37.8% | 37.8% | 37.8% | 20.6% | 20.6% |
| static cross-encoder | **100%** | **100%** | **100%** | 50.0% | 50.0% |
| reuse what worked (`query_memory`) | **100%** | **100%** | **100%** | **75.0%** | **64.5%** |
| MemRL-style | 76.7% | 74.9% | 70.5% | 63.5% | 56.3% |
| RRL, Gate E config (`rrl`) | 55.2% | 55.0% | 44.0% | 28.9% | 31.0% |
| RRL + reranker input (`rrl_ce`) | 96.1% | 91.6% | 89.6% | 59.3% | 51.8% |
| RRL combined (`rrl_combined`) | 96.1% | 94.1% | 92.9% | 70.8% | 57.7% |
| RRL per-question (`rrl_query`) | 61.8% | 59.0% | 49.4% | 32.5% | 31.5% |

## Paired differences (30 seeds; all p < 1e-5 unless marked)

- **Each component of the combined score helps where it is meant to.** Adding the reranker as an
  input: +40 points stationary, +16 drift. Adding source reliability and change detection on top:
  +11.5 drift, +5.9 drift + sycophantic, +3.3 sycophantic, +2.6 sparse verifier, 0 stationary.
- **Combined vs static reranker:** +20.8 points under drift, +7.7 under drift + sycophantic;
  −3.9 to −7.1 points when nothing changes (the cost of exploring when the ranking is right).
- **Combined vs MemRL-style:** +7 to +22 points in four scenarios; +1.4 under drift + sycophantic
  (p = 0.07, not significant).
- **Combined vs reuse-what-worked:** −3.9 to −7.1 points in every scenario. The simple memory
  wins overall.
- **Transfer after the switch.** In the first epoch after zyn 2.0 lands, every question is seen
  once, so reuse-what-worked fails all of them (0%); the combined score carries the lesson across
  a concept's four questions and reaches 17% (+16.8 points, 30/30 seeds). MemRL-style reaches 37%.

## What this says

1. Global per-document reputation alone (Gate E config) is worse than not learning: a sibling
   doc that is right for "ember rank" earns a good reputation and is then picked for "ember score".
   The reranker input fixes most of this.
2. Source reliability and change detection are worth keeping: they are the only additions that
   help under noisy feedback and drift, and they cost nothing when the world is stable.
3. RRL still loses to "reranker + move on when it fails" because Thompson exploration keeps
   paying a cost when the reranker is already right, and per-question learning gets only four
   visits per question here.

## Limits of this benchmark

- The reranker is perfect before the switch and its second choice is perfect after it, which
  favors `query_memory`. Poisoned guides never outranked the real docs, so poisoned *documents*
  were not stressed; poisoned *feedback* was.
- Controlled synthetic corpus (one fixed corpus; seeds vary order, samples and exploration).

---

# Realistic Recurring Benchmark Results (MBPP)

> [!WARNING]
> **Known problems with this benchmark (Gate E).** Treat the numbers below as superseded by
> Gate F (`sim/run_gate_matrix.py`, see the README).
> - **Answer-key leak.** Each query's own reference solution is in the corpus. In the committed
>   cache, retrieving it passes 78.8% of the time (96 of 170 completions copy it verbatim), while
>   any other document passes about 30%. The benchmark mostly measures whether a policy finds the
>   answer, not whether it finds useful context.
> - **Not reproducible from the repo.** `--replay` fails on every seed for the RRL arm: the
>   current code picks (query, document) pairs the original sweep never ran, so they are not in
>   `data/mbpp_sweep_cache.jsonl`. The static arm replays fine.
> - **Generator.** The harness calls `gemini-2.5-flash`; an earlier version of this page said
>   `gemini-2.5-pro`.

This document presents the empirical validation results of the **Retrieval Reputation Layer (RRL)** against a strong static baseline on a realistic, recurring-query setting using the Google MBPP dataset.

## Experimental Setup

- **Dataset:** 974 programming tasks from the Google MBPP dataset, grouped into naturally recurring problem families (e.g., list manipulation, math functions, string processing, tuples, etc.).
- **Retriever Pool:** Hybrid retrieval combining sparse (BM25) and dense embeddings, generating a top-5 candidate list.
- **Static Baseline:** A production-grade Cross-Encoder reranker selecting the top candidate from the top-5 hybrid retrieval list (no online learning or outcome awareness).
- **RRL Configuration:** Hybrid retriever wrapped with RRL using **global Beta counters** and a **Thompson-sampling exploration policy**. Downstream correctness (`passed = 1.0` or `0.0`) observed via isolated unit-test execution.
- **Sweep Parameters:** 10 independent random seeds, 8 recurring epochs (30 steps per epoch, totaling 240 steps per seed).
- **LLM Generator:** Gemini API (`gemini-2.5-flash`, temperature 0, Vertex AI or API key) producing Python code solutions.

---

## Performance Summary

| Metric | Static (Cross-Encoder) | RRL (Ours) | Absolute Lift | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Pass Rate** | 0.540 [0.406, 0.674] | **0.569 [0.509, 0.628]** | **+2.9 pts** | **+5.4%** |
| **Late-Stage Pass Rate** | 0.542 [0.399, 0.685] | **0.590 [0.525, 0.654]** | **+4.8 pts** | **+8.9%** |

*Note: In brackets `[...]` are the 95% confidence intervals across the 10 seeds.*

---

## Detailed Performance Analysis

### 1. The Learning Curve & Late-Stage Separation
During the early epochs, RRL pays a small **exploration tax** to probe alternative retrieval candidates and gather outcome statistics. 
* **Early stage:** The accuracies start closer as RRL populates the Beta counters.
* **Late stage (epochs 5-8):** As the reputation counters converge, RRL shifts toward exploitation and the gap widens to +4.8 pts: **0.590 [0.525, 0.654]** for RRL versus **0.542 [0.399, 0.685]** for the static reranker.
* **Statistical significance — not yet established.** With 10 seeds, the 95% confidence intervals overlap for both the overall and the late-stage pass rate. This benchmark therefore shows a consistent *direction*, not a statistically established lift. RRL's results do vary less across seeds (CI width `0.119` vs `0.268` overall), but a narrower spread is not a significance test.
* **What would settle it:** more seeds (Gate B needed 30 before its intervals separated), or a paired per-seed test, which fits here because both arms run on the same seeds and problem sets.

### 2. Mitigation of Distractor Noise
The core failure mode of the static cross-encoder is its susceptibility to highly semantically similar but logically incorrect distractor chunks (e.g., a function with the same name but slightly different parameters).
* RRL successfully identifies these distractors when they fail downstream unit tests.
* RRL decays their reputation, ensuring they are penalised and evicted from top rankings in subsequent epochs.

### 3. Verification of Stated Boundaries
These results are consistent with the central conditional claim of RRL:
1. **Recurrence + Verifier = Win:** In recurring tasks (MBPP problem families) with a reliable verifier (unit tests), online reputation tracking points the same way as the controlled gates: ahead of a strong static reranker overall, and further ahead once the counters converge. At 10 seeds that lift is directional, not yet statistically significant.
2. **Exploration Trade-Off:** The exploration tax is minor, meaning that RRL is a viable candidate for production integration in environments where queries recur.

---

## Visualization
The comparison plots showing the cumulative pass rates and learning trajectories across the epochs have been saved to:
`sim/gate_recurring_comparison.png`
