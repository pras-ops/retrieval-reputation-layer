# Results

All numbers below come from the current code and replay offline from committed data:

```bash
python3 sim/run_gate_t.py --scenario all --ablation        # Gate T (data/gate_t/, ToolRet-derived)
python3 sim/run_gate_w.py --scenario all --ablation        # Gate W (generated offline)
python3 sim/run_gate_g.py --variant hard --scenario all --ablation   # Gate G hard (data/gate_g/)
python3 sim/run_gate_g.py --variant easy --scenario all --ablation   # Gate G easy
python3 sim/summarize_results.py                           # prints the tables below
```

30 seeds per scenario; seeds vary the request order, feedback noise, flaky calls and exploration,
and every comparison is paired over the same seeds. Per-seed and paired statistics (95% CIs,
paired t-test, Wilcoxon) are in `sim/results/*.json`.

**Policies.** *Static reranker*: cross-encoder top-1, no learning. *Reuse what worked*: per
request, walk the reranker's order until something works, then keep it. *MemRL-style*:
`(1−λ)·z(sim) + λ·z(Q)`, `Q ← Q + 0.3(r − Q)`, taking feedback at face value. *RRL combined*:
reranker-blended relevance + Beta reputation + two-coin source reliability (per item) +
statistical change detection, with the benchmark exploration weight 0.30. *Explore 0.10* is
the same with the library's default exploration weight. *Per-context* adds counters per request
(or per site in Gate W), shrunk toward each item's global counters.

**Scenarios.** Verifier on every step (*stationary*), on 20% (*sparse verifier*, *spot check*),
users accepting 50% of answers at random (*sycophantic*), users accepting 90% of answers built
from a poisoned item (*fooled*), and a mid-stream change (*drift*: tool updates or an API
switch; *redesign*: site theme changes). Gate W's feedback is produced by the tools themselves:
each extractor's own success report and a schema validator, plus spot checks.

## Summary

1. **Learning which tools or documents work beats every non-learning baseline.** In Gate T the
   static reranker solves about 30% of requests; RRL 41–59%; the best possible pick 85–90%.
2. **Verifier-anchored trust is RRL's clear advantage.** When feedback is fooled by poisoned
   items, RRL beats MemRL-style: Gate T +3.0 / +3.7 (ToolBench) and +10.9 / +9.6 (ToolLens)
   points in *fooled* / *drift + fooled* (30/30 seeds each); Gate G hard +4.9 in *fooled* (28/30
   seeds); and in Gate W, where extractors over-report success, MemRL-style collapses to about 1%
   while RRL reaches 39–59% (30/30 seeds). The exception is Gate G hard *drift + fooled*, where
   the slow change detector costs more than trust gains (−8.3; with the old detector, a tie).
3. **Per-item reliability matters when sources fail on particular items.** In Gate W, where
   only some extractors over-report success, global-only reliability loses 8–9 points with
   spot checks. In Gate T and G it makes no difference.
4. **Per-context reputation helps when usefulness depends on context** (Gate W: +2.8 to +7.6
   points, best policy in every scenario) and does nothing or hurts when every request is new
   (Gate T) or contexts are few-shot (Gate G).
5. **Exploration weight 0.30 is too high.** The library default 0.10 is better in Gate T (+2 to
   +4 points) and in all but one Gate G scenario, and about equal in W. With it RRL also matches or beats MemRL-style on honest feedback in Gate T
   (ToolBench stationary 56.4 vs 55.2).
6. **Change detection is a trade-off, not a free win.** The new z-test detector cut false resets
   from about 20 to about 2 per run in Gate W (worth 5–7 points), but reacts later to a real API
   change: in Gate G hard under drift the old fast detector is 7–9 points better. In Gate T it
   rarely fires.
7. **Where MemRL-style still wins:** drift with honest or randomly noisy feedback (Gate G hard:
   −11.7 under *drift*, −18.3 under *drift + sycophantic*), and honest or randomly noisy feedback
   in Gate T with the 0.30 exploration weight (−2 to −4; with 0.10 the gap mostly closes). A running
   average that forgets quickly is hard to beat when feedback is not adversarial.

## Gate T — agent tool selection on real tool descriptions (ToolRet)

Requests and gold tools from ToolRet (ToolBench: 431 requests with a working tool in the top-8
pool out of 600; ToolLens: 200 of 314). Added on top: 30% of gold tools get a v2 replacement
(drift), 10% get a poisoned twin whose description copies example requests (the reranker picks
a poisoned tool first for 39% / 56% of requests), 20% of tools fail half their calls, and 20% of
requests are held out until a final block. Success = the chosen tool is a gold tool for the
request, deployed and not flaking — ToolRet's label standing in for execution.

### Gate T — tool selection, ToolBench (ToolRet)

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sparse_verifier | sycophantic | fooled | drift | drift_fooled |
|---|---|---|---|---|---|---|
| static reranker | 31.4 / 33.0 | 31.4 / 33.0 | 31.4 / 33.0 | 31.4 / 33.0 | 30.5 / 30.3 | 30.5 / 30.3 |
| reuse what worked | 46.0 / 33.0 | 46.0 / 33.0 | 40.9 / 33.0 | 39.6 / 33.0 | 41.0 / 30.3 | 34.9 / 30.3 |
| MemRL-style | 55.2 / 45.0 | 54.8 / 44.8 | 49.3 / 42.1 | 41.2 / 32.1 | 50.9 / 42.5 | 37.7 / 30.9 |
| RRL combined | 52.5 / 50.1 | 50.5 / 50.0 | 45.2 / 46.6 | 44.2 / 43.3 | 48.1 / 48.7 | 41.4 / 44.0 |
| RRL combined, explore 0.10 | 56.4 / 51.9 | 54.5 / 52.0 | 47.6 / 49.8 | 47.0 / 46.8 | 51.0 / 51.6 | 43.5 / 45.6 |
| RRL combined + per-context | 52.3 / 50.0 | 50.4 / 49.7 | 45.2 / 46.9 | 44.1 / 43.7 | 47.9 / 48.1 | 41.1 / 43.7 |
| best possible | 89.9 / 89.4 | 89.9 / 89.4 | 89.9 / 89.4 | 89.9 / 89.4 | 88.6 / 87.1 | 88.6 / 87.1 |


### Gate T — tool selection, ToolBench (ToolRet) — ablation

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sparse_verifier | sycophantic | fooled | drift | drift_fooled |
|---|---|---|---|---|---|---|
| RRL, reranker input only | 52.5 / 50.1 | 47.2 / 47.2 | 44.9 / 43.0 | 43.9 / 43.1 | 48.1 / 48.7 | 40.9 / 42.6 |
| combined, global-only reliability | 52.5 / 50.1 | 50.5 / 49.7 | 45.4 / 46.4 | 44.0 / 43.6 | 48.1 / 48.7 | 41.2 / 43.7 |
| combined, no change detection | 52.5 / 50.1 | 50.5 / 50.0 | 45.2 / 46.6 | 44.2 / 43.3 | 48.1 / 48.7 | 41.4 / 44.0 |
| combined, old detector (w=4, 0.6) | 52.5 / 50.1 | 50.5 / 50.0 | 45.2 / 46.6 | 44.2 / 43.3 | 48.1 / 48.7 | 41.4 / 44.0 |
| combined, exploration off | 58.9 / 50.9 | 56.8 / 50.3 | 48.4 / 47.1 | 47.7 / 45.8 | 52.9 / 52.0 | 44.6 / 46.5 |


### Gate T — tool selection, ToolLens (ToolRet)

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sparse_verifier | sycophantic | fooled | drift | drift_fooled |
|---|---|---|---|---|---|---|
| static reranker | 29.1 / 32.2 | 29.1 / 32.2 | 29.1 / 32.2 | 29.1 / 32.2 | 26.5 / 25.2 | 26.5 / 25.2 |
| reuse what worked | 42.9 / 32.2 | 42.9 / 32.2 | 38.1 / 32.2 | 34.2 / 32.2 | 36.2 / 25.2 | 29.3 / 25.2 |
| MemRL-style | 55.0 / 56.8 | 54.3 / 56.7 | 51.0 / 56.6 | 35.3 / 41.7 | 52.2 / 55.7 | 32.0 / 33.2 |
| RRL combined | 56.1 / 55.6 | 53.7 / 56.4 | 48.1 / 52.1 | 46.2 / 50.3 | 49.9 / 52.0 | 41.6 / 42.3 |
| RRL combined, explore 0.10 | 59.1 / 60.3 | 55.8 / 58.4 | 50.6 / 54.6 | 48.5 / 52.8 | 52.2 / 53.1 | 43.6 / 42.3 |
| RRL combined + per-context | 56.1 / 55.9 | 53.6 / 57.2 | 48.0 / 51.5 | 46.2 / 49.2 | 49.7 / 50.7 | 41.9 / 43.9 |
| best possible | 85.1 / 81.1 | 85.1 / 81.1 | 85.1 / 81.1 | 85.1 / 81.1 | 84.7 / 83.2 | 84.7 / 83.2 |


### Gate T — tool selection, ToolLens (ToolRet) — ablation

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sparse_verifier | sycophantic | fooled | drift | drift_fooled |
|---|---|---|---|---|---|---|
| RRL, reranker input only | 56.1 / 55.6 | 51.5 / 53.5 | 48.8 / 51.2 | 46.6 / 51.0 | 49.6 / 50.8 | 42.0 / 45.2 |
| combined, global-only reliability | 56.1 / 55.6 | 53.8 / 56.4 | 48.1 / 52.2 | 46.2 / 50.1 | 49.9 / 52.0 | 41.8 / 42.3 |
| combined, no change detection | 56.1 / 55.6 | 53.7 / 56.4 | 48.1 / 52.1 | 46.2 / 50.3 | 49.6 / 50.8 | 41.5 / 41.8 |
| combined, old detector (w=4, 0.6) | 56.0 / 55.2 | 53.7 / 56.3 | 48.1 / 52.2 | 46.2 / 50.3 | 50.3 / 51.5 | 41.8 / 43.2 |
| combined, exploration off | 60.2 / 59.8 | 57.4 / 60.0 | 51.5 / 54.8 | 49.1 / 52.3 | 53.4 / 50.2 | 42.8 / 41.2 |


## Gate W — web-scraping extractor selection (executable)

30 generated shops (6 held out), 8 extractors running real parsing code. The registry's
"universal AI extractor" matches every request best and the reranker picks it for all 30 sites;
it returns well-formed guesses, so its self-report and the schema validator both say success.
The *redesign* scenarios switch half the Shopify-like shops to a new theme halfway through.

### Gate W — web-scraping extractor selection

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | spot_check | redesign | redesign_spot_check |
|---|---|---|---|---|
| static reranker | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| reuse what worked | 19.0 / 5.0 | 19.0 / 5.0 | 17.8 / 4.3 | 17.8 / 4.3 |
| MemRL-style | 1.3 / 0.0 | 0.4 / 0.0 | 1.1 / 0.0 | 0.4 / 0.0 |
| RRL combined | 51.4 / 68.9 | 46.4 / 72.9 | 45.5 / 53.5 | 39.4 / 54.0 |
| RRL combined, explore 0.10 | 52.6 / 72.4 | 47.1 / 65.7 | 46.2 / 54.2 | 40.4 / 52.4 |
| RRL combined + per-context | 59.0 / 80.3 | 50.5 / 79.4 | 51.3 / 65.0 | 42.2 / 58.5 |
| best possible | 87.9 / 96.4 | 87.9 / 96.4 | 87.9 / 96.4 | 87.9 / 96.4 |


### Gate W — web-scraping extractor selection — ablation

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | spot_check | redesign | redesign_spot_check |
|---|---|---|---|---|
| RRL, reranker input only | 53.9 / 78.2 | 47.3 / 74.3 | 48.0 / 64.9 | 40.2 / 59.6 |
| combined, global-only reliability | 51.4 / 68.9 | 38.3 / 57.8 | 45.5 / 53.5 | 30.2 / 29.2 |
| combined, no change detection | 53.9 / 78.2 | 47.5 / 75.8 | 48.0 / 64.9 | 40.4 / 59.0 |
| combined, old detector (w=4, 0.6) | 44.6 / 60.3 | 40.9 / 59.9 | 39.7 / 39.4 | 34.6 / 31.8 |
| combined, exploration off | 51.7 / 66.0 | 45.8 / 58.6 | 45.2 / 49.3 | 38.6 / 39.7 |


Change-detector resets per run (stationary): old detector 23.0, new detector 1.9.

## Gate G hard — private-API documents with a poisoned reranker

Gemini 2.5 Flash writes code against a generated library (`zyn`) whose docs are the only way to
solve the task; poisoned guides fool the reranker on 79 of 80 poisoned questions; 40 questions
are held out; *drift* switches zyn 1.4 → 2.0 halfway.

### Gate G hard — private-API docs, poisoned reranker

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sycophantic | fooled | drift | drift_sycophantic | drift_fooled |
|---|---|---|---|---|---|---|
| static reranker | 50.8 / 50.0 | 50.8 / 50.0 | 50.8 / 50.0 | 23.5 / 0.0 | 23.5 / 0.0 | 23.5 / 0.0 |
| reuse what worked | 83.7 / 50.0 | 75.6 / 50.0 | 63.4 / 50.0 | 51.3 / 0.0 | 38.7 / 0.0 | 39.6 / 0.0 |
| MemRL-style | 96.1 / 100.0 | 92.1 / 99.8 | 83.2 / 96.0 | 83.4 / 98.1 | 70.7 / 90.1 | 65.0 / 84.0 |
| RRL combined | 96.5 / 100.0 | 90.2 / 99.8 | 88.0 / 98.8 | 71.7 / 96.7 | 52.4 / 65.2 | 56.7 / 75.7 |
| RRL combined, explore 0.10 | 97.7 / 100.0 | 93.1 / 99.8 | 90.5 / 97.8 | 75.5 / 95.0 | 54.2 / 67.9 | 59.9 / 78.8 |
| RRL combined + per-context | 96.5 / 100.0 | 89.9 / 99.8 | 87.5 / 99.2 | 61.3 / 76.0 | 48.4 / 51.6 | 52.9 / 70.2 |
| best possible | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |


### Gate G hard — private-API docs, poisoned reranker — ablation

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sycophantic | fooled | drift | drift_sycophantic | drift_fooled |
|---|---|---|---|---|---|---|
| RRL, reranker input only | 96.5 / 100.0 | 90.3 / 98.5 | 87.5 / 97.9 | 62.5 / 77.1 | 51.1 / 52.0 | 52.1 / 68.4 |
| combined, global-only reliability | 96.5 / 100.0 | 90.0 / 99.6 | 87.7 / 98.5 | 71.7 / 96.7 | 52.1 / 66.9 | 56.3 / 75.3 |
| combined, no change detection | 96.5 / 100.0 | 90.2 / 99.8 | 88.0 / 98.8 | 62.5 / 77.1 | 49.3 / 53.6 | 54.2 / 69.4 |
| combined, old detector (w=4, 0.6) | 96.5 / 100.0 | 90.2 / 99.7 | 88.0 / 98.7 | 80.6 / 99.1 | 60.6 / 78.9 | 63.9 / 82.3 |
| combined, exploration off | 97.7 / 100.0 | 89.9 / 97.3 | 88.2 / 95.8 | 76.8 / 100.0 | 50.6 / 58.3 | 56.1 / 72.6 |


The easy variant (`--variant easy`, results in `sim/results/gate_g_*.json`) has a reranker that
is always right before the switch; there RRL beats MemRL-style by about 19–22 points without
drift and loses by 2–4 with it.

## Gate F — MBPP, leak-free

`python3 sim/run_gate_matrix.py --generator gemini --scenario all --ablation` replays
`data/matrix_outcomes_gemini-2.5-flash.jsonl` (4,128 Gemini 2.5 Flash cells; one prompt timed
out on every retry and is recorded as a failure, `"error": "timeout"`). 30 questions per seed,
8 epochs, pool of 5 related MBPP solutions (never the question's own).

| Policy | stationary | sparse verifier | sycophantic | drift |
|---|---|---|---|---|
| no retrieval | 57.2 | 57.2 | 57.2 | 57.2 |
| static reranker | 57.1 | 57.1 | 57.1 | 56.3 |
| reuse what worked | 60.4 | 60.4 | 59.9 | 58.7 |
| MemRL-style | 57.9 | 57.8 | 57.8 | 57.6 |
| RRL combined | 58.4 | 57.7 | 57.7 | 57.1 |
| RRL combined, explore 0.10 | 58.3 | 58.1 | 58.0 | 57.4 |
| RRL + per-context | 58.7 | 58.0 | 57.9 | 57.3 |
| best possible | 61.3 | 61.3 | 61.3 | 61.3 |

**Retrieval barely matters here.** The reranker does no better than no retrieval (57.1 vs 57.2),
the most and least similar candidates pass at the same rate (56.9% vs 56.2% over 4,500 pairs),
a document helps in 2.0% of pairs and hurts in 2.8%, and the best possible pick is only 4 points
above the reranker. RRL gains +1.3 points over the reranker when nothing changes (p = 0.0001)
and is level with MemRL-style; per-question "reuse what worked" is best (+2.0 over RRL), since
with deterministic outcomes it settles on the best candidate per question. Gate F mainly shows
that a benchmark where the model already knows the answer cannot show the value of learning
which documents help — which is why Gates G, T and W were built.

## Limits

- **No live tool calls.** Gate T judges a choice by ToolRet's gold labels; Gate W runs real
  extraction code on generated pages; Gate G runs real code on a generated library.
- **Synthetic stress.** Tool updates, poisoned twins, flaky tools, redesigns and the feedback
  models are added on top of the data; their rates are choices, stated above.
- **MemRL-style is a reimplementation** of MemRL's ranking rule, not MemRL's code.
- **The benchmark exploration weight (0.30) predates these results**; the ablation favours the
  library default (0.10). Both are reported rather than re-picking the main configuration after
  seeing the results.

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
