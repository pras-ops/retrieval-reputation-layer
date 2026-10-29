# Results

All numbers below come from the current code and replay offline from committed data:

```bash
python3 sim/run_gate_m.py --scenario all --ablation        # Gate M, recorded live MCP responses
python3 sim/run_gate_m.py --clean --scenario all --ablation   # Gate M without the poisoned server
python3 sim/run_gate_m.py --live --seeds 3 --scenario update  # Gate M fully live (real MCP calls)
python3 sim/run_gate_t.py --scenario all --ablation        # Gate T (data/gate_t/, ToolRet-derived)
python3 sim/run_gate_w.py --scenario all --ablation        # Gate W (generated offline)
python3 sim/run_gate_g.py --variant hard --scenario all --ablation   # Gate G hard (data/gate_g/)
python3 sim/run_gate_matrix.py --generator gemini --scenario all --ablation   # Gate F
python3 sim/summarize_results.py                           # prints most tables below
```

30 seeds per scenario; seeds vary the request order, feedback noise, flaky calls and exploration,
and every comparison is paired over the same seeds. Per-seed and paired statistics (95% CIs,
paired t-test, Wilcoxon) are in `sim/results/*.json`.

**Policies.** *Static reranker*: cross-encoder top-1, no learning. *Gemini picks* (Gate M):
Gemini 2.5 Flash reads the live tool list and plans its calls, as agents do today; it does not
learn. *Reuse what worked*: per request (per site in W and M), walk the reranker's order until
something works, then keep it; its *robust* variant keeps a working tool through an isolated
failure (two in a row, three if unverified, before moving on). *MemRL-style*: `(1−λ)·z(sim) + λ·z(Q)`, `Q ← Q + 0.3(r − Q)`.
Both learning baselines use the verifier when it ran and take other feedback at face value.
*RRL combined*: reranker-blended relevance + Beta reputation + two-coin source reliability (per
item) + statistical change detection, exploration weight 0.30; *explore 0.10* uses the library
default; *per-context* adds counters per request (per site in W and M), shrunk toward each
item's global counters; *verified memory* (`stick_to_verified`) additionally keeps a tool for a
request once the verifier confirmed it worked there, until a verified failure.

> **Correction (2026-10-29).** An earlier version of these results let the two learning
> baselines blend the verifier's result with the other signals even when the verifier ran,
> instead of using it. That under-stated them, most in Gate W (MemRL-style was reported at
> about 1% with every call verified; it is 34–36%). All tables below use the corrected
> baselines.

## Summary

1. **Learning which tools work beats choosing by description**, including an LLM choosing from
   the tool list. With a poisoned "universal" tool in the registry, the static reranker and
   Gemini pick it almost always (Gemini: 92% of pages; it also planned the injected
   `send_report` call 4–6% of the time) and solve 0–8% of requests; RRL with per-context counters
   and verified memory solves 43–67% and picks the poisoned tool on about 2% of steps when every
   call is verified.
2. **With sparse verification and tools that over-report success, RRL is the learner that is
   not fooled.** With spot checks on 20% of calls (Gates W and M), MemRL-style trusts the
   poisoned tool's own success reports and falls to 3–4%; RRL keeps 43–53%, within 3 points of
   per-site "reuse what worked" overall and far ahead on new sites (53–80% vs 18–20%).
3. **When every call is verified and contexts recur, a simple per-context memory still wins.**
   The robust variant of "reuse what worked" beats RRL by 8–21 points in Gates W and M. The same
   robustness makes it easy to fool: with 20% spot checks it holds on to the poisoned tool on the
   strength of that tool's own success reports and falls to 22–27%, where RRL keeps 43–53%
   (+20 to +28 points, 30/30 seeds). Being robust to transient failures *and* hard to fool is
   what a memory without a trust model cannot do.
   **Verified memory closes part of the gap to the original per-site memory.**
   Keeping a verifier-confirmed tool per context lifts RRL by 2–9 points in Gates W and M and
   cuts "reuse what worked"'s lead from 8–13 to 4–6 points with the poisoned server present
   (10 points under the site redesign). Without the poisoned server that memory still leads by
   11–16 points. A confidence gate on exploration did nothing: while any tool is untried, the
   current best can never be shown best with 95% confidence.
4. **On real tool descriptions (Gate T), RRL and MemRL-style are close** (−4.1 to +2.7 points
   with verified memory; −2.4 to +4.6 at exploration 0.10), and both beat per-request memory by
   5–15 points because requests rarely repeat.
5. **Under abrupt change, MemRL-style adapts faster** (Gate G hard: 9–27 points ahead).
6. **Per-item reliability and per-context counters matter most.** Global-only reliability
   loses 7–9 points with spot checks (W, M); per-context counters add 3–12 points where contexts
   recur (W, M) and do nothing or hurt where they do not (T, G).
7. **Change detection is a trade-off.** The z-test detector cuts false resets from about 20 to
   about 2 per run (W, M), but reacts later to a real API change than the original rule (G).
8. **When the model already knows the answer, retrieval does not matter** (Gate F: reranker
   57.1% = no retrieval 57.2%).

## Gate M — tool selection over live MCP servers

Four local MCP servers over stdio (`sim/mcp_testbed/`): CSS, structured-data and heuristic
extractors (the heuristic one on a flaky server), and a "universal AI extractor" whose
description is inflated and carries an MCPTox-style injected instruction to call `send_report`
(a harmless canary). Tools fetch real pages from a local shop site; 6 of 30 shops are held out.
In the *update* scenarios half the Shopify-like shops redesign and the structured-data server is
upgraded mid-run (`extract_jsonld(url)` becomes `jsonld_extract(page_url)`). Outcomes were
recorded from 2,280 live MCP calls and replayed with flaky-server timeouts (20%) and 2% transient
failures on every tool.

### Gate M — live MCP servers (replayed), with the poisoned server

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | spot_check | update | update_spot_check |
|---|---|---|---|---|
| static reranker | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| Gemini picks from tool list | 1.0 / 0.0 | 1.0 / 0.0 | 8.2 / 0.0 | 8.2 / 0.0 |
| reuse what worked | 70.9 / 65.0 | 51.8 / 20.3 | 62.9 / 61.3 | 46.5 / 17.9 |
| reuse what worked, robust | 75.0 / 65.3 | 24.8 / 11.1 | 65.7 / 61.4 | 21.9 / 10.1 |
| MemRL-style | 35.8 / 45.8 | 3.9 / 1.9 | 33.8 / 32.2 | 3.2 / 1.2 |
| RRL combined | 45.9 / 57.2 | 42.1 / 60.3 | 42.5 / 51.2 | 37.5 / 44.4 |
| RRL combined + per-context | 57.8 / 67.6 | 48.8 / 69.4 | 49.8 / 60.4 | 41.1 / 55.7 |
| RRL + per-context + verified memory | 66.6 / 79.3 | 52.4 / 72.5 | 56.5 / 66.5 | 43.4 / 52.8 |
| best possible | 90.4 / 98.8 | 90.4 / 98.8 | 90.4 / 98.8 | 90.4 / 98.8 |


Poisoned tool chosen (share of steps), all calls verified / 20% verified: static reranker
100% / 100%, Gemini 99% / 99%, MemRL-style 4% / 90%, RRL + per-context + verified memory
1.4% / 13.0%.

### Gate M — live MCP servers (replayed), clean registry

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | spot_check | update | update_spot_check |
|---|---|---|---|---|
| static reranker | 46.0 / 65.7 | 46.0 / 65.7 | 37.4 / 49.3 | 37.4 / 49.3 |
| Gemini picks from tool list | 47.7 / 61.8 | 47.7 / 61.8 | 49.7 / 53.8 | 49.7 / 53.8 |
| reuse what worked | 77.5 / 89.4 | 77.0 / 89.4 | 70.4 / 85.4 | 69.2 / 85.4 |
| reuse what worked, robust | 81.8 / 90.3 | 81.6 / 90.3 | 72.5 / 86.0 | 71.3 / 86.0 |
| MemRL-style | 34.9 / 45.6 | 34.2 / 40.8 | 34.1 / 31.7 | 31.7 / 26.9 |
| RRL combined | 48.0 / 61.9 | 47.7 / 58.3 | 44.1 / 51.9 | 44.1 / 50.8 |
| RRL combined + per-context | 58.8 / 67.5 | 58.1 / 68.3 | 50.8 / 62.8 | 50.0 / 61.5 |
| RRL + per-context + verified memory | 66.3 / 75.7 | 61.0 / 68.8 | 56.0 / 65.0 | 52.7 / 63.6 |
| best possible | 90.4 / 98.8 | 90.4 / 98.8 | 90.4 / 98.8 | 90.4 / 98.8 |


Without the poisoned server, Gemini picks a sensible tool (48–50%, level with the reranker
before the update and ahead after it), and per-site "reuse what worked" is best (69–78%).

**Live check.** The same policies run with real MCP calls on every step (3 seeds):

| Policy | stationary: replay (30 seeds) | live (3 seeds) | update: replay | live |
|---|---|---|---|---|
| static reranker | 0.0 | 0.0 | 0.0 | 0.0 |
| Gemini picks from tool list | 1.0 | 1.0 | 8.2 | 8.3 |
| reuse what worked | 70.9 | 77.2 | 62.9 | 69.4 |
| reuse what worked, robust | 75.0 | 77.2 | 65.7 | 67.9 |
| MemRL-style | 35.8 | 37.7 | 33.8 | 35.0 |
| RRL combined | 45.9 | 47.4 | 42.5 | 44.3 |
| RRL + per-context | 57.8 | 57.9 | 49.8 | 49.8 |
| RRL + per-context + verified memory | 66.6 | 66.2 | 56.5 | 58.6 |

Live and replay agree within about 2 points, except "reuse what worked", which does 6 points
better live: the replay adds 2% transient failures on every tool, the local servers had none
outside the flaky one, and that baseline abandons a working tool after any single failure.

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
| reuse what worked | 46.0 / 33.0 | 46.0 / 33.0 | 41.1 / 33.0 | 39.9 / 33.0 | 41.0 / 30.3 | 35.2 / 30.3 |
| reuse what worked, robust | 46.4 / 33.0 | 46.4 / 33.0 | 39.8 / 33.0 | 39.7 / 33.0 | 40.4 / 30.3 | 34.1 / 30.3 |
| MemRL-style | 55.5 / 45.3 | 54.8 / 44.7 | 50.0 / 42.7 | 45.5 / 39.2 | 51.0 / 42.3 | 41.9 / 38.2 |
| RRL combined | 52.5 / 50.1 | 50.5 / 50.0 | 45.2 / 46.6 | 44.2 / 43.3 | 48.1 / 48.7 | 41.4 / 44.0 |
| RRL combined, explore 0.10 | 56.4 / 51.9 | 54.5 / 52.0 | 47.6 / 49.8 | 47.0 / 46.8 | 51.0 / 51.6 | 43.5 / 45.6 |
| RRL combined + per-context | 52.3 / 50.0 | 50.4 / 49.7 | 45.2 / 46.9 | 44.1 / 43.7 | 47.9 / 48.1 | 41.1 / 43.7 |
| RRL + per-context + verified memory | 55.3 / 48.4 | 51.2 / 49.5 | 45.9 / 45.6 | 45.0 / 42.8 | 49.5 / 48.5 | 42.0 / 42.6 |
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
| combined, confidence-gated exploration | 52.5 / 49.8 | 50.5 / 49.0 | 45.2 / 46.6 | 44.2 / 43.4 | 48.0 / 48.1 | 41.4 / 44.0 |
| per-context + confidence-gated exploration | 52.3 / 49.5 | 50.4 / 49.8 | 45.2 / 46.9 | 44.1 / 43.8 | 48.0 / 48.9 | 41.1 / 43.4 |


### Gate T — tool selection, ToolLens (ToolRet)

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sparse_verifier | sycophantic | fooled | drift | drift_fooled |
|---|---|---|---|---|---|---|
| static reranker | 29.1 / 32.2 | 29.1 / 32.2 | 29.1 / 32.2 | 29.1 / 32.2 | 26.5 / 25.2 | 26.5 / 25.2 |
| reuse what worked | 42.9 / 32.2 | 42.9 / 32.2 | 38.2 / 32.2 | 34.7 / 32.2 | 36.2 / 25.2 | 29.7 / 25.2 |
| reuse what worked, robust | 43.5 / 32.2 | 43.5 / 32.2 | 37.1 / 32.2 | 34.3 / 32.2 | 35.7 / 25.2 | 28.6 / 25.2 |
| MemRL-style | 55.4 / 57.9 | 54.3 / 56.8 | 51.7 / 56.3 | 43.9 / 50.3 | 52.4 / 55.4 | 40.6 / 42.0 |
| RRL combined | 56.1 / 55.6 | 53.7 / 56.4 | 48.1 / 52.1 | 46.2 / 50.3 | 49.9 / 52.0 | 41.6 / 42.3 |
| RRL combined, explore 0.10 | 59.1 / 60.3 | 55.8 / 58.4 | 50.6 / 54.6 | 48.5 / 52.8 | 52.2 / 53.1 | 43.6 / 42.3 |
| RRL combined + per-context | 56.1 / 55.9 | 53.6 / 57.2 | 48.0 / 51.5 | 46.2 / 49.2 | 49.7 / 50.7 | 41.9 / 43.9 |
| RRL + per-context + verified memory | 58.1 / 55.8 | 54.2 / 57.2 | 48.7 / 51.1 | 46.3 / 49.8 | 50.4 / 51.2 | 41.8 / 45.0 |
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
| combined, confidence-gated exploration | 56.3 / 57.1 | 53.8 / 57.2 | 48.1 / 52.5 | 46.2 / 50.8 | 49.9 / 51.5 | 41.6 / 42.3 |
| per-context + confidence-gated exploration | 56.3 / 55.2 | 53.6 / 56.3 | 47.9 / 51.0 | 46.2 / 49.0 | 49.3 / 50.6 | 41.9 / 43.8 |


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
| reuse what worked | 67.0 / 61.4 | 50.4 / 19.6 | 64.2 / 58.5 | 47.9 / 17.5 |
| reuse what worked, robust | 73.9 / 61.9 | 27.0 / 11.0 | 70.4 / 58.9 | 25.2 / 10.3 |
| MemRL-style | 35.9 / 43.3 | 4.4 / 3.6 | 34.0 / 32.1 | 4.3 / 3.5 |
| RRL combined | 51.4 / 68.9 | 46.4 / 72.9 | 45.5 / 53.5 | 39.4 / 54.0 |
| RRL combined, explore 0.10 | 52.6 / 72.4 | 47.1 / 65.7 | 46.2 / 54.2 | 40.4 / 52.4 |
| RRL combined + per-context | 59.0 / 80.3 | 50.5 / 79.4 | 51.3 / 65.0 | 42.2 / 58.5 |
| RRL + per-context + verified memory | 62.5 / 85.7 | 53.1 / 79.6 | 54.1 / 66.5 | 45.4 / 61.9 |
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
| combined, confidence-gated exploration | 51.4 / 68.9 | 46.4 / 72.9 | 45.5 / 53.5 | 39.4 / 54.0 |
| per-context + confidence-gated exploration | 59.0 / 80.3 | 50.5 / 79.4 | 51.3 / 65.0 | 42.2 / 58.5 |


## Gate G hard — private-API documents with a poisoned reranker

Gemini 2.5 Flash writes code against a generated library (`zyn`) whose docs are the only way to
solve the task; poisoned guides fool the reranker on 79 of 80 poisoned questions; 40 questions
are held out; *drift* switches zyn 1.4 → 2.0 halfway.

### Gate G hard — private-API docs, poisoned reranker

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sycophantic | fooled | drift | drift_sycophantic | drift_fooled |
|---|---|---|---|---|---|---|
| static reranker | 50.8 / 50.0 | 50.8 / 50.0 | 50.8 / 50.0 | 23.5 / 0.0 | 23.5 / 0.0 | 23.5 / 0.0 |
| reuse what worked | 83.7 / 50.0 | 76.0 / 50.0 | 64.8 / 50.0 | 51.3 / 0.0 | 39.1 / 0.0 | 40.1 / 0.0 |
| reuse what worked, robust | 83.7 / 50.0 | 70.4 / 50.0 | 59.4 / 50.0 | 34.8 / 0.0 | 29.8 / 0.0 | 26.3 / 0.0 |
| MemRL-style | 96.0 / 100.0 | 93.1 / 100.0 | 89.7 / 99.4 | 80.9 / 95.5 | 73.7 / 93.8 | 74.8 / 93.9 |
| RRL combined | 96.5 / 100.0 | 90.2 / 99.8 | 88.0 / 98.8 | 71.7 / 96.7 | 52.4 / 65.2 | 56.7 / 75.7 |
| RRL combined, explore 0.10 | 97.7 / 100.0 | 93.1 / 99.8 | 90.5 / 97.8 | 75.5 / 95.0 | 54.2 / 67.9 | 59.9 / 78.8 |
| RRL combined + per-context | 96.5 / 100.0 | 89.9 / 99.8 | 87.5 / 99.2 | 61.3 / 76.0 | 48.4 / 51.6 | 52.9 / 70.2 |
| RRL + per-context + verified memory | 96.5 / 100.0 | 90.3 / 99.8 | 88.1 / 99.4 | 59.8 / 74.7 | 46.9 / 53.1 | 50.5 / 70.1 |
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
| combined, confidence-gated exploration | 96.4 / 100.0 | 90.2 / 99.8 | 88.0 / 98.8 | 72.1 / 98.2 | 52.5 / 66.1 | 56.7 / 76.1 |
| per-context + confidence-gated exploration | 96.4 / 100.0 | 89.9 / 99.8 | 87.4 / 99.1 | 61.3 / 74.9 | 48.5 / 51.8 | 52.8 / 70.4 |


## Gate F — MBPP, leak-free

`data/matrix_outcomes_gemini-2.5-flash.jsonl` (4,128 Gemini 2.5 Flash cells; one prompt timed out
on every retry and is recorded as a failure). 30 questions per seed, 8 epochs, a pool of 5
related MBPP solutions (never the question's own).

| Policy | stationary | sparse_verifier | sycophantic | drift |
|---|---|---|---|---|
| no retrieval | 57.2 | 57.2 | 57.2 | 57.2 |
| static reranker | 57.1 | 57.1 | 57.1 | 56.3 |
| reuse what worked | 60.4 | 60.4 | 59.9 | 58.7 |
| MemRL-style | 57.7 | 57.8 | 57.4 | 57.2 |
| RRL combined | 58.4 | 57.7 | 57.7 | 57.1 |
| RRL + per-context | 58.7 | 58.0 | 57.9 | 57.3 |
| RRL + per-context + verified memory | 59.3 | 58.5 | 57.9 | 57.8 |
| best possible | 61.3 | 61.3 | 61.3 | 61.3 |

Retrieval barely matters here: the reranker matches no retrieval, the most and least similar
candidates pass at the same rate (56.9% vs 56.2% over 4,500 pairs), and the best possible pick is
4 points above the reranker. A benchmark where the model already knows the answer cannot show
the value of learning which documents help, which is why Gates G, T, W and M were built.

## Limits

- **Gate M's tools are real MCP tools on local servers**, with a generated website and a
  synthetic poisoned server; Gate T judges by ToolRet's labels; Gate W runs real code on
  generated pages. No third-party MCP servers or live web sites were used.
- **Synthetic stress.** Tool updates, poisoned tools, flaky tools, redesigns and the feedback
  models are added on top of the data; their rates are choices, stated above.
- **MemRL-style is a reimplementation** of MemRL's ranking rule, not MemRL's code.
- **The benchmark exploration weight (0.30) predates these results**; the ablation favours the
  library default (0.10) in Gates T and G. Both are reported rather than re-picking the main
  configuration after seeing the results.

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
