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
request once the verifier confirmed it worked there, until a verified failure; *fixed feedback
rule* is the same with learned source reliability replaced by the library's fixed rule
(`calculate_outcome`: the tool's own success report capped at 0.75 and averaged with the schema
check), so unverified success counts for less than unverified failure.

> **Correction (2026-10-29).** An earlier version of these results let the two learning
> baselines blend the verifier's result with the other signals even when the verifier ran,
> instead of using it. That under-stated them, most in Gate W (MemRL-style was reported at
> about 1% with every call verified; it is 34–36%). All tables below use the corrected
> baselines.

> **Fixes after a second review (2026-10-29).** Two implementation issues were fixed and every
> gate re-run: learned source reliability could give a source that always says "success" a
> positive vote when verified outcomes were few and unbalanced, and change detection ran a
> proportion test on soft outcomes. Change detection now tests verified 0/1 outcomes, globally
> and per context. Baseline numbers are unchanged; RRL numbers moved by about a point in most
> scenarios (Gate G hard drift: RRL + per-context + verified memory 59.8% → 68.2%).

## Summary

1. **Learning which tools work beats choosing by description**, including an LLM choosing from
   the tool list. With a poisoned "universal" tool in the registry, the static reranker and
   Gemini pick it almost always (Gemini: 92% of pages; it also planned the injected
   `send_report` call 4–6% of the time) and solve 0–8% of requests; RRL with per-context counters
   and verified memory solves 43–66% and picks the poisoned tool on about 1% of steps when every
   call is verified.
2. **With sparse verification and tools that over-report success, learners that trust the
   reports are fooled; RRL much less so.** With spot checks on 20% of calls (Gates W and M),
   MemRL-style trusts the poisoned tool's own success reports and falls to 3–4%; RRL keeps
   43–52%, within 3.5 points of per-site "reuse what worked" overall and far ahead on new sites
   (56–82% vs 18–20%). It still picks the poisoned tool on 14% of steps (Gate M).
3. **When every call is verified and contexts recur, a simple per-context memory still wins.**
   The robust variant of "reuse what worked" beats RRL by 9–16 points in Gates W and M (and by
   16–17 on the clean registry); bootstrapping over the 30 Gate M shops, that lead is 6.1 points
   per shop with an interval that includes zero. The same robustness makes it easy to fool: with
   20% spot checks it holds on to the poisoned tool on the strength of that tool's own success
   reports and falls to 22–27%, where RRL keeps 43–52% (+19 to +27 points, 30/30 seeds; +32 per
   shop, 95% interval +21 to +43). Being robust to transient failures *and* hard to fool needs
   unverified success to count for less than a verified one, which that memory lacks.
   **Verified memory closes part of the gap to the original per-site memory.**
   Keeping a verifier-confirmed tool per context lifts RRL by 2–9 points in Gates W and M and
   cuts "reuse what worked"'s lead from 9–14 to 5–7 points with the poisoned server present
   (10 points under the site redesign). Without the poisoned server that memory still leads by
   11–15 points. A confidence gate on exploration did nothing: while any tool is untried, the
   current best can never be shown best with 95% confidence.
4. **On real tool descriptions (Gate T), RRL and MemRL-style are close** (−3.9 to +2.7 points
   with verified memory; −2.4 to +4.2 at exploration 0.10), and both beat per-request memory by
   5–16 points because requests rarely repeat.
5. **Under abrupt change, MemRL-style adapts faster** (Gate G hard: 9–25 points ahead).
6. **Per-context counters and verified memory matter most; learned source trust does not.**
   Per-context counters add 2–11 points where contexts recur (W, M) and do nothing or hurt where
   they do not (T, G). Replacing learned per-tool reliability with the fixed feedback rule moves
   results by −3.2 to +2.9 points (learned trust better in 5 of 23 scenarios with unverified
   feedback, worse in 6; 2 and 1 after a Holm correction), and the fixed rule is ahead in all five
   spot-check scenarios with the poisoned server, the case learned trust was built for (Gate M:
   55.3% vs 52.1%). Global-only reliability is clearly worse (−8 to −10 points with spot checks,
   W and M).
7. **Change detection is a trade-off.** Testing verified 0/1 outcomes is better than testing the
   soft learning targets in 6 of 41 scenarios and worse in 2 (Gate G drift with a misleading
   judge). The z-test cuts false resets from about 20 to about 2 per run against the original
   fixed-threshold rule (W, M), but still costs 2–3 points when nothing changes (W, M), reacts
   later to a real API change than that rule (G), and does little against staged defection.
8. **When the model already knows the answer, retrieval does not matter** (Gate F: reranker
   57.1% = no retrieval 57.2%).
9. **A tool that turns liar after earning trust erases RRL's advantage.** When the most useful
   honest extractor starts returning wrong results that still report success (Gate M, spot
   checks), RRL loses 15 points and falls just behind plain per-site memory (36.8% vs 39.1%);
   no component of RRL protects against it.

## What drives the gains: learned trust or a fixed discount?

Replacing learned source reliability with the fixed feedback rule, everything else equal (RRL +
per-context + verified memory, 30 seeds, paired), in the 23 scenarios that have unverified
feedback: the two differ by −3.2 to +2.9 points; learned trust is significantly better in 5,
worse in 6 and indistinguishable in 12 (2 and 1 after a Holm correction across the 23 tests). It
helps a little where the verifier is sparse and other feedback mostly honest (sparse verifier on
Gate T: +1.6 to +2.9) and is worse in the case it was designed for: in all five spot-check
scenarios with the poisoned server on Gates M and W the fixed rule is ahead by 2.0–3.2 points
(Gate M spot check: 55.3% vs 52.1%). Learned trust is clearly better only than global-only
reliability (−8 to −10 points in Gates M and W). The gains over the baselines come from what both
variants share: unverified success is discounted, and only verified outcomes can lock in a
per-context choice. The full list is in Appendix B of `paper/main.pdf`, generated by
`paper/make_tables.py`.

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
| RRL combined | 45.9 / 57.2 | 42.6 / 63.7 | 42.5 / 51.2 | 36.6 / 49.0 |
| RRL combined + per-context | 57.1 / 67.9 | 47.2 / 68.5 | 48.8 / 45.4 | 39.8 / 53.3 |
| RRL + per-context + verified memory | 66.3 / 73.9 | 52.1 / 70.0 | 55.9 / 56.5 | 43.2 / 55.6 |
| same, fixed feedback rule instead of learned trust | 66.3 / 73.9 | 55.3 / 69.0 | 55.9 / 56.5 | 46.3 / 53.6 |
| best possible | 90.4 / 98.8 | 90.4 / 98.8 | 90.4 / 98.8 | 90.4 / 98.8 |


Poisoned tool chosen (share of steps), all calls verified / 20% verified: static reranker
100% / 100%, Gemini 99% / 99%, MemRL-style 4% / 90%, RRL + per-context + verified memory
1.4% / 14.2% (with the fixed feedback rule 1.4% / 12.8%).

### Gate M — live MCP servers (replayed), clean registry

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | spot_check | update | update_spot_check |
|---|---|---|---|---|
| static reranker | 46.0 / 65.7 | 46.0 / 65.7 | 37.4 / 49.3 | 37.4 / 49.3 |
| Gemini picks from tool list | 47.7 / 61.8 | 47.7 / 61.8 | 49.7 / 53.8 | 49.7 / 53.8 |
| reuse what worked | 77.5 / 89.4 | 77.0 / 89.4 | 70.4 / 85.4 | 69.2 / 85.4 |
| reuse what worked, robust | 81.8 / 90.3 | 81.6 / 90.3 | 72.5 / 86.0 | 71.3 / 86.0 |
| MemRL-style | 34.9 / 45.6 | 34.2 / 40.8 | 34.1 / 31.7 | 31.7 / 26.9 |
| RRL combined | 48.0 / 61.9 | 48.2 / 61.8 | 44.1 / 51.9 | 45.2 / 56.0 |
| RRL combined + per-context | 58.2 / 67.2 | 57.5 / 67.1 | 49.9 / 50.3 | 49.3 / 57.1 |
| RRL + per-context + verified memory | 66.3 / 72.4 | 60.0 / 72.2 | 55.8 / 56.0 | 51.2 / 58.8 |
| same, fixed feedback rule instead of learned trust | 66.3 / 72.4 | 59.6 / 70.1 | 55.8 / 56.0 | 51.1 / 57.1 |
| best possible | 90.4 / 98.8 | 90.4 / 98.8 | 90.4 / 98.8 | 90.4 / 98.8 |


Without the poisoned server, Gemini picks a sensible tool (48–50%, level with the reranker
before the update and ahead after it), and per-site "reuse what worked" is best (69–78%).

### Gate M — staged defection

In the *defect* scenarios the most useful honest extractor (`css_woo`) starts returning
well-formed wrong results halfway through the seen shops, while still reporting success and
passing the schema check (a TrustShift-style attack). Replay only.

### Gate M — staged defection, with the poisoned server

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | defect | defect_spot_check |
|---|---|---|
| static reranker | 0.0 / 0.0 | 0.0 / 0.0 |
| Gemini picks from tool list | 0.5 / 0.0 | 0.5 / 0.0 |
| reuse what worked | 60.5 / 48.3 | 39.1 / 16.8 |
| reuse what worked, robust | 61.9 / 48.6 | 17.6 / 8.9 |
| MemRL-style | 34.0 / 57.6 | 3.0 / 0.0 |
| RRL combined | 38.6 / 62.8 | 30.5 / 56.4 |
| RRL combined + per-context | 42.1 / 64.7 | 33.9 / 63.9 |
| RRL + per-context + verified memory | 50.4 / 63.3 | 36.2 / 62.6 |
| same, fixed feedback rule instead of learned trust | 50.4 / 63.3 | 39.6 / 60.3 |
| best possible | 83.6 / 82.2 | 83.6 / 82.2 |

### Gate M — staged defection, clean registry

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | defect | defect_spot_check |
|---|---|---|
| static reranker | 38.3 / 65.7 | 38.3 / 65.7 |
| Gemini picks from tool list | 40.0 / 61.8 | 40.0 / 61.8 |
| reuse what worked | 66.7 / 64.4 | 63.0 / 64.4 |
| reuse what worked, robust | 68.2 / 65.3 | 63.7 / 65.3 |
| MemRL-style | 33.8 / 58.5 | 17.2 / 4.0 |
| RRL combined | 39.8 / 61.7 | 37.6 / 60.6 |
| RRL combined + per-context | 43.8 / 65.3 | 40.6 / 62.1 |
| RRL + per-context + verified memory | 52.0 / 62.2 | 42.9 / 63.3 |
| same, fixed feedback rule instead of learned trust | 52.0 / 62.2 | 42.5 / 59.7 |
| best possible | 83.6 / 82.2 | 83.6 / 82.2 |

Every learner loses. Under spot checks RRL + per-context + verified memory loses 15.3 points
against the same scenario without defection (best possible: 6.7); it still beats the robust
memory (+19.2) but falls behind plain "reuse what worked" (36.8% vs 39.1%, 9/30 seeds). Without
the poisoned server the robust memory keeps its lead (63.7% vs 42.9%). The change detector now
tests verified outcomes, but at 20% verification a tool needs about 80 calls to fill two
windows of 8 verified outcomes and it fires 0.8 times per run; RRL combined solves 31.6% with it
and 31.2% without. The fixed feedback rule does not help either (40.0%).

**Live check.** The same policies run with real MCP calls on every step (3 seeds):

| Policy | stationary: replay (30 seeds) | live (3 seeds) | update: replay | live |
|---|---|---|---|---|
| static reranker | 0.0 | 0.0 | 0.0 | 0.0 |
| Gemini picks from tool list | 1.0 | 1.0 | 8.2 | 8.3 |
| reuse what worked | 70.9 | 77.2 | 62.9 | 69.4 |
| reuse what worked, robust | 75.0 | 77.2 | 65.7 | 67.9 |
| MemRL-style | 35.8 | 37.7 | 33.8 | 35.0 |
| RRL combined | 45.9 | 47.4 | 42.5 | 44.3 |
| RRL + per-context | 57.1 | 57.5 | 48.8 | 48.4 |
| RRL + per-context + verified memory | 66.3 | 65.7 | 55.9 | 56.9 |

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
| RRL combined | 52.5 / 50.1 | 50.4 / 49.6 | 45.4 / 45.4 | 44.5 / 43.6 | 48.1 / 48.7 | 41.5 / 42.1 |
| RRL combined, explore 0.10 | 56.4 / 51.9 | 54.3 / 52.0 | 47.6 / 49.1 | 46.9 / 46.0 | 51.0 / 51.6 | 43.4 / 45.2 |
| RRL combined + per-context | 52.3 / 50.0 | 50.4 / 49.6 | 45.3 / 45.8 | 44.3 / 43.1 | 47.9 / 48.1 | 41.5 / 43.7 |
| RRL + per-context + verified memory | 55.3 / 48.4 | 50.9 / 49.9 | 46.2 / 46.3 | 45.2 / 42.8 | 49.5 / 48.5 | 41.8 / 42.8 |
| same, fixed feedback rule instead of learned trust | 55.3 / 48.4 | 48.0 / 46.6 | 45.7 / 44.9 | 44.7 / 43.1 | 49.5 / 48.5 | 41.2 / 41.7 |
| best possible | 89.9 / 89.4 | 89.9 / 89.4 | 89.9 / 89.4 | 89.9 / 89.4 | 88.6 / 87.1 | 88.6 / 87.1 |


### Gate T — tool selection, ToolBench (ToolRet) — ablation

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sparse_verifier | sycophantic | fooled | drift | drift_fooled |
|---|---|---|---|---|---|---|
| RRL, reranker input only | 52.5 / 50.1 | 47.2 / 47.2 | 44.9 / 43.0 | 43.9 / 43.1 | 48.1 / 48.7 | 40.9 / 42.6 |
| combined, global-only reliability | 52.5 / 50.1 | 50.4 / 49.4 | 45.5 / 45.9 | 44.4 / 42.6 | 48.1 / 48.7 | 41.6 / 42.9 |
| combined, no change detection | 52.5 / 50.1 | 50.4 / 49.6 | 45.4 / 45.4 | 44.5 / 43.6 | 48.1 / 48.7 | 41.5 / 42.1 |
| combined, old detector (w=4, 0.6) | 52.5 / 50.1 | 50.4 / 49.6 | 45.4 / 45.4 | 44.5 / 43.6 | 48.1 / 48.7 | 41.5 / 42.1 |
| combined, exploration off | 58.9 / 50.9 | 57.0 / 50.4 | 48.5 / 47.2 | 47.6 / 45.6 | 52.9 / 52.0 | 44.5 / 46.5 |
| combined, confidence-gated exploration | 52.5 / 49.8 | 50.4 / 49.7 | 45.4 / 45.3 | 44.5 / 43.6 | 48.0 / 48.1 | 41.5 / 42.1 |
| per-context + confidence-gated exploration | 52.3 / 49.5 | 50.4 / 49.3 | 45.3 / 46.3 | 44.3 / 43.1 | 48.0 / 48.9 | 41.5 / 43.7 |


### Gate T — tool selection, ToolLens (ToolRet)

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sparse_verifier | sycophantic | fooled | drift | drift_fooled |
|---|---|---|---|---|---|---|
| static reranker | 29.1 / 32.2 | 29.1 / 32.2 | 29.1 / 32.2 | 29.1 / 32.2 | 26.5 / 25.2 | 26.5 / 25.2 |
| reuse what worked | 42.9 / 32.2 | 42.9 / 32.2 | 38.2 / 32.2 | 34.7 / 32.2 | 36.2 / 25.2 | 29.7 / 25.2 |
| reuse what worked, robust | 43.5 / 32.2 | 43.5 / 32.2 | 37.1 / 32.2 | 34.3 / 32.2 | 35.7 / 25.2 | 28.6 / 25.2 |
| MemRL-style | 55.4 / 57.9 | 54.3 / 56.8 | 51.7 / 56.3 | 43.9 / 50.3 | 52.4 / 55.4 | 40.6 / 42.0 |
| RRL combined | 56.1 / 55.6 | 53.4 / 56.0 | 48.1 / 51.5 | 45.7 / 49.3 | 49.9 / 52.0 | 41.6 / 45.8 |
| RRL combined, explore 0.10 | 59.1 / 60.3 | 56.4 / 59.5 | 50.5 / 55.9 | 48.0 / 50.6 | 52.2 / 53.1 | 42.9 / 41.9 |
| RRL combined + per-context | 56.1 / 55.9 | 53.4 / 57.1 | 48.0 / 52.1 | 45.5 / 47.7 | 49.9 / 51.9 | 41.4 / 45.5 |
| RRL + per-context + verified memory | 58.1 / 55.8 | 53.8 / 55.5 | 48.2 / 51.2 | 46.5 / 49.5 | 50.6 / 52.5 | 41.5 / 43.7 |
| same, fixed feedback rule instead of learned trust | 58.1 / 55.8 | 52.1 / 53.9 | 49.4 / 53.3 | 47.2 / 50.5 | 50.6 / 52.5 | 42.3 / 45.8 |
| best possible | 85.1 / 81.1 | 85.1 / 81.1 | 85.1 / 81.1 | 85.1 / 81.1 | 84.7 / 83.2 | 84.7 / 83.2 |


### Gate T — tool selection, ToolLens (ToolRet) — ablation

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sparse_verifier | sycophantic | fooled | drift | drift_fooled |
|---|---|---|---|---|---|---|
| RRL, reranker input only | 56.1 / 55.6 | 51.5 / 53.5 | 48.8 / 51.2 | 46.6 / 51.0 | 49.6 / 50.8 | 42.0 / 45.2 |
| combined, global-only reliability | 56.1 / 55.6 | 53.5 / 55.6 | 48.6 / 52.4 | 46.1 / 49.4 | 49.9 / 52.0 | 41.7 / 44.0 |
| combined, no change detection | 56.1 / 55.6 | 53.4 / 56.0 | 48.1 / 51.5 | 45.7 / 49.3 | 49.6 / 50.8 | 41.6 / 45.8 |
| combined, old detector (w=4, 0.6) | 56.0 / 55.2 | 53.5 / 56.5 | 48.1 / 51.4 | 45.7 / 49.2 | 50.3 / 51.5 | 41.9 / 45.7 |
| combined, exploration off | 60.2 / 59.8 | 57.6 / 59.8 | 51.7 / 55.8 | 48.9 / 52.5 | 53.4 / 50.2 | 42.6 / 37.8 |
| combined, confidence-gated exploration | 56.3 / 57.1 | 53.8 / 57.4 | 48.0 / 50.7 | 45.7 / 49.6 | 49.9 / 51.5 | 41.6 / 45.7 |
| per-context + confidence-gated exploration | 56.3 / 55.2 | 53.6 / 58.3 | 48.2 / 53.6 | 45.5 / 47.4 | 49.7 / 51.7 | 41.4 / 45.2 |


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
| RRL combined | 51.4 / 68.9 | 47.1 / 75.4 | 45.5 / 53.5 | 39.5 / 53.1 |
| RRL combined, explore 0.10 | 52.6 / 72.4 | 49.4 / 76.4 | 46.2 / 54.2 | 41.9 / 60.3 |
| RRL combined + per-context | 58.2 / 76.5 | 49.6 / 77.1 | 50.7 / 57.6 | 41.9 / 57.6 |
| RRL + per-context + verified memory | 62.0 / 79.6 | 52.1 / 81.8 | 54.1 / 60.7 | 44.4 / 61.0 |
| same, fixed feedback rule instead of learned trust | 62.0 / 79.6 | 54.1 / 77.6 | 54.1 / 60.7 | 46.6 / 61.0 |
| best possible | 87.9 / 96.4 | 87.9 / 96.4 | 87.9 / 96.4 | 87.9 / 96.4 |


### Gate W — web-scraping extractor selection — ablation

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | spot_check | redesign | redesign_spot_check |
|---|---|---|---|---|
| RRL, reranker input only | 53.9 / 78.2 | 47.3 / 74.3 | 48.0 / 64.9 | 40.2 / 59.6 |
| combined, global-only reliability | 51.4 / 68.9 | 38.1 / 56.5 | 45.5 / 53.5 | 29.7 / 24.4 |
| combined, no change detection | 53.9 / 78.2 | 47.3 / 76.8 | 48.0 / 64.9 | 40.0 / 56.9 |
| combined, old detector (w=4, 0.6) | 44.6 / 60.3 | 40.5 / 61.3 | 39.7 / 39.4 | 34.1 / 31.9 |
| combined, exploration off | 51.7 / 66.0 | 48.4 / 71.5 | 45.2 / 49.3 | 40.8 / 56.1 |
| combined, confidence-gated exploration | 51.4 / 68.9 | 47.1 / 75.7 | 45.5 / 53.5 | 39.5 / 53.8 |
| per-context + confidence-gated exploration | 58.2 / 76.5 | 49.6 / 77.1 | 50.7 / 57.8 | 41.9 / 57.6 |


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
| RRL combined | 96.5 / 100.0 | 90.7 / 99.7 | 87.8 / 98.9 | 71.7 / 96.7 | 51.3 / 56.2 | 54.2 / 70.8 |
| RRL combined, explore 0.10 | 97.7 / 100.0 | 92.7 / 99.8 | 90.4 / 98.9 | 75.5 / 95.0 | 50.3 / 50.7 | 55.5 / 66.8 |
| RRL combined + per-context | 96.5 / 100.0 | 90.6 / 99.8 | 87.5 / 98.8 | 70.0 / 95.6 | 50.7 / 57.5 | 53.3 / 71.2 |
| RRL + per-context + verified memory | 96.5 / 100.0 | 90.8 / 99.9 | 87.8 / 99.4 | 68.2 / 97.9 | 48.7 / 55.2 | 49.6 / 67.4 |
| same, fixed feedback rule instead of learned trust | 96.5 / 100.0 | 90.4 / 98.8 | 88.0 / 97.4 | 68.2 / 97.9 | 48.6 / 50.4 | 48.8 / 64.0 |
| best possible | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |


### Gate G hard — private-API docs, poisoned reranker — ablation

Overall / held-out pass rate (%), mean of 30 seeds.

| Policy | stationary | sycophantic | fooled | drift | drift_sycophantic | drift_fooled |
|---|---|---|---|---|---|---|
| RRL, reranker input only | 96.5 / 100.0 | 90.3 / 98.5 | 87.5 / 97.9 | 62.5 / 77.1 | 51.1 / 52.0 | 52.1 / 68.4 |
| combined, global-only reliability | 96.5 / 100.0 | 90.8 / 99.7 | 87.9 / 98.9 | 71.7 / 96.7 | 51.4 / 57.5 | 54.4 / 71.3 |
| combined, no change detection | 96.5 / 100.0 | 90.7 / 99.7 | 87.8 / 98.9 | 62.5 / 77.1 | 51.3 / 56.2 | 54.2 / 70.8 |
| combined, old detector (w=4, 0.6) | 96.5 / 100.0 | 90.7 / 99.7 | 87.8 / 98.8 | 80.6 / 99.1 | 62.4 / 80.2 | 64.2 / 82.5 |
| combined, exploration off | 97.7 / 100.0 | 91.1 / 98.8 | 89.5 / 97.4 | 76.8 / 100.0 | 45.4 / 36.8 | 50.8 / 52.1 |
| combined, confidence-gated exploration | 96.4 / 100.0 | 90.7 / 99.8 | 87.8 / 99.1 | 72.1 / 98.2 | 51.3 / 56.0 | 54.3 / 72.9 |
| per-context + confidence-gated exploration | 96.4 / 100.0 | 90.7 / 100.0 | 87.6 / 99.2 | 70.4 / 97.2 | 50.4 / 56.8 | 53.2 / 71.2 |


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
| RRL combined | 58.4 | 58.0 | 57.5 | 57.1 |
| RRL + per-context | 58.7 | 58.4 | 57.8 | 57.3 |
| RRL + per-context + verified memory | 59.3 | 58.5 | 58.2 | 57.8 |
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
- **Seed-level intervals** vary request order, noise and exploration over the same contexts;
  only the primary Gate M comparison also has a shop-level bootstrap (`paper/site_stats.py`).
- **A verified success stays authoritative** for a context until a verified failure; staged
  defection shows the cost. Version-aware memory, periodic audits and a limit on how long a
  success counts without re-verification are untested.

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
