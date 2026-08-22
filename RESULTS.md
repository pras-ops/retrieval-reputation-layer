# RRL results

All numbers below are reproducible offline with no API key. Ranking metrics need no LLM at
all, so they are measured at n = 4,800 query observations (10 seeds × 16 epochs × 30
queries) rather than the 10 per-seed means a task pass rate gives you.

```bash
python3 sim/run_bench.py --seeds 10 --epochs 16 --verifier oracle --baseline "B0 dense only"
python3 sim/run_bench.py --seeds 10 --epochs 16 --verifier cache  --baseline "B0 dense only"
python3 sim/run_delta_sweep.py --seeds 8 --epochs 120
python3 sim/run_staleness.py --seeds 8 --epochs 40
```

---

## 0. Withdrawn: the previous pass-rate results

The earlier headline numbers (+2.9% overall / +4.8% late-stage pass rate) are **withdrawn
as unmeasured**. The replay cache they came from mixed mock and real outcomes:

| signature | rows | label |
| :--- | ---: | :--- |
| completion is byte-for-byte `'    pass\n'` (mock's wrong-evidence branch) | 585 | all `0.0` |
| completion identical to the retrieved document (mock's right-evidence branch) | 73 | all `1.0` |
| **mock-generated total** | **658 / 1600 (41.1%)** | |

`--mock` shared `save_to_cache` with the real path, so a plumbing self-test wrote synthetic
labels into the reproducibility cache. The contamination is not noise: mock labels are a
*perfect function of retrieval correctness*, which fabricates exactly the "RRL learns to
retrieve the right document, so pass rate rises" result the benchmark was testing for.

Contaminated rows now live in `data/quarantine/`. `sim/outcome_cache.py` enforces separate
files for mock and real outcomes, requires a `generator`/`verifier` tag on every row, and
makes a strict replay hard-fail on anything untagged. `sim/quarantine_cache.py --apply`
reproduces the split. The 942 recovered rows are tagged `recovered-legacy` — recovered by
signature, not by recorded provenance — and should be regenerated before publication.

---

## 1. Four defects made outcome feedback inert, then backwards

Each is pinned by a regression test in `tests/test_regressions.py` that fails on the
pre-fix code (verified: 5/5 fail before, 15/15 pass after).

**Clock base.** `Candidate.__post_init__` tested `last_confirmed` for falsiness, so an
explicit `0.0` — a perfectly valid simulated timestamp — was overwritten with wall-clock
time (~1.79e9). Every simulation therefore computed `dt = step − 1.79e9 < 0` and **decay
never ran at all**. Fixed with `is None`, a `last_feedback` anchor, and a `Clock`
abstraction (`rrl/clock.py`) so production uses wall time and experiments use simulated
time without either touching the scoring path.

**Read-triggered decay.** `rescore` decayed α/β from a stale anchor and wrote the result
back without advancing the anchor, so decay re-applied on every read and compounded as
γ^Σ(tᵢ−t₀) — quadratic in read count rather than linear in elapsed time. Measured: nine
reads took α from 6.0 to **1.04**, where the correct value is 2.15. Decay is now computed
read-only via `Candidate.effective_counters()`; the stored record moves only on feedback.

**Asymmetric anchor.** `last_confirmed` advanced only when `y > 0.5`. A document that had
ever succeeded fell into the decay clock and lost its reward; a document that only ever
failed kept its original anchor and its penalty **never aged**. Successes were forgotten,
failures were immortal. `last_confirmed` now keeps its narrow meaning and a separate
`last_feedback` — advanced on every observation — is the decay anchor.

**Exploration outweighed relevance.** The benchmark ran `weights=(0.20, 0.40, 0.10, 0.30)`
with the term `w_explore · sim · (Beta sample + rarity)`, whose range is 1.5× the relevance
term it perturbs. Because the three defects above pinned α,β at the prior, `Beta(1,1) =
Uniform(0,1)` **forever** — the sampler never sharpened. Added `exploration_mode="ts"`
(pure Thompson sampling, bounded by `w_explore`), a warm-up phase, and a constructor
warning when `w_explore > w_sim/2`.

---

## 2. The comparison was not controlled

The RRL arm scored the **entire ~90-document corpus** at `w_sim=0.20`; the baseline
reranked a top-5 shortlist. That measures candidate-set size, not ranking quality.
`shortlist_k` now gates the layer to the same shortlist the baseline sees, and
`base_scores` lets it stack on an external ranker instead of competing with one.

Consequence, measured (Hit@1, n=4,800):

| configuration | Hit@1 | trend over 16 epochs |
| :--- | ---: | :--- |
| dense similarity only | 53.7% | flat |
| cross-encoder rerank | 46.7% | flat |
| RRF hybrid | 31.3% | flat |
| **RRL as originally shipped** | **24.4%** | **25.3% → 22.7% (declines)** |

As shipped, RRL retrieved the correct reference *less often than plain similarity* and got
worse with experience.

**A second finding worth acting on independently:** on this corpus **plain dense
similarity beats the cross-encoder** (53.7% vs 46.7%, paired p=0.0095), and RRF hybrid is
far worse than either (31.3%) — BM25 actively hurts on code retrieval. The original
benchmark was built on RRF, i.e. on a badly weakened base retriever.

---

## 3. Fixed and layered on the strongest base ranker, it wins

10 seeds × 16 epochs, shortlist 5, oracle verifier, paired against dense-only:

| arm | Hit@1 | MRR | nDCG@5 | ep1 → ep16 | vs B0 |
| :--- | ---: | ---: | ---: | :--- | ---: |
| B0 dense only | 53.7 [46.8, 60.5] | 0.579 | 0.592 | 53.7 → 53.7 | — |
| B1 cross-encoder | 46.7 [39.5, 53.9] | 0.530 | 0.555 | 46.7 → 46.7 | −7.0 (p=0.010) |
| B2 RRF hybrid | 31.3 [25.8, 36.9] | 0.422 | 0.472 | 31.3 → 31.3 | −22.3 (p=0.0001) |
| B3 RRF + **static** prior | 26.7 [22.6, 30.7] | 0.393 | 0.451 | 26.7 → 26.7 | −27.0 (p<0.0001) |
| B7 CE + RRL + pooling | 50.5 [44.3, 56.7] | 0.550 | 0.570 | 47.3 → 52.0 | −3.2 (n.s.) |
| B8 RRF + RRL + pooling | 48.7 [43.9, 53.4] | 0.535 | 0.557 | **26.0 → 56.0** | −5.0 (p=0.040) |
| B10 dense + RRL | 57.7 [52.7, 62.7] | 0.600 | 0.606 | 53.3 → 59.7 | **+4.0 (p=0.0050)** |
| **B11 dense + RRL + pooling** | **58.6 [53.7, 63.6]** | **0.605** | **0.610** | 53.3 → 60.3 | **+5.0 (p=0.0029)** |
| B12 dense + RRL + topic pooling | 58.1 [53.2, 63.0] | 0.602 | 0.608 | 53.3 → 60.3 | +4.5 (p=0.0037) |

Three things this separates that the old benchmark conflated:

* **Adaptivity is what pays, not having a prior.** B3 attaches a frozen, query-independent
  document quality prior and is the *worst* arm in the table. The gain comes from updating
  online, not from the existence of a reputation term.
* **The layer improves whatever base it is given, but does not replace a good one.** From
  RRF (31.3%) it climbs to 56.0% by epoch 16 — a +24.7 point recovery that overtakes the
  cross-encoder — yet still loses to dense-only. Stack it on the best available ranker.
* **Query-conditional pooling helps, but modestly.** +0.9 pts over global counters
  (58.6 vs 57.7). Global counters are *unbiased in this corpus by construction* — the
  distractors are never correct for any query — so this understates what pooling is for.
  Testing it properly needs a corpus where a document is right for some queries and wrong
  for others; that corpus does not exist here and building it is future work.

---

## 4. The binding constraint is verifier diagnosticity

Define Δ = P(tests pass | correct document) − P(tests pass | wrong document). Measured on
the 942 recovered rows:

| verifier | P(pass \| correct) | P(pass \| wrong) | Δ |
| :--- | ---: | ---: | ---: |
| binary pass/fail | 0.762 | 0.517 | **0.245** |
| graded per-assert fraction | 0.775 | 0.542 | **0.233** |

Swap the oracle for this verifier and every learning gain disappears — B11 goes from
**+5.0 pts (p=0.0029) to −1.0 pts (n.s.)**, with a flat curve (53.3 → 52.3).

### 4a. Densifying the reward does not help — a negative result

Grading each completion per individual assert (`sim/grade_cache.py`, fully offline, no API
calls) recovers partial credit on 4.7% of rows and **moves Δ from 0.245 to 0.233 — i.e.
slightly worse**. Δ is set by how often the model succeeds *without* the right evidence
(P=0.517), which is a property of task difficulty against model capability, not of output
granularity. Making the reward finer-grained cannot fix it.

What *does* move Δ is task selection:

| task set | tasks | Δ |
| :--- | ---: | ---: |
| all | 78 | 0.318 |
| drop tasks solvable without correct evidence | 42 | 0.476 |
| evidence-sensitive only | 24 | **0.810** |

**46% of tasks are solved regardless of which document is retrieved, and another 23% fail
regardless.** About 69% of this benchmark carries no learnable signal — and those tasks do
not merely dilute the measurement, they set the sample-complexity floor for every other
task. (One reward densification remains untested here because it needs generator logprobs
rather than task outcomes: using the model's log-likelihood of the reference as the reward,
as in PURPLE, arXiv:2601.12078.)

### 4b. Δ governs both the rate and the ceiling

`sim/run_delta_sweep.py`: real corpus, real embeddings, real shortlists; the verifier is
synthetic and calibrated so that only the signal gap changes (base rate held at 0.5).
8 seeds × 120 epochs, against a static dense baseline of 52.1%.

| Δ | crossover epoch | final Hit@1 | gain |
| ---: | ---: | ---: | ---: |
| 0.05 | **never** (120 ep) | 52.9% | +0.8 |
| 0.10 | 34 | 54.6% | +2.5 |
| 0.15 | 46 | 54.6% | +2.5 |
| 0.245 *(measured)* | 13 | 56.2% | +4.2 |
| 0.35 | 8 | 57.9% | +5.8 |
| 0.50 | 6 | 59.2% | +7.1 |
| 0.70 | 3 | 59.6% | +7.5 |
| 1.00 | 2 | 60.0% | +7.9 |

Rank correlation between the predicted and observed crossover epoch: **ρ = 0.964,
p = 0.0005**. Two readings:

* **The gain appears to saturate around Δ ≈ 0.5** (+7.1 → +7.9 from Δ=0.5 to Δ=1.0).
  **Superseded — see §8.3:** on a benchmark where the correct document is always present in
  the candidate set, the gain is linear in Δ all the way to 1.0. The saturation here was a
  shortlist-recall artifact, not a property of Δ.
* **Below Δ ≈ 0.1 the mechanism does not converge at all** within 120 epochs. That is a
  genuine floor, not slow progress.

### 4c. Correction to an earlier claim of ours

An earlier analysis in this project derived an analytic requirement of ≈56 observations per
document at Δ=0.245 and concluded the benchmark was "under-sampled 18×". **The sweep shows
that bound is far too pessimistic in absolute terms** — it predicts 327 epochs where the
measured crossover is 13. The bound is a two-sample hypothesis test for declaring one
document better with 80% power; a ranker does not need significance, it needs the argmax to
be right often enough, and it already has a similarity prior doing most of the work. The
bound predicts the *ordering* across verifiers well (ρ=0.96) and should be reported only
that way. Quote it as "under our measured Δ, ~56 observations to *statistically separate*
two documents", never as the recurrence a working system needs.

The honest diagnosis of the original benchmark is simpler: it ran **8 epochs**, and at its
own Δ=0.245 the crossover is around epoch 13. It measured before the mechanism could work.
With the exact measured verifier (p₁=0.762, p₂=0.517) and 120 epochs, the layer does reach
**+3.7 points**.

---

## 5. Staleness recovery: what decay is actually for

`sim/run_staleness.py`. The correct document for each query switches at the halfway point
to a successor inside the same shortlist. Relevance never changes, so a static ranker
cannot react by construction. 8 seeds × 40 epochs, oracle verifier.

| arm | Hit@1 | pre-switch | post-switch | final | demoted | demotion lag |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| static dense | 50.0% | 84.6% | 15.4% | 15.4% | 15% | — |
| RRL, γ=1.0 (no decay) | 51.4% | 98.7% | 1.3% | 15.7% | 16% | 11.1 visits |
| **RRL, γ=0.99** | **58.5%** | 96.0% | 7.0% | **35.9%** | **38%** | **7.6 visits** |
| RRL, γ=0.95 | 52.9% | 91.2% | 10.6% | 19.9% | 27% | 6.1 visits |
| RRL, γ=0.80 | 49.9% | 90.4% | 8.6% | 16.9% | 21% | 4.9 visits |

There is a real decay optimum, and it only becomes visible now that decay functions: γ=1.0
cannot unlearn (final 15.7%, no better than static), γ=0.80 destroys memory faster than it
accumulates, and γ=0.99 recovers 2.3× as much as static. Recovery is **partial** — 35.9%
against 96% pre-switch, and only 38% of queries abandon the stale document at all — because
the successor has lower similarity and `w_sim=0.70` dominates. Full recovery would need a
larger `w_c` or more post-switch recurrence. Reported as measured, not as a solved problem.

*Note on terminology:* `demotion_lag` counts **observations until a stale document stops
being selected**. Feedback Adaptation for RAG (arXiv:2604.06647) uses "correction lag" for
the wall-clock latency until an updated index is queryable. Different quantities; the names
are kept distinct deliberately.

---

## 6. What still holds from the earlier work

Gates A, B, C and D are untouched by the cache contamination — they never used the replay
cache. Gate C's boundary (a static reranker wins without recurrence) is *reinforced*: it is
the Δ→0 / no-recurrence corner of the sweep in §4b.

## 7. Honest limitations

* The 942 recovered cache rows are classified by mock signature, not by recorded
  provenance. Regenerate before publishing any pass-rate number from them.
* Pass-rate columns in §3 run at ~48% cache coverage. Misses are reported as reduced
  coverage and never imputed; treat those columns as indicative only.
* The evidence-sensitive task filter (§4a) cannot be tested end-to-end on the current
  cache — only ~3.6 queries per seed survive it. Its prediction is tested indirectly via
  the calibrated sweep in §4b, not directly.
* Query-conditional pooling is under-tested for the reason given in §3.
* The Δ sweep uses a synthetic verifier. Retrieval, corpus and shortlists are real; the
  outcome channel is not.
* **B9's noise model is mis-specified and should not be read as a validated noise
  correction.** A unit-test verifier is essentially exact about "did this code pass", so
  there is no meaningful false-positive rate to correct. B9 feeds it `rho_fp = P(pass |
  wrong document)`, which is not verifier error at all — it is *evidence-attribution*
  uncertainty wearing verifier error's clothes. The correction machinery itself is correct
  and unit-tested in isolation (`tests/test_regressions.py::TestGradedRewardAndNoise`), and
  the rates are additionally self-estimated from the same cache being evaluated, which is
  circular. Applying it properly needs a verifier that genuinely errs (an LLM judge, a
  flaky test suite) with rates measured on a held-out audited sample.
* **Failure attribution is implemented but untested end-to-end.** The taxonomy and the
  update gating exist and are unit-tested, but nothing classifies outcomes automatically,
  so every observation in §3–§5 is labelled `RETRIEVAL`. The attribution lever is available,
  not evaluated.


---

# Part II — Rebuilt on external ground truth

Part I diagnosed the mechanism on a corpus we built ourselves. Part II re-runs it on a
frozen benchmark whose correct documents are annotated by someone else, and answers the
verifier question with four real verifiers instead of one.

```bash
python3 sim/patch_evalplus_macos.py          # one-time, macOS only
python3 sim/regrade_mbpp_plus.py             # re-grade cached completions, no API calls
python3 sim/build_benchmark.py               # freeze the benchmark + hashed manifest
python3 sim/run_bench_v2.py --seeds 8 --epochs 64 --curve --tiers
python3 sim/run_bench_v2.py --seeds 8 --epochs 64 --delta-sweep 0.05,0.1,0.2,0.245,0.4,0.6,0.8,1.0
```

## 8.1 A stricter verifier does NOT help — hypothesis refuted

Hypothesis going in: Δ = 0.245 is low because MBPP's three asserts let *false passes*
through on wrong evidence; MBPP+ (~108 tests/problem, 36×) should catch them and sharpen Δ
for free. This is testable with no new generation, since the completions are already cached.

Re-graded 366 of the 942 cached completions (the ones whose task ids are among MBPP+'s 378):

| verifier | P(pass \| correct) | P(pass \| wrong) | Δ | Δ 95% CI |
| :--- | ---: | ---: | ---: | :--- |
| MBPP (3 asserts, binary) | 0.875 (n=24) | 0.637 (n=342) | **0.238** | [+0.096, +0.379] |
| MBPP (3 asserts, graded) | 0.875 | 0.650 | 0.225 | |
| MBPP+ (~108 tests, binary) | 0.750 | 0.599 | **0.151** | [−0.030, +0.331] |
| MBPP+ (~108 tests, graded) | 0.817 | 0.709 | 0.108 | |

**Δ gets worse, and MBPP+'s CI now includes zero.** MBPP+ does catch false passes — 47 of
239 MBPP passes are overturned, 19.7% — but the decisive question is whether they are
*concentrated* on wrong evidence:

| condition | false-pass rate |
| :--- | ---: |
| correct evidence | 4/21 = **0.190** [0.077, 0.400] |
| wrong evidence | 43/218 = **0.197** [0.150, 0.255] |
| difference | **+0.007** [−0.169, +0.183], Fisher exact **p = 1.000** |

Identical. A stricter verifier removes passes from both conditions at the same rate, so it
cannot sharpen the evidence signal — it only lowers both arms, and with n=24 on the correct
side that shows up as a *drop* in Δ.

**What this establishes:** the model passing on wrong evidence is not subtly-wrong code
slipping past weak tests. It is the model genuinely not needing the evidence. Verifier
strength is not the lever; task selection is. That was previously an inference from §4a —
it is now a measurement.

## 8.2 The frozen benchmark, and a trap it caught

`sim/build_benchmark.py` freezes a benchmark on [CodeRAG-Bench](https://arxiv.org/abs/2406.14497),
whose canonical document per problem is **manually annotated upstream** — so the ground
truth is not ours. What is ours, and declared in the manifest, is the distractor
construction: 1 correct + 4 same-family + 1 other-family + 1 irrelevant = 7 candidates,
fixed per task at a pinned seed. Definition files carry SHA-256 hashes and live apart from
any observed outcome, which is the structural fix for the contamination in §0.

**The trap:** the upstream canonical documents prefix the problem statement as a comment
(MBPP) or docstring (HumanEval). The "correct" document therefore *contains the query
verbatim*. First build:

| arm | Hit@1 |
| :--- | ---: |
| BM25 only | **99.7%** |
| dense only | 98.6% |
| cross-encoder | 99.4% |

Retrieval had collapsed into string matching and no reranking signal was measurable.
`strip_query_leakage()` removes the leading comment/docstring from **every** document,
correct and distractor alike, so no arm is advantaged. After stripping, BM25 falls from
99.7% to **26.3%** — that is how much of it was reading a copy of the query. Anyone using
this corpus for retrieval research should check for this.

## 8.3 External replication: RRL adds, and pooling finally proves out

297 tasks × 7 fixed candidates, 8 seeds, oracle verifier. Every arm ranks identical candidates.

| arm | Hit@1 | MRR | nDCG@5 | ep1 | ep8 | ep16 | ep64 | vs dense |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense only | 87.2 | 0.923 | 0.941 | 87.2 | 87.2 | 87.2 | 87.2 | — |
| BM25 only | 26.3 | 0.468 | 0.528 | flat | | | 26.3 | −60.9 |
| RRF hybrid | 50.2 | 0.692 | 0.763 | flat | | | 50.2 | −37.0 |
| cross-encoder | 76.8 | 0.864 | 0.893 | flat | | | 76.8 | −10.4 |
| dense + Thompson | 89.3 | 0.932 | 0.946 | 86.7 | 89.5 | 90.5 | 90.6 | +2.1 (p<0.0001) |
| dense + RRL | 88.9 | 0.930 | 0.945 | 86.4 | 89.0 | 90.2 | 90.9 | +1.7 (p<0.0001) |
| **dense + RRL + pooling** | **90.4** | **0.938** | **0.951** | 86.4 | 90.9 | 92.9 | **95.7** | **+3.2 (p<0.0001)** |
| CE + RRL + pooling | 83.4 | 0.892 | 0.914 | 77.4 | 84.1 | 87.1 | 89.2 | −3.8 |
| RRF + RRL + pooling | 58.9 | 0.729 | 0.789 | 48.5 | 58.8 | 65.8 | 70.8 | −28.3 |

Four things, and the third resolves a stated limitation from Part I:

* **The +5.0 result replicates on ground truth we did not annotate**: +3.2 pts at 16 epochs,
  rising to **+8.5 over the static baseline by epoch 64** (95.7 vs 87.2), monotone throughout.
* **Dense beats the cross-encoder again** (87.2 vs 76.8), independently confirming §2 on a
  different corpus with different candidate sets. Two corpora now agree: the cross-encoder
  is the wrong base ranker for this task.
* **Query-conditional pooling now clearly matters: +4.8 pts over global counters** (95.7 vs
  90.9 at epoch 64), against +0.9 in Part I. Part I flagged that as under-tested because its
  distractors were never correct for *any* query, so global counters were unbiased by
  construction. Here candidates are drawn from a shared 964-document pool, so a document
  genuinely is right for one query and wrong for others — the condition pooling exists for.
  The limitation is resolved, and pooling is vindicated.
* **The layer lifts a weak base a long way but does not rescue it**: RRF climbs +20.6
  (48.5 → 70.8) and CE +11.8 (77.4 → 89.2), yet neither passes dense-only. Stack on the best
  base ranker available.

Error composition confirms the distractor tiers are calibrated, and shows *where* the layer
works: tier-1 same-family distractors absorb ~90% of all errors, and RRL shifts errors
further onto them (89.5% → 94.2%) while cutting irrelevant-document errors from 5.3% to
1.6%. It clears the easy confusions; the residual is genuine same-family ambiguity.

## 8.4 Δ → gain is linear, and Part I's saturation was an artifact

Same frozen benchmark, same arm, verifier diagnosticity swept with a calibrated channel,
64 epochs, against the 87.2% static baseline:

| Δ | final Hit@1 | gain | ep4 | ep8 | ep16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 87.9% | +0.7 | 86.3 | 86.8 | 86.7 |
| 0.10 | 88.3% | +1.1 | 86.6 | 86.4 | 86.8 |
| 0.20 | 89.7% | +2.5 | 86.7 | 86.4 | 87.8 |
| **0.245** *(measured)* | **90.5%** | **+3.3** | 87.3 | 87.5 | 87.8 |
| 0.40 | 92.2% | +5.0 | 87.8 | 88.2 | 89.8 |
| 0.60 | 93.6% | +6.4 | 87.5 | 88.7 | 91.0 |
| 0.80 | 94.5% | +7.3 | 88.0 | 89.6 | 92.3 |
| 1.00 | 95.7% | +8.5 | 88.4 | 90.9 | 92.9 |

```
gain (Hit@1 pts) = 0.85 + 8.25 * Delta        R^2 = 0.961
```

**This corrects §4b.** Part I found the gain saturating past Δ ≈ 0.5 and read that as a
"good enough verifier" threshold. On a benchmark where the correct document is *always* in
the candidate set, the relationship is linear to Δ = 1.0. The saturation was a
shortlist-recall ceiling — the old shortlist sometimes did not contain the answer, so no
amount of verifier quality could help — not a property of diagnosticity. Retract the
threshold claim.

The law gives a falsifiable prediction: at the measured real verifier (Δ = 0.245) the
expected gain is **+2.9 to +3.3 points**, reached over tens of epochs. That is the number to
check once the cache is regenerated with real generation.

## 8.5 What Part II did not settle

* **Per-task Δ still needs generation.** MBPP+ grading gives ~108 tests per pair instead of
  1 bit, but only **24 tasks** have both a correct-evidence and a wrong-evidence observation
  in the existing cache, and the distribution stays lumpy (8 distinct values over 24 tasks).
  A real Δ_q distribution needs a fresh probe at **temperature > 0 with k ≥ 8 samples per
  (task, document) pair** — roughly 297 tasks × 7 docs × 8 ≈ 17k generations. Deterministic
  sampling cannot substitute: at temperature 0 a per-task rate is its own single sample.
* **Sample-split discipline is designed but unexercised.** When that probe runs, Δ_q must be
  estimated on samples 1–4 and RRL evaluated on samples 5–8, or the Δ-bucket result is
  selection on the dependent variable. Nothing here needed it yet because the sweep uses a
  calibrated channel rather than measured per-task rates.
* **Staleness is validated on Part I's corpus only** (§5, γ=0.99, 2.3× recovery). The port to
  the frozen benchmark is mechanical but not done, so treat §5 as v1 evidence.
* **The MBPP+ arm rests on n=24** correct-evidence rows. The false-pass concentration test
  (p=1.000) is the robust part; the Δ point estimates are not.
* **CodeSearchNet was evaluated and rejected**, not merely deferred: no verifiable outcome
  (so it cannot produce Δ at all), 32.8% of its docstrings are irrelevant to their own code
  by its authors' own count, and it has been public training data since 2019.
