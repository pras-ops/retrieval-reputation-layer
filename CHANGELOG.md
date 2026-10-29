# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Gate F** (`sim/run_gate_matrix.py`): a leak-free, fully replayable MBPP benchmark. The
  corpus never contains a query's own solution; every (query, candidate) outcome is
  pre-computed into a JSONL outcome table, so all policies are replayed offline and
  deterministically. Compares RRL against no retrieval, static RRF, a cross-encoder, a per-query
  "reuse what worked" memory, a MemRL-style utility ranker and an oracle, with paired t-tests
  and Wilcoxon tests over 30 seeds. Supports Gemini or a local Hugging Face model as generator;
  `--leaky` reproduces the Gate E corpus for comparison. A `--selftest` runs in CI.
- `RELATED_WORK.md`: MemRL, Memento, RRPO, LURE-RAG, StreamBench, CodeRAG-Bench, and a table of
  methods from other fields that address RRL's open problems.
- **`SourceReliability`** (`rrl/reliability.py`, opt-in via `ReputationLayer(reliability=...)`):
  learns each unverified feedback source's agreement with the verifier and combines unverified
  signals by a log-odds vote. Sources never checked against the verifier get no weight, so their
  feedback is skipped instead of trusted.
- **`ChangeDetector`** (`rrl/drift.py`, opt-in via `ReputationLayer(change_detector=...)`): resets a
  document's short-term counters when its recent outcomes shift abruptly.
- `update_counters_with_signals(..., y_override=..., outcome_fn=...)` to apply an externally
  combined outcome, per document when `outcome_fn` is given.
- `SourceReliability` also keeps per-document accuracies, shrunk toward the global ones, so a
  source that is reliable in general but fooled by one document is caught for that document.
- Gate G scenarios `fooled` and `drift_fooled`: users accept answers built from a poisoned guide
  90% of the time and are honest otherwise.
- Gate F scenarios (`--scenario stationary|sparse_verifier|sycophantic|drift|all`) that change only
  the feedback a policy sees or when usefulness shifts, so they replay the same outcome table. New
  policies: `rrl_ce` (reranker rank blended into the similarity input), `rrl_combined` (+ source
  reliability + change detection) and `rrl_query` (per-query counters shrunk toward the
  document's global reputation).
- `sim/plot_gate_f_map.py`: measured similarity-vs-usefulness map from a Gate F outcome table.
- **Gate G** (`sim/run_gate_g.py`): a benchmark where the documents carry knowledge the model
  lacks. A generated library, `zyn`, computes values from hidden formulas; tasks can only be
  solved by calling the right function with the right arguments. The corpus has look-alike
  sibling docs, poisoned "official guides" with the arguments swapped, and 1.x/2.0 docs for a
  breaking API change that can switch mid-stream. Generation is sampled (temperature 0.7, two
  samples per cell) and tests run against both library versions, so all scenarios replay from
  one outcome table.
- Gate G `--variant hard`: concept-specific, query-mirroring poisoned guides (after PoisonedRAG's
  black-box attack) that the cross-encoder ranks first for 79 of the 80 poisoned questions, and
  eight questions per concept of which two are held out until a final block, to measure transfer
  to unseen questions. Separate pools, outcome table and results files (`*_hard`).
- Gate F candidate pools are frozen to `data/gate_f_pools_*.json`, and the embedder and reranker
  run on CPU, so pools no longer vary with GPU numerics.
- **Gate T** (`sim/run_gate_t.py`): agent tool selection on real ToolRet requests and tool
  descriptions (ToolBench and ToolLens), with tool updates, poisoned tool twins, flaky tools and
  held-out requests. `--build` downloads ToolRet into `data/gate_t/raw/` (git-ignored; ToolRet
  states no license); only derived ids, scores and labels are committed.
- **Gate W** (`sim/run_gate_w.py`): web-scraping extractor selection with 8 executable extractors
  on 30 generated shops, site redesigns, held-out sites, and feedback produced by the tools
  themselves (self-reports, a schema validator, spot checks).
- `--ablation` on Gates F, G, T and W replays RRL variants with one component changed
  (global-only reliability, no or old change detection, long-term reset, exploration 0.10/0).
  Change-detector resets per run are reported.
- Policy `rrl_combined_ctx`: per-request (or per-site) counters shrunk toward each item's global
  counters.
- `run_policies` accepts `scenario["signals_fn"]` for benchmarks whose feedback comes from the
  tools rather than a simulated user.
- `sim/summarize_results.py` prints the RESULTS.md tables from `sim/results/*.json`.
- Gate F's 30-seed Gemini outcome table and results; `--timeouts-as-failures` records cells whose
  generation still times out after `--build` as failed (`"error": "timeout"`).
- `ChangeDetector.resets` and `ChangeDetector(reset_long_term=True)`; `ChangeDetector.check()`
  takes a cluster id and then checks that cluster's own counters.
- `ReputationLayer(cluster_strength=...)`.
- `pandas` and `pyarrow` in the `dev` extra, for Gate T's build step.

### Fixed
- The default exploration weight in `ReputationLayer` and `Retriever` was `0.0`, so the
  Thompson-sampling term never contributed to the score unless callers passed their own
  weights. Defaults are now `(0.55, 0.30, 0.15, 0.10)` and `(0.70, 0.20, 0.10, 0.10)`. Callers
  that pass explicit weights are unaffected.
- `RESULTS.md` named `gemini-2.5-pro`; the harness calls `gemini-2.5-flash`.
- `Retriever` broke ties in fused scores in hash order, so candidate rankings could differ
  between runs of the same code. Ties are now broken by candidate id.

### Changed
- `SourceReliability` is now a two-coin model: it learns each source's sensitivity and
  specificity against the verifier instead of its agreement rate, so a source that always says
  "worked" carries no weight even when most outcomes succeed. Unverified signals combine as the
  verifier's base rate plus each source's log-likelihood ratio. `doc_strength=None` keeps only
  global rates.
- `ChangeDetector` now defaults to a two-proportion z-test over two windows of 8 outcomes
  (α = 0.01), which avoids most false resets when an item's results vary by context. The
  original rule is `ChangeDetector(window=4, threshold=0.6)`.
- Cluster (per-context) counters are blended with the global ones by their own evidence,
  `n / (n + cluster_strength)` with `n = α_c + β_c − 2`, instead of `min(1, (A_c + B_c − 2) / 10)`,
  which used the slow long-term counters and left clusters almost unused.
- Gates A and B and the robustness ablation were re-run with the current code and their plots
  regenerated; Gate B's post-shift numbers are now 0.488 (decay off) vs 0.709 (decay on), the
  same conclusion as before.
- README and RESULTS.md describe the current formula, benchmarks and results; RESULTS.md now
  opens with Gates T, W and G and their ablations.
- `RESULTS.md` and the README now flag Gate E's answer-key leak and the fact that its RRL arm
  cannot be replayed from the committed cache.
- `RESULTS.md` and the Gate E entry in the README now state the realistic MBPP benchmark at the
  strength the data supports: with 10 seeds the confidence intervals overlap, so the lift over the
  cross-encoder baseline is directional, not yet statistically significant. Absolute lifts are
  labelled as percentage points.

### Fixed
- CI: the ruff rule set is pinned to `E4`, `E7`, `E9` and `F`. Newer ruff releases widened the
  default selection and failed the build on style suggestions.
- CI: mypy now ignores missing stubs for `google.*`, matching how `rrl/judge.py` imports
  `google.genai`.

## [0.2.1] — 2026-07-13

### Fixed
- Included `rrl.integrations` in setuptools packages configuration so that the LangChain and LlamaIndex adapters are correctly packaged into the wheel distribution.

## [0.2.0] — 2026-07-12


The layer becomes retriever-agnostic.

### Added
- **`ReputationLayer`** (`rrl/layer.py`) — the new core API. Feed it similarity scores from
  **any** retriever (`layer.rescore({"doc1": 0.9, ...})`), get reputation-adjusted ranking plus
  a `response_id`; close the loop with `layer.record_feedback(response_id, s_gt=..., ...)`.
  Works over both the in-memory and SQLite stores.
- `compute_credit_shares()` in `rrl/feedback.py` (moved out of the API handler).
- `save_pending` / `pop_pending` / `gc_pending` on the in-memory `CandidateStore`, matching the
  SQLite store's retrieve↔feedback bridge.
- Unknown candidate IDs passed to `rescore()` are tracked on the fly (`layer.track()`).

### Changed
- **Core install is now lightweight**: `pip install retrieval-reputation-layer` requires only
  numpy. `sentence-transformers` moved to the `[embeddings]` extra (lazy-imported); scipy and
  scikit-learn moved to `[dev]`.
- `Retriever` is now a thin wrapper: it computes hybrid (vector+BM25 RRF) similarities and
  delegates scoring/exploration to its `ReputationLayer` (`retriever.layer`). Public signature
  unchanged; `retriever.last_response_id` exposes the frozen credit-share handle.
- Query-conditional clustering now defaults **off** (`use_clustering=False`), matching the
  README's validation status — every validated result uses global counters.
- The FastAPI `/retrieve` and `/feedback` endpoints delegate to the layer instead of
  reimplementing credit shares and counter updates.

## [0.1.0] — 2026-06-22

Initial public release.

### Changed
- **Renamed the project `CAG` → `RRL` (Retrieval Reputation Layer)** to avoid the naming
  collision with *Cache-Augmented Generation*. The importable package is now `rrl`
  (`from rrl import Retriever, CandidateStore, OutcomeSignals, ...`); the PyPI distribution
  name is `retrieval-reputation-layer`. Environment variables are now `RRL_DB_PATH`,
  `RRL_DECAY_UNIT_SEC`, `RRL_GAMMA`, and the default DB file is `rrl.db`.

### Added
- `pyproject.toml` (PEP 621) — installable via `pip install -e ".[api,llm,dev]"`.
- GitHub Actions CI running the test suite on Python 3.10–3.13.
- This `CHANGELOG.md` and `CONTRIBUTING.md`.
- The realistic recurrence harness (`sim/run_gate_recurring.py`) and its MBPP dataset
  (`data/mbpp.jsonl`) are now tracked in git, so README references resolve from a fresh clone.

### Fixed
- Portable per-call timeout in the LLM judge: replaced the Unix/main-thread-only
  `signal.SIGALRM` with a `ThreadPoolExecutor`-based timeout that works off the main thread
  (e.g. FastAPI's threadpool) and on Windows.
- Removed a duplicated `save_setting`/`get_setting` pair in the SQLite store.
- Pinned `requirements.txt` to the exact versions the benchmark gates were validated against.

[0.2.0]: https://github.com/pras-ops/retrieval-reputation-layer/releases/tag/v0.2.0
[0.1.0]: https://github.com/pras-ops/retrieval-reputation-layer/releases/tag/v0.1.0
