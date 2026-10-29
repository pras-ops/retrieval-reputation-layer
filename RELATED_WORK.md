# Related Work & Positioning

RRL sits at the intersection of two active (2024–2026) research areas and one older
statistical lineage. This document states where it fits, how it differs, and — honestly —
where it has *not* yet been compared.

> **Note:** citations below are for orientation. Verify exact venues/IDs against the source
> before any formal submission; a few are cited by name where the identifier was not confirmed.

---

## 1. Agent memory & experience reuse (the closest neighbour)

A wave of recent work lets agents **reuse verified past experience** to improve later — the
same high-level goal as RRL.

- **Evo-Memory** (Google DeepMind + UIUC, [arXiv:2511.20857](https://arxiv.org/abs/2511.20857)) — a
  *streaming* benchmark for test-time learning with self-evolving memory, where each
  interaction encodes *whether the task succeeded and which strategies worked*. Their **ReMem**
  framework adds a think–act–memory-refine loop. This is essentially RRL's
  recurrence-with-a-verifier regime, standardized.
- **ExpeL** (Zhao et al., AAAI 2024) — collects trajectories, abstracts insights, retrieves
  successful past experiences at test time.
- **Agent Workflow Memory** (Wang et al., 2024) — induces reusable *workflows* from past
  trajectories and injects them into context.
- **Dynamic Cheatsheet** (2025) — adaptive memory of reusable strategies/insights.
- **Agentic Context Engineering** ([arXiv:2510.04618](https://arxiv.org/pdf/2510.04618)) — evolves
  the context itself for self-improvement.

**How RRL differs.** All of the above operate at the **prompt-construction level**: they
*extract and inject* workflows, insights, or strategy text. RRL operates one level lower, at
the **ranking level** — it keeps a lightweight **per-document Beta reputation** (with decay and
sycophancy safeguards) and changes *which existing documents surface*. It writes no new
memory artifacts and puts no LLM in the memory loop. This is a smaller, cheaper, more auditable
mechanism — and a narrower claim. The flip side: RRL does *not* synthesize new strategies the
way ExpRAG / AWM / Dynamic Cheatsheet do.

### 1a. Learned-utility memory retrieval (the closest prior work to RRL's mechanism)

- **MemRL** ([arXiv:2601.03192](https://arxiv.org/abs/2601.03192), Jan 2026) — retrieves
  candidates by similarity, then ranks them by `(1−λ)·z(sim) + λ·z(Q)`, where `Q` is a
  per-memory utility learned from environment feedback (`Q ← Q + α(r − Q)`), with the LLM frozen.
  Evaluated on BigCodeBench, ALFWorld, Lifelong Agent Bench and HLE against Mem0, MemP, RAG and
  Self-RAG. **This is essentially RRL's ranking rule.** RRL's remaining differences are the Beta
  posterior with Thompson sampling (vs a running mean), explicit staleness decay, and the
  noisy/sycophantic-feedback safeguards; MemRL has no decay and no per-source feedback model.
  Gate F includes a MemRL-style baseline (`memrl_q`).
- **Memento** ([arXiv:2508.16153](https://arxiv.org/abs/2508.16153)) — case bank with a learned
  case-retrieval policy in a memory-augmented MDP, LLM weights frozen.
- **RRPO** ([arXiv:2604.02091](https://arxiv.org/abs/2604.02091), ACL 2026) and **LURE-RAG**
  ([arXiv:2601.19535](https://arxiv.org/abs/2601.19535), ECIR 2026) — rerankers trained on the
  downstream utility of documents to the generator.

### 1b. Streaming benchmarks

- **StreamBench** ([arXiv:2406.08747](https://arxiv.org/abs/2406.08747), NeurIPS 2024 D&B) —
  continuous improvement of LLM agents over an input–feedback stream.
- **CodeRAG-Bench** ([arXiv:2406.14497](https://arxiv.org/abs/2406.14497), NAACL 2025 Findings) —
  realistic code retrieval corpora (docs, tutorials, StackOverflow, GitHub).

---

## 2. Online / feedback-driven RAG reranking & learning-to-rank

RRL's update loop is online learning-to-rank with bandit feedback, applied to RAG.

- **DynamicRAG** ([arXiv:2505.07233](https://arxiv.org/pdf/2505.07233)) — RL-optimizes a reranker
  from LLM-output feedback.
- **AutoRAG-HP** — frames RAG knobs as hierarchical bandits tuned from live feedback.
- **LTRR: Learning to Rank Retrievers for LLMs** ([arXiv:2506.13743](https://arxiv.org/pdf/2506.13743)).
- **Online-Optimized RAG for tool use / function calling** ([arXiv:2509.20415](https://arxiv.org/pdf/2509.20415)).
- **REARANK** — reranking as an RL reasoning agent; **RankArena** — eval platform with
  human/LLM feedback.

**How RRL differs.** These largely *train or prompt a (neural / LLM) reranker* from feedback.
RRL keeps the reranker fixed and adds a **non-parametric reputation prior** over documents that
updates in O(1) per feedback event, with explicit **staleness decay** and **noise/sycophancy
safeguards** — and it characterizes its own **boundary condition** (no recurrence → a strong
cross-encoder wins), which most of these do not.

---

## 3. Core statistical lineage (the math is not new — by design)

- **Beta Reputation System** (Jøsang & Ismail, 2002) — Beta-Bernoulli reputation from
  positive/negative feedback counts. RRL's α/β counters are this, per document.
- **Click models with Beta posteriors** — maintaining Beta posteriors per document to represent
  relevance/attractiveness uncertainty is standard in IR.
- **Online learning-to-rank with bandit feedback** (e.g. cascading/position-based bandits;
  LinUCB / Thompson-sampling rankers) — RRL's Thompson-sampling exploration is from this line.

**How RRL differs.** The contribution is *not* the estimator. It is the **integration**: a
clean, drop-in reputation layer that combines Beta reputation + similarity + decay + a
liar/deception counter, evaluated honestly with a stated failure mode.

---

## 3a. Methods from other fields that address RRL's open problems (future work)

| RRL problem | Field | Work | How it would apply |
|---|---|---|---|
| Reproducible offline evaluation | News recommendation | Li et al., *Unbiased offline evaluation of contextual-bandit-based news article recommendation*, WSDM 2011 ([arXiv:1003.5956](https://arxiv.org/abs/1003.5956)) | Replay policies against logged outcomes. Gate F uses the full-information form: every (query, candidate) outcome is pre-computed. |
| Dishonest / poisoned feedback | Bandit theory | Lykouris et al., *Stochastic bandits robust to adversarial corruptions*, STOC 2018 ([arXiv:1803.09353](https://arxiv.org/abs/1803.09353)); Xu et al., robust Thompson sampling ([arXiv:2410.19705](https://arxiv.org/abs/2410.19705)) | Principled baselines and guarantees for the liar counter and behavioral cap. |
| The ">50% dishonest" wall | Federated learning | Cao et al., *FLTrust*, NDSS 2021 ([arXiv:2012.13995](https://arxiv.org/abs/2012.13995)) | A small trusted root set (here: verifier outcomes) scores every other feedback source; FLTrust stays accurate with 40–60% malicious clients. |
| Weighting feedback sources | Crowdsourcing | Dawid & Skene, 1979 | Learn each source's (user, judge, behavior) reliability instead of fixed weights. |
| Staleness | Non-stationary bandits / stream mining | Garivier & Moulines, ALT 2011 ([arXiv:0805.3415](https://arxiv.org/abs/0805.3415)); discounted Thompson sampling ([arXiv:2305.10718](https://arxiv.org/abs/2305.10718)); ADWIN (Bifet & Gavaldà, 2007) | Compare fixed-γ decay against sliding windows and change detection. |
| Credit for top-k > 1 | Context attribution / click models | ContextCite, NeurIPS 2024 ([arXiv:2409.00729](https://arxiv.org/abs/2409.00729)); cascading bandits, ICML 2015 ([arXiv:1502.02763](https://arxiv.org/abs/1502.02763)) | Credit documents by measured influence on the answer, not by similarity. |
| Popularity bias | Recommender systems | Jiang et al., *Degenerate feedback loops*, AIES 2019 ([arXiv:1902.10730](https://arxiv.org/abs/1902.10730)) | Monitor retrieval concentration; keep exploration on. |
| Query-conditional reputation | Bandits | Gentile et al., *Online clustering of bandits*, ICML 2014 | Learn query clusters online; empirical-Bayes shrinkage instead of a fixed threshold. |

---

## 4. Evaluation methodology

- **"Benchmarking is Broken — Don't Let AI Be Its Own Judge"** ([arXiv:2510.07575](https://arxiv.org/html/2510.07575v2)) —
  warns against LLM-judge circularity. RRL's evaluation independently arrived at this: an
  earlier circular Gate A (training on the eval label) was identified and replaced with an
  independent verifier, and the LLM judge is deliberately demoted.

---

## Honest gaps relative to this work

- **Not yet evaluated on Evo-Memory** (the standard streaming benchmark). Current evidence is
  on controlled sims (Gates A/B), a synthetic-distractor recurrence demo (Gate D), a
  no-recurrence boundary on real HumanEval (Gate C), an MBPP recurrence harness with an
  answer-key leak (Gate E, superseded), and its leak-free, replayable replacement (Gate F,
  `run_gate_matrix.py`).
- **Not yet compared against MemRL itself** — only a MemRL-style baseline (`memrl_q` in Gate F)
  on RRL's own benchmark, not on MemRL's benchmarks.
- **Query-conditional reputation** — the per-experience granularity that ExpRAG/ReMem use is,
  in RRL, only the (unvalidated, future-work) clustering variant; validated results use global
  counters.
- **No strong-stack comparison** (hybrid retrieval + query rewriting + multi-query + agent
  memory) yet — retriever-level comparisons only.

## One-line positioning

> RRL is a lightweight, auditable **retrieval reputation layer** — Beta-Bernoulli document
> reputation with decay and sycophancy safeguards — that complements (rather than competes
> with) heavier experience-reuse memory systems, and helps specifically under **query
> recurrence with a trustworthy verifier**.
