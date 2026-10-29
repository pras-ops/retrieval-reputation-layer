# Related Work & Positioning

RRL sits at the intersection of two active (2024–2026) research areas and one older
statistical lineage. This document states where it fits, how it differs, and — honestly —
where it has *not* yet been compared.

> **Citation check (2026-10-29):** every arXiv identifier in this file, the README and
> RESULTS.md was looked up through the arXiv API and its title matched the work cited. Works
> cited without an identifier (Dawid & Skene 1979, GLAD, ADWIN, CLUB, ExpeL, Agent Workflow
> Memory, Dynamic Cheatsheet, AutoRAG-HP, REARANK, CASE) still need their venue details added
> before submission.

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

### 1c. Agent tool / MCP selection (the application RRL is now evaluated on)

- **ToolRet** ([arXiv:2503.01763](https://arxiv.org/abs/2503.01763), Findings of ACL 2025) —
  tool-retrieval benchmark (7.6k tasks, 43k tools); retrievers strong on ordinary IR do poorly
  on it. Gate T uses its ToolBench and ToolLens request sets and gold labels.
- **MCPTox** ([arXiv:2508.14925](https://arxiv.org/abs/2508.14925), AAAI 2026) and
  **MCP-SafetyBench** (ICLR 2026) — tool-poisoning attacks on real MCP servers, up to 72.8%
  attack success. Gate T's poisoned tool twins follow this threat model.
- **MCPEvol-Bench** ([arXiv:2607.14642](https://arxiv.org/abs/2607.14642)), **ProEvolve**
  ([arXiv:2603.05910](https://arxiv.org/abs/2603.05910)) and **ToolEVO / ToolQA-D**
  ([arXiv:2410.06617](https://arxiv.org/abs/2410.06617), ICLR 2025) — agents break when tools
  and APIs change; ToolEVO adapts by training the model. RRL adapts without training, at the
  selection layer. Gate T's tool updates follow MCPEvol-Bench's mutation idea.
- **AutoTool** ([arXiv:2511.14650](https://arxiv.org/abs/2511.14650)) and **ToolTree**
  ([arXiv:2603.12740](https://arxiv.org/abs/2603.12740)) — tool selection from execution
  feedback, without untrusted-report handling or change detection.
- **Tool-Guard** — *Think Twice Before You Act: Protecting LLM Agents Against Tool Description
  Poisoning via Isolated Planning* ([arXiv:2606.20922](https://arxiv.org/abs/2606.20922), ICML
  2026): partitions tools, plans in isolation and quarantines suspicious cross-tool influence;
  reports near-zero attack success on AgentDojo and ASB. **TRUSTDESC**
  ([arXiv:2604.07536](https://arxiv.org/abs/2604.07536)) regenerates trusted tool descriptions.
  These defend *within a plan* against a poisoned description; RRL learns *across requests*
  from verified outcomes which tools to trust. The two are complementary, and combining them is
  untested.
- Practice: MCP trust-scoring services (runtime reports, reporter-weighted reputation, schema
  drift checks) are emerging but early and not independently evaluated.

### 1d. Source reliability — the statistics RRL's trust axis comes from

- **Dawid & Skene (1979)**, **GLAD** (Whitehill et al., NeurIPS 2009: per-item difficulty), and
  **truth discovery** ([survey, arXiv:1505.02463](https://arxiv.org/abs/1505.02463); CRH,
  CATD for sources with few claims). RRL's `SourceReliability` is a two-coin Dawid–Skene model
  with the verifier as the known truth and per-document rates shrunk toward global ones. The
  statistics are not new; applying them to outcome-driven retrieval and tool selection is.

### 1e. Other settings with the same shape (not targeted)

- Semantic caching: **vCache** ([arXiv:2502.03771](https://arxiv.org/abs/2502.03771), ICLR 2026)
  learns per-entry thresholds with error bounds.
- Model routing with bandit feedback: PILOT (*Adaptive LLM Routing under Budget Constraints*,
  [arXiv:2508.21141](https://arxiv.org/abs/2508.21141)),
  BaRP ([arXiv:2510.07429](https://arxiv.org/abs/2510.07429)), ParetoBandit
  ([arXiv:2604.00136](https://arxiv.org/abs/2604.00136), with forgetting for drift).
- Demonstration selection as a bandit: CASE, and DEARICL (*Data Efficient Sample Selection for
  In-Context Learning*, [arXiv:2609.06670](https://arxiv.org/abs/2609.06670)).

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

- **No third-party tools.** Gate M makes real MCP calls, but to local servers written for the
  test against a generated website; Gate T judges by ToolRet's gold labels; Gate W runs real
  extraction code on generated pages. Third-party MCP servers (e.g. MCPEvol-Bench or MCPTox
  environments) are still needed.
- **Not compared against MemRL's code or benchmarks** — only a MemRL-style reimplementation of
  its ranking rule (`memrl_q`) on RRL's gates.
- **Not compared against Tool-Guard**, the closest security baseline for poisoned tool
  descriptions, nor run on MCPTox or MCPEvol-Bench tasks unmodified.
- **Not evaluated on Evo-Memory** or a natural document corpus with real version changes.
- **No strong-stack comparison** (query rewriting, multi-query, agent memory) — selection-level
  comparisons only.
- Evidence so far: controlled sims (Gates A/B/D), a no-recurrence boundary (C), a superseded
  MBPP harness (E), leak-free MBPP (F), a private-API library (G), real tool descriptions (T)
  and executable web-scraping tools (W).

## One-line positioning

> RRL is a lightweight, training-free **selection layer** for retrieval and tool use. It learns
> which documents or tools actually work from outcomes, trusts other feedback only as far as the
> verifier has confirmed it — per document or tool — and detects when an item's usefulness
> changes. Its ranking rule overlaps with MemRL; its distinct part is the verifier-anchored,
> per-item trust in feedback, which matters when only some calls can be verified and tools'
> descriptions or self-reports mislead. With every call verified, simpler memories do as well
> or better.
