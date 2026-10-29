# Paper plan

Working title: **When Tools Over-Report Success: Verifier-Anchored Tool Selection Under Partial
Feedback**

**Draft:** [`paper/main.tex`](paper/main.tex) (PDF: `paper/main.pdf`). Every table and quoted
number is generated from `sim/results/` by `paper/make_tables.py`; rebuild with
`python3 paper/make_tables.py && python3 paper/make_figure.py && cd paper && tectonic main.tex`. The author block is a
placeholder.

This file tracks the paper built on this repository: the claim, what the evidence supports today,
what must be done before submission, and how to post a preprint. Numbers come from
[RESULTS.md](RESULTS.md); related work and its verification status from
[RELATED_WORK.md](RELATED_WORK.md).

## Research question

Can a training-free, outcome-aware selection layer improve an agent's choice of tools when tool
descriptions or self-reported success can mislead, only some calls can be verified, and tools
change over time?

## Claim (narrow on purpose)

Under partial verification, learners that take unverified success reports at face value adopt
the tool that over-reports. Three training-free ingredients avoid it: discount unverified
success, let only verified outcomes lock in a per-context choice, and start new contexts from each
tool's global reputation. A learned model of each feedback source's reliability is *not* what
drives the gains. The approach loses when every call is verified and contexts recur, when tools
change abruptly and feedback is trustworthy, and gives no protection against a tool that defects
after earning trust.

## Reframing after the 2026-10-29 review

The first draft claimed learned per-tool source trust (two-coin Dawid–Skene, `rrl/reliability.py`)
as the main mechanism. A controlled ablation (`rrl_combined_ctx_verified_fixed`: the same layer
with the fixed feedback rule) showed it differs from a fixed discount by −3.4 to +3.2 points across
23 scenarios with unverified feedback, and is slightly *worse* in the case it was built for (Gate M
with the poisoned server under spot checks). The paper now reports this as a negative result and
keeps the pre-specified configuration as "RRL full". A staged-defection scenario (TrustShift-style)
was added to Gate M; it erases RRL's advantage.

## Novelty

- **RATED** (CIKM 2026, arXiv:2610.13889) learns which tools work from execution feedback with a
  contextual bandit, assuming the feedback is truthful.
- **ESA** (EWRL 2026, arXiv:2603.13356) learns per-source, per-context trust from sparse
  ground-truth audits, with regret bounds, but has no LLM or tool application. Its decoupling
  theorem explains why a MemRL-style learner collapses under spot checks.

The contribution is the finding that, for tool selection, anchoring memory to verified outcomes
and discounting the rest is enough to avoid an over-reporting tool while learned trust changes
little, plus the evaluation (live MCP servers, an LLM choosing from the tool list, partial
verification, staged defection) and an honest map of where simpler methods win.

## Contributions

1. **A simple, training-free selection layer** combining relevance, Beta reputation, a discount
   on unverified success, per-context counters shrunk toward global ones and verified-success
   memory (`rrl/layer.py`, `rrl/feedback.py`).
2. **A negative result on learned feedback trust** (`rrl/reliability.py`): per-tool trust is
   within a few points of a fixed discount; global-only trust is harmful.
3. **A reproducible evaluation** — six gates, including live MCP servers, an LLM choosing from the
   tool list and staged defection — that reports where simpler methods win.

## What the evidence supports today

| Supported | Not yet supported |
|---|---|
| Large gains over description-based selection, including Gemini choosing from the tool list, when a poisoned tool is present (Gate M) | Superiority over MemRL's official code, RATED or an ESA-style aggregator |
| Resistance to over-reporting tools under 20% spot checks, where a MemRL-style learner and a robust per-site memory are fooled (Gates M, W) | Results on unmodified MCPTox / MCPEvol-Bench tasks or third-party MCP servers |
| Far better accuracy on new contexts than per-context memory (Gates M, W) | Any benefit of learned source trust over a fixed discount |
| Each component's contribution, by ablation | Protection against staged defection; better adaptation than a fast-forgetting learner under abrupt change |

## Before submission (from the external review)

| # | Task | Status |
|---|---|---|
| 1 | Run MemRL's official implementation under the same pools, verifier budget and splits | To do |
| 1b | Compare against **RATED** (code public, EcoAgent benchmark) and an **ESA**-style audit-trust aggregator — the two closest methods | To do |
| 2 | Evaluate on MCPTox and MCPEvol-Bench (unmodified results reported separately from adapted ones), with Tool-Guard as a security baseline | To do |
| 3 | A robust per-context memory baseline that survives transient failures | **Done** (`query_memory_robust`): beats RRL by 8–21 points when every call is verified; with 20% spot checks it is fooled and RRL beats it by 20–28 |
| 4 | Source-trust validation: randomized vs. biased audits, deception concentrated on single tools, correlated sources; report calibration and false reputation updates | To do |
| 5 | Change detection: recovery delay, false resets under stable conditions, gradual vs. abrupt change, against a fast-forgetting baseline | Partly (resets reported; costs 2–3 points without change; no help against staged defection; delay and gradual drift to do) |
| 5b | Staged defection (TrustShift): a defence, e.g. change detection on a tool's agreement with the verifier | Scenario added; no defence yet |
| 6 | Verify every citation | **Done** for all arXiv identifiers; venue details for non-arXiv works to add |
| 7 | Separate benchmark weights (exploration 0.30) from the library default (0.10) in the text | Documented in RESULTS.md; report both in the paper |
| 8 | Publish an archived, versioned release with the exact code, outcome records and tables | To do (see below) |

## Verdict (2026-10-29)

- **Does it work?** Yes, in its regime: misleading tools + partial verification + new contexts,
  where it beats every baseline tested by large margins on 30/30 seeds. Outside it, a
  per-context memory (all calls verified) or a fast-forgetting learner (abrupt change) is better.
- **Is it useful?** For agents choosing among MCP tools or scrapers that can over-report success,
  with only some results checkable — a real and documented problem (agents adopt corrupted tool
  outputs over a third of the time; MCP has no server reputation system).
- **Publishable?** As an arXiv preprint and a workshop paper now, framed as an integration and
  empirical study. A main-conference paper needs tasks 1, 1b and 2: reviewers will ask how it
  compares with RATED and ESA, and with MemRL's own code.

## Where to submit

1. **arXiv preprint** once the paper is written (tasks 1–2 make it much stronger but are not
   required for a preprint).
2. **Workshop** on agents, tool use or AI security (NeurIPS / ICLR / ACL workshops) with the
   current evidence and its caveats.
3. **Main conference** (ACL/EMNLP for tool use, or a security venue for the poisoning angle) after
   tasks 1, 2, 4 and 5.

## Posting to arXiv

arXiv is a preprint server: it moderates submissions for scope and quality but does not peer
review them, and a posted paper is public and permanent (it can be replaced by new versions but
not removed).

1. **Write the paper first.** arXiv takes a complete research paper, normally as LaTeX source
   (PDF also accepted). Use a conference template (e.g. ACL or NeurIPS) so the same source can be
   submitted later. Survey and position papers in Computer Science now need prior peer review;
   a research paper with new methods and experiments, like this one, does not.
2. **Create an arXiv account** with your real name and affiliation (independent researchers may
   list none).
3. **Endorsement.** First-time submitters to a category need an endorsement from an established
   author in that category. Since January 2026 an institutional email alone no longer qualifies;
   ask a researcher who has published in the category (a co-author, advisor or colleague) to
   endorse you through arXiv's endorsement link.
4. **Categories.** Primary `cs.AI` or `cs.CL` (tool-using LLM agents); cross-list `cs.LG`, and
   `cs.CR` for the tool-poisoning angle.
5. **License.** CC BY 4.0 is the common choice for open work.
6. **Code and data.** Link this repository and an archived release (a GitHub release plus a
   Zenodo DOI) that matches the paper exactly; state that MemRL comparisons use a
   reimplementation until task 1 is done.
7. **Disclose AI assistance** in the acknowledgements as your venue requires; you remain
   responsible for every claim and citation.

Check arXiv's current help pages on endorsement and moderation before submitting; the rules
above were checked on 2026-10-29.

## Releasing the exact version

`.github/workflows/release.yml` publishes to PyPI whenever a `v*` tag is pushed, so tagging is a
release decision. When ready: merge the research branch into `main`, bump the version, tag it,
create a GitHub release, and connect the repository to Zenodo to mint a DOI for that release.
