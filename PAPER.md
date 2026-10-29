# Paper plan

Working title: **Trust, Not Just Relevance: Verifier-Anchored Tool Selection Under Partial
Feedback and Tool Drift**

This file tracks the paper built on this repository: the claim, what the evidence supports today,
what must be done before submission, and how to post a preprint. Numbers come from
[RESULTS.md](RESULTS.md); related work and its verification status from
[RELATED_WORK.md](RELATED_WORK.md).

## Research question

Can a training-free, outcome-aware selection layer improve an agent's choice of tools when tool
descriptions or self-reported success can mislead, only some calls can be verified, and tools
change over time?

## Claim (narrow on purpose)

RRL helps in a specific regime — feedback that is partially verified and selectively misleading,
with new contexts appearing — and its gains shrink or reverse when every call is verified and
contexts recur, or when tools change abruptly and feedback is trustworthy.

## Contributions

1. **Per-item feedback trust.** A two-coin (Dawid–Skene) model of each feedback source's
   sensitivity and specificity, learned against verified outcomes, globally and per tool
   (`rrl/reliability.py`).
2. **An online selection layer** combining relevance with Beta reputation, per-context counters
   shrunk toward global ones, verified-success memory and change detection (`rrl/layer.py`,
   `rrl/drift.py`).
3. **A reproducible evaluation** under poisoned tool descriptions, partial verification,
   unreliable feedback and tool evolution — six gates, including live MCP servers and an LLM
   choosing from the tool list — that reports where simpler methods win.

## What the evidence supports today

| Supported | Not yet supported |
|---|---|
| Large gains over description-based selection, including Gemini choosing from the tool list, when a poisoned tool is present (Gate M) | Superiority over MemRL's official code |
| Resistance to over-reporting tools under 20% spot checks, where a MemRL-style learner is fooled (Gates M, W) | Results on unmodified MCPTox / MCPEvol-Bench tasks or third-party MCP servers |
| Far better accuracy on new contexts than per-context memory (Gates M, W) | Better adaptation than a fast-forgetting learner under abrupt change (Gate G: it is worse) |
| Each component's contribution, by ablation | Calibration of source trust under correlated sources and biased audit sampling |

## Before submission (from the external review)

| # | Task | Status |
|---|---|---|
| 1 | Run MemRL's official implementation under the same pools, verifier budget and splits | To do |
| 2 | Evaluate on MCPTox and MCPEvol-Bench (unmodified results reported separately from adapted ones), with Tool-Guard as a security baseline | To do |
| 3 | A robust per-context memory baseline that survives transient failures | **Done** (`query_memory_robust`): stronger when every call is verified, more easily fooled under spot checks |
| 4 | Source-trust validation: randomized vs. biased audits, deception concentrated on single tools, correlated sources; report calibration and false reputation updates | To do |
| 5 | Change detection: recovery delay, false resets under stable conditions, gradual vs. abrupt change, against a fast-forgetting baseline | Partly (resets reported; delay and gradual drift to do) |
| 6 | Verify every citation | **Done** for all arXiv identifiers; venue details for non-arXiv works to add |
| 7 | Separate benchmark weights (exploration 0.30) from the library default (0.10) in the text | Documented in RESULTS.md; report both in the paper |
| 8 | Publish an archived, versioned release with the exact code, outcome records and tables | To do (see below) |

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
