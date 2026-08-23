# Paper: "When Does Outcome Feedback Help Retrieval?"

LaTeX source for the RRL boundary-study preprint.

## Contents

- `main.tex` — the complete manuscript (compiles with any standard LaTeX; no custom style file)
- `references.bib` — bibliography, all arXiv IDs and author lists verified against the sources
- `make_figures.py` — regenerates `figures/*.pdf` from the committed results JSONs (`python3 paper/make_figures.py` from the repo root)
- `figures/` — vector PDFs used by the paper (+ PNG previews)
- `overleaf-upload.zip` — ready-to-upload bundle (main.tex + references.bib + PDF figures)

## To compile

No local LaTeX needed:

1. Go to [overleaf.com](https://www.overleaf.com) → New Project → **Upload Project** → select `overleaf-upload.zip`.
2. Set the compiler to pdfLaTeX (default) and click Recompile. It should build with no errors.
3. Before submission: confirm the author name spelling in `main.tex` (marked with a TODO comment).

## To submit to arXiv

1. Register at arxiv.org; start a new submission (primary: `cs.IR`, cross-list: `cs.LG`).
2. Upload the same zip contents as **source** (arXiv requires LaTeX source, not just PDF).
3. License: arXiv non-exclusive (default). Handle the endorsement step if prompted.

## Number provenance

Every number in the paper derives from committed artifacts:

- Gate R table/figures → `sim/results/gate_r_*.json`
- Sensitivity analysis → `data/mbpp_sweep_cache.jsonl` (generator == `gemini-2.5-flash` entries)
- Gates A–D → `README.md` validation section and `sim/` gate scripts

All Gate R generations are cached, so results replay offline via
`python3 sim/run_gate_recurring.py --replay ...` with no API access.
