# Paper build

- `tex/main.tex`, `tex/refs.bib`, CVPR 2027 author-kit style (`cvpr.sty`, `ieeenat_fullname.bst`), review mode (anonymous, line numbers).
- Build: `cd paper/tex && tectonic main.tex` (XeTeX via tectonic; pdflatex also works with the same sources).
- Figures: `python scripts/make_figures.py` writes `paper/figures/fig1..fig9` (PDF + PNG) from the CSVs in `results/`. No number is typed by hand.
- Demo video: `python scripts/make_demo.py` writes `paper/demo/demo.mp4`; see `paper/demo/README.md` for what is live and what is replayed.
- Output: `observe_imagine_abstain_cvpr27.pdf` (8 pages + references + supplementary figures).
- `draft.md` is the working draft that the LaTeX was written from and is kept as the number ledger.

Before submission: verify the author fields of the 2026 arXiv entries in `refs.bib` (marked "Authors to be verified"), fill every `pending:` box from logged runs or remove the claim, re-run the novelty search, and add the AI-assistance disclosure required by the CVPR 2027 guidelines.
