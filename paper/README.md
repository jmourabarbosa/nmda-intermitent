# Paper

`main.pdf` is built from the paper's Google Doc, which is the only source of the text, with `doc_to_latex.py`:

1. Read the doc as JSON (Google Docs API `documents.get`) and save it, e.g. `doc.json`.
2. `python doc_to_latex.py doc.json` writes `main.tex` and downloads the figures embedded in the doc to
   `figures/figNN.png`, in the doc's order.
3. `pdflatex main && bibtex main && pdflatex main && pdflatex main`

`main_noAI.pdf` is the same paper without the AI methodology statement: `python doc_to_latex.py doc.json --exclude "AI methodology" --name main_noAI`.

The converter adds the layout the doc cannot hold, and stops or warns rather than guess:
- figures: each image and its "Figure N." / "Supplementary Figure N." paragraph become a float with a caption;
- style of Nature Neuroscience: numbered superscript citations (in order of appearance), "Fig. 1c,d" with the
  figure number as a link, captions "Fig. N | Title." with bold panel letters;
- citations: "(Author et al., 2024)" and "Author (2024)" become natbib citations to `refs.bib`; a citation or a doc
  reference without a bib entry stops the build, and the doc's reference list is checked against the cited entries;
- equations: centred equation paragraphs are typeset from their LaTeX version in `EQUATIONS` only if the doc's text is
  unchanged; an edited equation is kept as text and reported.

Pending suggestions count as not accepted (suggested insertions dropped, suggested deletions kept).
