# Paper

`main.pdf` is built from the paper's Google Doc (the source of the text), with `doc_to_latex.py`:

1. Read the doc as JSON (Google Docs API `documents.get`) and save it, e.g. `doc.json`.
2. `python doc_to_latex.py doc.json` writes `main.tex` and downloads the figures embedded in the doc to
   `figures/figNN.png` (in the doc's order).
3. `pdflatex main.tex`

Text, formatting (italic, bold, sub- and superscripts), headings, figures and the reference list are taken from the
doc as they are. Pending suggestions count as not accepted (suggested insertions dropped, suggested deletions kept).
