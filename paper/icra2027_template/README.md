# ICRA 2027 LaTeX template

This directory contains the official IEEE Robotics and Automation Society
PaperCept conference template referenced by the ICRA 2027 call for papers.

## Sources

- ICRA 2027 call for papers:
  https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/
- PaperCept template support:
  https://ras.papercept.net/conferences/support/support.php
- LaTeX template archive:
  https://ras.papercept.net/conferences/support/files/ieeeconf.zip
- IEEE BibTeX archive:
  https://ras.papercept.net/conferences/support/files/IEEEtranBST.zip

Downloaded on 2026-09-05.

## Contents

- `main.tex`: self-contained anonymous English manuscript using the ICRA class.
  The current version contains the overview figure and one explicitly labeled
  predecessor-router qualitative figure; quantitative results are reported in
  the tables and text.
- `references.bib`: references cited by `main.tex`.
- `ieeeconf.cls`: required conference document class.
- `bibtex/IEEEtran.bst`: standard IEEE numbered bibliography style.

The manuscript includes pre-exported PDF figures and does not require SVG or
Inkscape during compilation. On Overleaf, upload the complete final package,
choose `main.tex` as the main document, and compile with pdfLaTeX. The local
copy has been compiled successfully with pdfLaTeX and produces an eight-page
Letter-size PDF.

ICRA 2027 currently specifies an eight-page limit for the complete submission,
including figures, tables, acknowledgments, and references, and uses double-
anonymous review. Recheck the conference website before final submission in
case the instructions change.
