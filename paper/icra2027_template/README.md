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

- `main.tex`: anonymous English manuscript draft using the ICRA class. It
  contains four main figures; supplementary figures are kept separate.
- `supplement.tex`: standalone document containing S1--S4 supplementary figures.
- `references.bib`: references cited by `main.tex`.
- `ieeeconf.cls`: required conference document class.
- `bibtex/IEEEtran.bst`: standard IEEE numbered bibliography style.

The manuscript uses the `svg` package so the editable SVG figures remain the
source assets. On Overleaf, upload the required `generated/` figure folders
alongside these files and compile `main.tex`; switch the main document to
`supplement.tex` when reviewing S1--S4. The current machine does not have a
LaTeX engine installed, so PDF compilation and page-count inspection still need
to be run in Overleaf or another TeX environment.

ICRA 2027 currently specifies an eight-page limit for the complete submission,
including figures, tables, acknowledgments, and references, and uses double-
anonymous review. Recheck the conference website before final submission in
case the instructions change.
