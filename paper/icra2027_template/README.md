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

- `root.tex`: official example manuscript and starting point.
- `ieeeconf.cls`: required conference document class.
- `root.pdf`: official compiled example.
- `bibtex/IEEEtran.bst`: standard IEEE numbered bibliography style.
- `main.tex`: current anonymous English manuscript draft using the ICRA class.
- `references.bib`: references cited by `main.tex`.
- `piroute_overview.pdf`: cropped vector export of the current PIRoute Figure 1.
- `piroute_overview_2_2_2.pdf`: cropped vector export of
  `Section 2-2 2.svg`; the source SVG remains unchanged.
- `samarl_fig2_reference.pdf`: Fig. 2 cropped from Wang et al., ICRA 2024,
  for internal layout comparison only; it is not a submission asset.
- `piroute_icra2027_overleaf_preview.zip`: older internal layout preview; it is
  not the manuscript source.

The manuscript uses the `svg` package so the editable SVG figures remain the
source assets. On Overleaf, upload the repository's `paper/generated/` figure
folders alongside this directory and enable the standard SVG conversion
support. The current machine does not have a LaTeX engine installed, so PDF
compilation and page-count inspection still need to be run in Overleaf or
another TeX environment.

ICRA 2027 currently specifies an eight-page limit for the complete submission,
including figures, tables, acknowledgments, and references, and uses double-
anonymous review. Recheck the conference website before final submission in
case the instructions change.
