#!/bin/sh
# Build the paper PDF and an arXiv-ready source archive.
#   paper/main.pdf            the compiled paper
#   paper/arxiv-source.tar.gz main.tex (comments stripped: arXiv archives everything uploaded)
#                             + main.bbl (arXiv does not run BibTeX) + style files + figures
# Requires Tectonic (brew install tectonic). Run from anywhere: sh paper/build.sh
set -e
cd "$(dirname "$0")"
python3 ../scripts/19_paper_figures.py
tectonic -X compile --keep-intermediates main.tex

rm -rf arxiv-staging && mkdir -p arxiv-staging/figures
python3 - <<'EOF'
import re
out = []
for line in open("main.tex", encoding="utf-8"):
    if line.lstrip().startswith("%"):
        continue                                   # whole-line comment
    out.append(re.sub(r"(?<!\\)%.*$", "%", line.rstrip("\n")) + "\n")  # trailing comment
open("arxiv-staging/main.tex", "w", encoding="utf-8").write("".join(out))
EOF
cp main.bbl references.bib acl.sty acl_natbib.bst arxiv-staging/
cp figures/*.pdf arxiv-staging/figures/
(cd arxiv-staging && tectonic -X compile main.tex >/dev/null 2>&1 && rm -f main.pdf)  # the stripped copy must still compile
tar -czf arxiv-source.tar.gz -C arxiv-staging main.tex main.bbl references.bib acl.sty acl_natbib.bst figures
rm -rf arxiv-staging main.aux main.blg main.out main.log main.toc
echo "built main.pdf and arxiv-source.tar.gz"
