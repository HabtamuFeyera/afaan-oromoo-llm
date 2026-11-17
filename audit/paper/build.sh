#!/bin/sh
# Build the contamination-audit paper PDF (needs Tectonic: brew install tectonic).
# Figures are drawn first from the saved results: python audit/scripts/09_figures.py
set -e
cd "$(dirname "$0")"
python3 ../scripts/09_figures.py > /dev/null
tectonic -X compile main.tex
echo "built main.pdf"
