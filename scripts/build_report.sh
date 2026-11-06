#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../report"
mkdir -p ../build/report
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/report main.tex
