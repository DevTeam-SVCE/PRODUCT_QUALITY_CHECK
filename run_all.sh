#!/usr/bin/env bash
# Inspect every photo in this folder against the drawings in dxf_files/ (matched automatically).
# Usage: bash run_all.sh [TOL_MM]      (all options: python3 inspect_all.py --help)
cd "$(dirname "$0")"
python3 inspect_all.py --photos . --drawings dxf_files --out output ${1:+--tol "$1"}
