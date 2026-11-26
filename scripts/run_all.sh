#!/usr/bin/env bash
# Regenerate every figure and data table (about 25 minutes on 2 cores).
set -euo pipefail
cd "$(dirname "$0")"
rm -f ../data/gesn_null.csv
python3 fig1_validation.py
python3 fig2_dilute_gesn.py
python3 fig3_detectability.py
python3 fig4_power.py
python3 make_table.py
python3 snapshot_averaging.py
# needs the dataset doi:10.23673/RE-518 (set TAMM_TAR to the path of 2015-MC-DFT_structures.tar.gz)
if [ -n "${TAMM_TAR:-}" ]; then python3 tamm_chains.py "$TAMM_TAR"; fi
python3 reassess_literature.py
