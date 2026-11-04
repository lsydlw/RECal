#!/usr/bin/env bash
# RQ7(a): paired hierarchical bootstrap and Holm correction.
set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT="$RESULTS_ROOT/recal_significance_v1"
mkdir -p "$OUT"
"$PYTHON" -m refrec.analyze_recal_significance \
  --root "$RESULTS_ROOT/recal_asymmetric_8datasets_v1" \
  --output "$OUT/base_vs_recal_hierarchical_bootstrap.csv" \
  --bootstrap 5000 --seed 20261010
echo "RQ7 significance complete"
