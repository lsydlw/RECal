#!/usr/bin/env bash
# Generate the paper-facing RQ3/RQ5/RQ6 PDF figures from saved summaries.
set -euo pipefail
source "$(dirname "$0")/common.sh"

FIGURES="$RESULTS_ROOT/figures"
mkdir -p "$FIGURES"
"$PYTHON" -m refrec.plot_alternative_postprocessing \
  --output "$FIGURES/alternative_postprocessing_heatmap.pdf"
"$PYTHON" -m refrec.plot_rq5_ablation --output_dir "$FIGURES/rq5_ablation"
"$PYTHON" -m refrec.plot_rq6_figures --output_dir "$FIGURES/rq6"
echo "figures written to $FIGURES"
