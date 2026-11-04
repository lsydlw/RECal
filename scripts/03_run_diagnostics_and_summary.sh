#!/usr/bin/env bash
# RQ1 mismatch diagnosis and RQ2 main-result aggregation.
set -euo pipefail
source "$(dirname "$0")/common.sh"

MAIN="$RESULTS_ROOT/recal_asymmetric_8datasets_v1"
[[ -d "$MAIN/recal_runs" ]] || { echo "run scripts/02_run_main_test_once.sh first" >&2; exit 1; }

"$PYTHON" -m refrec.diagnose_intent_mass_mismatch \
  --input_roots "$MAIN/recal_runs" \
  --output_dir "$RESULTS_ROOT/recal_mismatch_diagnostic_v1" \
  --backbones gru4rec sasrec bert4rec repeatnet faesrec mamba4rec

"$PYTHON" -m refrec.summarize_recal_multiseed \
  --root "$MAIN/recal_runs" --output "$MAIN/multiseed_test_summary.csv"

echo "RQ1/RQ2 summaries written"
