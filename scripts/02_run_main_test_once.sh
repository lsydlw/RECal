#!/usr/bin/env bash
# Build the final asymmetric calibrator, freeze it, then evaluate test once.
set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT="$RESULTS_ROOT/recal_asymmetric_8datasets_v1"
LOG="$OUT/logs"
MANIFEST="$OUT/frozen_manifest.json"
mkdir -p "$OUT/recal_runs" "$LOG"

run_asymmetric_valid() {
  local ds="$1" bb="$2" seed="$3" base source out name
  base="$(base_dir "$ds" "$bb" "$seed")"
  source="$(source_recal_dir "$ds" "$bb" "$seed")"
  out="$(asym_recal_dir "$ds" "$bb" "$seed")"
  name="${ds}_${bb}_recal_asym_seed${seed}"
  [[ -f "$out/metrics.json" && -f "$out/intent_head.pt" ]] && return 0
  [[ -f "$base/best_model.pt" && -f "$source/intent_head.pt" ]] || {
    echo "missing frozen input for $ds $bb seed=$seed" >&2; return 1;
  }
  "$PYTHON" -m refrec.recalibrate_existing_head \
    --base_run_dir "$base" --source_recal_run_dir "$source" \
    --output_dir "$out" --device "$DEVICE" --seed "$seed" \
    --hidden_dim 64 --down_power 12 >"$LOG/${name}_valid.log" 2>&1
}

echo "asymmetric validation evaluation"
for ds in "${ALL_DATASETS[@]}"; do
  for bb in "${BACKBONES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      admit_job
      run_asymmetric_valid "$ds" "$bb" "$seed" &
      register_job
    done
  done
done
drain_jobs

"$PYTHON" tools/freeze_asymmetric_manifest.py \
  --results_root "$RESULTS_ROOT" --output "$MANIFEST"

running=0; stage_failed=0
echo "one-time test evaluation of frozen runs"
while IFS=$'\t' read -r base recal; do
  [[ -f "$recal/test_metrics.json" ]] && continue
  admit_job
  name="$(basename "$recal")"
  "$PYTHON" -m refrec.evaluate_recal_test \
    --base_run_dir "$base" --recal_run_dir "$recal" --device "$DEVICE" \
    >"$LOG/${name}_test.log" 2>&1 &
  register_job
done < <("$PYTHON" - "$MANIFEST" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1], encoding="utf-8"))["runs"]:
    print(f"{row['base_run_dir']}\t{row['recal_run_dir']}")
PY
)
drain_jobs

"$PYTHON" -m refrec.summarize_recal_multiseed \
  --root "$OUT/recal_runs" --output "$OUT/multiseed_test_summary.csv"
echo "main 8-dataset x 6-backbone x 3-seed test complete"
