#!/usr/bin/env bash
# RQ3: output-level controls, repeat-aware alternatives, and recent rerankers.
# Every invocation starts from the same frozen Base checkpoint.  A completed
# test_metrics.json makes the workflow resumable after interruption.
set -euo pipefail
source "$(dirname "$0")/common.sh"

STD_OUT="$RESULTS_ROOT/rq3_established_8d6b3s"
CAL_OUT="$RESULTS_ROOT/rq3_calibration_8d6b3s"
RECENT_OUT="$RESULTS_ROOT/rq3_recent_8d6b3s"
mkdir -p "$STD_OUT/logs" "$CAL_OUT/logs" "$RECENT_OUT/logs"

run_established() {
  local ds="$1" bb="$2" seed="$3" base out tag
  base="$(base_dir "$ds" "$bb" "$seed")"
  tag="${ds}_${bb}_seed${seed}"
  out="$STD_OUT/$ds/$tag"
  [[ -f "$out/test_metrics.json" ]] && return 0
  [[ -f "$base/best_model.pt" ]] || { echo "missing checkpoint: $base" >&2; return 1; }
  "$PYTHON" -m refrec.run_comparison_baselines \
    --base_run_dir "$base" --output_dir "$out" --device "$DEVICE" --seed "$seed" \
    >"$STD_OUT/logs/${tag}.log" 2>&1
}

run_controls() {
  local ds="$1" bb="$2" seed="$3" base out tag
  base="$(base_dir "$ds" "$bb" "$seed")"
  tag="${ds}_${bb}_seed${seed}"
  out="$CAL_OUT/$ds/$tag"
  [[ -f "$out/test_metrics.json" ]] && return 0
  [[ -f "$base/best_model.pt" ]] || { echo "missing checkpoint: $base" >&2; return 1; }
  "$PYTHON" -m refrec.run_tecrec_calibrators \
    --base_run_dir "$base" --output_dir "$out" --device "$DEVICE" --seed "$seed" \
    >"$CAL_OUT/logs/${tag}.log" 2>&1
}

run_recent() {
  local method="$1" ds="$2" bb="$3" seed="$4" base out tag
  base="$(base_dir "$ds" "$bb" "$seed")"
  tag="${ds}_${bb}_seed${seed}"
  out="$RECENT_OUT/$method/$ds/$tag"
  [[ -f "$out/test_metrics.json" ]] && return 0
  [[ -f "$base/best_model.pt" ]] || { echo "missing checkpoint: $base" >&2; return 1; }
  "$PYTHON" -m refrec.run_recent_output_baseline \
    --base_run_dir "$base" --output_dir "$out" --method "$method" \
    --device "$DEVICE" --seed "$seed" --pool_size 100 --topk 20 \
    >"$RECENT_OUT/logs/${method}_${tag}.log" 2>&1
}

for ds in "${ALL_DATASETS[@]}"; do
  for bb in "${BACKBONES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      admit_job; run_established "$ds" "$bb" "$seed" & register_job
      admit_job; run_controls "$ds" "$bb" "$seed" & register_job
      for method in topncal smmr duor; do
        admit_job; run_recent "$method" "$ds" "$bb" "$seed" & register_job
      done
    done
  done
done
drain_jobs

"$PYTHON" tools/summarize_rq3_runs.py \
  --established "$STD_OUT" --controls "$CAL_OUT" --recent "$RECENT_OUT" \
  --output "$RESULTS_ROOT/rq3_result_inventory.csv"
echo "RQ3 comparisons complete"
