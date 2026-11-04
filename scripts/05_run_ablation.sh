#!/usr/bin/env bash
# RQ5: eight datasets x six backbones x three seeds, evaluated on frozen test.
set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT="$RESULTS_ROOT/recal_ablation_test_8d4b3s_v1"
LOG="$OUT/logs"
mkdir -p "$OUT" "$LOG"

run_one() {
  local ds="$1" bb="$2" seed="$3" base recal name out
  base="$(base_dir "$ds" "$bb" "$seed")"
  recal="$(asym_recal_dir "$ds" "$bb" "$seed")"
  name="${ds}_${bb}_ablation_test_seed${seed}"
  out="$OUT/$ds/$name"
  [[ -f "$out/metrics.json" ]] && return 0
  [[ -f "$base/best_model.pt" && -f "$recal/intent_head.pt" ]] || {
    echo "missing frozen input for $ds $bb seed=$seed" >&2; return 1;
  }
  "$PYTHON" -m refrec.run_recal_ablations \
    --base_run_dir "$base" --recal_run_dir "$recal" \
    --output_dir "$out" --split test --device "$DEVICE" --seed "$seed" \
    --risk_power 12 >"$LOG/$name.log" 2>&1
}

for ds in "${ALL_DATASETS[@]}"; do
  for bb in "${BACKBONES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      admit_job
      run_one "$ds" "$bb" "$seed" &
      register_job
    done
  done
done
drain_jobs

"$PYTHON" -m refrec.summarize_recal_ablations --root "$OUT"
"$PYTHON" -m refrec.summarize_recal_ablations_multiseed --root "$OUT"
echo "RQ5 ablation complete"
