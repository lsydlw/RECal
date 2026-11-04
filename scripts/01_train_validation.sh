#!/usr/bin/env bash
# Train all frozen Base checkpoints and validation-only RECal intent heads.
set -euo pipefail
source "$(dirname "$0")/common.sh"

for ds in "${ALL_DATASETS[@]}"; do require_dataset "$ds"; done

run_base() {
  local ds="$1" bb="$2" seed="$3" family out name run log
  local -a extra=()
  family="$(family_for "$ds")"
  out="$RESULTS_ROOT/$family/base_runs"
  name="${ds}_${bb}_base_seed${seed}"
  run="$out/$ds/$name"
  log="$RESULTS_ROOT/$family/logs/${name}.log"
  [[ -f "$run/best_model.pt" && -f "$run/metrics.json" ]] && return 0
  mkdir -p "$(dirname "$log")"
  if [[ "$bb" == faesrec ]]; then
    extra=(--emb_dim 64 --num_blocks 2 --num_heads 2 --dropout 0.5 --cat_weight 0)
  elif [[ "$bb" == mamba4rec ]]; then
    extra=(--emb_dim 128 --mamba_layers 1 --mamba_d_state 32 --mamba_d_conv 4 --mamba_expand 2)
  fi
  "$PYTHON" -m refrec.run_seqrec \
    --dataset "$ds" --data_dir "$(data_dir "$ds")" \
    --output_dir "$out" --run_name "$name" --backbone "$bb" \
    --repeat_head base --negative_sampling all_except_target \
    --device "$DEVICE" --seed "$seed" --skip_test --no_progress "${extra[@]}" \
    >"$log" 2>&1
}

echo "BASE validation-only training"
for ds in "${ALL_DATASETS[@]}"; do
  for bb in "${BACKBONES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      admit_job
      run_base "$ds" "$bb" "$seed" &
      register_job
    done
  done
done
drain_jobs

run_head() {
  local ds="$1" bb="$2" seed="$3" family base out log
  family="$(family_for "$ds")"
  base="$(base_dir "$ds" "$bb" "$seed")"
  out="$(source_recal_dir "$ds" "$bb" "$seed")"
  log="$RESULTS_ROOT/$family/logs/${ds}_${bb}_recal_seed${seed}.log"
  [[ -f "$out/intent_head.pt" && -f "$out/metrics.json" ]] && return 0
  [[ -f "$base/best_model.pt" ]] || { echo "missing Base checkpoint: $base" >&2; return 1; }
  "$PYTHON" -m refrec.run_recal_frozen \
    --base_run_dir "$base" --output_dir "$out" --device "$DEVICE" \
    --seed "$seed" --hidden_dim 64 --calibration_mode asymmetric \
    --down_power 12 >"$log" 2>&1
}

running=0; stage_failed=0
echo "RECal intent-head validation training"
for ds in "${ALL_DATASETS[@]}"; do
  for bb in "${BACKBONES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      admit_job
      run_head "$ds" "$bb" "$seed" &
      register_job
    done
  done
done
drain_jobs

for family in core_multiseed_v1 external_multiseed_v1; do
  "$PYTHON" -m refrec.freeze_core_manifest \
    --root "$RESULTS_ROOT/$family/base_runs" \
    --recal_root "$RESULTS_ROOT/$family/recal_runs" \
    --output "$RESULTS_ROOT/$family/frozen_manifest.json"
done
echo "validation stage complete; test has not been read"
