#!/usr/bin/env bash
# RQ6: mismatch bins, intent-noise/risk sensitivity, and three-group KL projection.
set -euo pipefail
source "$(dirname "$0")/common.sh"

MECH="$RESULTS_ROOT/mechanism_3d2b3s_v1"
mkdir -p "$MECH/logs"
for ds in LastFM datafoundation-user online_retail; do
  for bb in gru4rec sasrec; do
    for seed in "${SEEDS[@]}"; do
      out="$MECH/$ds/${ds}_${bb}_seed${seed}"
      [[ -f "$out/metrics.json" ]] && continue
      admit_job
      "$PYTHON" -m refrec.scan_recal_shrinkage \
        --base_run_dir "$(base_dir "$ds" "$bb" "$seed")" \
        --recal_run_dir "$(asym_recal_dir "$ds" "$bb" "$seed")" \
        --output_dir "$out" --device "$DEVICE" --split test --seed "$seed" \
        --variants g0 asym1 asym2 asym4 asym8 asym12 asym16 asym24 \
        --noise_sigmas 0 0.1 0.2 0.4 0.8 \
        >"$MECH/logs/${ds}_${bb}_seed${seed}.log" 2>&1 &
      register_job
    done
  done
done
drain_jobs

MULTI="$RESULTS_ROOT/multigroup_kl_3d5b3s_v1"
mkdir -p "$MULTI/logs"
running=0; stage_failed=0
for ds in LastFM online_retail baidu_news_v2; do
  for bb in "${BACKBONES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out="$MULTI/$ds/${ds}_${bb}_seed${seed}"
      [[ -f "$out/metrics.json" ]] && continue
      admit_job
      "$PYTHON" -m refrec.run_multigroup_projection \
        --base_run_dir "$(base_dir "$ds" "$bb" "$seed")" \
        --recal_run_dir "$(asym_recal_dir "$ds" "$bb" "$seed")" \
        --output_dir "$out" --device "$DEVICE" --seed "$seed" \
        --lambdas 0.25,0.5,0.75,1.0 \
        >"$MULTI/logs/${ds}_${bb}_seed${seed}.log" 2>&1 &
      register_job
    done
  done
done
drain_jobs
"$PYTHON" -m refrec.summarize_multigroup_projection \
  --root "$MULTI" --output_dir "$MULTI/summary"
echo "RQ6 mechanism experiments complete"
