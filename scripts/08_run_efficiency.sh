#!/usr/bin/env bash
# RQ7(b): latency, memory and parameter overhead.
set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT="$RESULTS_ROOT/recal_efficiency_v3"
LOG="$OUT/logs"
mkdir -p "$OUT/runs" "$LOG"
datasets=(ml-100k LastFM online_retail)
backbones=(gru4rec sasrec)

for ds in "${datasets[@]}"; do
  for bb in "${backbones[@]}"; do
    seed=2023
    base="$(base_dir "$ds" "$bb" "$seed")"
    recal="$(asym_recal_dir "$ds" "$bb" "$seed")"
    name="${ds}_${bb}_seed${seed}"
    out="$OUT/runs/$ds/$name"
    mkdir -p "$out"
    [[ -f "$out/efficiency.json" ]] && continue
    "$PYTHON" -m refrec.benchmark_recal_efficiency \
      --base_run_dir "$base" --recal_run_dir "$recal" \
      --output_dir "$out" --device "$DEVICE" \
      --batch_size 32 --max_users 1024 --warmup 2 --repeats 5 \
      >"$LOG/$name.log" 2>&1
  done
done
"$PYTHON" -m refrec.summarize_recal_efficiency --root "$OUT"
echo "RQ7 efficiency complete"
