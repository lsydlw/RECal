#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
DEVICE="${DEVICE:-cuda}"
MAX_WORKERS="${MAX_WORKERS:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"
ADMISSION_DELAY="${ADMISSION_DELAY:-5}"

CORE_DATASETS=(ml-100k datafoundation-user LastFM Bookmarks)
EXTERNAL_DATASETS=(amazon_music foursquare online_retail baidu_news_v2)
ALL_DATASETS=("${CORE_DATASETS[@]}" "${EXTERNAL_DATASETS[@]}")
# Six heterogeneous sequence encoders used in the paper.
BACKBONES=(gru4rec sasrec bert4rec repeatnet faesrec mamba4rec)
SEEDS=(2023 2024 2025)

data_subdir() {
  local ds="$1"
  if [[ -d "$DATA_ROOT/$ds" ]]; then
    printf '%s\n' "$ds"
    return
  fi
  case "$ds" in
    ml-100k|datafoundation-user|LastFM|Bookmarks)
      if [[ -d "$DATA_ROOT/${ds}_v2" ]]; then
        printf '%s\n' "${ds}_v2"
        return
      fi
      ;;
  esac
  printf '%s\n' "$ds"
}

data_dir() {
  local ds="$1"
  printf '%s/%s\n' "$DATA_ROOT" "$(data_subdir "$ds")"
}

family_for() {
  case "$1" in
    ml-100k|datafoundation-user|LastFM|Bookmarks) printf '%s\n' core_multiseed_v1 ;;
    *) printf '%s\n' external_multiseed_v1 ;;
  esac
}

base_dir() {
  local ds="$1" bb="$2" seed="$3" family
  family="$(family_for "$ds")"
  printf '%s/%s/base_runs/%s/%s_%s_base_seed%s\n' \
    "$RESULTS_ROOT" "$family" "$ds" "$ds" "$bb" "$seed"
}

source_recal_dir() {
  local ds="$1" bb="$2" seed="$3" family
  family="$(family_for "$ds")"
  printf '%s/%s/recal_runs/%s/%s_%s_recal_seed%s\n' \
    "$RESULTS_ROOT" "$family" "$ds" "$ds" "$bb" "$seed"
}

asym_recal_dir() {
  local ds="$1" bb="$2" seed="$3"
  printf '%s/recal_asymmetric_8datasets_v1/recal_runs/%s/%s_%s_recal_asym_seed%s\n' \
    "$RESULTS_ROOT" "$ds" "$ds" "$bb" "$seed"
}

require_dataset() {
  local ds="$1" dir f
  dir="$(data_dir "$ds")"
  for f in train.csv valid.csv test.csv item_meta.csv; do
    [[ -f "$dir/$f" ]] || {
      echo "missing required file: $dir/$f" >&2
      return 1
    }
  done
}

gpu_free_mib() {
  if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      | head -1 | tr -d ' '
  else
    printf '%s\n' 999999
  fi
}

running=0
stage_failed=0

reap_one() {
  if ! wait -n; then stage_failed=1; fi
  running=$((running - 1))
}

admit_job() {
  while (( running >= MAX_WORKERS )); do reap_one; done
  while (( $(gpu_free_mib) < MIN_FREE_MIB )); do
    if (( running > 0 )); then reap_one; else sleep 30; fi
  done
}

register_job() {
  running=$((running + 1))
  sleep "$ADMISSION_DELAY"
}

drain_jobs() {
  while (( running > 0 )); do reap_one; done
  if (( stage_failed )); then
    echo "at least one experiment failed; inspect the per-run logs" >&2
    exit 1
  fi
}
