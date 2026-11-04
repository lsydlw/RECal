#!/usr/bin/env bash
# Full paper workflow. Use only after reading docs/EXPERIMENTS.md.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
"$here/01_train_validation.sh"
"$here/02_run_main_test_once.sh"
"$here/03_run_diagnostics_and_summary.sh"
"$here/04_run_alternative_methods.sh"
"$here/05_run_ablation.sh"
"$here/06_run_mechanism.sh"
"$here/07_run_significance.sh"
"$here/08_run_efficiency.sh"
"$here/09_make_figures.sh"
