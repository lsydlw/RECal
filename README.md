# RECal: Repeat--Exploration Calibration

This repository contains the result-free implementation used by the RECal
paper. RECal is a lightweight post-hoc layer for sequential recommenders: it
estimates request-level repeat intent, compares it with the frozen backbone's
probability mass on history items, applies a directional risk gate, and uses a
closed-form KL projection to redistribute mass while preserving within-group
ranking.

## What is included

- exact full-catalog evaluation with history items retained;
- GRU4Rec, SASRec, BERT4Rec, RepeatNet, FAESRec, and Mamba4Rec adapters;
- the two-group RECal model and the three-group KL extension;
- TS, GLS, Hard Mask, Top-N Cal., CalRec, PRM, TecRec-style, SMMR, DUoR,
  and ReCODE-style comparisons;
- mismatch diagnostics, grouped evaluation, ablations, robustness,
  significance, and efficiency analyses;
- leakage/protocol/projection unit tests and portable launch scripts.

No paper results, checkpoints, logs, raw datasets, server addresses, or local
absolute paths are included in this public package.

## Installation

```bash
conda env create -f environment.yml
conda activate recal
pip install -r requirements.txt
# Optional, required only for FAESRec/Mamba4Rec:
pip install -r requirements-faesrec.txt
python3 -m unittest discover -s tests -p "test_*.py" -v
```

## Data

Prepare the eight datasets with the four-CSV contract described in
[`docs/DATA.md`](docs/DATA.md). Raw datasets are not redistributed because
their licenses differ.

## Reproduce the paper workflow

```bash
export DATA_ROOT=/path/to/processed_data
export RESULTS_ROOT=$PWD/results
export DEVICE=cuda
export MAX_WORKERS=3
export MIN_FREE_MIB=12000

bash scripts/01_train_validation.sh
bash scripts/02_run_main_test_once.sh
bash scripts/03_run_diagnostics_and_summary.sh
bash scripts/04_run_alternative_methods.sh
bash scripts/05_run_ablation.sh
bash scripts/06_run_mechanism.sh
bash scripts/07_run_significance.sh
bash scripts/08_run_efficiency.sh
bash scripts/09_make_figures.sh
```

The scripts are resumable: a run with a completed `test_metrics.json` is not
overwritten. `MAX_WORKERS` and `MIN_FREE_MIB` bound parallel jobs and reserve
GPU memory for other users. To reproduce only one research question, see the
RQ-to-script mapping in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

Stage 1 selects checkpoints using validation data. Stage 2 freezes a manifest
and evaluates test exactly once. Do not use test outputs for tuning. See
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the RQ-by-RQ mapping and
[`docs/CODE_MAP.md`](docs/CODE_MAP.md) for source responsibilities.

## Paper protocol

- datasets: ML-100K, DataFoundation, Bookmarks, LastFM, Amazon Music,
  Foursquare, Online Retail, and Baidu News v2;
- backbones: GRU4Rec, SASRec, BERT4Rec, RepeatNet, FAESRec, and Mamba4Rec;
- seeds: 2023, 2024, and 2025;
- selection: validation NDCG@10;
- evaluation: exact full catalog, history candidates included;
- tie breaking: score descending, item ID ascending;
- default directional-risk exponent: 12.

## Third-party code and licensing

`third_party/FAESRec` contains the upstream encoder files required by the
adapter. Review the upstream project terms before redistribution. This package
does not select an open-source license on behalf of the authors. In
particular, remove or replace vendored third-party code if its license is
incompatible with the license chosen for a public release.
