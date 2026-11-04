# Paper experiments and commands

All commands run from the repository root. Set `DATA_ROOT`, `RESULTS_ROOT`,
`DEVICE`, `MAX_WORKERS`, and `MIN_FREE_MIB` as shown in the main README.
The formal grid uses eight datasets, six backbones, and seeds 2023/2024/2025.
Ranking is over the full item catalog with history candidates retained.

## Mandatory order

1. `bash scripts/01_train_validation.sh`
   trains 144 Base checkpoints and 144 lightweight intent heads. Selection is
   by validation NDCG@10 and test is not read.
2. `bash scripts/02_run_main_test_once.sh`
   converts the selected heads to the final asymmetric rule, writes a frozen
   manifest with hashes, and performs the one-time test evaluation.
3. Run scripts 03--09 to reproduce the paper analyses from the frozen runs.

## RQ mapping

| RQ | Question | Command | Main outputs |
|---|---|---|---|
| RQ1 | Does repeat--explore mass mismatch exist? | `bash scripts/03_run_diagnostics_and_summary.sh` | `recal_mismatch_diagnostic_v1/{mismatch_summary.json,mismatch_bins.csv}` |
| RQ2 | Does RECal improve six backbones? | scripts 01--03 | `recal_asymmetric_8datasets_v1/multiseed_test_summary.csv` and per-run `test_metrics.json` |
| RQ3 | Does RECal beat alternative output corrections? | `bash scripts/04_run_alternative_methods.sh` | baseline and calibrator three-seed summaries |
| RQ4 | Where do the gains come from? | no new training | repeat/explore groups in each `test_metrics.json` and `test_predictions.csv` |
| RQ5 | Which components matter? | `bash scripts/05_run_ablation.sh` | `ablation_summary_mean_std.csv` and per-request ablation ranks |
| RQ6 | Do observations match the mechanism? | `bash scripts/06_run_mechanism.sh` | noise/gamma scans and `multigroup_kl_3d5b3s_v1/summary/paper_table.csv` |
| RQ7 | Are gains significant and efficient? | scripts 07--08 | bootstrap/Holm CSV and efficiency summary CSV |

## RQ3 methods

`04_run_alternative_methods.sh` applies every method to the same frozen Base
checkpoint for every dataset--backbone--seed triple. It writes three output
families: (i) Hard Mask, CalRec, PRM, and ReCODE-style; (ii) TS, GLS, CalRec,
and TecRec-style; and (iii) Top-N Cal., SMMR, and DUoR. Listwise methods use
the Base Top-100 candidates; all parameters are selected using validation
requests only. TS changes probability sharpness but preserves ranking, so its
ranking metrics equal Base by construction.

## RQ5 variants

`run_recal_ablations.py` evaluates Base, fixed history boost, fixed global
gate, entropy confidence, intent-only (`q` without Base mass `m`), global
repeat rate, no explicit prefix statistics, symmetric risk gate, unguarded
full projection, and the final one-sided risk gate. The main paper may display
a subset; the full output is retained for the appendix.

## RQ6 settings

- noise/gamma: LastFM, DataFoundation, and Online Retail; GRU4Rec/SASRec;
  three seeds; noise `{0,0.1,0.2,0.4,0.8}`; risk exponents
  `{1,2,4,8,12,16,24}`;
- multi-group KL: LastFM, Online Retail, and Baidu News v2; all six
  backbones; three seeds; recent-repeat / long-term-repeat / exploration.

## Independent checks

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 tools/validate_data.py "$DATA_ROOT/ml-100k"
python3 -m compileall -q refrec tools
```

The private experiment archive, not this public repository, contains the
original logs, checkpoints, per-user ranks, and the paper-to-source index.
