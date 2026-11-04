# Code map

| File | Responsibility |
|---|---|
| `refrec/recal.py` | intent head, explicit prefix statistics, risk gate, and closed-form projection |
| `refrec/run_recal_frozen.py` | train/validate the lightweight head on a frozen Base |
| `refrec/evaluate_recal_test.py` | one-time full-catalog test evaluation and rank export |
| `refrec/run_seqrec.py` | common training entry for all six backbones |
| `refrec/seqrec.py` | GRU4Rec, SASRec, BERT4Rec, and RepeatNet implementations |
| `refrec/faesrec_adapter.py` | FAESRec encoder adapter under the shared protocol |
| `refrec/mamba4rec_adapter.py` | Mamba4Rec selective-state encoder adapter under the shared protocol |
| `third_party/FAESRec/` | vendored upstream FAESRec encoder code |
| `refrec/data_utils.py` | four-CSV loading, prefixes, history sets, and negative sampling |
| `refrec/evaluate.py`, `refrec/metrics.py` | exact chunked full ranking and metrics |
| `refrec/protocol.py` | candidate, history, tie-break, and output-protection contract |
| `refrec/run_tecrec_calibrators.py` | TS, GLS, CalRec, and TecRec-style controls |
| `refrec/run_comparison_baselines.py` | Hard Mask, PRM, and ReCODE-style comparison path |
| `refrec/run_recent_output_baseline.py` | Top-N Cal., SMMR, and DUoR frozen-output rerankers |
| `refrec/run_recal_ablations.py` | RQ5 variants and per-request rank export |
| `refrec/scan_recal_shrinkage.py` | intent-noise and risk-exponent scans |
| `refrec/run_multigroup_projection.py` | three-group KL projection experiment |
| `refrec/analyze_recal_significance.py` | paired hierarchical bootstrap and Holm correction |
| `refrec/benchmark_recal_efficiency.py` | parameter, latency, and memory overhead |
| `scripts/01...09` | portable end-to-end paper workflow; script 04 runs all RQ3 families |

Files related to the earlier REFRec development path are retained for audit
compatibility but are not required by the final RECal workflow.
