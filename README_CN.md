# RECal 复现代码

本目录是可公开发布的、**不包含实验结果**的 RECal 复现包。它实现了请求级重复意图估计、非对称风险门控和闭式 KL 概率质量投影，并覆盖论文的 RQ1--RQ7。

代码不包含原始数据、缓存、检查点、日志、逐用户排名、服务器地址或密钥。请先阅读英文主说明 [`README.md`](README.md)、数据格式说明 [`docs/DATA.md`](docs/DATA.md) 和实验映射 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

主要覆盖范围：

- 六种骨干：GRU4Rec、SASRec、BERT4Rec、RepeatNet、FAESRec 和 Mamba4Rec；
- 八个数据集的全目录排序与验证集选参；
- RQ1 失配诊断、RQ2 总体性能、RQ3 输出校准/重排对比、RQ4 分组分析、RQ5 消融、RQ6 机制分析、RQ7 显著性与效率；
- Top-N Cal.、SMMR、DUoR、CalRec、PRM、TecRec-style、ReCODE-style 等对照；
- 可恢复脚本：完成的 `test_metrics.json` 不会被覆盖。

典型运行顺序为：

```bash
conda env create -f environment.yml
conda activate recal
pip install -r requirements.txt
pip install -r requirements-faesrec.txt  # 仅 FAESRec/Mamba4Rec 所需
export DATA_ROOT=/path/to/data
export RESULTS_ROOT=$PWD/results
bash scripts/01_train_validation.sh
bash scripts/02_run_main_test_once.sh
```

之后依次执行 `scripts/03` 到 `scripts/09` 即可复现对应 RQ。上传 GitHub 前，请按 [`docs/PUBLISH_CHECKLIST.md`](docs/PUBLISH_CHECKLIST.md) 选择许可证并确认第三方代码许可。
