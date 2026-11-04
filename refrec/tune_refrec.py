#!/usr/bin/env python3
"""REFRec 随机参数搜索：每个 trial 完整训练 + 测试，并逐个报告测试集指标。

设计要点：
- 每个 trial 通过子进程调用 refrec/run_refrec.py，复用现有 CLI（隔离崩溃、OOM）。
- 驱动进程只依赖标准库，不需要 torch/optuna。
- 每个 trial 的完整输出（config.json / metrics.json / train.log / 分组指标 / 预测）都落在
  <output_dir>/<dataset>/trial_XXXX/ 下；汇总写入 <output_dir>/<dataset>/trials.csv。
- 参数对每个 trial 用 (seed, dataset, trial) 确定性采样，可 --resume 断点续跑且不改变后续参数。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 搜索空间（可直接增删）。key 对应 run_refrec.py 的 --<key> 参数，value 是候选值列表，
# 随机搜索从每个列表里等概率采样。
# 说明：固定 repeat_mode=residual（带残差），因此这里的重复专家相关超参（
# refrec_init_repeat_weight / lambda_refrec_orth / gamma_repeat）都在搜。
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, list[Any]] = {
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "emb_dim": [64, 128, 256],
    "dropout": [0.0, 0.1, 0.3, 0.5],
    "weight_decay": [0.0, 1e-5, 1e-4],
    "batch_size": [128, 256, 512],
    "num_neg": [50, 100, 200],
    "gamma_weak": [0.5, 0.7, 0.8, 0.9],
    "gamma_repeat": [0.6, 0.7, 0.8, 0.9, 1.0],
    "alpha_main": [0.5, 1.0, 2.0],
    "alpha_recent": [0.5, 1.0, 2.0],
    "cat_weight": [0.0, 0.1, 0.2, 0.5],
    "cat_smooth_k": [1.0, 2.0, 3.0, 5.0],
    "refrec_init_repeat_weight": [0.2, 0.5, 0.8, 1.0],
    "lambda_refrec_orth": [0.0, 0.01, 0.1],
}

# 每个 trial 固定不变的参数（与 residual_rw0.8 参考一致；main_topk/min_main_count/
# recent_window/max_seq_len 不传，走 run_refrec 的 dataset 默认值，避免反复重建样本缓存）。
FIXED: dict[str, Any] = {
    "repeat_mode": "residual",
    "seed": 2023,
    "epochs": 100,
    "patience": 10,
    "max_seq_len": 50,
    "recent_window": 5,
    "device": "cuda",
    "no_progress": True,
}

METRIC_KEYS = ["recall@5", "ndcg@5", "recall@10", "ndcg@10", "recall@20", "ndcg@20", "mrr@10"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="REFRec random parameter search (per-trial test report)")
    p.add_argument("--dataset", required=True, help="例如 LastFM 或 Bookmarks")
    p.add_argument("--data_root", default="data")
    p.add_argument("--output_dir", default="results/refrec/param_search")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--device", default="cuda")
    p.add_argument("--trial_timeout", type=int, default=7200, help="单个 trial 超时秒数，超时判 FAIL")
    p.add_argument("--run_refrec", default=str(ROOT / "refrec" / "run_refrec.py"))
    p.add_argument("--resume", action="store_true", help="跳过已存在 metrics.json 的 trial，续跑失败/未完成的 trial")
    p.add_argument("--dry_run", action="store_true", help="只打印每个 trial 的命令，不真正执行")
    return p.parse_args()


def sample_params(dataset: str, seed: int, trial_num: int) -> dict[str, Any]:
    # 每个 trial 的参数由 (seed, dataset, trial) 唯一确定，resume 不影响尚未跑的 trial 的采样。
    rng = random.Random(f"{seed}:{dataset}:{trial_num}")
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def build_command(args: argparse.Namespace, dataset: str, trial_num: int,
                  params: dict[str, Any]) -> tuple[list[str], Path]:
    out_root = Path(args.output_dir).resolve()
    data_root = Path(args.data_root).resolve()
    run_name = f"trial_{trial_num:04d}"
    cmd = [
        sys.executable, args.run_refrec,
        "--dataset", dataset,
        "--data_root", str(data_root),
        "--output_dir", str(out_root),
        "--run_name", run_name,
        "--seed", "2023",
        "--device", args.device,
        "--repeat_mode", "residual",
        "--no_progress",
    ]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]
    return cmd, out_root / dataset / run_name


def _fmt(v: Any) -> Any:
    return round(float(v), 6) if isinstance(v, float) else v


def _metrics_from(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    out = Path(args.output_dir).resolve() / dataset
    out.mkdir(parents=True, exist_ok=True)

    (out / "search_space.json").write_text(
        json.dumps({"dataset": dataset, "search_space": SEARCH_SPACE, "fixed": FIXED},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    param_cols = list(SEARCH_SPACE.keys())
    metric_cols = [f"test_{k}" for k in METRIC_KEYS]
    cols = ["trial_number", "state", "best_epoch", "valid_ndcg@10",
            "wall_time_s", *metric_cols, *param_cols]
    trials_csv = out / "trials.csv"

    if not args.resume and trials_csv.exists():
        raise SystemExit(
            f"{trials_csv} 已存在。用新的 --output_dir 开新搜索，或加 --resume 续跑。"
        )

    rows: list[dict[str, Any]] = []
    if trials_csv.exists():
        with trials_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    def write_csv() -> None:
        if not rows:
            return
        tmp = trials_csv.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(trials_csv)

    def fail_row(trial_num: int, params: dict[str, Any], wall: float, why: str) -> dict[str, Any]:
        row = {c: "" for c in cols}
        row.update({"trial_number": trial_num, "state": "FAIL",
                    "wall_time_s": round(wall, 2), **params})
        print(f"[FAIL] trial={trial_num} {why}")
        return row

    best = {"score": -1.0, "trial": None, "run_name": None}
    for i in range(args.trials):
        run_name = f"trial_{i:04d}"
        run_dir = out / run_name
        if args.resume and _metrics_from(run_dir) is not None:
            print(f"[skip] trial={i} 已有 metrics.json，跳过")
            continue

        params = sample_params(dataset, args.seed, i)
        cmd, run_dir = build_command(args, dataset, i, params)

        print(f"\n=== trial {i} ===")
        print("params:", json.dumps({k: _fmt(v) for k, v in params.items()}, ensure_ascii=False))
        print("cmd:", " ".join(cmd))
        if args.dry_run:
            continue

        # 清掉可能残留的半成品目录（run_refrec 会拒绝覆盖已存在的 config/metrics）
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                                  timeout=args.trial_timeout)
            wall = time.perf_counter() - t0
            (run_dir / "process.stdout.log").write_text(proc.stdout, encoding="utf-8")
            (run_dir / "process.stderr.log").write_text(proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                rows.append(fail_row(i, params, wall, f"returncode={proc.returncode}"))
                write_csv()
                print(proc.stderr[-3000:])
                continue
            m = _metrics_from(run_dir)
            test_m = (m or {}).get("test_metrics") or {}
            if not test_m:
                rows.append(fail_row(i, params, wall, "缺 test_metrics（可能未跑到测试）"))
                write_csv()
                continue
            row = {c: "" for c in cols}
            row.update({
                "trial_number": i, "state": "COMPLETE",
                "best_epoch": (m or {}).get("best_epoch", ""),
                "valid_ndcg@10": _fmt(((m or {}).get("valid_metrics") or {}).get("ndcg@10", 0.0)),
                "wall_time_s": round(wall, 2),
                **{f"test_{k}": _fmt(test_m.get(k, 0.0)) for k in METRIC_KEYS},
                **params,
            })
            rows.append(row)
            write_csv()
            score = float(test_m.get("ndcg@10", 0.0))
            print(f"[OK] trial={i} test_ndcg@10={score:.6f} "
                  f"test_recall@10={test_m.get('recall@10', 0):.6f} wall={wall:.0f}s")
            if score > best["score"]:
                best = {"score": score, "trial": i, "run_name": run_name}
                (out / "best_config.json").write_text(
                    json.dumps({"trial": i, "test_ndcg@10": score, "run_name": run_name,
                                "params": params, "test_metrics": test_m,
                                "best_epoch": (m or {}).get("best_epoch"),
                                "valid_metrics": (m or {}).get("valid_metrics")},
                               indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - t0
            rows.append(fail_row(i, params, wall, f"timeout>{args.trial_timeout}s"))
            write_csv()
        except Exception as e:  # noqa: BLE001
            wall = time.perf_counter() - t0
            rows.append(fail_row(i, params, wall, f"exception={e}"))
            write_csv()

    write_csv()
    print(f"\ndone. best trial={best['trial']} run={best['run_name']} "
          f"test_ndcg@10={best['score']:.6f}")


if __name__ == "__main__":
    main()
