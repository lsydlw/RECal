#!/usr/bin/env python3
"""REFRec 消融实验 runner（A/B/C 组）。

每个变体通过子进程调用 refrec/run_refrec.py，与正式训练路径完全一致；
每个变体的完整输出落在 <output_dir>/<dataset>/<variant>/ 下，
汇总写入 <output_dir>/<dataset>/summary.csv 与 summary.md。

用法示例（服务器）:
  python refrec/run_ablation.py --datasets LastFM,Bookmarks --device cuda
  python refrec/run_ablation.py --datasets ml-100k --variants full,raw,none --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# 每个变体只写「与基准不同」的开关；基准固定为 repeat_mode=residual、
# refrec_init_repeat_weight=0.8、lambda_refrec_orth=0、seed=2023、epochs=100、patience=10。
VARIANTS: dict[str, dict[str, Any]] = {
    "full":       {"label": "A1/C1 full residual",  "flags": []},
    "raw":        {"label": "A2 raw (no beta sub)", "flags": ["--repeat_mode", "raw"]},
    "none":       {"label": "A3 no repeat expert",  "flags": ["--repeat_mode", "none"]},
    "no_support": {"label": "A4 no support mask",   "flags": ["--no_repeat_support"]},
    "beta1":      {"label": "B1 fix beta=1",        "flags": ["--fix_beta", "1"]},
    "fixscale":   {"label": "B3 fix scale=0.8",     "flags": ["--fix_scale", "0.8"]},
    "orth001":    {"label": "C2 lambda_orth=0.01",  "flags": ["--lambda_refrec_orth", "0.01"]},
    "softmax":    {"label": "E1 repeat-in-softmax (FARRec)", "flags": ["--fuse_mode", "softmax"]},
}

BASE_FLAGS = [
    "--repeat_mode", "residual",
    "--refrec_init_repeat_weight", "0.8",
    "--lambda_refrec_orth", "0",
    "--seed", "2023",
    "--epochs", "100",
    "--patience", "10",
    "--no_progress",
]

METRIC_KEYS = ["recall@5", "ndcg@5", "recall@10", "ndcg@10", "recall@20", "ndcg@20", "mrr@10"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="REFRec ablation runner")
    p.add_argument("--datasets", required=True, help="逗号分隔，如 LastFM,Bookmarks")
    p.add_argument("--variants", default="", help="逗号分隔的变体子集，缺省=全部")
    p.add_argument("--data_root", default="data")
    p.add_argument("--output_dir", default="results/refrec/ablation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval_batch_size", type=int, default=256,
                   help="评估批量大小；显存不足时可调小。")
    p.add_argument("--decode_chunk", type=int, default=2048,
                   help="评估候选分块大小；只影响显存/耗时，不影响分数。默认 2048 避免大目录 OOM。")
    p.add_argument("--run_refrec", default=str(ROOT / "refrec" / "run_refrec.py"))
    p.add_argument("--skip_existing", action="store_true", help="跳过已有 metrics.json 的变体")
    p.add_argument("--dry_run", action="store_true", help="只打印命令不执行")
    return p.parse_args()


def build_command(args: argparse.Namespace, dataset: str, variant: str) -> list[str]:
    out_root = Path(args.output_dir).resolve()
    data_root = Path(args.data_root).resolve()
    return [
        sys.executable, args.run_refrec,
        "--dataset", dataset,
        "--data_root", str(data_root),
        "--output_dir", str(out_root),
        "--run_name", variant,
        "--device", args.device,
        "--eval_batch_size", str(args.eval_batch_size),
        "--decode_chunk", str(args.decode_chunk),
        *BASE_FLAGS,
        *VARIANTS[variant]["flags"],
    ]


def _fmt(v: Any) -> Any:
    return round(float(v), 6) if isinstance(v, float) else v


def load_metrics(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_repeat_groups(run_dir: Path) -> dict[str, dict[str, str]]:
    p = run_dir / "group_metrics_repeat.csv"
    out: dict[str, dict[str, str]] = {}
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row.get("group", "")] = row
    return out


def _write_summary(out: Path, cols: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path = out / "summary.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(csv_path)

    md_path = out / "summary.md"
    lines = [
        "# REFRec 消融结果",
        "",
        "| variant | label | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | repeat_cnt | repeat_NDCG@10 | nonrepeat_NDCG@10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("state") != "COMPLETE":
            lines.append(f"| {r['variant']} | {r['label']} | FAIL | | | | | | |")
            continue
        lines.append(
            f"| {r['variant']} | {r['label']} "
            f"| {r.get('test_recall@10', '')} | {r.get('test_ndcg@10', '')} "
            f"| {r.get('test_recall@20', '')} | {r.get('test_ndcg@20', '')} "
            f"| {r.get('repeat_count', '')} | {r.get('repeat_ndcg@10', '')} "
            f"| {r.get('nonrepeat_ndcg@10', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    variants = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants else list(VARIANTS.keys())
    )
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"未知变体: {unknown}；可用: {list(VARIANTS.keys())}")

    cols = [
        "dataset", "variant", "label", "state", "best_epoch", "wall_time_s",
        *[f"test_{k}" for k in METRIC_KEYS],
        "repeat_count", "repeat_ndcg@10", "nonrepeat_ndcg@10",
    ]

    for dataset in datasets:
        out = Path(args.output_dir).resolve() / dataset
        out.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for variant in variants:
            run_dir = out / variant
            if args.skip_existing and load_metrics(run_dir) is not None:
                print(f"[skip] {dataset}/{variant} 已有结果")
                continue
            cmd = build_command(args, dataset, variant)
            print(f"\n=== {dataset}/{variant} ({VARIANTS[variant]['label']}) ===")
            print("cmd:", " ".join(cmd))
            if args.dry_run:
                continue
            shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            wall = time.perf_counter() - t0
            (run_dir / "process.stdout.log").write_text(proc.stdout, encoding="utf-8")
            (run_dir / "process.stderr.log").write_text(proc.stderr, encoding="utf-8")
            m = load_metrics(run_dir)
            test_m = (m or {}).get("test_metrics") or {}
            if proc.returncode != 0 or not test_m:
                rows.append({"dataset": dataset, "variant": variant,
                             "label": VARIANTS[variant]["label"], "state": "FAIL",
                             "wall_time_s": round(wall, 2)})
                print(f"[FAIL] {dataset}/{variant} rc={proc.returncode}")
                if proc.returncode != 0:
                    print(proc.stderr[-2500:])
                continue
            grp = read_repeat_groups(run_dir)
            rep = grp.get("repeat_target", {})
            nonrep = grp.get("non_repeat_target", {})
            rows.append({
                "dataset": dataset, "variant": variant,
                "label": VARIANTS[variant]["label"], "state": "COMPLETE",
                "best_epoch": (m or {}).get("best_epoch", ""),
                "wall_time_s": round(wall, 2),
                **{f"test_{k}": _fmt(test_m.get(k, 0.0)) for k in METRIC_KEYS},
                "repeat_count": rep.get("count", ""),
                "repeat_ndcg@10": rep.get("ndcg@10", ""),
                "nonrepeat_ndcg@10": nonrep.get("ndcg@10", ""),
            })
            print(f"[OK] {dataset}/{variant} test_ndcg@10={test_m.get('ndcg@10', 0):.6f} "
                  f"wall={wall:.0f}s")
            _write_summary(out, cols, rows)

        _write_summary(out, cols, rows)


if __name__ == "__main__":
    main()
