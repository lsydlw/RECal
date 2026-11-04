#!/usr/bin/env python3
"""Summarize the three-seed comparison-baseline test runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("base", "hard_mask", "calibrated_recommendation", "prm", "recode")
GROUPS = ("overall", "repeat", "explore")
METRICS = ("recall@10", "ndcg@10")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/comparison_baselines_8d3s_v1")
    return p.parse_args()


def main():
    args = parse_args(); root = Path(args.root)
    files = sorted(root.glob("*/seed*/test_metrics.json"))
    if len(files) != 24:
        raise RuntimeError(f"expected 24 test files, found {len(files)}")
    records = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset, seed = payload["dataset"], int(payload["seed"])
        for method in METHODS:
            for group in GROUPS:
                result = payload["methods"][method][group]
                records.append({"dataset": dataset, "seed": seed, "method": method,
                                "group": group, "count": result["count"],
                                **{metric: result[metric] for metric in METRICS}})
    summary = []
    datasets = sorted({r["dataset"] for r in records})
    for dataset in datasets:
        for method in METHODS:
            for group in GROUPS:
                chosen = [r for r in records if r["dataset"] == dataset and
                          r["method"] == method and r["group"] == group]
                row = {"dataset": dataset, "method": method, "group": group,
                       "seeds": len(chosen), "count_mean": np.mean([r["count"] for r in chosen])}
                for metric in METRICS:
                    values = np.asarray([r[metric] for r in chosen], dtype=float)
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_std"] = float(values.std(ddof=1))
                summary.append(row)
    base_lookup = {(r["dataset"], r["group"]): r for r in summary if r["method"] == "base"}
    for row in summary:
        base = base_lookup[(row["dataset"], row["group"])]
        for metric in METRICS:
            b = base[f"{metric}_mean"]
            row[f"{metric}_relative_pct"] = 100.0 * (row[f"{metric}_mean"] - b) / b if b else 0.0
    columns = list(summary[0])
    with (root / "three_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(summary)
    (root / "three_seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("dataset\tmethod\tR@10 mean±std\tN@10 mean±std\tN@10 vs Base")
    for row in summary:
        if row["group"] != "overall": continue
        print(f"{row['dataset']}\t{row['method']}\t"
              f"{row['recall@10_mean']:.6f}±{row['recall@10_std']:.6f}\t"
              f"{row['ndcg@10_mean']:.6f}±{row['ndcg@10_std']:.6f}\t"
              f"{row['ndcg@10_relative_pct']:+.2f}%")


if __name__ == "__main__":
    main()
