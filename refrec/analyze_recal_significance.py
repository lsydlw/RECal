#!/usr/bin/env python3
"""Paired hierarchical bootstrap significance for Base versus RECal."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Root containing recal_runs/<dataset>/<run>/test_predictions.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261010)
    return parser.parse_args()


def ndcg10(rank):
    rank = int(rank)
    return 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0


def recall10(rank):
    return float(int(rank) <= 10)


def mrr10(rank):
    rank = int(rank)
    return 1.0 / rank if rank <= 10 else 0.0


def load_run(path):
    data = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            user = str(row["user_id"])
            br, rr = int(row["base_rank"]), int(row["recal_rank"])
            data[user] = {
                "base_ndcg@10": ndcg10(br), "recal_ndcg@10": ndcg10(rr),
                "base_recall@10": recall10(br), "recal_recall@10": recall10(rr),
                "base_mrr@10": mrr10(br), "recal_mrr@10": mrr10(rr),
            }
    return data


def hierarchical_bootstrap(diff, repeats, rng, batch_size=32):
    """Resample model seeds and aligned users; return replicate mean deltas."""
    seeds, users = diff.shape
    values = []
    for start in range(0, repeats, batch_size):
        size = min(batch_size, repeats - start)
        seed_index = rng.integers(0, seeds, size=(size, seeds))
        selected = diff[seed_index].mean(axis=1)
        user_index = rng.integers(0, users, size=(size, users))
        values.append(np.take_along_axis(selected, user_index, axis=1).mean(axis=1))
    return np.concatenate(values)


def holm_adjust(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_bootstrap"])
    adjusted = [1.0] * len(rows)
    running = 0.0
    total = len(rows)
    for position, index in enumerate(order):
        value = min(1.0, (total - position) * rows[index]["p_bootstrap"])
        running = max(running, value)
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
        row["significant_holm_0.05"] = value < 0.05


def main():
    args = parse_args(); root = Path(args.root); rng = np.random.default_rng(args.seed)
    grouped = defaultdict(list)
    for path in sorted(root.glob("recal_runs/*/*/test_predictions.csv")):
        dataset = path.parents[1].name
        run_name = path.parent.name
        backbone = next((b for b in ("gru4rec", "sasrec", "bert4rec", "repeatnet")
                         if f"_{b}_" in run_name), None)
        if backbone is None: raise RuntimeError(f"Cannot parse backbone from {run_name}")
        grouped[(dataset, backbone)].append(path)
    rows = []
    for (dataset, backbone), paths in sorted(grouped.items()):
        if len(paths) != 3:
            raise RuntimeError(f"{dataset}/{backbone}: expected 3 seeds, got {len(paths)}")
        runs = [load_run(path) for path in paths]
        users = sorted(set.intersection(*(set(run) for run in runs)))
        if not users: raise RuntimeError(f"No aligned users for {dataset}/{backbone}")
        metric_values = {}
        for metric in ("ndcg@10", "recall@10", "mrr@10"):
            base = np.asarray([[run[user][f"base_{metric}"] for user in users] for run in runs])
            recal = np.asarray([[run[user][f"recal_{metric}"] for user in users] for run in runs])
            metric_values[metric] = (float(base.mean()), float(recal.mean()), recal - base)
        base_mean, recal_mean, diff = metric_values["ndcg@10"]
        boot = hierarchical_bootstrap(diff, args.bootstrap, rng)
        below = (np.count_nonzero(boot <= 0.0) + 1) / (len(boot) + 1)
        above = (np.count_nonzero(boot >= 0.0) + 1) / (len(boot) + 1)
        row = {
            "dataset": dataset, "backbone": backbone, "num_seeds": len(runs),
            "aligned_users": len(users), "bootstrap_replicates": args.bootstrap,
            "base_ndcg@10": base_mean, "recal_ndcg@10": recal_mean,
            "delta_ndcg@10": recal_mean - base_mean,
            "relative_ndcg@10_pct": 100.0 * (recal_mean / base_mean - 1.0) if base_mean else float("nan"),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "p_bootstrap": min(1.0, 2.0 * min(below, above)),
        }
        for metric in ("recall@10", "mrr@10"):
            base_m, recal_m, _ = metric_values[metric]
            row[f"base_{metric}"] = base_m; row[f"recal_{metric}"] = recal_m
            row[f"delta_{metric}"] = recal_m - base_m
        rows.append(row)
    if len(rows) != 32:
        raise RuntimeError(f"Expected 32 dataset/backbone comparisons, got {len(rows)}")
    holm_adjust(rows)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} comparisons to {output}; Holm-significant: "
          f"{sum(r['significant_holm_0.05'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
