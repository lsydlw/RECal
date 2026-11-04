#!/usr/bin/env python3
"""Aggregate multi-seed validation sensitivity scans into tidy CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected_seeds", nargs="+", type=int, default=[2023, 2024, 2025])
    return parser.parse_args()


def mean_std(values):
    values = [float(v) for v in values]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    args = parse_args()
    root = Path(args.root)
    seed_rows = []
    grouped = defaultdict(list)
    for path in sorted(root.glob("runs/*/*/metrics.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        seed = int(path.parent.name.rsplit("seed", 1)[-1])
        for variant, groups in obj["variants"].items():
            power = float(variant.removeprefix("asym"))
            row = {
                "dataset": obj["dataset"], "backbone": obj["backbone"],
                "seed": seed, "power": power,
            }
            for group in ("overall", "repeat", "explore"):
                for metric in ("recall@10", "ndcg@10", "mrr@10"):
                    value = groups[group].get(metric, float("nan"))
                    row[f"{group}_{metric}"] = value
            seed_rows.append(row)
            grouped[(obj["dataset"], obj["backbone"], power)].append(row)

    if not seed_rows:
        raise SystemExit(f"No metrics.json files found under {root / 'runs'}")
    expected = sorted(args.expected_seeds)
    aggregate = []
    metric_cols = [k for k in seed_rows[0] if "@10" in k]
    for (dataset, backbone, power), rows in sorted(grouped.items()):
        seeds = sorted(r["seed"] for r in rows)
        if seeds != expected:
            raise RuntimeError(f"{dataset}/{backbone}/p={power}: expected {expected}, got {seeds}")
        out = {"dataset": dataset, "backbone": backbone, "power": power,
               "num_seeds": len(rows), "seeds": ",".join(map(str, seeds))}
        for col in metric_cols:
            vals = [r[col] for r in rows if math.isfinite(float(r[col]))]
            avg, std = mean_std(vals) if vals else (float("nan"), float("nan"))
            out[f"{col}_mean"] = avg
            out[f"{col}_std"] = std
        aggregate.append(out)

    for name, rows in (("sensitivity_seed_results.csv", seed_rows),
                       ("sensitivity_mean_std.csv", aggregate)):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(seed_rows)} seed rows and {len(aggregate)} aggregate rows to {root}")


if __name__ == "__main__":
    main()
