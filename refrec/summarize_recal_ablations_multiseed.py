#!/usr/bin/env python3
"""Aggregate RECal ablation metrics across frozen random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


GROUPS = ("overall", "repeat", "explore")
METRICS = ("recall@10", "ndcg@10", "mrr@10")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected_seeds", nargs="+", type=int,
                        default=(2023, 2024, 2025))
    args = parser.parse_args()
    root = Path(args.root)
    expected = tuple(sorted(args.expected_seeds))
    grouped = defaultdict(list)
    for path in sorted(root.glob("*/*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        seed = int(result["seed"])
        for variant, groups in result["variants"].items():
            key = (result["dataset"], result["backbone"],
                   result.get("split", "valid"), variant)
            grouped[key].append((seed, groups))
    if not grouped:
        raise SystemExit(f"No metrics.json files under {root}")

    rows = []
    for (dataset, backbone, split, variant), seed_groups in sorted(grouped.items()):
        seeds = tuple(sorted(seed for seed, _ in seed_groups))
        if seeds != expected:
            raise SystemExit(
                f"Expected seeds {expected}, got {seeds} for "
                f"{dataset}/{backbone}/{variant}")
        row = {"dataset": dataset, "backbone": backbone, "split": split,
               "variant": variant, "num_seeds": len(seeds),
               "seeds": ",".join(map(str, seeds))}
        for group in GROUPS:
            counts = [int(groups[group].get("count", 0))
                      for _, groups in seed_groups]
            row[f"{group}_count"] = counts[0] if len(set(counts)) == 1 else "|".join(map(str, counts))
            for metric in METRICS:
                values = [float(groups[group][metric]) for _, groups in seed_groups]
                row[f"{group}_{metric}_mean"] = statistics.fmean(values)
                row[f"{group}_{metric}_std"] = statistics.stdev(values)
        rows.append(row)

    output = root / "ablation_summary_mean_std.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
