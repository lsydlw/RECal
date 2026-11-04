#!/usr/bin/env python3
"""Summarize Base and RECal test metrics over random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("recall@5", "ndcg@5", "recall@10", "ndcg@10",
           "recall@20", "ndcg@20", "mrr@10")


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.root)
    grouped = defaultdict(list)
    for path in sorted(root.glob("*/*/test_metrics.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        grouped[(obj["dataset"], obj["backbone"])].append(obj)
    if not grouped:
        raise SystemExit(f"no test_metrics.json under {root}")
    rows = []
    for (dataset, backbone), runs in sorted(grouped.items()):
        row = {"dataset": dataset, "backbone": backbone,
               "seeds": ",".join(str(x["seed"]) for x in sorted(runs, key=lambda x: x["seed"])),
               "num_seeds": len(runs)}
        for metric in METRICS:
            bm, bs = mean_std([x["base"][metric] for x in runs])
            rm, rs = mean_std([x["recal"][metric] for x in runs])
            row[f"base_{metric}_mean"] = bm
            row[f"base_{metric}_std"] = bs
            row[f"recal_{metric}_mean"] = rm
            row[f"recal_{metric}_std"] = rs
            row[f"relative_{metric}_pct"] = (rm / bm - 1.0) * 100.0 if bm else ""
        for group in ("repeat", "explore"):
            row[f"{group}_count"] = runs[0]["groups"][group]["count"]
            for metric in ("recall@10", "ndcg@10", "mrr@10"):
                bm, bs = mean_std([x["groups"][group]["base"][metric] for x in runs])
                rm, rs = mean_std([x["groups"][group]["recal"][metric] for x in runs])
                row[f"{group}_base_{metric}_mean"] = bm
                row[f"{group}_base_{metric}_std"] = bs
                row[f"{group}_recal_{metric}_mean"] = rm
                row[f"{group}_recal_{metric}_std"] = rs
        rows.append(row)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"summarized {len(rows)} dataset-backbone pairs -> {out}")


if __name__ == "__main__":
    main()
