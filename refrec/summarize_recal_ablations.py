#!/usr/bin/env python3
"""Summarize core RECal ablation metrics into tidy CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("*/*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        ranks_path = path.with_name(f"{result.get('split', 'valid')}_ablation_ranks.csv")
        group_counts = {}
        if ranks_path.exists():
            with ranks_path.open(newline="", encoding="utf-8") as f:
                rank_rows = list(csv.DictReader(f))
            group_counts = {
                "overall": len(rank_rows),
                "repeat": sum(r["group"] == "repeat" for r in rank_rows),
                "explore": sum(r["group"] == "explore" for r in rank_rows),
            }
        for variant, groups in result["variants"].items():
            row = {"dataset": result["dataset"], "backbone": result["backbone"],
                   "seed": result.get("seed", ""),
                   "split": result.get("split", "valid"),
                   "variant": variant}
            for group in ("overall", "repeat", "explore"):
                row[f"{group}_count"] = groups[group].get(
                    "count", group_counts.get(group, ""))
                for metric in ("recall@10", "ndcg@10", "mrr@10"):
                    row[f"{group}_{metric}"] = groups[group][metric]
            rows.append(row)
    if not rows:
        raise SystemExit(f"No metrics.json files under {root}")
    output = root / "ablation_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
