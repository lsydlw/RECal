#!/usr/bin/env python3
"""Summarize the immutable one-shot test outputs in a frozen manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("recall@5", "ndcg@5", "recall@10", "ndcg@10",
           "recall@20", "ndcg@20", "mrr@10")


def mean(values):
    return statistics.fmean(values)


def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_frozen_test.py MANIFEST OUTPUT_DIR")
    manifest_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest["runs"]
    if len(runs) != 48:
        raise RuntimeError(f"expected 48 frozen runs, found {len(runs)}")

    flat = []
    grouped = defaultdict(list)
    for frozen in runs:
        base_dir = Path(frozen["run_dir"])
        recal_dir = Path(frozen["recal_run_dir"])
        base_result = json.loads((base_dir / "test_metrics.json").read_text(encoding="utf-8"))
        recal_result = json.loads((recal_dir / "test_metrics.json").read_text(encoding="utf-8"))
        if not base_result.get("test_evaluated_once") or not recal_result.get("test_evaluated_once"):
            raise RuntimeError(f"non-final test artifact: {base_dir} or {recal_dir}")
        base_metrics = base_result["test_metrics"]
        recal_base = recal_result["base"]
        for metric in METRICS:
            if abs(float(base_metrics[metric]) - float(recal_base[metric])) > 1e-12:
                raise RuntimeError(f"Base mismatch for {base_dir}: {metric}")
        key = (frozen["dataset"], frozen["backbone"])
        record = {
            "dataset": key[0], "backbone": key[1], "seed": int(frozen["seed"]),
            "base_dir": str(base_dir), "recal_dir": str(recal_dir),
        }
        for metric in METRICS:
            record[f"base_{metric}"] = float(base_metrics[metric])
            record[f"recal_{metric}"] = float(recal_result["recal"][metric])
        for group in ("repeat", "explore"):
            group_result = recal_result["groups"][group]
            record[f"{group}_count"] = int(group_result["count"])
            for metric in METRICS:
                record[f"{group}_base_{metric}"] = float(group_result["base"][metric])
                record[f"{group}_recal_{metric}"] = float(group_result["recal"][metric])
        flat.append(record)
        grouped[key].append(record)

    summary = []
    group_summary = []
    for (dataset, backbone), records in sorted(grouped.items()):
        seeds = sorted(r["seed"] for r in records)
        if seeds != [2023, 2024, 2025]:
            raise RuntimeError(f"unexpected seeds for {dataset}/{backbone}: {seeds}")
        row = {"dataset": dataset, "backbone": backbone, "seeds": "2023|2024|2025"}
        for metric in METRICS:
            base_values = [r[f"base_{metric}"] for r in records]
            recal_values = [r[f"recal_{metric}"] for r in records]
            base_mean, recal_mean = mean(base_values), mean(recal_values)
            row[f"base_{metric}_mean"] = base_mean
            row[f"base_{metric}_std"] = std(base_values)
            row[f"recal_{metric}_mean"] = recal_mean
            row[f"recal_{metric}_std"] = std(recal_values)
            row[f"delta_{metric}"] = recal_mean - base_mean
            row[f"relative_{metric}_pct"] = ((recal_mean / base_mean - 1.0) * 100.0
                                                if base_mean != 0.0 else "")
        summary.append(row)
        for group in ("repeat", "explore"):
            grow = {"dataset": dataset, "backbone": backbone, "group": group,
                    "count_per_seed": "|".join(str(r[f"{group}_count"]) for r in records)}
            for metric in METRICS:
                base_values = [r[f"{group}_base_{metric}"] for r in records]
                recal_values = [r[f"{group}_recal_{metric}"] for r in records]
                grow[f"base_{metric}_mean"] = mean(base_values)
                grow[f"base_{metric}_std"] = std(base_values)
                grow[f"recal_{metric}_mean"] = mean(recal_values)
                grow[f"recal_{metric}_std"] = std(recal_values)
            group_summary.append(grow)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("test_runs.csv", flat), ("test_summary.csv", summary),
                       ("test_group_summary.csv", group_summary)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "run_count": len(flat), "group_count": len(summary),
        "seeds": [2023, 2024, 2025], "test_reexecuted": False,
    }
    (output_dir / "test_summary_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
