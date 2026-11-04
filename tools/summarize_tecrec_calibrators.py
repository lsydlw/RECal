#!/usr/bin/env python3
"""Aggregate the RQ3 TS/GLS/CalRec/TecRec-style runs over random seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in sorted(Path(args.root).glob("*/*/test_metrics.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        dataset = obj["dataset"]
        for method, payload in obj["methods"].items():
            for metric, value in payload["overall"].items():
                grouped[(dataset, method, metric)].append(float(value))

    rows = []
    for (dataset, method, metric), values in sorted(grouped.items()):
        rows.append({
            "dataset": dataset,
            "method": method,
            "metric": metric,
            "num_seeds": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
