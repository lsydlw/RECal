#!/usr/bin/env python3
"""Write a schema-agnostic inventory of resumable RQ3 outputs.

The three baseline runners have intentionally different JSON schemas because
some methods train a listwise adapter and others are parameter-free.  This
small utility records every completed test artifact and is useful for checking
coverage before paper-table aggregation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--established", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--recent", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def rows(root: Path, family: str):
    for path in sorted(root.rglob("test_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield {
            "family": family,
            "path": str(path.relative_to(root)),
            "dataset": payload.get("dataset", "unknown"),
            "backbone": payload.get("backbone", payload.get("base_backbone", "unknown")),
            "seed": payload.get("seed", "unknown"),
        }


def main() -> None:
    args = parse_args()
    table = []
    table.extend(rows(Path(args.established), "established"))
    table.extend(rows(Path(args.controls), "controls"))
    table.extend(rows(Path(args.recent), "recent"))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "dataset", "backbone", "seed", "path"])
        writer.writeheader(); writer.writerows(table)
    print(f"wrote {len(table)} completed RQ3 artifacts to {output}")


if __name__ == "__main__":
    main()
