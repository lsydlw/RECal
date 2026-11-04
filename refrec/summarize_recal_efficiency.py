#!/usr/bin/env python3
"""Collect RECal efficiency JSON files into a paper-ready CSV."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True)
    args = parser.parse_args(); root = Path(args.root); rows = []
    for path in sorted((root / "runs").glob("*/*/efficiency.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "dataset": obj["dataset"], "backbone": obj["backbone"],
            "seed": obj["seed"], "catalogue_items": obj["catalogue_items"],
            "users": obj["base"]["users"], "batch_size": obj["batch_size"],
            "base_params": obj["base_parameters"],
            "recal_extra_params": obj["recal_extra_parameters"],
            "param_overhead_pct": obj["parameter_overhead_percent"],
            "base_ms_per_user": obj["base"]["milliseconds_per_user"],
            "recal_ms_per_user": obj["recal"]["milliseconds_per_user"],
            "latency_overhead_pct": obj["latency_overhead_percent"],
            "base_users_per_s": obj["base"]["users_per_second"],
            "recal_users_per_s": obj["recal"]["users_per_second"],
            "base_peak_mib": obj["base"]["peak_allocated_mib"],
            "recal_peak_mib": obj["recal"]["peak_allocated_mib"],
            "peak_memory_overhead_mib": obj["peak_memory_overhead_mib"],
        })
    if not rows: raise SystemExit(f"No efficiency results under {root / 'runs'}")
    with (root / "efficiency_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {root / 'efficiency_summary.csv'}")


if __name__ == "__main__": main()
