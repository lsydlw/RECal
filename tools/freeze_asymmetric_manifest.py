#!/usr/bin/env python3
"""Freeze all validation-selected asymmetric RECal runs before test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.results_root)
    recal_root = root / "recal_asymmetric_8datasets_v1" / "recal_runs"
    runs = []
    for metrics_path in sorted(recal_root.glob("*/*/metrics.json")):
        obj = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parent
        dataset = obj["dataset"]
        backbone = obj["backbone"]
        seed = int(obj["seed"])
        family = ("core_multiseed_v1" if dataset in {
            "ml-100k", "datafoundation-user", "LastFM", "Bookmarks"
        } else "external_multiseed_v1")
        base = root / family / "base_runs" / dataset / f"{dataset}_{backbone}_base_seed{seed}"
        head = run_dir / "intent_head.pt"
        if not (base / "best_model.pt").is_file() or not head.is_file():
            raise FileNotFoundError(f"incomplete frozen run: {run_dir}")
        runs.append({
            "dataset": dataset,
            "backbone": backbone,
            "seed": seed,
            "base_run_dir": str(base),
            "recal_run_dir": str(run_dir),
            "mode": "asymmetric",
            "down_power": 12.0,
            "intent_head_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
        })
    if len(runs) != 120:
        raise RuntimeError(f"expected 120 frozen runs, found {len(runs)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "selection_split": "valid",
        "selection_metric": "ndcg@10",
        "test_not_used_for_selection": True,
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
