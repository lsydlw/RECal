#!/usr/bin/env python3
"""Diagnose repeat-intent versus Base history-mass mismatch from saved ranks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from refrec.metrics import aggregate_from_ranks
from refrec.utils import save_json


BINS = (
    ("strong_over", -math.inf, -0.30),
    ("moderate_over", -0.30, -0.10),
    ("matched", -0.10, 0.10),
    ("moderate_under", 0.10, 0.30),
    ("strong_under", 0.30, math.inf),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_roots", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--backbones", nargs="*", default=None,
                   help="Optional backbone filter, e.g. gru4rec sasrec.")
    return p.parse_args()


def metrics(rows, key):
    if not rows:
        return {k: 0.0 for k in ("recall@5", "ndcg@5", "recall@10",
                                  "ndcg@10", "recall@20", "ndcg@20", "mrr@10")}
    return aggregate_from_ranks([int(r[key]) for r in rows])


def projected_mass(m, delta):
    m = min(max(m, 1e-12), 1 - 1e-12)
    z = math.log(m / (1 - m)) + delta
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1 + ez)


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(rows):
    y = np.asarray([r["is_repeat"] for r in rows], dtype=float)
    m = np.asarray([r["m"] for r in rows], dtype=float)
    q = np.asarray([r["q"] for r in rows], dtype=float)
    target = np.asarray([r["target_mass"] for r in rows], dtype=float)
    gap = q - m
    gain = np.asarray([r["base_rank"] - r["recal_rank"] for r in rows], dtype=float)
    out = {
        "count": len(rows), "repeat_rate": float(y.mean()),
        "mean_m": float(m.mean()), "mean_q": float(q.mean()),
        "mean_gap_q_minus_m": float(gap.mean()),
        "mean_abs_gap": float(np.abs(gap).mean()),
        "base_mass_mae": float(np.abs(m - y).mean()),
        "intent_q_mae": float(np.abs(q - y).mean()),
        "recal_target_mass_mae": float(np.abs(target - y).mean()),
        "base_mass_brier": float(np.mean((m - y) ** 2)),
        "intent_q_brier": float(np.mean((q - y) ** 2)),
        "recal_target_mass_brier": float(np.mean((target - y) ** 2)),
        "base": metrics(rows, "base_rank"),
        "recal": metrics(rows, "recal_rank"),
        "mean_rank_gain": float(gain.mean()),
        "median_rank_gain": float(np.median(gain)),
        "improved_fraction": float(np.mean(gain > 0)),
        "worsened_fraction": float(np.mean(gain < 0)),
        "unchanged_fraction": float(np.mean(gain == 0)),
        "corr_abs_gap_rank_gain": pearson(np.abs(gap), gain),
        "corr_signed_gap_rank_gain": pearson(gap, gain),
    }
    return out


def main():
    args = parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    detail_rows, summaries = [], []
    files = []
    for root in map(Path, args.input_roots):
        files.extend(root.glob("*/*/valid_predictions.csv"))
    for path in sorted(set(files)):
        metrics_path = path.parent / "metrics.json"
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        dataset, backbone = meta["dataset"], meta["backbone"]
        if args.backbones and backbone not in set(args.backbones):
            continue
        mode = meta.get("calibration_mode", "unknown")
        down_power = meta.get("down_power")
        up_power = meta.get("up_power")
        with path.open(encoding="utf-8") as f:
            rows = []
            for raw in csv.DictReader(f):
                row = {"is_repeat": int(raw["is_repeat"]),
                       "m": float(raw["m"]), "q": float(raw["q"]),
                       "delta": float(raw["delta"]),
                       "base_rank": int(raw["base_rank"]),
                       "recal_rank": int(raw["recal_rank"])}
                row["target_mass"] = projected_mass(row["m"], row["delta"])
                row["gap"] = row["q"] - row["m"]
                rows.append(row)
        overall = summarize(rows)
        summaries.append({"dataset": dataset, "backbone": backbone,
                          "calibration_mode": mode, "down_power": down_power,
                          "up_power": up_power, **overall})
        for name, lo, hi in BINS:
            selected = [r for r in rows if lo <= r["gap"] < hi]
            s = summarize(selected) if selected else {"count": 0}
            flat = {"dataset": dataset, "backbone": backbone,
                    "calibration_mode": mode, "down_power": down_power,
                    "up_power": up_power, "gap_bin": name,
                    "gap_lower": lo, "gap_upper": hi}
            for key, value in s.items():
                if key in {"base", "recal"}:
                    for metric, score in value.items():
                        flat[f"{key}_{metric}"] = score
                else:
                    flat[key] = value
            detail_rows.append(flat)
    if not detail_rows:
        raise SystemExit("No valid_predictions.csv files found")
    fields = []
    for row in detail_rows:
        for key in row:
            if key not in fields: fields.append(key)
    with (out / "mismatch_bins.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(detail_rows)
    save_json(out / "mismatch_summary.json", {"runs": summaries, "bins": list(BINS)})
    print(f"runs={len(summaries)} bins={len(detail_rows)} output={out.resolve()}")


if __name__ == "__main__":
    main()
