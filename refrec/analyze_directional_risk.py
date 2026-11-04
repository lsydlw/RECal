#!/usr/bin/env python3
"""Analyse directional calibration risk from per-user RECal ablation ranks.

The input files are produced by ``run_recal_ablations.py`` and contain only
validation predictions.  Samples are separated by the sign of q - m, where q
is estimated repeat intent and m is Base history probability mass.  The script
reports accuracy together with the fraction of users whose target rank is
improved or harmed by each calibration rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


VARIANTS = ("base", "full_projection", "one_sided_risk", "bidirectional_risk")


def parse_args():
    p = argparse.ArgumentParser(description="Directional RECal risk analysis")
    p.add_argument("--input_root", required=True,
                   help="Root containing valid_ablation_ranks.csv files")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pattern", default="valid_ablation_ranks.csv")
    p.add_argument("--k", type=int, default=10)
    return p.parse_args()


def ndcg(rank: int, k: int) -> float:
    return 1.0 / math.log2(rank + 1.0) if rank <= k else 0.0


def infer_run(path: Path, root: Path):
    rel = path.relative_to(root)
    run = path.parent.name
    dataset = "unknown"
    backbone = "unknown"
    seed = "unknown"
    metrics = path.parent / "metrics.json"
    if metrics.exists():
        obj = json.loads(metrics.read_text(encoding="utf-8"))
        dataset = str(obj.get("dataset", dataset))
        backbone = str(obj.get("backbone", backbone))
    else:
        parts = [p.lower() for p in rel.parts]
        for candidate in ("gru4rec", "sasrec", "bert4rec", "repeatnet"):
            if candidate in run.lower() or candidate in parts:
                backbone = candidate
                break
        if len(rel.parts) >= 2:
            dataset = rel.parts[-3] if len(rel.parts) >= 3 else rel.parts[0]
    for token in run.replace("-", "_").split("_"):
        if token.startswith("seed") and token[4:].isdigit():
            seed = token[4:]
    return dataset, backbone, seed, str(rel)


def direction(row):
    q, m = float(row["q"]), float(row["m"])
    if q > m:
        return "up_q_gt_m"
    if q < m:
        return "down_q_lt_m"
    return "matched"


def summarize(rows, k):
    out = {
        "count": len(rows),
        "repeat_rate": sum(r["group"] == "repeat" for r in rows) / len(rows),
        "mean_m": sum(float(r["m"]) for r in rows) / len(rows),
        "mean_q": sum(float(r["q"]) for r in rows) / len(rows),
        "mean_abs_gap": sum(abs(float(r["q"]) - float(r["m"])) for r in rows) / len(rows),
    }
    base_ranks = [int(r["rank_base"]) for r in rows]
    base_ndcg = [ndcg(x, k) for x in base_ranks]
    for v in VARIANTS:
        ranks = [int(r[f"rank_{v}"]) for r in rows]
        values = [ndcg(x, k) for x in ranks]
        out[f"ndcg@{k}_{v}"] = sum(values) / len(values)
        out[f"recall@{k}_{v}"] = sum(x <= k for x in ranks) / len(ranks)
        if v != "base":
            out[f"ndcg@{k}_delta_{v}"] = (
                out[f"ndcg@{k}_{v}"] - out[f"ndcg@{k}_base"])
            diffs = [x - b for x, b in zip(values, base_ndcg)]
            mean = sum(diffs) / len(diffs)
            variance = (sum((x - mean) ** 2 for x in diffs) /
                        max(1, len(diffs) - 1))
            half = 1.96 * math.sqrt(variance / len(diffs))
            out[f"ndcg@{k}_delta_ci95_low_{v}"] = mean - half
            out[f"ndcg@{k}_delta_ci95_high_{v}"] = mean + half
            out[f"rank_improved_{v}"] = sum(x < b for x, b in zip(ranks, base_ranks)) / len(ranks)
            out[f"rank_harmed_{v}"] = sum(x > b for x, b in zip(ranks, base_ranks)) / len(ranks)
            out[f"rank_unchanged_{v}"] = sum(x == b for x, b in zip(ranks, base_ranks)) / len(ranks)
            out[f"mean_rank_delta_{v}"] = sum(x - b for x, b in zip(ranks, base_ranks)) / len(ranks)
    one = [ndcg(int(r["rank_one_sided_risk"]), k) for r in rows]
    bi = [ndcg(int(r["rank_bidirectional_risk"]), k) for r in rows]
    diffs = [x - y for x, y in zip(one, bi)]
    mean = sum(diffs) / len(diffs)
    variance = sum((x - mean) ** 2 for x in diffs) / max(1, len(diffs) - 1)
    half = 1.96 * math.sqrt(variance / len(diffs))
    out[f"ndcg@{k}_one_minus_bi"] = mean
    out[f"ndcg@{k}_one_minus_bi_ci95_low"] = mean - half
    out[f"ndcg@{k}_one_minus_bi_ci95_high"] = mean + half
    return out


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.input_root).resolve()
    out = Path(args.output_dir).resolve()
    files = sorted(root.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No {args.pattern} under {root}")

    by_run = []
    pooled = defaultdict(list)
    manifest = []
    for path in files:
        dataset, backbone, seed, rel = infer_run(path, root)
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        required = {"group", "m", "q", *(f"rank_{v}" for v in VARIANTS)}
        missing = required - set(rows[0]) if rows else required
        if missing:
            raise ValueError(f"{path} misses columns: {sorted(missing)}")
        manifest.append({"dataset": dataset, "backbone": backbone,
                         "seed": seed, "file": rel, "users": len(rows)})
        for name in ("all", "up_q_gt_m", "down_q_lt_m", "matched"):
            directional = rows if name == "all" else [r for r in rows if direction(r) == name]
            for target_group in ("all", "repeat", "explore"):
                chosen = (directional if target_group == "all" else
                          [r for r in directional if r["group"] == target_group])
                if not chosen:
                    continue
                rec = {"dataset": dataset, "backbone": backbone, "seed": seed,
                       "direction": name, "target_group": target_group,
                       **summarize(chosen, args.k)}
                by_run.append(rec)
                pooled[(dataset, backbone, name, target_group)].extend(chosen)

    aggregate = []
    for (dataset, backbone, name, target_group), rows in sorted(pooled.items()):
        aggregate.append({"dataset": dataset, "backbone": backbone,
                          "direction": name, "target_group": target_group,
                          **summarize(rows, args.k)})
    write_csv(out / "directional_risk_by_run.csv", by_run)
    write_csv(out / "directional_risk_aggregate.csv", aggregate)
    (out / "manifest.json").write_text(
        json.dumps({"split": "valid", "k": args.k, "files": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"analysed {len(files)} rank files; outputs -> {out}")


if __name__ == "__main__":
    main()
