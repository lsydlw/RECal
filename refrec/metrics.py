"""Recall / NDCG / MRR 指标与分组聚合。"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

EVAL_KS = (5, 10, 20)


def recall_at_k(topk_items: list[int], target: int, k: int) -> float:
    return 1.0 if target in topk_items[:k] else 0.0


def ndcg_at_k(topk_items: list[int], target: int, k: int) -> float:
    for i, item in enumerate(topk_items[:k]):
        if item == target:
            return 1.0 / math.log2(i + 2.0)
    return 0.0


def mrr_at_k(topk_items: list[int], target: int, k: int) -> float:
    for i, item in enumerate(topk_items[:k]):
        if item == target:
            return 1.0 / (i + 1)
    return 0.0


def rank_in_list(items: list[int], target: int) -> int:
    for i, item in enumerate(items):
        if item == target:
            return i + 1
    return len(items) + 1


def per_sample_metrics(topk_model: list[int], target_model: int, ks: Iterable[int] = EVAL_KS) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(topk_model, target_model, k)
        out[f"ndcg@{k}"] = ndcg_at_k(topk_model, target_model, k)
    out["mrr@10"] = mrr_at_k(topk_model, target_model, 10)
    return out


def aggregate_metrics(rows: list[dict[str, float]], ks: Iterable[int] = EVAL_KS) -> dict[str, float]:
    if not rows:
        return _empty_metrics(ks)
    sums: dict[str, float] = defaultdict(float)
    for row in rows:
        for key, val in row.items():
            sums[key] += val
    n = len(rows)
    return {k: v / n for k, v in sums.items()}


def aggregate_from_ranks(ranks: list[int], ks: Iterable[int] = EVAL_KS) -> dict[str, float]:
    if not ranks:
        return _empty_metrics(ks)
    rows = []
    for rank in ranks:
        row: dict[str, float] = {}
        for k in ks:
            hit = 1.0 if rank <= k else 0.0
            row[f"recall@{k}"] = hit
            row[f"ndcg@{k}"] = (1.0 / math.log2(rank + 1.0)) if rank <= k else 0.0
        row["mrr@10"] = (1.0 / rank) if rank <= 10 else 0.0
        rows.append(row)
    return aggregate_metrics(rows, ks)


def aggregate_group_metrics(
    samples: list[dict[str, Any]],
    group_key: str,
    ks: Iterable[int] = EVAL_KS,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for sample in samples:
        buckets[sample[group_key]].append(sample["metrics"])
    out: list[dict[str, Any]] = []
    for group in sorted(buckets.keys()):
        agg = aggregate_metrics(buckets[group], ks)
        rec: dict[str, Any] = {"group": group, "count": len(buckets[group])}
        rec.update(agg)
        out.append(rec)
    return out


def save_group_metrics_csv(path: Path, rows: list[dict[str, Any]], ks: Iterable[int] = EVAL_KS) -> None:
    ks = list(ks)
    cols = ["group", "count"]
    for k in ks:
        cols.extend([f"recall@{k}", f"ndcg@{k}"])
    cols.append("mrr@10")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _empty_metrics(ks: Iterable[int]) -> dict[str, float]:
    ks = list(ks)
    out = {f"{m}@{k}": 0.0 for k in ks for m in ("recall", "ndcg")}
    out["mrr@10"] = 0.0
    return out
