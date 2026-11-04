#!/usr/bin/env python3
"""Export a base model's full-catalog repeat probability mass.

For every validation example this computes

    m = sum_{i in unique(history)} softmax(base_logits)_i

over the complete catalogue.  The repeat label is defined only from the
training prefix, so validation/test targets never enter the history feature.
The implementation is chunked and numerically stable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refrec.data_utils import (FARRecDataset, collate_eval_batch,
                               compute_item_popularity, load_dataset,
                               load_or_build_cache)
from refrec.run_seqrec import build_model
from refrec.utils import save_json


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    vals, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    del vals
    sums = np.bincount(inv, weights=ranks)
    ranks = sums[inv] / counts[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_summary(labels: np.ndarray, masses: np.ndarray, n_bins: int = 10):
    eps = 1e-12
    rows = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (masses >= lo) & ((masses < hi) if b + 1 < n_bins else (masses <= hi))
        count = int(mask.sum())
        if count:
            mean_m = float(masses[mask].mean())
            rate = float(labels[mask].mean())
            gap = mean_m - rate
            ece += count / len(labels) * abs(gap)
        else:
            mean_m = rate = gap = float("nan")
        rows.append({"bin": b, "lower": lo, "upper": hi, "count": count,
                     "mean_m": mean_m, "repeat_rate": rate,
                     "signed_gap_m_minus_y": gap})
    clipped = np.clip(masses, eps, 1 - eps)
    summary = {
        "users": int(len(labels)),
        "repeat_prevalence": float(labels.mean()),
        "mean_base_repeat_mass": float(masses.mean()),
        "signed_global_gap_m_minus_y": float(masses.mean() - labels.mean()),
        "brier": float(np.mean((masses - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "ece_10_equal_width": float(ece),
        "auc_repeat_intent": _auc(labels, masses),
        "nonempty_underconfident_bins": sum(r["count"] > 0 and r["signed_gap_m_minus_y"] < 0 for r in rows),
        "nonempty_overconfident_bins": sum(r["count"] > 0 and r["signed_gap_m_minus_y"] > 0 for r in rows),
    }
    return summary, rows


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose full-catalog repeat-mass miscalibration")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--split", choices=["valid", "test"], default="valid")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--decode_chunk", type=int, default=8192)
    return p


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("repeat_head") != "base":
        raise ValueError("Repeat-mass diagnosis requires a run trained with --repeat_head base")
    data_dir = Path(args.data_dir or cfg["data_dir"])
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "repeat_mass_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    data = load_dataset(data_dir)
    ns = SimpleNamespace(**cfg)
    model = build_model(ns, data, compute_item_popularity(data)).to(device)
    state = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    cp = {k: int(cfg[k]) for k in ("max_seq_len", "recent_window", "main_topk", "min_main_count")}
    samples = load_or_build_cache(data_dir, data, args.split, **cp)
    loader = DataLoader(FARRecDataset(samples), batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_eval_batch)
    all_items = torch.arange(1, data.num_items + 1, device=device)
    exported = []

    for batch in loader:
        bt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        enc = model.encode(bt["seq_items"], bt["seq_cats"], bt["valid_mask"],
                           bt["weak_mask"], bt["recent_mask"])
        bsz = bt["seq_items"].size(0)
        log_z = torch.full((bsz,), -torch.inf, device=device)
        log_z_hist = torch.full((bsz,), -torch.inf, device=device)
        target_scores = torch.empty((bsz,), device=device)
        greater = torch.zeros((bsz,), dtype=torch.long, device=device)
        equal_lower_id = torch.zeros((bsz,), dtype=torch.long, device=device)
        targets = bt["targets"]

        # First pass obtains target scores and both partition functions.
        for start in range(0, data.num_items, args.decode_chunk):
            ids = all_items[start:start + args.decode_chunk]
            cand = ids.unsqueeze(0).expand(bsz, -1)
            scores = model.compute_base_score(enc, bt["seq_items"], bt["valid_mask"], cand)
            log_z = torch.logaddexp(log_z, torch.logsumexp(scores, dim=1))
            for row, history in enumerate(batch["history_sets"]):
                if history:
                    hmask = torch.isin(ids, torch.as_tensor(sorted(history), device=device))
                    if hmask.any():
                        log_z_hist[row] = torch.logaddexp(log_z_hist[row], torch.logsumexp(scores[row, hmask], dim=0))
            inside = (targets >= ids[0]) & (targets <= ids[-1])
            if inside.any():
                rr = torch.nonzero(inside, as_tuple=False).squeeze(1)
                target_scores[rr] = scores[rr, targets[rr] - ids[0]]

        # Second pass computes exact deterministic rank without storing |I| logits.
        for start in range(0, data.num_items, args.decode_chunk):
            ids = all_items[start:start + args.decode_chunk]
            cand = ids.unsqueeze(0).expand(bsz, -1)
            scores = model.compute_base_score(enc, bt["seq_items"], bt["valid_mask"], cand)
            greater += (scores > target_scores[:, None]).sum(dim=1)
            equal_lower_id += ((scores == target_scores[:, None]) & (ids[None, :] < targets[:, None])).sum(dim=1)

        mass = torch.exp(log_z_hist - log_z)
        mass = torch.where(torch.isfinite(mass), mass, torch.zeros_like(mass))
        for i in range(bsz):
            uid = int(bt["user_id"][i])
            target = int(targets[i])
            history = batch["history_sets"][i]
            exported.append({
                "user_id": data.user_unmap.get(uid, uid),
                "target_item": data.item_unmap.get(target, target),
                "is_repeat_in_prefix": int(target in history),
                "base_repeat_mass_m": float(mass[i]),
                "target_rank": int(1 + greater[i] + equal_lower_id[i]),
                "history_unique_items": len(history),
                "catalog_items": data.num_items,
            })

    with (out_dir / f"{args.split}_repeat_mass.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(exported[0]))
        w.writeheader(); w.writerows(exported)
    labels = np.asarray([r["is_repeat_in_prefix"] for r in exported], dtype=np.float64)
    masses = np.asarray([r["base_repeat_mass_m"] for r in exported], dtype=np.float64)
    summary, bins = calibration_summary(labels, masses)
    summary.update({"split": args.split, "dataset": cfg["dataset"], "backbone": cfg["backbone"],
                    "checkpoint": str((run_dir / "best_model.pt").resolve()),
                    "mass_definition": "sum softmax(base_logits) over unique full-prefix history items",
                    "target_used_in_features": False})
    save_json(out_dir / f"{args.split}_repeat_mass_summary.json", summary)
    with (out_dir / f"{args.split}_reliability_bins.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bins[0])); w.writeheader(); w.writerows(bins)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
