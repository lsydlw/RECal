#!/usr/bin/env python3
"""Validate a new RECal gate while reusing a frozen intent head.

This avoids retraining either the recommendation backbone or the target-
independent repeat-intent predictor.  The output is a self-contained RECal run
directory that can be passed to ``evaluate_recal_test``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from refrec.data_utils import compute_item_popularity, load_dataset, load_or_build_cache
from refrec.diagnose_repeat_mass import calibration_summary
from refrec.recal import RepeatIntentHead
from refrec.run_recal_frozen import evaluate_recal, rank_metrics
from refrec.run_seqrec import build_model
from refrec.utils import save_json


def parse_args():
    p = argparse.ArgumentParser(description="Reuse a frozen head with an asymmetric gate")
    p.add_argument("--base_run_dir", required=True)
    p.add_argument("--source_recal_run_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--decode_chunk", type=int, default=8192)
    p.add_argument("--down_power", type=float, default=12.0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    base = Path(args.base_run_dir).resolve()
    source = Path(args.source_recal_run_dir).resolve()
    out = Path(args.output_dir).resolve()
    if (out / "test_metrics.json").exists():
        raise FileExistsError(f"test output already exists: {out / 'test_metrics.json'}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data = load_dataset(Path(cfg["data_dir"]))
    model = build_model(SimpleNamespace(**cfg), data, compute_item_popularity(data)).to(device)
    model.load_state_dict(torch.load(base / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    head = RepeatIntentHead(cfg["emb_dim"], hidden_dim=args.hidden_dim, dropout=0.1).to(device)
    head.load_state_dict(torch.load(source / "intent_head.pt", map_location=device, weights_only=True))
    head.eval()
    cp = {k: int(cfg[k]) for k in ("max_seq_len", "recent_window", "main_topk", "min_main_count")}
    valid = load_or_build_cache(Path(cfg["data_dir"]), data, "valid", **cp)
    ea = SimpleNamespace(eval_batch_size=args.eval_batch_size,
                         decode_chunk=args.decode_chunk,
                         calibration_mode="asymmetric",
                         down_power=args.down_power,
                         up_power=0.0)
    rows = evaluate_recal(model, head, valid, data, ea, device, cfg["recent_window"])
    with (out / "valid_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    labels = np.asarray([r["is_repeat"] for r in rows], dtype=float)
    qvals = np.asarray([r["q"] for r in rows], dtype=float)
    q_summary, q_bins = calibration_summary(labels, qvals)
    groups = {}
    for name, flag in (("repeat", 1), ("explore", 0)):
        subset = [r for r in rows if r["is_repeat"] == flag]
        groups[name] = {"count": len(subset),
                        "base": rank_metrics(subset, "base_rank"),
                        "recal": rank_metrics(subset, "recal_rank")}
    src_metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    result = {
        "dataset": cfg["dataset"], "backbone": cfg["backbone"], "seed": cfg["seed"],
        "negative_sampling": cfg["negative_sampling"],
        "best_intent_epoch": src_metrics.get("best_intent_epoch"),
        "base": rank_metrics(rows, "base_rank"), "recal": rank_metrics(rows, "recal_rank"),
        "groups": groups, "intent_calibration": q_summary,
        "mean_m": float(np.mean([r["m"] for r in rows])), "mean_q": float(qvals.mean()),
        "mean_delta": float(np.mean([r["delta"] for r in rows])),
        "calibration_mode": "asymmetric", "down_power": args.down_power, "up_power": 0.0,
        "source_recal_run_dir": str(source), "intent_head_reused": True,
        "target_used_in_features": False, "test_evaluated": False,
    }
    save_json(out / "metrics.json", result)
    torch.save(head.state_dict(), out / "intent_head.pt")
    with (out / "intent_reliability_bins.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(q_bins[0])); writer.writeheader(); writer.writerows(q_bins)
    history = source / "training_history.json"
    if history.exists():
        shutil.copy2(history, out / "training_history.json")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
