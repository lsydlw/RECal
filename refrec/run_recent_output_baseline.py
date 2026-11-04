#!/usr/bin/env python3
"""Output-only recent-baseline runner for RECal RQ3.

The runner never updates the frozen sequential backbone.  It implements three
candidate-list methods under the paper's common full-catalogue protocol:

* ``topncal``: rank-binned Top-N calibration (Sato, RecSys 2024);
* ``smmr``: sampled MMR reranking (Liakhnovich et al., SIGIR 2025);
* ``duor``: dynamic user-oriented popularity calibration (Gulsoy et al.,
  IJHCS 2025), reimplemented from the published stepwise procedure.

All method choices are made on validation data only.  A completed test output
is write-once, which lets the shell controller resume safely after interruption.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refrec.data_utils import collate_eval_batch, compute_item_popularity, load_dataset, load_or_build_cache
from refrec.metrics import aggregate_from_ranks
from refrec.protocol import PROTOCOL, merge_topk
from refrec.run_seqrec import build_model
from refrec.utils import save_json


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--method", choices=("topncal", "smmr", "duor"), required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--decode_chunk", type=int, default=4096)
    p.add_argument("--pool_size", type=int, default=100)
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args()


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def collect(base, samples, data, args, device):
    """Materialize the shared Base Top-100 lists once for a split."""
    base.eval(); all_items = torch.arange(1, data.num_items + 1, device=device)
    rows = []
    for start in range(0, len(samples), args.eval_batch_size):
        sb = samples[start:start + args.eval_batch_size]
        raw = collate_eval_batch(sb); bt = to_device(raw, device); bsz = len(sb)
        enc = base.encode(bt["seq_items"], bt["seq_cats"], bt["valid_mask"], bt["weak_mask"], bt["recent_mask"])
        enc = base.prepare_full_catalog(enc, bt["seq_items"], bt["valid_mask"], all_items, chunk_size=args.decode_chunk)
        top_scores = top_items = None
        for off in range(0, data.num_items, args.decode_chunk):
            ids = all_items[off:off + args.decode_chunk]
            scores = base.compute_base_score(enc, bt["seq_items"], bt["valid_mask"], ids[None, :].expand(bsz, -1))
            top_scores, top_items = merge_topk(top_scores, top_items, scores, ids[None, :].expand(bsz, -1), args.pool_size)
        embs = base.item_repr(top_items).detach().cpu().numpy().astype(np.float32)
        for r, sample in enumerate(sb):
            rows.append({
                "user_id": int(bt["user_id"][r]), "target": int(bt["targets"][r]),
                "history": sorted(map(int, sample["history_set"])),
                "group": "repeat" if int(bt["targets"][r]) in sample["history_set"] else "explore",
                "items": top_items[r].detach().cpu().numpy().astype(np.int64),
                "scores": top_scores[r].detach().cpu().numpy().astype(np.float64),
                "embeddings": embs[r],
            })
        print(f"collected {min(start + bsz, len(samples))}/{len(samples)}", flush=True)
    return rows


def rank20(order, target, topk=20):
    found = np.flatnonzero(np.asarray(order[:topk]) == int(target))
    return int(found[0] + 1) if len(found) else topk + 1


def ndcg(rows, orders, topk=20):
    ranks = [rank20(o, r["target"], topk) for r, o in zip(rows, orders)]
    return aggregate_from_ranks(ranks)["ndcg@10"]


def fit_topncal(rows, n_bins):
    """Smoothed rank-binned likelihood calibration fitted on validation only."""
    hit = np.zeros(n_bins, dtype=np.float64); total = np.zeros(n_bins, dtype=np.float64)
    for row in rows:
        target_pos = np.flatnonzero(row["items"] == row["target"])
        pos = int(target_pos[0]) if len(target_pos) else -1
        for j in range(len(row["items"])):
            b = min(n_bins - 1, (j * n_bins) // len(row["items"]))
            total[b] += 1.0
            if j == pos: hit[b] += 1.0
    # Laplace smoothing avoids a -infinity offset for rare top-N hits.
    prior = (hit + 1.0) / (total + 2.0)
    offset = np.log(prior / (1.0 - prior))
    return offset


def apply_topncal(row, offset):
    n = len(row["items"]); b = np.minimum(len(offset) - 1, np.arange(n) * len(offset) // n)
    # Scores are standardized per request before rank-bin offsets are applied.
    s = row["scores"]
    norm = (s - s.mean()) / max(s.std(), 1e-6)
    return row["items"][np.lexsort((row["items"], -(norm + offset[b])))]


def cosine_matrix(emb):
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    x = emb / np.maximum(norm, 1e-12)
    return x @ x.T


def sampled_mmr(row, lam, temperature, rng, topk=20):
    """Sampling-based MMR on the Base pool; mirrors the SMMR inference interface."""
    items, scores = row["items"], row["scores"]
    sim = cosine_matrix(row["embeddings"])
    rel = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
    remaining = list(range(len(items))); chosen = []
    for _ in range(min(topk, len(remaining))):
        div = np.zeros(len(remaining)) if not chosen else np.max(sim[np.ix_(remaining, chosen)], axis=1)
        utility = lam * rel[remaining] - (1.0 - lam) * div
        # Official SMMR samples instead of greedy selection.  The fixed RNG
        # makes each seed exactly reproducible under the shared protocol.
        prob = np.exp((utility - utility.max()) / max(temperature, 1e-4)); prob /= prob.sum()
        local = int(rng.choice(len(remaining), p=prob)); chosen.append(remaining.pop(local))
    return np.concatenate((items[chosen], np.asarray([i for i in items if i not in set(items[chosen])], dtype=np.int64)))


def popularity_flags(data):
    pop = compute_item_popularity(data)
    # Project versions return either a 1-indexed Tensor or a mapping.
    if isinstance(pop, torch.Tensor):
        value = pop.detach().cpu().numpy().astype(np.float64)[1:data.num_items + 1]
    else:
        value = np.asarray([float(pop.get(i, 0.0)) for i in range(1, data.num_items + 1)])
    threshold = float(np.quantile(value, 0.8))
    return {i: bool(value[i - 1] >= threshold) for i in range(1, data.num_items + 1)}


def duor_rerank(row, lam, popular, topk=20):
    """Published DUoR-style stepwise popularity-profile calibration."""
    items, scores = row["items"], row["scores"]
    hist = row["history"]
    # A smoothed individual desired fraction of popular items.
    desired = (sum(popular.get(i, False) for i in hist) + 1.0) / (len(hist) + 2.0)
    rel = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
    remaining = list(range(len(items))); selected = []; chosen_pop = 0
    for step in range(min(topk, len(remaining))):
        values = []
        for idx in remaining:
            after = (chosen_pop + int(popular.get(int(items[idx]), False))) / (step + 1.0)
            values.append((1.0 - lam) * rel[idx] - lam * abs(after - desired))
        utilities = np.asarray(values)
        tie_items = np.asarray([items[i] for i in remaining])
        local = int(np.lexsort((tie_items, -utilities))[0]); picked = remaining.pop(local)
        selected.append(picked); chosen_pop += int(popular.get(int(items[picked]), False))
    selected_items = items[selected]
    tail = np.asarray([i for i in items if i not in set(selected_items)], dtype=np.int64)
    return np.concatenate((selected_items, tail))


def orders_for(rows, method, config, data, seed):
    if method == "topncal": return [apply_topncal(x, config["offset"]) for x in rows]
    if method == "smmr":
        return [sampled_mmr(x, config["lambda"], config["temperature"], np.random.default_rng(seed + i)) for i, x in enumerate(rows)]
    popular = config["popular"]
    return [duor_rerank(x, config["lambda"], popular) for x in rows]


def choose(rows, method, data, seed):
    candidates = []
    if method == "topncal":
        for b in (5, 10, 20): candidates.append({"bins": b, "offset": fit_topncal(rows, b)})
    elif method == "smmr":
        for l in (0.80, 0.90, 0.95, 0.99):
            for t in (0.01, 0.05): candidates.append({"lambda": l, "temperature": t})
    else:
        popular = popularity_flags(data)
        for l in (0.05, 0.10, 0.20, 0.35, 0.50): candidates.append({"lambda": l, "popular": popular})
    values = [ndcg(rows, orders_for(rows, method, c, data, seed), 20) for c in candidates]
    best = int(np.argmax(values)); return candidates[best], values, candidates


def metrics(rows, orders, topk=20):
    ranks = [rank20(o, x["target"], topk) for x, o in zip(rows, orders)]
    out = {"overall": aggregate_from_ranks(ranks)}
    for group in ("repeat", "explore"):
        ids = [i for i, x in enumerate(rows) if x["group"] == group]
        out[group] = aggregate_from_ranks([ranks[i] for i in ids]) | {"count": len(ids)}
    return out, ranks


def main():
    args = args_parser(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    out, base_dir = Path(args.output_dir), Path(args.base_run_dir)
    if (out / "test_metrics.json").exists(): raise FileExistsError(f"write-once output exists: {out}")
    out.mkdir(parents=True, exist_ok=True); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8")); data = load_dataset(Path(cfg["data_dir"]))
    base = build_model(SimpleNamespace(**cfg), data, compute_item_popularity(data)).to(device)
    base.load_state_dict(torch.load(base_dir / "best_model.pt", map_location=device, weights_only=True)); base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    cp = {k: int(cfg[k]) for k in ("max_seq_len", "recent_window", "main_topk", "min_main_count")}
    valid = load_or_build_cache(Path(cfg["data_dir"]), data, "valid", **cp)
    test = load_or_build_cache(Path(cfg["data_dir"]), data, "test", **cp)
    valid_rows = collect(base, valid, data, args, device)
    selected, grid, configs = choose(valid_rows, args.method, data, args.seed)
    valid_orders = orders_for(valid_rows, args.method, selected, data, args.seed)
    test_rows = collect(base, test, data, args, device)
    test_orders = orders_for(test_rows, args.method, selected, data, args.seed + 900000)
    valid_result, _ = metrics(valid_rows, valid_orders, args.topk); test_result, ranks = metrics(test_rows, test_orders, args.topk)
    with (out / "test_ranks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=("user_id", "target", "group", "rank")); w.writeheader()
        for row, rank in zip(test_rows, ranks): w.writerow({"user_id": row["user_id"], "target": row["target"], "group": row["group"], "rank": rank})
    serial = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in selected.items() if k != "popular"}
    notes = {"topncal": "rank-binned Top-N calibration reproduction; bins are fitted from validation top-N hit rates",
             "smmr": "official SMMR inference principle on frozen Base Top-100 plus frozen item embeddings",
             "duor": "reimplementation of DUoR's published stepwise user-popularity calibration under the common frozen-output protocol"}
    save_json(out / "config.json", vars(args) | {"dataset": cfg["dataset"], "backbone": cfg["backbone"], "base_seed": cfg["seed"], "base_run_dir": str(base_dir.resolve()), "selected": serial, "validation_grid": grid, "protocol": PROTOCOL, "method_note": notes[args.method], "test_evaluated_once": True})
    save_json(out / "valid_metrics.json", {"method": args.method, "selected": serial, "metrics": valid_result, "grid": grid})
    save_json(out / "test_metrics.json", {"method": args.method, "dataset": cfg["dataset"], "seed": cfg["seed"], "selected": serial, "metrics": test_result, "selected_by": "validation only"})
    print(json.dumps(test_result, indent=2), flush=True)


if __name__ == "__main__": main()
