#!/usr/bin/env python3
"""Run Hard Mask, calibrated recommendation, PRM and ReCODE fairly.

One invocation consumes one frozen SASRec checkpoint and one seed.  Add-on
parameters are selected without reading test labels; the test output is
write-once.  All methods start from exactly the same full-catalogue Base scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refrec.data_utils import (collate_eval_batch, compute_item_popularity,
                               load_dataset, load_or_build_cache, sample_negatives)
from refrec.metrics import aggregate_from_ranks
from refrec.protocol import PROTOCOL, merge_topk
from refrec.run_seqrec import build_model
from refrec.strong_baselines import PRMReranker, ReCODEHead, recent_gap_bins
from refrec.utils import save_json


METHODS = ("base", "hard_mask", "calibrated_recommendation", "prm", "recode")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--train_list_size", type=int, default=50)
    p.add_argument("--rerank_size", type=int, default=100)
    p.add_argument("--train_samples", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--decode_chunk", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--ode_steps", type=int, default=51)
    p.add_argument("--time_scalar", type=int, default=604800,
                   help="Seconds per ReCODE time bin; official default is one week.")
    p.add_argument("--calibration_lambdas", nargs="+", type=float,
                   default=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5])
    p.add_argument("--calibration_tune_users", type=int, default=2000)
    return p.parse_args()


def temporal_records(data, split: str):
    if split == "train":
        for uid in sorted(data.user_train):
            items, times = data.user_train[uid], data.user_train_times[uid]
            for pos in range(1, len(items)):
                yield items[:pos], times[:pos], times[pos]
    elif split == "valid":
        for uid in sorted(data.user_valid):
            yield (list(data.user_train[uid]), list(data.user_train_times[uid]),
                   data.user_valid_times[uid])
    elif split == "test":
        for uid in sorted(data.user_test):
            yield (list(data.user_train[uid]) + [data.user_valid[uid]],
                   list(data.user_train_times[uid]) + [data.user_valid_times[uid]],
                   data.user_test_times[uid])
    else:
        raise ValueError(split)


def choose_training_subset(samples, records, maximum: int, seed: int):
    n = len(samples)
    if maximum <= 0 or maximum >= n:
        return samples, list(records)
    chosen = np.sort(np.random.default_rng(seed).choice(n, maximum, replace=False))
    chosen_set = set(chosen.tolist())
    selected_samples = [samples[i] for i in chosen]
    selected_records = [record for i, record in enumerate(records) if i in chosen_set]
    return selected_samples, selected_records


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def candidate_batch(samples, records, num_items, list_size, rng, scalar, steps):
    collated = collate_eval_batch(samples)
    candidates, gaps = [], []
    for sample, (hist_items, hist_times, target_time) in zip(samples, records):
        target = int(sample["target_item"])
        negs = sample_negatives(target, sample["history_set"], num_items,
                                list_size - 1, rng, mode="all_except_target",
                                num_history_neg=0)
        ids = [target] + negs
        candidates.append(ids)
        gaps.append(recent_gap_bins(hist_items, hist_times, ids, target_time,
                                    scalar, steps))
    collated["candidate_items"] = torch.tensor(candidates, dtype=torch.long)
    collated["gap_bins"] = torch.stack(gaps)
    return collated


@torch.no_grad()
def encode_contexts(base, samples, batch_size, device):
    outputs = []
    base.eval()
    for start in range(0, len(samples), batch_size):
        bt = to_device(collate_eval_batch(samples[start:start + batch_size]), device)
        enc = base.encode(bt["seq_items"], bt["seq_cats"], bt["valid_mask"],
                          bt["weak_mask"], bt["recent_mask"])
        outputs.append(enc["h_user"].cpu())
    return torch.cat(outputs)


def addon_losses(base, prm, recode, batch, device, cached_user=None):
    bt = to_device(batch, device)
    with torch.no_grad():
        cand = bt["candidate_items"]
        if cached_user is None:
            enc = base.encode(bt["seq_items"], bt["seq_cats"], bt["valid_mask"],
                              bt["weak_mask"], bt["recent_mask"])
            user = enc["h_user"]
        else:
            user = cached_user.to(device)
        item = base.item_repr(cand)
        scores = (user[:, None, :] * item).sum(-1) + base.item_bias(cand).squeeze(-1)
    hist = torch.zeros_like(cand, dtype=torch.float32)
    for row, history in enumerate(batch["history_sets"]):
        hist[row] = torch.isin(cand[row], torch.as_tensor(sorted(history), device=device)).float()
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    sorted_scores = scores.gather(1, order)
    sorted_items = item.gather(1, order[:, :, None].expand_as(item))
    sorted_hist = hist.gather(1, order)
    labels = (order == 0).nonzero(as_tuple=False)[:, 1]
    prm_loss = F.cross_entropy(prm(user, sorted_items, sorted_scores, sorted_hist), labels)
    recode_scores = scores + recode(user, item, bt["gap_bins"])
    recode_loss = F.cross_entropy(recode_scores, torch.zeros(len(samples := batch["history_sets"]),
                                                              dtype=torch.long, device=device))
    return prm_loss, recode_loss


def train_addons(base, prm, recode, train_samples, train_records,
                 valid_samples, valid_records, args, device):
    optimizer = torch.optim.AdamW(list(prm.parameters()) + list(recode.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    print("encoding frozen backbone contexts once", flush=True)
    train_context = encode_contexts(base, train_samples, args.batch_size, device)
    valid_context = encode_contexts(base, valid_samples, args.batch_size, device)
    best, stale, best_epoch, best_state = math.inf, 0, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_samples)); prm.train(); recode.train()
        total_p = total_r = seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            samples = [train_samples[i] for i in idx]
            records = [train_records[i] for i in idx]
            batch = candidate_batch(samples, records, base.num_items,
                                    args.train_list_size, rng, args.time_scalar,
                                    args.ode_steps)
            optimizer.zero_grad(set_to_none=True)
            lp, lr = addon_losses(base, prm, recode, batch, device,
                                  train_context[idx])
            (lp + lr).backward(); optimizer.step()
            total_p += float(lp) * len(idx); total_r += float(lr) * len(idx); seen += len(idx)
        prm.eval(); recode.eval(); vp = vr = vn = 0
        # Reuse the same validation candidate lists every epoch; otherwise
        # early stopping would partly select negative-sampling noise.
        valid_rng = np.random.default_rng(args.seed + 100000)
        with torch.no_grad():
            for start in range(0, len(valid_samples), args.batch_size):
                samples = valid_samples[start:start + args.batch_size]
                records = valid_records[start:start + args.batch_size]
                batch = candidate_batch(samples, records, base.num_items,
                                        args.train_list_size, valid_rng,
                                        args.time_scalar, args.ode_steps)
                lp, lr = addon_losses(base, prm, recode, batch, device,
                                      valid_context[start:start + len(samples)])
                vp += float(lp) * len(samples); vr += float(lr) * len(samples); vn += len(samples)
        score = (vp + vr) / max(vn, 1)
        row = {"epoch": epoch, "prm_train_ce": total_p / seen,
               "recode_train_ce": total_r / seen, "prm_valid_ce": vp / vn,
               "recode_valid_ce": vr / vn, "selection_loss": score}
        history.append(row); print(row, flush=True)
        if score < best - 1e-6:
            best, stale, best_epoch = score, 0, epoch
            best_state = {"prm": {k: v.detach().cpu().clone() for k, v in prm.state_dict().items()},
                          "recode": {k: v.detach().cpu().clone() for k, v in recode.state_dict().items()}}
        else:
            stale += 1
            if stale >= args.patience:
                break
    prm.load_state_dict(best_state["prm"]); recode.load_state_dict(best_state["recode"])
    return best_epoch, history


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    middle = 0.5 * (p + q)
    left = np.where(p > 0, p * np.log((p + 1e-12) / (middle + 1e-12)), 0).sum()
    right = np.where(q > 0, q * np.log((q + 1e-12) / (middle + 1e-12)), 0).sum()
    return float(0.5 * (left + right))


def calibrated_top20(items, scores, history, item2cat, num_categories, weight):
    counts = Counter(item2cat.get(i, 0) for i in history if item2cat.get(i, 0) > 0)
    if not counts:
        return items[:20]
    profile = np.zeros(num_categories + 1, dtype=np.float64)
    for cat, count in counts.items(): profile[cat] = count
    profile /= profile.sum()
    scores = np.asarray(scores, dtype=np.float64)
    relevance = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
    item_array = np.asarray(items, dtype=np.int64)
    cats = np.asarray([item2cat.get(i, 0) for i in items], dtype=np.int64)
    remaining = np.arange(len(items)); selected = []; slate = np.zeros_like(profile)
    for _ in range(min(20, len(items))):
        penalties = {}
        for cat in np.unique(cats[remaining]):
            trial = slate.copy()
            if cat > 0: trial[cat] += 1
            penalties[int(cat)] = js_divergence(profile, trial / max(trial.sum(), 1))
        values = ((1 - weight) * relevance[remaining] - weight *
                  np.asarray([penalties[int(cat)] for cat in cats[remaining]]))
        # Primary key: descending objective. Secondary key: ascending item id.
        local = np.lexsort((item_array[remaining], -values))[0]
        best_idx = int(remaining[local])
        selected.append(items[best_idx]); cat = item2cat.get(items[best_idx], 0)
        if cat > 0: slate[cat] += 1
        remaining = np.delete(remaining, local)
    return selected


def rank_from_top(items, target):
    try: return items.index(target) + 1
    except ValueError: return 21


@torch.no_grad()
def evaluate(base, prm, recode, samples, records, data, args, device, cal_weights):
    base.eval(); prm.eval(); recode.eval(); all_items = torch.arange(1, data.num_items + 1, device=device)
    rows = []
    for start_batch in range(0, len(samples), args.eval_batch_size):
        sb = samples[start_batch:start_batch + args.eval_batch_size]
        rb = records[start_batch:start_batch + args.eval_batch_size]
        batch = collate_eval_batch(sb); bt = to_device(batch, device); bsz = len(sb)
        enc = base.encode(bt["seq_items"], bt["seq_cats"], bt["valid_mask"],
                          bt["weak_mask"], bt["recent_mask"])
        enc = base.prepare_full_catalog(enc, bt["seq_items"], bt["valid_mask"],
                                        all_items, chunk_size=args.decode_chunk)
        targets = bt["targets"]
        target_base = base.compute_base_score(enc, bt["seq_items"], bt["valid_mask"],
                                              targets[:, None]).squeeze(1)
        hist_ids, hist_final, target_final = [], [], target_base.clone()
        for row, (sample, record) in enumerate(zip(sb, rb)):
            ids = sorted(sample["history_set"]); hist_ids.append(ids)
            if ids:
                ids_t = torch.tensor(ids, device=device)[None, :]
                hbase = base.compute_base_score(
                    {k: v[row:row+1] if isinstance(v, torch.Tensor) and v.size(0) == bsz else v
                     for k, v in enc.items()}, bt["seq_items"][row:row+1],
                    bt["valid_mask"][row:row+1], ids_t).squeeze(0)
                gap = recent_gap_bins(*record[:2], ids, record[2], args.time_scalar,
                                      args.ode_steps).to(device)[None, :, :]
                dyn = recode(enc["h_user"][row:row+1], base.item_repr(ids_t), gap).squeeze(0)
                final = hbase + dyn; hist_final.append(final)
                if int(targets[row]) in sample["history_set"]:
                    target_final[row] = final[ids.index(int(targets[row]))]
            else:
                hist_final.append(torch.empty(0, device=device))
        greater_b = torch.zeros(bsz, dtype=torch.long, device=device)
        equal_b = torch.zeros_like(greater_b); greater_h = torch.zeros_like(greater_b)
        equal_h = torch.zeros_like(greater_b); greater_r = torch.zeros_like(greater_b)
        equal_r = torch.zeros_like(greater_b)
        top_scores = top_items = None
        for offset in range(0, data.num_items, args.decode_chunk):
            ids = all_items[offset:offset + args.decode_chunk]
            scores = base.compute_base_score(enc, bt["seq_items"], bt["valid_mask"],
                                             ids[None, :].expand(bsz, -1))
            top_scores, top_items = merge_topk(top_scores, top_items, scores,
                                               ids[None, :].expand(bsz, -1), args.rerank_size)
            greater_b += (scores > target_base[:, None]).sum(1)
            equal_b += ((scores == target_base[:, None]) & (ids[None, :] < targets[:, None])).sum(1)
            for row, history in enumerate(batch["history_sets"]):
                hm = torch.isin(ids, torch.as_tensor(sorted(history), device=device))
                hard_scores = scores[row].masked_fill(hm, -torch.inf)
                if int(targets[row]) not in history:
                    greater_h[row] += (hard_scores > target_base[row]).sum()
                    equal_h[row] += ((hard_scores == target_base[row]) & (ids < targets[row])).sum()
                rec_scores = scores[row].clone()
                if hm.any():
                    positions = torch.nonzero(hm, as_tuple=False).squeeze(1)
                    local = {item: j for j, item in enumerate(hist_ids[row])}
                    replacement = [local[int(ids[p])] for p in positions]
                    rec_scores[positions] = hist_final[row][replacement]
                greater_r[row] += (rec_scores > target_final[row]).sum()
                equal_r[row] += ((rec_scores == target_final[row]) & (ids < targets[row])).sum()
        for row, sample in enumerate(sb):
            items = top_items[row].tolist(); scores = top_scores[row].tolist(); target = int(targets[row])
            item_t = top_items[row:row+1]
            history_flag = torch.isin(item_t, torch.as_tensor(sorted(sample["history_set"]), device=device)).float()
            prm_scores = prm(enc["h_user"][row:row+1], base.item_repr(item_t),
                             top_scores[row:row+1], history_flag).squeeze(0)
            prm_order = torch.argsort(prm_scores, descending=True, stable=True).tolist()
            out = {"user_id": int(bt["user_id"][row]),
                   "group": "repeat" if target in sample["history_set"] else "explore",
                   "rank_base": int(1 + greater_b[row] + equal_b[row]),
                   "rank_hard_mask": (data.num_items + 1 if target in sample["history_set"]
                                      else int(1 + greater_h[row] + equal_h[row])),
                   "rank_prm": rank_from_top([items[i] for i in prm_order[:20]], target),
                   "rank_recode": int(1 + greater_r[row] + equal_r[row])}
            for weight in cal_weights:
                key = f"rank_cal_{weight:g}"
                out[key] = rank_from_top(calibrated_top20(
                    items, scores, sample["history_set"], data.item2cat,
                    data.num_categories, weight), target)
            rows.append(out)
        print(f"evaluated {min(start_batch + bsz, len(samples))}/{len(samples)}", flush=True)
    return rows


def grouped_metrics(rows, rank_key, group=None):
    chosen = rows if group is None else [r for r in rows if r["group"] == group]
    return {**aggregate_from_ranks([int(r[rank_key]) for r in chosen]), "count": len(chosen)}


def summarize(rows, cal_weight):
    keys = {"base": "rank_base", "hard_mask": "rank_hard_mask",
            "calibrated_recommendation": f"rank_cal_{cal_weight:g}",
            "prm": "rank_prm", "recode": "rank_recode"}
    return {method: {"overall": grouped_metrics(rows, key),
                     "repeat": grouped_metrics(rows, key, "repeat"),
                     "explore": grouped_metrics(rows, key, "explore")}
            for method, key in keys.items()}


def main():
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_dir, out = Path(args.base_run_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "test_metrics.json").exists():
        raise FileExistsError(f"write-once test output already exists: {out}")
    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    if cfg["backbone"] != "sasrec" or cfg.get("repeat_head") != "base":
        raise ValueError("comparison protocol requires a frozen plain SASRec checkpoint")
    data = load_dataset(Path(cfg["data_dir"])); base = build_model(
        SimpleNamespace(**cfg), data, compute_item_popularity(data)).to(device)
    base.load_state_dict(torch.load(base_dir / "best_model.pt", map_location=device, weights_only=True))
    base.eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    cp = {k: int(cfg[k]) for k in ("max_seq_len", "recent_window", "main_topk", "min_main_count")}
    train = load_or_build_cache(Path(cfg["data_dir"]), data, "train", **cp)
    valid = load_or_build_cache(Path(cfg["data_dir"]), data, "valid", **cp)
    test = load_or_build_cache(Path(cfg["data_dir"]), data, "test", **cp)
    train, train_records = choose_training_subset(
        train, temporal_records(data, "train"), args.train_samples, args.seed)
    valid_records = list(temporal_records(data, "valid")); test_records = list(temporal_records(data, "test"))
    if not (len(train) == len(train_records) and len(valid) == len(valid_records) and len(test) == len(test_records)):
        raise RuntimeError("temporal records are not aligned with recommendation samples")
    prm = PRMReranker(cfg["emb_dim"], list_size=args.rerank_size).to(device)
    recode = ReCODEHead(cfg["emb_dim"], steps=args.ode_steps).to(device)
    best_epoch, training_history = train_addons(
        base, prm, recode, train, train_records, valid, valid_records, args, device)
    torch.save({"prm": prm.state_dict(), "recode": recode.state_dict()}, out / "addons.pt")
    save_json(out / "training_history.json", {"best_epoch": best_epoch, "epochs": training_history})
    tune_count = min(len(valid), max(1, args.calibration_tune_users))
    if tune_count < len(valid):
        tune_idx = np.sort(np.random.default_rng(args.seed + 700000).choice(
            len(valid), tune_count, replace=False))
        tune_samples = [valid[i] for i in tune_idx]
        tune_records = [valid_records[i] for i in tune_idx]
    else:
        tune_samples, tune_records = valid, valid_records
    tune_rows = evaluate(base, prm, recode, tune_samples, tune_records, data, args, device,
                         args.calibration_lambdas)
    valid_scores = {weight: aggregate_from_ranks(
        [r[f"rank_cal_{weight:g}"] for r in tune_rows])["ndcg@10"]
        for weight in args.calibration_lambdas}
    best_weight = max(args.calibration_lambdas, key=lambda x: (valid_scores[x], -x))
    if tune_count == len(valid):
        valid_rows = tune_rows
    else:
        valid_rows = evaluate(base, prm, recode, valid, valid_records, data, args,
                              device, [best_weight])
    valid_result = summarize(valid_rows, best_weight)
    test_rows = evaluate(base, prm, recode, test, test_records, data, args, device, [best_weight])
    test_result = summarize(test_rows, best_weight)
    with (out / "test_ranks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0])); writer.writeheader(); writer.writerows(test_rows)
    config = vars(args) | {"dataset": cfg["dataset"], "backbone": "sasrec",
                           "base_seed": cfg["seed"], "base_run_dir": str(base_dir.resolve()),
                           "selected_calibration_lambda": best_weight,
                           "protocol": PROTOCOL, "test_evaluated_once": True,
                           "target_used_in_features": False,
                           "method_notes": {
                               "hard_mask": "all prefix items receive -infinity, with no target exception",
                               "calibrated_recommendation": "Steck-style category-distribution reranking of Base Top-100",
                               "prm": "personalized Transformer reranking of Base Top-100",
                               "recode": "official-style five-gap Euler neural ODE repeat head on frozen Base"}}
    save_json(out / "config.json", config)
    save_json(out / "valid_metrics.json", {"methods": valid_result,
              "calibration_grid": {str(k): v for k, v in valid_scores.items()}})
    save_json(out / "test_metrics.json", {"dataset": cfg["dataset"], "seed": cfg["seed"],
              "methods": test_result, "selected_by": "validation only",
              "selected_calibration_lambda": best_weight})
    print(json.dumps(test_result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
