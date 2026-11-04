#!/usr/bin/env python3
"""Train and validate the frozen-backbone RECal plug-in."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refrec.data_utils import (FARRecDataset, collate_eval_batch,
                               compute_item_popularity, load_dataset,
                               load_or_build_cache)
from refrec.diagnose_repeat_mass import calibration_summary
from refrec.metrics import aggregate_from_ranks
from refrec.recal import (RepeatIntentHead, asymmetric_risk_trust,
                          bidirectional_risk_trust,
                          group_logit_shift, sequence_statistics,
                          shrink_target_mass)
from refrec.run_seqrec import build_model
from refrec.utils import save_json


def parse_args():
    p = argparse.ArgumentParser(description="Frozen-backbone RECal")
    p.add_argument("--base_run_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--encode_batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--decode_chunk", type=int, default=8192)
    p.add_argument("--calibration_mode", choices=["full", "risk", "asymmetric"], default="risk")
    p.add_argument("--down_power", type=float, default=12.0)
    p.add_argument("--up_power", type=float, default=12.0)
    return p.parse_args()


@torch.no_grad()
def encode_samples(model, samples, device, batch_size, recent_window):
    loader = DataLoader(FARRecDataset(samples), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_eval_batch)
    hs, zs, ys = [], [], []
    model.eval()
    for batch in loader:
        bt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        enc = model.encode(bt["seq_items"], bt["seq_cats"], bt["valid_mask"],
                           bt["weak_mask"], bt["recent_mask"])
        hs.append(enc["h_user"].cpu())
        zs.append(sequence_statistics(bt["seq_items"], bt["valid_mask"], recent_window).cpu())
        ys.append(torch.tensor([float(t in h) for t, h in zip(batch["targets"].tolist(),
                                                               batch["history_sets"]) ]))
    return torch.cat(hs), torch.cat(zs), torch.cat(ys)


def train_head(head, train_tensors, valid_tensors, args, device):
    htr, ztr, ytr = train_tensors
    hva, zva, yva = valid_tensors
    head.set_stats_normalization(ztr.to(device))
    loader = DataLoader(TensorDataset(htr, ztr, ytr), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_epoch, stale, best_state = math.inf, 0, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train(); total = count = 0
        for h, z, y in loader:
            h, z, y = h.to(device), z.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(h, z), y)
            loss.backward(); opt.step()
            total += float(loss) * len(y); count += len(y)
        head.eval()
        with torch.no_grad():
            logits = head(hva.to(device), zva.to(device))
            valid_loss = float(F.binary_cross_entropy_with_logits(logits, yva.to(device)))
            valid_brier = float(((torch.sigmoid(logits) - yva.to(device)) ** 2).mean())
        history.append({"epoch": epoch, "train_bce": total / count,
                        "valid_bce": valid_loss, "valid_brier": valid_brier})
        print(f"intent epoch={epoch} train_bce={total/count:.6f} valid_bce={valid_loss:.6f} valid_brier={valid_brier:.6f}", flush=True)
        if valid_loss < best - 1e-7:
            best, best_epoch, stale = valid_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best_state)
    return best_epoch, history


def rank_metrics(rows, key):
    return aggregate_from_ranks([int(r[key]) for r in rows])


@torch.no_grad()
def evaluate_recal(model, head, samples, data, args, device, recent_window):
    loader = DataLoader(FARRecDataset(samples), batch_size=args.eval_batch_size,
                        shuffle=False, collate_fn=collate_eval_batch)
    all_items = torch.arange(1, data.num_items + 1, device=device)
    rows = []
    model.eval(); head.eval()
    for batch in loader:
        bt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        enc = model.encode(bt["seq_items"], bt["seq_cats"], bt["valid_mask"],
                           bt["weak_mask"], bt["recent_mask"])
        enc = model.prepare_full_catalog(enc, bt["seq_items"], bt["valid_mask"],
                                         all_items, chunk_size=args.decode_chunk)
        q = torch.sigmoid(head(enc["h_user"], sequence_statistics(
            bt["seq_items"], bt["valid_mask"], recent_window)))
        bsz = len(batch["history_sets"])
        log_z = torch.full((bsz,), -torch.inf, device=device)
        log_zh = torch.full((bsz,), -torch.inf, device=device)
        targets = bt["targets"]
        target_base = torch.empty(bsz, device=device)
        for start in range(0, data.num_items, args.decode_chunk):
            ids = all_items[start:start + args.decode_chunk]
            scores = model.compute_base_score(enc, bt["seq_items"], bt["valid_mask"],
                                              ids[None, :].expand(bsz, -1))
            log_z = torch.logaddexp(log_z, torch.logsumexp(scores, 1))
            for row, history in enumerate(batch["history_sets"]):
                hm = torch.isin(ids, torch.as_tensor(sorted(history), device=device))
                if hm.any():
                    log_zh[row] = torch.logaddexp(log_zh[row], torch.logsumexp(scores[row, hm], 0))
            inside = (targets >= ids[0]) & (targets <= ids[-1])
            rr = torch.nonzero(inside, as_tuple=False).squeeze(1)
            target_base[rr] = scores[rr, targets[rr] - ids[0]]
        m = torch.exp(log_zh - log_z)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        if args.calibration_mode == "risk":
            trust = bidirectional_risk_trust(m, q, args.down_power, args.up_power)
            target_mass = shrink_target_mass(m, q, trust)
        elif args.calibration_mode == "asymmetric":
            trust = asymmetric_risk_trust(m, q, args.down_power)
            target_mass = shrink_target_mass(m, q, trust)
        else:
            trust = torch.ones_like(q)
            target_mass = q
        delta = group_logit_shift(target_mass, m)
        repeat_target = torch.tensor([float(t in h) for t, h in zip(targets.tolist(), batch["history_sets"])], device=device)
        target_cal = target_base + delta * repeat_target
        bg = torch.zeros(bsz, dtype=torch.long, device=device); be = torch.zeros_like(bg)
        cg = torch.zeros_like(bg); ce = torch.zeros_like(bg)
        for start in range(0, data.num_items, args.decode_chunk):
            ids = all_items[start:start + args.decode_chunk]
            base = model.compute_base_score(enc, bt["seq_items"], bt["valid_mask"],
                                            ids[None, :].expand(bsz, -1))
            cal = base.clone()
            for row, history in enumerate(batch["history_sets"]):
                hm = torch.isin(ids, torch.as_tensor(sorted(history), device=device))
                cal[row, hm] += delta[row]
            bg += (base > target_base[:, None]).sum(1)
            be += ((base == target_base[:, None]) & (ids[None, :] < targets[:, None])).sum(1)
            cg += (cal > target_cal[:, None]).sum(1)
            ce += ((cal == target_cal[:, None]) & (ids[None, :] < targets[:, None])).sum(1)
        for i in range(bsz):
            rows.append({"user_id": data.user_unmap.get(int(bt['user_id'][i]), int(bt['user_id'][i])),
                         "is_repeat": int(repeat_target[i]), "m": float(m[i]), "q": float(q[i]),
                         "trust": float(trust[i]), "delta": float(delta[i]), "base_rank": int(1 + bg[i] + be[i]),
                         "recal_rank": int(1 + cg[i] + ce[i])})
    return rows


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    base_dir, out = Path(args.base_run_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("repeat_head") != "base": raise ValueError("RECal requires a base checkpoint")
    data = load_dataset(Path(cfg["data_dir"]))
    model = build_model(SimpleNamespace(**cfg), data, compute_item_popularity(data)).to(device)
    model.load_state_dict(torch.load(base_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    cp = {k: int(cfg[k]) for k in ("max_seq_len", "recent_window", "main_topk", "min_main_count")}
    train = load_or_build_cache(Path(cfg["data_dir"]), data, "train", **cp)
    valid = load_or_build_cache(Path(cfg["data_dir"]), data, "valid", **cp)
    train_t = encode_samples(model, train, device, args.encode_batch_size, cfg["recent_window"])
    valid_t = encode_samples(model, valid, device, args.encode_batch_size, cfg["recent_window"])
    head = RepeatIntentHead(cfg["emb_dim"], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    best_epoch, history = train_head(head, train_t, valid_t, args, device)
    torch.save(head.state_dict(), out / "intent_head.pt")
    rows = evaluate_recal(model, head, valid, data, args, device, cfg["recent_window"])
    with (out / "valid_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    labels = np.asarray([r["is_repeat"] for r in rows], dtype=float)
    qvals = np.asarray([r["q"] for r in rows], dtype=float)
    q_summary, q_bins = calibration_summary(labels, qvals)
    groups = {}
    for name, flag in (("repeat", 1), ("explore", 0)):
        subset = [r for r in rows if r["is_repeat"] == flag]
        groups[name] = {"count": len(subset), "base": rank_metrics(subset, "base_rank"),
                        "recal": rank_metrics(subset, "recal_rank")}
    result = {"dataset": cfg["dataset"], "backbone": cfg["backbone"],
              "negative_sampling": cfg["negative_sampling"], "best_intent_epoch": best_epoch,
              "base": rank_metrics(rows, "base_rank"), "recal": rank_metrics(rows, "recal_rank"),
              "groups": groups, "intent_calibration": q_summary,
              "mean_m": float(np.mean([r['m'] for r in rows])),
              "mean_q": float(qvals.mean()), "mean_delta": float(np.mean([r['delta'] for r in rows])),
              "calibration_mode": args.calibration_mode,
              "down_power": args.down_power, "up_power": args.up_power,
              "target_used_in_features": False, "test_evaluated": False}
    save_json(out / "metrics.json", result)
    save_json(out / "training_history.json", {"epochs": history})
    with (out / "intent_reliability_bins.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(q_bins[0])); w.writeheader(); w.writerows(q_bins)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
