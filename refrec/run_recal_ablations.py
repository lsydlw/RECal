#!/usr/bin/env python3
"""Run the ten core RECal ablations with shared full-catalogue passes.

All variants use the same frozen backbone and validation users.  The standard
intent head is loaded from ``recal_run_dir``.  A second context-only head is
trained for the ``no_explicit_stats`` ablation; it never receives the five
hand-crafted sequence statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refrec.data_utils import (FARRecDataset, collate_eval_batch,
                               compute_item_popularity, load_dataset,
                               load_or_build_cache)
from refrec.metrics import aggregate_from_ranks
from refrec.recal import (RepeatIntentHead, asymmetric_risk_trust,
                          bidirectional_risk_trust, entropy_confidence,
                          group_logit_shift, sequence_statistics,
                          shrink_target_mass)
from refrec.run_recal_frozen import encode_samples, train_head
from refrec.run_seqrec import build_model
from refrec.utils import save_json


VARIANTS = (
    "base",
    "fixed_history_boost",
    "full_projection",
    "fixed_global_g",
    "entropy_confidence",
    "q_only_no_m",
    "global_repeat_rate",
    "no_explicit_stats",
    "one_sided_risk",
    "bidirectional_risk",
)


def parse_args():
    p = argparse.ArgumentParser(description="Core RECal ablations")
    p.add_argument("--base_run_dir", required=True)
    p.add_argument("--recal_run_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", choices=("valid", "test"), default="valid",
                   help="Evaluation split. Heads are always selected on validation data.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--encode_batch_size", type=int, default=256)
    p.add_argument("--decode_chunk", type=int, default=8192)
    p.add_argument("--fixed_boost", type=float, default=math.log(2.0),
                   help="Fixed history-logit boost; log(2) doubles history odds.")
    p.add_argument("--fixed_g", type=float, default=0.1)
    p.add_argument("--risk_power", type=float, default=12.0)
    p.add_argument("--head_epochs", type=int, default=100)
    p.add_argument("--head_patience", type=int, default=10)
    p.add_argument("--head_batch_size", type=int, default=512)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--head_weight_decay", type=float, default=1e-5)
    p.add_argument("--variants", nargs="+", choices=VARIANTS,
                   default=list(VARIANTS),
                   help="Evaluate only these variants and merge them into any existing outputs.")
    return p.parse_args()


def _metrics(rows, variant, group=None):
    chosen = rows if group is None else [r for r in rows if r["group"] == group]
    metrics = aggregate_from_ranks([int(r[f"rank_{variant}"]) for r in chosen])
    metrics["count"] = len(chosen)
    return metrics


def _head_args(args):
    return SimpleNamespace(batch_size=args.head_batch_size, seed=args.seed,
                           lr=args.head_lr, weight_decay=args.head_weight_decay,
                           epochs=args.head_epochs, patience=args.head_patience)


def load_or_train_context_head(model, train, valid, cfg, args, out, device):
    """Train a clean context-only q head for the statistics ablation."""
    head = RepeatIntentHead(cfg["emb_dim"], num_stats=0,
                            hidden_dim=args.hidden_dim).to(device)
    checkpoint = out / "context_only_intent_head.pt"
    history_path = out / "context_only_training_history.json"
    train_t = encode_samples(model, train, device, args.encode_batch_size,
                             cfg["recent_window"])
    valid_t = encode_samples(model, valid, device, args.encode_batch_size,
                             cfg["recent_window"])
    train_t = (train_t[0], train_t[1][:, :0], train_t[2])
    valid_t = (valid_t[0], valid_t[1][:, :0], valid_t[2])
    if checkpoint.exists():
        head.load_state_dict(torch.load(checkpoint, map_location=device,
                                        weights_only=True))
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
        best_epoch = history.get("best_epoch", 0)
    else:
        best_epoch, epochs = train_head(head, train_t, valid_t,
                                        _head_args(args), device)
        torch.save(head.state_dict(), checkpoint)
        history = {"best_epoch": best_epoch, "epochs": epochs}
        save_json(history_path, history)
    return head, train_t, valid_t, best_epoch


@torch.no_grad()
def evaluate(model, standard_head, context_head, samples, data, cfg, args,
             device, global_repeat_rate, variants):
    loader = DataLoader(FARRecDataset(samples), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_eval_batch)
    all_items = torch.arange(1, data.num_items + 1, device=device)
    rows = []
    model.eval(); standard_head.eval(); context_head.eval()
    for batch in loader:
        bt = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}
        enc = model.encode(bt["seq_items"], bt["seq_cats"], bt["valid_mask"],
                           bt["weak_mask"], bt["recent_mask"])
        # Some backbones (notably RepeatNet) mix normalized probabilities and
        # therefore need a single full-catalogue partition function before
        # chunked scoring.  Without this hook each chunk is normalized on its
        # own and the resulting ranks depend on decode_chunk.
        enc = model.prepare_full_catalog(
            enc, bt["seq_items"], bt["valid_mask"], all_items,
            chunk_size=args.decode_chunk,
        )
        stats = sequence_statistics(bt["seq_items"], bt["valid_mask"],
                                    cfg["recent_window"])
        q = torch.sigmoid(standard_head(enc["h_user"], stats))
        empty_stats = stats[:, :0]
        q_context = torch.sigmoid(context_head(enc["h_user"], empty_stats))
        bsz = len(batch["history_sets"]); targets = bt["targets"]
        log_z = torch.full((bsz,), -torch.inf, device=device)
        log_zh = torch.full((bsz,), -torch.inf, device=device)
        target_base = torch.empty(bsz, device=device)
        for start in range(0, data.num_items, args.decode_chunk):
            ids = all_items[start:start + args.decode_chunk]
            base = model.compute_base_score(enc, bt["seq_items"], bt["valid_mask"],
                                            ids[None].expand(bsz, -1))
            log_z = torch.logaddexp(log_z, torch.logsumexp(base, 1))
            for i, hist in enumerate(batch["history_sets"]):
                hm = torch.isin(ids, torch.as_tensor(sorted(hist), device=device))
                if hm.any():
                    log_zh[i] = torch.logaddexp(log_zh[i],
                                                torch.logsumexp(base[i, hm], 0))
            inside = (targets >= ids[0]) & (targets <= ids[-1])
            rr = torch.nonzero(inside, as_tuple=False).squeeze(1)
            target_base[rr] = base[rr, targets[rr] - ids[0]]
        m = torch.exp(log_zh - log_z)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        global_q = torch.full_like(q, global_repeat_rate)
        zero = torch.zeros_like(q)
        deltas = {
            "base": zero,
            "fixed_history_boost": torch.full_like(q, args.fixed_boost),
            "full_projection": group_logit_shift(q, m),
            "fixed_global_g": group_logit_shift(
                shrink_target_mass(m, q, args.fixed_g), m),
            "entropy_confidence": group_logit_shift(
                shrink_target_mass(m, q, entropy_confidence(q)), m),
            # q is used as a candidate-independent additive history log-odds;
            # this deliberately never reads the Base history mass m.
            "q_only_no_m": torch.logit(q.clamp(1e-6, 1 - 1e-6)),
            "global_repeat_rate": group_logit_shift(global_q, m),
            "no_explicit_stats": group_logit_shift(
                shrink_target_mass(m, q_context, asymmetric_risk_trust(
                    m, q_context, args.risk_power)), m),
            "one_sided_risk": group_logit_shift(
                shrink_target_mass(m, q, asymmetric_risk_trust(
                    m, q, args.risk_power)), m),
            "bidirectional_risk": group_logit_shift(
                shrink_target_mass(m, q, bidirectional_risk_trust(
                    m, q, args.risk_power, args.risk_power)), m),
        }
        repeated = torch.tensor([float(t in h) for t, h in
                                 zip(targets.tolist(), batch["history_sets"])],
                                device=device)
        target_scores = {name: target_base + deltas[name] * repeated
                         for name in variants}
        greater = {name: torch.zeros(bsz, dtype=torch.long, device=device)
                   for name in variants}
        equals = {name: torch.zeros(bsz, dtype=torch.long, device=device)
                  for name in variants}
        for start in range(0, data.num_items, args.decode_chunk):
            ids = all_items[start:start + args.decode_chunk]
            base = model.compute_base_score(enc, bt["seq_items"], bt["valid_mask"],
                                            ids[None].expand(bsz, -1))
            hist_masks = [torch.isin(ids, torch.as_tensor(sorted(h), device=device))
                          for h in batch["history_sets"]]
            for name in variants:
                scores = base.clone()
                for i, hm in enumerate(hist_masks):
                    scores[i, hm] += deltas[name][i]
                greater[name] += (scores > target_scores[name][:, None]).sum(1)
                equals[name] += ((scores == target_scores[name][:, None]) &
                                 (ids[None] < targets[:, None])).sum(1)
        for i in range(bsz):
            row = {"user_id": int(bt["user_id"][i]),
                   "group": "repeat" if repeated[i] else "explore",
                   "m": float(m[i]), "q": float(q[i]),
                   "q_context_only": float(q_context[i])}
            for name in variants:
                row[f"rank_{name}"] = int(1 + greater[name][i] + equals[name][i])
            rows.append(row)
    return rows


def merge_rank_rows(path, new_rows, variants):
    """Merge selectively recomputed rank columns without touching other variants."""
    if not path.exists() or set(variants) == set(VARIANTS):
        return new_rows
    with path.open(newline="", encoding="utf-8") as f:
        old_rows = list(csv.DictReader(f))
    if len(old_rows) != len(new_rows):
        raise RuntimeError(f"Cannot merge {len(new_rows)} rows into {len(old_rows)} rows")
    for old, new in zip(old_rows, new_rows):
        if old["user_id"] != str(new["user_id"]) or old["group"] != new["group"]:
            raise RuntimeError("Existing and recomputed validation rows are not aligned")
        old["m"] = new["m"]
        old["q"] = new["q"]
        old["q_context_only"] = new["q_context_only"]
        for name in variants:
            old[f"rank_{name}"] = new[f"rank_{name}"]
    return old_rows


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device.startswith("cuda") and
                          torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_run_dir); recal_dir = Path(args.recal_run_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    if args.split == "test" and metrics_path.exists():
        raise FileExistsError(f"test ablation was already evaluated: {metrics_path}")
    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    data = load_dataset(Path(cfg["data_dir"]))
    model = build_model(SimpleNamespace(**cfg), data,
                        compute_item_popularity(data)).to(device)
    model.load_state_dict(torch.load(base_dir / "best_model.pt",
                                     map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    standard_head = RepeatIntentHead(cfg["emb_dim"], hidden_dim=args.hidden_dim).to(device)
    standard_head.load_state_dict(torch.load(recal_dir / "intent_head.pt",
                                             map_location=device, weights_only=True))
    cp = {k: int(cfg[k]) for k in ("max_seq_len", "recent_window",
                                   "main_topk", "min_main_count")}
    train = load_or_build_cache(Path(cfg["data_dir"]), data, "train", **cp)
    valid = load_or_build_cache(Path(cfg["data_dir"]), data, "valid", **cp)
    context_head, train_t, _, context_best_epoch = load_or_train_context_head(
        model, train, valid, cfg, args, out, device)
    global_repeat_rate = float(train_t[2].mean())
    samples = valid if args.split == "valid" else load_or_build_cache(
        Path(cfg["data_dir"]), data, "test", **cp)
    variants = tuple(args.variants)
    rows = evaluate(model, standard_head, context_head, samples, data, cfg, args,
                    device, global_repeat_rate, variants)
    ranks_path = out / f"{args.split}_ablation_ranks.csv"
    rows = merge_rank_rows(ranks_path, rows, variants)
    with ranks_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if metrics_path.exists() and set(variants) != set(VARIANTS):
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["global_repeat_rate_train"] = global_repeat_rate
        result["context_only_best_epoch"] = context_best_epoch
    else:
        result = {"dataset": cfg["dataset"], "backbone": cfg["backbone"],
                  "seed": cfg["seed"], "split": args.split,
                  "full_catalogue": True,
                  "selected_by": "valid.ndcg@10",
                  "test_evaluated": args.split == "test",
                  "test_evaluated_once": args.split == "test",
                  "global_repeat_rate_train": global_repeat_rate,
                  "fixed_boost": args.fixed_boost, "fixed_g": args.fixed_g,
                  "risk_power": args.risk_power,
                  "context_only_best_epoch": context_best_epoch, "variants": {}}
    result["variant_definitions"] = {
        "no_explicit_stats": "context-only q with the same asymmetric risk gate as one_sided_risk"
    }
    for name in variants:
        result["variants"][name] = {
            "overall": _metrics(rows, name),
            "repeat": _metrics(rows, name, "repeat"),
            "explore": _metrics(rows, name, "explore"),
        }
    save_json(metrics_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
