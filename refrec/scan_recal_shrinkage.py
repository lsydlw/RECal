#!/usr/bin/env python3
"""Evaluate many RECal shrinkage rules with two shared catalogue passes."""

from __future__ import annotations

import argparse
import csv
import json
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
from refrec.recal import (RepeatIntentHead, asymmetric_risk_trust, bidirectional_risk_trust, entropy_confidence,
                          group_logit_shift, sequence_statistics,
                          shrink_target_mass)
from refrec.run_seqrec import build_model
from refrec.utils import save_json


FIXED_G = (0.0, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0)
ASYM_POWERS = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0)
ALL_VARIANTS = ([f"g{g:g}" for g in FIXED_G] +
                ["entropy"] + [f"asym{p:g}" for p in ASYM_POWERS] +
                [
                 "risk_up1", "risk_up2", "risk_up4", "risk_up8", "risk_up12"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_dir", required=True)
    p.add_argument("--recal_run_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--decode_chunk", type=int, default=8192)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--split", choices=("valid", "test"), default="valid")
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--noise_sigmas", nargs="+", type=float, default=[])
    p.add_argument("--variants", nargs="+", choices=ALL_VARIANTS,
                   default=ALL_VARIANTS,
                   help="Evaluate only the requested shrinkage rules.")
    return p.parse_args()


def metrics(rows, variant, group=None):
    selected = rows if group is None else [r for r in rows if r["group"] == group]
    return aggregate_from_ranks([r[f"rank_{variant}"] for r in selected])


@torch.no_grad()
def main():
    args = parse_args(); base_dir = Path(args.base_run_dir); recal_dir = Path(args.recal_run_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data = load_dataset(Path(cfg["data_dir"]))
    model = build_model(SimpleNamespace(**cfg), data, compute_item_popularity(data)).to(device)
    model.load_state_dict(torch.load(base_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    head = RepeatIntentHead(cfg["emb_dim"], hidden_dim=args.hidden_dim).to(device)
    head.load_state_dict(torch.load(recal_dir / "intent_head.pt", map_location=device, weights_only=True))
    head.eval()
    cp = {k: int(cfg[k]) for k in ("max_seq_len", "recent_window", "main_topk", "min_main_count")}
    samples = load_or_build_cache(Path(cfg["data_dir"]), data, args.split, **cp)
    loader = DataLoader(FARRecDataset(samples), batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_eval_batch)
    all_items = torch.arange(1, data.num_items + 1, device=device)
    noise_names = [f"noise{s:g}" for s in args.noise_sigmas]
    variants = list(args.variants) + noise_names
    generator = torch.Generator(device=device).manual_seed(args.seed + 91073)
    rows = []
    for batch in loader:
        bt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        enc = model.encode(bt["seq_items"], bt["seq_cats"], bt["valid_mask"],
                           bt["weak_mask"], bt["recent_mask"])
        q = torch.sigmoid(head(enc["h_user"], sequence_statistics(
            bt["seq_items"], bt["valid_mask"], cfg["recent_window"])))
        bsz = len(batch["history_sets"]); targets = bt["targets"]
        log_z = torch.full((bsz,), -torch.inf, device=device)
        log_zh = torch.full((bsz,), -torch.inf, device=device)
        target_base = torch.empty(bsz, device=device)
        for start in range(0, data.num_items, args.decode_chunk):
            ids = all_items[start:start + args.decode_chunk]
            base = model.compute_base_score(enc, bt["seq_items"], bt["valid_mask"], ids[None].expand(bsz, -1))
            log_z = torch.logaddexp(log_z, torch.logsumexp(base, 1))
            for i, hist in enumerate(batch["history_sets"]):
                hm = torch.isin(ids, torch.as_tensor(sorted(hist), device=device))
                if hm.any(): log_zh[i] = torch.logaddexp(log_zh[i], torch.logsumexp(base[i, hm], 0))
            inside = (targets >= ids[0]) & (targets <= ids[-1])
            rr = torch.nonzero(inside, as_tuple=False).squeeze(1)
            target_base[rr] = base[rr, targets[rr] - ids[0]]
        m = torch.exp(log_zh - log_z); m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        all_trust = {f"g{g:g}": torch.full_like(m, g) for g in FIXED_G}
        all_trust["entropy"] = entropy_confidence(q)
        for power in ASYM_POWERS:
            all_trust[f"asym{power:g}"] = asymmetric_risk_trust(m, q, power)
        all_trust["risk_up1"] = bidirectional_risk_trust(m, q, 12.0, 1.0)
        all_trust["risk_up2"] = bidirectional_risk_trust(m, q, 12.0, 2.0)
        all_trust["risk_up4"] = bidirectional_risk_trust(m, q, 12.0, 4.0)
        all_trust["risk_up8"] = bidirectional_risk_trust(m, q, 12.0, 8.0)
        all_trust["risk_up12"] = bidirectional_risk_trust(m, q, 12.0, 12.0)
        trust = {name: all_trust[name] for name in args.variants}
        deltas = {name: group_logit_shift(shrink_target_mass(m, q, t), m) for name, t in trust.items()}
        # Controlled, target-independent perturbations test robustness to an
        # imperfect intent estimator. Noise is added in logit space so q stays
        # in (0,1); sigma=0 exactly recovers the full asymmetric method.
        for sigma, name in zip(args.noise_sigmas, noise_names):
            noise = torch.randn(q.shape, generator=generator, device=device)
            noisy_q = torch.sigmoid(torch.logit(q.clamp(1e-6, 1-1e-6)) + sigma * noise)
            noisy_trust = asymmetric_risk_trust(m, noisy_q, 12.0)
            deltas[name] = group_logit_shift(shrink_target_mass(m, noisy_q, noisy_trust), m)
        repeated = torch.tensor([float(t in h) for t, h in zip(targets.tolist(), batch["history_sets"])], device=device)
        target_scores = {name: target_base + d * repeated for name, d in deltas.items()}
        greater = {name: torch.zeros(bsz, dtype=torch.long, device=device) for name in variants}
        equals = {name: torch.zeros(bsz, dtype=torch.long, device=device) for name in variants}
        for start in range(0, data.num_items, args.decode_chunk):
            ids = all_items[start:start + args.decode_chunk]
            base = model.compute_base_score(enc, bt["seq_items"], bt["valid_mask"], ids[None].expand(bsz, -1))
            hist_masks = [torch.isin(ids, torch.as_tensor(sorted(h), device=device)) for h in batch["history_sets"]]
            for name in variants:
                sc = base.clone()
                for i, hm in enumerate(hist_masks): sc[i, hm] += deltas[name][i]
                greater[name] += (sc > target_scores[name][:, None]).sum(1)
                equals[name] += ((sc == target_scores[name][:, None]) & (ids[None] < targets[:, None])).sum(1)
        for i in range(bsz):
            row = {"user_id": int(bt["user_id"][i]), "group": "repeat" if repeated[i] else "explore",
                   "m": float(m[i]), "q": float(q[i]),
                   "entropy_trust": float(all_trust["entropy"][i])}
            for name in variants: row[f"rank_{name}"] = int(1 + greater[name][i] + equals[name][i])
            rows.append(row)
    with (out / f"{args.split}_shrinkage_ranks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result = {"dataset": cfg["dataset"], "backbone": cfg["backbone"],
              "split": args.split, "seed": args.seed, "noise_space": "logit(q)",
              "variants": {}}
    for name in variants:
        result["variants"][name] = {"overall": metrics(rows, name),
                                    "repeat": metrics(rows, name, "repeat"),
                                    "explore": metrics(rows, name, "explore")}
    result["mean_m"] = sum(r["m"] for r in rows) / len(rows)
    result["mean_q"] = sum(r["q"] for r in rows) / len(rows)
    result["mean_entropy_trust"] = sum(r["entropy_trust"] for r in rows) / len(rows)
    # Fixed mismatch bins make the mechanism plot comparable across datasets.
    edges = (0.0, 0.1, 0.2, 0.3, 1.000001)
    result["mismatch_bins"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        chosen = [r for r in rows if lo <= abs(r["q"] - r["m"]) < hi]
        if not chosen:
            continue
        block = {"lower": lo, "upper": min(hi, 1.0), "count": len(chosen)}
        for name in variants:
            block[name] = aggregate_from_ranks([r[f"rank_{name}"] for r in chosen])
        result["mismatch_bins"].append(block)
    save_json(out / "metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
