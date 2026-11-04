#!/usr/bin/env python3
"""Fair Base-versus-RECal inference benchmark on identical full-catalog batches."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from refrec.data_utils import (FARRecDataset, collate_eval_batch,
                               compute_item_popularity, load_dataset,
                               load_or_build_cache)
from refrec.recal import (RepeatIntentHead, asymmetric_risk_trust,
                          group_logit_shift, sequence_statistics,
                          shrink_target_mass)
from refrec.run_seqrec import build_model
from refrec.utils import save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run_dir", required=True)
    parser.add_argument("--recal_run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--decode_chunk", type=int, default=8192)
    parser.add_argument("--max_users", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--down_power", type=float, default=12.0)
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run_once(model, head, batches, all_items, cfg, args, device, use_recal):
    users = 0
    checksum = 0
    for batch in batches:
        bt = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}
        enc = model.encode(bt["seq_items"], bt["seq_cats"], bt["valid_mask"],
                           bt["weak_mask"], bt["recent_mask"])
        enc = model.prepare_full_catalog(enc, bt["seq_items"], bt["valid_mask"],
                                         all_items, chunk_size=args.decode_chunk)
        score_parts = []
        for start in range(0, len(all_items), args.decode_chunk):
            ids = all_items[start:start + args.decode_chunk]
            score_parts.append(model.compute_base_score(
                enc, bt["seq_items"], bt["valid_mask"],
                ids[None, :].expand(len(batch["history_sets"]), -1)))
        scores = torch.cat(score_parts, dim=1)
        if use_recal:
            q = torch.sigmoid(head(enc["h_user"], sequence_statistics(
                bt["seq_items"], bt["valid_mask"], cfg["recent_window"])))
            row_tensor = bt["history_row_indices"]
            item_tensor = bt["history_item_indices"]
            log_z = torch.logsumexp(scores, dim=1)
            log_zh = torch.full_like(log_z, -torch.inf)
            if row_tensor.numel():
                history_scores = scores[row_tensor, item_tensor]
                history_max = torch.full_like(log_z, -torch.inf)
                history_max.scatter_reduce_(0, row_tensor, history_scores,
                                            reduce="amax", include_self=True)
                history_sum = torch.zeros_like(log_z)
                history_sum.scatter_add_(
                    0, row_tensor,
                    torch.exp(history_scores - history_max[row_tensor]))
                present = history_sum > 0
                log_zh[present] = (history_max[present] +
                                   torch.log(history_sum[present]))
            m = torch.exp(log_zh - log_z)
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            trust = asymmetric_risk_trust(m, q, args.down_power)
            delta = group_logit_shift(shrink_target_mass(m, q, trust), m)
            if row_tensor.numel():
                scores[row_tensor, item_tensor] += delta[row_tensor]
        top = torch.topk(scores, k=min(args.topk, scores.shape[1]), dim=1).indices
        checksum += int(top[:, 0].sum().item())
        users += scores.shape[0]
    return users, checksum


def benchmark(label, model, head, batches, all_items, cfg, args, device, use_recal):
    for _ in range(args.warmup):
        run_once(model, head, batches, all_items, cfg, args, device, use_recal)
    synchronize(device)
    times = []
    peak = 0
    users = 0
    checksum = 0
    for _ in range(args.repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        start = time.perf_counter()
        users, checksum = run_once(model, head, batches, all_items, cfg, args,
                                   device, use_recal)
        synchronize(device)
        times.append(time.perf_counter() - start)
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device))
    mean = statistics.mean(times)
    return {
        "mode": label, "users": users, "repeats": args.repeats,
        "seconds_mean": mean,
        "seconds_std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "milliseconds_per_user": 1000.0 * mean / users,
        "users_per_second": users / mean,
        "peak_allocated_mib": peak / (1024 ** 2),
        "checksum": checksum,
    }


def main():
    args = parse_args()
    base_dir = Path(args.base_run_dir)
    recal_dir = Path(args.recal_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if args.device.startswith("cuda") and
                          torch.cuda.is_available() else "cpu")
    data = load_dataset(Path(cfg["data_dir"]))
    model = build_model(SimpleNamespace(**cfg), data,
                        compute_item_popularity(data)).to(device)
    model.load_state_dict(torch.load(base_dir / "best_model.pt",
                                     map_location=device, weights_only=True))
    model.eval()
    head = RepeatIntentHead(cfg["emb_dim"], hidden_dim=64, dropout=0.1).to(device)
    head.load_state_dict(torch.load(recal_dir / "intent_head.pt",
                                    map_location=device, weights_only=True))
    head.eval()
    cp = {k: int(cfg[k]) for k in ("max_seq_len", "recent_window",
                                   "main_topk", "min_main_count")}
    samples = load_or_build_cache(Path(cfg["data_dir"]), data, "test", **cp)
    samples = samples[:min(len(samples), args.max_users)]
    loader = DataLoader(FARRecDataset(samples), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_eval_batch,
                        pin_memory=device.type == "cuda")
    batches = list(loader)
    # Candidate-group membership is request metadata, not model computation.
    # Convert it to a reusable sparse representation once, just as production
    # serving systems cache a user's interacted-item IDs.
    for batch in batches:
        row_ids = []
        item_ids = []
        for row, history in enumerate(batch["history_sets"]):
            ids = [int(i) - 1 for i in history if 0 < int(i) <= data.num_items]
            row_ids.extend([row] * len(ids)); item_ids.extend(ids)
        batch["history_row_indices"] = torch.tensor(row_ids, dtype=torch.long)
        batch["history_item_indices"] = torch.tensor(item_ids, dtype=torch.long)
    all_items = torch.arange(1, data.num_items + 1, device=device)
    base_params = sum(p.numel() for p in model.parameters())
    recal_params = sum(p.numel() for p in head.parameters())
    base = benchmark("base", model, head, batches, all_items, cfg, args,
                     device, False)
    recal = benchmark("recal", model, head, batches, all_items, cfg, args,
                      device, True)
    result = {
        "dataset": cfg["dataset"], "backbone": cfg["backbone"],
        "seed": cfg["seed"], "full_catalogue": True,
        "catalogue_items": data.num_items, "batch_size": args.batch_size,
        "max_users": args.max_users, "warmup": args.warmup,
        "base_parameters": base_params, "recal_extra_parameters": recal_params,
        "parameter_overhead_percent": 100.0 * recal_params / base_params,
        "base": base, "recal": recal,
        "latency_overhead_percent": 100.0 * (recal["seconds_mean"] / base["seconds_mean"] - 1.0),
        "peak_memory_overhead_mib": recal["peak_allocated_mib"] - base["peak_allocated_mib"],
    }
    save_json(output_dir / "efficiency.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
