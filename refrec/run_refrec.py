#!/usr/bin/env python3
"""Standalone training entry point for REFRec.

Model-specific code, arguments and loss all live in ``refrec/``.  The shared
FARRec modules imported here are only dataset parsing, full-catalog evaluation,
and generic output helpers.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refrec.data_utils import (FARRecDataset, collate_eval_batch, collate_train_batch,
                               compute_item_popularity, load_dataset, load_or_build_cache,
                               save_category_map)
from refrec.evaluate import evaluate_full_ranking, save_group_metrics, save_predictions
from refrec.utils import (SUPPORTED_DATASETS, TeeLogger, apply_dataset_defaults,
                           args_to_dict, resolve_run_dir, save_json, timestamp)
from refrec.model import REFRec
from refrec.protocol import PROTOCOL, ensure_new_run, manifest
from refrec.train import train_one_epoch


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="REFRec training")
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    p.add_argument("--data_root", default="data")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default="results/refrec")
    p.add_argument("--run_name", default="default")
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--max_seq_len", type=int, default=50)
    p.add_argument("--recent_window", type=int, default=5)
    p.add_argument("--main_topk", type=int, default=1)
    p.add_argument("--min_main_count", type=int, default=2)
    p.add_argument("--cat_weight", type=float, default=0.2)
    p.add_argument("--gamma_weak", type=float, default=0.8)
    p.add_argument("--gamma_repeat", type=float, default=0.8)
    p.add_argument("--alpha_main", type=float, default=1.0)
    p.add_argument("--alpha_recent", type=float, default=1.0)
    p.add_argument("--cat_smooth_k", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_neg", type=int, default=100)
    p.add_argument("--negative_sampling", choices=["unseen_only", "all_except_target", "stratified_history", "target_conditional"],
                   default="unseen_only",
                   help="Training-negative policy; evaluation candidates are controlled separately by the protocol.")
    p.add_argument("--num_history_neg", type=int, default=20,
                   help="History-negative slots used by stratified_history (the total remains --num_neg).")
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--decode_chunk", type=int, default=8192)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lambda_refrec_orth", type=float, default=0.0)
    p.add_argument("--refrec_init_repeat_weight", type=float, default=0.20)
    p.add_argument("--repeat_mode", choices=["none", "raw", "residual"], default="residual")
    p.add_argument("--disable_repeat_expert", action="store_true", default=False)
    p.add_argument("--disable_recent_expert", action="store_true", default=False)
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--skip_test", action="store_true")
    p.add_argument("--eval_history_mask", action="store_true",
                   help="Diagnostic: hard-mask every item in the full available history during validation/test.")
    p.add_argument("--fix_beta", type=float, default=None,
                   help="Ablation: fix residual beta to a constant (None=learned).")
    p.add_argument("--fix_scale", type=float, default=None,
                   help="Ablation: fix residual scale to a constant (None=learned).")
    p.add_argument("--no_repeat_support", action="store_true",
                   help="Ablation: disable repeat-support mask (apply residual to all candidates).")
    p.add_argument("--fuse_mode", choices=["residual", "softmax", "residual_full"], default="residual",
                   help="softmax = FARRec 4-expert softmax (double-counting baseline); "
                        "residual = repeat-blind softmax base + additive repeat residual (REFRec); "
                        "residual_full = Gram-Schmidt residual chain over recent/long/weak + repeat residual.")
    p.add_argument("--cosine_base", action="store_true",
                   help="L2-normalize the recent/long query and item vectors before the base dot "
                        "products (turns base experts into cosine similarities).")
    p.add_argument("--use_gru", action="store_true",
                   help="Replace the long-term expert's mean-pooling readout with a GRU sequence "
                        "encoder (learned long-interest representation).")
    return p


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--dataset")
    known, _ = pre.parse_known_args()
    parser = build_parser(); apply_dataset_defaults(parser, known.dataset)
    return parser.parse_args()


def cache_params(args: argparse.Namespace) -> dict[str, int]:
    return {"max_seq_len": args.max_seq_len, "recent_window": args.recent_window,
            "main_topk": args.main_topk, "min_main_count": args.min_main_count}


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if not 0 <= args.num_history_neg <= args.num_neg:
        raise ValueError("Require 0 <= --num_history_neg <= --num_neg")
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / args.dataset
    out_dir = resolve_run_dir(args.output_dir, args.dataset, args.run_name)
    ensure_new_run(out_dir)
    logger = TeeLogger(out_dir / "train.log")
    started_at = timestamp(); t0 = time.perf_counter()
    logger.log(f"[{started_at}] REFRec dataset={args.dataset} device={device}")
    logger.log(f"run_dir={out_dir.resolve()}")
    candidate_mode = "all_except_full_history" if args.eval_history_mask else "all_including_history"
    logger.log(f"protocol={PROTOCOL['version']} candidates={candidate_mode} selection=valid.ndcg@10")
    history_slots = args.num_history_neg if args.negative_sampling in {"stratified_history", "target_conditional"} else 0
    logger.log(f"training_negatives={args.negative_sampling} num_neg={args.num_neg} history_slots={history_slots}")
    logger.log(f"command={' '.join(sys.argv)}")

    data = load_dataset(data_dir)
    item_popularity = compute_item_popularity(data)
    save_category_map(out_dir / "category_map.csv", data.cat_unmap)
    save_json(out_dir / "item2cat.json", {str(k): v for k, v in sorted(data.item2cat.items())})
    cp = cache_params(args)
    train_samples = load_or_build_cache(data_dir, data, "train", **cp)
    valid_samples = load_or_build_cache(data_dir, data, "valid", **cp)
    test_samples = load_or_build_cache(data_dir, data, "test", **cp)
    logger.log(f"samples train={len(train_samples)} valid={len(valid_samples)} test={len(test_samples)} items={data.num_items} categories={data.num_categories}")

    config = args_to_dict(args)
    config.update({"run_dir": str(out_dir.resolve()), "data_dir": str(data_dir.resolve()),
                   "num_items": data.num_items, "num_categories": data.num_categories,
                   "num_users": data.num_users, "started_at": started_at,
                   "model_version": "refrec", "evaluation_protocol": PROTOCOL})
    save_json(out_dir / "config.json", config)
    save_json(out_dir / "evaluation_protocol.json", manifest(data_dir, [ROOT / "refrec"]))

    rng = np.random.default_rng(args.seed)
    train_loader = DataLoader(FARRecDataset(train_samples), batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_train_batch(
            b, data.num_items, args.num_neg, data.item2cat, rng,
            negative_sampling=args.negative_sampling, num_history_neg=args.num_history_neg,
        ), num_workers=0)
    valid_loader = DataLoader(FARRecDataset(valid_samples), batch_size=args.eval_batch_size, shuffle=False,
                              collate_fn=collate_eval_batch)
    test_loader = DataLoader(FARRecDataset(test_samples), batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=collate_eval_batch)

    model = REFRec(num_items=data.num_items, num_categories=data.num_categories, item2cat=data.item2cat,
        item_popularity=item_popularity, emb_dim=args.emb_dim, cat_weight=args.cat_weight,
        max_seq_len=args.max_seq_len, dropout=args.dropout, gamma_weak=args.gamma_weak,
        gamma_repeat=args.gamma_repeat, alpha_main=args.alpha_main, alpha_recent=args.alpha_recent,
        cat_smooth_k=args.cat_smooth_k, disable_repeat_expert=args.disable_repeat_expert,
        disable_recent_expert=args.disable_recent_expert,
        repeat_mode=("none" if args.disable_repeat_expert else args.repeat_mode),
        fuse_mode=args.fuse_mode,
        cosine_base=args.cosine_base,
        use_gru=args.use_gru,
        refrec_init_repeat_weight=args.refrec_init_repeat_weight,
        fix_beta=args.fix_beta, fix_scale=args.fix_scale,
        no_repeat_support=args.no_repeat_support).to(device)
    logger.log(f"model params={model.count_parameters()} {model.complexity_note(args.num_neg)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score, best_epoch, stale, best_state, best_valid = -1.0, 0, 0, None, {}
    epoch_times: list[float] = []; valid_eval_time = 0.0
    train_t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        stats = train_one_epoch(model, train_loader, optimizer, device, args, epoch, args.epochs, not args.no_progress)
        tv = time.perf_counter()
        valid_metrics, _, _, _ = evaluate_full_ranking(model, valid_loader, device, data.item_unmap, data.user_unmap,
            split="valid", eval_chunk_size=args.decode_chunk, show_progress=not args.no_progress,
            compute_group_metrics=False, mask_history=args.eval_history_mask)
        valid_time = time.perf_counter() - tv; valid_eval_time += valid_time
        epoch_times.append(time.perf_counter() - epoch_t0)
        logger.log(f"epoch={epoch} train_loss={stats['train_loss']:.4f} orth_loss={stats['orth_loss']:.4f} valid_recall@10={valid_metrics.get('recall@10', 0):.4f} valid_ndcg@10={valid_metrics.get('ndcg@10', 0):.4f}")
        if valid_metrics["ndcg@10"] > best_score:
            best_score, best_epoch, stale = valid_metrics["ndcg@10"], epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; best_valid = dict(valid_metrics)
        else:
            stale += 1
            if stale >= args.patience:
                logger.log(f"early stop epoch={epoch} best_epoch={best_epoch}"); break

    train_time = time.perf_counter() - train_t0
    if best_state is not None: model.load_state_dict(best_state)
    torch.save(best_state or model.state_dict(), out_dir / "best_model.pt")
    if args.skip_test:
        save_json(out_dir / "metrics.json", {"seed": args.seed, "best_epoch": best_epoch,
            "valid_best_score": best_score, "valid_metrics": best_valid, "evaluation_protocol": PROTOCOL})
        logger.log(f"valid metrics: {best_valid}; test not evaluated (tuning)"); logger.close(); return

    tt = time.perf_counter()
    test_metrics, predictions, group_samples, test_timing = evaluate_full_ranking(model, test_loader, device, data.item_unmap, data.user_unmap,
        split="test", eval_chunk_size=args.decode_chunk, show_progress=not args.no_progress,
        compute_group_metrics=True, mask_history=args.eval_history_mask)
    test_time = time.perf_counter() - tt
    save_predictions(out_dir / "predictions_top20.csv", predictions)
    if group_samples: save_group_metrics(out_dir, group_samples)
    save_json(out_dir / "metrics.json", {"seed": args.seed, "best_epoch": best_epoch,
        "valid_best_score": best_score, "valid_metrics": best_valid, "test_metrics": test_metrics,
        "evaluation_protocol": PROTOCOL})
    save_json(out_dir / "speed_report.json", {"avg_epoch_time": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "total_train_time": train_time, "valid_eval_time": valid_eval_time, "test_eval_time": test_time,
        "users_per_second_test": test_timing["users_per_second"], "model_parameters": model.count_parameters(),
        "decode_chunk": args.decode_chunk, "complexity": model.complexity_note(args.num_neg),
        "wall_time": time.perf_counter() - t0})
    logger.log(f"test metrics: {test_metrics}"); logger.log(f"outputs -> {out_dir.resolve()}"); logger.close()


if __name__ == "__main__":
    main()
