#!/usr/bin/env python3
"""Training entry point for pluggable sequential backbones + the residual repeat head.

Reuses the exact same data pipeline, full-catalog evaluation, and repeat / new-target
decomposition protocol as ``run_refrec.py``; only the model is swapped via ``--backbone``.
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
from refrec.seqrec import GRU4Rec, SASRec, BERT4Rec, RepeatNet
from refrec.faesrec_adapter import FAESRecBackbone
from refrec.mamba4rec_adapter import Mamba4RecBackbone
from refrec.protocol import PROTOCOL, ensure_new_run, manifest
from refrec.train import train_one_epoch

BACKBONES = {"gru4rec", "sasrec", "bert4rec", "repeatnet", "faesrec", "mamba4rec"}
REPEAT_HEADS = {"base", "naive", "residual"}


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sequential backbone + residual repeat head")
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    p.add_argument("--data_root", default="data")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default="results/seqrec")
    p.add_argument("--run_name", default="default")
    p.add_argument("--backbone", choices=sorted(BACKBONES), default="sasrec")
    p.add_argument("--repeat_head", choices=sorted(REPEAT_HEADS), default="residual",
                   help="base = plain backbone; naive = B + w*R (double-counts); "
                        "residual = B + w*(R - beta*B) (the de-double-counting head).")
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--attn_hidden", type=int, default=64)
    p.add_argument("--faes_expert_num", type=int, default=4)
    p.add_argument("--faes_temperature", type=float, default=1.0)
    p.add_argument("--faes_cl_weight", type=float, default=1e-2)
    p.add_argument("--faes_l1_weight", type=float, default=1e-3)
    p.add_argument("--faes_d_state", type=int, default=32)
    p.add_argument("--faes_d_conv", type=int, default=4)
    p.add_argument("--faes_expand", type=int, default=2)
    p.add_argument("--mamba_layers", type=int, default=1)
    p.add_argument("--mamba_d_state", type=int, default=32)
    p.add_argument("--mamba_d_conv", type=int, default=4)
    p.add_argument("--mamba_expand", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=50)
    p.add_argument("--recent_window", type=int, default=5)
    p.add_argument("--main_topk", type=int, default=1)
    p.add_argument("--min_main_count", type=int, default=2)
    p.add_argument("--cat_weight", type=float, default=0.2)
    p.add_argument("--gamma_repeat", type=float, default=0.8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_neg", type=int, default=100)
    p.add_argument("--negative_sampling", choices=["unseen_only", "all_except_target", "stratified_history", "target_conditional"],
                   default="unseen_only")
    p.add_argument("--num_history_neg", type=int, default=20)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--decode_chunk", type=int, default=8192)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lambda_refrec_orth", type=float, default=0.0)
    p.add_argument("--refrec_init_repeat_weight", type=float, default=0.8)
    p.add_argument("--fix_beta", type=float, default=None)
    p.add_argument("--fix_scale", type=float, default=None)
    p.add_argument("--no_repeat_support", action="store_true")
    p.add_argument("--valid_subsample", type=int, default=0)
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--skip_test", action="store_true")
    p.add_argument("--eval_history_mask", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Resume this run from latest_checkpoint.pt if present.")
    return p


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--dataset")
    known, _ = pre.parse_known_args()
    parser = build_parser(); apply_dataset_defaults(parser, known.dataset)
    return parser.parse_args()


def cache_params(args: argparse.Namespace) -> dict[str, int]:
    return {"max_seq_len": args.max_seq_len, "recent_window": args.recent_window,
            "main_topk": args.main_topk, "min_main_count": args.min_main_count}


def build_model(args: argparse.Namespace, data, item_popularity: torch.Tensor):
    common = dict(
        num_items=data.num_items, num_categories=data.num_categories, item2cat=data.item2cat,
        item_popularity=item_popularity, emb_dim=args.emb_dim, cat_weight=args.cat_weight,
        max_seq_len=args.max_seq_len, dropout=args.dropout, gamma_repeat=args.gamma_repeat,
        repeat_head=args.repeat_head, refrec_init_repeat_weight=args.refrec_init_repeat_weight,
        no_repeat_support=args.no_repeat_support, fix_beta=args.fix_beta, fix_scale=args.fix_scale,
    )
    if args.backbone == "gru4rec":
        return GRU4Rec(**common, gru_layers=args.gru_layers)
    if args.backbone == "sasrec":
        return SASRec(**common, num_blocks=args.num_blocks, num_heads=args.num_heads,
                      attn_dropout=args.attn_dropout)
    if args.backbone == "bert4rec":
        return BERT4Rec(**common, num_blocks=args.num_blocks, num_heads=args.num_heads,
                        attn_dropout=args.attn_dropout)
    if args.backbone == "repeatnet":
        return RepeatNet(**common, gru_layers=args.gru_layers, attn_hidden=args.attn_hidden)
    if args.backbone == "faesrec":
        return FAESRecBackbone(
            **common, num_blocks=args.num_blocks, num_heads=args.num_heads,
            faes_expert_num=args.faes_expert_num,
            faes_temperature=args.faes_temperature,
            faes_cl_weight=args.faes_cl_weight,
            faes_l1_weight=args.faes_l1_weight,
            faes_d_state=args.faes_d_state, faes_d_conv=args.faes_d_conv,
            faes_expand=args.faes_expand,
        )
    if args.backbone == "mamba4rec":
        return Mamba4RecBackbone(
            **common, mamba_layers=args.mamba_layers,
            mamba_d_state=args.mamba_d_state, mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
        )
    raise ValueError(f"Unknown backbone: {args.backbone}")


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
    if not args.resume:
        ensure_new_run(out_dir)
    logger = TeeLogger(out_dir / "train.log", append=args.resume)
    started_at = timestamp(); t0 = time.perf_counter()
    logger.log(f"[{started_at}] {args.backbone}/{args.repeat_head} dataset={args.dataset} device={device}")
    logger.log(f"run_dir={out_dir.resolve()}")
    logger.log(f"protocol={PROTOCOL['version']} candidates=all_including_history selection=valid.ndcg@10")
    logger.log(f"command={' '.join(sys.argv)}")

    data = load_dataset(data_dir)
    item_popularity = compute_item_popularity(data)
    save_category_map(out_dir / "category_map.csv", data.cat_unmap)
    save_json(out_dir / "item2cat.json", {str(k): v for k, v in sorted(data.item2cat.items())})
    cp = cache_params(args)
    train_samples = load_or_build_cache(data_dir, data, "train", **cp)
    valid_samples = load_or_build_cache(data_dir, data, "valid", **cp)
    test_samples = load_or_build_cache(data_dir, data, "test", **cp)
    logger.log(f"samples train={len(train_samples)} valid={len(valid_samples)} test={len(test_samples)} "
               f"items={data.num_items} categories={data.num_categories}")

    config = args_to_dict(args)
    config.update({"run_dir": str(out_dir.resolve()), "data_dir": str(data_dir.resolve()),
                   "num_items": data.num_items, "num_categories": data.num_categories,
                   "num_users": data.num_users, "started_at": started_at,
                   "model_version": "seqrec", "evaluation_protocol": PROTOCOL})
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

    monitor_samples = valid_samples
    monitor_loader = valid_loader
    valid_subsample = max(0, int(args.valid_subsample))
    if 0 < valid_subsample < len(valid_samples):
        sel = np.random.default_rng(args.seed).choice(len(valid_samples), size=valid_subsample, replace=False)
        monitor_samples = [valid_samples[i] for i in np.sort(sel)]
        monitor_loader = DataLoader(FARRecDataset(monitor_samples), batch_size=args.eval_batch_size,
                                    shuffle=False, collate_fn=collate_eval_batch)

    model = build_model(args, data, item_popularity).to(device)
    logger.log(f"model params={model.count_parameters()} {model.complexity_note(args.num_neg)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score, best_epoch, stale, best_state, best_valid = -1.0, 0, 0, None, {}
    epoch_times: list[float] = []; valid_eval_time = 0.0
    start_epoch = 1
    checkpoint_path = out_dir / "latest_checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        stale = int(checkpoint["stale"])
        best_state = checkpoint["best_state"]
        best_valid = checkpoint["best_valid"]
        epoch_times = list(checkpoint.get("epoch_times", []))
        valid_eval_time = float(checkpoint.get("valid_eval_time", 0.0))
        start_epoch = int(checkpoint["epoch"]) + 1
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        rng.bit_generator.state = checkpoint["numpy_generator_state"]
        logger.log(f"resumed checkpoint epoch={start_epoch - 1} best_epoch={best_epoch}")
    train_t0 = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        stats = train_one_epoch(model, train_loader, optimizer, device, args, epoch, args.epochs, not args.no_progress)
        tv = time.perf_counter()
        valid_metrics, _, _, _ = evaluate_full_ranking(model, monitor_loader, device, data.item_unmap, data.user_unmap,
            split="valid_monitor", eval_chunk_size=args.decode_chunk, show_progress=not args.no_progress,
            compute_group_metrics=False, mask_history=args.eval_history_mask)
        valid_time = time.perf_counter() - tv; valid_eval_time += valid_time
        epoch_times.append(time.perf_counter() - epoch_t0)
        logger.log(f"epoch={epoch} train_loss={stats['train_loss']:.4f} orth_loss={stats['orth_loss']:.4f} "
                   f"valid_recall@10={valid_metrics.get('recall@10', 0):.4f} valid_ndcg@10={valid_metrics.get('ndcg@10', 0):.4f}")
        if valid_metrics["ndcg@10"] > best_score:
            best_score, best_epoch, stale = valid_metrics["ndcg@10"], epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; best_valid = dict(valid_metrics)
        else:
            stale += 1
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "best_score": best_score,
            "best_epoch": best_epoch, "stale": stale, "best_state": best_state,
            "best_valid": best_valid, "epoch_times": epoch_times,
            "valid_eval_time": valid_eval_time, "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(),
            "numpy_generator_state": rng.bit_generator.state,
        }, checkpoint_path)
        if stale >= args.patience:
            logger.log(f"early stop epoch={epoch} best_epoch={best_epoch}"); break

    train_time = time.perf_counter() - train_t0
    if best_state is not None: model.load_state_dict(best_state)
    torch.save(best_state or model.state_dict(), out_dir / "best_model.pt")
    monitor_best_score = best_score
    if monitor_samples is not valid_samples:
        best_valid, _, _, _ = evaluate_full_ranking(model, valid_loader, device, data.item_unmap, data.user_unmap,
            split="valid_full", eval_chunk_size=args.decode_chunk, show_progress=not args.no_progress,
            compute_group_metrics=False, mask_history=args.eval_history_mask)
        best_score = float(best_valid.get("ndcg@10", 0.0))
        logger.log(f"full valid at best_epoch={best_epoch}: ndcg@10={best_score:.6f}")
    if args.skip_test:
        save_json(out_dir / "metrics.json", {"seed": args.seed, "best_epoch": best_epoch,
            "valid_best_score": best_score, "valid_metrics": best_valid,
            "valid_monitor_best_score": monitor_best_score, "valid_subsample": valid_subsample,
            "evaluation_protocol": PROTOCOL})
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
        "valid_monitor_best_score": monitor_best_score, "valid_subsample": valid_subsample,
        "evaluation_protocol": PROTOCOL})
    save_json(out_dir / "speed_report.json", {"avg_epoch_time": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "total_train_time": train_time, "valid_eval_time": valid_eval_time, "test_eval_time": test_time,
        "users_per_second_test": test_timing["users_per_second"], "model_parameters": model.count_parameters(),
        "decode_chunk": args.decode_chunk, "complexity": model.complexity_note(args.num_neg),
        "wall_time": time.perf_counter() - t0})
    logger.log(f"test metrics: {test_metrics}"); logger.log(f"outputs -> {out_dir.resolve()}"); logger.close()


if __name__ == "__main__":
    main()
