#!/usr/bin/env python3
"""One-shot test evaluation for a validation-selected SeqRec checkpoint.

Training/tuning always uses ``--skip_test``.  This entry point refuses to
overwrite a previous test result, which makes the validation/test boundary
auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from refrec.data_utils import (FARRecDataset, collate_eval_batch,
                               compute_item_popularity, load_dataset,
                               load_or_build_cache)
from refrec.evaluate import evaluate_full_ranking, save_group_metrics, save_predictions
from refrec.run_seqrec import build_model
from refrec.utils import save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--decode_chunk", type=int, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out = run_dir / "test_metrics.json"
    if out.exists():
        raise FileExistsError(f"test was already evaluated: {out}")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    valid = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if "test_metrics" in valid:
        raise RuntimeError("training artifact already contains test metrics")

    device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data_dir = Path(cfg["data_dir"])
    data = load_dataset(data_dir)
    model = build_model(SimpleNamespace(**cfg), data, compute_item_popularity(data)).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    cp = {k: int(cfg[k]) for k in ("max_seq_len", "recent_window", "main_topk", "min_main_count")}
    samples = load_or_build_cache(data_dir, data, "test", **cp)
    loader = DataLoader(FARRecDataset(samples),
                        batch_size=args.eval_batch_size or int(cfg["eval_batch_size"]),
                        shuffle=False, collate_fn=collate_eval_batch)
    metrics, predictions, groups, timing = evaluate_full_ranking(
        model, loader, device, data.item_unmap, data.user_unmap, split="test",
        eval_chunk_size=args.decode_chunk or int(cfg["decode_chunk"]),
        show_progress=False, compute_group_metrics=True,
        mask_history=bool(cfg.get("eval_history_mask", False)),
    )
    save_predictions(run_dir / "test_predictions_top20.csv", predictions)
    if groups:
        save_group_metrics(run_dir / "test_groups", groups)
    result = {
        "dataset": cfg["dataset"], "backbone": cfg["backbone"], "seed": cfg["seed"],
        "checkpoint": str((run_dir / "best_model.pt").resolve()),
        "selected_by": "valid.ndcg@10", "best_epoch": valid["best_epoch"],
        "test_metrics": metrics, "timing": timing,
        "test_evaluated_once": True,
    }
    save_json(out, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
