"""Full ranking 评估与预测导出。"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch

from refrec.metrics import (
    aggregate_from_ranks,
    aggregate_group_metrics,
    per_sample_metrics,
    rank_in_list,
    save_group_metrics_csv,
)
from refrec.utils import tqdm_bar
from refrec.protocol import PROTOCOL, merge_topk


def _merge_topk(
    running_scores: torch.Tensor | None,
    running_items: torch.Tensor | None,
    chunk_scores: torch.Tensor,
    chunk_items: torch.Tensor,
    k: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    return merge_topk(running_scores, running_items, chunk_scores, chunk_items, k)


@torch.no_grad()
def evaluate_full_ranking(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    item_unmap: dict[int, int],
    user_unmap: dict[int, int],
    split: str = "eval",
    eval_chunk_size: int = 8192,
    show_progress: bool = True,
    compute_group_metrics: bool = False,
    mask_history: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    model.eval()
    if eval_chunk_size < 1:
        raise ValueError("eval_chunk_size must be positive")
    num_items = model.num_items
    all_items = torch.arange(1, num_items + 1, dtype=torch.long, device=device)
    ranks: list[int] = []
    predictions: list[dict[str, Any]] = []
    group_samples: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    score_time = 0.0
    num_users = 0

    pbar = tqdm_bar(loader, disable=not show_progress, desc=f"{split} [full rank]", unit="batch")
    for batch in pbar:
        batch_t = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bsz = batch_t["seq_items"].size(0)

        enc = model.encode(
            batch_t["seq_items"], batch_t["seq_cats"], batch_t["valid_mask"],
            batch_t["weak_mask"], batch_t["recent_mask"],
        )
        enc = model.prepare_full_catalog(
            enc, batch_t["seq_items"], batch_t["valid_mask"], all_items,
            chunk_size=eval_chunk_size,
        )

        running_scores: torch.Tensor | None = None
        running_items: torch.Tensor | None = None
        t_score = time.perf_counter()
        for start in range(0, len(all_items), eval_chunk_size):
            chunk_items = all_items[start:start + eval_chunk_size]
            chunk_scores = model.score_chunk(
                enc, batch_t["seq_items"], batch_t["valid_mask"], chunk_items,
            )
            if mask_history:
                # Diagnostic hard filter over the complete available prefix,
                # not merely the max_seq_len encoder window.
                for row, history in enumerate(batch["history_sets"]):
                    if history:
                        history_tensor = torch.as_tensor(list(history), device=device)
                        chunk_scores[row].masked_fill_(torch.isin(chunk_items, history_tensor), float("-inf"))
            chunk_items_b = chunk_items.unsqueeze(0).expand(bsz, -1)
            running_scores, running_items = _merge_topk(
                running_scores, running_items, chunk_scores, chunk_items_b,
            )
        score_time += time.perf_counter() - t_score

        assert running_items is not None
        targets = batch_t["targets"].tolist()
        for i in range(bsz):
            topk_model = running_items[i].cpu().tolist()
            target_model = targets[i]
            rank = rank_in_list(topk_model, target_model)
            ranks.append(rank)

            topk_raw = [item_unmap.get(mid, mid - 1) for mid in topk_model]
            target_raw = item_unmap.get(target_model, target_model - 1)
            uid_model = batch_t["user_id"][i].item()
            uid_raw = user_unmap.get(uid_model, uid_model)

            predictions.append({
                "user_id": uid_raw,
                "target_item": target_raw,
                "rank": rank,
                "target_in_top20": int(rank <= 20),
                "topk_items": topk_raw,
            })
            if compute_group_metrics:
                group_samples.append({
                    "group_interest_shift": batch["group_interest_shift"][i],
                    "group_target_alignment": batch["group_target_alignment"][i],
                    "group_seen_unseen": batch["group_seen_unseen"][i],
                    "group_repeat": batch["group_repeat"][i],
                    "group_weak_category": batch["group_weak_category"][i],
                    "metrics": per_sample_metrics(topk_model, target_model),
                })
        num_users += bsz
        pbar.set_postfix(users=num_users, items=num_items)

    elapsed = time.perf_counter() - t0
    timing = {
        "eval_time": elapsed,
        "score_time": score_time,
        "num_users": num_users,
        "users_per_second": num_users / max(elapsed, 1e-9),
    }
    return aggregate_from_ranks(ranks), predictions, group_samples, timing


def save_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "target_item", "rank", "target_in_top20", "top20_items"])
        for pred in predictions:
            writer.writerow([
                pred["user_id"],
                pred["target_item"],
                pred["rank"],
                pred["target_in_top20"],
                " ".join(str(x) for x in pred["topk_items"]),
            ])


def save_group_metrics(output_dir: Path, group_samples: list[dict[str, Any]]) -> None:
    save_group_metrics_csv(
        output_dir / "group_metrics_interest_shift.csv",
        aggregate_group_metrics(group_samples, "group_interest_shift"),
    )
    save_group_metrics_csv(
        output_dir / "group_metrics_target_alignment.csv",
        aggregate_group_metrics(group_samples, "group_target_alignment"),
    )
    save_group_metrics_csv(
        output_dir / "group_metrics_seen_unseen.csv",
        aggregate_group_metrics(group_samples, "group_seen_unseen"),
    )
    save_group_metrics_csv(
        output_dir / "group_metrics_repeat.csv",
        aggregate_group_metrics(group_samples, "group_repeat"),
    )
    save_group_metrics_csv(
        output_dir / "group_metrics_weak_category.csv",
        aggregate_group_metrics(group_samples, "group_weak_category"),
    )
