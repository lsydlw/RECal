"""Self-contained evaluation/run protocol used by REFRec."""

from __future__ import annotations

import hashlib
from pathlib import Path


PROTOCOL = {
    "version": "repeat_full_v2",
    "candidates": "all_catalog_items_except_padding",
    "history_items": "included_without_target_exceptions",
    "tie_break": "score_desc_item_id_asc",
    "selection_split": "valid",
    "selection_metric": "ndcg@10",
    "catalog_assumption": "known_catalog_union_splits_and_metadata",
    "group_history": "full_available_prefix_not_only_encoder_window",
    "rank_export": "top20_censored_21_means_greater_than_20",
}


def merge_topk(running_scores, running_items, chunk_scores, chunk_items, k=20):
    """Stable score-descending, item-id-ascending top-k merge."""
    import torch
    scores = chunk_scores if running_scores is None else torch.cat([running_scores, chunk_scores], 1)
    items = chunk_items if running_items is None else torch.cat([running_items, chunk_items], 1)
    by_id = torch.argsort(items, dim=1, stable=True)
    scores, items = scores.gather(1, by_id), items.gather(1, by_id)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :min(k, scores.size(1))]
    return scores.gather(1, order), items.gather(1, order)


def ensure_new_run(path: Path) -> None:
    """Refuse to silently overwrite a prior run."""
    path = Path(path)
    if any((path / name).exists() for name in ("config.json", "metrics.json", "best_model.pt")):
        raise FileExistsError(f"Refusing to overwrite existing run: {path}. Choose a new --run_name.")


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def manifest(data_dir: Path, source_dirs: list[Path]) -> dict:
    """Write hashes when source/data files exist; tolerate legacy layouts."""
    data_dir = Path(data_dir)
    return {
        **PROTOCOL,
        "data_sha256": {name: _digest(data_dir / name)
                        for name in ("train.csv", "valid.csv", "test.csv", "item_meta.csv")},
        "source_sha256": {f"{Path(folder).name}/{p.name}": _digest(p)
                          for folder in source_dirs for p in sorted(Path(folder).glob("*.py"))},
        "protocol_source": "refrec/protocol.py",
    }
