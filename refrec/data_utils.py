"""FARRec 数据读取、兴趣状态预标注与缓存。"""

from __future__ import annotations

import csv
import hashlib
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


PADDING_ID = 0
CATEGORY_COLUMNS = (
    "category",
    "item_category",
    "cate_id",
    "genre",
    "beer_style",
    "cluster_id",
    "category_id",
)
CACHE_SUBDIR = "refrec_cache"


def _parse_category(raw: str) -> str:
    raw = str(raw).strip()
    if "|" in raw:
        return raw.split("|")[0].strip()
    return raw


def detect_category_column(fieldnames: list[str] | None) -> str:
    if not fieldnames:
        raise ValueError("item_meta.csv is empty or missing header")
    lower_map = {name.lower(): name for name in fieldnames}
    for col in CATEGORY_COLUMNS:
        if col in lower_map:
            return lower_map[col]
    raise ValueError(
        f"Cannot find category column in item_meta.csv. "
        f"Expected one of: {', '.join(CATEGORY_COLUMNS)}. Got: {fieldnames}"
    )


def load_interactions(path: Path) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["user_id"]), int(row["item_id"]), int(row["timestamp"])))
    rows.sort(key=lambda x: (x[0], x[2], x[1]))
    return rows


def load_item_meta(path: Path) -> dict[int, str]:
    meta: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cat_col = detect_category_column(reader.fieldnames)
        for row in reader:
            meta[int(row["item_id"])] = _parse_category(row[cat_col])
    return meta


def build_id_maps(
    train_rows: list[tuple[int, int, int]],
    valid_rows: list[tuple[int, int, int]],
    test_rows: list[tuple[int, int, int]],
    raw_item_meta: dict[int, str],
) -> dict[str, Any]:
    raw_users: set[int] = set()
    raw_items: set[int] = set()
    for uid, iid, _ in train_rows + valid_rows + test_rows:
        raw_users.add(uid)
        raw_items.add(iid)
    raw_items.update(raw_item_meta.keys())

    user_map = {uid: idx + 1 for idx, uid in enumerate(sorted(raw_users))}
    user_unmap = {v: k for k, v in user_map.items()}
    item_map = {iid: idx + 1 for idx, iid in enumerate(sorted(raw_items))}
    item_unmap = {v: k for k, v in item_map.items()}

    raw_cats: set[str] = set()
    for iid in raw_items:
        if iid in raw_item_meta:
            raw_cats.add(raw_item_meta[iid])
    cat_map = {cat: idx + 1 for idx, cat in enumerate(sorted(raw_cats))}
    cat_unmap = {v: k for k, v in cat_map.items()}

    item2cat: dict[int, int] = {}
    for raw_iid, model_iid in item_map.items():
        cat_str = raw_item_meta.get(raw_iid, "")
        item2cat[model_iid] = cat_map.get(cat_str, PADDING_ID)

    return {
        "user_map": user_map,
        "user_unmap": user_unmap,
        "item_map": item_map,
        "item_unmap": item_unmap,
        "cat_map": cat_map,
        "cat_unmap": cat_unmap,
        "item2cat": item2cat,
        "num_users": len(user_map),
        "num_items": len(item_map),
        "num_categories": len(cat_map),
    }


def save_category_map(path: Path, cat_unmap: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category_id", "category_raw"])
        for cid in sorted(cat_unmap.keys()):
            writer.writerow([cid, cat_unmap[cid]])


def left_pad_sequence(items: list[int], max_seq_len: int) -> list[int]:
    if len(items) >= max_seq_len:
        return items[-max_seq_len:]
    return [PADDING_ID] * (max_seq_len - len(items)) + items


def _main_categories(counter: Counter[int], main_topk: int, min_main_count: int) -> set[int]:
    if not counter:
        return set()
    ranked = counter.most_common()
    selected = {c for c, _ in ranked[: max(main_topk, 1)]}
    if all(counter[c] < min_main_count for c in selected):
        selected = {ranked[0][0]}
    return selected


def _repeat_ratio(items: list[int]) -> float:
    if not items:
        return 0.0
    return 1.0 - len(set(items)) / len(items)


def build_interest_state_labels(
    history_items: list[int],
    target_item: int,
    item2cat: dict[int, int],
    main_topk: int,
    min_main_count: int,
    recent_window: int,
) -> dict[str, Any]:
    valid_items = [i for i in history_items if i > PADDING_ID]
    history_cats = [item2cat.get(i, PADDING_ID) for i in valid_items]
    history_cats = [c for c in history_cats if c > PADDING_ID]

    cat_counter = Counter(history_cats)
    main_categories = _main_categories(cat_counter, main_topk, min_main_count)

    recent_items = valid_items[-recent_window:] if valid_items else []
    recent_categories = {item2cat.get(i, PADDING_ID) for i in recent_items}
    recent_categories.discard(PADDING_ID)

    history_cat_set = set(history_cats)
    weak_categories = history_cat_set - main_categories - recent_categories

    target_cat = item2cat.get(target_item, PADDING_ID)
    in_main = target_cat in main_categories
    in_recent = target_cat in recent_categories

    is_interest_shift = not in_main
    is_target_align_neither = (not in_main) and (not in_recent)
    is_seen_non_main = (
        target_cat in history_cat_set
        and target_cat not in main_categories
        and target_cat not in recent_categories
    )
    is_repeat_target = float(target_item in valid_items)
    is_weak_category = float(
        target_cat in history_cat_set
        and target_cat not in main_categories
        and target_cat not in recent_categories
    )

    if in_main and in_recent:
        group_target_alignment = "target_align_both"
    elif in_main:
        group_target_alignment = "target_align_long_only"
    elif in_recent:
        group_target_alignment = "target_align_recent_only"
    else:
        group_target_alignment = "target_align_neither"

    if in_main or in_recent:
        group_seen_unseen = "align_long_or_recent"
    elif target_cat in history_cat_set:
        group_seen_unseen = "seen_non_main"
    else:
        group_seen_unseen = "unseen"

    group_interest_shift = "interest_shift" if is_interest_shift else "interest_continuation"
    group_repeat = "repeat_target" if is_repeat_target > 0 else "non_repeat_target"
    group_weak_category = (
        "weak_category_target" if is_weak_category > 0 else "non_weak_category_target"
    )

    weak_count = sum(1 for c in history_cats if c in weak_categories)
    valid_len = max(len(valid_items), 1)
    weak_ratio = weak_count / valid_len
    num_weak_categories = float(len(weak_categories))
    weak_exists = 1.0 if (weak_count > 0 or len(weak_categories) > 0) else 0.0

    ent = 0.0
    if cat_counter:
        total = sum(cat_counter.values())
        for cnt in cat_counter.values():
            p = cnt / total
            ent -= p * math.log(p + 1e-12)
    category_entropy = ent

    repeat_ratio = _repeat_ratio(valid_items)
    recent_repeat_ratio = _repeat_ratio(recent_items)
    if valid_items:
        last_item = valid_items[-1]
        last_item_frequency = valid_items.count(last_item) / len(valid_items)
    else:
        last_item_frequency = 0.0

    return {
        "is_interest_shift": float(is_interest_shift),
        "is_target_align_neither": float(is_target_align_neither),
        "is_seen_non_main": float(is_seen_non_main),
        "is_repeat_target": is_repeat_target,
        "is_weak_category": is_weak_category,
        "group_interest_shift": group_interest_shift,
        "group_target_alignment": group_target_alignment,
        "group_seen_unseen": group_seen_unseen,
        "group_repeat": group_repeat,
        "group_weak_category": group_weak_category,
        "weak_categories": weak_categories,
        "weak_ratio": weak_ratio,
        "category_entropy": category_entropy,
        "num_weak_categories": num_weak_categories,
        "weak_exists": weak_exists,
        "target_cat": target_cat,
        "repeat_ratio": repeat_ratio,
        "recent_repeat_ratio": recent_repeat_ratio,
        "last_item_frequency": last_item_frequency,
    }


def _build_masks(
    seq_items: list[int],
    seq_cats: list[int],
    weak_categories: set[int],
    recent_window: int,
    max_seq_len: int,
) -> tuple[list[float], list[float], list[float]]:
    valid_mask = [1.0 if i > PADDING_ID else 0.0 for i in seq_items]
    weak_mask = [
        1.0 if (c in weak_categories and i > PADDING_ID) else 0.0
        for i, c in zip(seq_items, seq_cats)
    ]
    recent_mask = [0.0] * max_seq_len
    valid_indices = [idx for idx, v in enumerate(valid_mask) if v > 0]
    if valid_indices:
        for idx in valid_indices[-recent_window:]:
            recent_mask[idx] = 1.0
    return valid_mask, weak_mask, recent_mask


def annotate_sample(
    user_id: int,
    history: list[int],
    target_item: int,
    item2cat: dict[int, int],
    max_seq_len: int,
    recent_window: int,
    main_topk: int,
    min_main_count: int,
) -> dict[str, Any]:
    seq_items = left_pad_sequence(history, max_seq_len)
    seq_cats = [item2cat.get(i, PADDING_ID) for i in seq_items]
    labels = build_interest_state_labels(
        history, target_item, item2cat, main_topk, min_main_count, recent_window,
    )
    valid_mask, weak_mask, recent_mask = _build_masks(
        seq_items, seq_cats, labels["weak_categories"], recent_window, max_seq_len,
    )
    history_set = set(i for i in history if i > PADDING_ID)

    return {
        "user_id": user_id,
        "seq_items": seq_items,
        "seq_cats": seq_cats,
        "target_item": target_item,
        "target_cat": labels["target_cat"],
        "valid_mask": valid_mask,
        "weak_mask": weak_mask,
        "recent_mask": recent_mask,
        "is_interest_shift": labels["is_interest_shift"],
        "is_target_align_neither": labels["is_target_align_neither"],
        "is_seen_non_main": labels["is_seen_non_main"],
        "is_repeat_target": labels["is_repeat_target"],
        "is_weak_category": labels["is_weak_category"],
        "group_interest_shift": labels["group_interest_shift"],
        "group_target_alignment": labels["group_target_alignment"],
        "group_seen_unseen": labels["group_seen_unseen"],
        "group_repeat": labels["group_repeat"],
        "group_weak_category": labels["group_weak_category"],
        "weak_ratio": labels["weak_ratio"],
        "category_entropy": labels["category_entropy"],
        "num_weak_categories": labels["num_weak_categories"],
        "weak_exists": labels["weak_exists"],
        "repeat_ratio": labels["repeat_ratio"],
        "recent_repeat_ratio": labels["recent_repeat_ratio"],
        "last_item_frequency": labels["last_item_frequency"],
        "history_set": history_set,
    }


@dataclass
class DataBundle:
    user_train: dict[int, list[int]]
    user_train_times: dict[int, list[int]]
    user_valid: dict[int, int]
    user_valid_times: dict[int, int]
    user_test: dict[int, int]
    user_test_times: dict[int, int]
    item2cat: dict[int, int]
    item_unmap: dict[int, int]
    user_unmap: dict[int, int]
    cat_unmap: dict[int, str]
    num_users: int
    num_items: int
    num_categories: int
    user_map: dict[int, int]
    item_map: dict[int, int]


def load_dataset(data_dir: Path) -> DataBundle:
    train_rows = load_interactions(data_dir / "train.csv")
    valid_rows = load_interactions(data_dir / "valid.csv")
    test_rows = load_interactions(data_dir / "test.csv")
    raw_meta = load_item_meta(data_dir / "item_meta.csv")
    maps = build_id_maps(train_rows, valid_rows, test_rows, raw_meta)

    user_train: dict[int, list[int]] = defaultdict(list)
    user_train_times: dict[int, list[int]] = defaultdict(list)
    for uid, iid, ts in train_rows:
        if uid in maps["user_map"] and iid in maps["item_map"]:
            mapped_uid = maps["user_map"][uid]
            user_train[mapped_uid].append(maps["item_map"][iid])
            user_train_times[mapped_uid].append(ts)

    user_valid: dict[int, int] = {}
    user_valid_times: dict[int, int] = {}
    for uid, iid, ts in valid_rows:
        if uid in maps["user_map"] and iid in maps["item_map"]:
            mapped_uid = maps["user_map"][uid]
            user_valid[mapped_uid] = maps["item_map"][iid]
            user_valid_times[mapped_uid] = ts

    user_test: dict[int, int] = {}
    user_test_times: dict[int, int] = {}
    for uid, iid, ts in test_rows:
        if uid in maps["user_map"] and iid in maps["item_map"]:
            mapped_uid = maps["user_map"][uid]
            user_test[mapped_uid] = maps["item_map"][iid]
            user_test_times[mapped_uid] = ts

    return DataBundle(
        user_train=dict(user_train),
        user_train_times=dict(user_train_times),
        user_valid=user_valid,
        user_valid_times=user_valid_times,
        user_test=user_test,
        user_test_times=user_test_times,
        item2cat=maps["item2cat"],
        item_unmap=maps["item_unmap"],
        user_unmap=maps["user_unmap"],
        cat_unmap=maps["cat_unmap"],
        num_users=maps["num_users"],
        num_items=maps["num_items"],
        num_categories=maps["num_categories"],
        user_map=maps["user_map"],
        item_map=maps["item_map"],
    )


def compute_item_popularity(data: DataBundle) -> torch.Tensor:
    counts = torch.zeros(data.num_items + 1, dtype=torch.float32)
    for seq in data.user_train.values():
        for iid in seq:
            if 0 < iid <= data.num_items:
                counts[iid] += 1.0
    total = counts.sum().clamp_min(1.0)
    return torch.log1p(counts * data.num_items / total)


def cache_path(
    data_dir: Path,
    split: str,
    max_seq_len: int,
    recent_window: int,
    main_topk: int,
    min_main_count: int,
) -> Path:
    name = f"{split}_msl{max_seq_len}_rw{recent_window}_top{main_topk}_min{min_main_count}.pt"
    return data_dir / CACHE_SUBDIR / name


def _data_fingerprint(data_dir: Path) -> str:
    parts = []
    for fname in ("train.csv", "valid.csv", "test.csv", "item_meta.csv"):
        p = data_dir / fname
        if p.exists():
            parts.append(p.read_bytes())
    return hashlib.md5(b"".join(parts)).hexdigest()


def build_split_samples(
    data: DataBundle,
    split: str,
    max_seq_len: int,
    recent_window: int,
    main_topk: int,
    min_main_count: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if split == "train":
        for uid in sorted(data.user_train.keys()):
            seq = data.user_train[uid]
            for pos in range(1, len(seq)):
                history = seq[:pos]
                target = seq[pos]
                samples.append(annotate_sample(
                    uid, history, target, data.item2cat,
                    max_seq_len, recent_window, main_topk, min_main_count,
                ))
    elif split == "valid":
        for uid in sorted(data.user_valid.keys()):
            history = list(data.user_train[uid])
            target = data.user_valid[uid]
            samples.append(annotate_sample(
                uid, history, target, data.item2cat,
                max_seq_len, recent_window, main_topk, min_main_count,
            ))
    elif split == "test":
        for uid in sorted(data.user_test.keys()):
            history = list(data.user_train[uid]) + [data.user_valid[uid]]
            target = data.user_test[uid]
            samples.append(annotate_sample(
                uid, history, target, data.item2cat,
                max_seq_len, recent_window, main_topk, min_main_count,
            ))
    else:
        raise ValueError(f"unknown split: {split}")
    return samples


def _try_load_cache(path: Path, fp: str) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        cached = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    if cached.get("fingerprint") != fp or "samples" not in cached:
        return None
    return cached["samples"]


def _save_cache(path: Path, fp: str, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save({"fingerprint": fp, "samples": samples}, tmp)
    tmp.replace(path)


def load_or_build_cache(
    data_dir: Path,
    data: DataBundle,
    split: str,
    max_seq_len: int,
    recent_window: int,
    main_topk: int,
    min_main_count: int,
) -> list[dict[str, Any]]:
    path = cache_path(data_dir, split, max_seq_len, recent_window, main_topk, min_main_count)
    fp = _data_fingerprint(data_dir)
    samples = _try_load_cache(path, fp)
    if samples is not None:
        return samples
    samples = build_split_samples(
        data, split, max_seq_len, recent_window, main_topk, min_main_count,
    )
    _save_cache(path, fp, samples)
    return samples


class FARRecDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def sample_negatives(
    target: int,
    seen: set[int],
    num_items: int,
    num_neg: int,
    rng: np.random.Generator,
    mode: str = "unseen_only",
    num_history_neg: int = 20,
) -> list[int]:
    """Sample training negatives under an explicit history-exposure policy.

    ``unseen_only`` preserves the original REFRec behavior. ``all_except_target``
    samples uniformly from the inference catalogue, while ``stratified_history``
    reserves up to ``num_history_neg`` slots for items in the user's prefix and
    fills the rest with unseen items. ``target_conditional`` applies the
    stratified policy only to non-repeat targets and preserves unseen-only
    negatives for repeat targets. The positive target is never sampled,
    including when it is itself a repeat interaction.
    """
    if mode not in {"unseen_only", "all_except_target", "stratified_history", "target_conditional"}:
        raise ValueError(f"Unknown negative-sampling mode: {mode}")
    if num_neg < 0 or not 0 <= num_history_neg <= num_neg:
        raise ValueError("Require 0 <= num_history_neg <= num_neg")

    all_items = np.arange(1, num_items + 1, dtype=np.int64)
    is_repeat_target = target in seen
    valid_seen = {int(x) for x in seen if 1 <= int(x) <= num_items and int(x) != target}
    effective_mode = (
        "unseen_only" if mode == "target_conditional" and is_repeat_target
        else "stratified_history" if mode == "target_conditional"
        else mode
    )

    def draw(pool: np.ndarray, count: int) -> list[int]:
        if count == 0:
            return []
        if not len(pool):
            return []
        return rng.choice(pool, size=count, replace=len(pool) < count).tolist()

    if effective_mode == "all_except_target":
        candidates = all_items[all_items != target]
        negatives = draw(candidates, num_neg)
    else:
        unseen_mask = np.ones(num_items, dtype=bool)
        for item in valid_seen:
            unseen_mask[item - 1] = False
        if 1 <= target <= num_items:
            unseen_mask[target - 1] = False
        unseen_pool = all_items[unseen_mask]
        if effective_mode == "unseen_only":
            negatives = draw(unseen_pool, num_neg)
        else:
            history_pool = np.asarray(sorted(valid_seen), dtype=np.int64)
            history_count = min(num_history_neg, num_neg) if len(history_pool) else 0
            negatives = draw(history_pool, history_count)
            negatives.extend(draw(unseen_pool, num_neg - len(negatives)))
            if len(negatives) < num_neg:
                fallback = all_items[all_items != target]
                negatives.extend(draw(fallback, num_neg - len(negatives)))
            rng.shuffle(negatives)

    if len(negatives) != num_neg:
        raise ValueError("No eligible negative items; catalogue contains only the target")
    return [int(x) for x in negatives]


def collate_train_batch(
    batch: list[dict[str, Any]],
    num_items: int,
    num_neg: int,
    item2cat: dict[int, int],
    rng: np.random.Generator,
    negative_sampling: str = "unseen_only",
    num_history_neg: int = 20,
) -> dict[str, torch.Tensor]:
    bsz = len(batch)
    max_len = len(batch[0]["seq_items"])
    out: dict[str, torch.Tensor] = {
        "user_id": torch.zeros(bsz, dtype=torch.long),
        "seq_items": torch.zeros(bsz, max_len, dtype=torch.long),
        "seq_cats": torch.zeros(bsz, max_len, dtype=torch.long),
        "valid_mask": torch.zeros(bsz, max_len, dtype=torch.float32),
        "weak_mask": torch.zeros(bsz, max_len, dtype=torch.float32),
        "recent_mask": torch.zeros(bsz, max_len, dtype=torch.float32),
        "target_item": torch.zeros(bsz, dtype=torch.long),
        "target_cat": torch.zeros(bsz, dtype=torch.long),
        "sampled_items": torch.zeros(bsz, 1 + num_neg, dtype=torch.long),
        "sampled_cats": torch.zeros(bsz, 1 + num_neg, dtype=torch.long),
        "labels": torch.zeros(bsz, dtype=torch.long),
        "weak_loss_mask": torch.zeros(bsz, dtype=torch.float32),
    }
    for i, s in enumerate(batch):
        out["user_id"][i] = s["user_id"]
        out["seq_items"][i] = torch.tensor(s["seq_items"], dtype=torch.long)
        out["seq_cats"][i] = torch.tensor(s["seq_cats"], dtype=torch.long)
        out["valid_mask"][i] = torch.tensor(s["valid_mask"], dtype=torch.float32)
        out["weak_mask"][i] = torch.tensor(s["weak_mask"], dtype=torch.float32)
        out["recent_mask"][i] = torch.tensor(s["recent_mask"], dtype=torch.float32)
        target = s["target_item"]
        out["target_item"][i] = target
        out["target_cat"][i] = s["target_cat"]
        negs = sample_negatives(
            target, s["history_set"], num_items, num_neg, rng,
            mode=negative_sampling, num_history_neg=num_history_neg,
        )
        sampled = [target] + negs
        out["sampled_items"][i] = torch.tensor(sampled, dtype=torch.long)
        out["sampled_cats"][i] = torch.tensor(
            [item2cat.get(x, PADDING_ID) for x in sampled], dtype=torch.long,
        )
        if s["is_seen_non_main"] > 0 or s.get("is_weak_category", 0) > 0:
            out["weak_loss_mask"][i] = 1.0
    return out


def collate_eval_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    bsz = len(batch)
    max_len = len(batch[0]["seq_items"])
    seq_items = torch.zeros(bsz, max_len, dtype=torch.long)
    seq_cats = torch.zeros(bsz, max_len, dtype=torch.long)
    valid_mask = torch.zeros(bsz, max_len, dtype=torch.float32)
    weak_mask = torch.zeros(bsz, max_len, dtype=torch.float32)
    recent_mask = torch.zeros(bsz, max_len, dtype=torch.float32)
    targets = torch.zeros(bsz, dtype=torch.long)
    user_ids: list[int] = []
    history_sets: list[set[int]] = []
    groups: dict[str, list[str]] = {
        "group_interest_shift": [],
        "group_target_alignment": [],
        "group_seen_unseen": [],
        "group_repeat": [],
        "group_weak_category": [],
    }
    for i, s in enumerate(batch):
        seq_items[i] = torch.tensor(s["seq_items"], dtype=torch.long)
        seq_cats[i] = torch.tensor(s["seq_cats"], dtype=torch.long)
        valid_mask[i] = torch.tensor(s["valid_mask"], dtype=torch.float32)
        weak_mask[i] = torch.tensor(s["weak_mask"], dtype=torch.float32)
        recent_mask[i] = torch.tensor(s["recent_mask"], dtype=torch.float32)
        targets[i] = s["target_item"]
        user_ids.append(s["user_id"])
        history_sets.append(set(s["history_set"]))
        groups["group_interest_shift"].append(s["group_interest_shift"])
        groups["group_target_alignment"].append(s["group_target_alignment"])
        groups["group_seen_unseen"].append(s["group_seen_unseen"])
        groups["group_repeat"].append(s.get("group_repeat", "non_repeat_target"))
        groups["group_weak_category"].append(s.get("group_weak_category", "non_weak_category_target"))
    return {
        "user_id": torch.tensor(user_ids, dtype=torch.long),
        "seq_items": seq_items,
        "seq_cats": seq_cats,
        "valid_mask": valid_mask,
        "weak_mask": weak_mask,
        "recent_mask": recent_mask,
        "target_item": targets,
        "targets": targets,
        "history_sets": history_sets,
        **groups,
    }
