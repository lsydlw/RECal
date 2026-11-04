#!/usr/bin/env python3
"""Validate the four-CSV dataset contract without loading model code."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CATEGORY_COLUMNS = (
    "category", "item_category", "cate_id", "genre",
    "beer_style", "cluster_id", "category_id",
)


def read_interactions(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "item_id", "timestamp"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, raw in enumerate(reader, 2):
            try:
                rows.append((int(raw["user_id"]), int(raw["item_id"]), int(float(raw["timestamp"]))))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: IDs/timestamp must be numeric") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()
    root = args.dataset_dir

    splits = {name: read_interactions(root / f"{name}.csv")
              for name in ("train", "valid", "test")}
    meta_path = root / "item_meta.csv"
    with meta_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "item_id" not in fields:
            raise ValueError(f"{meta_path}: missing item_id")
        category = next((name for name in CATEGORY_COLUMNS if name in fields), None)
        if category is None:
            raise ValueError(f"{meta_path}: missing a recognized category column")
        meta_items = {int(row["item_id"]) for row in reader}

    by_split_user = {}
    all_items = set()
    all_users = set()
    for split, rows in splits.items():
        grouped = defaultdict(list)
        for user, item, timestamp in rows:
            grouped[user].append((timestamp, item))
            all_users.add(user); all_items.add(item)
        for user, events in grouped.items():
            if events != sorted(events, key=lambda row: (row[0], row[1])):
                raise ValueError(f"{split}.csv: user {user} is not sorted by timestamp/item")
        by_split_user[split] = grouped

    chronology_errors = []
    for user in all_users:
        train = by_split_user["train"].get(user, [])
        valid = by_split_user["valid"].get(user, [])
        test = by_split_user["test"].get(user, [])
        if train and valid and train[-1][0] > valid[0][0]: chronology_errors.append(user)
        if valid and test and valid[-1][0] > test[0][0]: chronology_errors.append(user)
    if chronology_errors:
        raise ValueError(f"cross-split chronology violated for users {sorted(set(chronology_errors))[:10]}")
    missing_meta = sorted(all_items.difference(meta_items))
    if missing_meta:
        raise ValueError(f"item_meta.csv misses {len(missing_meta)} interacted items; first={missing_meta[:10]}")

    summary = {
        "dataset_dir": str(root.resolve()),
        "users": len(all_users),
        "items_in_interactions": len(all_items),
        "items_in_metadata": len(meta_items),
        "category_column": category,
        "rows": {name: len(rows) for name, rows in splits.items()},
        "status": "ok",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

