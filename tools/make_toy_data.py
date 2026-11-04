#!/usr/bin/env python3
"""Create a deterministic toy dataset for installation smoke tests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def write(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/toy"))
    args = parser.parse_args(); out = args.output
    train, valid, test = [], [], []
    for user in range(1, 17):
        base = (user * 3) % 24 + 1
        history = [((base + step * 5 - 1) % 30) + 1 for step in range(10)]
        if user % 2 == 0:
            history[7] = history[2]
        for step, item in enumerate(history[:8], 1):
            train.append({"user_id": user, "item_id": item, "timestamp": user * 100 + step})
        valid.append({"user_id": user, "item_id": history[8], "timestamp": user * 100 + 9})
        target = history[2] if user % 3 == 0 else history[9]
        test.append({"user_id": user, "item_id": target, "timestamp": user * 100 + 10})
    fields = ["user_id", "item_id", "timestamp"]
    write(out / "train.csv", fields, train)
    write(out / "valid.csv", fields, valid)
    write(out / "test.csv", fields, test)
    write(out / "item_meta.csv", ["item_id", "category_id"], [
        {"item_id": item, "category_id": (item - 1) % 6 + 1}
        for item in range(1, 31)
    ])
    print(out.resolve())


if __name__ == "__main__":
    main()

