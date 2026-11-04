#!/usr/bin/env python3
"""Compare one-sided and bidirectional RECal on external validation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--one_sided_root", required=True)
    p.add_argument("--bidirectional_root", required=True)
    args = p.parse_args()
    one_root = Path(args.one_sided_root)
    bi_root = Path(args.bidirectional_root)
    rows = []
    for path in sorted(one_root.glob("*/*/metrics.json")):
        one = json.loads(path.read_text(encoding="utf-8"))
        dataset, backbone = one["dataset"], one["backbone"]
        bi_path = bi_root / dataset / f"{dataset}_{backbone}_risk_seed2023" / "metrics.json"
        bi = json.loads(bi_path.read_text(encoding="utf-8"))
        asym = one["variants"]["asym12"]
        row = {"dataset": dataset, "backbone": backbone,
               "base_r10": bi["base"]["recall@10"],
               "one_r10": asym["overall"]["recall@10"],
               "bi_r10": bi["recal"]["recall@10"],
               "base_n10": bi["base"]["ndcg@10"],
               "one_n10": asym["overall"]["ndcg@10"],
               "bi_n10": bi["recal"]["ndcg@10"],
               "one_n10_rel": asym["overall"]["ndcg@10"] / bi["base"]["ndcg@10"] - 1,
               "bi_n10_rel": bi["recal"]["ndcg@10"] / bi["base"]["ndcg@10"] - 1,
               "repeat_one_n10": asym["repeat"]["ndcg@10"],
               "repeat_bi_n10": bi["groups"]["repeat"]["recal"]["ndcg@10"],
               "explore_one_n10": asym["explore"]["ndcg@10"],
               "explore_bi_n10": bi["groups"]["explore"]["recal"]["ndcg@10"]}
        rows.append(row)
    if not rows:
        raise SystemExit("No one-sided metrics found")
    output = one_root / "one_sided_vs_bidirectional.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
