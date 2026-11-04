#!/usr/bin/env python3
"""Merge the 4-backbone RECal results with SASRec comparison baselines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BACKBONES = ("bert4rec", "gru4rec", "repeatnet", "sasrec")
LABELS = {
    "bert4rec": "BERT4Rec", "gru4rec": "GRU4Rec",
    "repeatnet": "RepeatNet", "sasrec": "SASRec",
    "hard_mask": "SASRec+HardMask",
    "calibrated_recommendation": "SASRec+CalRec",
    "prm": "SASRec+PRM", "recode": "SASRec+ReCODE-style",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--main", default="results/recal_asymmetric_8datasets_v1/multiseed_test_summary.csv")
    p.add_argument("--comparisons", default="results/comparison_baselines_8d3s_v1/three_seed_summary.csv")
    p.add_argument("--output", default="results/combined_recal_results_8d3s.csv")
    return p.parse_args()


def read_csv(path, encoding="utf-8"):
    with Path(path).open(newline="", encoding=encoding) as handle:
        return list(csv.DictReader(handle))


def fmt(mean, std):
    return f"{float(mean):.4f}±{float(std):.4f}"


def main():
    args = parse_args()
    main_rows = read_csv(args.main, "utf-8-sig")
    comp_rows = [r for r in read_csv(args.comparisons) if r["group"] == "overall"]
    out = []
    for row in main_rows:
        dataset, backbone = row["dataset"], row["backbone"]
        out.append({
            "dataset": dataset, "configuration": LABELS[backbone], "family": "backbone",
            "recall@10_mean": row["base_recall@10_mean"], "recall@10_std": row["base_recall@10_std"],
            "ndcg@10_mean": row["base_ndcg@10_mean"], "ndcg@10_std": row["base_ndcg@10_std"],
        })
        out.append({
            "dataset": dataset, "configuration": LABELS[backbone] + "+RECal", "family": "recal",
            "recall@10_mean": row["recal_recall@10_mean"], "recall@10_std": row["recal_recall@10_std"],
            "ndcg@10_mean": row["recal_ndcg@10_mean"], "ndcg@10_std": row["recal_ndcg@10_std"],
        })
    for row in comp_rows:
        if row["method"] == "base":
            continue  # identical to the SASRec base already imported above
        out.append({
            "dataset": row["dataset"], "configuration": LABELS[row["method"]], "family": "comparison",
            "recall@10_mean": row["recall@10_mean"], "recall@10_std": row["recall@10_std"],
            "ndcg@10_mean": row["ndcg@10_mean"], "ndcg@10_std": row["ndcg@10_std"],
        })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0])); writer.writeheader(); writer.writerows(out)
    order = []
    for backbone in BACKBONES:
        order.extend((LABELS[backbone], LABELS[backbone] + "+RECal"))
    order.extend(("SASRec+HardMask", "SASRec+CalRec", "SASRec+PRM", "SASRec+ReCODE-style"))
    datasets = sorted({r["dataset"] for r in out})
    lookup = {(r["dataset"], r["configuration"]): r for r in out}
    for metric in ("ndcg@10", "recall@10"):
        print(f"\n{metric.upper()}")
        print("dataset\t" + "\t".join(order))
        for dataset in datasets:
            values = [fmt(lookup[(dataset, name)][f"{metric}_mean"],
                          lookup[(dataset, name)][f"{metric}_std"]) for name in order]
            print(dataset + "\t" + "\t".join(values))


if __name__ == "__main__":
    main()
