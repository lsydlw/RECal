#!/usr/bin/env python3
"""Aggregate the eight validation-only RECal shrinkage scans."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/recal_v1/shrinkage_scan")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/recal_v1/shrinkage_summary.csv")
runs = [json.loads(p.read_text()) for p in sorted(root.glob("*/*/metrics.json"))]
if not runs: raise SystemExit("no scan metrics found")
variants = list(runs[0]["variants"])
rows=[]
for x in runs:
    base=x["variants"]["g0"]
    for v in variants:
        z=x["variants"][v]
        row={"dataset":x["dataset"],"backbone":x["backbone"],"variant":v,
             "overall_n10":z["overall"]["ndcg@10"],"overall_r10":z["overall"]["recall@10"],
             "overall_n10_rel":z["overall"]["ndcg@10"]/max(base["overall"]["ndcg@10"],1e-12)-1,
             "repeat_n10":z["repeat"]["ndcg@10"],"explore_n10":z["explore"]["ndcg@10"],
             "repeat_rel":z["repeat"]["ndcg@10"]/max(base["repeat"]["ndcg@10"],1e-12)-1 if base["repeat"]["ndcg@10"] else "",
             "explore_rel":z["explore"]["ndcg@10"]/max(base["explore"]["ndcg@10"],1e-12)-1 if base["explore"]["ndcg@10"] else ""}
        rows.append(row)
out.parent.mkdir(parents=True,exist_ok=True)
with out.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print("variant,mean_rel,improved_or_tied,worst_rel,mean_n10")
for v in variants:
    rr=[r for r in rows if r["variant"]==v]
    rel=[float(r["overall_n10_rel"]) for r in rr]
    print(f"{v},{sum(rel)/len(rel):.6f},{sum(x>=-1e-12 for x in rel)}/{len(rel)},{min(rel):.6f},{sum(float(r['overall_n10']) for r in rr)/len(rr):.6f}")
print("\nper-run NDCG@10")
print("dataset,backbone,"+",".join(variants))
for x in runs:
    print(x["dataset"]+","+x["backbone"]+","+",".join(f"{x['variants'][v]['overall']['ndcg@10']:.6f}" for v in variants))
print(out.resolve())
