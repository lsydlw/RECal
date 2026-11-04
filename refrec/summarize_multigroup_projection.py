#!/usr/bin/env python3
"""Aggregate three-group KL results into paper and appendix CSV files."""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

METHODS=('base','binary','three_group')
GROUPS=('recent','long_term','explore')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output_dir',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in sorted(root.glob('*/*/metrics.json')):
        x=json.loads(path.read_text(encoding='utf-8'))
        for method in METHODS:
            row={'dataset':x['dataset'],'backbone':x['backbone'],'seed':x['seed'],'method':method,
                 'selected_lambda':x.get('selected_lambda',''),'overall_ndcg@10':x[method]['overall']['ndcg@10'],
                 'group_mae':x[method]['group_mae']}
            for g in GROUPS:
                z=x[method]['groups'][g];row[f'{g}_count']=z['count'];row[f'{g}_ndcg@10']=z['metrics']['ndcg@10'] if z['metrics'] else ''
            records.append(row)
    expected=3*5*3*3
    if len(records)!=expected: raise RuntimeError(f'expected {expected} method records, found {len(records)}')
    fields=list(records[0]);
    with (out/'appendix_all_runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    grouped=defaultdict(list)
    for r in records: grouped[(r['dataset'],r['method'])].append(r)
    paper=[]
    for (ds,method),rows in sorted(grouped.items()):
        z={'dataset':ds,'method':method}
        for key in ('overall_ndcg@10','recent_ndcg@10','long_term_ndcg@10','explore_ndcg@10','group_mae'):
            vals=[float(r[key]) for r in rows if r[key]!=''];z[key]=float(np.mean(vals)) if vals else ''
        paper.append(z)
    with (out/'paper_table.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(paper[0]));w.writeheader();w.writerows(paper)
    checks={}
    for ds in sorted(set(r['dataset'] for r in paper)):
        d={r['method']:r for r in paper if r['dataset']==ds}
        checks[ds]={'three_vs_binary_overall':d['three_group']['overall_ndcg@10']-d['binary']['overall_ndcg@10'],
                    'three_vs_binary_long_term':d['three_group']['long_term_ndcg@10']-d['binary']['long_term_ndcg@10'],
                    'three_vs_binary_mae':d['three_group']['group_mae']-d['binary']['group_mae']}
    (out/'conclusion_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(f'wrote {len(records)} run-method rows and {len(paper)} paper rows to {out}')

if __name__=='__main__':main()
