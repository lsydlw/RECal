#!/usr/bin/env python3
"""Freeze validation-selected checkpoints before any test evaluation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--recal_root', required=True); p.add_argument('--output', required=True)
    a=p.parse_args(); root=Path(a.root); out=Path(a.output)
    if out.exists(): raise FileExistsError(f'frozen manifest already exists: {out}')
    rows=[]
    for m in sorted(root.glob('*/*/metrics.json')):
        run=m.parent; cfg=json.loads((run/'config.json').read_text()); met=json.loads(m.read_text())
        if 'test_metrics' in met or (run/'test_metrics.json').exists():
            raise RuntimeError(f'test contamination before freeze: {run}')
        ck=run/'best_model.pt'
        rr=Path(a.recal_root)/cfg['dataset']/f"{cfg['dataset']}_{cfg['backbone']}_recal_seed{cfg['seed']}"
        if not (rr/'metrics.json').exists() or not (rr/'intent_head.pt').exists():
            raise RuntimeError(f'missing RECal validation artifact: {rr}')
        if (rr/'test_metrics.json').exists(): raise RuntimeError(f'RECal test contamination before freeze: {rr}')
        rows.append({'dataset':cfg['dataset'],'backbone':cfg['backbone'],'seed':cfg['seed'],
                     'run_dir':str(run.resolve()),'best_epoch':met['best_epoch'],
                     'valid_ndcg@10':met['valid_metrics']['ndcg@10'],
                     'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
                     'recal_run_dir':str(rr.resolve()),
                     'intent_head_sha256':hashlib.sha256((rr/'intent_head.pt').read_bytes()).hexdigest()})
    expected=4*4*3
    if len(rows)!=expected: raise RuntimeError(f'require {expected} complete validation runs, found {len(rows)}')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'protocol':'repeat_full_v2','selection':'valid.ndcg@10',
                               'test_seen':False,'runs':rows},indent=2),encoding='utf-8')
    print(out.resolve())
if __name__=='__main__': main()
