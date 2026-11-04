#!/usr/bin/env python3
"""One-shot test evaluation for a frozen RECal head and base checkpoint."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from refrec.data_utils import compute_item_popularity, load_dataset, load_or_build_cache
from refrec.diagnose_repeat_mass import calibration_summary
from refrec.recal import RepeatIntentHead
from refrec.run_recal_frozen import evaluate_recal, rank_metrics
from refrec.run_seqrec import build_model
from refrec.utils import save_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--base_run_dir',required=True)
    p.add_argument('--recal_run_dir',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--output_dir',default=None,
                   help='Write final-test artifacts here without modifying validation runs.')
    a=p.parse_args(); base=Path(a.base_run_dir).resolve(); recal=Path(a.recal_run_dir).resolve()
    artifact=Path(a.output_dir).resolve() if a.output_dir else recal
    artifact.mkdir(parents=True,exist_ok=True)
    out=artifact/'test_metrics.json'
    if out.exists(): raise FileExistsError(f'test was already evaluated: {out}')
    cfg=json.loads((base/'config.json').read_text()); rm=json.loads((recal/'metrics.json').read_text())
    device=torch.device('cuda' if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    data=load_dataset(Path(cfg['data_dir']))
    model=build_model(SimpleNamespace(**cfg),data,compute_item_popularity(data)).to(device)
    model.load_state_dict(torch.load(base/'best_model.pt',map_location=device,weights_only=True)); model.eval()
    head=RepeatIntentHead(cfg['emb_dim'],hidden_dim=64,dropout=0.1).to(device)
    head.load_state_dict(torch.load(recal/'intent_head.pt',map_location=device,weights_only=True)); head.eval()
    cp={k:int(cfg[k]) for k in ('max_seq_len','recent_window','main_topk','min_main_count')}
    samples=load_or_build_cache(Path(cfg['data_dir']),data,'test',**cp)
    ea=SimpleNamespace(eval_batch_size=64,decode_chunk=int(cfg['decode_chunk']),
                       calibration_mode=rm['calibration_mode'],down_power=rm['down_power'],up_power=rm['up_power'])
    rows=evaluate_recal(model,head,samples,data,ea,device,cfg['recent_window'])
    with (artifact/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups={}
    for name,flag in (('repeat',1),('explore',0)):
        sub=[r for r in rows if r['is_repeat']==flag]
        groups[name]={'count':len(sub),'base':rank_metrics(sub,'base_rank'),'recal':rank_metrics(sub,'recal_rank')}
    result={'dataset':cfg['dataset'],'backbone':cfg['backbone'],'seed':cfg['seed'],
            'selected_by':'valid.ndcg@10','base':rank_metrics(rows,'base_rank'),
            'recal':rank_metrics(rows,'recal_rank'),'groups':groups,'test_evaluated_once':True}
    save_json(out,result); print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
