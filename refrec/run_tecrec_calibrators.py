#!/usr/bin/env python3
"""Evaluate lightweight calibration controls and a TecRec-style adapter.

The frozen backbone and full-catalogue candidate protocol are shared by every
method. Hyperparameters are selected on validation only; test is written once.
"""
from __future__ import annotations

import argparse, csv, json, math, random, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from refrec.data_utils import collate_eval_batch, compute_item_popularity, load_dataset, load_or_build_cache
from refrec.metrics import aggregate_from_ranks
from refrec.protocol import PROTOCOL, merge_topk
from refrec.run_seqrec import build_model
from refrec.run_comparison_baselines import js_divergence, grouped_metrics, temporal_records, to_device
from refrec.utils import save_json


class TastePredictor(nn.Module):
    """Small category-GRU predicting the next-interaction taste distribution."""
    def __init__(self, num_categories: int, hidden: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_categories + 1, hidden, padding_idx=0)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, num_categories + 1)

    def forward(self, cats, valid_mask):
        lengths = valid_mask.long().sum(1).clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(cats), lengths.cpu(),
                                                    batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.output(h[-1])


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_dir", required=True); p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--batch_size", type=int, default=128); p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--decode_chunk", type=int, default=4096); p.add_argument("--rerank_size", type=int, default=100)
    p.add_argument("--taste_epochs", type=int, default=8); p.add_argument("--taste_patience", type=int, default=3)
    p.add_argument("--taste_hidden", type=int, default=32); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_samples", type=int, default=100000)
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.5,0.75,1.0,1.5,2.0,3.0])
    p.add_argument("--logit_shifts", nargs="+", type=float, default=[-3,-2,-1,-0.5,0,0.5,1,2,3])
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.01,0.03,0.05,0.1,0.2,0.5])
    return p.parse_args()


def train_taste(model, train, valid, item2cat, args, device):
    if args.train_samples and len(train) > args.train_samples:
        idx = np.sort(np.random.default_rng(args.seed).choice(len(train), args.train_samples, replace=False))
        train = [train[i] for i in idx]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best, state, stale, history = math.inf, None, 0, []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.taste_epochs + 1):
        model.train(); total = seen = 0
        order = rng.permutation(len(train))
        for start in range(0, len(train), args.batch_size):
            ids = order[start:start+args.batch_size]
            batch = to_device(collate_eval_batch([train[i] for i in ids]), device)
            labels = torch.tensor([item2cat.get(int(x),0) for x in batch["targets"].tolist()], device=device)
            opt.zero_grad(set_to_none=True); loss = F.cross_entropy(model(batch["seq_cats"], batch["valid_mask"]), labels)
            loss.backward(); opt.step(); total += float(loss)*len(ids); seen += len(ids)
        model.eval(); vtotal = vn = 0
        with torch.no_grad():
            for start in range(0, len(valid), args.batch_size):
                batch = to_device(collate_eval_batch(valid[start:start+args.batch_size]), device)
                labels = torch.tensor([item2cat.get(int(x),0) for x in batch["targets"].tolist()], device=device)
                loss = F.cross_entropy(model(batch["seq_cats"], batch["valid_mask"]), labels)
                vtotal += float(loss)*len(labels); vn += len(labels)
        row = {"epoch":epoch,"train_ce":total/max(seen,1),"valid_ce":vtotal/max(vn,1)}
        history.append(row); print("taste", row, flush=True)
        if row["valid_ce"] < best-1e-6:
            best=row["valid_ce"]; stale=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.taste_patience: break
    model.load_state_dict(state); return history


def rerank_profile(items, scores, profile, item2cat, weight):
    scores=np.asarray(scores,dtype=np.float64); relevance=(scores-scores.min())/max(scores.max()-scores.min(),1e-12)
    item_array=np.asarray(items,dtype=np.int64); cats=np.asarray([item2cat.get(i,0) for i in items],dtype=np.int64)
    remaining=np.arange(len(items)); selected=[]; slate=np.zeros_like(profile)
    for _ in range(min(20,len(items))):
        penalties={}
        for cat in np.unique(cats[remaining]):
            trial=slate.copy()
            if cat>0: trial[cat]+=1
            penalties[int(cat)]=js_divergence(profile,trial/max(trial.sum(),1))
        values=(1-weight)*relevance[remaining]-weight*np.asarray([penalties[int(c)] for c in cats[remaining]])
        local=np.lexsort((item_array[remaining],-values))[0]; chosen=int(remaining[local]); selected.append(items[chosen])
        if cats[chosen]>0: slate[cats[chosen]]+=1
        remaining=np.delete(remaining,local)
    return selected


def history_profile(history,item2cat,ncat):
    p=np.zeros(ncat+1,dtype=np.float64)
    for i in history:
        c=item2cat.get(int(i),0)
        if c>0:p[c]+=1
    if p.sum()==0: p[1:]=1/max(ncat,1)
    else:p/=p.sum()
    return p


def top_rank(items,target):
    try:return items.index(target)+1
    except ValueError:return 21


@torch.no_grad()
def evaluate(base,taste,samples,data,args,device,grid=True,selected=None):
    base.eval();taste.eval(); all_items=torch.arange(1,data.num_items+1,device=device); rows=[]
    # The test pass must retain the unmodified Base (shift=0) as well as the
    # validation-selected shift; otherwise Base cannot be reconstructed when
    # the selected value is non-zero.
    shifts=args.logit_shifts if grid else list(dict.fromkeys([0.0, selected["shift"]]))
    lambdas=args.lambdas if grid else [selected["cal_lambda"],selected["tec_lambda"]]
    temps=args.temperatures if grid else [selected["temperature"]]
    nll={t:0.0 for t in temps}
    for begin in range(0,len(samples),args.eval_batch_size):
        sb=samples[begin:begin+args.eval_batch_size]; batch=collate_eval_batch(sb); bt=to_device(batch,device); bsz=len(sb)
        enc=base.encode(bt["seq_items"],bt["seq_cats"],bt["valid_mask"],bt["weak_mask"],bt["recent_mask"])
        enc=base.prepare_full_catalog(enc,bt["seq_items"],bt["valid_mask"],all_items,chunk_size=args.decode_chunk)
        targets=bt["targets"]; target_score=base.compute_base_score(enc,bt["seq_items"],bt["valid_mask"],targets[:,None]).squeeze(1)
        target_hist=torch.tensor([int(int(t) in h) for t,h in zip(targets.tolist(),batch["history_sets"])],device=device,dtype=torch.float32)
        greater={d:torch.zeros(bsz,dtype=torch.long,device=device) for d in shifts}; equal={d:torch.zeros(bsz,dtype=torch.long,device=device) for d in shifts}
        lse={t:None for t in temps}; top_scores=top_items=None
        for off in range(0,data.num_items,args.decode_chunk):
            ids=all_items[off:off+args.decode_chunk]; scores=base.compute_base_score(enc,bt["seq_items"],bt["valid_mask"],ids[None,:].expand(bsz,-1))
            top_scores,top_items=merge_topk(top_scores,top_items,scores,ids[None,:].expand(bsz,-1),args.rerank_size)
            hist=torch.stack([torch.isin(ids,torch.as_tensor(sorted(h),device=device)) for h in batch["history_sets"]]).float()
            for d in shifts:
                shifted=scores+d*hist; ts=target_score+d*target_hist
                greater[d]+=(shifted>ts[:,None]).sum(1); equal[d]+=((shifted==ts[:,None])&(ids[None,:]<targets[:,None])).sum(1)
            for t in temps:
                block=torch.logsumexp(scores/t,dim=1); lse[t]=block if lse[t] is None else torch.logaddexp(lse[t],block)
        taste_p=F.softmax(taste(bt["seq_cats"],bt["valid_mask"]),dim=-1).cpu().numpy()
        for t in temps:nll[t]+=float((lse[t]-target_score/t).sum())
        for r,sample in enumerate(sb):
            items=top_items[r].tolist(); scores=top_scores[r].tolist(); target=int(targets[r]); row={"user_id":int(bt["user_id"][r]),"group":"repeat" if target in sample["history_set"] else "explore"}
            for d in shifts:row[f"rank_shift_{d:g}"]=int(1+greater[d][r]+equal[d][r])
            hp=history_profile(sample["history_set"],data.item2cat,data.num_categories)
            for w in lambdas:
                row[f"rank_cal_{w:g}"]=top_rank(rerank_profile(items,scores,hp,data.item2cat,w),target)
                row[f"rank_tec_{w:g}"]=top_rank(rerank_profile(items,scores,taste_p[r],data.item2cat,w),target)
            rows.append(row)
        print(f"evaluated {min(begin+bsz,len(samples))}/{len(samples)}",flush=True)
    return rows,{t:nll[t]/len(samples) for t in temps}


def metrics(rows,key):
    return {"overall":grouped_metrics(rows,key),"repeat":grouped_metrics(rows,key,"repeat"),"explore":grouped_metrics(rows,key,"explore")}


def main():
    args=args_parser(); random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() else "cpu"); base_dir=Path(args.base_run_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/"test_metrics.json").exists():raise FileExistsError(f"write-once test exists: {out}")
    cfg=json.loads((base_dir/"config.json").read_text(encoding="utf-8"));data=load_dataset(Path(cfg["data_dir"]));base=build_model(SimpleNamespace(**cfg),data,compute_item_popularity(data)).to(device)
    base.load_state_dict(torch.load(base_dir/"best_model.pt",map_location=device,weights_only=True));base.eval()
    for p in base.parameters():p.requires_grad_(False)
    cp={k:int(cfg[k]) for k in ("max_seq_len","recent_window","main_topk","min_main_count")};train=load_or_build_cache(Path(cfg["data_dir"]),data,"train",**cp);valid=load_or_build_cache(Path(cfg["data_dir"]),data,"valid",**cp);test=load_or_build_cache(Path(cfg["data_dir"]),data,"test",**cp)
    taste=TastePredictor(data.num_categories,args.taste_hidden).to(device);history=train_taste(taste,train,valid,data.item2cat,args,device);torch.save(taste.state_dict(),out/"taste_predictor.pt")
    vrows,vnll=evaluate(base,taste,valid,data,args,device,True)
    shift=max(args.logit_shifts,key=lambda d:(aggregate_from_ranks([r[f"rank_shift_{d:g}"] for r in vrows])["ndcg@10"],-abs(d)))
    cal=max(args.lambdas,key=lambda w:(aggregate_from_ranks([r[f"rank_cal_{w:g}"] for r in vrows])["ndcg@10"],-w))
    tec=max(args.lambdas,key=lambda w:(aggregate_from_ranks([r[f"rank_tec_{w:g}"] for r in vrows])["ndcg@10"],-w));temp=min(args.temperatures,key=lambda t:vnll[t]);selected={"shift":shift,"cal_lambda":cal,"tec_lambda":tec,"temperature":temp}
    trows,tnll=evaluate(base,taste,test,data,args,device,False,selected)
    result={"base":metrics(trows,f"rank_shift_0"),"temperature_scaling":metrics(trows,f"rank_shift_0"),"global_logit_shift":metrics(trows,f"rank_shift_{shift:g}"),"calrec":metrics(trows,f"rank_cal_{cal:g}"),"tecrec_style":metrics(trows,f"rank_tec_{tec:g}")}
    with (out/"test_ranks.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(trows[0]));w.writeheader();w.writerows(trows)
    save_json(out/"config.json",vars(args)|{"dataset":cfg["dataset"],"backbone":cfg["backbone"],"base_run_dir":str(base_dir.resolve()),"selected":selected,"protocol":PROTOCOL,"test_evaluated_once":True,"tecrec_note":"unified-protocol category-GRU taste prediction plus calibrated Top-100 reranking"})
    save_json(out/"valid_selection.json",{"selected":selected,"temperature_nll":vnll,"taste_training":history})
    save_json(out/"test_metrics.json",{"dataset":cfg["dataset"],"seed":cfg["seed"],"methods":result,"selected_by":"validation only","test_temperature_nll":tnll})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
