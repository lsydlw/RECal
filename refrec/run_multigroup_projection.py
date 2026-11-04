#!/usr/bin/env python3
"""Three-group KL projection experiment on a frozen sequential backbone.

Groups are recent-repeat, long-term-repeat, and explore.  The script trains a
small three-way intent head on training prefixes, selects one shrinkage value
on validation NDCG@10, and evaluates test exactly once.  Base, the existing
binary RECal head, and the three-group projection share the same catalog scores.
"""
from __future__ import annotations

import argparse, csv, json, math, random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from refrec.data_utils import (FARRecDataset, collate_eval_batch,
                               compute_item_popularity, load_dataset,
                               load_or_build_cache)
from refrec.metrics import aggregate_from_ranks
from refrec.recal import (MultiGroupIntentHead, RepeatIntentHead,
                          asymmetric_risk_trust, group_logit_shift,
                          multigroup_kl_shifts, sequence_statistics,
                          shrink_target_mass)
from refrec.run_seqrec import build_model
from refrec.utils import save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_run_dir', required=True)
    p.add_argument('--recal_run_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--hidden_dim', type=int, default=64); p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=512); p.add_argument('--encode_batch_size', type=int, default=256)
    p.add_argument('--eval_batch_size', type=int, default=64); p.add_argument('--decode_chunk', type=int, default=8192)
    p.add_argument('--lambdas', default='0.25,0.5,0.75,1.0')
    return p.parse_args()


def recent_set(seq, valid, window):
    vals = seq[valid > 0].tolist()
    return set(int(x) for x in vals[-window:] if int(x) > 0)


def label_for(target, history, recent):
    return 0 if target in recent else (1 if target in history else 2)


@torch.no_grad()
def encode(model, samples, device, batch_size, window):
    loader = DataLoader(FARRecDataset(samples), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_eval_batch)
    hs, zs, ys = [], [], []
    model.eval()
    for batch in loader:
        bt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        enc = model.encode(bt['seq_items'], bt['seq_cats'], bt['valid_mask'],
                           bt['weak_mask'], bt['recent_mask'])
        hs.append(enc['h_user'].cpu())
        zs.append(sequence_statistics(bt['seq_items'], bt['valid_mask'], window).cpu())
        for i, target in enumerate(batch['targets'].tolist()):
            rs = recent_set(batch['seq_items'][i], batch['valid_mask'][i], window)
            ys.append(label_for(target, batch['history_sets'][i], rs))
    return torch.cat(hs), torch.cat(zs), torch.tensor(ys, dtype=torch.long)


def train_head(head, train_t, valid_t, args, device):
    htr, ztr, ytr = train_t; hva, zva, yva = valid_t
    head.set_stats_normalization(ztr.to(device))
    loader = DataLoader(TensorDataset(htr, ztr, ytr), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, stale, best_epoch, state = math.inf, 0, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train(); total = count = 0
        for h, z, y in loader:
            h, z, y = h.to(device), z.to(device), y.to(device)
            opt.zero_grad(set_to_none=True); loss = F.cross_entropy(head(h, z), y)
            loss.backward(); opt.step(); total += float(loss) * len(y); count += len(y)
        head.eval()
        with torch.no_grad(): loss_v = float(F.cross_entropy(head(hva.to(device), zva.to(device)), yva.to(device)))
        history.append({'epoch': epoch, 'train_ce': total/count, 'valid_ce': loss_v})
        print(f'multigroup epoch={epoch} train_ce={total/count:.6f} valid_ce={loss_v:.6f}', flush=True)
        if loss_v < best - 1e-7:
            best, stale, best_epoch = loss_v, 0, epoch
            state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience: break
    head.load_state_dict(state)
    return best_epoch, history


def metrics(rows, key):
    return aggregate_from_ranks([int(x[key]) for x in rows])


@torch.no_grad()
def evaluate(model, binary_head, multi_head, samples, data, cfg, args, device, lambdas):
    loader = DataLoader(FARRecDataset(samples), batch_size=args.eval_batch_size,
                        shuffle=False, collate_fn=collate_eval_batch)
    items = torch.arange(1, data.num_items + 1, device=device)
    rows=[]; model.eval(); binary_head.eval(); multi_head.eval(); window=int(cfg['recent_window'])
    for batch in loader:
        bt={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        enc=model.encode(bt['seq_items'],bt['seq_cats'],bt['valid_mask'],bt['weak_mask'],bt['recent_mask'])
        enc=model.prepare_full_catalog(enc,bt['seq_items'],bt['valid_mask'],items,chunk_size=args.decode_chunk)
        stats=sequence_statistics(bt['seq_items'],bt['valid_mask'],window)
        q2=torch.sigmoid(binary_head(enc['h_user'],stats))
        q3=torch.softmax(multi_head(enc['h_user'],stats),dim=1)
        bsz=len(batch['history_sets']); logz=torch.full((bsz,),-torch.inf,device=device)
        logzg=torch.full((bsz,3),-torch.inf,device=device); target_score=torch.empty(bsz,device=device)
        recent=[]; longterm=[]; labels=[]
        for i,t in enumerate(batch['targets'].tolist()):
            rs=recent_set(batch['seq_items'][i],batch['valid_mask'][i],window)
            ls=set(batch['history_sets'][i])-rs; recent.append(rs); longterm.append(ls)
            labels.append(label_for(t,batch['history_sets'][i],rs))
        for start in range(0,data.num_items,args.decode_chunk):
            ids=items[start:start+args.decode_chunk]
            score=model.compute_base_score(enc,bt['seq_items'],bt['valid_mask'],ids[None,:].expand(bsz,-1))
            logz=torch.logaddexp(logz,torch.logsumexp(score,1))
            for i in range(bsz):
                masks=(torch.isin(ids,torch.as_tensor(sorted(recent[i]),device=device)),
                       torch.isin(ids,torch.as_tensor(sorted(longterm[i]),device=device)))
                masks=masks+(~(masks[0]|masks[1]),)
                for k,mask in enumerate(masks):
                    if mask.any(): logzg[i,k]=torch.logaddexp(logzg[i,k],torch.logsumexp(score[i,mask],0))
            inside=(bt['targets']>=ids[0])&(bt['targets']<=ids[-1]); rr=torch.nonzero(inside).squeeze(1)
            target_score[rr]=score[rr,bt['targets'][rr]-ids[0]]
        m3=torch.exp(logzg-logz[:,None]); m3=torch.where(torch.isfinite(m3),m3,torch.zeros_like(m3))
        available=m3>1e-12; q3=q3*available; q3=q3/q3.sum(1,keepdim=True).clamp_min(1e-12)
        mh=m3[:,:2].sum(1); trust=asymmetric_risk_trust(mh,q2,12.0)
        th=shrink_target_mass(mh,q2,trust); d2=group_logit_shift(th,mh)
        binary_mass=torch.zeros_like(m3); ratio=m3[:,:2]/mh[:,None].clamp_min(1e-12)
        binary_mass[:,:2]=ratio*th[:,None]; binary_mass[:,2]=1-th
        targets={lam:m3+lam*(q3-m3) for lam in lambdas}
        shifts={lam:multigroup_kl_shifts(t,m3) for lam,t in targets.items()}
        methods=['base','binary']+[f'multi_{lam:g}' for lam in lambdas]
        greater={x:torch.zeros(bsz,dtype=torch.long,device=device) for x in methods}
        equal={x:torch.zeros(bsz,dtype=torch.long,device=device) for x in methods}
        label_t=torch.tensor(labels,device=device); target_binary=target_score+d2*(label_t<2)
        target_multi={lam:target_score+shifts[lam].gather(1,label_t[:,None]).squeeze(1) for lam in lambdas}
        for start in range(0,data.num_items,args.decode_chunk):
            ids=items[start:start+args.decode_chunk]
            base=model.compute_base_score(enc,bt['seq_items'],bt['valid_mask'],ids[None,:].expand(bsz,-1))
            scores={'base':base}; bin_score=base.clone()
            for i in range(bsz):
                hm=torch.isin(ids,torch.as_tensor(sorted(batch['history_sets'][i]),device=device)); bin_score[i,hm]+=d2[i]
            scores['binary']=bin_score
            for lam in lambdas:
                sc=base.clone()
                for i in range(bsz):
                    rm=torch.isin(ids,torch.as_tensor(sorted(recent[i]),device=device)); lm=torch.isin(ids,torch.as_tensor(sorted(longterm[i]),device=device))
                    gm=(rm,lm,~(rm|lm))
                    for k,mask in enumerate(gm): sc[i,mask]+=shifts[lam][i,k]
                scores[f'multi_{lam:g}']=sc
            target_scores={'base':target_score,'binary':target_binary,**{f'multi_{x:g}':target_multi[x] for x in lambdas}}
            for name in methods:
                greater[name]+=(scores[name]>target_scores[name][:,None]).sum(1)
                equal[name]+=((scores[name]==target_scores[name][:,None])&(ids[None,:]<bt['targets'][:,None])).sum(1)
        onehot=F.one_hot(label_t,3).float()
        for i in range(bsz):
            row={'user_id':data.user_unmap.get(int(bt['user_id'][i]),int(bt['user_id'][i])),
                 'target_group':('recent','long_term','explore')[labels[i]],
                 'base_group_mae':float((m3[i]-onehot[i]).abs().mean()),
                 'binary_group_mae':float((binary_mass[i]-onehot[i]).abs().mean())}
            for name in methods: row[f'{name}_rank']=int(1+greater[name][i]+equal[name][i])
            for lam in lambdas: row[f'multi_{lam:g}_group_mae']=float((targets[lam][i]-onehot[i]).abs().mean())
            rows.append(row)
    return rows


def summarize(rows,key,mae_key):
    out={'overall':metrics(rows,key),'group_mae':float(np.mean([r[mae_key] for r in rows])),'groups':{}}
    for g in ('recent','long_term','explore'):
        sub=[r for r in rows if r['target_group']==g]
        out['groups'][g]={'count':len(sub),'metrics':metrics(sub,key) if sub else None}
    return out


def main():
    args=parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device('cuda' if args.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    base_dir=Path(args.base_run_dir); recal_dir=Path(args.recal_run_dir); out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'non-empty output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((base_dir/'config.json').read_text(encoding='utf-8')); rm=json.loads((recal_dir/'metrics.json').read_text(encoding='utf-8'))
    data=load_dataset(Path(cfg['data_dir'])); model=build_model(SimpleNamespace(**cfg),data,compute_item_popularity(data)).to(device)
    model.load_state_dict(torch.load(base_dir/'best_model.pt',map_location=device,weights_only=True)); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    binary=RepeatIntentHead(cfg['emb_dim'],hidden_dim=64,dropout=.1).to(device)
    binary.load_state_dict(torch.load(recal_dir/'intent_head.pt',map_location=device,weights_only=True)); binary.eval()
    cp={k:int(cfg[k]) for k in ('max_seq_len','recent_window','main_topk','min_main_count')}
    train=load_or_build_cache(Path(cfg['data_dir']),data,'train',**cp); valid=load_or_build_cache(Path(cfg['data_dir']),data,'valid',**cp)
    test=load_or_build_cache(Path(cfg['data_dir']),data,'test',**cp)
    tr=encode(model,train,device,args.encode_batch_size,cp['recent_window']); va=encode(model,valid,device,args.encode_batch_size,cp['recent_window'])
    head=MultiGroupIntentHead(cfg['emb_dim'],hidden_dim=args.hidden_dim,dropout=args.dropout).to(device)
    best_epoch,history=train_head(head,tr,va,args,device); torch.save(head.state_dict(),out/'multigroup_intent_head.pt')
    lambdas=[float(x) for x in args.lambdas.split(',')]
    valid_rows=evaluate(model,binary,head,valid,data,cfg,args,device,lambdas)
    valid_scores={x:metrics(valid_rows,f'multi_{x:g}_rank')['ndcg@10'] for x in lambdas}
    selected=max(lambdas,key=lambda x:(valid_scores[x],-x))
    test_rows=evaluate(model,binary,head,test,data,cfg,args,device,[selected])
    key=f'multi_{selected:g}'
    result={'dataset':cfg['dataset'],'backbone':cfg['backbone'],'seed':args.seed,'recent_window':cp['recent_window'],
            'selected_lambda':selected,'selection_split':'valid','selection_metric':'ndcg@10','valid_lambda_ndcg@10':valid_scores,
            'base':summarize(test_rows,'base_rank','base_group_mae'),
            'binary':summarize(test_rows,'binary_rank','binary_group_mae'),
            'three_group':summarize(test_rows,f'{key}_rank',f'{key}_group_mae'),
            'groups':['recent','long_term','explore'],'test_evaluated_once':True}
    with (out/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(test_rows[0]));w.writeheader();w.writerows(test_rows)
    save_json(out/'metrics.json',result);save_json(out/'training_history.json',{'best_epoch':best_epoch,'epochs':history})
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
