#!/usr/bin/env python3
"""Staged training: GraphMAE pretrain -> mild DANN/CORAL adapt -> detector threshold tuning.
Goal: keep AUC while improving recall/F1.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path('data/processed/downstream/downstream_v2_informative.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l=l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l*g, None


def grl(x,l=1.0): return GRL.apply(x,l)


def coral(xs, xt):
    xs = xs - xs.mean(0, keepdim=True); xt = xt - xt.mean(0, keepdim=True)
    cs = (xs.t() @ xs) / max(xs.size(0)-1,1)
    ct = (xt.t() @ xt) / max(xt.size(0)-1,1)
    return ((cs-ct)**2).mean()


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64))
        self.dec = nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,d))
        self.cls = nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        self.reg = nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        self.dom = nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self, x, l=0.0):
        z = self.enc(x)
        xh = self.dec(z)
        yc = self.cls(z).squeeze(-1)
        yr = self.reg(z).squeeze(-1)
        yd = self.dom(grl(z,l)).squeeze(-1)
        return z, xh, yc, yr, yd


def standardize(Xs, Xt):
    mu = Xs.mean(0, keepdims=True); sd = Xs.std(0, keepdims=True)+1e-6
    return (Xs-mu)/sd, (Xt-mu)/sd


def pick_threshold(scores, y):
    best=(0,-1,0,0)
    for th in [0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.25]:
        p=(scores>=th).astype(int)
        f1=f1_score(y,p,zero_division=0); pr=precision_score(y,p,zero_division=0); rc=recall_score(y,p,zero_division=0)
        score=f1*0.7+rc*0.2+pr*0.1
        if score>best[1]: best=(th,score,pr,rc)
    return float(best[0])


def train_fold(df, feats, target):
    src=df[df.network!=target].copy(); tgt=df[df.network==target].copy()
    Xs,Xt=src[feats].astype(float).values,tgt[feats].astype(float).values
    Xs,Xt=standardize(Xs,Xt)
    ys=src.y_cls_v2.astype(float).values; yt=tgt.y_cls_v2.astype(float).values
    yrs=src.y_reg.astype(float).values; yrt=tgt.y_reg.astype(float).values

    xs=torch.tensor(Xs,dtype=torch.float32,device=DEVICE)
    xt=torch.tensor(Xt,dtype=torch.float32,device=DEVICE)
    ysc=torch.tensor(ys,dtype=torch.float32,device=DEVICE)
    ysr=torch.tensor(np.log1p(np.clip(yrs,0,None)),dtype=torch.float32,device=DEVICE)

    net=Net(xs.size(1)).to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    posw=torch.tensor([(len(ys)-ys.sum())/max(ys.sum(),1.0)],device=DEVICE)
    bce=nn.BCEWithLogitsLoss(pos_weight=posw)

    bs=1024

    # Stage A: pretrain representation (recon only)
    for ep in range(6):
        idx=np.random.permutation(len(xs))
        for i in range(max(len(xs)//bs,1)):
            b=idx[i*bs:(i+1)*bs]
            if len(b)==0: continue
            xb=xs[b]
            m=(torch.rand_like(xb)<0.3)
            xm=xb.clone(); xm[m]=0
            _,xh,_,_,_=net(xm,l=0)
            loss=F.mse_loss(xh[m],xb[m]) if m.any() else F.mse_loss(xh,xb)
            opt.zero_grad(); loss.backward(); opt.step()

    # Stage B: mild domain adaptation + supervised heads
    for ep in range(10):
        isrc=np.random.permutation(len(xs)); itgt=np.random.permutation(len(xt))
        nb=max(len(xs)//bs,1)
        lamb=0.2 + 0.4*(ep/9)  # mild ramp
        for i in range(nb):
            bs_i=isrc[i*bs:(i+1)*bs]; bt_i=itgt[i*bs:(i+1)*bs]
            if len(bs_i)==0 or len(bt_i)==0: continue
            xsb,yscb,ysrb=xs[bs_i],ysc[bs_i],ysr[bs_i]
            xtb=xt[bt_i]

            z_s,xh_s,yc_s,yr_s,yd_s=net(xsb,l=lamb)
            z_t,xh_t,yc_t,yr_t,yd_t=net(xtb,l=lamb)

            m=(torch.rand_like(xsb)<0.2)
            xsm=xsb.clone(); xsm[m]=0
            z_m,xh_m,_,_,_=net(xsm,l=lamb)
            loss_rec=F.mse_loss(xh_m[m],xsb[m]) if m.any() else F.mse_loss(xh_m,xsb)
            loss_cls=bce(yc_s,yscb)
            loss_reg=F.mse_loss(yr_s,ysrb)
            dom_s=torch.zeros_like(yd_s); dom_t=torch.ones_like(yd_t)
            loss_dom=F.binary_cross_entropy_with_logits(yd_s,dom_s)+F.binary_cross_entropy_with_logits(yd_t,dom_t)
            loss_cor=coral(z_s,z_t)

            loss = loss_rec + 1.0*loss_cls + 0.4*loss_reg + 0.12*loss_dom + 0.12*loss_cor
            opt.zero_grad(); loss.backward(); opt.step()

    # Stage C: threshold tuning on source predictions
    net.eval()
    with torch.no_grad():
        _,_,yc_s,yr_s,_=net(xs,l=0)
        _,_,yc_t,yr_t,_=net(xt,l=0)
        ss=torch.sigmoid(yc_s).cpu().numpy()
        st=torch.sigmoid(yc_t).cpu().numpy()
        th=pick_threshold(ss, ys.astype(int))
        pt=(st>=th).astype(int)
        pr=np.expm1(yr_t.cpu().numpy()); pr=np.clip(pr,0,None)

    cls={'test_network':target,'auc':float(roc_auc_score(yt,st)),'ap':float(average_precision_score(yt,st)),'f1':float(f1_score(yt,pt,zero_division=0)),'precision':float(precision_score(yt,pt,zero_division=0)),'recall':float(recall_score(yt,pt,zero_division=0)),'threshold':th}
    reg={'test_network':target,'mae':float(mean_absolute_error(yrt,pr)),'rmse':float(mean_squared_error(yrt,pr)**0.5),'r2':float(r2_score(yrt,pr))}
    return cls,reg


def main():
    torch.manual_seed(42); np.random.seed(42)
    df=pd.read_parquet(DATA)
    feats=[c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_') or c.endswith('_tshift')]
    feats += [c for c in ['n_samples','pos_rate','yreg_mean','yreg_std'] if c in df.columns]

    crows=[]; rrows=[]
    for n in sorted(df.network.unique()):
        print('fold',n,flush=True)
        c,r=train_fold(df,feats,n)
        crows.append(c); rrows.append(r)

    cdf=pd.DataFrame(crows); rdf=pd.DataFrame(rrows)
    cdf.to_csv(OUT/'v2_graphmae_dann_coral_staged_cls.csv',index=False)
    rdf.to_csv(OUT/'v2_graphmae_dann_coral_staged_reg.csv',index=False)
    summ={'classification_mean':cdf[['auc','ap','f1','precision','recall']].mean().to_dict(),'regression_mean':rdf[['mae','rmse','r2']].mean().to_dict()}
    (OUT/'v2_graphmae_dann_coral_staged_summary.json').write_text(json.dumps(summ,indent=2))
    print(json.dumps(summ,indent=2))

if __name__=='__main__':
    main()
