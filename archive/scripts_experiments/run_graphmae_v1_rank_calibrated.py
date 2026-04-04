#!/usr/bin/env python3
"""GraphMAE v1 + ranking-oriented threshold calibration (strict LONO)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

OPF_ROOT = Path('data/opfdata')
DOWN_PATH = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_graph(path: Path):
    d = json.loads(path.read_text())
    x = torch.tensor(d['grid']['nodes']['bus'], dtype=torch.float)
    n = x.size(0)
    edges = []
    ge = d['grid']['edges']
    for et in ['ac_line', 'transformer']:
        obj = ge.get(et, {})
        s = obj.get('senders', []) if isinstance(obj, dict) else []
        r = obj.get('receivers', []) if isinstance(obj, dict) else []
        for a,b in zip(s,r):
            a,b = int(a), int(b)
            if a<n and b<n:
                edges += [(a,b),(b,a)]
    if not edges:
        for i in range(n-1): edges += [(i,i+1),(i+1,i)]
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    case='unknown'
    for p in path.parts:
        if 'case' in p.lower(): case=p; break
    return Data(x=x, edge_index=ei), case


class Model(nn.Module):
    def __init__(self, in_dim, hid=64, z=32):
        super().__init__()
        self.c1=GCNConv(in_dim,hid); self.c2=GCNConv(hid,z)
        self.dec=nn.Sequential(nn.Linear(z,hid),nn.ReLU(),nn.Linear(hid,in_dim))
    def enc(self,x,ei):
        return self.c2(F.relu(self.c1(x,ei)),ei)
    def forward(self,d,mask=0.3):
        x=d.x.clone(); m=torch.rand(x.size(0),device=x.device)<mask; x[m]=0
        z=self.enc(x,d.edge_index); xh=self.dec(z)
        return F.mse_loss(xh[m], d.x[m]) if m.any() else F.mse_loss(xh,d.x)


def case_proxy(c):
    c=c.lower()
    if 'case14' in c: return 'case14'
    if 'case30' in c: return 'case30'
    if 'case57' in c: return 'case57'
    if 'case118' in c: return 'case118'
    return 'other'

def net_proxy(n):
    return {'ieee24':'case30','ieee39':'case57','ieee118':'case118','uk':'case118'}.get(n,'case57')


def select_threshold_cv(train_df, score_col='score'):
    nets=sorted(train_df.network.unique())
    grid=[0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.30,0.35]
    rows=[]
    for th in grid:
        fs,rs,ps=[],[],[]
        for vn in nets:
            te=train_df[train_df.network==vn]; s=te[score_col].values; y=te.y_cls.values
            p=(s>=th).astype(int)
            fs.append(f1_score(y,p,zero_division=0)); rs.append(recall_score(y,p,zero_division=0)); ps.append(precision_score(y,p,zero_division=0))
        rows.append((th,float(np.mean(fs)),float(np.mean(rs)),float(np.mean(ps))))
    t=pd.DataFrame(rows,columns=['th','f1','recall','precision'])
    # prioritize F1 with minimum recall guard
    t2=t[t.recall>=0.18]
    if len(t2)==0: t2=t
    b=t2.sort_values(['f1','recall'],ascending=False).iloc[0]
    return float(b.th), t


def main():
    torch.manual_seed(42)
    files=list(OPF_ROOT.rglob('*.json'))
    if len(files)>20000:
        rng=np.random.default_rng(42); files=list(rng.choice(files,size=20000,replace=False))

    graphs=[]; cases=[]
    for i,p in enumerate(files,1):
        try:
            g,c=load_graph(p); graphs.append(g); cases.append(c)
        except Exception:
            continue
        if i%5000==0: print('loaded',i,flush=True)

    model=Model(in_dim=graphs[0].x.size(1)).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loader=DataLoader(graphs,batch_size=128,shuffle=True)
    model.train()
    for ep in range(1,9):
        tot=0
        for b in loader:
            b=b.to(DEVICE); opt.zero_grad(); loss=model(b); loss.backward(); opt.step(); tot+=float(loss.item())
        print('epoch',ep,'loss',tot/len(loader),flush=True)

    # embeddings per graph then proto by case
    model.eval(); embs=[]
    with torch.no_grad():
        for b in DataLoader(graphs,batch_size=256,shuffle=False):
            b=b.to(DEVICE)
            z=model.enc(b.x,b.edge_index)
            # mean per graph (manual using ptr)
            ptr=b.ptr.cpu().numpy(); zz=z.cpu().numpy()
            g=[]
            for i in range(len(ptr)-1): g.append(zz[ptr[i]:ptr[i+1]].mean(axis=0))
            embs.append(np.array(g))
    E=np.concatenate(embs,axis=0)
    proto=pd.DataFrame(E,columns=[f'gmae_z{i}' for i in range(E.shape[1])]); proto['case_name']=cases
    proto['proxy']=proto.case_name.map(case_proxy)
    pcols=[c for c in proto.columns if c.startswith('gmae_z')]
    proto2=proto.groupby('proxy')[pcols].mean().reset_index()

    down=pd.read_parquet(DOWN_PATH)
    down['proxy']=down.network.map(net_proxy)
    fused=down.merge(proto2,on='proxy',how='left').fillna(0)
    feats=[c for c in fused.columns if c.startswith('ef_') or c.startswith('efnc_') or c.startswith('gmae_z')]

    rows=[]
    for net in sorted(fused.network.unique()):
        tr=fused[fused.network!=net].copy(); te=fused[fused.network==net].copy()
        Xtr,Xte=tr[feats].astype(float),te[feats].astype(float)
        ytr,yte=tr.y_cls.astype(int).values,te.y_cls.astype(int).values
        w=np.ones(len(ytr)); w[ytr==1]=12.0
        clf=RandomForestClassifier(n_estimators=450,max_depth=20,random_state=42,n_jobs=-1)
        clf.fit(Xtr,ytr,sample_weight=w)
        tr['score']=clf.predict_proba(Xtr)[:,1]
        s_te=clf.predict_proba(Xte)[:,1]
        th,th_tbl=select_threshold_cv(tr[['network','y_cls','score']])
        p=(s_te>=th).astype(int)
        rows.append({'test_network':net,'threshold':th,'auc':float(roc_auc_score(yte,s_te)),'ap':float(average_precision_score(yte,s_te)),'f1':float(f1_score(yte,p,zero_division=0)),'precision':float(precision_score(yte,p,zero_division=0)),'recall':float(recall_score(yte,p,zero_division=0))})

    res=pd.DataFrame(rows)
    res.to_csv(OUT/'graphmae_v1_rankcal_lono.csv',index=False)
    summary=res[['auc','ap','f1','precision','recall']].mean().to_dict()
    (OUT/'graphmae_v1_rankcal_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
