#!/usr/bin/env python3
"""F1 boost on fused SSL latent + downstream features (LONO)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

DOWN = Path('data/processed/downstream/downstream_full.parquet')
LAT = Path('data/processed/ssl/opf_ssl_latent.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def map_case_to_proxy(case_name: str):
    c = case_name.lower()
    if 'case14' in c: return 'case14'
    if 'case30' in c: return 'case30'
    if 'case57' in c: return 'case57'
    if 'case118' in c: return 'case118'
    return 'other'


def net_to_proxy(n):
    return {'ieee24':'case30','ieee39':'case57','ieee118':'case118','uk':'case118'}.get(n,'case57')


def choose_th(scores, y):
    best_f1, best_th = -1, 0.15
    for th in [0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.30,0.35,0.40,0.50]:
        p=(scores>=th).astype(int)
        f1=f1_score(y,p,zero_division=0)
        if f1>best_f1:
            best_f1,best_th=f1,th
    return best_th


def main():
    down = pd.read_parquet(DOWN)
    lat = pd.read_parquet(LAT)
    zcols = [c for c in lat.columns if c.startswith('z_')]
    lat['proxy'] = lat['case_name'].map(map_case_to_proxy)
    proto = lat.groupby('proxy')[zcols].mean().reset_index()
    proto = proto.rename(columns={c:f'lat_{c}' for c in zcols})

    df = down.copy()
    df['proxy'] = df['network'].map(net_to_proxy)
    df = df.merge(proto, on='proxy', how='left')
    df = df.fillna(0.0)

    feats = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_') or c.startswith('lat_z_')]

    rows=[]
    for net in sorted(df.network.unique()):
        tr=df[df.network!=net]
        te=df[df.network==net]
        Xtr,Xte=tr[feats].astype(float),te[feats].astype(float)
        ytr,yte=tr.y_cls.astype(int).values,te.y_cls.astype(int).values

        w=np.ones(len(ytr)); w[ytr==1]=10.0
        clf=RandomForestClassifier(n_estimators=600,max_depth=20,min_samples_leaf=2,random_state=42,n_jobs=-1)
        clf.fit(Xtr,ytr,sample_weight=w)

        s_tr=clf.predict_proba(Xtr)[:,1]
        s_te=clf.predict_proba(Xte)[:,1]
        th=choose_th(s_tr,ytr)
        p=(s_te>=th).astype(int)

        rows.append({
            'test_network':net,'threshold':th,
            'auc':float(roc_auc_score(yte,s_te)),
            'f1':float(f1_score(yte,p,zero_division=0)),
            'precision':float(precision_score(yte,p,zero_division=0)),
            'recall':float(recall_score(yte,p,zero_division=0)),
            'acc':float(accuracy_score(yte,p)),
        })

    res=pd.DataFrame(rows)
    res.to_csv(OUT/'f1_boost_fused_lono_by_network.csv',index=False)
    summ=res[['auc','f1','precision','recall','acc']].mean().to_dict()
    pd.DataFrame([summ]).to_csv(OUT/'f1_boost_fused_lono_summary.csv',index=False)

    print('=== FUSED F1 BOOST ===')
    print(res.to_string(index=False))
    print('mean:',summ)


if __name__=='__main__':
    main()
