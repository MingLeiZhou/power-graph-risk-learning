#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def recall_at_k(y_true, scores, k_frac=0.1):
    k = max(1, int(len(y_true)*k_frac))
    idx = np.argsort(scores)[::-1][:k]
    pos = y_true.sum()
    return float(0.0 if pos<=0 else y_true[idx].sum()/pos)


def eval_cls(tr, va, feats, wb, md, qm):
    Xtr, Xva = tr[feats].astype(float), va[feats].astype(float)
    ytr, yva = tr.y_cls.astype(int).values, va.y_cls.astype(int).values
    w = np.ones(len(ytr)); w[ytr==1] = wb
    clf = RandomForestClassifier(n_estimators=260,max_depth=md,min_samples_leaf=1,random_state=42,n_jobs=-1)
    clf.fit(Xtr,ytr,sample_weight=w)
    s = clf.predict_proba(Xva)[:,1]
    src_prior = float((tr.y_cls==1).mean())
    q = min(0.30, max(0.02, src_prior*qm))
    th = float(np.quantile(s, 1-q))
    p = (s>=th).astype(int)
    return {
        'auc': float(roc_auc_score(yva,s)),
        'ap': float(average_precision_score(yva,s)),
        'f1': float(f1_score(yva,p,zero_division=0)),
        'precision': float(precision_score(yva,p,zero_division=0)),
        'recall': float(recall_score(yva,p,zero_division=0)),
        'r10': recall_at_k(yva,s,0.10),
    }


def select_cls(source, feats):
    nets = sorted(source.network.unique())
    grid = [(8,18,1.6),(12,18,1.6),(12,22,1.6),(12,22,2.0),(16,22,2.0)]
    rows=[]
    for wb,md,qm in grid:
        ms=[]
        for vn in nets:
            tr=source[source.network!=vn]; va=source[source.network==vn]
            ms.append(eval_cls(tr,va,feats,wb,md,qm))
        avg={k:float(np.mean([m[k] for m in ms])) for k in ms[0]}
        rows.append({'wb':wb,'md':md,'qm':qm,**avg})
    t=pd.DataFrame(rows).sort_values(['f1','ap','auc'],ascending=False)
    b=t.iloc[0]
    return {'wb':float(b.wb),'md':int(b.md),'qm':float(b.qm)}, t


def eval_reg(tr, va, feats, md, ml):
    Xtr,Xva = tr[feats].astype(float), va[feats].astype(float)
    ytr,yva = tr.y_reg.astype(float).values, va.y_reg.astype(float).values
    reg=ExtraTreesRegressor(n_estimators=260,max_depth=md,min_samples_leaf=ml,random_state=42,n_jobs=-1)
    reg.fit(Xtr,np.log1p(np.clip(ytr,0,None)))
    pred=np.expm1(reg.predict(Xva)); pred=np.clip(pred,0,None)
    return {'mae':float(mean_absolute_error(yva,pred)),'rmse':float(mean_squared_error(yva,pred)**0.5),'r2':float(r2_score(yva,pred))}


def select_reg(source,feats):
    nets=sorted(source.network.unique())
    grid=[(18,1),(18,2),(22,1),(22,2)]
    rows=[]
    for md,ml in grid:
        ms=[]
        for vn in nets:
            tr=source[source.network!=vn]; va=source[source.network==vn]
            ms.append(eval_reg(tr,va,feats,md,ml))
        avg={k:float(np.mean([m[k] for m in ms])) for k in ms[0]}
        rows.append({'md':md,'ml':ml,**avg})
    t=pd.DataFrame(rows).sort_values(['rmse','mae'],ascending=True)
    b=t.iloc[0]
    return {'md':int(b.md),'ml':int(b.ml)}, t


def main():
    df=pd.read_parquet(DATA)
    feats=[c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]
    cls_rows=[]; reg_rows=[]; innerc=[]; innerr=[]
    for net in sorted(df.network.unique()):
        print('target',net,flush=True)
        src=df[df.network!=net]; tgt=df[df.network==net]
        cls_p, t1 = select_cls(src,feats)
        reg_p, t2 = select_reg(src,feats)
        t1['target_net']=net; t2['target_net']=net
        innerc.append(t1); innerr.append(t2)

        # final cls
        Xs,Xt=src[feats].astype(float),tgt[feats].astype(float)
        ys,yt=src.y_cls.astype(int).values,tgt.y_cls.astype(int).values
        w=np.ones(len(ys)); w[ys==1]=cls_p['wb']
        clf=RandomForestClassifier(n_estimators=350,max_depth=cls_p['md'],min_samples_leaf=1,random_state=42,n_jobs=-1)
        clf.fit(Xs,ys,sample_weight=w)
        s=clf.predict_proba(Xt)[:,1]
        q=min(0.30,max(0.02,float((src.y_cls==1).mean())*cls_p['qm']))
        th=float(np.quantile(s,1-q)); p=(s>=th).astype(int)
        cls_rows.append({'test_network':net,'threshold':th,'auc':float(roc_auc_score(yt,s)),'ap':float(average_precision_score(yt,s)),'f1':float(f1_score(yt,p,zero_division=0)),'precision':float(precision_score(yt,p,zero_division=0)),'recall':float(recall_score(yt,p,zero_division=0)),'recall_at_10pct':recall_at_k(yt,s,0.10),'params':json.dumps(cls_p)})

        # final reg
        yr_s,yr_t=src.y_reg.astype(float).values,tgt.y_reg.astype(float).values
        reg=ExtraTreesRegressor(n_estimators=350,max_depth=reg_p['md'],min_samples_leaf=reg_p['ml'],random_state=42,n_jobs=-1)
        reg.fit(Xs,np.log1p(np.clip(yr_s,0,None)))
        pred=np.expm1(reg.predict(Xt)); pred=np.clip(pred,0,None)
        reg_rows.append({'test_network':net,'mae':float(mean_absolute_error(yr_t,pred)),'rmse':float(mean_squared_error(yr_t,pred)**0.5),'r2':float(r2_score(yr_t,pred)),'params':json.dumps(reg_p)})

    cls_df=pd.DataFrame(cls_rows); reg_df=pd.DataFrame(reg_rows)
    cls_df.to_csv(OUT/'dual_domain_fast_cls_by_network.csv',index=False)
    reg_df.to_csv(OUT/'dual_domain_fast_reg_by_network.csv',index=False)
    pd.concat(innerc,ignore_index=True).to_csv(OUT/'dual_domain_fast_cls_innercv.csv',index=False)
    pd.concat(innerr,ignore_index=True).to_csv(OUT/'dual_domain_fast_reg_innercv.csv',index=False)

    summary={'classification_mean':cls_df[['auc','ap','f1','precision','recall','recall_at_10pct']].mean().to_dict(),'regression_mean':reg_df[['mae','rmse','r2']].mean().to_dict()}
    (OUT/'dual_domain_fast_summary.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame([{'task':'classification',**summary['classification_mean']},{'task':'regression',**summary['regression_mean']}]).to_csv(OUT/'paper_main_results.csv',index=False)
    Path(OUT/'paper_main_results.md').write_text('# Paper Main Results (Strict LONO)\n\n## Classification\n```\n'+cls_df.to_string(index=False)+'\n```\n\n## Regression\n```\n'+reg_df.to_string(index=False)+'\n```\n\n## Mean\n```\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
