#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path('data/processed/downstream/downstream_v2_informative.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def eval_th(df, feats, th):
    crows, rrows = [], []
    for net in sorted(df.network.unique()):
        tr=df[df.network!=net]; te=df[df.network==net]
        Xtr,Xte=tr[feats].astype(float).values,te[feats].astype(float).values
        ytr,yte=tr.y_cls_v2.astype(int).values,te.y_cls_v2.astype(int).values
        w=np.ones(len(ytr)); w[ytr==1]=10.0
        clf=RandomForestClassifier(n_estimators=220,max_depth=20,random_state=42,n_jobs=-1)
        clf.fit(Xtr,ytr,sample_weight=w)
        s=clf.predict_proba(Xte)[:,1]; p=(s>=th).astype(int)
        crows.append({'test_network':net,'auc':float(roc_auc_score(yte,s)),'ap':float(average_precision_score(yte,s)),'f1':float(f1_score(yte,p,zero_division=0)),'precision':float(precision_score(yte,p,zero_division=0)),'recall':float(recall_score(yte,p,zero_division=0))})

        yrtr,yrte=tr.y_reg.astype(float).values,te.y_reg.astype(float).values
        reg=ExtraTreesRegressor(n_estimators=220,max_depth=20,random_state=42,n_jobs=-1)
        reg.fit(Xtr,np.log1p(np.clip(yrtr,0,None)))
        pr=np.expm1(reg.predict(Xte)); pr=np.clip(pr,0,None)
        rrows.append({'test_network':net,'mae':float(mean_absolute_error(yrte,pr)),'rmse':float(mean_squared_error(yrte,pr)**0.5),'r2':float(r2_score(yrte,pr))})
    return pd.DataFrame(crows), pd.DataFrame(rrows)


def main():
    df=pd.read_parquet(DATA)
    feats=[c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_') or c.endswith('_tshift')]
    feats += [c for c in ['n_samples','pos_rate','yreg_mean','yreg_std'] if c in df.columns]

    grid=[0.12,0.16,0.20,0.25]
    rows=[]; best=None; bestu=-1
    for th in grid:
        print('threshold',th,flush=True)
        c,r=eval_th(df,feats,th)
        cm=c[['auc','ap','f1','precision','recall']].mean().to_dict()
        rm=r[['mae','rmse','r2']].mean().to_dict()
        u=cm['f1']*0.5+cm['ap']*0.2+cm['auc']*0.2+cm['precision']*0.1
        rows.append({'threshold':th,**cm,'utility':u})
        if u>bestu:
            bestu=u; best=(th,c,r,cm,rm)

    sweep=pd.DataFrame(rows).sort_values('utility',ascending=False)
    sweep.to_csv(OUT/'v2_threshold_sweep.csv',index=False)
    th,c,r,cm,rm=best
    c.to_csv(OUT/'v2_final_cls_by_network.csv',index=False)
    r.to_csv(OUT/'v2_final_reg_by_network.csv',index=False)

    summary={'best_balanced_threshold':float(th),'best_balanced_classification':cm,'regression_mean':rm,'deployment_profiles':{}}
    m=sweep.set_index('threshold')
    for name,t in [('high_recall',0.12),('balanced',float(th)),('high_precision',0.25)]:
        if t in m.index:
            summary['deployment_profiles'][name]={'threshold':t,'metrics':m.loc[t,['auc','ap','f1','precision','recall','utility']].to_dict()}

    (OUT/'v2_final_summary.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame([{'section':'classification_final','threshold':th,**cm},{'section':'regression_final','threshold':np.nan,**rm}]).to_csv(OUT/'paper_main_results_final.csv',index=False)
    Path(OUT/'paper_main_results_final.md').write_text('# Final Paper/Deployment Results (v2)\n\n## Sweep\n```\n'+sweep.to_string(index=False)+'\n```\n\n## Summary\n```\n'+json.dumps(summary,indent=2)+'\n```\n\n## Classification by network\n```\n'+c.to_string(index=False)+'\n```\n\n## Regression by network\n```\n'+r.to_string(index=False)+'\n```\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
