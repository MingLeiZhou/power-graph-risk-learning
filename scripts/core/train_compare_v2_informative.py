#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path('data/processed/downstream/downstream_v2_informative.parquet')
OUT = Path('analysis/training')
PRIOR = OUT / 'augmented_compare_summary.json'


def gmm_threshold(scores):
    gm = GaussianMixture(n_components=2, random_state=42)
    s = scores.reshape(-1, 1)
    gm.fit(s)
    m = gm.means_.flatten()
    return float((np.min(m) + np.max(m)) / 2.0)


def main():
    df = pd.read_parquet(DATA)
    feat = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_') or c.endswith('_tshift')]
    meta = ['n_samples','pos_rate','yreg_mean','yreg_std']
    feat = feat + [c for c in meta if c in df.columns]

    cls_rows, reg_rows = [], []
    for net in sorted(df.network.unique()):
        tr = df[df.network != net]
        te = df[df.network == net]
        Xtr, Xte = tr[feat].astype(float).values, te[feat].astype(float).values

        ytr = tr['y_cls_v2'].astype(int).values
        yte = te['y_cls_v2'].astype(int).values
        w = np.ones(len(ytr)); w[ytr == 1] = 10.0

        clf = RandomForestClassifier(n_estimators=500, max_depth=22, random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr, sample_weight=w)
        s = clf.predict_proba(Xte)[:, 1]
        th = gmm_threshold(s)
        p = (s >= th).astype(int)

        cls_rows.append({
            'test_network': net,
            'auc': float(roc_auc_score(yte, s)),
            'ap': float(average_precision_score(yte, s)),
            'f1': float(f1_score(yte, p, zero_division=0)),
            'precision': float(precision_score(yte, p, zero_division=0)),
            'recall': float(recall_score(yte, p, zero_division=0)),
        })

        yr_tr = tr['y_reg'].astype(float).values
        yr_te = te['y_reg'].astype(float).values
        reg = ExtraTreesRegressor(n_estimators=500, max_depth=22, random_state=42, n_jobs=-1)
        reg.fit(Xtr, np.log1p(np.clip(yr_tr, 0, None)))
        pr = np.expm1(reg.predict(Xte)); pr = np.clip(pr, 0, None)
        reg_rows.append({
            'test_network': net,
            'mae': float(mean_absolute_error(yr_te, pr)),
            'rmse': float(mean_squared_error(yr_te, pr) ** 0.5),
            'r2': float(r2_score(yr_te, pr)),
        })

    cls_df = pd.DataFrame(cls_rows)
    reg_df = pd.DataFrame(reg_rows)
    cls_df.to_csv(OUT / 'v2_lono_cls.csv', index=False)
    reg_df.to_csv(OUT / 'v2_lono_reg.csv', index=False)

    cur = {
        'classification': cls_df[['auc','ap','f1','precision','recall']].mean().to_dict(),
        'regression': reg_df[['mae','rmse','r2']].mean().to_dict()
    }

    prior = json.loads(PRIOR.read_text()) if PRIOR.exists() else None
    report = {'prior_augmented': prior, 'current_v2': cur}
    (OUT / 'v2_compare_summary.json').write_text(json.dumps(report, indent=2))

    # delta vs previous augmented current
    rows = []
    if prior and 'current_augmented' in prior:
        pcls = prior['current_augmented']['classification']
        preg = prior['current_augmented']['regression']
        for k in ['auc','ap','f1','precision','recall']:
            rows.append({'task':'classification','metric':k,'prev_aug':pcls.get(k),'v2':cur['classification'].get(k),'delta':cur['classification'].get(k)-pcls.get(k)})
        for k in ['mae','rmse','r2']:
            rows.append({'task':'regression','metric':k,'prev_aug':preg.get(k),'v2':cur['regression'].get(k),'delta':cur['regression'].get(k)-preg.get(k)})
    pd.DataFrame(rows).to_csv(OUT / 'v2_compare_delta.csv', index=False)

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
