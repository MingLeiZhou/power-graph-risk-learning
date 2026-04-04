#!/usr/bin/env python3
"""Retrain on augmented dataset and compare with prior best summary."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture

AUG = Path('data/processed/downstream/downstream_augmented.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)
PRIOR = OUT / 'seq123_final_summary.json'


def gmm_threshold(scores):
    s = scores.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=42)
    gm.fit(s)
    m = gm.means_.flatten()
    return float((np.min(m) + np.max(m)) / 2.0)


def eval_lono(df, feat_cols):
    cls_rows, reg_rows = [], []
    for net in sorted(df.network.unique()):
        tr = df[df.network != net]
        te = df[df.network == net]
        Xtr, Xte = tr[feat_cols].astype(float).values, te[feat_cols].astype(float).values
        ytr, yte = tr.y_cls.astype(int).values, te.y_cls.astype(int).values

        w = np.ones(len(ytr)); w[ytr == 1] = 10.0
        clf = RandomForestClassifier(n_estimators=450, max_depth=20, random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr, sample_weight=w)
        s_te = clf.predict_proba(Xte)[:, 1]
        th = gmm_threshold(s_te)
        p = (s_te >= th).astype(int)

        cls_rows.append({
            'test_network': net,
            'auc': float(roc_auc_score(yte, s_te)),
            'ap': float(average_precision_score(yte, s_te)),
            'f1': float(f1_score(yte, p, zero_division=0)),
            'precision': float(precision_score(yte, p, zero_division=0)),
            'recall': float(recall_score(yte, p, zero_division=0)),
        })

        yr_tr, yr_te = tr.y_reg.astype(float).values, te.y_reg.astype(float).values
        reg = ExtraTreesRegressor(n_estimators=450, max_depth=20, random_state=42, n_jobs=-1)
        reg.fit(Xtr, np.log1p(np.clip(yr_tr, 0, None)))
        pr = np.expm1(reg.predict(Xte)); pr = np.clip(pr, 0, None)
        reg_rows.append({
            'test_network': net,
            'mae': float(mean_absolute_error(yr_te, pr)),
            'rmse': float(mean_squared_error(yr_te, pr) ** 0.5),
            'r2': float(r2_score(yr_te, pr)),
        })
    return pd.DataFrame(cls_rows), pd.DataFrame(reg_rows)


def main():
    df = pd.read_parquet(AUG)
    feat_cols = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]
    cls_df, reg_df = eval_lono(df, feat_cols)
    cls_df.to_csv(OUT / 'augmented_lono_cls.csv', index=False)
    reg_df.to_csv(OUT / 'augmented_lono_reg.csv', index=False)

    cur = {
        'classification': cls_df[['auc','ap','f1','precision','recall']].mean().to_dict(),
        'regression': reg_df[['mae','rmse','r2']].mean().to_dict()
    }

    prior = None
    if PRIOR.exists():
        p = json.loads(PRIOR.read_text())
        prior = {
            'classification': p.get('final_cls_mean', {}),
            'regression': p.get('final_reg_mean', {})
        }

    report = {'prior': prior, 'current_augmented': cur}
    (OUT / 'augmented_compare_summary.json').write_text(json.dumps(report, indent=2))

    # delta table
    rows = []
    if prior:
        for k in ['auc','ap','f1','precision','recall']:
            rows.append({'task':'classification','metric':k,'prior':prior['classification'].get(k), 'augmented':cur['classification'].get(k), 'delta':cur['classification'].get(k)-prior['classification'].get(k)})
        for k in ['mae','rmse','r2']:
            rows.append({'task':'regression','metric':k,'prior':prior['regression'].get(k), 'augmented':cur['regression'].get(k), 'delta':cur['regression'].get(k)-prior['regression'].get(k)})
    pd.DataFrame(rows).to_csv(OUT / 'augmented_compare_delta.csv', index=False)

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
