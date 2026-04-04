#!/usr/bin/env python3
"""Final v2 threshold calibration: precision-recall tradeoff under LONO.
Outputs deployment threshold suggestions + paper table.
"""
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


def eval_threshold(df, feat_cols, th):
    cls_rows, reg_rows = [], []
    for net in sorted(df.network.unique()):
        tr = df[df.network != net]
        te = df[df.network == net]
        Xtr, Xte = tr[feat_cols].astype(float).values, te[feat_cols].astype(float).values

        ytr = tr['y_cls_v2'].astype(int).values
        yte = te['y_cls_v2'].astype(int).values

        w = np.ones(len(ytr)); w[ytr == 1] = 10.0
        clf = RandomForestClassifier(n_estimators=550, max_depth=22, random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr, sample_weight=w)
        s = clf.predict_proba(Xte)[:, 1]
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
        reg = ExtraTreesRegressor(n_estimators=550, max_depth=22, random_state=42, n_jobs=-1)
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
    return cls_df, reg_df


def main():
    df = pd.read_parquet(DATA)
    feat = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_') or c.endswith('_tshift')]
    meta = [c for c in ['n_samples','pos_rate','yreg_mean','yreg_std'] if c in df.columns]
    feat_cols = feat + meta

    grid = [0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.25,0.30]
    rows = []
    best_mode = None
    best_score = -1
    per_th = {}

    for th in grid:
        cls_df, reg_df = eval_threshold(df, feat_cols, th)
        cm = cls_df[['auc','ap','f1','precision','recall']].mean().to_dict()
        rm = reg_df[['mae','rmse','r2']].mean().to_dict()
        per_th[str(th)] = {'classification': cm, 'regression': rm}

        # balanced utility: keep high recall while improving precision/f1
        score = cm['f1'] * 0.5 + cm['ap'] * 0.2 + cm['auc'] * 0.2 + cm['precision'] * 0.1
        rows.append({'threshold': th, **cm, 'utility': score})
        if score > best_score:
            best_score = score
            best_mode = (th, cls_df, reg_df, cm, rm)

    sweep = pd.DataFrame(rows).sort_values('utility', ascending=False)
    sweep.to_csv(OUT / 'v2_threshold_sweep.csv', index=False)

    th, cls_best, reg_best, cm, rm = best_mode
    cls_best.to_csv(OUT / 'v2_final_cls_by_network.csv', index=False)
    reg_best.to_csv(OUT / 'v2_final_reg_by_network.csv', index=False)

    # deployment suggestions
    deploy = {
        'high_recall': {'threshold': 0.12},
        'balanced': {'threshold': float(th)},
        'high_precision': {'threshold': 0.25}
    }

    # evaluate selected deployment thresholds from sweep table
    idx = sweep.set_index('threshold')
    for k, v in deploy.items():
        t = v['threshold']
        if t in idx.index:
            v['metrics'] = idx.loc[t, ['auc','ap','f1','precision','recall','utility']].to_dict()

    summary = {
        'best_balanced_threshold': float(th),
        'best_balanced_classification': cm,
        'regression_mean': rm,
        'deployment_profiles': deploy,
    }
    (OUT / 'v2_final_summary.json').write_text(json.dumps(summary, indent=2))

    # paper main table (final)
    paper = pd.DataFrame([
        {'section':'classification_final', 'threshold': th, **cm},
        {'section':'regression_final', 'threshold': np.nan, **rm},
    ])
    paper.to_csv(OUT / 'paper_main_results_final.csv', index=False)

    md = []
    md.append('# Final Paper/Deployment Results (v2)\n')
    md.append('## Threshold sweep (classification)\n')
    md.append('```\n' + sweep.to_string(index=False) + '\n```\n')
    md.append('## Final selected (balanced)\n')
    md.append('```\n' + json.dumps(summary, indent=2) + '\n```\n')
    md.append('## By network\n')
    md.append('### Classification\n```\n' + cls_best.to_string(index=False) + '\n```\n')
    md.append('### Regression\n```\n' + reg_best.to_string(index=False) + '\n```\n')
    (OUT / 'paper_main_results_final.md').write_text('\n'.join(md))

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
