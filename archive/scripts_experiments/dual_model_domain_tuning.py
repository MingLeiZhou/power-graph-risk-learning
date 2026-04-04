#!/usr/bin/env python3
"""Dual-model domain tuning under strict LONO.
- Classification: optimize F1/Recall with source-network CV + prior thresholding
- Regression: optimize RMSE with source-network CV (log1p target)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def recall_at_k(y_true, scores, k_frac=0.1):
    n = len(y_true)
    k = max(1, int(n * k_frac))
    idx = np.argsort(scores)[::-1][:k]
    pos = y_true.sum()
    if pos <= 0:
        return 0.0
    return float(y_true[idx].sum() / pos)


def train_cls_predict(train_df, valid_df, feats, weight_boost, max_depth, q_mult):
    Xtr, Xva = train_df[feats].astype(float), valid_df[feats].astype(float)
    ytr = train_df.y_cls.astype(int).values
    yva = valid_df.y_cls.astype(int).values

    w = np.ones(len(ytr), dtype=float)
    w[ytr == 1] = weight_boost

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=max_depth,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(Xtr, ytr, sample_weight=w)

    s = clf.predict_proba(Xva)[:, 1]
    src_prior = float((train_df.y_cls == 1).mean())
    q = min(0.30, max(0.02, src_prior * q_mult))
    thr = float(np.quantile(s, 1 - q))
    p = (s >= thr).astype(int)

    return {
        'auc': float(roc_auc_score(yva, s)),
        'ap': float(average_precision_score(yva, s)),
        'f1': float(f1_score(yva, p, zero_division=0)),
        'precision': float(precision_score(yva, p, zero_division=0)),
        'recall': float(recall_score(yva, p, zero_division=0)),
        'r10': recall_at_k(yva, s, 0.10)
    }


def select_cls_params(source_df, feats):
    nets = sorted(source_df.network.unique())
    grid = []
    for wb in [8.0, 12.0, 16.0]:
        for md in [16, 22]:
            for qm in [1.2, 1.6, 2.0]:
                grid.append((wb, md, qm))

    rows = []
    for wb, md, qm in grid:
        mets = []
        for va_net in nets:
            tr = source_df[source_df.network != va_net]
            va = source_df[source_df.network == va_net]
            m = train_cls_predict(tr, va, feats, wb, md, qm)
            mets.append(m)
        avg = {k: float(np.mean([x[k] for x in mets])) for k in mets[0].keys()}
        rows.append({'weight_boost': wb, 'max_depth': md, 'q_mult': qm, **avg})

    tbl = pd.DataFrame(rows).sort_values(['f1', 'ap', 'auc'], ascending=False)
    best = tbl.iloc[0]
    return {
        'weight_boost': float(best.weight_boost),
        'max_depth': int(best.max_depth),
        'q_mult': float(best.q_mult)
    }, tbl


def train_reg_predict(train_df, valid_df, feats, max_depth, min_leaf):
    Xtr, Xva = train_df[feats].astype(float), valid_df[feats].astype(float)
    ytr = train_df.y_reg.astype(float).values
    yva = valid_df.y_reg.astype(float).values

    reg = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        random_state=42,
        n_jobs=-1
    )
    reg.fit(Xtr, np.log1p(np.clip(ytr, 0, None)))
    pred = np.expm1(reg.predict(Xva))
    pred = np.clip(pred, 0, None)

    return {
        'mae': float(mean_absolute_error(yva, pred)),
        'rmse': float(mean_squared_error(yva, pred) ** 0.5),
        'r2': float(r2_score(yva, pred))
    }


def select_reg_params(source_df, feats):
    nets = sorted(source_df.network.unique())
    grid = []
    for md in [16, 20, None]:
        for ml in [1, 2, 4]:
            grid.append((md, ml))

    rows = []
    for md, ml in grid:
        mets = []
        for va_net in nets:
            tr = source_df[source_df.network != va_net]
            va = source_df[source_df.network == va_net]
            mets.append(train_reg_predict(tr, va, feats, md, ml))
        avg = {k: float(np.mean([x[k] for x in mets])) for k in mets[0].keys()}
        rows.append({'max_depth': md if md is not None else -1, 'min_leaf': ml, **avg})

    tbl = pd.DataFrame(rows).sort_values(['rmse', 'mae'], ascending=True)
    best = tbl.iloc[0]
    return {
        'max_depth': None if int(best.max_depth) == -1 else int(best.max_depth),
        'min_leaf': int(best.min_leaf)
    }, tbl


def run_target(df, target_net, feats):
    source_df = df[df.network != target_net].copy()
    target_df = df[df.network == target_net].copy()

    cls_params, cls_tbl = select_cls_params(source_df, feats)
    reg_params, reg_tbl = select_reg_params(source_df, feats)

    # final cls
    Xs, Xt = source_df[feats].astype(float), target_df[feats].astype(float)
    ys, yt = source_df.y_cls.astype(int).values, target_df.y_cls.astype(int).values
    w = np.ones(len(ys)); w[ys == 1] = cls_params['weight_boost']
    clf = RandomForestClassifier(
        n_estimators=700,
        max_depth=cls_params['max_depth'],
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(Xs, ys, sample_weight=w)
    s = clf.predict_proba(Xt)[:, 1]
    src_prior = float((source_df.y_cls == 1).mean())
    q = min(0.30, max(0.02, src_prior * cls_params['q_mult']))
    thr = float(np.quantile(s, 1 - q))
    p = (s >= thr).astype(int)

    cls_out = {
        'test_network': target_net,
        'threshold': thr,
        'auc': float(roc_auc_score(yt, s)),
        'ap': float(average_precision_score(yt, s)),
        'f1': float(f1_score(yt, p, zero_division=0)),
        'precision': float(precision_score(yt, p, zero_division=0)),
        'recall': float(recall_score(yt, p, zero_division=0)),
        'recall_at_10pct': recall_at_k(yt, s, 0.10),
        'pos_rate_test': float(yt.mean()),
        'cls_params': json.dumps(cls_params),
    }

    # final reg
    yr_s, yr_t = source_df.y_reg.astype(float).values, target_df.y_reg.astype(float).values
    reg = ExtraTreesRegressor(
        n_estimators=700,
        max_depth=reg_params['max_depth'],
        min_samples_leaf=reg_params['min_leaf'],
        random_state=42,
        n_jobs=-1
    )
    reg.fit(Xs, np.log1p(np.clip(yr_s, 0, None)))
    pred = np.expm1(reg.predict(Xt))
    pred = np.clip(pred, 0, None)

    reg_out = {
        'test_network': target_net,
        'mae': float(mean_absolute_error(yr_t, pred)),
        'rmse': float(mean_squared_error(yr_t, pred) ** 0.5),
        'r2': float(r2_score(yr_t, pred)),
        'reg_params': json.dumps(reg_params),
    }

    return cls_out, reg_out, cls_tbl, reg_tbl


def main():
    df = pd.read_parquet(DATA)
    feats = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]

    cls_rows, reg_rows = [], []
    cv_tables_cls, cv_tables_reg = [], []

    for net in sorted(df.network.unique()):
        print(f'Running target={net} ...', flush=True)
        c, r, tcls, treg = run_target(df, net, feats)
        tcls['target_net'] = net
        treg['target_net'] = net
        cv_tables_cls.append(tcls)
        cv_tables_reg.append(treg)
        cls_rows.append(c)
        reg_rows.append(r)

    cls_df = pd.DataFrame(cls_rows)
    reg_df = pd.DataFrame(reg_rows)
    cls_df.to_csv(OUT / 'dual_domain_cls_by_network.csv', index=False)
    reg_df.to_csv(OUT / 'dual_domain_reg_by_network.csv', index=False)

    pd.concat(cv_tables_cls, ignore_index=True).to_csv(OUT / 'dual_domain_cls_innercv.csv', index=False)
    pd.concat(cv_tables_reg, ignore_index=True).to_csv(OUT / 'dual_domain_reg_innercv.csv', index=False)

    summary = {
        'classification_mean': cls_df[['auc','ap','f1','precision','recall','recall_at_10pct']].mean().to_dict(),
        'regression_mean': reg_df[['mae','rmse','r2']].mean().to_dict(),
    }
    (OUT / 'dual_domain_summary.json').write_text(json.dumps(summary, indent=2))

    # paper main table
    paper = pd.DataFrame([
        {'task': 'classification', **summary['classification_mean']},
        {'task': 'regression', **summary['regression_mean']},
    ])
    paper.to_csv(OUT / 'paper_main_results.csv', index=False)

    md = []
    md.append('# Paper Main Results (Strict LONO)\n')
    md.append('## Classification (best practical detector)\n')
    md.append('```\n' + cls_df.to_string(index=False) + '\n```\n')
    md.append('## Regression (best practical regressor)\n')
    md.append('```\n' + reg_df.to_string(index=False) + '\n```\n')
    md.append('## Mean summary\n')
    md.append('```\n' + json.dumps(summary, indent=2) + '\n```\n')
    (OUT / 'paper_main_results.md').write_text('\n'.join(md))

    print('=== DUAL DOMAIN SUMMARY ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
