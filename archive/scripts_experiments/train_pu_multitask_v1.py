#!/usr/bin/env python3
"""PU learning + multitask surrogate on v2 informative dataset (strict LONO)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path('data/processed/downstream/downstream_v2_informative.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def train_pu_classifier(X, y, n_estimators=500):
    # Step1: PN warm-start
    clf1 = RandomForestClassifier(n_estimators=n_estimators, max_depth=22, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    clf1.fit(X, y)
    s = clf1.predict_proba(X)[:, 1]

    # Step2: identify reliable negatives from unlabeled (y=0) low-score tail
    unl = (y == 0)
    if unl.sum() > 20:
        thr = np.quantile(s[unl], 0.25)
    else:
        thr = 0.2
    reliable_neg = (unl & (s <= thr))

    # sample weights: positives high, reliable negatives medium, uncertain unlabeled low
    w = np.full(len(y), 0.4, dtype=float)
    w[reliable_neg] = 0.8
    w[y == 1] = 3.0

    clf2 = RandomForestClassifier(n_estimators=n_estimators, max_depth=22, random_state=42, n_jobs=-1)
    clf2.fit(X, y, sample_weight=w)
    return clf2


def train_multitask_encoder(X, y_cls, y_reg):
    # shared encoder proxy via autoencoder-ish regressor on concatenated targets
    # y targets: [y_cls, log1p(y_reg)] to force shared representation
    Y = np.column_stack([y_cls.astype(float), np.log1p(np.clip(y_reg, 0, None))])
    m = MLPRegressor(hidden_layer_sizes=(128, 32, 128), max_iter=80, random_state=42, early_stopping=True)
    # multitask reconstruction surrogate: reconstruct X plus Y in joint space
    XY = np.column_stack([X, Y])
    m.fit(XY, XY)

    # bottleneck extraction from trained network
    W1, W2 = m.coefs_[0], m.coefs_[1]
    b1, b2 = m.intercepts_[0], m.intercepts_[1]
    H = np.maximum(0, XY @ W1 + b1)
    Z = np.maximum(0, H @ W2 + b2)
    return m, Z


def transform_multitask(m, X, y_cls_hint=None, y_reg_hint=None):
    # at inference no labels; feed zeros for target slots
    zc = np.zeros((len(X), 1)) if y_cls_hint is None else y_cls_hint.reshape(-1, 1)
    zr = np.zeros((len(X), 1)) if y_reg_hint is None else np.log1p(np.clip(y_reg_hint, 0, None)).reshape(-1, 1)
    XY = np.column_stack([X, zc, zr])
    W1, W2 = m.coefs_[0], m.coefs_[1]
    b1, b2 = m.intercepts_[0], m.intercepts_[1]
    H = np.maximum(0, XY @ W1 + b1)
    Z = np.maximum(0, H @ W2 + b2)
    return Z


def main():
    df = pd.read_parquet(DATA)
    feat = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_') or c.endswith('_tshift')]
    feat += [c for c in ['n_samples','pos_rate','yreg_mean','yreg_std'] if c in df.columns]

    cls_rows, reg_rows = [], []
    for net in sorted(df.network.unique()):
        tr = df[df.network != net]
        te = df[df.network == net]
        Xtr = tr[feat].astype(float).values
        Xte = te[feat].astype(float).values
        ytr = tr['y_cls_v2'].astype(int).values
        yte = te['y_cls_v2'].astype(int).values
        yrtr = tr['y_reg'].astype(float).values
        yrte = te['y_reg'].astype(float).values

        # multitask shared representation
        enc, Ztr = train_multitask_encoder(Xtr, ytr, yrtr)
        Zte = transform_multitask(enc, Xte)

        # PU classifier on Z
        clf = train_pu_classifier(Ztr, ytr, n_estimators=320)
        s = clf.predict_proba(Zte)[:, 1]
        # choose threshold by source prior budget
        q = min(0.25, max(0.03, float(ytr.mean()) * 1.5))
        th = float(np.quantile(s, 1 - q))
        p = (s >= th).astype(int)

        cls_rows.append({
            'test_network': net,
            'auc': float(roc_auc_score(yte, s)),
            'ap': float(average_precision_score(yte, s)),
            'f1': float(f1_score(yte, p, zero_division=0)),
            'precision': float(precision_score(yte, p, zero_division=0)),
            'recall': float(recall_score(yte, p, zero_division=0)),
            'threshold': th,
        })

        reg = ExtraTreesRegressor(n_estimators=320, max_depth=20, random_state=42, n_jobs=-1)
        reg.fit(Ztr, np.log1p(np.clip(yrtr, 0, None)))
        pr = np.expm1(reg.predict(Zte)); pr = np.clip(pr, 0, None)
        reg_rows.append({
            'test_network': net,
            'mae': float(mean_absolute_error(yrte, pr)),
            'rmse': float(mean_squared_error(yrte, pr) ** 0.5),
            'r2': float(r2_score(yrte, pr)),
        })

    cls_df = pd.DataFrame(cls_rows)
    reg_df = pd.DataFrame(reg_rows)
    cls_df.to_csv(OUT / 'pu_multitask_v1_cls.csv', index=False)
    reg_df.to_csv(OUT / 'pu_multitask_v1_reg.csv', index=False)

    summary = {
        'classification_mean': cls_df[['auc','ap','f1','precision','recall']].mean().to_dict(),
        'regression_mean': reg_df[['mae','rmse','r2']].mean().to_dict(),
    }
    (OUT / 'pu_multitask_v1_summary.json').write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
