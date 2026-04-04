#!/usr/bin/env python3
"""Top3: domain adaptation via CORAL-style feature alignment under strict LONO.
Uses unlabeled target features for covariate alignment.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def matrix_sqrt_psd(M, eps=1e-6):
    w, v = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (v * np.sqrt(w)) @ v.T


def matrix_inv_sqrt_psd(M, eps=1e-6):
    w, v = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (v * (1.0 / np.sqrt(w))) @ v.T


def coral_align(Xs, Xt):
    """Align source to target covariances: Xs' = (Xs-mu_s)As + mu_t"""
    mu_s = Xs.mean(axis=0, keepdims=True)
    mu_t = Xt.mean(axis=0, keepdims=True)
    Xs0 = Xs - mu_s
    Xt0 = Xt - mu_t

    Cs = np.cov(Xs0, rowvar=False) + np.eye(Xs.shape[1]) * 1e-3
    Ct = np.cov(Xt0, rowvar=False) + np.eye(Xt.shape[1]) * 1e-3

    As = matrix_inv_sqrt_psd(Cs)
    At = matrix_sqrt_psd(Ct)

    Xs_aligned = (Xs0 @ As @ At) + mu_t
    return Xs_aligned, Xt


def threshold_by_train(train_scores, y_train, mult=1.4):
    prior = float(y_train.mean())
    q = min(0.30, max(0.02, prior * mult))
    return float(np.quantile(train_scores, 1 - q))


def main():
    df = pd.read_parquet(DATA)
    feats = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]

    cls_rows, reg_rows = [], []
    for net in sorted(df.network.unique()):
        tr = df[df.network != net].copy()
        te = df[df.network == net].copy()

        Xs = tr[feats].astype(float).values
        Xt = te[feats].astype(float).values

        # Top3 domain adaptation step
        Xs_a, Xt_a = coral_align(Xs, Xt)

        # ----- classification -----
        ys = tr.y_cls.astype(int).values
        yt = te.y_cls.astype(int).values
        w = np.ones(len(ys)); w[ys == 1] = 12.0

        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(Xs_a, ys, sample_weight=w)

        s_tr = clf.predict_proba(Xs_a)[:, 1]
        s_te = clf.predict_proba(Xt_a)[:, 1]
        th = threshold_by_train(s_tr, ys, mult=1.6)
        p = (s_te >= th).astype(int)

        cls_rows.append({
            'test_network': net,
            'threshold': th,
            'auc': float(roc_auc_score(yt, s_te)),
            'ap': float(average_precision_score(yt, s_te)),
            'f1': float(f1_score(yt, p, zero_division=0)),
            'precision': float(precision_score(yt, p, zero_division=0)),
            'recall': float(recall_score(yt, p, zero_division=0)),
        })

        # ----- regression -----
        yr_s = tr.y_reg.astype(float).values
        yr_t = te.y_reg.astype(float).values
        reg = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=1, random_state=42, n_jobs=-1)
        reg.fit(Xs_a, np.log1p(np.clip(yr_s, 0, None)))
        pr = np.expm1(reg.predict(Xt_a))
        pr = np.clip(pr, 0, None)

        reg_rows.append({
            'test_network': net,
            'mae': float(mean_absolute_error(yr_t, pr)),
            'rmse': float(mean_squared_error(yr_t, pr) ** 0.5),
            'r2': float(r2_score(yr_t, pr)),
        })

    cls_df = pd.DataFrame(cls_rows)
    reg_df = pd.DataFrame(reg_rows)
    cls_df.to_csv(OUT / 'top3_coral_cls_by_network.csv', index=False)
    reg_df.to_csv(OUT / 'top3_coral_reg_by_network.csv', index=False)

    summary = {
        'classification_mean': cls_df[['auc', 'ap', 'f1', 'precision', 'recall']].mean().to_dict(),
        'regression_mean': reg_df[['mae', 'rmse', 'r2']].mean().to_dict(),
    }
    (OUT / 'top3_coral_summary.json').write_text(json.dumps(summary, indent=2))

    print('=== TOP3 CORAL SUMMARY ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
