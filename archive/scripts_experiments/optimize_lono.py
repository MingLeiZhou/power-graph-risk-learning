#!/usr/bin/env python3
"""Further optimization for cross-topology (LONO):
1) class imbalance handling + threshold search
2) log1p regression target
3) report improvements
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, mean_squared_error, mean_absolute_error, r2_score

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def best_threshold_by_f1(y_true, y_score):
    p, r, th = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    # thresholds has len n-1 vs p/r n
    idx = int(np.nanargmax(f1[:-1])) if len(th) > 0 else 0
    return (th[idx] if len(th) > 0 else 0.5), float(f1[:-1][idx] if len(th) > 0 else 0.0)


def lono_optimized(df, feature_cols):
    nets = sorted(df['network'].unique())
    cls_rows, reg_rows = [], []

    for net in nets:
        tr = df[df.network != net]
        te = df[df.network == net]
        Xtr = tr[feature_cols].astype(float)
        Xte = te[feature_cols].astype(float)

        # ---- Classification with stronger imbalance handling ----
        ytr = tr['y_cls'].astype(int)
        yte = te['y_cls'].astype(int)
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )
        clf.fit(Xtr, ytr)
        s_tr = clf.predict_proba(Xtr)[:, 1]
        s_te = clf.predict_proba(Xte)[:, 1]

        th, train_best_f1 = best_threshold_by_f1(ytr.values, s_tr)
        p_te = (s_te >= th).astype(int)

        cls_rows.append({
            'test_network': net,
            'auc': float(roc_auc_score(yte, s_te)),
            'f1': float(f1_score(yte, p_te, zero_division=0)),
            'threshold': float(th),
            'train_best_f1': float(train_best_f1),
            'pos_rate_test': float(yte.mean())
        })

        # ---- Regression with log1p target ----
        ytr_r = tr['y_reg'].astype(float)
        yte_r = te['y_reg'].astype(float)
        ytr_log = np.log1p(np.clip(ytr_r.values, 0, None))

        reg = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        reg.fit(Xtr, ytr_log)
        pred_log = reg.predict(Xte)
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)

        reg_rows.append({
            'test_network': net,
            'mae': float(mean_absolute_error(yte_r, pred)),
            'rmse': float(mean_squared_error(yte_r, pred) ** 0.5),
            'r2': float(r2_score(yte_r, pred))
        })

    return pd.DataFrame(cls_rows), pd.DataFrame(reg_rows)


def main():
    df = pd.read_parquet(DATA)
    feats = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]

    cls_df, reg_df = lono_optimized(df, feats)
    cls_df.to_csv(OUT / 'optimized_lono_classification.csv', index=False)
    reg_df.to_csv(OUT / 'optimized_lono_regression.csv', index=False)

    summary = {
        'cls_mean': cls_df[['auc', 'f1']].mean().to_dict(),
        'reg_mean': reg_df[['mae', 'rmse', 'r2']].mean().to_dict(),
        'cls_by_network': cls_df.to_dict(orient='records'),
        'reg_by_network': reg_df.to_dict(orient='records')
    }
    (OUT / 'optimized_lono_summary.json').write_text(json.dumps(summary, indent=2))

    print('=== OPTIMIZED LONO SUMMARY ===')
    print('Classification mean:', summary['cls_mean'])
    print('Regression mean    :', summary['reg_mean'])
    print('\nBy network (classification):')
    print(cls_df.to_string(index=False))
    print('\nBy network (regression):')
    print(reg_df.to_string(index=False))
    print('\nSaved to', OUT)


if __name__ == '__main__':
    main()
