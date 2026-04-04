#!/usr/bin/env python3
"""Fast LONO optimization v2 with threshold sweep + log-reg target."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(DATA)
    feats = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]
    nets = sorted(df.network.unique())

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    cls_rows, reg_rows = [], []

    for net in nets:
        print(f'[LONO] test={net}', flush=True)
        tr = df[df.network != net]
        te = df[df.network == net]
        Xtr, Xte = tr[feats].astype(float), te[feats].astype(float)

        # classifier
        ytr, yte = tr.y_cls.astype(int), te.y_cls.astype(int)
        clf = RandomForestClassifier(
            n_estimators=220, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1, class_weight='balanced_subsample'
        )
        clf.fit(Xtr, ytr)
        s = clf.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, s))

        # threshold sweep on test (analysis upper bound, not deployment threshold)
        f1_best, th_best = -1.0, 0.5
        for th in thresholds:
            p = (s >= th).astype(int)
            f1 = f1_score(yte, p, zero_division=0)
            if f1 > f1_best:
                f1_best, th_best = float(f1), float(th)

        cls_rows.append({
            'test_network': net,
            'auc': auc,
            'best_f1_on_test_grid': f1_best,
            'best_threshold_on_grid': th_best,
            'pos_rate_test': float(yte.mean())
        })

        # regressor with log1p target
        ytr_r, yte_r = tr.y_reg.astype(float), te.y_reg.astype(float)
        ytr_log = np.log1p(np.clip(ytr_r.values, 0, None))
        reg = ExtraTreesRegressor(n_estimators=220, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)
        reg.fit(Xtr, ytr_log)
        pred = np.expm1(reg.predict(Xte))
        pred = np.clip(pred, 0, None)

        reg_rows.append({
            'test_network': net,
            'mae': float(mean_absolute_error(yte_r, pred)),
            'rmse': float(mean_squared_error(yte_r, pred) ** 0.5),
            'r2': float(r2_score(yte_r, pred))
        })

    cls_df = pd.DataFrame(cls_rows)
    reg_df = pd.DataFrame(reg_rows)
    cls_df.to_csv(OUT / 'optimized_lono_v2_classification.csv', index=False)
    reg_df.to_csv(OUT / 'optimized_lono_v2_regression.csv', index=False)

    summary = {
        'cls_mean': cls_df[['auc', 'best_f1_on_test_grid']].mean().to_dict(),
        'reg_mean': reg_df[['mae', 'rmse', 'r2']].mean().to_dict(),
    }
    (OUT / 'optimized_lono_v2_summary.json').write_text(json.dumps(summary, indent=2))

    print('\n=== OPTIMIZED V2 SUMMARY ===')
    print('Classification mean:', summary['cls_mean'])
    print('Regression mean    :', summary['reg_mean'])
    print('\nClassification detail:')
    print(cls_df.to_string(index=False))
    print('\nRegression detail:')
    print(reg_df.to_string(index=False))


if __name__ == '__main__':
    main()
