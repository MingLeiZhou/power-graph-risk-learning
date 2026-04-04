#!/usr/bin/env python3
"""Continue training: cross-topology generalization evaluation.
Leave-one-network-out for classification/regression baselines.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(DATA)
    feats = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]
    nets = sorted(df['network'].unique())
    print('Networks:', nets)

    cls_rows, reg_rows = [], []

    for test_net in nets:
        print(f'\n[Fold] test network = {test_net}')
        train_df = df[df.network != test_net]
        test_df = df[df.network == test_net]

        Xtr, Xte = train_df[feats], test_df[feats]

        # classification
        ytr_c, yte_c = train_df['y_cls'].astype(int), test_df['y_cls'].astype(int)
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=500, class_weight='balanced'))
        ])
        clf.fit(Xtr, ytr_c)
        p = clf.predict(Xte)
        s = clf.predict_proba(Xte)[:,1]
        auc = roc_auc_score(yte_c, s)
        f1 = f1_score(yte_c, p, zero_division=0)
        acc = accuracy_score(yte_c, p)
        cls_rows.append({'test_network': test_net, 'auc': auc, 'f1': f1, 'acc': acc, 'pos_rate_test': float(yte_c.mean())})

        # regression
        ytr_r, yte_r = train_df['y_reg'].astype(float), test_df['y_reg'].astype(float)
        reg = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))])
        reg.fit(Xtr, ytr_r)
        pred = reg.predict(Xte)
        mae = mean_absolute_error(yte_r, pred)
        rmse = mean_squared_error(yte_r, pred) ** 0.5
        r2 = r2_score(yte_r, pred)
        reg_rows.append({'test_network': test_net, 'mae': mae, 'rmse': rmse, 'r2': r2})

        print(f'  CLS auc={auc:.4f} f1={f1:.4f} acc={acc:.4f}')
        print(f'  REG rmse={rmse:.4f} r2={r2:.4f}')

    cls_df = pd.DataFrame(cls_rows)
    reg_df = pd.DataFrame(reg_rows)
    cls_df.to_csv(OUT / 'generalization_classification_lono.csv', index=False)
    reg_df.to_csv(OUT / 'generalization_regression_lono.csv', index=False)

    print('\n=== LONO SUMMARY ===')
    print('\nClassification by held-out network:')
    print(cls_df.to_string(index=False))
    print('\nMean metrics:', cls_df[['auc','f1','acc']].mean().to_dict())

    print('\nRegression by held-out network:')
    print(reg_df.to_string(index=False))
    print('\nMean metrics:', reg_df[['mae','rmse','r2']].mean().to_dict())
    print('\nSaved to', OUT)


if __name__ == '__main__':
    main()
