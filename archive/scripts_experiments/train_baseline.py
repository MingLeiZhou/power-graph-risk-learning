#!/usr/bin/env python3
"""Train baseline models for early warning (classification) and risk (regression)."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import json

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print('Loading dataset...')
    df = pd.read_parquet(DATA)
    feature_cols = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]
    X = df[feature_cols].astype(float)

    # ---- Classification ----
    y_cls = df['y_cls'].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

    print('Training classification baselines...')
    clf_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, class_weight='balanced'))
    ])
    clf_rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced_subsample')

    clf_lr.fit(X_tr, y_tr)
    clf_rf.fit(X_tr, y_tr)

    def eval_clf(model, name):
        p = model.predict(X_te)
        if hasattr(model, 'predict_proba'):
            s = model.predict_proba(X_te)[:,1]
        else:
            s = p
        auc = roc_auc_score(y_te, s)
        f1 = f1_score(y_te, p)
        acc = accuracy_score(y_te, p)
        pr, rc, f1b, _ = precision_recall_fscore_support(y_te, p, average='binary', zero_division=0)
        return {'model': name, 'auc': float(auc), 'f1': float(f1), 'acc': float(acc), 'precision': float(pr), 'recall': float(rc)}

    cls_results = [eval_clf(clf_lr, 'logistic_regression'), eval_clf(clf_rf, 'random_forest')]

    # ---- Regression ----
    y_reg = df['y_reg'].astype(float)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    print('Training regression baselines...')
    reg_ridge = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))])
    reg_rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    reg_ridge.fit(Xr_tr, yr_tr)
    reg_rf.fit(Xr_tr, yr_tr)

    def eval_reg(model, name):
        pred = model.predict(Xr_te)
        mae = mean_absolute_error(yr_te, pred)
        rmse = mean_squared_error(yr_te, pred) ** 0.5
        r2 = r2_score(yr_te, pred)
        return {'model': name, 'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}

    reg_results = [eval_reg(reg_ridge, 'ridge'), eval_reg(reg_rf, 'random_forest')]

    # save models
    joblib.dump(clf_rf, OUT / 'clf_random_forest.joblib')
    joblib.dump(reg_rf, OUT / 'reg_random_forest.joblib')

    # save reports
    pd.DataFrame(cls_results).to_csv(OUT / 'classification_metrics.csv', index=False)
    pd.DataFrame(reg_results).to_csv(OUT / 'regression_metrics.csv', index=False)

    summary = {
        'n_samples': int(len(df)),
        'n_features': int(len(feature_cols)),
        'positive_rate': float(y_cls.mean()),
        'classification': cls_results,
        'regression': reg_results,
    }
    (OUT / 'training_summary.json').write_text(json.dumps(summary, indent=2))

    print('\n=== TRAINING SUMMARY ===')
    print('samples:', len(df), 'features:', len(feature_cols), 'positive_rate:', y_cls.mean())
    print('\nClassification:')
    print(pd.DataFrame(cls_results).to_string(index=False))
    print('\nRegression:')
    print(pd.DataFrame(reg_results).to_string(index=False))
    print('\nSaved to', OUT)


if __name__ == '__main__':
    main()
