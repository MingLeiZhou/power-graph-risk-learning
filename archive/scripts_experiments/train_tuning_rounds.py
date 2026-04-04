#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def get_features(df):
    return [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]


def add_network_normalized(df, feature_cols):
    out = df.copy()
    for c in feature_cols:
        g = out.groupby('network')[c]
        mu = g.transform('mean')
        sd = g.transform('std').replace(0, 1e-6)
        out[c] = (out[c] - mu) / sd
    return out


def build_clf(name, params):
    if name == 'logreg':
        return Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=800, class_weight='balanced', **params))])
    if name == 'rf':
        return RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample', **params)
    if name == 'et':
        return ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight='balanced', **params)
    raise ValueError(name)


def build_reg(name, params):
    if name == 'ridge':
        return Pipeline([('scaler', StandardScaler()), ('reg', Ridge(**params))])
    if name == 'rf':
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if name == 'et':
        return ExtraTreesRegressor(random_state=42, n_jobs=-1, **params)
    raise ValueError(name)


def tune_round(df, round_name, max_rows=120000):
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        print(f'[{round_name}] sampled rows: {len(df)}', flush=True)

    feats = get_features(df)
    X = df[feats].astype(float)
    y_cls = df['y_cls'].astype(int)
    y_reg = df['y_reg'].astype(float)

    Xtr, Xte, ytr_c, yte_c, ytr_r, yte_r = train_test_split(X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls)
    Xtr2, Xva, ytr2_c, yva_c, ytr2_r, yva_r = train_test_split(Xtr, ytr_c, ytr_r, test_size=0.25, random_state=42, stratify=ytr_c)

    clf_space = {
        'logreg': {'C': [0.5, 1.5]},
        'rf': {'n_estimators': [250], 'max_depth': [20], 'min_samples_leaf': [1, 3]},
        'et': {'n_estimators': [250], 'max_depth': [20], 'min_samples_leaf': [1, 3]},
    }
    reg_space = {
        'ridge': {'alpha': [0.5, 2.0]},
        'rf': {'n_estimators': [250], 'max_depth': [20], 'min_samples_leaf': [1, 3]},
        'et': {'n_estimators': [250], 'max_depth': [20], 'min_samples_leaf': [1, 3]},
    }

    best_clf, best_auc = None, -1
    clf_rows = []
    for m, grid in clf_space.items():
        for p in ParameterGrid(grid):
            print(f'[{round_name}] clf {m} {p}', flush=True)
            model = build_clf(m, p)
            model.fit(Xtr2, ytr2_c)
            pred = model.predict(Xva)
            score = model.predict_proba(Xva)[:,1] if hasattr(model, 'predict_proba') else pred
            met = {'auc': roc_auc_score(yva_c, score), 'f1': f1_score(yva_c, pred, zero_division=0), 'acc': accuracy_score(yva_c, pred)}
            row = {'round': round_name, 'model': m, **p, **{k: float(v) for k,v in met.items()}}
            clf_rows.append(row)
            if met['auc'] > best_auc:
                best_auc = met['auc']
                best_clf = (m, p)

    best_reg, best_rmse = None, 1e18
    reg_rows = []
    for m, grid in reg_space.items():
        for p in ParameterGrid(grid):
            print(f'[{round_name}] reg {m} {p}', flush=True)
            model = build_reg(m, p)
            model.fit(Xtr2, ytr2_r)
            pr = model.predict(Xva)
            met = {'mae': mean_absolute_error(yva_r, pr), 'rmse': mean_squared_error(yva_r, pr)**0.5, 'r2': r2_score(yva_r, pr)}
            row = {'round': round_name, 'model': m, **p, **{k: float(v) for k,v in met.items()}}
            reg_rows.append(row)
            if met['rmse'] < best_rmse:
                best_rmse = met['rmse']
                best_reg = (m, p)

    # test
    cm, cp = best_clf
    clf = build_clf(cm, cp)
    clf.fit(Xtr, ytr_c)
    pred = clf.predict(Xte)
    score = clf.predict_proba(Xte)[:,1] if hasattr(clf, 'predict_proba') else pred
    test_cls = {'round': round_name, 'model': cm, 'params': cp,
                'auc': float(roc_auc_score(yte_c, score)), 'f1': float(f1_score(yte_c, pred, zero_division=0)), 'acc': float(accuracy_score(yte_c, pred))}

    rm, rp = best_reg
    reg = build_reg(rm, rp)
    reg.fit(Xtr, ytr_r)
    pr = reg.predict(Xte)
    test_reg = {'round': round_name, 'model': rm, 'params': rp,
                'mae': float(mean_absolute_error(yte_r, pr)), 'rmse': float(mean_squared_error(yte_r, pr)**0.5), 'r2': float(r2_score(yte_r, pr))}

    return pd.DataFrame(clf_rows), pd.DataFrame(reg_rows), test_cls, test_reg, best_clf, best_reg


def lono(df, clf_cfg, reg_cfg, round_name):
    feats = get_features(df)
    nets = sorted(df.network.unique())
    cm, cp = clf_cfg
    rm, rp = reg_cfg
    cls_rows, reg_rows = [], []
    for net in nets:
        print(f'[{round_name}] LONO test {net}', flush=True)
        tr = df[df.network != net]
        te = df[df.network == net]
        Xtr, Xte = tr[feats].astype(float), te[feats].astype(float)

        ytrc, ytec = tr.y_cls.astype(int), te.y_cls.astype(int)
        clf = build_clf(cm, cp); clf.fit(Xtr, ytrc)
        p = clf.predict(Xte)
        s = clf.predict_proba(Xte)[:,1] if hasattr(clf,'predict_proba') else p
        cls_rows.append({'round': round_name, 'test_network': net, 'auc': float(roc_auc_score(ytec, s)), 'f1': float(f1_score(ytec, p, zero_division=0)), 'acc': float(accuracy_score(ytec, p))})

        ytrr, yter = tr.y_reg.astype(float), te.y_reg.astype(float)
        reg = build_reg(rm, rp); reg.fit(Xtr, ytrr)
        pr = reg.predict(Xte)
        reg_rows.append({'round': round_name, 'test_network': net, 'mae': float(mean_absolute_error(yter, pr)), 'rmse': float(mean_squared_error(yter, pr)**0.5), 'r2': float(r2_score(yter, pr))})

    return pd.DataFrame(cls_rows), pd.DataFrame(reg_rows)


def main():
    df = pd.read_parquet(DATA)

    print('\n=== Round 1 baseline ===', flush=True)
    r1c, r1r, r1tc, r1tr, bclf1, breg1 = tune_round(df, 'round1_baseline')

    print('\n=== Round 2 network-normalized ===', flush=True)
    dfn = add_network_normalized(df, get_features(df))
    r2c, r2r, r2tc, r2tr, bclf2, breg2 = tune_round(dfn, 'round2_netnorm')

    # choose by random split
    choose_round = 'round1_baseline' if r1tc['auc'] >= r2tc['auc'] else 'round2_netnorm'
    choose_df = df if choose_round == 'round1_baseline' else dfn
    choose_clf = bclf1 if choose_round == 'round1_baseline' else bclf2
    choose_reg = breg1 if choose_round == 'round1_baseline' else breg2

    print('\n=== Round 3 LONO ===', flush=True)
    l3c, l3r = lono(choose_df, choose_clf, choose_reg, 'round3_lono')

    r1c.to_csv(OUT/'tuning_round1_classification.csv', index=False)
    r1r.to_csv(OUT/'tuning_round1_regression.csv', index=False)
    r2c.to_csv(OUT/'tuning_round2_classification.csv', index=False)
    r2r.to_csv(OUT/'tuning_round2_regression.csv', index=False)
    l3c.to_csv(OUT/'tuning_round3_lono_classification.csv', index=False)
    l3r.to_csv(OUT/'tuning_round3_lono_regression.csv', index=False)

    summary = {
        'round1_test': {'cls': r1tc, 'reg': r1tr},
        'round2_test': {'cls': r2tc, 'reg': r2tr},
        'selected_round': choose_round,
        'selected_clf': {'model': choose_clf[0], 'params': choose_clf[1]},
        'selected_reg': {'model': choose_reg[0], 'params': choose_reg[1]},
        'lono_cls_mean': l3c[['auc','f1','acc']].mean().to_dict(),
        'lono_reg_mean': l3r[['mae','rmse','r2']].mean().to_dict(),
    }
    (OUT/'tuning_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
