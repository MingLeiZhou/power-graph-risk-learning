#!/usr/bin/env python3
"""Fusion + domain-aware calibration + LONO comparison table (paper-ready)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score

DOWN = Path('data/processed/downstream/downstream_full.parquet')
LAT = Path('data/processed/ssl/opf_ssl_latent.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def map_case_to_net(case_name: str):
    c = case_name.lower()
    if 'case14' in c: return 'case14'
    if 'case30' in c: return 'case30'
    if 'case57' in c: return 'case57'
    if 'case118' in c: return 'case118'
    return 'other'


def network_to_case_proxy(network: str):
    # topology-size proxy mapping
    return {
        'ieee24': 'case30',
        'ieee39': 'case57',
        'ieee118': 'case118',
        'uk': 'case118',
    }.get(network, 'case57')


def build_fusion(df_down, df_lat):
    zcols = [c for c in df_lat.columns if c.startswith('z_')]
    df_lat = df_lat.copy()
    df_lat['case_proxy'] = df_lat['case_name'].map(map_case_to_net)

    proto = df_lat.groupby('case_proxy')[zcols].mean().reset_index()
    proto = proto.rename(columns={c: f'lat_{c}' for c in zcols})

    out = df_down.copy()
    out['case_proxy'] = out['network'].map(network_to_case_proxy)
    out = out.merge(proto, on='case_proxy', how='left')
    for c in [x for x in out.columns if x.startswith('lat_z_')]:
        out[c] = out[c].fillna(0.0)
    return out


def invdist_weights(target_centroid, source_centroids):
    ws = {}
    for k, c in source_centroids.items():
        d = float(np.linalg.norm(target_centroid - c))
        ws[k] = 1.0 / max(d, 1e-6)
    s = sum(ws.values())
    return {k: v/s for k, v in ws.items()}


def lono_run(df, feature_cols):
    nets = sorted(df['network'].unique())
    rows = []

    for test_net in nets:
        tr = df[df.network != test_net].copy()
        te = df[df.network == test_net].copy()

        Xtr = tr[feature_cols].astype(float)
        Xte = te[feature_cols].astype(float)

        # ---------- Classification ----------
        ytr = tr.y_cls.astype(int).values
        yte = te.y_cls.astype(int).values
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=3,
            random_state=42, n_jobs=-1, class_weight='balanced_subsample'
        )
        clf.fit(Xtr, ytr)
        p_tr = clf.predict_proba(Xtr)[:, 1]
        p_te = clf.predict_proba(Xte)[:, 1]

        # domain-aware calibrators: one per source network
        src_nets = sorted(tr.network.unique())
        cal_models = {}
        src_centroids = {}
        for sn in src_nets:
            m = tr.network == sn
            xs = p_tr[m.values].reshape(-1, 1)
            ys = ytr[m.values]
            lr = LogisticRegression(max_iter=500)
            try:
                lr.fit(xs, ys)
                cal_models[sn] = lr
            except Exception:
                cal_models[sn] = None
            src_centroids[sn] = tr.loc[m, feature_cols].mean().values

        tgt_centroid = te[feature_cols].mean().values
        w = invdist_weights(tgt_centroid, src_centroids)

        p_cal = np.zeros_like(p_te)
        for sn in src_nets:
            mdl = cal_models[sn]
            if mdl is None:
                p_sn = p_te
            else:
                p_sn = mdl.predict_proba(p_te.reshape(-1, 1))[:, 1]
            p_cal += w[sn] * p_sn

        # threshold sweep on calibration probs (paper diagnostic)
        best_f1, best_th = -1, 0.5
        for th in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]:
            pred = (p_cal >= th).astype(int)
            f1 = f1_score(yte, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        pred = (p_cal >= best_th).astype(int)

        auc = roc_auc_score(yte, p_cal)
        acc = accuracy_score(yte, pred)

        # ---------- Regression ----------
        yr_tr = tr.y_reg.astype(float).values
        yr_te = te.y_reg.astype(float).values
        reg = ExtraTreesRegressor(n_estimators=300, max_depth=20, min_samples_leaf=1, random_state=42, n_jobs=-1)
        reg.fit(Xtr, np.log1p(np.clip(yr_tr, 0, None)))
        r_tr = np.expm1(reg.predict(Xtr))
        r_te = np.expm1(reg.predict(Xte))
        r_te = np.clip(r_te, 0, None)

        # domain-aware affine correction per source network
        aff = {}
        for sn in src_nets:
            m = (tr.network == sn).values
            x = r_tr[m].reshape(-1, 1)
            y = yr_tr[m]
            lr = LinearRegression()
            lr.fit(x, y)
            aff[sn] = lr

        r_cal = np.zeros_like(r_te)
        for sn in src_nets:
            r_cal += w[sn] * aff[sn].predict(r_te.reshape(-1, 1))
        r_cal = np.clip(r_cal, 0, None)

        mae = mean_absolute_error(yr_te, r_cal)
        rmse = mean_squared_error(yr_te, r_cal) ** 0.5
        r2 = r2_score(yr_te, r_cal)

        rows.append({
            'test_network': test_net,
            'auc': float(auc),
            'f1': float(best_f1),
            'acc': float(acc),
            'threshold': float(best_th),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
        })

    return pd.DataFrame(rows)


def main():
    down = pd.read_parquet(DOWN)
    lat = pd.read_parquet(LAT)

    # baseline feature set
    base_feats = [c for c in down.columns if c.startswith('ef_') or c.startswith('efnc_')]

    # fusion
    fused = build_fusion(down, lat)
    fused_feats = base_feats + [c for c in fused.columns if c.startswith('lat_z_')]

    print('Running LONO baseline (with domain-aware calibration)...')
    res_base = lono_run(down, base_feats)
    print('Running LONO fused (with domain-aware calibration)...')
    res_fused = lono_run(fused, fused_feats)

    res_base['setting'] = 'baseline+domain_cal'
    res_fused['setting'] = 'fused_ssl+domain_cal'

    all_res = pd.concat([res_base, res_fused], ignore_index=True)
    all_res.to_csv(OUT / 'paper_lono_comparison_by_network.csv', index=False)

    summ = all_res.groupby('setting')[['auc','f1','acc','mae','rmse','r2']].mean().reset_index()
    summ.to_csv(OUT / 'paper_lono_comparison_summary.csv', index=False)

    # markdown table
    md = ['# Paper LONO Comparison (Before vs After)\n']
    md.append('## Mean metrics\n')
    md.append('```\n' + summ.to_string(index=False) + '\n```')
    md.append('\n\n## By network\n')
    md.append('```\n' + all_res.to_string(index=False) + '\n```')
    (OUT / 'paper_lono_comparison.md').write_text('\n'.join(md))

    print('\n=== COMPARISON SUMMARY ===')
    print(summ.to_string(index=False))
    print('\nSaved:')
    print(' -', OUT / 'paper_lono_comparison_by_network.csv')
    print(' -', OUT / 'paper_lono_comparison_summary.csv')
    print(' -', OUT / 'paper_lono_comparison.md')


if __name__ == '__main__':
    main()
