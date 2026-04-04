#!/usr/bin/env python3
"""Execute 1/2/3 in sequence, run ablation keep/remove, then final retrain.
1) Two-stage detector (ranking + domain calibration)
2) Unsupervised target thresholding (GMM on target scores)
3) Sample-level transfer (supervised encoder)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.mixture import GaussianMixture

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def centroid_weights(train_df, feat_cols):
    nets = sorted(train_df.network.unique())
    C = {n: train_df[train_df.network == n][feat_cols].mean().values for n in nets}
    global_c = train_df[feat_cols].mean().values
    ws = {}
    for n in nets:
        d = float(np.linalg.norm(global_c - C[n]))
        ws[n] = 1.0 / max(d, 1e-6)
    z = sum(ws.values())
    return {k: v / z for k, v in ws.items()}


def gmm_threshold(scores):
    s = scores.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=42)
    gm.fit(s)
    m = gm.means_.flatten()
    m1, m2 = float(np.min(m)), float(np.max(m))
    # midpoint in score space (robust simple approximation)
    return float((m1 + m2) / 2.0)


def train_encoder_features(Xtr, Xte):
    # sample-level transfer surrogate: supervised encoder via MLP autoencoder bottleneck
    ae = MLPRegressor(hidden_layer_sizes=(128, 32, 128), max_iter=60, random_state=42, early_stopping=True)
    ae.fit(Xtr, Xtr)
    W1, W2 = ae.coefs_[0], ae.coefs_[1]
    b1, b2 = ae.intercepts_[0], ae.intercepts_[1]
    Htr = np.maximum(0, Xtr @ W1 + b1)
    Ztr = np.maximum(0, Htr @ W2 + b2)
    Hte = np.maximum(0, Xte @ W1 + b1)
    Zte = np.maximum(0, Hte @ W2 + b2)
    return Ztr, Zte


def eval_fold(tr, te, feat_cols, mode):
    Xtr = tr[feat_cols].astype(float).values
    Xte = te[feat_cols].astype(float).values
    ytr = tr.y_cls.astype(int).values
    yte = te.y_cls.astype(int).values
    yrtr = tr.y_reg.astype(float).values
    yrte = te.y_reg.astype(float).values

    if mode == 'm3_sample_transfer':
        Xtr_use, Xte_use = train_encoder_features(Xtr, Xte)
    else:
        Xtr_use, Xte_use = Xtr, Xte

    # ranking model
    w = np.ones(len(ytr)); w[ytr == 1] = 10.0
    clf = RandomForestClassifier(n_estimators=350, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(Xtr_use, ytr, sample_weight=w)
    s_tr = clf.predict_proba(Xtr_use)[:, 1]
    s_te = clf.predict_proba(Xte_use)[:, 1]

    if mode == 'm1_two_stage':
        # per-domain calibration + weighted blend
        src_nets = sorted(tr.network.unique())
        ws = centroid_weights(tr, feat_cols)
        p_cal = np.zeros_like(s_te)
        for sn in src_nets:
            m = (tr.network == sn).values
            lr = LogisticRegression(max_iter=300)
            try:
                lr.fit(s_tr[m].reshape(-1, 1), ytr[m])
                p_cal += ws[sn] * lr.predict_proba(s_te.reshape(-1, 1))[:, 1]
            except Exception:
                p_cal += ws[sn] * s_te
        s = p_cal
        # threshold selected by source prior budget
        q = min(0.30, max(0.02, float(ytr.mean()) * 1.5))
        th = float(np.quantile(s_tr, 1 - q))
    elif mode == 'm2_unsup_threshold':
        s = s_te
        th = gmm_threshold(s)
    elif mode == 'm3_sample_transfer':
        s = s_te
        q = min(0.30, max(0.02, float(ytr.mean()) * 1.5))
        th = float(np.quantile(s_tr, 1 - q))
    else:
        s = s_te
        q = min(0.30, max(0.02, float(ytr.mean()) * 1.5))
        th = float(np.quantile(s_tr, 1 - q))

    p = (s >= th).astype(int)

    cls = {
        'auc': float(roc_auc_score(yte, s)),
        'ap': float(average_precision_score(yte, s)),
        'f1': float(f1_score(yte, p, zero_division=0)),
        'precision': float(precision_score(yte, p, zero_division=0)),
        'recall': float(recall_score(yte, p, zero_division=0)),
    }

    # regression shared strong baseline
    reg = ExtraTreesRegressor(n_estimators=350, max_depth=20, random_state=42, n_jobs=-1)
    reg.fit(Xtr_use, np.log1p(np.clip(yrtr, 0, None)))
    pr = np.expm1(reg.predict(Xte_use)); pr = np.clip(pr, 0, None)
    rg = {
        'mae': float(mean_absolute_error(yrte, pr)),
        'rmse': float(mean_squared_error(yrte, pr) ** 0.5),
        'r2': float(r2_score(yrte, pr)),
    }
    return cls, rg


def run_mode(df, feat_cols, mode):
    cls_rows, reg_rows = [], []
    for net in sorted(df.network.unique()):
        tr = df[df.network != net].copy()
        te = df[df.network == net].copy()
        cls, rg = eval_fold(tr, te, feat_cols, mode)
        cls_rows.append({'mode': mode, 'test_network': net, **cls})
        reg_rows.append({'mode': mode, 'test_network': net, **rg})
    return pd.DataFrame(cls_rows), pd.DataFrame(reg_rows)


def main():
    df = pd.read_parquet(DATA)
    # keep runtime tractable while preserving imbalance
    if len(df) > 140000:
        df = pd.concat([
            df[df.y_cls == 1],
            df[df.y_cls == 0].sample(n=120000, random_state=42)
        ], ignore_index=True)

    feat_cols = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]

    modes = ['m1_two_stage', 'm2_unsup_threshold', 'm3_sample_transfer']
    all_cls, all_reg = [], []
    for m in modes:
        print('running', m, flush=True)
        c, r = run_mode(df, feat_cols, m)
        all_cls.append(c); all_reg.append(r)

    cls_df = pd.concat(all_cls, ignore_index=True)
    reg_df = pd.concat(all_reg, ignore_index=True)
    cls_df.to_csv(OUT / 'seq123_cls_by_network.csv', index=False)
    reg_df.to_csv(OUT / 'seq123_reg_by_network.csv', index=False)

    cls_sum = cls_df.groupby('mode')[['auc','ap','f1','precision','recall']].mean().reset_index()
    reg_sum = reg_df.groupby('mode')[['mae','rmse','r2']].mean().reset_index()

    # keep/remove decision (benefit if top-2 on any key metric)
    keep = []
    for m in modes:
        row = cls_sum[cls_sum['mode'] == m].iloc[0]
        score = row['f1'] * 0.5 + row['ap'] * 0.3 + row['auc'] * 0.2
        keep.append((m, score))
    keep = sorted(keep, key=lambda x: x[1], reverse=True)
    kept_modes = [keep[0][0], keep[1][0]]
    removed_modes = [m for m in modes if m not in kept_modes]

    # final retrain = best kept mode (single robust winner)
    best_mode = kept_modes[0]
    final_cls, final_reg = run_mode(df, feat_cols, best_mode)
    final_cls.to_csv(OUT / 'final_retrain_cls_by_network.csv', index=False)
    final_reg.to_csv(OUT / 'final_retrain_reg_by_network.csv', index=False)

    final_summary = {
        'best_mode': best_mode,
        'kept_modes': kept_modes,
        'removed_modes': removed_modes,
        'seq123_cls_summary': cls_sum.to_dict(orient='records'),
        'seq123_reg_summary': reg_sum.to_dict(orient='records'),
        'final_cls_mean': final_cls[['auc','ap','f1','precision','recall']].mean().to_dict(),
        'final_reg_mean': final_reg[['mae','rmse','r2']].mean().to_dict(),
    }

    (OUT / 'seq123_final_summary.json').write_text(json.dumps(final_summary, indent=2))

    md = []
    md.append('# Sequence 1-2-3 Execution + Ablation + Final Retrain\n')
    md.append('## Classification summary by mode\n')
    md.append('```\n' + cls_sum.to_string(index=False) + '\n```\n')
    md.append('## Regression summary by mode\n')
    md.append('```\n' + reg_sum.to_string(index=False) + '\n```\n')
    md.append(f"Kept modes: {kept_modes}\\n")
    md.append(f"Removed modes: {removed_modes}\\n")
    md.append(f"Final retrain mode: {best_mode}\\n")
    md.append('## Final mean metrics\n')
    md.append('```\n' + json.dumps({'classification': final_summary['final_cls_mean'], 'regression': final_summary['final_reg_mean']}, indent=2) + '\n```\n')
    (OUT / 'seq123_final_report.md').write_text('\n'.join(md))

    print(json.dumps(final_summary, indent=2))


if __name__ == '__main__':
    main()
