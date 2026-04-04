#!/usr/bin/env python3
"""LONO with prior-aware thresholding and ranking metrics.
Focus: improve actionable detection under severe imbalance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def recall_at_k(y_true, scores, k_frac=0.05):
    n = len(y_true)
    k = max(1, int(n * k_frac))
    idx = np.argsort(scores)[::-1][:k]
    pos = y_true.sum()
    if pos <= 0:
        return 0.0
    return float(y_true[idx].sum() / pos)


def weighted_source_prior(train_df, feat_cols, target_net):
    src = sorted(train_df.network.unique())
    tgt_c = train_df[train_df.network == src[0]][feat_cols].mean().values * 0
    # for target centroid, use nearest source proxy by name fallback overall mean
    # we do not have target samples in strict deploy; use global centroid proxy
    tgt_c = train_df[feat_cols].mean().values
    ws = {}
    for s in src:
        c = train_df[train_df.network == s][feat_cols].mean().values
        d = np.linalg.norm(tgt_c - c)
        ws[s] = 1.0 / max(d, 1e-6)
    z = sum(ws.values())
    ws = {k: v/z for k, v in ws.items()}
    prior = 0.0
    for s in src:
        prior += ws[s] * float((train_df[train_df.network == s].y_cls == 1).mean())
    return float(prior)


def main():
    df = pd.read_parquet(DATA)
    feats = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]
    nets = sorted(df.network.unique())

    rows = []
    for net in nets:
        tr = df[df.network != net].copy()
        te = df[df.network == net].copy()
        Xtr, Xte = tr[feats].astype(float), te[feats].astype(float)
        ytr, yte = tr.y_cls.astype(int).values, te.y_cls.astype(int).values

        # stronger positive weighting
        w = np.ones(len(ytr), dtype=float)
        w[ytr == 1] = 12.0

        clf = RandomForestClassifier(
            n_estimators=700,
            max_depth=22,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr, sample_weight=w)
        s = clf.predict_proba(Xte)[:, 1]

        # prior-aware threshold by target positive budget (ranking cut)
        src_prior = weighted_source_prior(tr, feats, net)
        # conservative multiplier to avoid all-negative prediction
        q = min(0.25, max(0.02, src_prior * 1.6))
        thr = float(np.quantile(s, 1 - q))
        p = (s >= thr).astype(int)

        rows.append({
            'test_network': net,
            'prior_est': float(src_prior),
            'pred_pos_rate': float(p.mean()),
            'threshold': thr,
            'auc': float(roc_auc_score(yte, s)),
            'ap': float(average_precision_score(yte, s)),
            'f1': float(f1_score(yte, p, zero_division=0)),
            'precision': float(precision_score(yte, p, zero_division=0)),
            'recall': float(recall_score(yte, p, zero_division=0)),
            'recall_at_5pct': recall_at_k(yte, s, 0.05),
            'recall_at_10pct': recall_at_k(yte, s, 0.10),
            'pos_rate_test': float(yte.mean()),
        })

    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'lono_prior_threshold_by_network.csv', index=False)
    summary = res[['auc','ap','f1','precision','recall','recall_at_5pct','recall_at_10pct']].mean().to_dict()
    pd.DataFrame([summary]).to_csv(OUT / 'lono_prior_threshold_summary.csv', index=False)
    (OUT / 'lono_prior_threshold_summary.json').write_text(json.dumps(summary, indent=2))

    print('=== LONO PRIOR-THRESHOLD SUMMARY ===')
    print(pd.DataFrame([summary]).to_string(index=False))
    print('\nBy network:')
    print(res.to_string(index=False))


if __name__ == '__main__':
    main()
