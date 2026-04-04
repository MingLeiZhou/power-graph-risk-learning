#!/usr/bin/env python3
"""F1-focused LONO optimization with sample weighting + threshold policy.
Produces paper-ready ablation table.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)


def make_weights(y, boost=6.0):
    w = np.ones(len(y), dtype=float)
    w[y == 1] = boost
    return w


def threshold_policy_from_train(scores, y):
    # robust threshold policy: maximize F1 on train with constraints
    grid = np.array([0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40,0.50])
    best = None
    for th in grid:
        p = (scores >= th).astype(int)
        f1 = f1_score(y, p, zero_division=0)
        rc = recall_score(y, p, zero_division=0)
        pr = precision_score(y, p, zero_division=0)
        # constrain to avoid degenerate ultra-low precision policy
        if rc < 0.20:
            continue
        score = f1 - 0.05 * max(0, 0.10 - pr)  # slight penalty when precision too tiny
        cur = (score, th, f1, pr, rc)
        if best is None or cur[0] > best[0]:
            best = cur
    if best is None:
        return 0.20
    return float(best[1])


def run_variant(df, weighted=False, netnorm=False, name='variant'):
    work = df.copy()
    feat_cols = [c for c in work.columns if c.startswith('ef_') or c.startswith('efnc_')]

    if netnorm:
        for c in feat_cols:
            g = work.groupby('network')[c]
            mu = g.transform('mean')
            sd = g.transform('std').replace(0, 1e-6)
            work[c] = (work[c] - mu) / sd

    nets = sorted(work.network.unique())
    rows = []

    for net in nets:
        tr = work[work.network != net]
        te = work[work.network == net]
        Xtr, Xte = tr[feat_cols].astype(float), te[feat_cols].astype(float)
        ytr, yte = tr.y_cls.astype(int).values, te.y_cls.astype(int).values

        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample' if not weighted else None,
        )

        if weighted:
            w = make_weights(ytr, boost=8.0)
            clf.fit(Xtr, ytr, sample_weight=w)
        else:
            clf.fit(Xtr, ytr)

        s_tr = clf.predict_proba(Xtr)[:, 1]
        s_te = clf.predict_proba(Xte)[:, 1]

        th = threshold_policy_from_train(s_tr, ytr)
        p = (s_te >= th).astype(int)

        rows.append({
            'variant': name,
            'test_network': net,
            'threshold': th,
            'auc': float(roc_auc_score(yte, s_te)),
            'f1': float(f1_score(yte, p, zero_division=0)),
            'precision': float(precision_score(yte, p, zero_division=0)),
            'recall': float(recall_score(yte, p, zero_division=0)),
            'acc': float(accuracy_score(yte, p)),
            'pos_rate_test': float(yte.mean()),
        })

    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(DATA)

    v1 = run_variant(df, weighted=False, netnorm=False, name='rf_balanced')
    v2 = run_variant(df, weighted=True,  netnorm=False, name='rf_sample_weighted')
    v3 = run_variant(df, weighted=True,  netnorm=True,  name='rf_weighted_netnorm')

    all_df = pd.concat([v1, v2, v3], ignore_index=True)
    all_df.to_csv(OUT / 'f1_boost_lono_by_network.csv', index=False)

    summ = all_df.groupby('variant')[['auc','f1','precision','recall','acc']].mean().reset_index()
    summ.to_csv(OUT / 'f1_boost_lono_summary.csv', index=False)

    # select best by mean F1 then AUC
    best = summ.sort_values(['f1','auc'], ascending=False).iloc[0]

    md = []
    md.append('# F1 Boost LONO Report\n')
    md.append('## Mean metrics by variant\n')
    md.append('```\n' + summ.to_string(index=False) + '\n```\n')
    md.append('## By network\n')
    md.append('```\n' + all_df.to_string(index=False) + '\n```\n')
    md.append(f"Best variant (by F1 then AUC): {best['variant']}\n")
    (OUT / 'f1_boost_lono_report.md').write_text('\n'.join(md))

    out = {
        'best_variant': str(best['variant']),
        'summary': summ.to_dict(orient='records')
    }
    (OUT / 'f1_boost_lono_summary.json').write_text(json.dumps(out, indent=2))

    print('=== F1 BOOST SUMMARY ===')
    print(summ.to_string(index=False))
    print('\nBest variant:', best['variant'])
    print('Saved to', OUT)


if __name__ == '__main__':
    main()
