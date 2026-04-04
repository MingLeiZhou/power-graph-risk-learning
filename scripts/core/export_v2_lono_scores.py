#!/usr/bin/env python3
"""Export sample-level LONO classification scores for ROC/PR plotting."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DATA = Path('data/processed/downstream/downstream_v2_informative.parquet')
OUT = Path('analysis/training/v2_lono_scores.parquet')


def main():
    df = pd.read_parquet(DATA)
    feat = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_') or c.endswith('_tshift')]
    feat += [c for c in ['n_samples', 'pos_rate', 'yreg_mean', 'yreg_std'] if c in df.columns]

    rows = []
    for net in sorted(df.network.unique()):
        tr = df[df.network != net]
        te = df[df.network == net]

        Xtr = tr[feat].astype(float).values
        Xte = te[feat].astype(float).values
        ytr = tr['y_cls_v2'].astype(int).values

        w = np.ones(len(ytr))
        w[ytr == 1] = 10.0
        clf = RandomForestClassifier(n_estimators=220, max_depth=20, random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr, sample_weight=w)
        s = clf.predict_proba(Xte)[:, 1]

        out = te[['network', 'sample_idx', 'y_cls_v2']].copy()
        out['y_true'] = out['y_cls_v2'].astype(int)
        out['y_score'] = s
        out = out.drop(columns=['y_cls_v2'])
        rows.append(out)

    score_df = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_parquet(OUT, index=False)
    print(f'saved {OUT} rows={len(score_df)}')


if __name__ == '__main__':
    main()
