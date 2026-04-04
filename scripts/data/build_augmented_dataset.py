#!/usr/bin/env python3
"""Build augmented + aligned downstream dataset for cross-topology robustness."""
from pathlib import Path
import numpy as np
import pandas as pd

INP = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('data/processed/downstream/downstream_augmented.parquet')


def main():
    rng = np.random.default_rng(42)
    df = pd.read_parquet(INP).copy()
    feat_cols = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]

    # 1) robust per-network normalization
    for c in feat_cols:
        g = df.groupby('network')[c]
        med = g.transform('median')
        q1 = g.transform(lambda s: s.quantile(0.25))
        q3 = g.transform(lambda s: s.quantile(0.75))
        iqr = (q3 - q1).replace(0, 1e-6)
        df[c] = (df[c] - med) / iqr

    # 2) noise + missingness simulation (copy 40% rows)
    n_aug = int(len(df) * 0.4)
    idx = rng.choice(len(df), size=n_aug, replace=False)
    aug = df.iloc[idx].copy().reset_index(drop=True)

    X = aug[feat_cols].values.astype(float)
    # gaussian noise by feature std
    std = np.std(X, axis=0, keepdims=True)
    noise = rng.normal(0, 0.05, size=X.shape) * np.maximum(std, 1e-6)
    Xn = X + noise
    # random feature dropout 5%
    mask = rng.random(Xn.shape) < 0.05
    Xn[mask] = 0.0
    aug[feat_cols] = Xn

    aug['is_augmented'] = 1
    df['is_augmented'] = 0

    out = pd.concat([df, aug], ignore_index=True)
    out.to_parquet(OUT, index=False)
    print('saved', OUT, 'rows', len(out), 'orig', len(df), 'aug', len(aug))


if __name__ == '__main__':
    main()
