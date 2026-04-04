#!/usr/bin/env python3
"""Build v2 informative dataset:
- multi-level risk labels
- hard-example oversampling near decision boundary
- temporal-like perturbation features + noise/missing simulation
- network meta features
"""
from pathlib import Path
import numpy as np
import pandas as pd

INP = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('data/processed/downstream/downstream_v2_informative.parquet')


def main():
    rng = np.random.default_rng(42)
    df = pd.read_parquet(INP).copy()
    feat = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]

    # --- multi-level risk from y_reg ---
    y = df['y_reg'].values
    q1, q2, q3 = np.quantile(y, [0.70, 0.90, 0.98])
    risk = np.zeros(len(df), dtype=int)
    risk[y > q1] = 1
    risk[y > q2] = 2
    risk[y > q3] = 3
    df['risk_level'] = risk
    # binary target: moderate/severe as positive
    df['y_cls_v2'] = (df['risk_level'] >= 2).astype(int)

    # --- network meta features ---
    net_stats = df.groupby('network').agg(
        n_samples=('network', 'size'),
        pos_rate=('y_cls_v2', 'mean'),
        yreg_mean=('y_reg', 'mean'),
        yreg_std=('y_reg', 'std')
    ).reset_index()
    df = df.merge(net_stats, on='network', how='left')

    # --- temporal-like perturbation features (lightweight surrogate) ---
    # simulate short-term drift by adding smoothed jitter on base means
    for c in [x for x in feat if x.endswith('_mean')]:
        jitter = rng.normal(0, 0.02, size=len(df))
        df[c + '_tshift'] = df[c] * (1.0 + jitter)

    # --- hard-example oversampling near boundary around q2 ---
    # hard = near moderate-risk threshold ± band
    band = max(1e-6, (q3 - q1) * 0.08)
    hard_mask = (df['y_reg'] >= (q2 - band)) & (df['y_reg'] <= (q2 + band))
    hard = df[hard_mask].copy()
    n_hard = min(len(hard), int(len(df) * 0.35))
    if n_hard > 0:
        hard = hard.sample(n=n_hard, random_state=42, replace=(len(hard) < n_hard))

        Xh = hard[feat].values.astype(float)
        std = np.std(Xh, axis=0, keepdims=True)
        noise = rng.normal(0, 0.04, size=Xh.shape) * np.maximum(std, 1e-6)
        Xh = Xh + noise
        # missingness simulation
        m = rng.random(Xh.shape) < 0.04
        Xh[m] = 0.0
        hard[feat] = Xh
        hard['is_hard_aug'] = 1
    else:
        hard = df.head(0).copy()
        hard['is_hard_aug'] = []

    df['is_hard_aug'] = 0
    out = pd.concat([df, hard], ignore_index=True)

    # robust normalize per network for all numeric model features
    model_cols = [c for c in out.columns if c.startswith('ef_') or c.startswith('efnc_') or c.endswith('_tshift')]
    for c in model_cols:
        g = out.groupby('network')[c]
        med = g.transform('median')
        ql = g.transform(lambda s: s.quantile(0.25))
        qh = g.transform(lambda s: s.quantile(0.75))
        iqr = (qh - ql).replace(0, 1e-6)
        out[c] = (out[c] - med) / iqr

    out.to_parquet(OUT, index=False)
    print('saved', OUT)
    print('rows', len(out), 'orig', len(df), 'hard_aug', len(hard))
    print('risk thresholds', {'q1': float(q1), 'q2': float(q2), 'q3': float(q3)})
    print('v2 positive rate', float(out['y_cls_v2'].mean()))


if __name__ == '__main__':
    main()
