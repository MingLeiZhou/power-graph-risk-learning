#!/usr/bin/env python3
"""Build downstream supervised dataset from PowerGraph MAT files.
Creates sample-level features from Ef/Ef_nc and labels from dns_MW (of_reg).
Outputs CSV/Parquet in data/processed/downstream/
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import h5py

ROOT = Path('data/powergraph/dataset_cascades_extracted/dataset_cascades')
OUT = Path('data/processed/downstream')
OUT.mkdir(parents=True, exist_ok=True)

NETWORKS = None  # None -> auto-discover from ROOT


def deref_dataset(hf, ds, idx):
    """Read one element from object-reference dataset and resolve to array."""
    ref = ds[0, idx]
    arr = hf[ref][()]
    return np.array(arr)


def stat_features(arr, prefix):
    arr = np.array(arr, dtype=float)
    if arr.ndim == 1:
        rows = [arr]
    else:
        rows = [arr[i].ravel() for i in range(arr.shape[0])]
    feats = {}
    for i, r in enumerate(rows[:4]):
        feats[f'{prefix}_r{i}_mean'] = float(np.mean(r))
        feats[f'{prefix}_r{i}_std'] = float(np.std(r))
        feats[f'{prefix}_r{i}_min'] = float(np.min(r))
        feats[f'{prefix}_r{i}_max'] = float(np.max(r))
    feats[f'{prefix}_global_mean'] = float(np.mean(arr))
    feats[f'{prefix}_global_std'] = float(np.std(arr))
    return feats


def build_for_network(net):
    raw = ROOT / net / net / 'raw'
    f_ef = h5py.File(raw / 'Ef.mat', 'r')
    f_efnc = h5py.File(raw / 'Ef_nc.mat', 'r')
    f_of = h5py.File(raw / 'of_reg.mat', 'r')

    ds_ef = f_ef['E_f_post']
    ds_efnc = f_efnc['E_f_kenza']
    y = np.array(f_of['dns_MW'][()]).reshape(-1)

    n = len(y)
    rows = []
    for i in range(n):
        ef = deref_dataset(f_ef, ds_ef, i)
        efnc = deref_dataset(f_efnc, ds_efnc, i)
        row = {'network': net, 'sample_idx': i, 'dns_mw': float(y[i])}
        row.update(stat_features(ef, 'ef'))
        row.update(stat_features(efnc, 'efnc'))
        # downstream labels
        row['y_reg'] = float(y[i])
        row['y_cls'] = int(y[i] > 0.0)
        rows.append(row)
        if (i + 1) % 5000 == 0:
            print(f'[{net}] processed {i+1}/{n}')

    f_ef.close(); f_efnc.close(); f_of.close()
    return pd.DataFrame(rows)


def discover_networks(root: Path) -> list[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build downstream dataset from PowerGraph MAT files")
    parser.add_argument("--networks", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.networks:
        networks = [n.strip() for n in args.networks.split(",") if n.strip()]
    elif NETWORKS:
        networks = NETWORKS
    else:
        networks = discover_networks(ROOT)

    if not networks:
        raise ValueError(f"No networks found under {ROOT}")

    all_df = []
    for net in networks:
        print(f'Building network: {net}')
        df = build_for_network(net)
        print(f'  -> rows: {len(df)} | positive cls: {df.y_cls.mean():.3f}')
        df.to_parquet(OUT / f'{net}_downstream.parquet', index=False)
        all_df.append(df)

    full = pd.concat(all_df, ignore_index=True)
    full.to_parquet(OUT / 'downstream_full.parquet', index=False)
    full.to_csv(OUT / 'downstream_full.csv', index=False)
    print('Saved:', OUT / 'downstream_full.parquet')
    print('Total rows:', len(full))
    print('Overall positive rate:', full.y_cls.mean())


if __name__ == '__main__':
    main()
