#!/usr/bin/env python3
"""Self-supervised pretraining (denoising autoencoder) on OPFData.

- Extracts fixed-size aggregate graph features from OPF JSONs
- Trains denoising autoencoder (sklearn MLPRegressor) to reconstruct clean features
- Exports latent representation for transfer tasks
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

ROOT = Path('data/opfdata')
OUT = Path('data/processed/ssl')
OUT.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 60000  # full for now
RANDOM_SEED = 42
MASK_RATIO = 0.2


def safe_stats(mat, max_dim=6):
    """Return mean/std for first max_dim dimensions from list[list[float]]."""
    if not mat:
        return [0.0] * (2 * max_dim)
    arr = np.array(mat, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    d = min(arr.shape[1], max_dim)
    mean = arr[:, :d].mean(axis=0)
    std = arr[:, :d].std(axis=0)
    # pad to max_dim
    out_mean = np.zeros(max_dim)
    out_std = np.zeros(max_dim)
    out_mean[:d] = mean
    out_std[:d] = std
    return np.concatenate([out_mean, out_std]).tolist()


def extract_feature_from_json(path: Path):
    d = json.loads(path.read_text())
    g = d.get('grid', {})
    nodes = g.get('nodes', {})
    edges = g.get('edges', {})

    feat = {}
    # node counts
    for t in ['bus', 'generator', 'load', 'shunt']:
        arr = nodes.get(t, []) if isinstance(nodes, dict) else []
        feat[f'n_{t}'] = len(arr)
        s = safe_stats(arr, max_dim=6)
        for i, v in enumerate(s):
            feat[f'{t}_stat_{i}'] = v

    # edge counts + features (ac_line, transformer have features)
    for t in ['ac_line', 'transformer', 'generator_link', 'load_link', 'shunt_link']:
        obj = edges.get(t, {}) if isinstance(edges, dict) else {}
        senders = obj.get('senders', []) if isinstance(obj, dict) else []
        feat[f'n_{t}'] = len(senders)
        ef = obj.get('features', []) if isinstance(obj, dict) else []
        s = safe_stats(ef, max_dim=6)
        for i, v in enumerate(s):
            feat[f'{t}_stat_{i}'] = v

    objv = d.get('metadata', {}).get('objective', 0.0)
    feat['objective'] = float(objv)

    # case label from path
    case_name = 'unknown'
    for p in path.parts:
        if 'case' in p.lower():
            case_name = p
            break
    feat['case_name'] = case_name
    feat['relpath'] = str(path)
    return feat


def build_feature_table(max_samples=MAX_SAMPLES):
    files = list(ROOT.rglob('*.json'))[:max_samples]
    rows = []
    for i, p in enumerate(files, 1):
        try:
            rows.append(extract_feature_from_json(p))
        except Exception:
            continue
        if i % 5000 == 0:
            print(f'Extracted {i}/{len(files)}')
    df = pd.DataFrame(rows)
    return df


def apply_mask(X, ratio=MASK_RATIO, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    M = rng.random(X.shape) < ratio
    Xm = X.copy()
    Xm[M] = 0.0
    return Xm


def relu(x):
    return np.maximum(0, x)


def get_latent(model: MLPRegressor, X):
    """Extract bottleneck (2nd hidden layer) for architecture [128,32,128]."""
    W1, W2, W3, W4 = model.coefs_
    b1, b2, b3, b4 = model.intercepts_
    h1 = relu(X @ W1 + b1)
    z = relu(h1 @ W2 + b2)  # bottleneck (32)
    return z


def main():
    feat_path = OUT / 'opf_ssl_features.parquet'
    if feat_path.exists():
        print('Loading cached features...')
        df = pd.read_parquet(feat_path)
    else:
        print('Building feature table from OPFData JSON...')
        df = build_feature_table(MAX_SAMPLES)
        df.to_parquet(feat_path, index=False)
        print('Saved feature table:', feat_path)

    # model matrix
    num_cols = [c for c in df.columns if c not in ['case_name', 'relpath']]
    X = df[num_cols].fillna(0.0).to_numpy(dtype=float)

    # scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # masked inputs
    X_masked = apply_mask(Xs, ratio=MASK_RATIO)
    Xtr, Xva, Ytr, Yva = train_test_split(X_masked, Xs, test_size=0.2, random_state=RANDOM_SEED)

    print('Training denoising autoencoder (MLPRegressor)...')
    ae = MLPRegressor(
        hidden_layer_sizes=(128, 32, 128),
        activation='relu',
        solver='adam',
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=RANDOM_SEED,
        verbose=True,
    )
    ae.fit(Xtr, Ytr)

    pred = ae.predict(Xva)
    rmse = mean_squared_error(Yva, pred) ** 0.5
    print(f'Validation reconstruction RMSE: {rmse:.6f}')

    # latent representation (all samples)
    Z = get_latent(ae, Xs)
    zcols = [f'z_{i}' for i in range(Z.shape[1])]
    zdf = pd.DataFrame(Z, columns=zcols)
    zdf['case_name'] = df['case_name'].values
    zdf['objective'] = df['objective'].values
    zdf.to_parquet(OUT / 'opf_ssl_latent.parquet', index=False)

    # artifacts
    joblib.dump(ae, OUT / 'ssl_ae_model.joblib')
    joblib.dump(scaler, OUT / 'ssl_scaler.joblib')

    report = {
        'n_samples': int(len(df)),
        'n_features': int(X.shape[1]),
        'latent_dim': int(Z.shape[1]),
        'mask_ratio': MASK_RATIO,
        'val_recon_rmse': float(rmse),
        'iterations': int(getattr(ae, 'n_iter_', -1)),
    }
    (OUT / 'pretrain_report.json').write_text(json.dumps(report, indent=2))

    print('Saved:')
    print(' -', OUT / 'ssl_ae_model.joblib')
    print(' -', OUT / 'opf_ssl_latent.parquet')
    print(' -', OUT / 'pretrain_report.json')


if __name__ == '__main__':
    main()
