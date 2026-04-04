#!/usr/bin/env python3
"""GraphMAE v0 (proxy) on downstream feature-graphs + LONO evaluation.

Note: Builds a graph from per-sample feature vector (node-per-feature) for fast iteration.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, r2_score, mean_absolute_error

DATA = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_edge_index(n):
    edges = []
    for i in range(n - 1):
        edges += [(i, i + 1), (i + 1, i)]
    for i in range(n - 2):
        edges += [(i, i + 2), (i + 2, i)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class Encoder(nn.Module):
    def __init__(self, in_dim=1, hid=64, out=32):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, out)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = self.c2(x, edge_index)
        return x


class GraphMAE(nn.Module):
    def __init__(self, in_dim=1, hid=64, z=32):
        super().__init__()
        self.enc = Encoder(in_dim, hid, z)
        self.dec = nn.Sequential(nn.Linear(z, hid), nn.ReLU(), nn.Linear(hid, in_dim))

    def forward(self, data, mask_ratio=0.3):
        x = data.x.clone()
        m = torch.rand(x.size(0), device=x.device) < mask_ratio
        x[m] = 0.0
        z = self.enc(x, data.edge_index)
        xhat = self.dec(z)
        loss = F.mse_loss(xhat[m], data.x[m]) if m.any() else F.mse_loss(xhat, data.x)
        return loss

    @torch.no_grad()
    def embed(self, data):
        z = self.enc(data.x, data.edge_index)
        g = global_mean_pool(z, data.batch)
        return g


def build_graphs(df, feat_cols):
    eidx = make_edge_index(len(feat_cols))
    graphs = []
    X = df[feat_cols].values.astype(np.float32)
    for i in range(len(df)):
        x = torch.from_numpy(X[i]).view(-1, 1)
        d = Data(x=x, edge_index=eidx)
        d.y_cls = int(df.iloc[i]['y_cls'])
        d.y_reg = float(df.iloc[i]['y_reg'])
        d.network = df.iloc[i]['network']
        graphs.append(d)
    return graphs


def train_mae(model, loader, epochs=8):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        for b in loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = model(b)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        print(f'epoch {ep} loss={tot/len(loader):.6f}', flush=True)


@torch.no_grad()
def extract_embeddings(model, graphs, bs=512):
    loader = DataLoader(graphs, batch_size=bs, shuffle=False)
    model.eval()
    embs = []
    for b in loader:
        b = b.to(DEVICE)
        g = model.embed(b).cpu().numpy()
        embs.append(g)
    return np.concatenate(embs, axis=0)


def lono_eval(df_meta, emb):
    nets = sorted(df_meta['network'].unique())
    cls_rows, reg_rows = [], []
    y_cls = df_meta['y_cls'].values
    y_reg = df_meta['y_reg'].values
    net = df_meta['network'].values

    for t in nets:
        tr = net != t
        te = net == t
        Xtr, Xte = emb[tr], emb[te]
        yc_tr, yc_te = y_cls[tr], y_cls[te]
        yr_tr, yr_te = y_reg[tr], y_reg[te]

        clf = LogisticRegression(max_iter=400, class_weight='balanced')
        clf.fit(Xtr, yc_tr)
        s = clf.predict_proba(Xte)[:, 1]
        p = (s >= 0.20).astype(int)
        cls_rows.append({
            'test_network': t,
            'auc': float(roc_auc_score(yc_te, s)),
            'f1': float(f1_score(yc_te, p, zero_division=0))
        })

        reg = Ridge(alpha=1.0)
        reg.fit(Xtr, np.log1p(np.clip(yr_tr, 0, None)))
        pr = np.expm1(reg.predict(Xte))
        pr = np.clip(pr, 0, None)
        reg_rows.append({
            'test_network': t,
            'mae': float(mean_absolute_error(yr_te, pr)),
            'rmse': float(mean_squared_error(yr_te, pr) ** 0.5),
            'r2': float(r2_score(yr_te, pr))
        })

    return pd.DataFrame(cls_rows), pd.DataFrame(reg_rows)


def main():
    torch.manual_seed(42)
    df = pd.read_parquet(DATA)
    # keep runtime manageable for first graphmae round
    if len(df) > 90000:
        df = df.sample(n=90000, random_state=42)

    feat_cols = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_')]
    meta = df[['network', 'y_cls', 'y_reg']].reset_index(drop=True)

    print('Building graphs...', flush=True)
    graphs = build_graphs(df.reset_index(drop=True), feat_cols)
    loader = DataLoader(graphs, batch_size=256, shuffle=True)

    model = GraphMAE(in_dim=1, hid=64, z=32).to(DEVICE)
    print('Training GraphMAE v0 on', DEVICE, flush=True)
    train_mae(model, loader, epochs=8)

    print('Extracting embeddings...', flush=True)
    emb = extract_embeddings(model, graphs)

    print('Running LONO downstream...', flush=True)
    cls_df, reg_df = lono_eval(meta, emb)
    cls_df.to_csv(OUT / 'graphmae_v0_lono_cls.csv', index=False)
    reg_df.to_csv(OUT / 'graphmae_v0_lono_reg.csv', index=False)

    summary = {
        'n_samples': int(len(df)),
        'emb_dim': int(emb.shape[1]),
        'cls_mean': cls_df[['auc', 'f1']].mean().to_dict(),
        'reg_mean': reg_df[['mae', 'rmse', 'r2']].mean().to_dict()
    }
    (OUT / 'graphmae_v0_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
