#!/usr/bin/env python3
"""GraphMAE v1 on real OPF topology graphs (bus graph), then transfer to downstream LONO.
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OPF_ROOT = Path('data/opfdata')
DOWN_PATH = Path('data/processed/downstream/downstream_full.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_opf_graph(path: Path):
    d = json.loads(path.read_text())
    nodes = d['grid']['nodes']['bus']  # [n_bus, feat]
    x = torch.tensor(nodes, dtype=torch.float)
    n_bus = x.size(0)

    edges = []
    ge = d['grid']['edges']
    for et in ['ac_line', 'transformer']:
        if et in ge and isinstance(ge[et], dict):
            s = ge[et].get('senders', [])
            r = ge[et].get('receivers', [])
            for a, b in zip(s, r):
                a, b = int(a), int(b)
                if a < n_bus and b < n_bus:
                    edges.append((a, b))
                    edges.append((b, a))
    if not edges:
        # fallback simple chain
        for i in range(n_bus - 1):
            edges.append((i, i + 1)); edges.append((i + 1, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    case_name = 'unknown'
    for p in path.parts:
        if 'case' in p.lower():
            case_name = p
            break
    return Data(x=x, edge_index=edge_index), case_name


class GMAE(nn.Module):
    def __init__(self, in_dim, hid=64, z=32):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, z)
        self.dec = nn.Sequential(nn.Linear(z, hid), nn.ReLU(), nn.Linear(hid, in_dim))

    def encode(self, x, edge_index):
        h = F.relu(self.c1(x, edge_index))
        z = self.c2(h, edge_index)
        return z

    def forward(self, data, mask_ratio=0.3):
        x = data.x.clone()
        m = torch.rand(x.size(0), device=x.device) < mask_ratio
        x[m] = 0.0
        z = self.encode(x, data.edge_index)
        xh = self.dec(z)
        if m.any():
            return F.mse_loss(xh[m], data.x[m])
        return F.mse_loss(xh, data.x)

    @torch.no_grad()
    def graph_embed(self, data):
        z = self.encode(data.x, data.edge_index)
        g = global_mean_pool(z, data.batch)
        return g


def train_mae(graphs, epochs=10):
    in_dim = graphs[0].x.size(1)
    model = GMAE(in_dim=in_dim, hid=64, z=32).to(DEVICE)
    loader = DataLoader(graphs, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(1, epochs + 1):
        loss_sum = 0.0
        for b in loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = model(b)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
        print(f'epoch {ep} loss={loss_sum/len(loader):.6f}', flush=True)
    return model


@torch.no_grad()
def extract_case_proto(model, graphs, case_names):
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    model.eval()
    embs = []
    for b in loader:
        b = b.to(DEVICE)
        embs.append(model.graph_embed(b).cpu().numpy())
    E = np.concatenate(embs, axis=0)
    df = pd.DataFrame(E, columns=[f'gmae_z{i}' for i in range(E.shape[1])])
    df['case_name'] = case_names
    return df.groupby('case_name').mean().reset_index()


def case_to_proxy(case_name: str):
    c = case_name.lower()
    if 'case14' in c: return 'case14'
    if 'case30' in c: return 'case30'
    if 'case57' in c: return 'case57'
    if 'case118' in c: return 'case118'
    return 'other'


def net_to_proxy(n):
    return {'ieee24':'case30','ieee39':'case57','ieee118':'case118','uk':'case118'}.get(n, 'case57')


def lono_eval(df, feats):
    nets = sorted(df.network.unique())
    cls_rows, reg_rows = [], []
    for net in nets:
        tr = df[df.network != net]
        te = df[df.network == net]
        Xtr, Xte = tr[feats].astype(float), te[feats].astype(float)

        ytr, yte = tr.y_cls.astype(int).values, te.y_cls.astype(int).values
        w = np.ones(len(ytr)); w[ytr==1] = 10.0
        clf = RandomForestClassifier(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr, sample_weight=w)
        s = clf.predict_proba(Xte)[:,1]
        q = min(0.30, max(0.02, float((tr.y_cls==1).mean())*1.6))
        th = float(np.quantile(s, 1-q))
        p = (s >= th).astype(int)
        cls_rows.append({
            'test_network': net,
            'auc': float(roc_auc_score(yte, s)),
            'ap': float(average_precision_score(yte, s)),
            'f1': float(f1_score(yte, p, zero_division=0)),
            'precision': float(precision_score(yte, p, zero_division=0)),
            'recall': float(recall_score(yte, p, zero_division=0))
        })

        yr_tr, yr_te = tr.y_reg.astype(float).values, te.y_reg.astype(float).values
        reg = ExtraTreesRegressor(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1)
        reg.fit(Xtr, np.log1p(np.clip(yr_tr,0,None)))
        pr = np.expm1(reg.predict(Xte)); pr = np.clip(pr, 0, None)
        reg_rows.append({
            'test_network': net,
            'mae': float(mean_absolute_error(yr_te, pr)),
            'rmse': float(mean_squared_error(yr_te, pr)**0.5),
            'r2': float(r2_score(yr_te, pr))
        })

    return pd.DataFrame(cls_rows), pd.DataFrame(reg_rows)


def main():
    torch.manual_seed(42)
    files = list(OPF_ROOT.rglob('*.json'))
    if len(files) > 30000:
        rng = np.random.default_rng(42)
        files = list(rng.choice(files, size=30000, replace=False))

    graphs, case_names = [], []
    for i, p in enumerate(files, 1):
        try:
            g, c = load_opf_graph(p)
            graphs.append(g); case_names.append(c)
        except Exception:
            continue
        if i % 5000 == 0:
            print(f'loaded {i}/{len(files)}', flush=True)

    print('train graphmae v1 on', len(graphs), 'graphs', DEVICE, flush=True)
    model = train_mae(graphs, epochs=10)

    print('extract case prototypes', flush=True)
    proto = extract_case_proto(model, graphs, case_names)
    proto['proxy'] = proto['case_name'].map(case_to_proxy)
    pcols = [c for c in proto.columns if c.startswith('gmae_z')]
    proto2 = proto.groupby('proxy')[pcols].mean().reset_index()

    down = pd.read_parquet(DOWN_PATH)
    down['proxy'] = down['network'].map(net_to_proxy)
    fused = down.merge(proto2, on='proxy', how='left').fillna(0.0)

    base_feats = [c for c in down.columns if c.startswith('ef_') or c.startswith('efnc_')]
    fused_feats = base_feats + [c for c in fused.columns if c.startswith('gmae_z')]

    print('lono evaluate baseline', flush=True)
    cls_b, reg_b = lono_eval(down, base_feats)
    print('lono evaluate graphmae-v1 fused', flush=True)
    cls_f, reg_f = lono_eval(fused, fused_feats)

    cls_b['setting'] = 'baseline'
    cls_f['setting'] = 'graphmae_v1_fused'
    reg_b['setting'] = 'baseline'
    reg_f['setting'] = 'graphmae_v1_fused'

    all_cls = pd.concat([cls_b, cls_f], ignore_index=True)
    all_reg = pd.concat([reg_b, reg_f], ignore_index=True)
    all_cls.to_csv(OUT/'graphmae_v1_cls_compare.csv', index=False)
    all_reg.to_csv(OUT/'graphmae_v1_reg_compare.csv', index=False)

    summary = {
        'cls_mean': all_cls.groupby('setting')[['auc','ap','f1','precision','recall']].mean().reset_index().to_dict(orient='records'),
        'reg_mean': all_reg.groupby('setting')[['mae','rmse','r2']].mean().reset_index().to_dict(orient='records')
    }
    (OUT/'graphmae_v1_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
