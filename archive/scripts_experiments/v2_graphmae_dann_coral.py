#!/usr/bin/env python3
"""Rollback to v2 features and run sample-level GraphMAE + DANN/CORAL joint adaptation.
Practical surrogate implementation on tabular sample-level features.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path('data/processed/downstream/downstream_v2_informative.parquet')
OUT = Path('analysis/training')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def coral_loss(xs, xt):
    # xs [n,d], xt [m,d]
    xs = xs - xs.mean(0, keepdim=True)
    xt = xt - xt.mean(0, keepdim=True)
    cs = (xs.t() @ xs) / max(xs.size(0) - 1, 1)
    ct = (xt.t() @ xt) / max(xt.size(0) - 1, 1)
    return ((cs - ct) ** 2).mean()


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class Model(nn.Module):
    def __init__(self, in_dim, z=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, z)
        )
        self.dec = nn.Sequential(nn.Linear(z, 128), nn.ReLU(), nn.Linear(128, in_dim))
        self.cls = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, 1))
        self.reg = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, 1))
        self.dom = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, grl=0.0):
        z = self.enc(x)
        xh = self.dec(z)
        yc = self.cls(z).squeeze(-1)
        yr = self.reg(z).squeeze(-1)
        zd = grad_reverse(z, grl)
        yd = self.dom(zd).squeeze(-1)
        return z, xh, yc, yr, yd


def standardize_by_source(Xs, Xt):
    mu = Xs.mean(0, keepdims=True)
    sd = Xs.std(0, keepdims=True) + 1e-6
    return (Xs - mu) / sd, (Xt - mu) / sd


def train_one_fold(df, feat_cols, target_net, epochs=18, batch_size=1024):
    src = df[df.network != target_net].copy()
    tgt = df[df.network == target_net].copy()

    Xs = src[feat_cols].astype(float).values
    Xt = tgt[feat_cols].astype(float).values
    Xs, Xt = standardize_by_source(Xs, Xt)

    ys_cls = src['y_cls_v2'].astype(float).values
    yt_cls = tgt['y_cls_v2'].astype(float).values
    ys_reg = src['y_reg'].astype(float).values
    yt_reg = tgt['y_reg'].astype(float).values

    xsrc = torch.tensor(Xs, dtype=torch.float32, device=DEVICE)
    xtgt = torch.tensor(Xt, dtype=torch.float32, device=DEVICE)
    ysc = torch.tensor(ys_cls, dtype=torch.float32, device=DEVICE)
    ysr = torch.tensor(np.log1p(np.clip(ys_reg, 0, None)), dtype=torch.float32, device=DEVICE)

    model = Model(in_dim=xsrc.size(1), z=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    pos_weight = torch.tensor([(len(ys_cls) - ys_cls.sum()) / max(ys_cls.sum(), 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ns, nt = xsrc.size(0), xtgt.size(0)
    for ep in range(1, epochs + 1):
        model.train()
        idxs = np.random.permutation(ns)
        idxt = np.random.permutation(nt)
        nb = max(ns // batch_size, 1)
        grl = min(1.0, ep / epochs)

        losses = []
        for i in range(nb):
            bs = idxs[i * batch_size:(i + 1) * batch_size]
            bt = idxt[i * batch_size:(i + 1) * batch_size]
            if len(bs) == 0 or len(bt) == 0:
                continue

            xs_b, ys_c_b, ys_r_b = xsrc[bs], ysc[bs], ysr[bs]
            xt_b = xtgt[bt]

            # mask for MAE-style reconstruction
            m = (torch.rand_like(xs_b) < 0.3)
            xs_m = xs_b.clone(); xs_m[m] = 0.0

            z_s, xh_s, yc_s, yr_s, yd_s = model(xs_m, grl=grl)
            z_t, xh_t, yc_t, yr_t, yd_t = model(xt_b, grl=grl)

            loss_rec = F.mse_loss(xh_s[m], xs_b[m]) if m.any() else F.mse_loss(xh_s, xs_b)
            loss_cls = bce(yc_s, ys_c_b)
            loss_reg = F.mse_loss(yr_s, ys_r_b)

            # domain labels: source=0, target=1
            yds = torch.zeros_like(yd_s)
            ydt = torch.ones_like(yd_t)
            loss_dom = F.binary_cross_entropy_with_logits(yd_s, yds) + F.binary_cross_entropy_with_logits(yd_t, ydt)
            loss_coral = coral_loss(z_s, z_t)

            loss = loss_rec + 1.0 * loss_cls + 0.5 * loss_reg + 0.2 * loss_dom + 0.2 * loss_coral
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))

        if ep % 6 == 0:
            print(f'[{target_net}] epoch {ep} loss={np.mean(losses):.4f}', flush=True)

    # inference on target
    model.eval()
    with torch.no_grad():
        zt, _, yc_t, yr_t, _ = model(xtgt, grl=0.0)
        s = torch.sigmoid(yc_t).cpu().numpy()
        # prior-aware threshold
        q = min(0.25, max(0.03, float(ys_cls.mean()) * 1.4))
        th = float(np.quantile(s, 1 - q))
        p = (s >= th).astype(int)

        pr = np.expm1(yr_t.cpu().numpy())
        pr = np.clip(pr, 0, None)

    cls = {
        'test_network': target_net,
        'auc': float(roc_auc_score(yt_cls, s)),
        'ap': float(average_precision_score(yt_cls, s)),
        'f1': float(f1_score(yt_cls, p, zero_division=0)),
        'precision': float(precision_score(yt_cls, p, zero_division=0)),
        'recall': float(recall_score(yt_cls, p, zero_division=0)),
        'threshold': th,
    }
    reg = {
        'test_network': target_net,
        'mae': float(mean_absolute_error(yt_reg, pr)),
        'rmse': float(mean_squared_error(yt_reg, pr) ** 0.5),
        'r2': float(r2_score(yt_reg, pr)),
    }
    return cls, reg


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_parquet(DATA)
    feat_cols = [c for c in df.columns if c.startswith('ef_') or c.startswith('efnc_') or c.endswith('_tshift')]
    feat_cols += [c for c in ['n_samples','pos_rate','yreg_mean','yreg_std'] if c in df.columns]

    cls_rows, reg_rows = [], []
    for net in sorted(df.network.unique()):
        print('running fold', net, flush=True)
        c, r = train_one_fold(df, feat_cols, net)
        cls_rows.append(c); reg_rows.append(r)

    cls_df = pd.DataFrame(cls_rows)
    reg_df = pd.DataFrame(reg_rows)
    cls_df.to_csv(OUT / 'v2_graphmae_dann_coral_cls.csv', index=False)
    reg_df.to_csv(OUT / 'v2_graphmae_dann_coral_reg.csv', index=False)

    summary = {
        'classification_mean': cls_df[['auc','ap','f1','precision','recall']].mean().to_dict(),
        'regression_mean': reg_df[['mae','rmse','r2']].mean().to_dict(),
    }
    (OUT / 'v2_graphmae_dann_coral_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
