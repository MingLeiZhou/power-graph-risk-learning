#!/usr/bin/env python3
"""Train multiple model families under LONO and compare results."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.mixture import GaussianMixture

try:
    import xgboost as xgb  # type: ignore
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False
    xgb = None  # type: ignore[assignment]

try:
    import torch
    from torch import nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import DataLoader as PyGLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    _PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYG_AVAILABLE = False
    PyGData = None  # type: ignore[assignment]
    PyGLoader = None  # type: ignore[assignment]
    GCNConv = None  # type: ignore[assignment]
    global_mean_pool = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from power_graph_builder import build_graph_from_json

DEFAULT_DATA = Path("data/processed/downstream/downstream_v2_informative.parquet")
DEFAULT_OUT = Path("analysis/training")
DEFAULT_GCN_ROOT = Path("data/opfdata")
DEFAULT_THRESHOLD_GRID = [0.12, 0.16, 0.20, 0.25]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def gmm_threshold(scores: np.ndarray) -> float:
    gm = GaussianMixture(n_components=2, random_state=42)
    s = scores.reshape(-1, 1)
    gm.fit(s)
    m = gm.means_.flatten()
    return float((np.min(m) + np.max(m)) / 2.0)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.sum(y_true) == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def safe_r2(y_true: np.ndarray, preds: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, preds))


def utility_from_metrics(metrics: dict) -> float:
    auc = np.nan_to_num(metrics.get("auc", float("nan")), nan=0.0)
    ap = np.nan_to_num(metrics.get("ap", float("nan")), nan=0.0)
    f1 = np.nan_to_num(metrics.get("f1", float("nan")), nan=0.0)
    precision = np.nan_to_num(metrics.get("precision", float("nan")), nan=0.0)
    return float(f1 * 0.5 + ap * 0.2 + auc * 0.2 + precision * 0.1)


def sweep_threshold(scores: np.ndarray, y_true: np.ndarray, grid: list[float]) -> tuple[float, list[dict]]:
    rows = []
    best_th, best_u = grid[0], -1.0
    for th in grid:
        preds = (scores >= th).astype(int)
        metrics = {
            "auc": safe_auc(y_true, scores),
            "ap": safe_ap(y_true, scores),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
        }
        u = utility_from_metrics(metrics)
        rows.append({"threshold": float(th), **metrics, "utility": u})
        if u > best_u:
            best_th, best_u = float(th), u
    return best_th, rows


def choose_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    method: str,
    grid: list[float],
) -> tuple[float, list[dict]]:
    if method == "gmm":
        return gmm_threshold(scores), []
    return sweep_threshold(scores, y_true, grid)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    feat = [c for c in df.columns if c.startswith("ef_") or c.startswith("efnc_") or c.endswith("_tshift")]
    meta = ["n_samples", "pos_rate", "yreg_mean", "yreg_std"]
    feat.extend([c for c in meta if c in df.columns])
    return feat


def build_model_factories(random_state: int):
    return {
        "rf": {
            "cls": lambda: RandomForestClassifier(
                n_estimators=600,
                max_depth=22,
                random_state=random_state,
                n_jobs=-1,
            ),
            "reg": lambda: RandomForestRegressor(
                n_estimators=600,
                max_depth=22,
                random_state=random_state,
                n_jobs=-1,
            ),
        },
        "xgb": {
            "cls": lambda: xgb.XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                tree_method="hist",
                eval_metric="auc",
                n_jobs=-1,
            ),
            "reg": lambda: xgb.XGBRegressor(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                tree_method="hist",
                objective="reg:squarederror",
                n_jobs=-1,
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LONO training across multiple model families")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", type=str, default="rf,xgb,gcn")
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--threshold-method", type=str, default="sweep", choices=["sweep", "gmm"])
    parser.add_argument("--threshold-grid", type=str, default="0.12,0.16,0.20,0.25")
    parser.add_argument("--gcn-root", type=Path, default=DEFAULT_GCN_ROOT)
    parser.add_argument("--gcn-epochs", type=int, default=100)
    parser.add_argument("--gcn-hidden", type=int, default=64)
    parser.add_argument("--gcn-lr", type=float, default=1e-3)
    parser.add_argument("--gcn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gcn-batch", type=int, default=16)
    parser.add_argument("--gcn-max-graphs", type=int, default=0)
    parser.add_argument("--gcn-max-graphs-per-net", type=int, default=0)
    return parser.parse_args()


def _infer_network_name(path: Path) -> str:
    match = re.search(r"pglib_opf_case(\d+)_ieee", str(path))
    if match:
        return f"case{match.group(1)}"
    return path.parent.name


def _iter_opf_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob("example_*.json"))


def _load_gcn_graphs(
    root: Path,
    max_graphs: int,
    max_graphs_per_net: int,
    seed: int,
) -> list[PyGData]:
    if not (_TORCH_AVAILABLE and _PYG_AVAILABLE):
        raise ImportError("GCN requires torch and torch_geometric")

    rng = np.random.default_rng(seed)
    by_net: dict[str, list[Path]] = {}
    for path in _iter_opf_jsons(root):
        net = _infer_network_name(path)
        by_net.setdefault(net, []).append(path)

    graphs: list[PyGData] = []
    for net, paths in sorted(by_net.items()):
        if max_graphs_per_net > 0 and len(paths) > max_graphs_per_net:
            idx = rng.choice(len(paths), size=max_graphs_per_net, replace=False)
            paths = [paths[i] for i in idx]
        for path in paths:
            graph = build_graph_from_json(
                path,
                normalize_features=True,
                include_solution=False,
                include_links=True,
                normalization_mode="graph",
            )
            if not isinstance(graph, PyGData):
                raise RuntimeError("GCN requires torch_geometric outputs")
            graph.net = net
            graphs.append(graph)

    if max_graphs > 0 and len(graphs) > max_graphs:
        idx = rng.choice(len(graphs), size=max_graphs, replace=False)
        graphs = [graphs[i] for i in idx]

    return graphs


class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return self.head(x).view(-1)


def _prepare_gcn_labels(graphs: Iterable[PyGData], q1: float, q2: float, q3: float) -> None:
    for g in graphs:
        y_reg = float(g.y.item()) if hasattr(g.y, "item") else float(g.y)
        risk = 0
        if y_reg > q1:
            risk = 1
        if y_reg > q2:
            risk = 2
        if y_reg > q3:
            risk = 3
        y_cls = 1 if risk >= 2 else 0
        g.y_reg = torch.tensor([y_reg], dtype=torch.float32)
        g.y_cls = torch.tensor([y_cls], dtype=torch.float32)


def _run_gcn_fold(
    train_graphs: list[PyGData],
    test_graphs: list[PyGData],
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    pos_weight: float,
    device: str,
    log_prefix: str,
):
    model_cls = SimpleGCN(train_graphs[0].num_node_features, hidden_dim).to(device)
    model_reg = SimpleGCN(train_graphs[0].num_node_features, hidden_dim).to(device)

    cls_loader = PyGLoader(train_graphs, batch_size=batch_size, shuffle=True)
    reg_loader = PyGLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = PyGLoader(test_graphs, batch_size=batch_size, shuffle=False)

    cls_opt = torch.optim.Adam(model_cls.parameters(), lr=lr, weight_decay=weight_decay)
    reg_opt = torch.optim.Adam(model_reg.parameters(), lr=lr, weight_decay=weight_decay)
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    reg_loss = nn.MSELoss()

    model_cls.train()
    for epoch in range(epochs):
        losses = []
        for batch in cls_loader:
            batch = batch.to(device)
            cls_opt.zero_grad()
            logits = model_cls(batch.x, batch.edge_index, batch.batch)
            loss = cls_loss(logits, batch.y_cls.view(-1))
            loss.backward()
            cls_opt.step()
            losses.append(loss.item())
        print(f"{log_prefix} cls_epoch={epoch + 1} loss={float(np.mean(losses)):.6f}", flush=True)

    model_reg.train()
    for epoch in range(epochs):
        losses = []
        for batch in reg_loader:
            batch = batch.to(device)
            reg_opt.zero_grad()
            preds = model_reg(batch.x, batch.edge_index, batch.batch)
            target = torch.log1p(torch.clamp(batch.y_reg.view(-1), min=0.0))
            loss = reg_loss(preds, target)
            loss.backward()
            reg_opt.step()
            losses.append(loss.item())
        print(f"{log_prefix} reg_epoch={epoch + 1} loss={float(np.mean(losses)):.6f}", flush=True)

    model_cls.eval()
    model_reg.eval()
    cls_scores, reg_preds, reg_true = [], [], []
    train_scores, train_labels = [], []
    with torch.no_grad():
        for batch in cls_loader:
            batch = batch.to(device)
            logits = model_cls(batch.x, batch.edge_index, batch.batch)
            scores = torch.sigmoid(logits).cpu().numpy()
            train_scores.extend(scores.tolist())
            train_labels.extend(batch.y_cls.view(-1).cpu().numpy().tolist())
        for batch in test_loader:
            batch = batch.to(device)
            logits = model_cls(batch.x, batch.edge_index, batch.batch)
            scores = torch.sigmoid(logits).cpu().numpy()
            cls_scores.extend(scores.tolist())

            z = model_reg(batch.x, batch.edge_index, batch.batch).cpu().numpy()
            pred = np.expm1(z)
            pred = np.clip(pred, 0, None)
            reg_preds.extend(pred.tolist())
            reg_true.extend(batch.y_reg.view(-1).cpu().numpy().tolist())

    return (
        np.array(train_scores),
        np.array(train_labels, dtype=int),
        np.array(cls_scores),
        np.array(reg_preds),
        np.array(reg_true),
    )


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = [args.random_state]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    threshold_grid = [float(v) for v in args.threshold_grid.split(",") if v.strip()]
    if not threshold_grid:
        threshold_grid = DEFAULT_THRESHOLD_GRID

    print(
        "Data note: RF/XGB use downstream_v2_informative; GCN uses OPF JSON graphs. "
        "These are not directly comparable unless samples/labels/networks are aligned.",
        flush=True,
    )
    print(
        "GCN label note: build_graph_from_json sets g.y to OPF objective (metadata.objective), "
        "not dns_MW. Classification labels are derived from objective quantiles.",
        flush=True,
    )

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    cls_rows, reg_rows = [], []
    validation = {
        "tabular": {"networks": [], "counts": {}},
        "gcn": {"networks": [], "counts": {}},
        "label_source": {
            "rf": "downstream_v2_informative: y_cls_v2 / y_reg (dns_mw)",
            "xgb": "downstream_v2_informative: y_cls_v2 / y_reg (dns_mw)",
            "gcn": "OPF JSON graphs: y=objective, y_cls derived from objective quantiles",
        },
    }

    tabular_df = None
    feature_cols = []
    if any(m in {"rf", "xgb"} for m in model_names):
        tabular_df = pd.read_parquet(args.data)
        feature_cols = get_feature_cols(tabular_df)
        validation["tabular"]["networks"] = sorted(tabular_df.network.unique().tolist())
        validation["tabular"]["counts"] = tabular_df.groupby("network").size().to_dict()

    gcn_graphs = None
    if "gcn" in model_names:
        if not (_TORCH_AVAILABLE and _PYG_AVAILABLE):
            raise ImportError("GCN requires torch and torch_geometric")
        gcn_graphs = _load_gcn_graphs(
            args.gcn_root,
            max_graphs=args.gcn_max_graphs,
            max_graphs_per_net=args.gcn_max_graphs_per_net,
            seed=seeds[0],
        )
        gcn_counts = {}
        for g in gcn_graphs:
            gcn_counts[g.net] = gcn_counts.get(g.net, 0) + 1
        validation["gcn"]["networks"] = sorted(gcn_counts)
        validation["gcn"]["counts"] = gcn_counts

    print("Validation summary:", json.dumps(validation, indent=2), flush=True)

    for seed in seeds:
        set_seeds(seed)
        print(f"Seed {seed}: starting", flush=True)

        if tabular_df is not None:
            if "xgb" in model_names and not _XGB_AVAILABLE:
                raise ImportError("xgboost is required for the xgb model")

            factories = build_model_factories(seed)
            for net in sorted(tabular_df.network.unique()):
                print(f"Seed {seed}: tabular LONO network={net}", flush=True)
                tr = tabular_df[tabular_df.network != net]
                te = tabular_df[tabular_df.network == net]
                Xtr = tr[feature_cols].astype(float).values
                Xte = te[feature_cols].astype(float).values

                ytr = tr["y_cls_v2"].astype(int).values
                yte = te["y_cls_v2"].astype(int).values
                w = np.ones(len(ytr))
                w[ytr == 1] = args.pos_weight

                yr_tr = np.clip(tr["y_reg"].astype(float).values, 0, None)
                yr_te = np.clip(te["y_reg"].astype(float).values, 0, None)
                ztr = np.log1p(yr_tr)

                for model_name in [m for m in model_names if m in {"rf", "xgb"}]:
                    clf = factories[model_name]["cls"]()
                    clf.fit(Xtr, ytr, sample_weight=w)
                    s_train = clf.predict_proba(Xtr)[:, 1]
                    s_test = clf.predict_proba(Xte)[:, 1]
                    th, sweep_rows = choose_threshold(s_train, ytr, args.threshold_method, threshold_grid)
                    if sweep_rows:
                        print(f"Seed {seed}: {model_name} threshold_sweep rows={len(sweep_rows)}", flush=True)
                    p = (s_test >= th).astype(int)

                    cls_rows.append(
                        {
                            "seed": seed,
                            "model": model_name,
                            "test_network": net,
                            "threshold": th,
                            "auc": safe_auc(yte, s_test),
                            "ap": safe_ap(yte, s_test),
                            "f1": float(f1_score(yte, p, zero_division=0)),
                            "precision": float(precision_score(yte, p, zero_division=0)),
                            "recall": float(recall_score(yte, p, zero_division=0)),
                        }
                    )

                    reg = factories[model_name]["reg"]()
                    reg.fit(Xtr, ztr)
                    pr = np.expm1(reg.predict(Xte))
                    pr = np.clip(pr, 0, None)

                    reg_rows.append(
                        {
                            "seed": seed,
                            "model": model_name,
                            "test_network": net,
                            "mae": float(mean_absolute_error(yr_te, pr)),
                            "rmse": float(mean_squared_error(yr_te, pr) ** 0.5),
                            "r2": safe_r2(yr_te, pr),
                        }
                    )

        if gcn_graphs is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for net in sorted({g.net for g in gcn_graphs}):
                print(f"Seed {seed}: GCN LONO network={net}", flush=True)
                train_graphs = [g for g in gcn_graphs if g.net != net]
                test_graphs = [g for g in gcn_graphs if g.net == net]

                y_train = np.array([float(g.y.item()) for g in train_graphs])
                q1, q2, q3 = np.quantile(y_train, [0.70, 0.90, 0.98])

                _prepare_gcn_labels(train_graphs, q1, q2, q3)
                _prepare_gcn_labels(test_graphs, q1, q2, q3)

                (
                    train_scores,
                    train_labels,
                    s_test,
                    reg_pred,
                    reg_true,
                ) = _run_gcn_fold(
                    train_graphs,
                    test_graphs,
                    hidden_dim=args.gcn_hidden,
                    epochs=args.gcn_epochs,
                    lr=args.gcn_lr,
                    weight_decay=args.gcn_weight_decay,
                    batch_size=args.gcn_batch,
                    pos_weight=args.pos_weight,
                    device=device,
                    log_prefix=f"Seed {seed} net={net}",
                )

                th, sweep_rows = choose_threshold(train_scores, train_labels, args.threshold_method, threshold_grid)
                if sweep_rows:
                    print(f"Seed {seed}: gcn threshold_sweep rows={len(sweep_rows)}", flush=True)
                p = (s_test >= th).astype(int)
                yte = np.array([int(g.y_cls.item()) for g in test_graphs])

                cls_rows.append(
                    {
                        "seed": seed,
                        "model": "gcn",
                        "test_network": net,
                        "threshold": th,
                        "auc": safe_auc(yte, s_test),
                        "ap": safe_ap(yte, s_test),
                        "f1": float(f1_score(yte, p, zero_division=0)),
                        "precision": float(precision_score(yte, p, zero_division=0)),
                        "recall": float(recall_score(yte, p, zero_division=0)),
                    }
                )

                reg_rows.append(
                    {
                        "seed": seed,
                        "model": "gcn",
                        "test_network": net,
                        "mae": float(mean_absolute_error(reg_true, reg_pred)),
                        "rmse": float(mean_squared_error(reg_true, reg_pred) ** 0.5),
                        "r2": safe_r2(reg_true, reg_pred),
                    }
                )

    cls_df = pd.DataFrame(cls_rows)
    reg_df = pd.DataFrame(reg_rows)

    cls_df.to_csv(args.out / "v2_models_lono_cls.csv", index=False)
    if not reg_df.empty:
        reg_df.to_csv(args.out / "v2_models_lono_reg.csv", index=False)

    cls_mean = cls_df.groupby("model")[["auc", "ap", "f1", "precision", "recall"]].mean()
    cls_std = cls_df.groupby("model")[["auc", "ap", "f1", "precision", "recall"]].std()
    reg_mean = reg_df.groupby("model")[["mae", "rmse", "r2"]].mean() if not reg_df.empty else pd.DataFrame()
    reg_std = reg_df.groupby("model")[["mae", "rmse", "r2"]].std() if not reg_df.empty else pd.DataFrame()

    summary = {
        "classification": {
            model: {
                metric: {"mean": float(cls_mean.loc[model, metric]), "std": float(cls_std.loc[model, metric])}
                for metric in cls_mean.columns
            }
            for model in cls_mean.index
        },
        "regression": {
            model: {
                metric: {"mean": float(reg_mean.loc[model, metric]), "std": float(reg_std.loc[model, metric])}
                for metric in reg_mean.columns
            }
            for model in reg_mean.index
        },
        "data_source": {
            "rf": "downstream_v2_informative",
            "xgb": "downstream_v2_informative",
            "gcn": "opfdata_graphs",
        },
    }
    (args.out / "v2_models_summary.json").write_text(json.dumps(summary, indent=2))

    metadata = {
        "data_source_note": (
            "RF/XGB use downstream_v2_informative; GCN uses OPF JSON graphs. "
            "Not directly comparable unless samples/labels/networks are aligned."
        ),
        "feature_columns": feature_cols,
        "label_columns": {
            "rf": ["y_cls_v2", "y_reg"],
            "xgb": ["y_cls_v2", "y_reg"],
            "gcn": ["objective (g.y)", "derived y_cls (objective quantiles)", "y_reg=objective"],
        },
        "networks": {
            "tabular": validation["tabular"],
            "gcn": validation["gcn"],
        },
        "sample_counts": {
            "tabular": validation["tabular"]["counts"],
            "gcn": validation["gcn"]["counts"],
        },
        "threshold_method": args.threshold_method,
        "threshold_grid": threshold_grid,
        "seeds": seeds,
        "package_availability": {
            "xgboost": _XGB_AVAILABLE,
            "torch": _TORCH_AVAILABLE,
            "torch_geometric": _PYG_AVAILABLE,
        },
        "validation": validation,
    }
    (args.out / "v2_models_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

