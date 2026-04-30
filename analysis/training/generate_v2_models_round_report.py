#!/usr/bin/env python3
"""Generate plots and a markdown report for the v2 model comparison."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    _HAVE_SEABORN = True
except ImportError:  # pragma: no cover
    sns = None  # type: ignore[assignment]
    _HAVE_SEABORN = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v2 model comparison report")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/training"))
    parser.add_argument("--fig-dir", type=Path, default=Path("analysis/training/v2_models_round_figs"))
    parser.add_argument("--cls-csv", type=Path, default=Path("analysis/training/v2_models_lono_cls.csv"))
    parser.add_argument("--reg-csv", type=Path, default=Path("analysis/training/v2_models_lono_reg.csv"))
    parser.add_argument("--meta-json", type=Path, default=Path("analysis/training/v2_models_metadata.json"))
    parser.add_argument("--md-out", type=Path, default=Path("analysis/training/v2_models_round.md"))
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def safe_read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_fig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_counts(series: pd.Series, title: str, y_label: str, out_path: Path, dpi: int) -> None:
    plt.figure(figsize=(6, 3))
    if series.empty:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        if _HAVE_SEABORN:
            sns.barplot(x=series.index, y=series.values, color="#4C72B0")
        else:
            plt.bar(series.index, series.values, color="#4C72B0")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(y_label)
    plt.title(title)
    save_fig(out_path, dpi)


def plot_metric_bars(df: pd.DataFrame, metrics: List[str], title: str, out_path: Path, dpi: int) -> None:
    if df.empty:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.title(title)
        save_fig(out_path, dpi)
        return

    if _HAVE_SEABORN:
        melted = df.melt(id_vars=["seed", "model", "test_network"], value_vars=metrics, var_name="metric")
        g = sns.catplot(
            data=melted,
            x="test_network",
            y="value",
            hue="model",
            col="metric",
            kind="bar",
            col_wrap=3,
            height=3.2,
            sharey=False,
        )
        g.set_titles("{col_name}")
        g.set_axis_labels("Network", "Value")
        g.fig.suptitle(title, y=1.03)
        save_fig(out_path, dpi)
        return

    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows), squeeze=False)
    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        pivot = df.pivot_table(index="test_network", columns="model", values=metric, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Network")
        ax.set_ylabel("Value")
        ax.legend(title="Model")
    for idx in range(len(metrics), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    fig.suptitle(title)
    save_fig(out_path, dpi)


def plot_thresholds(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    plt.figure(figsize=(6, 3))
    if "threshold" not in df.columns or df.empty:
        plt.text(0.5, 0.5, "No threshold data", ha="center", va="center")
    else:
        if _HAVE_SEABORN:
            sns.boxplot(data=df, x="model", y="threshold")
        else:
            labels = sorted(df["model"].unique())
            groups = [df[df["model"] == m]["threshold"].values for m in labels]
            try:
                plt.boxplot(groups, tick_labels=labels)
            except TypeError:
                plt.boxplot(groups, labels=labels)
    plt.title("Threshold by model")
    save_fig(out_path, dpi)


def build_summary_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby("model")[metrics].agg(["mean", "std"])


def render_markdown(
    md_path: Path,
    figs: Dict[str, Path],
    metadata: Dict,
    cls_summary: pd.DataFrame,
    reg_summary: pd.DataFrame,
) -> None:
    def md_img(path: Path) -> str:
        return f"![]({path.as_posix()})"

    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "(no table)"
        try:
            return df.to_markdown()
        except ImportError:
            headers = ["model"] + ["_".join(map(str, col)) for col in df.columns]
            rows = []
            for idx, row in df.iterrows():
                rows.append([idx] + [f"{v:.6f}" if isinstance(v, (int, float, np.floating)) else str(v) for v in row])
            lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            for r in rows:
                lines.append("| " + " | ".join(map(str, r)) + " |")
            return "\n".join(lines)

    label_cols = metadata.get("label_columns", {})
    data_note = metadata.get(
        "data_source_note",
        "RF/XGB use downstream_v2_informative; GCN uses OPF JSON graphs. Not directly comparable unless aligned.",
    )

    lines = [
        "# V2 Models Two-Round Report",
        "",
        "## Data scope note",
        data_note,
        "",
        "Label sources:",
        "```json",
        json.dumps(label_cols, indent=2),
        "```",
        "",
        "## Round 1: Quick diagnostic",
        "",
        "### Network coverage (tabular)",
        md_img(figs["tabular_counts"]),
        "",
        "### Network coverage (GCN graphs)",
        md_img(figs["gcn_counts"]),
        "",
        "### Classification metrics by network",
        md_img(figs["cls_metrics"]),
        "",
        "### Threshold distribution",
        md_img(figs["thresholds"]),
        "",
        "## Round 2: Deeper comparison",
        "",
        "### Regression metrics by network",
        md_img(figs["reg_metrics"]),
        "",
        "### Summary statistics (classification)",
        md_table(cls_summary),
        "",
        "### Summary statistics (regression)",
        md_table(reg_summary),
        "",
        "## 这一轮的思路",
        "",
        "- **区分数据来源**：RF/XGBoost 使用 downstream v2 表格数据；GCN 使用 OPF JSON 图数据且标签来自 OPF objective。未对齐样本/标签/网络时，不做严格公平对比。",
        "- **校准一致性**：阈值选择采用与 `final_v2_threshold_sweep_fast.py` 同样的 utility-based sweep，减少阈值差异带来的比较偏差。",
        "- **稳定性与可复现**：多 seed 统计均值/方差，避免单次波动导致的结论偏差。",
        "- **网络差异诊断**：LONO 主要反映跨网络分布漂移，异常网络更适合作为诊断而非优化目标。",
        "- **下一步**：若需要严格横向比较，建议对齐 GCN 的标签为 dns_MW，或从同一 OPF JSON 源重构表格特征。",
    ]

    md_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    if _HAVE_SEABORN:
        sns.set_theme(style="whitegrid")
    else:
        print("seaborn not available; using matplotlib-only plots", flush=True)

    cls_df = pd.read_csv(args.cls_csv)
    reg_df = pd.read_csv(args.reg_csv) if args.reg_csv.exists() else pd.DataFrame()
    metadata = safe_read_json(args.meta_json)

    tabular_counts = pd.Series(metadata.get("sample_counts", {}).get("tabular", {})).sort_index()
    gcn_counts = pd.Series(metadata.get("sample_counts", {}).get("gcn", {})).sort_index()

    figs = {
        "tabular_counts": args.fig_dir / "tabular_counts.png",
        "gcn_counts": args.fig_dir / "gcn_counts.png",
        "cls_metrics": args.fig_dir / "cls_metrics_by_network.png",
        "thresholds": args.fig_dir / "thresholds.png",
        "reg_metrics": args.fig_dir / "reg_metrics_by_network.png",
    }

    plot_counts(tabular_counts, "Tabular samples by network", "Samples", figs["tabular_counts"], args.dpi)
    plot_counts(gcn_counts, "GCN graphs by network", "Graphs", figs["gcn_counts"], args.dpi)

    plot_metric_bars(
        cls_df,
        ["auc", "ap", "f1", "precision", "recall"],
        "Classification metrics by network",
        figs["cls_metrics"],
        args.dpi,
    )

    plot_thresholds(cls_df, figs["thresholds"], args.dpi)

    plot_metric_bars(
        reg_df,
        ["mae", "rmse", "r2"],
        "Regression metrics by network",
        figs["reg_metrics"],
        args.dpi,
    )

    cls_summary = build_summary_table(cls_df, ["auc", "ap", "f1", "precision", "recall"])
    reg_summary = build_summary_table(reg_df, ["mae", "rmse", "r2"])

    render_markdown(args.md_out, figs, metadata, cls_summary, reg_summary)
    print(f"Saved report: {args.md_out}")


if __name__ == "__main__":
    main()

