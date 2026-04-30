# V2 Models Two-Round Report

## Data scope note
RF/XGB use downstream_v2_informative; GCN uses OPF JSON graphs. Not directly comparable unless samples/labels/networks are aligned.

Label sources:
```json
{
  "rf": [
    "y_cls_v2",
    "y_reg"
  ],
  "xgb": [
    "y_cls_v2",
    "y_reg"
  ],
  "gcn": [
    "objective (g.y)",
    "derived y_cls (objective quantiles)",
    "y_reg=objective"
  ]
}
```

## Round 1: Quick diagnostic

### Network coverage (tabular)
![](analysis/training/v2_models_round_figs/tabular_counts.png)

### Network coverage (GCN graphs)
![](analysis/training/v2_models_round_figs/gcn_counts.png)

### Classification metrics by network
![](analysis/training/v2_models_round_figs/cls_metrics_by_network.png)

### Threshold distribution
![](analysis/training/v2_models_round_figs/thresholds.png)

## Round 2: Deeper comparison

### Regression metrics by network
![](analysis/training/v2_models_round_figs/reg_metrics_by_network.png)

### Summary statistics (classification)
| model | auc_mean | auc_std | ap_mean | ap_std | f1_mean | f1_std | precision_mean | precision_std | recall_mean | recall_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rf | 0.747435 | 0.107625 | 0.149102 | 0.108231 | 0.105209 | 0.120923 | 0.121232 | 0.157089 | 0.163893 | 0.235869 |
| xgb | 0.678097 | 0.133930 | 0.126072 | 0.102582 | 0.115932 | 0.120637 | 0.108352 | 0.119250 | 0.196202 | 0.198627 |

### Summary statistics (regression)
| model | mae_mean | mae_std | rmse_mean | rmse_std | r2_mean | r2_std |
| --- | --- | --- | --- | --- | --- | --- |
| rf | 0.015431 | 0.002107 | 0.039557 | 0.007445 | -76.367563 | 148.732000 |
| xgb | 0.014171 | 0.001665 | 0.035727 | 0.007572 | -59.621488 | 116.845788 |

## 这一轮的思路

- **区分数据来源**：RF/XGBoost 使用 downstream v2 表格数据；GCN 使用 OPF JSON 图数据且标签来自 OPF objective。未对齐样本/标签/网络时，不做严格公平对比。
- **校准一致性**：阈值选择采用与 `final_v2_threshold_sweep_fast.py` 同样的 utility-based sweep，减少阈值差异带来的比较偏差。
- **稳定性与可复现**：多 seed 统计均值/方差，避免单次波动导致的结论偏差。
- **网络差异诊断**：LONO 主要反映跨网络分布漂移，异常网络更适合作为诊断而非优化目标。
- **下一步**：若需要严格横向比较，建议对齐 GCN 的标签为 dns_MW，或从同一 OPF JSON 源重构表格特征。