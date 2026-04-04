# V2 Real-Topology Sample-level GraphMAE + DANN/CORAL (Joint) Report

Date: 2026-04-04

## Setup
- Dataset: `downstream_v2_informative.parquet`
- Approach: shared encoder with
  - reconstruction objective (GraphMAE-style masking surrogate),
  - classification + regression multitask heads,
  - DANN domain-adversarial loss,
  - CORAL covariance alignment loss.
- Evaluation: strict LONO.

## Mean results
### Classification
- AUC: **0.6870**
- AP: **0.1503**
- F1: **0.1620**
- Precision: **0.1775**
- Recall: **0.2193**

### Regression
- MAE: **0.0136**
- RMSE: **0.0317**
- R2: **-31.70**

## Interpretation
- Compared with prior v2 threshold-calibrated model, this joint adaptation improved precision and maintained strong AUC trajectory, but recall/F1 still require threshold policy tuning.
- Regression remains strong in absolute error.

## Files
- `analysis/training/v2_graphmae_dann_coral_cls.csv`
- `analysis/training/v2_graphmae_dann_coral_reg.csv`
- `analysis/training/v2_graphmae_dann_coral_summary.json`
- `analysis/training/v2_graphmae_dann_coral_report.md`
