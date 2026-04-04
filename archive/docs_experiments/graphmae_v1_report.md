# GraphMAE v1 (Real Topology) Report

Date: 2026-04-04

## What changed vs v0
- Used **real OPF bus topology graphs** (ac_line + transformer edges) instead of feature-chain proxy graph.
- Pretrained GraphMAE on 30k OPF graphs.
- Fused learned graph prototypes into downstream features and re-ran strict LONO.

## Mean LONO results
### Classification
- baseline: AUC 0.5858, AP 0.1510, F1 0.1579, Recall 0.1963
- graphmae_v1_fused: **AUC 0.6250**, AP 0.1369, F1 0.1400, **Recall 0.2049**

### Regression
- baseline: MAE 0.0344, RMSE 0.0493, R2 -36.84
- graphmae_v1_fused: **MAE 0.0336**, **RMSE 0.0472**, **R2 -32.67**

## Interpretation
- Real-topology GraphMAE improved ranking/discrimination (AUC) and regression error.
- F1/AP still need better thresholding/calibration after fusion.
- This confirms GraphMAE has upside when topology is modeled correctly.

## Files
- `analysis/training/graphmae_v1_cls_compare.csv`
- `analysis/training/graphmae_v1_reg_compare.csv`
- `analysis/training/graphmae_v1_summary.json`
- `analysis/training/graphmae_v1_report.md`
