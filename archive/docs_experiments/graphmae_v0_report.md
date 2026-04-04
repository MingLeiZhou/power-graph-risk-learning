# GraphMAE v0 First Run

Date: 2026-04-04

## Setup
- Script: `scripts/run_graphmae_v0.py`
- Runtime: PyTorch + PyG on CPU
- Samples used: 90,000 (subsample for quick first iteration)
- Graph construction: node-per-feature proxy graph

## Pretraining
Masked graph autoencoding loss decreased:
- epoch1: 0.1134
- epoch8: 0.0870

## LONO downstream results (mean)
- Classification AUC: **0.4895**
- Classification F1: **0.0822**
- Regression MAE: **0.0089**
- Regression RMSE: **0.0290**
- Regression R2: **-3.75**

## Interpretation
- This v0 proxy GraphMAE setup did **not** improve classification transfer (AUC below prior baselines).
- Regression absolute error is low, but classification signal is weak.
- Next version should use true topology/edge semantics from PowerGraph or OPF structural graph objects instead of proxy feature-chain graphs.
