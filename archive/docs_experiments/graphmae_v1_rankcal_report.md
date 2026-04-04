# GraphMAE v1 + Ranking Calibration Report

Date: 2026-04-04

## Run
- Script: `scripts/run_graphmae_v1_rank_calibrated.py`
- Pretrain: real-topology GraphMAE on 20k OPF graphs
- Downstream: strict LONO with weighted RF + source-domain CV threshold selection

## Mean LONO metrics
- AUC: **0.6485**
- AP: **0.1552**
- F1: **0.0345**
- Precision: **0.0214**
- Recall: **0.0889**

## Interpretation
- Ranking quality improved (AUC/AP), but threshold transfer remains brittle and suppresses F1.
- This run confirms GraphMAE helps discrimination, while decision calibration still needs stronger domain adaptation strategy.

## Files
- `analysis/training/graphmae_v1_rankcal_lono.csv`
- `analysis/training/graphmae_v1_rankcal_summary.json`
