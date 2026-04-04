# Staged GraphMAE + DANN/CORAL (v2) Report

Date: 2026-04-04

## Result (strict LONO mean)
- Classification:
  - AUC: 0.6817
  - AP: 0.1340
  - F1: 0.1251
  - Precision: 0.2227
  - Recall: 0.3873
- Regression:
  - MAE: 0.0462
  - RMSE: 0.0656
  - R2: -104.37

## Comparison note
- Relative to previous non-staged DANN/CORAL run, precision improved but overall detector quality (AUC/F1/AP) did not surpass the best v2 threshold-calibrated baseline.
- Regression worsened notably in this staged setting.

## Verdict
Staged schedule alone did not beat current best operational v2 model.
