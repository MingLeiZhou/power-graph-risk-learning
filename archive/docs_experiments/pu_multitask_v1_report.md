# PU + MultiTask v1 Report

Date: 2026-04-04

## Mean metrics (strict LONO)
- Classification:
  - AUC: 0.6187
  - AP: 0.0953
  - F1: 0.1662
  - Precision: 0.0947
  - Recall: 0.9652
- Regression:
  - MAE: 0.0355
  - RMSE: 0.0508
  - R2: -58.31

## Comparison vs current v2 final (threshold-calibrated)
Current v2 final reference:
- AUC ~0.7444, AP ~0.1405, F1 ~0.1662, Precision ~0.1331, Recall ~0.4534

PU+MultiTask v1:
- Recall is much higher (0.97) but precision/AP collapse.
- AUC/regression are significantly worse than v2 final.

## Verdict
- **Not better as overall primary model**.
- Could be used as an extreme high-recall pre-filter, but high false positives make it expensive.
