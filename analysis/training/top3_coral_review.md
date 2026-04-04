# Top3 (Domain Adaptation / CORAL) Run + Quick Code Review

Date: 2026-04-04

## Run result
Script: `scripts/top3_domain_coral_lono.py`

### Mean metrics
- Classification:
  - AUC: **0.6379**
  - AP: **0.1390**
  - F1: **0.0485**
  - Precision: **0.0733**
  - Recall: **0.2682**
- Regression:
  - MAE: **0.0124**
  - RMSE: **0.0301**
  - R2: **-4.40**

## Interpretation
- CORAL improved ranking/recall and significantly improved regression error.
- But F1 dropped vs prior-threshold detector, so this is not yet best end-to-end detector.

## Quick review: highest-impact improvement directions
1. **Two-stage detector**
   - Stage A: ranking model (maximize AUC/AP/Recall@K)
   - Stage B: calibration model for binary decision threshold transfer (network-conditioned calibration)
   - Expected to keep AUC gains while restoring F1.

2. **Per-target unsupervised thresholding**
   - Replace fixed prior multiplier with target-score-shape method (mixture modeling / elbow / quantile from score entropy).
   - Current threshold rule likely too aggressive after CORAL shift.

3. **True sample-level graph transfer**
   - Current GraphMAE transfer still uses prototype fusion at proxy level.
   - Move to sample-level encoder + downstream head to avoid prototype information collapse.

4. **Metric-aware model selection**
   - Select model checkpoints by AP + Recall@10% composite, not only AUC.
   - Add confidence interval via bootstrap across LONO folds.

## Artifacts
- `analysis/training/top3_coral_cls_by_network.csv`
- `analysis/training/top3_coral_reg_by_network.csv`
- `analysis/training/top3_coral_summary.json`
- `analysis/training/top3_coral_review.md`
