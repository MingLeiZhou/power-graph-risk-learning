# 1-2-3 Sequential Execution: Keep/Remove Judgement + Final Retrain

Date: 2026-04-04

## Sequential results (mean over strict LONO folds)

### 1) Two-stage detector (ranking + domain calibration)
- AUC: 0.6052
- AP: 0.2342
- F1: 0.0000
- Precision/Recall: 0.0000 / 0.0000
- Verdict: **partial** (ranking okay, decision failure)

### 2) Unsupervised target thresholding (GMM)
- AUC: 0.6052
- AP: 0.2342
- F1: **0.2785**
- Precision/Recall: **0.2083 / 0.5750**
- Verdict: **strong positive gain** (best detector behavior)

### 3) Sample-level transfer surrogate encoder
- AUC: 0.5866
- AP: 0.2135
- F1: 0.0000
- Regression RMSE improved but R2 deteriorated strongly
- Verdict: **negative for detection** in current implementation

## Keep / Remove
- **Keep:** `m2_unsup_threshold`, `m1_two_stage` (as ranking component)
- **Remove:** `m3_sample_transfer` (current version)

## Final retrain (best retained mode)
- Selected final mode: **m2_unsup_threshold**

Final mean metrics:
- Classification:
  - AUC: 0.6052
  - AP: 0.2342
  - F1: **0.2785**
  - Precision: 0.2083
  - Recall: **0.5750**
- Regression:
  - MAE: 0.0401
  - RMSE: 0.0590
  - R2: -22.98

## Academic significance threshold (practical guidance)
For this strict cross-topology setting, a result is usually meaningful if it achieves roughly:
- **AUC >= 0.70** (or AP clearly above prior baselines by >=20% relative), and
- **F1 >= 0.25** with non-trivial recall (>=0.40), and
- stable performance across all held-out networks (not dominated by one network).

Current status:
- F1/Recall reached practical warning levels (**good**),
- AUC still around 0.61 (**needs improvement for strong academic claim**).

## Deployment practicality
- Current detector can be used as a **high-recall screening model** (human-in-the-loop triage),
- not yet ideal as a standalone autonomous decision model due to moderate discrimination (AUC).

## Files
- `analysis/training/seq123_cls_by_network.csv`
- `analysis/training/seq123_reg_by_network.csv`
- `analysis/training/final_retrain_cls_by_network.csv`
- `analysis/training/final_retrain_reg_by_network.csv`
- `analysis/training/seq123_final_summary.json`
- `analysis/training/seq123_final_report.md`
