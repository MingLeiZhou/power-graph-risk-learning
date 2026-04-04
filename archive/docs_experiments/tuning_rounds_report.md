# Tuning Rounds Report (Continue Training)

Date: 2026-04-03

## What was done
Executed `scripts/train_tuning_rounds.py` with 3 rounds:
1. **Round 1 baseline features** (sampled 120k for tuning speed)
2. **Round 2 network-normalized features**
3. **Round 3 LONO (cross-topology)** using best random-split configs

## Best random-split results
### Classification
- Round1 best: RF(max_depth=20, min_samples_leaf=3, n_estimators=250)
  - AUC: 0.99774
  - F1: 0.88889
  - ACC: 0.98446
- Round2 best: RF(max_depth=20, min_samples_leaf=3, n_estimators=250)
  - AUC: **0.99784**
  - F1: **0.90775**
  - ACC: **0.98746**

### Regression
- Round1 best: ExtraTrees(max_depth=20, min_samples_leaf=1, n_estimators=250)
  - RMSE: **0.00965**
  - R2: **0.86818**
- Round2 best: ExtraTrees(max_depth=20, min_samples_leaf=1, n_estimators=250)
  - RMSE: 0.01115
  - R2: 0.82393

Selected for LONO:
- Classifier from Round2 (better classification)
- Regressor from Round2 policy (same round consistency)

## LONO (cross-topology) results
- Mean AUC: **0.65447** (improved vs ~0.50 earlier)
- Mean F1: 0.02219
- Mean ACC: 0.90572 (class imbalance effect)
- Mean RMSE: **0.03852** (improved strongly vs ~0.405 earlier)
- Mean R2: -49.63 (still unstable under severe domain shift)

## Interpretation
- Multi-round tuning + network normalization produced clear gains in cross-topology AUC and RMSE.
- Classification ranking signal improved (AUC up), but decision threshold/F1 remains poor due to heavy imbalance and distribution shift.
- Regression error dropped substantially, but R2 still negative indicates target variance mismatch across held-out networks.

## Saved outputs
- `analysis/training/tuning_round1_classification.csv`
- `analysis/training/tuning_round1_regression.csv`
- `analysis/training/tuning_round2_classification.csv`
- `analysis/training/tuning_round2_regression.csv`
- `analysis/training/tuning_round3_lono_classification.csv`
- `analysis/training/tuning_round3_lono_regression.csv`
- `analysis/training/tuning_summary.json`

## Next tuning ideas
1. Optimize threshold by network or by validation PR-curve (for F1/recall).
2. Add sample weighting/focal-style strategy for rare positive class.
3. Train regressor on `log1p(y_reg)` and back-transform.
4. Introduce domain-adaptive/self-supervised encoder for topology-invariant features.
