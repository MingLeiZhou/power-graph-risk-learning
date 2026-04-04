# Optimization Progress Update

Date: 2026-04-03 23:25

## What was run
1. `scripts/optimize_lono.py` (class imbalance + train-threshold + log1p regression)
2. `scripts/optimize_lono_v2.py` (faster sweep + test-threshold grid analysis + log1p regression)

## Latest metrics
### Optimized V2 (current best diagnostic)
- **LONO mean AUC**: 0.6262
- **LONO mean best-F1-on-grid**: 0.2140
- **LONO mean RMSE**: 0.0469
- **LONO mean MAE**: 0.0317

Per-network classification (AUC / best-F1):
- ieee118: 0.6007 / 0.1124
- ieee24: 0.6203 / 0.4153
- ieee39: 0.5147 / 0.1711
- uk: 0.7690 / 0.1572

Per-network regression (RMSE):
- ieee118: 0.0166
- ieee24: 0.0443
- ieee39: 0.0747
- uk: 0.0519

## Interpretation
- Cross-topology AUC remains in moderate range (~0.63-0.65), better than random but not yet strong.
- F1 is highly sensitive to threshold and class imbalance; fixed 0.5 is too conservative.
- Regression absolute error is low, but R² still unstable across networks due to large target-distribution shift.

## Artifacts
- `analysis/training/optimized_lono_classification.csv`
- `analysis/training/optimized_lono_regression.csv`
- `analysis/training/optimized_lono_summary.json`
- `analysis/training/optimized_lono_v2_classification.csv`
- `analysis/training/optimized_lono_v2_regression.csv`
- `analysis/training/optimized_lono_v2_summary.json`

## Next high-impact steps
1. Use **network-aware calibration** (Platt/Isotonic per source network blend).
2. Train with **domain-adversarial / invariant features** using SSL latent + downstream head.
3. For regression, use **per-network affine correction** after log1p prediction.
4. Evaluate with both strict LONO and mixed-domain validation for model selection.
