# Continue Training Report (LONO Generalization)

Date: 2026-04-03

## What was run
- Script: `scripts/train_continue_generalization.py`
- Protocol: Leave-One-Network-Out (LONO)
- Networks: `ieee118`, `ieee24`, `ieee39`, `uk`
- Models:
  - Classification: Logistic Regression (balanced)
  - Regression: Ridge

## Key results
### Classification (held-out network)
- Mean AUC: **0.5067**
- Mean F1: **0.1419**
- Mean Accuracy: **0.3188**

Per-network AUC:
- ieee118: 0.5056
- ieee24: 0.5000
- ieee39: 0.5267
- uk: 0.4946

### Regression (held-out network)
- Mean RMSE: **0.4053**
- Mean R²: **very negative** (poor transfer)

## Interpretation
- Random split baseline looked very strong, but LONO shows almost-random transfer performance.
- This indicates strong **domain shift across topologies** and likely **network-specific feature distributions**.
- This is expected and supports the need for your planned self-supervised transferable representation learning.

## Errors fixed in this stage
- Installed missing dependencies for training (`scikit-learn`, `pyarrow` in miniforge3 base).
- Re-ran dataset build successfully (`236,000` rows).

## Output files
- `analysis/training/generalization_classification_lono.csv`
- `analysis/training/generalization_regression_lono.csv`
- `analysis/training/continue_training_report.md`

## Next recommended training steps
1. Train topology-aware encoder (self-supervised pretraining on OPFData).
2. Fine-tune on PowerGraph labels.
3. Add domain-invariant normalization / adversarial domain loss.
4. Evaluate again with LONO as the primary metric.
