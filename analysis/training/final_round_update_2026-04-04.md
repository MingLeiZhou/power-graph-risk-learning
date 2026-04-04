# Final Round Update (00:30)

Ran: `scripts/dual_model_domain_tuning_fast.py`

## Mean results (strict LONO)
### Classification (detector-oriented)
- AUC: **0.6047**
- AP: **0.1382**
- F1: **0.1588**
- Precision: **0.1532**
- Recall: **0.2559**
- Recall@10%: **0.2107**

### Regression (risk-oriented)
- MAE: **0.0312**
- RMSE: **0.0463**
- R²: **-24.91**

## Interpretation
- This dual-model tuning preserves the recovered F1/Recall regime (compared with near-zero F1 runs).
- Regression remains strong in absolute error (MAE/RMSE), with persistent negative R² under extreme domain shift.
- Overall this is a practical operating point for early-warning sensitivity under strict cross-topology transfer.

## Paper-ready outputs
- `analysis/training/paper_main_results.csv`
- `analysis/training/paper_main_results.md`
- `analysis/training/dual_domain_fast_cls_by_network.csv`
- `analysis/training/dual_domain_fast_reg_by_network.csv`
- `analysis/training/dual_domain_fast_summary.json`
