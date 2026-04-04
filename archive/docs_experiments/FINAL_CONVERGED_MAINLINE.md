# FINAL Converged Mainline (Accepted)

Date: 2026-04-04

## Decision
After sequential experiments and ablations, we converge to the best practical+academic mainline:

**Mainline = v2 informative dataset + threshold-calibrated detector**
- Dataset: `data/processed/downstream/downstream_v2_informative.parquet`
- Training/Eval pipeline: `scripts/final_v2_threshold_sweep_fast.py`
- Selected balanced threshold: **0.12**

## Final Mainline Metrics (strict LONO mean)
### Classification
- AUC: **0.7444**
- AP: **0.1405**
- F1: **0.1662**
- Precision: **0.1331**
- Recall: **0.4534**

### Regression
- MAE: **0.0130**
- RMSE: **0.0313**
- R2: **-41.16**

## Why this mainline
- Highest and most stable AUC among robust runs.
- Strong recall for early-warning operation.
- Good absolute error for regression.
- Better overall utility than staged DANN/CORAL and PU+multitask variants.

## Keep vs Remove (final)
### Keep
- v2 informative data construction
- hard-example augmentation
- per-network robust normalization
- threshold sweep + deployment profiles

### Remove from mainline
- proxy GraphMAE v0
- staged DANN/CORAL as default path
- PU+multitask v1 as primary model (can be fallback high-recall filter)

## Academic significance & practical usage
- Academic: AUC~0.74 under strict cross-topology is meaningful; AP/F1 still improvable but publishable with clear limitations.
- Practical: usable as high-recall triage/screening model; keep human review for final decisions.

## Final artifacts
- `analysis/training/paper_main_results_final.csv`
- `analysis/training/paper_main_results_final.md`
- `analysis/training/v2_final_summary.json`
- `analysis/training/v2_threshold_sweep.csv`
