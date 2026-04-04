# LONO Prior-Threshold Optimization Report

Date: 2026-04-04

## Objective
Improve practical detection under severe class imbalance by using:
- stronger positive sample weighting,
- prior-aware thresholding (predictive positive budget),
- ranking diagnostics (AP, Recall@K).

## Mean results
- AUC: **0.6206**
- AP: **0.1405**
- F1: **0.1593**
- Precision: **0.1555**
- Recall: **0.2372**
- Recall@5%: **0.1291**
- Recall@10%: **0.2199**

## Key takeaway
Compared with previous near-zero F1 runs, this setup gives a **major F1/Recall recovery** in strict LONO.
Even if AUC is not maximal, this is better aligned with early-warning use cases where missing positives is costly.

## Output files
- `analysis/training/lono_prior_threshold_by_network.csv`
- `analysis/training/lono_prior_threshold_summary.csv`
- `analysis/training/lono_prior_threshold_summary.json`
- `analysis/training/lono_prior_threshold_report.md`
