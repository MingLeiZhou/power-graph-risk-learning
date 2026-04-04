# F1 Boost Progress (LONO)

Date: 2026-04-03 23:50

## Runs
1. `scripts/f1_boost_lono.py`
2. `scripts/f1_boost_fused_lono.py`

## Results summary
### Variant comparison (non-fused)
- rf_balanced: AUC 0.6318, F1 0.0000
- rf_sample_weighted: **AUC 0.6520, F1 0.0051** (best F1)
- rf_weighted_netnorm: **AUC 0.6622**, F1 0.0044

### Fused latent + weighted
- Mean AUC: 0.6294
- Mean F1: 0.0000

## Interpretation
- Class imbalance remains severe under strict LONO.
- Weighted training improves AUC, but F1 remains close to zero due threshold transfer failure across topologies.
- Fused latent (current proxy-join strategy) did not improve F1; likely needs sample-level alignment rather than case-prototype merge.

## Produced files
- `analysis/training/f1_boost_lono_by_network.csv`
- `analysis/training/f1_boost_lono_summary.csv`
- `analysis/training/f1_boost_lono_summary.json`
- `analysis/training/f1_boost_lono_report.md`
- `analysis/training/f1_boost_fused_lono_by_network.csv`
- `analysis/training/f1_boost_fused_lono_summary.csv`

## Next best move
Implement true sample-level transfer with learned encoder + downstream head (PyTorch), and evaluate PR-AUC / Recall@K alongside F1.
