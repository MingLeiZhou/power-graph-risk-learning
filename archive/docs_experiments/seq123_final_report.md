# Sequence 1-2-3 Execution + Ablation + Final Retrain

## Classification summary by mode

```
              mode      auc       ap       f1  precision   recall
      m1_two_stage 0.605231 0.234158 0.000000   0.000000 0.000000
m2_unsup_threshold 0.605232 0.234158 0.278529   0.208274 0.574985
m3_sample_transfer 0.586553 0.213489 0.000000   0.000000 0.000000
```

## Regression summary by mode

```
              mode      mae     rmse         r2
      m1_two_stage 0.040118 0.058979 -22.981418
m2_unsup_threshold 0.040118 0.058979 -22.981418
m3_sample_transfer 0.035714 0.049026 -38.807879
```

Kept modes: ['m2_unsup_threshold', 'm1_two_stage']\n
Removed modes: ['m3_sample_transfer']\n
Final retrain mode: m2_unsup_threshold\n
## Final mean metrics

```
{
  "classification": {
    "auc": 0.6052311526954564,
    "ap": 0.23416127997763012,
    "f1": 0.2785292693199111,
    "precision": 0.2082744949356221,
    "recall": 0.5749847071256409
  },
  "regression": {
    "mae": 0.040117595372569426,
    "rmse": 0.058979180092459484,
    "r2": -22.981418068615792
  }
}
```
