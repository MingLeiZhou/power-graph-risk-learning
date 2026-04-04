# F1 Boost LONO Report

## Mean metrics by variant

```
            variant      auc       f1  precision   recall      acc
        rf_balanced 0.631760 0.000000   0.000000 0.000000 0.905744
 rf_sample_weighted 0.651966 0.005080   0.130000 0.002591 0.905762
rf_weighted_netnorm 0.662194 0.004367   0.092593 0.002236 0.905634
```

## By network

```
            variant test_network  threshold      auc       f1  precision   recall      acc  pos_rate_test
        rf_balanced      ieee118        0.5 0.595321 0.000000    0.00000 0.000000 0.949102       0.050898
        rf_balanced       ieee24        0.5 0.616921 0.000000    0.00000 0.000000 0.798419       0.201581
        rf_balanced       ieee39        0.5 0.467855 0.000000    0.00000 0.000000 0.910393       0.089607
        rf_balanced           uk        0.5 0.846943 0.000000    0.00000 0.000000 0.965063       0.034938
 rf_sample_weighted      ieee118        0.5 0.635813 0.000000    0.00000 0.000000 0.949102       0.050898
 rf_sample_weighted       ieee24        0.5 0.652886 0.000000    0.00000 0.000000 0.798419       0.201581
 rf_sample_weighted       ieee39        0.5 0.482058 0.020320    0.52000 0.010363 0.910464       0.089607
 rf_sample_weighted           uk        0.5 0.837105 0.000000    0.00000 0.000000 0.965063       0.034938
rf_weighted_netnorm      ieee118        0.5 0.508925 0.000000    0.00000 0.000000 0.949102       0.050898
rf_weighted_netnorm       ieee24        0.5 0.722466 0.000000    0.00000 0.000000 0.798233       0.201581
rf_weighted_netnorm       ieee39        0.5 0.552746 0.000000    0.00000 0.000000 0.910357       0.089607
rf_weighted_netnorm           uk        0.5 0.864640 0.017467    0.37037 0.008945 0.964844       0.034938
```

Best variant (by F1 then AUC): rf_sample_weighted
