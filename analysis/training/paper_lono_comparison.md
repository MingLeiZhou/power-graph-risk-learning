# Paper LONO Comparison (Before vs After)

## Mean metrics

```
             setting      auc      f1      acc      mae     rmse         r2
 baseline+domain_cal 0.615227 0.00451 0.905789 0.033898 0.048707 -39.363698
fused_ssl+domain_cal 0.638228 0.00000 0.905735 0.032735 0.046163 -18.119446
```


## By network

```
test_network      auc       f1      acc  threshold      mae     rmse          r2              setting
     ieee118 0.612540 0.000000 0.949102        0.1 0.023115 0.024099 -140.699495  baseline+domain_cal
      ieee24 0.616033 0.000000 0.798419        0.1 0.024260 0.045349   -0.060587  baseline+domain_cal
      ieee39 0.485847 0.018039 0.910571        0.1 0.057792 0.074901  -16.534286  baseline+domain_cal
          uk 0.746487 0.000000 0.965063        0.1 0.030426 0.050478   -0.160422  baseline+domain_cal
     ieee118 0.596253 0.000000 0.949102        0.1 0.013736 0.015537  -57.900451 fused_ssl+domain_cal
      ieee24 0.622458 0.000000 0.798419        0.1 0.026577 0.046315   -0.106279 fused_ssl+domain_cal
      ieee39 0.507961 0.000000 0.910357        0.1 0.054643 0.069708  -14.187363 fused_ssl+domain_cal
          uk 0.826238 0.000000 0.965063        0.1 0.035982 0.053092   -0.283691 fused_ssl+domain_cal
```