# Baseline Model V3 — AOD Spatial Features

**Date**: 2026-04-29 14:40
**Target**: station mean PM2.5 (40 stations, LOO-CV)
**New features**: per-station means of AOT_inner_mean, AOT_outer_mean, AOT_spatial_std, AOT_local_vs_regional, AOT_grad_mag

## New Feature Correlations with Station Mean PM2.5

| Feature | Pearson | Spearman | New? |
|---------|:---:|:---:|:---:|
| nbr_diurnal_range | +0.7617 | +0.8049 |  |
| mean_PM25_nn_idw | +0.8157 | +0.7713 |  |
| mean_AOT_inner_mean | +0.6391 | +0.7653 | NEW |
| mean_AOT_outer_mean | +0.6595 | +0.7490 | NEW |
| std_PM25_nn_idw | +0.7733 | +0.7392 |  |
| iqr_PM25_nn_idw | +0.7827 | +0.7290 |  |
| ACAG_annual_mean | +0.6789 | +0.6425 |  |
| median_AOT | +0.5444 | +0.6376 |  |
| mean_AOT_spatial_std | +0.4804 | +0.5357 | NEW |
| longitude | -0.4534 | -0.5301 |  |
| mean_AOD_PBLH_ratio | +0.4675 | +0.5301 |  |
| building_area_3km | +0.3592 | +0.4787 |  |
| mean_WS | -0.3575 | -0.4599 |  |
| AOT_p95 | +0.3776 | +0.3904 |  |
| building_count_3km | +0.3095 | +0.3857 |  |
| latitude | +0.4082 | +0.3419 |  |
| mean_AOT_local_vs_regional | -0.0640 | +0.3406 | NEW |
| mean_VC | -0.3790 | -0.3394 |  |
| nbr_seasonal_amp | +0.3582 | +0.3346 |  |
| mean_Humidity | -0.2163 | -0.3273 |  |
| AOT_std | +0.3466 | +0.3147 |  |
| mean_AOT_grad_mag | -0.2414 | -0.3040 | NEW |
| AOT_valid_frac | -0.2829 | -0.2677 |  |
| elevation_m | -0.0912 | +0.2537 |  |
| rain_freq | +0.1467 | +0.2076 |  |
| mean_Pressure | +0.0759 | +0.2031 |  |
| mean_Temp | -0.1427 | -0.0882 |  |
| slope_deg | +0.2799 | -0.0677 |  |
| mean_PBLH | -0.1847 | -0.0170 |  |

## All Results (sorted by R²)

| # | Model | R² | MAE (µg/m³) |
|---|-------|:---:|:---:|
| 1 | Best-8 Ridge | 0.7038 | 5.50 |
| 2 | Best-7 Ridge | 0.6952 | 5.51 |
| 3 | Best-7 ElasticNet | 0.6859 | 5.57 |
| 4 | Best-6 Ridge | 0.6859 | 5.47 |
| 5 | Best-8 ElasticNet | 0.6844 | 5.65 |
| 6 | Best-6 ElasticNet | 0.6841 | 5.54 |
| 7 | Best-5 Ridge | 0.6828 | 5.54 |
| 8 | Best-5 ElasticNet | 0.6812 | 5.58 |
| 9 | Best-5 + mean_AOT_grad_mag | 0.6799 | 5.98 |
| 10 | v2 Best-5 Ridge (baseline) | 0.6580 | 5.99 |
| 11 | Best-5 + mean_AOT_spatial_std | 0.6428 | 6.18 |
| 12 | Best-5 + mean_AOT_inner_mean | 0.6405 | 6.26 |
| 13 | Best-5 RF | 0.6390 | 6.17 |
| 14 | Best-5 + mean_AOT_outer_mean | 0.6372 | 6.19 |
| 15 | Best-6 RF | 0.6302 | 6.31 |
| 16 | Best-5 + mean_AOT_local_vs_regional | 0.6298 | 6.25 |
| 17 | Best-5 + all 5 AOD spatial | 0.6283 | 6.57 |
| 18 | Best-7 RF | 0.6081 | 6.54 |
| 19 | Best-8 RF | 0.5992 | 6.54 |
| 20 | Best-5 + all 5 AOD spatial RF | 0.5603 | 7.08 |
| 21 | v3 All 29 LOSO-safe RF | 0.5363 | 6.82 |
| 22 | v3 All 29 LOSO-safe ElasticNet | 0.4806 | 7.66 |
| 23 | v2 All 24 LOSO-safe Ridge | 0.4220 | 8.29 |
| 24 | v3 All 29 LOSO-safe Ridge | 0.3970 | 8.02 |

## Best Subsets

**Best-5**: mean_PM25_nn_idw, AOT_valid_frac, mean_WS, slope_deg, mean_AOT_grad_mag (new: mean_AOT_grad_mag)
  Ridge R²=0.6828, MAE=5.54

**Best-6**: mean_PM25_nn_idw, AOT_valid_frac, mean_WS, rain_freq, slope_deg, mean_AOT_grad_mag (new: mean_AOT_grad_mag)
  Ridge R²=0.6859, MAE=5.47

**Best-7**: mean_PM25_nn_idw, AOT_p95, AOT_valid_frac, mean_WS, mean_VC, slope_deg, mean_AOT_grad_mag (new: mean_AOT_grad_mag)
  Ridge R²=0.6952, MAE=5.51

**Best-8**: mean_PM25_nn_idw, AOT_p95, AOT_valid_frac, mean_WS, mean_VC, mean_Temp, slope_deg, mean_AOT_grad_mag (new: mean_AOT_grad_mag)
  Ridge R²=0.7038, MAE=5.50

## v2 Best-5 + Individual AOD Features

| Added Feature | R² | MAE | Δ R² vs Best-5 |
|---------------|:---:|:---:|:---:|
| mean_AOT_inner_mean | 0.6405 | 6.26 | -0.0175 |
| mean_AOT_outer_mean | 0.6372 | 6.19 | -0.0207 |
| mean_AOT_spatial_std | 0.6428 | 6.18 | -0.0152 |
| mean_AOT_local_vs_regional | 0.6298 | 6.25 | -0.0282 |
| mean_AOT_grad_mag | 0.6799 | 5.98 | +0.0219 |

## Conclusion

Best model: **Best-8 Ridge** (R²=0.7038, MAE=5.50)

v2 baseline (Best-5 Ridge): R²=0.6580
Improvement: +0.0458 R²
