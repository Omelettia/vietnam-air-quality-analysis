# Station Baseline Features — Model Analysis

**Date**: 2026-04-29 12:35
**Stations**: 40
**All predictor features**: 31
**LOSO-safe features**: 24

## Feature Correlations

| Feature | Leaky? | Pearson→PM2.5 | Spearman→PM2.5 | Pearson→R² | Spearman→R² |
|---------|:---:|:---:|:---:|:---:|:---:|
| seasonal_pm25_amp | LEAK | +0.9088 | +0.8650 | +0.5848 | +0.6073 |
| nbr_diurnal_range |  | +0.7617 | +0.8049 | +0.3610 | +0.3443 |
| mean_PM25_nn_idw |  | +0.8157 | +0.7713 | +0.3974 | +0.3615 |
| diurnal_pm25_range | LEAK | +0.7385 | +0.7537 | +0.3492 | +0.3989 |
| std_PM25_nn_idw |  | +0.7733 | +0.7392 | +0.3726 | +0.3039 |
| iqr_PM25_nn_idw |  | +0.7827 | +0.7290 | +0.4062 | +0.2947 |
| ACAG_annual_mean |  | +0.6789 | +0.6425 | +0.4555 | +0.4222 |
| median_AOT |  | +0.5444 | +0.6376 | +0.2661 | +0.3994 |
| mean_AOT |  | +0.5055 | +0.6292 | +0.1933 | +0.3474 |
| longitude |  | -0.4534 | -0.5301 | -0.2126 | -0.2594 |
| mean_AOD_PBLH_ratio |  | +0.4675 | +0.5301 | +0.1419 | +0.2105 |
| mean_AOD_physics |  | +0.5003 | +0.5167 | +0.2963 | +0.3528 |
| AOT_iqr |  | +0.4443 | +0.4899 | +0.1324 | +0.2400 |
| nn1_corr | LEAK | +0.4545 | +0.4864 | +0.4282 | +0.3806 |
| building_area_3km |  | +0.3592 | +0.4787 | +0.4312 | +0.4400 |
| mean_WS |  | -0.3575 | -0.4599 | -0.3121 | -0.2857 |
| AOT_p95 |  | +0.3776 | +0.3904 | +0.0540 | +0.0823 |
| building_count_3km |  | +0.3095 | +0.3857 | +0.3951 | +0.3675 |
| latitude |  | +0.4082 | +0.3419 | +0.1954 | -0.0034 |
| mean_VC |  | -0.3790 | -0.3394 | -0.3011 | -0.1848 |
| nbr_seasonal_amp |  | +0.3582 | +0.3346 | +0.3945 | +0.3499 |
| mean_Humidity |  | -0.2163 | -0.3273 | -0.4299 | -0.5139 |
| AOT_std |  | +0.3466 | +0.3147 | +0.0432 | +0.0718 |
| AOT_valid_frac |  | -0.2829 | -0.2677 | -0.0112 | -0.0266 |
| elevation_m |  | -0.0912 | +0.2537 | -0.0547 | +0.2911 |
| rain_freq |  | +0.1467 | +0.2076 | +0.2032 | +0.1313 |
| mean_Pressure |  | +0.0759 | +0.2031 | +0.0445 | +0.0598 |
| mean_Temp |  | -0.1427 | -0.0882 | -0.1029 | -0.0735 |
| slope_deg |  | +0.2799 | -0.0677 | +0.2076 | +0.1011 |
| mean_PBLH |  | -0.1847 | -0.0170 | -0.0318 | +0.1157 |
| nn1_km |  | -0.0982 | +0.0041 | -0.0230 | -0.0115 |

## LOO-CV Results — All Features (target = station mean PM2.5)

| Model | R² | MAE (µg/m³) |
|-------|:---:|:---:|
| Ridge (all 31 features) | 0.8291 | 4.63 |
| ElasticNet (all 31 features) | 0.8588 | 4.17 |
| Random Forest (all 31 features) | 0.8059 | 4.90 |
| Ridge (best 5) | 0.9059 | 3.20 |

**Best 5-feature subset**: ACAG_annual_mean, mean_Temp, diurnal_pm25_range, seasonal_pm25_amp, nbr_diurnal_range

## LOO-CV Results — LOSO-Safe Features Only

These features do NOT use the station's own PM2.5, so they are valid as a Stage 1 spatial baseline in LOSO evaluation.

| Model | R² | MAE (µg/m³) |
|-------|:---:|:---:|
| Ridge (24 features) | 0.4220 | 8.29 |
| ElasticNet (24 features) | 0.4922 | 7.42 |
| Random Forest (24 features) | 0.5537 | 6.88 |
| Ridge (best 5 LOSO-safe) | 0.6580 | 5.99 |

**Best 5 LOSO-safe features**: mean_PM25_nn_idw, mean_PBLH, mean_VC, rain_freq, slope_deg

## LOO-CV predicting LOSO R² (LOSO-safe features)

| Model | R² | MAE |
|-------|:---:|:---:|
| Ridge (24 features) | -0.4225 | 0.4618 |

## Interpretation

Features flagged **LEAK** use the station's own PM2.5 data (diurnal_pm25_range, seasonal_pm25_amp, nn1_corr). These inflate the all-features LOO R² but are unavailable in LOSO evaluation.

The LOSO-safe LOO R² represents the real ceiling for a Stage 1 spatial baseline that predicts station mean PM2.5 without using the target station's own observations.
