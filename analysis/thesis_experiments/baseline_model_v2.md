# Baseline Model V2 — Pushing LOSO-Safe LOO R²

**Date**: 2026-04-29 13:01
**Target**: station mean PM2.5 (40 stations, LOO-CV)

## All Results

| # | Model | R² | MAE (µg/m³) |
|---|-------|:---:|:---:|
| 1 | Best-8 + 6 interactions RF | 0.6695 | 5.79 |
| 2 | Best-7 RF (200 trees, depth=4) | 0.6659 | 5.97 |
| 3 | Best-8 RF (200 trees, depth=4) | 0.6613 | 5.99 |
| 4 | Best-6 RF (200 trees, depth=4) | 0.6606 | 6.20 |
| 5 | Best-8 Ridge LOSO-safe | 0.6581 | 6.12 |
| 6 | Best-5 Ridge (baseline) | 0.6580 | 5.99 |
| 7 | Best-7 ElasticNet | 0.6573 | 6.07 |
| 8 | Best-8 ElasticNet | 0.6569 | 6.09 |
| 9 | Best-7 Ridge LOSO-safe | 0.6566 | 6.06 |
| 10 | Best-6 Ridge LOSO-safe | 0.6560 | 5.97 |
| 11 | Best-6 ElasticNet | 0.6542 | 5.95 |
| 12 | Best-5 + nbr_diurnal_range | 0.6317 | 6.24 |
| 13 | Best-5 + ACAG_annual_mean | 0.6276 | 6.23 |
| 14 | Best-8 + 6 interactions ElasticNet | 0.6256 | 6.42 |
| 15 | Best-8 + 12 ACAG monthly ElasticNet | 0.6145 | 6.36 |
| 16 | Best-8 + 6 interactions Ridge | 0.6122 | 6.59 |
| 17 | Best-5 + 6 interactions Ridge | 0.6119 | 6.42 |
| 18 | Best-5 + nbr_diurnal_range + ACAG_annual | 0.5977 | 6.48 |
| 19 | Best-8 + 12 ACAG monthly Ridge | 0.5971 | 6.51 |
| 20 | All LOSO-safe + interactions + ACAG monthly RF | 0.5773 | 6.47 |
| 21 | All 24 LOSO-safe RF | 0.5537 | 6.88 |
| 22 | Best-5 + 12 ACAG monthly | 0.5497 | 7.19 |
| 23 | Best-5 + ACAG_annual + 12 monthly | 0.5496 | 7.19 |
| 24 | All LOSO-safe + interactions + ACAG monthly EN | 0.5193 | 7.17 |
| 25 | All 24 LOSO-safe ElasticNet | 0.4922 | 7.42 |
| 26 | All 24 LOSO-safe Ridge | 0.4220 | 8.29 |

## Best Subsets

**Best 6**: mean_PM25_nn_idw, mean_PBLH, mean_VC, rain_freq, building_area_3km, slope_deg (R²=0.6560, MAE=5.97)

**Best 7**: mean_PM25_nn_idw, mean_PBLH, mean_WS, rain_freq, building_area_3km, longitude, slope_deg (R²=0.6566, MAE=6.06)

**Best 8**: mean_PM25_nn_idw, AOT_valid_frac, mean_PBLH, mean_WS, rain_freq, building_area_3km, longitude, slope_deg (R²=0.6581, MAE=6.12)

## Feature Interactions Tested

| Name | Formula |
|------|--------|
| idw_x_ACAG | mean_PM25_nn_idw × ACAG_annual_mean |
| ndr_x_bldg | nbr_diurnal_range × building_area_3km |
| VC_x_rain | mean_VC × rain_freq |
| idw_x_ndr | mean_PM25_nn_idw × nbr_diurnal_range |
| ACAG_x_lat | ACAG_annual_mean × latitude |
| idw_x_PBLH | mean_PM25_nn_idw × mean_PBLH |

## Per-Station Errors — Best Model: Best-8 + 6 interactions RF

R² = 0.6695, MAE = 5.79

| Station | Region | Actual | Predicted | Error | |Error| |
|---------|--------|:---:|:---:|:---:|:---:|
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK) | North | 52.2 | 31.0 | +21.1 | 21.1 |
| Thái Nguyên Sân vận động Gang thép - P Trung  | North | 55.2 | 34.5 | +20.7 | 20.7 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa - TP  | North | 23.4 | 41.4 | -18.0 | 18.0 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | 6.2 | 21.3 | -15.1 | 15.1 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì (KK) | North | 27.1 | 41.5 | -14.4 | 14.4 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị trấn H | North | 27.9 | 41.2 | -13.2 | 13.2 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. S | South | 6.7 | 16.2 | -9.5 | 9.5 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Quảng N | Central | 27.4 | 18.1 | +9.3 | 9.3 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 48.5 | 39.3 | +9.3 | 9.3 |
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK) | North | 6.8 | 15.4 | -8.5 | 8.5 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | 13.2 | 21.1 | -7.9 | 7.9 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK | North | 46.8 | 40.2 | +6.6 | 6.6 |
| Long An UBND Tp Tân An - 76 Hùng Vương - P.2  | South | 21.2 | 14.7 | +6.5 | 6.5 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy (KK) | North | 15.5 | 21.1 | -5.7 | 5.7 |
| Bình Định Khuôn viên Cây xanh gần cầu chui đư | Central | 18.5 | 13.1 | +5.4 | 5.4 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên | North | 43.2 | 37.8 | +5.4 | 5.4 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà Nẵng  | Central | 23.6 | 18.8 | +4.8 | 4.8 |
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | 9.2 | 14.0 | -4.8 | 4.8 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | 25.4 | 29.9 | -4.5 | 4.5 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | 10.9 | 15.0 | -4.1 | 4.1 |
| Trà Vinh xã Đông Hải, huyện Duyên Hải (KK) | South | 5.7 | 9.6 | -3.9 | 3.9 |
| Bình Dương số 593 Đại lộ Bình Dương, P. Hiệp  | South | 24.2 | 20.8 | +3.4 | 3.4 |
| Bình Định huyện Tuy Phước (KK) | Central | 12.1 | 15.3 | -3.3 | 3.3 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phường 8 - | South | 14.7 | 11.6 | +3.1 | 3.1 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC Đ. Hồ | Central | 19.9 | 16.9 | +3.0 | 3.0 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái Tông - | North | 37.2 | 34.4 | +2.8 | 2.8 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả | North | 7.0 | 9.2 | -2.3 | 2.3 |
| Hải Dương UBND TP. Hải Dương - 106 Đường Trần | North | 37.6 | 35.6 | +2.1 | 2.1 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | 10.9 | 9.0 | +1.9 | 1.9 |
| Hà Nội Công viên Nhân Chính - Khuất Duy Tiến  | North | 36.9 | 38.8 | -1.8 | 1.8 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | 7.6 | 9.5 | -1.8 | 1.8 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - Quận | South | 22.6 | 20.9 | +1.6 | 1.6 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 9.5 | 8.4 | +1.1 | 1.1 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Môi Trư | South | 21.3 | 22.4 | -1.0 | 1.0 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - | North | 6.6 | 7.6 | -1.0 | 1.0 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | 9.1 | 9.9 | -0.7 | 0.7 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơn - P | Central | 18.1 | 17.4 | +0.7 | 0.7 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Thống N | Central | 15.5 | 16.1 | -0.6 | 0.6 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung - TP | North | 39.2 | 38.6 | +0.6 | 0.6 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | 10.6 | 10.4 | +0.2 | 0.2 |

## Key Findings

- Baseline (best-5 Ridge): R²=0.6580, MAE=5.99
- Best model: Best-8 + 6 interactions RF — R²=0.6695, MAE=5.79
- Improvement: +0.0115 R², -0.20 MAE
