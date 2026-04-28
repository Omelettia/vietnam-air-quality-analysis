# Experiment 08: Two-Stage Architecture

**Date:** 2026-04-28 15:18
**Dataset:** 727,635 rows, 40 stations
**Stage 1:** Ridge regression on station means (8 static features, α=1.0)
**Stage 2:** XGBoost on residuals (met + AOD + RFSI, no geographic features)
**XGBoost:** v3.2.0, n_estimators=500, max_depth=7, lr=0.05, device=cuda
**RFSI:** K=5 nearest neighbors

## Stage 1: Station-Mean Prediction (Ridge LOO)

- **LOO R²:** 0.3240
- **LOO RMSE:** 11.44 µg/m³
- **LOO MAE:** 9.03 µg/m³

### Ridge Coefficients (standardized)

| Feature | Coefficient |
|---------|------------|
| building_area_1km | +8.204 |
| building_count_1km | -7.138 |
| building_area_3km | +4.177 |
| longitude | -4.118 |
| latitude | +4.053 |
| elevation_m | -1.599 |
| slope_deg | +1.301 |
| building_count_3km | -0.048 |
| intercept | 21.885 |

### Station-Level Predictions (sorted by |error|)

| Station | Region | Actual Mean | Predicted | Error |
|---------|--------|------------|-----------|-------|
| Thái Nguyên Sân vận động Gang thép - P Trung  | North | 55.2 | 24.4 | -30.8 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK) | North | 52.2 | 22.6 | -29.6 |
| Bình Định Khuôn viên Cây xanh gần cầu chui đư | Central | 18.5 | -1.3 | -19.8 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả | North | 7.0 | 26.2 | +19.2 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | 10.6 | 27.3 | +16.7 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên | North | 43.2 | 27.0 | -16.2 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | 7.6 | 23.3 | +15.7 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. S | South | 6.7 | 20.8 | +14.1 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 48.5 | 34.9 | -13.7 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - | North | 6.6 | 20.2 | +13.6 |
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | 9.2 | 22.2 | +13.0 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | 6.2 | 17.3 | +11.1 |
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK) | North | 6.9 | 17.3 | +10.4 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa - TP  | North | 23.4 | 33.5 | +10.0 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 9.5 | 18.8 | +9.3 |
| Hải Dương UBND TP. Hải Dương - 106 Đường Trần | North | 37.6 | 29.3 | -8.3 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Quảng N | Central | 27.4 | 19.2 | -8.2 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | 25.4 | 17.4 | -8.0 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | 10.9 | 18.9 | +7.9 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì (KK) | North | 27.1 | 34.8 | +7.7 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Môi Trư | South | 21.3 | 28.7 | +7.3 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phường 8 - | South | 14.7 | 8.2 | -6.5 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung - TP | North | 39.2 | 33.0 | -6.1 |
| Hà Nội Công viên Nhân Chính - Khuất Duy Tiến  | North | 36.9 | 43.0 | +6.0 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơn - P | Central | 18.1 | 23.9 | +5.8 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - Quận | South | 22.6 | 17.2 | -5.4 |
| Bình Định huyện Tuy Phước (KK) | Central | 12.1 | 7.2 | -4.8 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK | North | 46.8 | 42.1 | -4.7 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Thống N | Central | 15.5 | 10.9 | -4.6 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | 13.2 | 17.6 | +4.4 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC Đ. Hồ | Central | 19.9 | 15.9 | -3.9 |
| Bình Dương số 593 Đại lộ Bình Dương, P. Hiệp  | South | 24.2 | 27.9 | +3.7 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị trấn H | North | 27.9 | 24.7 | -3.3 |
| Trà Vinh xã Đông Hải, huyện Duyên Hải (KK) | South | 5.7 | 8.5 | +2.8 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà Nẵng  | Central | 23.6 | 21.4 | -2.2 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy (KK) | North | 15.5 | 17.4 | +1.9 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | 10.9 | 9.2 | -1.7 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | 9.1 | 7.6 | -1.5 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái Tông - | North | 37.2 | 36.3 | -0.9 |
| Long An UBND Tp Tân An - 76 Hùng Vương - P.2  | South | 21.2 | 21.4 | +0.3 |

## Comparison Table

| Config | Description | Features | KFold R² | LOSO R² (mean) | LOSO R² (median) | Neg Stations | Gap |
|--------|-------------|----------|----------|----------------|------------------|--------------|-----|
| C (Exp01) | Absolute baseline | 62 | 0.7262 | -0.4953 | -0.0004 | 20 | 1.2215 |
| E (Exp02) | Oracle anomaly | 55 | 0.6926 | 0.2252 | 0.2640 | 7 | 0.4674 |
| K2 (Exp04) | Full + RFSI | — | 0.8099 | -0.1139 | 0.0058 | 20 | 0.9238 |
| B1 (Exp07) | K2 + buildings | — | 0.8105 | -0.0197 | -0.0038 | 20 | 0.8302 |
| **S1** | **Two-stage: Ridge + XGB (met+AOD+RFSI)** | 67 | 0.7978 | **-0.4731** | -0.1461 | 24 | 1.2709 |
| S2 | Two-stage: Ridge + XGB (met+RFSI) | 44 | 0.7966 | -0.4872 | -0.1376 | 24 | 1.2838 |
| S3 | Ridge baseline only (no temporal) | 8 | 0.3240 | -0.8007 | -0.1661 | 40 | 1.1247 |

## Per-Station LOSO: S1 vs B1 vs K2 vs C

| Station | Region | C R² | K2 R² | B1 R² | S1 R² | Δ vs B1 | S1 baseline err | S1 RMSE |
|---------|--------|------|-------|-------|-------|---------|----------------|--------|
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK) | North | 0.0952 | -0.7762 | -0.4785 | -3.9428 | -3.4643 | +10.4 | 16.7 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả | North | 0.0782 | 0.1731 | -0.6991 | -3.1302 | -2.4311 | +19.2 | 14.8 |
| Bình Định Khuôn viên Cây xanh gần cầu chui đư | Central | -0.2247 | -0.0540 | -0.1164 | -2.8613 | -2.7449 | -19.8 | 23.8 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | -0.2355 | -0.1338 | -0.0728 | -1.6769 | -1.6041 | +15.7 | 14.7 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | -6.0323 | -1.0131 | -0.4196 | -1.4995 | -1.0799 | +7.9 | 14.2 |
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | -0.5723 | -1.2574 | -0.6345 | -1.4392 | -0.8047 | +13.0 | 13.0 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. S | South | -8.7966 | -0.6451 | -0.5035 | -1.3544 | -0.8509 | +14.1 | 15.6 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị trấn H | North | 0.2329 | -0.5003 | -0.3357 | -1.2625 | -0.9268 | -3.3 | 37.1 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - | North | 0.0295 | -0.0174 | 0.1467 | -1.1086 | -1.2553 | +13.6 | 11.1 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy (KK) | North | -0.1564 | -0.2798 | -0.1226 | -0.9852 | -0.8626 | +1.9 | 28.9 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | -0.8672 | -1.3008 | -1.3350 | -0.9100 | +0.4250 ✓ | +11.1 | 16.2 |
| Bình Định huyện Tuy Phước (KK) | Central | 0.1857 | 0.0278 | 0.0197 | -0.6247 | -0.6444 | -4.8 | 17.2 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phường 8 - | South | -0.1059 | 0.1480 | 0.0911 | -0.6023 | -0.6934 | -6.5 | 10.2 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Thống N | Central | -0.0129 | 0.2004 | -0.1248 | -0.4358 | -0.3110 | -4.6 | 10.9 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | -0.9883 | -0.1736 | -0.0526 | -0.3662 | -0.3136 | -1.5 | 7.5 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì (KK) | North | -0.1844 | -0.5124 | -0.4438 | -0.2821 | +0.1617 ✓ | +7.7 | 23.8 |
| Thái Nguyên Sân vận động Gang thép - P Trung  | North | 0.0805 | 0.2434 | 0.2878 | -0.2714 | -0.5592 | -30.8 | 49.7 |
| Trà Vinh xã Đông Hải, huyện Duyên Hải (KK) | South | -3.8829 | -3.7899 | -1.5549 | -0.2710 | +1.2839 ✓ | +2.8 | 5.6 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà Nẵng  | Central | -0.3357 | -0.2679 | 0.1215 | -0.2442 | -0.3657 | -2.2 | 16.2 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK) | North | -0.1850 | -0.0503 | -0.0285 | -0.2032 | -0.1747 | -29.6 | 49.8 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | -0.1673 | -0.3183 | -0.0273 | -0.0889 | -0.0616 | -1.7 | 14.2 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | -1.6790 | -0.2470 | 0.0614 | -0.0815 | -0.1429 | +16.7 | 9.5 |
| Long An UBND Tp Tân An - 76 Hùng Vương - P.2  | South | -0.4058 | -0.2831 | 0.2623 | -0.0577 | -0.3200 | +0.3 | 13.5 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 0.1150 | 0.2949 | 0.3115 | -0.0122 | -0.3237 | +9.3 | 12.8 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | 0.0121 | -0.2810 | -0.1860 | 0.0010 | +0.1870 ✓ | -8.0 | 19.6 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | -0.0285 | -0.0161 | -0.3612 | 0.0035 | +0.3647 ✓ | +4.4 | 18.6 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Quảng N | Central | 0.0164 | 0.1198 | 0.1273 | 0.0255 | -0.1018 | -8.2 | 24.6 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC Đ. Hồ | Central | 0.1732 | 0.1745 | 0.1388 | 0.0425 | -0.0963 | -3.9 | 13.8 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơn - P | Central | 0.2038 | 0.1678 | -0.2252 | 0.0567 | +0.2819 ✓ | +5.8 | 20.1 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa - TP  | North | -0.0312 | 0.2371 | -0.2515 | 0.1196 | +0.3711 ✓ | +10.0 | 20.1 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - Quận | South | 0.3235 | 0.5877 | 0.5127 | 0.1290 | -0.3837 | -5.4 | 12.8 |
| Bình Dương số 593 Đại lộ Bình Dương, P. Hiệp  | South | -0.0190 | 0.2193 | 0.3564 | 0.2081 | -0.1483 | +3.7 | 14.9 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Môi Trư | South | 0.1903 | 0.5170 | 0.4356 | 0.2303 | -0.2053 | +7.3 | 11.0 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái Tông - | North | 0.4875 | 0.5125 | 0.5066 | 0.4318 | -0.0748 | -0.9 | 25.8 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 0.4249 | 0.5406 | 0.5897 | 0.4616 | -0.1281 | -13.7 | 24.5 |
| Hà Nội Công viên Nhân Chính - Khuất Duy Tiến  | North | 0.3721 | 0.7550 | 0.7110 | 0.4838 | -0.2272 | +6.0 | 16.9 |
| Hải Dương UBND TP. Hải Dương - 106 Đường Trần | North | 0.5190 | 0.4237 | 0.5904 | 0.5973 | +0.0069 ✓ | -8.3 | 18.9 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK | North | 0.6220 | 0.7812 | 0.7593 | 0.6126 | -0.1467 | -4.7 | 23.1 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên | North | 0.4371 | 0.5073 | 0.4148 | 0.6571 | +0.2423 ✓ | -16.2 | 21.2 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung - TP | North | 0.4997 | 0.7311 | 0.7393 | 0.7291 | -0.0102 | -6.1 | 15.7 |

## Regional Breakdown

| Region | C R² | K2 R² | B1 R² | S1 R² | S2 R² | S3 R² |
|--------|------|------|------|------|------|------|
| North | 0.0458 | 0.0545 | 0.0888 | -0.4906 | -0.4390 | -1.1166 |
| Central | -0.0211 | 0.0038 | -0.0497 | -0.4585 | -0.6208 | -0.3799 |
| South | -2.1908 | -0.4925 | -0.0969 | -0.3982 | -0.4130 | -0.4740 |

## Feature Importance (Config S1 Stage 2, top 20)

| Rank | Feature | Gain | Type |
|------|---------|------|------|
| 1 | PM25_nn_idw | 400155 | RFSI |
| 2 | PM25_nn_mean | 138799 | RFSI |
| 3 | month_cos | 78816 |  |
| 4 | AOT_outer_mean | 59038 |  |
| 5 | dist_nn4 | 54215 | RFSI |
| 6 | dist_nn5 | 53556 | RFSI |
| 7 | PM25_nn2 | 52435 | RFSI |
| 8 | dist_nn2 | 50281 | RFSI |
| 9 | PM25_nn1 | 46880 | RFSI |
| 10 | AOT_valid_count | 46546 |  |
| 11 | dist_nn1 | 38503 | RFSI |
| 12 | dist_nn3 | 34675 | RFSI |
| 13 | AOT_mean | 30581 |  |
| 14 | day_of_year_cos | 29670 |  |
| 15 | PM25_nn4 | 29187 | RFSI |
| 16 | month_sin | 29064 |  |
| 17 | wind_u_local | 26060 |  |
| 18 | wind_dir_sin_local | 21249 |  |
| 19 | PM25_nn3 | 20777 | RFSI |
| 20 | rain_sum_48h | 20767 |  |

## Analysis

### 1. Stage 1 quality — can Ridge predict station means?

- Stage 1 LOO R² = 0.3240
- Stage 1 LOO RMSE = 11.44 µg/m³
- Ridge captures 30-50% — moderate baseline, room for improvement

### 2. Two-stage vs single-stage (S1 vs B1 vs K2)

- K2 (single-stage, full+RFSI): LOSO R² = -0.1139
- B1 (single-stage, +buildings): LOSO R² = -0.0197
- S1 (two-stage): LOSO R² = -0.4731
- Delta S1 vs B1: -0.4534

### 3. Does AOD help in Stage 2? (S1 vs S2)

- S1 (met+AOD+RFSI): LOSO R² = -0.4731
- S2 (met+RFSI):     LOSO R² = -0.4872
- AOD effect: +0.0141

### 4. S3 Ridge-only — how much does the baseline alone explain?

- S3 LOSO R² = -0.8007 (predicting station mean for every hour)
- S1 LOSO R² = -0.4731
- Stage 2 adds +0.3276 R² on top of the baseline

### 5. KFold-LOSO gap

- S1: KFold=0.7978, LOSO=-0.4731, gap=1.2709
- S2: KFold=0.7966, LOSO=-0.4872, gap=1.2838
- S3: KFold=0.3240, LOSO=-0.8007, gap=1.1247
- Baseline (C) gap: 1.2215
- Oracle (E) gap: 0.4674

### 6. Stations where Stage 1 error determines outcome

Stations with |Stage 1 error| > 10 µg/m³:

- Thái Nguyên Sân vận động Gang thép - P Trung : baseline err=-30.8, LOSO R²=-0.2714
- Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK): baseline err=-29.6, LOSO R²=-0.2032
- Bình Định Khuôn viên Cây xanh gần cầu chui đư: baseline err=-19.8, LOSO R²=-2.8613
- Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả: baseline err=+19.2, LOSO R²=-3.1302
- Quảng Ninh UBND TP Uông Bí (KK): baseline err=+16.7, LOSO R²=-0.0815
- Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên: baseline err=-16.2, LOSO R²=0.6571
- Quảng Ninh Gần KCN Cái Lân (KK): baseline err=+15.7, LOSO R²=-1.6769
- Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. S: baseline err=+14.1, LOSO R²=-1.3544
- Hà Nội 556 Nguyễn Văn Cừ (KK): baseline err=-13.7, LOSO R²=0.4616
- Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng -: baseline err=+13.6, LOSO R²=-1.1086
- Quảng Ninh Nhuệ Hổ - Đông Triều (KK): baseline err=+13.0, LOSO R²=-1.4392
- Đà Nẵng Phạm Hùng (KK): baseline err=+11.1, LOSO R²=-0.9100
- Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK): baseline err=+10.4, LOSO R²=-3.9428
- Bắc Ninh TT Quan trắc - phường Suối Hoa - TP : baseline err=+10.0, LOSO R²=0.1196