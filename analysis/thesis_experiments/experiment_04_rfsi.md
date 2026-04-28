# Experiment 04: RFSI Nearest-Station Features

**Date:** 2026-04-28 09:41
**Dataset:** 727,635 rows, 40 stations
**XGBoost:** v3.2.0, n_estimators=500, max_depth=7, lr=0.05, device=cuda
**K6 overrides:** max_depth=5, colsample_bytree=0.6, min_child_weight=20, monotonic constraints
**RFSI:** K=5 nearest neighbors, haversine distances

## Comparison Table

| Config | Description | Features | KFold R² | LOSO R² (mean) | LOSO R² (median) | Neg Stations | Gap |
|--------|-------------|----------|----------|----------------|------------------|--------------|-----|
| C (Exp01) | Absolute baseline | 62 | 0.7262 | -0.4953 | -0.0004 | 20 | 1.2215 |
| E (Exp02) | Oracle anomaly | 55 | 0.6926 | 0.2252 | 0.2640 | 7 | 0.4674 |
| K1 | RFSI + temporal | 19 | 0.7855 | -0.3397 | 0.0060 | 19 | 1.1252 |
| K2 | Full + RFSI | 75 | 0.8099 | -0.1139 | 0.0058 | 20 | 0.9238 |
| K3 | Met+AOD+RFSI (no geo) | 67 | 0.7977 | -0.3927 | -0.0537 | 21 | 1.1904 |
| K4 | Met+RFSI (no AOD) | 44 | 0.7967 | -0.4353 | 0.0597 | 19 | 1.2320 |
| K5 | Minimal physics | 17 | 0.7622 | -0.4226 | 0.0236 | 19 | 1.1848 |
| K6 | K5 + constrained XGB | 17 | 0.7225 | -0.5850 | -0.0606 | 21 | 1.3075 |

## Per-Station LOSO: Config K2 vs Config C

| Station | Region | C R² | K2 R² | Delta | K2 RMSE |
|---------|--------|------|------|-------|--------|
| Trà Vinh xã Đông Hải, huyện Duyên Hải (KK) | South | -3.8829 | -3.7899 | +0.0930 ✓ | 10.9 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | -0.8672 | -1.3008 | -0.4336 | 17.8 |
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | -0.5723 | -1.2574 | -0.6851 | 12.5 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | -6.0323 | -1.0131 | +5.0192 ✓ | 12.7 |
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK) | North | 0.0952 | -0.7762 | -0.8714 | 10.0 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. Sóc Tr | South | -8.7966 | -0.6451 | +8.1515 ✓ | 13.0 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì (KK) | North | -0.1844 | -0.5124 | -0.3280 | 25.8 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị trấn Hồ (KK | North | 0.2329 | -0.5003 | -0.7332 | 30.2 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | -0.1673 | -0.3183 | -0.1510 | 15.6 |
| Long An UBND Tp Tân An - 76 Hùng Vương - P.2 (KK) | South | -0.4058 | -0.2831 | +0.1227 ✓ | 14.8 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | 0.0121 | -0.2810 | -0.2931 | 22.2 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy (KK) | North | -0.1564 | -0.2798 | -0.1234 | 23.2 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà Nẵng (KK) | Central | -0.3357 | -0.2679 | +0.0678 ✓ | 16.3 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | -1.6790 | -0.2470 | +1.4320 ✓ | 10.2 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | -0.9883 | -0.1736 | +0.8147 ✓ | 7.0 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | -0.2355 | -0.1338 | +0.1017 ✓ | 9.5 |
| Bình Định Khuôn viên Cây xanh gần cầu chui đường H | Central | -0.2247 | -0.0540 | +0.1707 ✓ | 12.4 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK) | North | -0.1850 | -0.0503 | +0.1347 ✓ | 46.6 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - Hạ L | North | 0.0295 | -0.0174 | -0.0469 | 7.7 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | -0.0285 | -0.0161 | +0.0124 ✓ | 18.8 |
| Bình Định huyện Tuy Phước (KK) | Central | 0.1857 | 0.0278 | -0.1579 | 13.3 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Quảng Ngãi ( | Central | 0.0164 | 0.1198 | +0.1034 ✓ | 23.4 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phường 8 - TP V | South | -0.1059 | 0.1480 | +0.2539 ✓ | 7.5 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơn - Phường | Central | 0.2038 | 0.1678 | -0.0360 | 18.9 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả, đườ | North | 0.0782 | 0.1731 | +0.0949 ✓ | 6.6 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC Đ. Hồ Xuân | Central | 0.1732 | 0.1745 | +0.0013 ✓ | 12.8 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Thống Nhất - | Central | -0.0129 | 0.2004 | +0.2133 ✓ | 8.1 |
| Bình Dương số 593 Đại lộ Bình Dương, P. Hiệp Thành | South | -0.0190 | 0.2193 | +0.2383 ✓ | 14.8 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa - TP Bắc N | North | -0.0312 | 0.2371 | +0.2683 ✓ | 18.7 |
| Thái Nguyên Sân vận động Gang thép - P Trung Thành | North | 0.0805 | 0.2434 | +0.1629 ✓ | 38.4 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 0.1150 | 0.2949 | +0.1799 ✓ | 10.7 |
| Hải Dương UBND TP. Hải Dương - 106 Đường Trần Hưng | North | 0.5190 | 0.4237 | -0.0953 | 22.6 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên (KK) | North | 0.4371 | 0.5073 | +0.0702 ✓ | 25.4 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái Tông - P. B | North | 0.4875 | 0.5125 | +0.0250 ✓ | 23.9 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Môi Trường - | South | 0.1903 | 0.5170 | +0.3267 ✓ | 8.7 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 0.4249 | 0.5406 | +0.1157 ✓ | 22.6 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - Quận 2 (N | South | 0.3235 | 0.5877 | +0.2642 ✓ | 8.8 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung - TP. Phủ | North | 0.4997 | 0.7311 | +0.2314 ✓ | 15.7 |
| Hà Nội Công viên Nhân Chính - Khuất Duy Tiến (KK) | North | 0.3721 | 0.7550 | +0.3829 ✓ | 11.6 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK) | North | 0.6220 | 0.7812 | +0.1592 ✓ | 17.4 |

## Regional Breakdown

| Region | C LOSO R² | K1 R² | K2 R² | K3 R² | K4 R² | K5 R² | K6 R² |
|--------|-----------|----------|----------|----------|----------|----------|----------|
| North | 0.0458 | -0.2522 | 0.0545 | -0.3793 | -0.3656 | -0.4378 | -0.6908 |
| Central | -0.0211 | -0.0603 | 0.0038 | -0.0552 | -0.0646 | -0.0492 | -0.0904 |
| South | -2.1908 | -0.4777 | -0.4925 | -0.4806 | -0.4768 | -0.4639 | -0.7295 |

## Feature Importance (Config K2, top 20)

| Rank | Feature | Gain | RFSI? |
|------|---------|------|-------|
| 1 | PM25_nn_idw | 792610 | yes |
| 2 | PM25_nn_mean | 539207 | yes |
| 3 | slope_deg | 137732 |  |
| 4 | longitude | 132755 |  |
| 5 | dist_nn4 | 66265 | yes |
| 6 | aspect_cos | 61022 |  |
| 7 | AOT_inner_mean | 59118 |  |
| 8 | aspect_sin | 58604 |  |
| 9 | latitude | 56525 |  |
| 10 | elevation_m | 45805 |  |
| 11 | dist_nn5 | 44746 | yes |
| 12 | month_cos | 40968 |  |
| 13 | PM25_nn1 | 39072 | yes |
| 14 | day_of_year_cos | 35977 |  |
| 15 | dist_nn2 | 34965 | yes |
| 16 | dist_nn1 | 34304 | yes |
| 17 | AOT_outer_mean | 31644 |  |
| 18 | dist_nn3 | 25604 | yes |
| 19 | wind_dir_sin_local | 25128 |  |
| 20 | AOT_mean | 24796 |  |

### RFSI Feature Rankings

| Feature | Rank | Gain |
|---------|------|------|
| PM25_nn1 | 13 | 39072 |
| PM25_nn2 | 22 | 23227 |
| PM25_nn3 | 42 | 12158 |
| PM25_nn4 | 30 | 15391 |
| PM25_nn5 | 39 | 12520 |
| dist_nn1 | 16 | 34304 |
| dist_nn2 | 15 | 34965 |
| dist_nn3 | 18 | 25604 |
| dist_nn4 | 5 | 66265 |
| dist_nn5 | 11 | 44746 |
| n_neighbors_available | not used | 0 |
| PM25_nn_mean | 2 | 539207 |
| PM25_nn_idw | 1 | 792610 |

## Analysis

### 1. Does RFSI alone (K1) beat the baseline?

- Config C baseline: LOSO R² = -0.4953
- Config K1 (RFSI only): LOSO R² = -0.3397
- RFSI alone beats the baseline by +0.1556
- Oracle ceiling (Config E): LOSO R² = 0.2252

### 2. Does Full + RFSI (K2) set a new best?

- Config K2: LOSO R² = -0.1139
- Better than baseline but below oracle (0.2252)

### 3. K5 minimal physics vs K2 kitchen-sink

- K2 (75 features): LOSO R² = -0.1139
- K5 (17 features): LOSO R² = -0.4226
- Delta: -0.3087

### 4. Do monotonic constraints help? (K6 vs K5)

- K5 (unconstrained): LOSO R² = -0.4226, neg=19
- K6 (constrained):   LOSO R² = -0.5850, neg=21
- Delta: -0.1624
- Constraints do not help

### 5. KFold-LOSO gap

- K1: KFold=0.7855, LOSO=-0.3397, gap=1.1252
- K2: KFold=0.8099, LOSO=-0.1139, gap=0.9238
- K3: KFold=0.7977, LOSO=-0.3927, gap=1.1904
- K4: KFold=0.7967, LOSO=-0.4353, gap=1.2320
- K5: KFold=0.7622, LOSO=-0.4226, gap=1.1848
- K6: KFold=0.7225, LOSO=-0.5850, gap=1.3075
- Baseline (C) gap: 1.2215