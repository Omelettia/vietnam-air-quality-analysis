# Experiment 07: Building Density as Spatial Baseline Proxy

**Date:** 2026-04-28 11:25
**Dataset:** 727,635 rows, 40 stations
**XGBoost:** v3.2.0, n_estimators=500, max_depth=7, lr=0.05, device=cuda
**RFSI:** K=5 nearest neighbors
**Building density:** Google Open Buildings, 1km and 3km radius

## Hypothesis

The architecture review identified station-mean PM2.5 (driven by local emissions and urbanization) as the strongest predictor of LOSO failure (Spearman r=0.60). Building density from Google Open Buildings serves as a proxy for urbanization intensity — the missing spatial baseline signal that the model needs to distinguish clean rural stations from polluted urban ones without memorizing station identity.

## Building Density Statistics

| Station | Region | PM2.5 mean | Bldg count 1km | Bldg area 1km (m²) | Bldg count 3km |
|---------|--------|-----------|----------------|---------------------|----------------|
| Trà Vinh xã Đông Hải, huyện Duyên Hải (KK) | South | 5.7 | 1,137 | 88,552 | 3,218 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | 6.2 | 10,070 | 731,721 | 69,361 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - | North | 6.6 | 7,421 | 503,538 | 33,315 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. S | South | 6.7 | 8,509 | 648,582 | 50,762 |
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK) | North | 6.9 | 7,797 | 463,730 | 26,614 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả | North | 7.0 | 19,432 | 1,130,495 | 56,913 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | 7.6 | 2,161 | 306,725 | 23,554 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | 9.1 | 1,196 | 140,789 | 5,479 |
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | 9.2 | 2,815 | 274,278 | 25,781 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 9.5 | 4,458 | 319,533 | 12,238 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | 10.6 | 11,091 | 812,310 | 37,740 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | 10.9 | 7,578 | 656,835 | 35,505 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | 10.9 | 1,680 | 432,928 | 14,979 |
| Bình Định huyện Tuy Phước (KK) | Central | 12.1 | 11,044 | 755,846 | 56,755 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | 13.2 | 24,542 | 1,381,606 | 117,427 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phường 8 - | South | 14.7 | 7,128 | 515,396 | 26,374 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy (KK) | North | 15.5 | 1,852 | 104,138 | 22,818 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Thống N | Central | 15.5 | 14,579 | 1,078,770 | 75,208 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơn - P | Central | 18.1 | 14,328 | 1,299,741 | 73,556 |
| Bình Định Khuôn viên Cây xanh gần cầu chui đư | Central | 18.5 | 14,069 | 695,750 | 59,122 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC Đ. Hồ | Central | 19.9 | 15,938 | 1,140,437 | 69,385 |
| Long An UBND Tp Tân An - 76 Hùng Vương - P.2  | South | 21.2 | 14,792 | 1,075,336 | 67,700 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Môi Trư | South | 21.3 | 19,372 | 1,191,166 | 180,765 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - Quận | South | 22.6 | 9,875 | 742,088 | 51,613 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa - TP  | North | 23.4 | 7,384 | 726,050 | 54,599 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà Nẵng  | Central | 23.6 | 14,839 | 1,044,518 | 105,100 |
| Bình Dương số 593 Đại lộ Bình Dương, P. Hiệp  | South | 24.2 | 11,654 | 1,002,761 | 83,660 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | 25.4 | 2,112 | 151,503 | 16,908 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì (KK) | North | 27.1 | 7,464 | 678,896 | 45,790 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Quảng N | Central | 27.4 | 17,749 | 1,324,589 | 79,042 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị trấn H | North | 27.9 | 5,133 | 439,302 | 28,705 |
| Hà Nội Công viên Nhân Chính - Khuất Duy Tiến  | North | 36.9 | 5,162 | 673,091 | 114,969 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái Tông - | North | 37.2 | 10,705 | 933,021 | 65,988 |
| Hải Dương UBND TP. Hải Dương - 106 Đường Trần | North | 37.6 | 12,969 | 937,079 | 68,676 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung - TP | North | 39.2 | 11,912 | 1,036,340 | 57,930 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên | North | 43.2 | 7,246 | 598,127 | 42,383 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK | North | 46.8 | 15,685 | 1,079,642 | 180,144 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 48.5 | 9,870 | 872,509 | 73,659 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK) | North | 52.2 | 4,042 | 415,747 | 34,963 |
| Thái Nguyên Sân vận động Gang thép - P Trung  | North | 55.2 | 5,849 | 594,943 | 26,071 |

## Comparison Table (all configs)

| Config | Description | Features | KFold R² | LOSO R² (mean) | LOSO R² (median) | Neg Stations | Gap |
|--------|-------------|----------|----------|----------------|------------------|--------------|-----|
| C (Exp01) | Absolute baseline | 62 | 0.7262 | -0.4953 | -0.0004 | 20 | 1.2215 |
| E (Exp02) | Oracle anomaly | 55 | 0.6926 | 0.2252 | 0.2640 | 7 | 0.4674 |
| K1 (Exp04) | RFSI + temporal | — | 0.7855 | -0.3397 | 0.0060 | 19 | 1.1252 |
| K2 (Exp04) | Full + RFSI | — | 0.8099 | -0.1139 | 0.0058 | 20 | 0.9238 |
| K3 (Exp04) | Met+AOD+RFSI (no geo) | — | 0.7977 | -0.3927 | -0.0537 | 21 | 1.1904 |
| K4 (Exp04) | Met+RFSI (no AOD) | — | 0.7967 | -0.4353 | 0.0597 | 19 | 1.2320 |
| K5 (Exp04) | Minimal physics | — | 0.7622 | -0.4226 | 0.0236 | 19 | 1.1848 |
| K6 (Exp04) | K5 + constrained XGB | — | 0.7225 | -0.5850 | -0.0606 | 21 | 1.3075 |
| **B1** | **K2 + buildings** | 79 | 0.8105 | **-0.0197** | -0.0038 | 20 | 0.8302 |
| **B2** | **K3 + buildings (no geo)** | 71 | 0.8053 | **-0.2254** | -0.0506 | 23 | 1.0307 |
| **B3** | **Minimal RFSI + buildings + physics** | 17 | 0.7520 | **-0.1370** | 0.0024 | 20 | 0.8890 |

## Per-Station LOSO: Config B1 vs K2 vs Config C

| Station | Region | C R² | K2 R² | B1 R² | Δ vs K2 | B1 RMSE |
|---------|--------|------|-------|-------|---------|--------|
| Trà Vinh xã Đông Hải, huyện Duyên Hải (KK) | South | -3.8829 | -3.7899 | -1.5549 | +2.2350 ✓ | 7.9 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | -0.8672 | -1.3008 | -1.3350 | -0.0342 | 17.9 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả | North | 0.0782 | 0.1731 | -0.6991 | -0.8722 | 9.5 |
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | -0.5723 | -1.2574 | -0.6345 | +0.6229 ✓ | 10.6 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. S | South | -8.7966 | -0.6451 | -0.5035 | +0.1416 ✓ | 12.4 |
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK) | North | 0.0952 | -0.7762 | -0.4785 | +0.2977 ✓ | 9.1 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì (KK) | North | -0.1844 | -0.5124 | -0.4438 | +0.0686 ✓ | 25.2 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | -6.0323 | -1.0131 | -0.4196 | +0.5935 ✓ | 10.7 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | -0.0285 | -0.0161 | -0.3612 | -0.3451 | 21.8 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị trấn H | North | 0.2329 | -0.5003 | -0.3357 | +0.1646 ✓ | 28.5 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa - TP  | North | -0.0312 | 0.2371 | -0.2515 | -0.4886 | 24.0 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơn - P | Central | 0.2038 | 0.1678 | -0.2252 | -0.3930 | 22.9 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | 0.0121 | -0.2810 | -0.1860 | +0.0950 ✓ | 21.3 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Thống N | Central | -0.0129 | 0.2004 | -0.1248 | -0.3252 | 9.6 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy (KK) | North | -0.1564 | -0.2798 | -0.1226 | +0.1572 ✓ | 21.7 |
| Bình Định Khuôn viên Cây xanh gần cầu chui đư | Central | -0.2247 | -0.0540 | -0.1164 | -0.0624 | 12.8 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | -0.2355 | -0.1338 | -0.0728 | +0.0610 ✓ | 9.3 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | -0.9883 | -0.1736 | -0.0526 | +0.1210 ✓ | 6.6 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK) | North | -0.1850 | -0.0503 | -0.0285 | +0.0218 ✓ | 46.1 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | -0.1673 | -0.3183 | -0.0273 | +0.2910 ✓ | 13.8 |
| Bình Định huyện Tuy Phước (KK) | Central | 0.1857 | 0.0278 | 0.0197 | -0.0081 | 13.4 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | -1.6790 | -0.2470 | 0.0614 | +0.3084 ✓ | 8.9 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phường 8 - | South | -0.1059 | 0.1480 | 0.0911 | -0.0569 | 7.7 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà Nẵng  | Central | -0.3357 | -0.2679 | 0.1215 | +0.3894 ✓ | 13.6 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Quảng N | Central | 0.0164 | 0.1198 | 0.1273 | +0.0075 ✓ | 23.3 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC Đ. Hồ | Central | 0.1732 | 0.1745 | 0.1388 | -0.0357 | 13.1 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - | North | 0.0295 | -0.0174 | 0.1467 | +0.1641 ✓ | 7.1 |
| Long An UBND Tp Tân An - 76 Hùng Vương - P.2  | South | -0.4058 | -0.2831 | 0.2623 | +0.5454 ✓ | 11.3 |
| Thái Nguyên Sân vận động Gang thép - P Trung  | North | 0.0805 | 0.2434 | 0.2878 | +0.0444 ✓ | 37.2 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 0.1150 | 0.2949 | 0.3115 | +0.0166 ✓ | 10.6 |
| Bình Dương số 593 Đại lộ Bình Dương, P. Hiệp  | South | -0.0190 | 0.2193 | 0.3564 | +0.1371 ✓ | 13.5 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên | North | 0.4371 | 0.5073 | 0.4148 | -0.0925 | 27.7 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Môi Trư | South | 0.1903 | 0.5170 | 0.4356 | -0.0814 | 9.4 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái Tông - | North | 0.4875 | 0.5125 | 0.5066 | -0.0059 | 24.0 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - Quận | South | 0.3235 | 0.5877 | 0.5127 | -0.0750 | 9.5 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 0.4249 | 0.5406 | 0.5897 | +0.0491 ✓ | 21.4 |
| Hải Dương UBND TP. Hải Dương - 106 Đường Trần | North | 0.5190 | 0.4237 | 0.5904 | +0.1667 ✓ | 19.1 |
| Hà Nội Công viên Nhân Chính - Khuất Duy Tiến  | North | 0.3721 | 0.7550 | 0.7110 | -0.0440 | 12.6 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung - TP | North | 0.4997 | 0.7311 | 0.7393 | +0.0082 ✓ | 15.4 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK | North | 0.6220 | 0.7812 | 0.7593 | -0.0219 | 18.2 |

## Regional Breakdown

| Region | C R² | K2 R² | B1 R² | B2 R² | B3 R² |
|--------|------|------|------|------|------|
| North | 0.0458 | 0.0545 | 0.0888 | -0.1245 | 0.0558 |
| Central | -0.0211 | 0.0038 | -0.0497 | -0.1291 | -0.3307 |
| South | -2.1908 | -0.4925 | -0.0969 | -0.2041 | -0.2290 |

## Feature Importance (Config B1, top 20)

| Rank | Feature | Gain | Type |
|------|---------|------|------|
| 1 | PM25_nn_idw | 804152 | RFSI |
| 2 | PM25_nn_mean | 504430 | RFSI |
| 3 | building_area_3km | 238116 | BUILDING |
| 4 | longitude | 183220 |  |
| 5 | slope_deg | 119685 |  |
| 6 | building_count_3km | 59114 | BUILDING |
| 7 | aspect_sin | 52549 |  |
| 8 | dist_nn4 | 49808 | RFSI |
| 9 | aspect_cos | 49743 |  |
| 10 | elevation_m | 49284 |  |
| 11 | dist_nn5 | 43760 | RFSI |
| 12 | dist_nn3 | 42353 | RFSI |
| 13 | month_cos | 42164 |  |
| 14 | PM25_nn1 | 40113 | RFSI |
| 15 | building_area_1km | 37215 | BUILDING |
| 16 | AOT_inner_mean | 36589 |  |
| 17 | latitude | 35123 |  |
| 18 | day_of_year_cos | 34533 |  |
| 19 | dist_nn1 | 30158 | RFSI |
| 20 | building_count_1km | 29270 | BUILDING |

### Building Feature Rankings

| Feature | Rank | Gain |
|---------|------|------|
| building_count_1km | 20 | 29270 |
| building_area_1km | 15 | 37215 |
| building_count_3km | 6 | 59114 |
| building_area_3km | 3 | 238116 |

## Analysis

### 1. Do buildings help the kitchen-sink model? (B1 vs K2)

- K2 (full+RFSI, no buildings): LOSO R² = -0.1139
- B1 (K2 + buildings): LOSO R² = -0.0197
- Delta: +0.0942
- Buildings improve the full model

### 2. Can buildings replace geography? (B2 vs K3 vs K2)

- K2 (with geo): LOSO R² = -0.1139
- K3 (no geo, no buildings): LOSO R² = -0.3927
- B2 (no geo, with buildings): LOSO R² = -0.2254
- Buildings recover +0.1673 of the geo gap

### 3. Minimal model with buildings (B3)

- B3 (17 features): LOSO R² = -0.1370, neg=20
- Oracle ceiling (Config E): LOSO R² = 0.2252

### 4. KFold-LOSO gap (identity leakage diagnostic)

- B1: KFold=0.8105, LOSO=-0.0197, gap=0.8302
- B2: KFold=0.8053, LOSO=-0.2254, gap=1.0307
- B3: KFold=0.7520, LOSO=-0.1370, gap=0.8890
- Baseline (C) gap: 1.2215
- Oracle (E) gap: 0.4674

### 5. Impact on disaster stations

- Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. S: B1=-0.5035, K2=-0.6451, C=-8.7966, (bldg_1km=8,509)

### 6. Correlation: building density vs station PM2.5 mean

- Spearman(PM2.5 mean, building_count_1km) = 0.162 (p=0.3193)
- Spearman(PM2.5 mean, building_count_3km) = 0.385 (p=0.0143)
- Building density is a weak proxy for station PM2.5 level