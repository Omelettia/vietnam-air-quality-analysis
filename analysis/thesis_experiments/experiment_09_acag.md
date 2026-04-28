# Experiment 09: ACAG Satellite Climatology

**Date:** 2026-04-28 21:30
**Dataset:** 727,635 rows, 40 stations
**ACAG:** V6.GL.02.04, 0.1° monthly AS, 2020-2023 climatology
**XGBoost:** v3.2.0, n_estimators=500, max_depth=7, lr=0.05, device=cuda
**RFSI:** K=5 nearest neighbors

## ACAG Diagnostics

- Correlation ACAG_annual vs PM2.5: 0.3717
- Correlation ACAG_monthly vs PM2.5: 0.4275
- ACAG annual mean vs station mean R² (across 40 stations): 0.3082
  (Compare to Ridge LOO R² from Exp08 which was ~0.3-0.5)
- ACAG vs station mean |error|: median=9.0, max=26.4 µg/m³

## Comparison Table

| Config | Description | Features | KFold R² | LOSO R² (mean) | LOSO R² (median) | Neg Stations | Gap |
|--------|-------------|----------|----------|----------------|------------------|--------------|-----|
| C (Exp01) | Absolute baseline | 62 | 0.7262 | -0.4953 | -0.0004 | 20 | 1.2215 |
| E (Exp02) | Oracle anomaly | 55 | 0.6926 | 0.2252 | 0.2640 | 7 | 0.4674 |
| K2 (Exp04) | Full + RFSI | — | 0.8099 | -0.1139 | 0.0058 | 20 | 0.9238 |
| B1 (Exp07) | K2 + buildings | — | 0.8105 | -0.0197 | -0.0038 | 20 | 0.8302 |
| S1 (Exp08) | Two-stage Ridge+XGB | — | 0.7978 | -0.4731 | -0.1461 | 24 | 1.2709 |
| **A1** | **B1 + ACAG features (single-stage)** | 81 | 0.8116 | **-0.0535** | -0.0298 | 21 | 0.8651 |
| A2 | ACAG baseline + XGB residual (met+AOD+RFSI) | 67 | 0.7950 | -0.3387 | -0.0079 | 21 | 1.1337 |
| A3 | ACAG baseline + XGB residual (+buildings) | 71 | 0.8023 | -0.2419 | -0.1142 | 26 | 1.0442 |

## Per-Station LOSO: A1 vs B1 vs S1

| Station | Region | B1 R² | S1 R² | A1 R² | Δ vs B1 | ACAG diff | A1 RMSE |
|---------|--------|-------|-------|-------|---------|-----------|--------|
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | -0.6345 | -1.4392 | -1.4033 | -0.7688 | +19.3 | 12.9 |
| Trà Vinh xã Đông Hải, huyện Duyên Hải (KK) | South | -1.5549 | -0.2710 | -1.3912 | +0.1637 ✓ | +11.8 | 7.7 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | -1.3350 | -0.9100 | -1.3904 | -0.0554 | +15.2 | 18.1 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả | North | -0.6991 | -3.1302 | -1.1864 | -0.4873 | +14.1 | 10.8 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị trấn H | North | -0.3357 | -1.2625 | -0.4936 | -0.1579 | +8.6 | 30.1 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì (KK) | North | -0.4438 | -0.2821 | -0.4346 | +0.0092 ✓ | +10.4 | 25.2 |
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK) | North | -0.4785 | -3.9428 | -0.3852 | +0.0933 ✓ | +13.1 | 8.8 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | -0.1860 | 0.0010 | -0.3076 | -0.1216 | +8.8 | 22.4 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | -0.3612 | 0.0035 | -0.3062 | +0.0550 ✓ | +7.0 | 21.3 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa - TP  | North | -0.2515 | 0.1196 | -0.2860 | -0.0345 | +12.0 | 24.3 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Thống N | Central | -0.1248 | -0.4358 | -0.1897 | -0.0649 | +7.0 | 9.9 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. S | South | -0.5035 | -1.3544 | -0.1852 | +0.3183 ✓ | +14.8 | 11.1 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơn - P | Central | -0.2252 | 0.0567 | -0.1717 | +0.0535 ✓ | +1.7 | 22.4 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | 0.0614 | -0.0815 | -0.1600 | -0.2214 | +26.4 | 9.9 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | -0.0728 | -1.6769 | -0.1453 | -0.0725 | +14.1 | 9.6 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | -0.0526 | -0.3662 | -0.1399 | -0.0873 | +6.0 | 6.8 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy (KK) | North | -0.1226 | -0.9852 | -0.1364 | -0.0138 | +8.9 | 21.9 |
| Bình Định Khuôn viên Cây xanh gần cầu chui đư | Central | -0.1164 | -2.8613 | -0.1094 | +0.0070 ✓ | -0.0 | 12.7 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK) | North | -0.0285 | -0.2032 | -0.0453 | -0.0168 | -15.7 | 46.5 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | -0.4196 | -1.4995 | -0.0399 | +0.3797 ✓ | +13.8 | 9.2 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phường 8 - | South | 0.0911 | -0.6023 | -0.0197 | -0.1108 | +7.0 | 8.2 |
| Bình Định huyện Tuy Phước (KK) | Central | 0.0197 | -0.6247 | 0.0236 | +0.0039 ✓ | +6.4 | 13.3 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | -0.0273 | -0.0889 | 0.0270 | +0.0543 ✓ | +7.8 | 13.4 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Quảng N | Central | 0.1273 | 0.0255 | 0.0912 | -0.0361 | -9.2 | 23.8 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC Đ. Hồ | Central | 0.1388 | 0.0425 | 0.0978 | -0.0410 | +0.6 | 13.4 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - | North | 0.1467 | -1.1086 | 0.1230 | -0.0237 | +14.2 | 7.2 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà Nẵng  | Central | 0.1215 | -0.2442 | 0.1633 | +0.0418 ✓ | -1.4 | 13.3 |
| Long An UBND Tp Tân An - 76 Hùng Vương - P.2  | South | 0.2623 | -0.0577 | 0.2296 | -0.0327 | +3.2 | 11.5 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 0.3115 | -0.0122 | 0.2761 | -0.0354 | +15.3 | 10.9 |
| Thái Nguyên Sân vận động Gang thép - P Trung  | North | 0.2878 | -0.2714 | 0.2897 | +0.0019 ✓ | -23.7 | 37.2 |
| Bình Dương số 593 Đại lộ Bình Dương, P. Hiệp  | South | 0.3564 | 0.2081 | 0.3197 | -0.0367 | +4.2 | 13.8 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên | North | 0.4148 | 0.6571 | 0.3630 | -0.0518 | -10.9 | 28.9 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Môi Trư | South | 0.4356 | 0.2303 | 0.4240 | -0.0116 | +13.6 | 9.5 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - Quận | South | 0.5127 | 0.1290 | 0.4737 | -0.0390 | +12.1 | 9.9 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái Tông - | North | 0.5066 | 0.4318 | 0.4991 | -0.0075 | -8.5 | 24.2 |
| Hải Dương UBND TP. Hải Dương - 106 Đường Trần | North | 0.5904 | 0.5973 | 0.5794 | -0.0110 | -4.6 | 19.3 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 0.5897 | 0.4616 | 0.5826 | -0.0071 | -10.7 | 21.6 |
| Hà Nội Công viên Nhân Chính - Khuất Duy Tiến  | North | 0.7110 | 0.4838 | 0.7223 | +0.0113 ✓ | +4.3 | 12.4 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung - TP | North | 0.7393 | 0.7291 | 0.7400 | +0.0007 ✓ | -7.5 | 15.4 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK | North | 0.7593 | 0.6126 | 0.7603 | +0.0010 ✓ | -8.9 | 18.2 |

## Regional Breakdown

| Region | K2 R² | B1 R² | S1 R² | A1 R² | A2 R² | A3 R² |
|--------|------|------|------|------|------|------|
| North | 0.0545 | 0.0888 | -0.4906 | -0.0023 | -0.3505 | -0.2266 |
| Central | 0.0038 | -0.0497 | -0.4585 | -0.0416 | -0.0921 | -0.0989 |
| South | -0.4925 | -0.0969 | -0.3982 | -0.0365 | -0.4132 | -0.2585 |

## Feature Importance (Config A1, top 20)

| Rank | Feature | Gain | Type |
|------|---------|------|------|
| 1 | PM25_nn_idw | 776471 | RFSI |
| 2 | PM25_nn_mean | 506805 | RFSI |
| 3 | ACAG_annual_mean | 283564 | ACAG |
| 4 | building_area_3km | 252766 | building |
| 5 | longitude | 118329 |  |
| 6 | slope_deg | 111109 |  |
| 7 | building_count_3km | 56608 | building |
| 8 | elevation_m | 53492 |  |
| 9 | aspect_sin | 50958 |  |
| 10 | aspect_cos | 50608 |  |
| 11 | dist_nn4 | 45809 | RFSI |
| 12 | dist_nn5 | 43807 | RFSI |
| 13 | building_count_1km | 41703 | building |
| 14 | dist_nn3 | 39634 | RFSI |
| 15 | month_cos | 39312 |  |
| 16 | building_area_1km | 37495 | building |
| 17 | latitude | 36945 |  |
| 18 | PM25_nn1 | 36601 | RFSI |
| 19 | day_of_year_cos | 32594 |  |
| 20 | dist_nn1 | 31808 | RFSI |

## Analysis

### 1. ACAG quality as spatial baseline

- ACAG annual vs station mean R² = 0.3082
- Median |error| = 9.0 µg/m³
- ACAG is satellite-derived (van Donkelaar et al.) — no ground station leakage in LOSO
- This is a key advantage over Ridge baseline which uses station-derived building density

### 2. A1: Does adding ACAG to B1 help?

- B1 LOSO R² = -0.0197
- A1 LOSO R² = -0.0535
- Delta: -0.0338

### 3. A2 vs A3: Do buildings help Stage 2?

- A2 (met+AOD+RFSI): LOSO R² = -0.3387
- A3 (+buildings):    LOSO R² = -0.2419
- Building effect: +0.0968

### 4. ACAG two-stage vs Ridge two-stage (A2 vs S1)

- S1 (Ridge baseline): LOSO R² = -0.4731
- A2 (ACAG baseline):  LOSO R² = -0.3387
- Delta: +0.1344
- ACAG uses external satellite data, Ridge uses station-derived static features

### 5. Progression summary

| Experiment | Config | LOSO R² | Key addition |
|------------|--------|---------|--------------|
| Exp01 | C | -0.4953 | Absolute baseline |
| Exp02 | E | +0.2252 | Oracle anomaly (ceiling) |
| Exp04 | K2 | -0.1139 | RFSI neighbors |
| Exp07 | B1 | -0.0197 | Building density |
| Exp08 | S1 | -0.4731 | Two-stage Ridge |
| Exp09 | A1 | -0.0535 | ACAG satellite climatology |
| — | Oracle | +0.2252 | Perfect per-station mean |

### 6. KFold-LOSO gap

- A1: KFold=0.8116, LOSO=-0.0535, gap=0.8651
- A2: KFold=0.7950, LOSO=-0.3387, gap=1.1337
- A3: KFold=0.8023, LOSO=-0.2419, gap=1.0442