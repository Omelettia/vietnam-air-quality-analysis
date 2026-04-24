# Experiment 03: Regional Daily Anomaly XGBoost

**Date:** 2026-04-24 13:51
**Dataset:** 727,635 rows, 40 stations
**XGBoost:** v3.2.0, n_estimators=500, max_depth=7, lr=0.05
**Best base config:** H3 | Overall best: H4

## Comparison Table (all R² on absolute PM2.5)

| Config | Description | Features | KFold R² | LOSO R² (mean) | LOSO R² (median) | Neg R² | Gap |
|--------|-------------|----------|----------|----------------|------------------|--------|-----|
| C (Exp01) | Full baseline | 62 | 0.6926 | 0.2252 | 0.2640 | 7 | 0.4674 |
| E (Exp02) | Oracle station anomaly | 55 | 0.6926 | 0.2252 | 0.2640 | 7 | 0.4674 |
| F K=5 (Exp02) | Neighbor station anomaly | 55 | 0.6926 | -0.3030 | -0.0364 | 22 | 0.9956 |
| H1 | National day+hour anomaly | 54 | 0.5597 | -1.0873 | -0.2405 | 27 | 1.6471 |
| H2 | Regional day+hour anomaly | 54 | 0.5794 | -1.2349 | -0.1975 | 26 | 1.8143 |
| H3 | National daily anomaly | 54 | 0.5922 | -0.9993 | -0.2298 | 29 | 1.5915 |
| H4 | H3 + terrain | 60 | 0.7134 | -0.8345 | -0.3753 | 30 | 1.5479 |

## Per-Station LOSO: H4 vs Config C vs Config F(K=3)

| Station | Region | C R² | F R² | H4 R² | Δ(C) | RMSE |
|---------|--------|------|------|--------|------|------|
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | -0.9883 | — | -7.3064 | -6.3181 | 18.5 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. Sóc Tr | South | -8.7966 | — | -4.9026 | +3.8940 ✓ | 24.6 |
| Trà Vinh xã Đông Hải, huyện Duyên Hải (KK) | South | -3.8829 | — | -3.7652 | +0.1177 ✓ | 10.8 |
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | -0.5723 | — | -2.6560 | -2.0837 | 15.9 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - Quận 2 (N | South | 0.3235 | — | -2.3200 | -2.6435 | 24.9 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | -1.6790 | — | -2.0132 | -0.3342 | 15.9 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Môi Trường - | South | 0.1903 | — | -1.8368 | -2.0271 | 21.1 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phường 8 - TP V | South | -0.1059 | — | -1.7714 | -1.6655 | 13.4 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | -0.2355 | — | -1.3628 | -1.1273 | 13.8 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | -6.0323 | — | -1.1871 | +4.8452 ✓ | 13.3 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | -0.1673 | — | -0.8521 | -0.6848 | 18.5 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả, đườ | North | 0.0782 | — | -0.7854 | -0.8636 | 9.8 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - Hạ L | North | 0.0295 | — | -0.7326 | -0.7621 | 10.1 |
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK) | North | 0.0952 | — | -0.6139 | -0.7091 | 9.5 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Quảng Ngãi ( | Central | 0.0164 | — | -0.5492 | -0.5656 | 31.0 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Thống Nhất - | Central | -0.0129 | — | -0.4782 | -0.4653 | 11.0 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | -0.0285 | — | -0.4614 | -0.4329 | 22.6 |
| Bình Định Khuôn viên Cây xanh gần cầu chui đường H | Central | -0.2247 | — | -0.4460 | -0.2213 | 14.5 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | 0.0121 | — | -0.4251 | -0.4372 | 23.4 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | -0.8672 | — | -0.4180 | +0.4492 ✓ | 14.0 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung - TP. Phủ | North | 0.4997 | — | -0.3326 | -0.8323 | 34.9 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên (KK) | North | 0.4371 | — | -0.2978 | -0.7349 | 41.2 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà Nẵng (KK) | Central | -0.3357 | — | -0.1610 | +0.1747 ✓ | 15.6 |
| Hà Nội Công viên Nhân Chính - Khuất Duy Tiến (KK) | North | 0.3721 | — | -0.0829 | -0.4550 | 24.4 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 0.1150 | — | -0.0810 | -0.1960 | 13.3 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy (KK) | North | -0.1564 | — | -0.0692 | +0.0872 ✓ | 21.2 |
| Bình Dương số 593 Đại lộ Bình Dương, P. Hiệp Thành | South | -0.0190 | — | -0.0553 | -0.0363 | 17.2 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK) | North | -0.1850 | — | -0.0368 | +0.1482 ✓ | 46.3 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa - TP Bắc N | North | -0.0312 | — | -0.0138 | +0.0174 ✓ | 21.6 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC Đ. Hồ Xuân | Central | 0.1732 | — | -0.0101 | -0.1833 | 14.2 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị trấn Hồ (KK | North | 0.2329 | — | 0.0192 | -0.2137 | 24.4 |
| Bình Định huyện Tuy Phước (KK) | Central | 0.1857 | — | 0.0623 | -0.1234 | 13.1 |
| Long An UBND Tp Tân An - 76 Hùng Vương - P.2 (KK) | South | -0.4058 | — | 0.0683 | +0.4741 ✓ | 12.7 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơn - Phường | Central | 0.2038 | — | 0.1228 | -0.0810 | 19.4 |
| Thái Nguyên Sân vận động Gang thép - P Trung Thành | North | 0.0805 | — | 0.1789 | +0.0984 ✓ | 40.0 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì (KK) | North | -0.1844 | — | 0.2491 | +0.4335 ✓ | 18.2 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 0.4249 | — | 0.4620 | +0.0371 ✓ | 24.5 |
| Hải Dương UBND TP. Hải Dương - 106 Đường Trần Hưng | North | 0.5190 | — | 0.4743 | -0.0447 | 21.6 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK) | North | 0.6220 | — | 0.5019 | -0.1201 | 26.2 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái Tông - P. B | North | 0.4875 | — | 0.5053 | +0.0178 ✓ | 24.0 |

## Regional Comparison

| Region | Config C | Config F(K=3) | H4 | Oracle E |
|--------|---------|---------------|------|----------|
| North | +0.0458 | +nan | -0.3387 | +0.2313 |
| Central | -0.1057 | +nan | -0.3191 | +0.2501 |
| South | -2.1908 | +nan | -2.5641 | +0.1888 |

## Feature Importance (H4, top 20)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | slope_deg | 732333760 |
| 2 | day_of_year_cos | 204714480 |
| 3 | elevation_m | 113343400 |
| 4 | AOT_outer_mean | 96519544 |
| 5 | aspect_sin | 91763184 |
| 6 | aspect_cos | 64749632 |
| 7 | WS_local | 62376620 |
| 8 | Pressure_regional_anom | 54726032 |
| 9 | VC | 54065152 |
| 10 | day_of_year_sin | 52745876 |
| 11 | PBLH_regional_anom | 43443908 |
| 12 | month_cos | 43067620 |
| 13 | Temperature_regional_anom | 38835848 |
| 14 | elev_x_PBLH | 36661900 |
| 15 | WS_regional_anom | 35551068 |
| 16 | dWS_6h | 34575660 |
| 17 | AOT_mean | 33329750 |
| 18 | AOT_ffill_48h | 33005436 |
| 19 | dRH_6h | 31620388 |
| 20 | Humidity_regional_anom | 29723002 |

## Anomaly-Space KFold R²

- H1: R²_anom = 0.5051 (absolute R² = 0.5597)
- H2: R²_anom = 0.4285 (absolute R² = 0.5794)
- H3: R²_anom = 0.5187 (absolute R² = 0.5922)
- H4: R²_anom = 0.6618 (absolute R² = 0.7134)

## Analysis

### 1. Does regional anomaly beat neighbor station anomaly? (H vs F)

- Best H config: H3 LOSO R² = -0.9993
- Config F (K=5): LOSO R² = -0.3030
- Regional anomaly beats neighbor? **NO** (Δ = -0.6963)

### 2. Does it approach the oracle ceiling? (H vs E)

- Best H: LOSO R² = -0.9993
- Oracle E: LOSO R² = 0.2252
- Gap to oracle: 1.2245

### 3. National vs region-specific mean — which works better?

- H1 (national): LOSO R² = -1.0873
- H2 (regional): LOSO R² = -1.2349
- Winner: **National (H1)**

### 4. Day+hour vs day-only — which is better?

- H1 (day+hour): LOSO R² = -1.0873
- H3 (day-only): LOSO R² = -0.9993
- Winner: **Day-only (H3)**

### 5. Does terrain help? (H4 vs best H)

- H3: LOSO R² = -0.9993
- H4 (H3+terrain): LOSO R² = -0.8345
- Terrain effect: +0.1648

### 6. Which features dominate?

Top 5 by gain: slope_deg, day_of_year_cos, elevation_m, AOT_outer_mean, aspect_sin
Geography in top 20: No — physics-based ✓