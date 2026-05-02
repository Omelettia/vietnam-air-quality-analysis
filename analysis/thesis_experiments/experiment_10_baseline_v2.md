# Experiment 10: Improved Two-Stage with LOSO-Safe Baseline

**Date:** 2026-04-29 13:19
**Dataset:** 727,635 rows, 40 stations
**Stage 1:** Ridge on 6 LOSO-safe features (α=1.0)
**Stage 1 LOO R²:** 0.6548
**XGBoost:** v3.2.0, n_estimators=500, max_depth=7, lr=0.05, device=cuda

## Stage 1 Features

| Feature | Coefficient (standardized) |
|---------|:---:|
| mean_PM25_nn_idw | +10.175 |
| mean_PBLH | +3.765 |
| mean_VC | -4.719 |
| rain_freq | -2.862 |
| slope_deg | +2.213 |
| building_area_3km | +1.680 |

## Comparison Table

| Config | Description | KFold R² | LOSO R² (mean) | LOSO R² (median) | Neg Stations |
|--------|-------------|:---:|:---:|:---:|:---:|
| B1 (Exp07) | Full+RFSI+buildings | 0.8105 | -0.0197 | -0.0038 | 20 |
| S1 (Exp08) | Two-stage: static Ridge + XGB | 0.7978 | -0.4731 | -0.1461 | 24 |
| T1 | Two-stage: LOSO-safe Ridge + XGB (met+AOD+RFSI) | 0.7978 | -0.4555 | 0.0004 | 20 |
| **T2** | **B1 + Ridge baseline as feature** | 0.8117 | **0.0052** | 0.1129 | 15 |
| T3 | Two-stage + buildings + ACAG_monthly | 0.8061 | -0.3996 | -0.0169 | 22 |

## Per-Station LOSO: T2 vs B1 vs S1

| Station | Region | B1 R² | S1 R² | T2 R² | Δ vs B1 | Stage1 err | T2 RMSE |
|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (K | North | -0.4785 | -3.9428 | -1.4840 | -1.0055 | +8.6 | 11.8 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | -1.3350 | -0.9100 | -1.3742 | -0.0392 | +13.6 | 18.1 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì ( | North | -0.4438 | -0.2821 | -0.9042 | -0.4604 | +12.3 | 29.0 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | -0.4196 | -1.4995 | -0.7006 | -0.2810 | +8.9 | 11.7 |
| Trà Vinh xã Đông Hải, huyện Duyên Hải (K | South | -1.5549 | -0.2710 | -0.6500 | +0.9049 | -4.0 | 6.4 |
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | -0.6345 | -1.4392 | -0.5753 | +0.0592 | +8.6 | 10.4 |
| Bình Định huyện Tuy Phước (KK) | Central | 0.0197 | -0.6247 | -0.3792 | -0.3989 | +7.2 | 15.9 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy ( | North | -0.1226 | -0.9852 | -0.3051 | -0.1825 | +5.8 | 23.4 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị t | North | -0.3357 | -1.2625 | -0.1535 | +0.1822 | +7.7 | 26.5 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | -0.0526 | -0.3662 | -0.1497 | -0.0971 | -12.5 | 6.9 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | -0.3612 | 0.0035 | -0.1386 | +0.2226 | +2.4 | 19.9 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | -0.0728 | -1.6769 | -0.1299 | -0.0571 | -4.7 | 9.5 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | 0.0614 | -0.0815 | -0.0693 | -0.1307 | +5.9 | 9.5 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành  | North | -0.0285 | -0.2032 | -0.0526 | -0.0241 | -22.9 | 46.6 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | -0.0273 | -0.0889 | -0.0372 | -0.0099 | +1.9 | 13.8 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Th | Central | -0.1248 | -0.4358 | 0.0375 | +0.1623 | +3.0 | 8.9 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà  | Central | 0.1215 | -0.2442 | 0.0717 | -0.0498 | -7.4 | 14.0 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6,  | South | -0.5035 | -1.3544 | 0.0807 | +0.5842 | +1.6 | 9.7 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Qu | Central | 0.1273 | 0.0255 | 0.0888 | -0.0385 | -6.0 | 23.8 |
| Bình Định Khuôn viên Cây xanh gần cầu ch | Central | -0.1164 | -2.8613 | 0.1001 | +0.2165 | +1.3 | 11.5 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Tr | North | 0.1467 | -1.1086 | 0.1258 | -0.0209 | -1.9 | 7.2 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC  | Central | 0.1388 | 0.0425 | 0.1272 | -0.0116 | -4.2 | 13.2 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | -0.1860 | 0.0010 | 0.1351 | +0.3211 | -5.0 | 18.2 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơ | Central | -0.2252 | 0.0567 | 0.1442 | +0.3694 | +0.1 | 19.1 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phườn | South | 0.0911 | -0.6023 | 0.1755 | +0.0844 | +0.6 | 7.3 |
| Bình Dương số 593 Đại lộ Bình Dương, P.  | South | 0.3564 | 0.2081 | 0.1770 | -0.1794 | -10.9 | 15.2 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩ | North | -0.6991 | -3.1302 | 0.2179 | +0.9170 | -1.4 | 6.5 |
| Thái Nguyên Sân vận động Gang thép - P T | North | 0.2878 | -0.2714 | 0.2289 | -0.0589 | -21.6 | 38.7 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 0.3115 | -0.0122 | 0.2465 | -0.0650 | +0.4 | 11.1 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Mô | South | 0.4356 | 0.2303 | 0.2582 | -0.1774 | +2.7 | 10.8 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa  | North | -0.2515 | 0.1196 | 0.3547 | +0.6062 | +11.4 | 17.2 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái T | North | 0.5066 | 0.4318 | 0.3609 | -0.1457 | -2.7 | 27.3 |
| Hà Nội Công viên Nhân Chính - Khuất Duy  | North | 0.7110 | 0.4838 | 0.3703 | -0.3407 | +6.2 | 18.6 |
| Long An UBND Tp Tân An - 76 Hùng Vương - | South | 0.2623 | -0.0577 | 0.3712 | +0.1089 | +0.5 | 10.4 |
| Hải Dương UBND TP. Hải Dương - 106 Đường | North | 0.5904 | 0.5973 | 0.4290 | -0.1614 | -4.3 | 22.5 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 0.5897 | 0.4616 | 0.5135 | -0.0762 | -6.2 | 23.3 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - | South | 0.5127 | 0.1290 | 0.5893 | +0.0766 | -0.9 | 8.8 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung | North | 0.7393 | 0.7291 | 0.6075 | -0.1318 | -2.7 | 18.9 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phón | North | 0.7593 | 0.6126 | 0.7366 | -0.0227 | +0.1 | 19.1 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưn | North | 0.4148 | 0.6571 | 0.7630 | +0.3482 | -2.3 | 17.6 |

## Summary

- **T1**: LOSO R²=-0.4555 (median=0.0004), RMSE=18.28, 20 negative stations
- **T2**: LOSO R²=0.0052 (median=0.1129), RMSE=16.46, 15 negative stations
- **T3**: LOSO R²=-0.3996 (median=-0.0169), RMSE=18.07, 22 negative stations

**Best config (T2) vs B1:** +0.0249
**Best config (T2) vs S1 (Exp08):** +0.4783