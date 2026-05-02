# Experiment 11: Satellite-Only vs Full Stage 1 Baselines

**Date:** 2026-05-02 11:44
**Dataset:** unified_thesis_v2.csv — 727,635 rows, 40 stations
**XGBoost:** v3.2.0, n_estimators=500, max_depth=7, lr=0.05, device=cuda

## Stage 1 Baselines

| Baseline | Features | LOSO R² | LOSO MAE |
|----------|---------|:---:|:---:|
| S1_full (Best-7) | mean_PM25_nn_idw, mean_AOT_valid_frac, mean_WS, mean_VC, rain_freq, slope_deg, mean_AOT_grad_mag | 0.6758 | 6.13 |
| S1_sat (Best-7) | mean_AOT_outer_mean, mean_AOT_inner_mean, mean_AOT_grad_mag, latitude, mean_SSA_inner_mean_clean, mean_SSA_grad_mag_clean, mean_SSA_local_vs_regional_clean | 0.6214 | 6.54 |

## Configs

- **V1**: S1_full baseline + B1 features (single-stage)
- **V2**: S1_sat baseline + B1 features (single-stage)
- **V3**: S1_full two-stage (met+AOD+RFSI, no geography)
- **V4**: S1_sat two-stage (met+AOD+RFSI, no geography)
- **V5**: S1_full baseline + met+AOD+RFSI+buildings (no geo)

## Comparison Table

| Config | Description | KFold R² | LOSO R² (mean) | LOSO R² (median) | LOSO MAE | Neg Stations |
|--------|-------------|:---:|:---:|:---:|:---:|:---:|
| B1 (ref) | Exp07: Full+RFSI+buildings | 0.8105 | -0.0197 | -0.0038 | 11.52 | 20 |
| T2 (ref) | Exp10: B1 + Ridge baseline | 0.8117 | 0.0052 | 0.1129 | 11.26 | 15 |
| **V1** | S1_full baseline + B1 features (single-stage) | 0.8117 | **-0.0426** | 0.1084 | 11.29 | 16 |
| **V2** | S1_sat baseline + B1 features (single-stage) | 0.8117 | **0.0072** | 0.1035 | 11.24 | 16 |
| **V3** | S1_full two-stage (met+AOD+RFSI, no geography) | 0.7978 | **-0.4793** | -0.1198 | 13.21 | 25 |
| **V4** | S1_sat two-stage (met+AOD+RFSI, no geography) | 0.7978 | **-0.3212** | -0.0103 | 12.44 | 20 |
| **V5** | S1_full baseline + met+AOD+RFSI+buildings (no geo) | 0.8089 | **-0.0721** | 0.0701 | 11.49 | 19 |

## Per-Station LOSO: V2

| Station | Region | B1 R² | T2 R² | V2 R² | Δ vs B1 | S1_full err | S1_sat err |
|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | -0.6345 | -0.5753 | -1.8277 | -1.1932 | +14.1 | +18.8 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | -0.4196 | -0.7006 | -1.1864 | -0.7668 | +6.2 | +14.1 |
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (K | North | -0.4785 | -1.4840 | -1.1024 | -0.6239 | +12.4 | +7.7 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | -1.3350 | -1.3742 | -0.7164 | +0.6186 | +12.2 | +9.1 |
| Trà Vinh xã Đông Hải, huyện Duyên Hải (K | South | -1.5549 | -0.6500 | -0.6500 | +0.9049 | +5.6 | +0.1 |
| Bình Định Khuôn viên Cây xanh gần cầu ch | Central | -0.1164 | 0.1001 | -0.4548 | -0.3384 | -5.0 | -10.9 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị t | North | -0.3357 | -0.1535 | -0.1535 | +0.1822 | +6.4 | +6.5 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | -0.3612 | -0.1386 | -0.1386 | +0.2226 | +4.7 | +4.4 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy ( | North | -0.1226 | -0.3051 | -0.1158 | +0.0068 | +3.2 | +0.9 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | -0.0526 | -0.1497 | -0.1147 | -0.0621 | -8.5 | -1.4 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | -0.0728 | -0.1299 | -0.0835 | -0.0107 | -5.4 | +1.3 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành  | North | -0.0285 | -0.0526 | -0.0526 | -0.0241 | -17.6 | -19.3 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | 0.0614 | -0.0693 | -0.0482 | -0.1096 | +7.0 | +15.4 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | -0.0273 | -0.0372 | -0.0427 | -0.0154 | +1.8 | +3.3 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà  | Central | 0.1215 | 0.0717 | -0.0350 | -0.1565 | -1.0 | -9.2 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩ | North | -0.6991 | 0.2179 | -0.0101 | +0.6890 | +1.1 | +6.1 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Th | Central | -0.1248 | 0.0375 | 0.0288 | +0.1536 | -1.9 | -2.4 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6,  | South | -0.5035 | 0.0807 | 0.0596 | +0.5631 | -4.3 | +3.5 |
| Bình Định huyện Tuy Phước (KK) | Central | 0.0197 | -0.3792 | 0.0610 | +0.0413 | +0.1 | -8.1 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Qu | Central | 0.1273 | 0.0888 | 0.0811 | -0.0462 | -10.7 | -7.8 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Tr | North | 0.1467 | 0.1258 | 0.1258 | -0.0209 | -6.2 | -1.8 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì ( | North | -0.4438 | -0.9042 | 0.1276 | +0.5714 | +10.6 | +9.3 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơ | Central | -0.2252 | 0.1442 | 0.1442 | +0.3694 | +4.0 | -1.0 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phườn | South | 0.0911 | 0.1755 | 0.1755 | +0.0844 | -4.2 | -0.6 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | -0.1860 | 0.1351 | 0.2041 | +0.3901 | +16.4 | +4.3 |
| Thái Nguyên Sân vận động Gang thép - P T | North | 0.2878 | 0.2289 | 0.2289 | -0.0589 | -21.6 | -22.1 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Mô | South | 0.4356 | 0.2582 | 0.2586 | -0.1770 | -4.6 | +2.9 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC  | Central | 0.1388 | 0.1272 | 0.2787 | +0.1399 | -0.5 | +2.7 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưn | North | 0.4148 | 0.7630 | 0.2807 | -0.1341 | -2.2 | -7.5 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 0.3115 | 0.2465 | 0.3041 | -0.0074 | +0.4 | +10.0 |
| Long An UBND Tp Tân An - 76 Hùng Vương - | South | 0.2623 | 0.3712 | 0.3304 | +0.0681 | -0.9 | -2.2 |
| Bình Dương số 593 Đại lộ Bình Dương, P.  | South | 0.3564 | 0.1770 | 0.3473 | -0.0091 | -8.6 | -0.4 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa  | North | -0.2515 | 0.3547 | 0.3547 | +0.6062 | +10.7 | +8.3 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái T | North | 0.5066 | 0.3609 | 0.3609 | -0.1457 | -1.8 | -3.5 |
| Hà Nội Công viên Nhân Chính - Khuất Duy  | North | 0.7110 | 0.3703 | 0.3713 | -0.3397 | +4.2 | +1.9 |
| Hải Dương UBND TP. Hải Dương - 106 Đường | North | 0.5904 | 0.4290 | 0.4290 | -0.1614 | -5.1 | -2.0 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 0.5897 | 0.5135 | 0.5380 | -0.0517 | -1.9 | -12.7 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - | South | 0.5127 | 0.5893 | 0.5841 | +0.0714 | -3.5 | +1.4 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung | North | 0.7393 | 0.6075 | 0.6075 | -0.1318 | -5.0 | -6.9 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phón | North | 0.7593 | 0.7366 | 0.7385 | -0.0208 | -3.3 | -9.6 |

## Summary

- **V1**: LOSO R²=-0.0426 (median=0.1084), MAE=11.29, 16 negative stations
- **V2**: LOSO R²=0.0072 (median=0.1035), MAE=11.24, 16 negative stations
- **V3**: LOSO R²=-0.4793 (median=-0.1198), MAE=13.21, 25 negative stations
- **V4**: LOSO R²=-0.3212 (median=-0.0103), MAE=12.44, 20 negative stations
- **V5**: LOSO R²=-0.0721 (median=0.0701), MAE=11.49, 19 negative stations

**Best (V2) vs B1:** +0.0269
**Best (V2) vs T2 (Exp10):** +0.0020