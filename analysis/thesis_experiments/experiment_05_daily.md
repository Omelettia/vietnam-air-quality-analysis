# Experiment 05: Daily Aggregation Test

**Date:** 2026-04-24 17:17
**Hourly dataset:** 727,635 rows
**Daily dataset:** 31,935 rows, 40 stations
**XGBoost:** v3.2.0, n_estimators=500, max_depth=7, lr=0.05, device=cuda
**RFSI:** K=5 nearest neighbors (daily PM2.5)

## Comparison Table — Hourly vs Daily

| Config | Resolution | Features | KFold R² | LOSO R² (mean) | LOSO R² (median) | Neg Stations | Gap |
|--------|------------|----------|----------|----------------|------------------|--------------|-----|
| K1 (Exp04) | Hourly | 19 | 0.7855 | -0.1520 | -0.0030 | ? | ? |
| K2 (Exp04) | Hourly | 75 | 0.8099 | -0.1140 | 0.0480 | ? | ? |
| E (Exp02) | Hourly | 55 | 0.6926 | 0.2252 | 0.2640 | 7 | 0.4674 |
| D_K1 | Daily | 17 | 0.8634 | -0.6524 | -0.0464 | 22 | 1.5158 |
| D_K2 | Daily | 72 | 0.8788 | -0.2521 | -0.0472 | 22 | 1.1309 |

## Per-Station LOSO: Daily D_K2 vs Hourly K2

| Station | Region | Hourly K2 R² | Daily D_K2 R² | Delta | Daily RMSE |
|---------|--------|--------------|------------|-------|------------|
| Trà Vinh xã Đông Hải, huyện Duyên Hải (KK) | South | -3.7899 | -7.9430 | -4.1531 | 11.0 |
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | -1.2574 | -1.7233 | -0.4659 | 12.6 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | -1.0131 | -1.5095 | -0.4964 | 9.9 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. Sóc Tr | South | -0.6451 | -1.4758 | -0.8307 | 10.5 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | -1.3008 | -1.0682 | +0.2326 ✓ | 13.0 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị trấn Hồ (KK | North | -0.5003 | -0.7809 | -0.2806 | 26.9 |
| Long An UBND Tp Tân An - 76 Hùng Vương - P.2 (KK) | South | -0.2831 | -0.6785 | -0.3954 | 12.0 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | -0.2470 | -0.5737 | -0.3267 | 9.9 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy (KK) | North | -0.2798 | -0.5636 | -0.2838 | 20.5 |
| Bình Định Khuôn viên Cây xanh gần cầu chui đường H | Central | -0.0540 | -0.5194 | -0.4654 | 9.0 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà Nẵng (KK) | Central | -0.2679 | -0.3695 | -0.1016 | 12.2 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK) | North | -0.0503 | -0.2381 | -0.1878 | 40.6 |
| Thái Nguyên Sân vận động Gang thép - P Trung Thành | North | 0.2434 | -0.2220 | -0.4654 | 38.5 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì (KK) | North | -0.5124 | -0.2140 | +0.2984 ✓ | 20.6 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | -0.2810 | -0.1673 | +0.1137 ✓ | 17.4 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | -0.1338 | -0.1507 | -0.0169 | 8.6 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | -0.0161 | -0.0916 | -0.0755 | 11.0 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - Hạ L | North | -0.0174 | -0.0788 | -0.0614 | 7.1 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Quảng Ngãi ( | Central | 0.1198 | -0.0713 | -0.1911 | 18.3 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | -0.1736 | -0.0567 | +0.1169 ✓ | 5.9 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | -0.3183 | -0.0377 | +0.2806 ✓ | 12.0 |
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK) | North | -0.7762 | -0.0060 | +0.7702 ✓ | 20.1 |
| Bình Định huyện Tuy Phước (KK) | Central | 0.0278 | 0.0111 | -0.0167 | 9.8 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa - TP Bắc N | North | 0.2371 | 0.0769 | -0.1602 | 17.3 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả, đườ | North | 0.1731 | 0.1524 | -0.0207 | 5.9 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Thống Nhất - | Central | 0.2004 | 0.1967 | -0.0037 | 5.2 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phường 8 - TP V | South | 0.1480 | 0.2189 | +0.0709 ✓ | 5.1 |
| Bình Dương số 593 Đại lộ Bình Dương, P. Hiệp Thành | South | 0.2193 | 0.2877 | +0.0684 ✓ | 10.9 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC Đ. Hồ Xuân | Central | 0.1745 | 0.3031 | +0.1286 ✓ | 7.9 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơn - Phường | Central | 0.1678 | 0.3230 | +0.1552 ✓ | 12.8 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 0.2949 | 0.3625 | +0.0676 ✓ | 9.1 |
| Hải Dương UBND TP. Hải Dương - 106 Đường Trần Hưng | North | 0.4237 | 0.4872 | +0.0635 ✓ | 18.3 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 0.5406 | 0.6443 | +0.1037 ✓ | 16.6 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái Tông - P. B | North | 0.5125 | 0.7260 | +0.2135 ✓ | 13.8 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - Quận 2 (N | South | 0.5877 | 0.7428 | +0.1551 ✓ | 4.8 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Môi Trường - | South | 0.5170 | 0.7461 | +0.2291 ✓ | 4.4 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung - TP. Phủ | North | 0.7311 | 0.7636 | +0.0325 ✓ | 12.9 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên (KK) | North | 0.5073 | 0.7961 | +0.2888 ✓ | 14.1 |
| Hà Nội Công viên Nhân Chính - Khuất Duy Tiến (KK) | North | 0.7550 | 0.8061 | +0.0511 ✓ | 8.9 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK) | North | 0.7812 | 0.8096 | +0.0284 ✓ | 14.1 |

## Regional Breakdown

| Region | D_K1 R² | D_K2 R² |
|--------|----------|----------|
| North | -0.3172 | 0.0432 |
| Central | -0.2625 | -0.0284 |
| South | -1.2165 | -1.0742 |

## Feature Importance (Config D_K2, top 20)

| Rank | Feature | Gain | RFSI? |
|------|---------|------|-------|
| 1 | PM25_nn_idw | 58871 | yes |
| 2 | PM25_nn_mean | 35298 | yes |
| 3 | slope_deg | 11496 |  |
| 4 | dist_nn4 | 7623 | yes |
| 5 | longitude | 6931 |  |
| 6 | aspect_cos | 4233 |  |
| 7 | elevation_m | 4063 |  |
| 8 | aspect_sin | 3457 |  |
| 9 | month_cos | 3238 |  |
| 10 | dist_nn5 | 3213 | yes |
| 11 | latitude | 2926 |  |
| 12 | PM25_nn1 | 2525 | yes |
| 13 | PM25_nn2 | 2473 | yes |
| 14 | dist_nn3 | 2122 | yes |
| 15 | dist_nn1 | 2000 | yes |
| 16 | day_of_year_cos | 1991 |  |
| 17 | month_sin | 1874 |  |
| 18 | wind_dir_cos_local | 1827 |  |
| 19 | dist_nn2 | 1712 | yes |
| 20 | WS_local | 1542 |  |

## Key Question: Does daily resolution fix LOSO?

- Hourly K2 (Exp04): LOSO R² = -0.1140
- Daily D_K2:        LOSO R² = -0.2521

**NO** — daily resolution does not fix LOSO. The problem is deeper: sparse network + Vietnam heterogeneity.

- Daily D_K1 (RFSI only): LOSO R² = -0.6524
- Oracle ceiling (Exp02): LOSO R² = 0.2252