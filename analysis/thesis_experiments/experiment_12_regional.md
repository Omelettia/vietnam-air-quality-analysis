# Experiment 12: Regional Clustering + Per-Cluster LOSO

**Date:** 2026-05-02 13:00
**Dataset:** unified_thesis_v2.csv — 727,635 rows, 40 stations
**XGBoost:** v3.2.0, n_estimators=500, max_depth=7, lr=0.05, device=cuda

## Part 1: Clustering Results

Clustering features (12): mean_AOT_outer_mean, mean_AOT_inner_mean, mean_SSA_inner_mean_clean, mean_SSA_grad_mag_clean, mean_SSA_local_vs_regional_clean, latitude, building_area_3km, elevation_m, mean_PBLH, mean_VC, rain_freq, mean_Temp

| Method | k | Silhouette | PM2.5 Coherence | Within-Cluster Std |
|--------|---|:---:|:---:|:---:|
| Hierarchical k=2 | 2 | 0.316 | 0.261 | 10.3 |
| Hierarchical k=3 | 3 | 0.338 | 0.385 | 8.6 |
| Hierarchical k=4 | 4 | 0.370 | 0.476 | 7.3 |
| Hierarchical k=5 | 5 | 0.385 | 0.496 | 7.0 |
| KMeans k=2 | 2 | 0.316 | 0.261 | 10.3 |
| KMeans k=3 | 3 | 0.338 | 0.385 | 8.6 |
| KMeans k=4 | 4 | 0.370 | 0.476 | 7.3 |
| **KMeans k=5** | 5 | 0.385 | **0.496** | 7.0 |

**Selected:** KM5 (highest PM2.5 coherence)

### Cluster Assignments (KM5)

**Cluster 0** (n=6, PM2.5 mean=17.8, std=6.6):

- Tây Ninh Thị xã Trảng Bàng (KK) (South, PM2.5=10.9)
- Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6, Tp. Sóc Tr (South, PM2.5=6.7)
- Bình Dương số 593 Đại lộ Bình Dương, P. Hiệp Thành (South, PM2.5=24.2)
- HCM Khu Liên cơ quan Bộ Tài Nguyên và Môi Trường - (South, PM2.5=21.3)
- HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - Quận 2 (N (South, PM2.5=22.6)
- Long An UBND Tp Tân An - 76 Hùng Vương - P.2 (KK) (South, PM2.5=21.2)

**Cluster 1** (n=6, PM2.5 mean=12.6, std=4.2):

- Trà Vinh xã Dân Thành, TX Duyên Hải (KK) (South, PM2.5=9.1)
- Trà Vinh xã Đông Hải, huyện Duyên Hải (KK) (South, PM2.5=5.7)
- Bình Định Khuôn viên Cây xanh gần cầu chui đường H (Central, PM2.5=18.5)
- Ninh Thuận Công viên (bến xe cũ) - Đ. Thống Nhất - (Central, PM2.5=15.5)
- Vũng Tàu Đ. Huyền Trân Công Chúa - Phường 8 - TP V (South, PM2.5=14.7)
- Bình Định huyện Tuy Phước (KK) (Central, PM2.5=12.1)

**Cluster 2** (n=11, PM2.5 mean=13.0, std=7.2):

- Đà Nẵng 41 đường Lê Duẩn (KK) (Central, PM2.5=13.2)
- Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK) (North, PM2.5=6.9)
- Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - Hạ L (North, PM2.5=6.6)
- Thái Bình xã Thái Thọ, huyện Thái Thụy (KK) (North, PM2.5=15.5)
- Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả, đườ (North, PM2.5=7.0)
- Quảng Ninh Gần KCN Cái Lân (KK) (North, PM2.5=7.6)
- Quảng Ninh Km11 - Minh Thành (KK) (North, PM2.5=9.5)
- Đà Nẵng Khuôn viên trường ĐH sư phạm Đà Nẵng (KK) (Central, PM2.5=23.6)
- Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC Đ. Hồ Xuân (Central, PM2.5=19.9)
- Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Quảng Ngãi ( (Central, PM2.5=27.4)
- Đà Nẵng Phạm Hùng (KK) (Unknown, PM2.5=6.2)

**Cluster 3** (n=15, PM2.5 mean=34.7, std=13.5):

- Phú Thọ đường Hùng Vương - Tp Việt Trì (KK) (North, PM2.5=27.1)
- Hà Nội 556 Nguyễn Văn Cừ (KK) (North, PM2.5=48.5)
- Quảng Ninh UBND TP Uông Bí (KK) (North, PM2.5=10.6)
- Bắc Ninh Khu liên cơ Thuận Thành - thị trấn Hồ (KK (North, PM2.5=27.9)
- Bắc Ninh UBND xã Xuân Lâm - Thuận Thành (KK) (North, PM2.5=52.2)
- Bắc Ninh TT Quan trắc - phường Suối Hoa - TP Bắc N (North, PM2.5=23.4)
- Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) (North, PM2.5=25.4)
- Quảng Ninh Nhuệ Hổ - Đông Triều (KK) (North, PM2.5=9.2)
- Hải Dương UBND TP. Hải Dương - 106 Đường Trần Hưng (North, PM2.5=37.6)
- Hà Nam Công Viên Nam Cao - P.Quang Trung - TP. Phủ (North, PM2.5=39.2)
- Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK) (North, PM2.5=46.8)
- Hà Nội Công viên Nhân Chính - Khuất Duy Tiến (KK) (North, PM2.5=36.9)
- Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên (KK) (North, PM2.5=43.2)
- Thái Bình Cầu Thái Bình - Đ. Trần Thái Tông - P. B (North, PM2.5=37.2)
- Thái Nguyên Sân vận động Gang thép - P Trung Thành (North, PM2.5=55.2)

**Cluster 4** (n=2, PM2.5 mean=14.5, std=3.6):

- Gia Lai KCN Trà Đa - Tp Pleiku (KK) (Central, PM2.5=10.9)
- Lâm Đồng Vườn hoa - đối diện THCS Lam Sơn - Phường (Central, PM2.5=18.1)

## Part 2: LOSO Results

| Config | Description | LOSO R² (mean) | LOSO R² (median) | LOSO MAE | Neg Stations |
|--------|-------------|:---:|:---:|:---:|:---:|
| B1 (ref) | Exp07 reference | -0.0197 | -0.0038 | 11.52 | 20 |
| V2 (ref) | Exp11: B1 + sat baseline | 0.0072 | 0.1035 | 11.24 | 16 |
| **C1** | Per-cluster LOSO (KM5) | **-0.1186** | 0.0169 | 11.68 | 18 |
| **C2** | All stations + cluster_id feature (KM5) | **0.0093** | 0.0820 | 11.26 | 16 |
| **C3** | All stations, no cluster (V2 reference) | **0.0072** | 0.1035 | 11.24 | 16 |

### Per-Cluster Breakdown

**C1** (Per-cluster LOSO (KM5)):

| Cluster | Stations | Mean R² | Mean MAE |
|:---:|:---:|:---:|:---:|
| 0 | 6 | -0.0582 | 8.63 |
| 1 | 6 | -0.2958 | 7.06 |
| 2 | 11 | -0.3150 | 10.23 |
| 3 | 15 | 0.1263 | 15.62 |
| 4 | 2 | -0.5248 | 13.08 |

**C2** (All stations + cluster_id feature (KM5)):

| Cluster | Stations | Mean R² | Mean MAE |
|:---:|:---:|:---:|:---:|
| 0 | 6 | 0.0416 | 8.36 |
| 1 | 6 | -0.1342 | 6.61 |
| 2 | 11 | -0.1279 | 9.46 |
| 3 | 15 | 0.1479 | 15.57 |
| 4 | 2 | 0.0576 | 11.57 |

**C3** (All stations, no cluster (V2 reference)):

| Cluster | Stations | Mean R² | Mean MAE |
|:---:|:---:|:---:|:---:|
| 0 | 6 | 0.0656 | 8.23 |
| 1 | 6 | -0.1590 | 6.57 |
| 2 | 11 | -0.1284 | 9.43 |
| 3 | 15 | 0.1439 | 15.57 |
| 4 | 2 | 0.0507 | 11.79 |


### Per-Station: C2

| Station | Region | Cluster | B1 R² | C2 R² | Δ vs B1 |
|---------|--------|:---:|:---:|:---:|:---:|
| Quảng Ninh Nhuệ Hổ - Đông Triều (KK) | North | 3 | -0.6345 | -1.7736 | -1.1391 |
| Tây Ninh Thị xã Trảng Bàng (KK) | South | 0 | -0.4196 | -1.2442 | -0.8246 |
| Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (K | North | 2 | -0.4785 | -0.8476 | -0.3691 |
| Đà Nẵng Phạm Hùng (KK) | Unknown | 2 | -1.3350 | -0.8213 | +0.5137 |
| Bình Định Khuôn viên Cây xanh gần cầu ch | Central | 1 | -0.1164 | -0.5310 | -0.4146 |
| Trà Vinh xã Đông Hải, huyện Duyên Hải (K | South | 1 | -1.5549 | -0.4013 | +1.1536 |
| Thái Bình xã Thái Thọ, huyện Thái Thụy ( | North | 2 | -0.1226 | -0.1390 | -0.0164 |
| Đà Nẵng 41 đường Lê Duẩn (KK) | Central | 2 | -0.3612 | -0.1283 | +0.2329 |
| Quảng Ninh UBND TP Uông Bí (KK) | North | 3 | 0.0614 | -0.1197 | -0.1811 |
| Bắc Ninh Khu liên cơ Thuận Thành - thị t | North | 3 | -0.3357 | -0.1119 | +0.2238 |
| Trà Vinh xã Dân Thành, TX Duyên Hải (KK) | South | 1 | -0.0526 | -0.1014 | -0.0488 |
| Quảng Ninh Gần KCN Cái Lân (KK) | North | 2 | -0.0728 | -0.0920 | -0.0192 |
| Gia Lai KCN Trà Đa - Tp Pleiku (KK) | Central | 4 | -0.0273 | -0.0652 | -0.0379 |
| Đà Nẵng Khuôn viên trường ĐH sư phạm Đà  | Central | 2 | 0.1215 | -0.0493 | -0.1708 |
| Bắc Ninh UBND xã Xuân Lâm - Thuận Thành  | North | 3 | -0.0285 | -0.0472 | -0.0187 |
| Quảng Ninh Trung tâm văn hóa thể thao Cẩ | North | 2 | -0.6991 | -0.0445 | +0.6546 |
| Sóc Trăng Sở TNMT - 16 Hùng Vương, P.6,  | South | 0 | -0.5035 | 0.0225 | +0.5260 |
| Bình Định huyện Tuy Phước (KK) | Central | 1 | 0.0197 | 0.0356 | +0.0159 |
| Ninh Thuận Công viên (bến xe cũ) - Đ. Th | Central | 1 | -0.1248 | 0.0704 | +0.1952 |
| Quảng Ninh Nhà máy tuyển than Nam Cầu Tr | North | 2 | 0.1467 | 0.0732 | -0.0735 |
| Quảng Ngãi UBND P. Nguyễn Nghiêm - TP Qu | Central | 2 | 0.1273 | 0.0909 | -0.0364 |
| Thái Nguyên Sân vận động Gang thép - P T | North | 3 | 0.2878 | 0.1085 | -0.1793 |
| Vũng Tàu Đ. Huyền Trân Công Chúa - Phườn | South | 1 | 0.0911 | 0.1224 | +0.0313 |
| Phú Thọ đường Hùng Vương - Tp Việt Trì ( | North | 3 | -0.4438 | 0.1580 | +0.6018 |
| Lâm Đồng Vườn hoa - đối diện THCS Lam Sơ | Central | 4 | -0.2252 | 0.1805 | +0.4057 |
| HCM Khu Liên cơ quan Bộ Tài Nguyên và Mô | South | 0 | 0.4356 | 0.1908 | -0.2448 |
| Bắc Ninh UBND xã Cao Đức - Gia Bình (KK) | North | 3 | -0.1860 | 0.2450 | +0.4310 |
| Quảng Ninh Km11 - Minh Thành (KK) | North | 2 | 0.3115 | 0.2686 | -0.0429 |
| Quảng Nam Tiếp giáp Đ. Hùng Vương - KDC  | Central | 2 | 0.1388 | 0.2828 | +0.1440 |
| Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưn | North | 3 | 0.4148 | 0.2950 | -0.1198 |
| Long An UBND Tp Tân An - 76 Hùng Vương - | South | 0 | 0.2623 | 0.3354 | +0.0731 |
| Bắc Ninh TT Quan trắc - phường Suối Hoa  | North | 3 | -0.2515 | 0.3491 | +0.6006 |
| Hà Nội Công viên Nhân Chính - Khuất Duy  | North | 3 | 0.7110 | 0.3594 | -0.3516 |
| Bình Dương số 593 Đại lộ Bình Dương, P.  | South | 0 | 0.3564 | 0.3612 | +0.0048 |
| Thái Bình Cầu Thái Bình - Đ. Trần Thái T | North | 3 | 0.5066 | 0.3623 | -0.1443 |
| Hải Dương UBND TP. Hải Dương - 106 Đường | North | 3 | 0.5904 | 0.4342 | -0.1562 |
| Hà Nội 556 Nguyễn Văn Cừ (KK) | North | 3 | 0.5897 | 0.5839 | -0.0058 |
| HCM Đ. Lê Hữu Kiều - P. Bình Trưng Tây - | South | 0 | 0.5127 | 0.5840 | +0.0713 |
| Hà Nam Công Viên Nam Cao - P.Quang Trung | North | 3 | 0.7393 | 0.6322 | -0.1071 |
| Hà Nội ĐHBK cổng Parabol đường Giải Phón | North | 3 | 0.7593 | 0.7434 | -0.0159 |

## Summary

- **C1**: LOSO R²=-0.1186 (median=0.0169), MAE=11.68, 18 negative stations
- **C2**: LOSO R²=0.0093 (median=0.0820), MAE=11.26, 16 negative stations
- **C3**: LOSO R²=0.0072 (median=0.1035), MAE=11.24, 16 negative stations

**Best (C2) vs B1:** +0.0290