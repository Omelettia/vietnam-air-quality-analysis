# Mã nguồn đồ án tốt nghiệp

**Đề tài:** Ước tính nồng độ bụi mịn PM2.5 mặt đất từ dữ liệu vệ tinh và khí tượng sử dụng học máy
**Tác giả:** Nguyễn Tài Khoa — MSSV 20215066
**GVHD:** TS. Trần Nguyên Ngọc
**Đơn vị:** Trường Công nghệ Thông tin và Truyền thông, Đại học Bách khoa Hà Nội

Gói này chứa mã nguồn do tác giả viết để thực hiện đồ án: thu thập dữ liệu, xử lý
và kiểm soát chất lượng, xây dựng đặc trưng, và các thí nghiệm mô hình. Đây là phần
minh chứng cho công việc đã thực hiện; dữ liệu thô và kết quả không kèm theo do dung
lượng lớn (xem mục Dữ liệu).

## Cấu trúc gói

```
README.md
requirements.txt
Thesis/scripts/
  01_collection/    # thu thập dữ liệu (trạm, vệ tinh, khí tượng, mưa)
  02_processing/    # hợp nhất + kiểm soát chất lượng
  03_features/      # xây dựng đặc trưng
  04_experiments/   # thí nghiệm mô hình
```

Các script tham chiếu lẫn nhau theo đường dẫn tương đối từ thư mục gốc của gói
(ví dụ `04_experiments` dùng `02_processing/pm25_qc.py`), nên hãy giữ nguyên cấu trúc
thư mục `Thesis/scripts/` khi giải nén và chạy từ thư mục gốc của gói.

## Cài đặt

```bash
conda create -n airquality python=3.10
conda activate airquality
pip install -r requirements.txt
```

## Quy trình (chạy theo thứ tự)

### 01_collection — Thu thập dữ liệu
- `get_stations.py`, `fetch_id.py` — danh sách trạm và mã bản ghi từ API TEDP.
- `fetch_historical.py` — tải PM2.5/khí tượng theo giờ (giới hạn tốc độ, có resume,
  ghi từng trạm); dùng `detail_parser.py`.
- `fetch_weather.py` — khí tượng tái phân tích (Open-Meteo).
- `download_himawari.py`, `process_himawari.py`, `extract_himawari_stations.py` — AOD Himawari.
- `download_gpm.py`, `extract_gpm_stations.py` — mưa GPM IMERG.
- `extract_building_density.py` — mật độ xây dựng.
- Các tệp `*.js` — trích xuất trên Google Earth Engine (MAIAC AOD, TROPOMI, MODIS LST,
  GEOS-CF/MERRA-2, GHAP, đặc trưng theo hướng).

### 02_processing — Hợp nhất và kiểm soát chất lượng
- `pm25_qc.py` — bộ lọc chất lượng PM2.5 ở cấp từng dòng (dùng chung cho mọi thí nghiệm).
- `build_unified.py` — hợp nhất PM2.5 + Himawari AOD + GEE/TROPOMI + MODIS + khí tượng + mưa + DEM và bối cảnh tĩnh → `unified_thesis.csv` (183 cột). Chính sách khí tượng: chỉ dùng nguồn tái phân tích ERA5/Open-Meteo cho các cột khí tượng của mô hình — huấn luyện, đánh giá độc lập và lưới bản đồ dùng chung một nguồn.
- `convert_embassy_data.py`, `convert_gee_directional.py`, `build_static_features_123.py` — chuẩn hóa dữ liệu Đại sứ quán và dựng các bảng bối cảnh tĩnh theo hướng.
- `data_profile.py`, `diagnose_pm25_qc.py`, `validate_pm25_qc_effect.py` — hồ sơ dữ liệu và chẩn đoán QC.

### 03_features — Xây dựng đặc trưng
- `regional_feature_pipeline.py` — nguồn duy nhất cho các đặc trưng không phụ thuộc fold (anomaly/rolling GEE–MODIS, bối cảnh tĩnh, tương tác vật lý, ràng buộc đơn điệu); được `build_unified.py` gọi khi dựng bảng và được mọi thí nghiệm vùng import. RFSI không ghi sẵn mà tính lại trong từng lượt LOSO để chống rò rỉ.
- `aod_pm25_correlation_paper.py` — chẩn đoán tương quan AOD–PM2.5 (đầu vào `unified_aod_filled.csv` là bảng chuẩn bị sẵn từ giai đoạn khảo sát AOD, kèm trong repo dữ liệu đầy đủ).
- `build_station_feature_table.py` — bảng tóm tắt PM2.5 theo trạm sau QC (mean/std/p90/số giờ + tọa độ) cho chẩn đoán địa lý/trạm neo.

### 04_experiments — Thí nghiệm mô hình
- `exp_random_sample_kfold.py` — đánh giá chia ngẫu nhiên theo mẫu (KFold) trên 40 trạm.
- `exp_national_loso_diagnostic.py` — ước tính toàn quốc theo phương án bỏ một trạm ra.
- `exp_red_river_delta.py` — **mô hình vùng được lựa chọn** (XGBoost + RFSI
  có xét hướng gió); LOSO trên 12 trạm vùng và đánh giá tại 42 trạm ngoài tập
  nghiên cứu (41 LCS + Đại sứ quán Hoa Kỳ), đặc trưng trích tại đúng tọa độ trạm.
- `exp_satellite_products.py` — đánh giá sản phẩm PM2.5 toàn cầu (GEOS-CF, MERRA-2, GHAP).
- `grid_mapping/` — các script tạo bản đồ lưới 0,02° vùng đồng bằng sông Hồng.

## Dữ liệu

Dữ liệu thô và bảng hợp nhất không kèm theo do dung lượng lớn. Nguồn dữ liệu: PM2.5
trạm KK/LCS từ hệ thống quan trắc tự động (Envisoft/CEM, TEDP), khí tượng tái phân tích
từ Open-Meteo, AOD từ Himawari và MODIS MAIAC, khí vết từ Sentinel-5P/TROPOMI (qua
Google Earth Engine), mưa từ GPM IMERG, và PM2.5 trạm Đại sứ quán Hoa Kỳ từ chương
trình quan trắc của Bộ Ngoại giao Hoa Kỳ. Các script trong `01_collection/` mô tả cách
thu thập từng nguồn.

Ngoài dữ liệu thô, các thí nghiệm đọc một số bảng metadata đã chuẩn bị sẵn trong repo
đầy đủ: `Thesis/results/01_stations/station_selection_final.csv` (danh sách 40 trạm KK),
`feature_list.csv` (bộ đặc trưng của chẩn đoán toàn quốc), `station_selection_lcs.csv`
(danh sách LCS đạt kiểm tra), và `data/stations/metadata/*.csv` (bối cảnh tĩnh theo trạm).
Nhóm script bản đồ (`grid_mapping/`) dùng ổ dữ liệu làm việc lớn; vị trí mặc định
`D:/map_data` có thể đổi qua biến môi trường `MAP_DATA`.
