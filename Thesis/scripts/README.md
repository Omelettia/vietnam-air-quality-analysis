# Thesis Scripts — PM2.5 Prediction from Satellite Remote Sensing

## Data Sources & Collection Scripts

### 1. Ground-truth PM2.5 + Meteorology (Envisoft/TEDP)
- **Data**: `data/stations/historical_full/*.csv` (123 stations)
- **Metadata**: `data/stations/metadata/envisoft_station_map.csv`
- **Scripts**:
  - `01_collection/get_stations.py` — fetch station list from TEDP API
  - `01_collection/fetch_id.py` — Stage 1: fetch record IDs per station → `data/stations/historical_index/`
  - `01_collection/fetch_historical.py` — Stage 2: download hourly observations → `data/stations/historical_full/` (rate-limited, resumable, atomic per-station writes)
  - `01_collection/detail_parser.py` — shared parser for Vietnamese API keys

### 2. ERA5 / OpenMeteo Weather (PBLH, wind, temperature)
- **Data**: `data/stations/weather/weather_*.csv` (125 stations)
- **Script**: `01_collection/fetch_weather.py` — OpenMeteo Historical API

### 3. Himawari-8 AOD (JAXA P-Tree FTP)
- **Data**: `data/station_aod/L2/*.csv` (123 stations, hourly 5x5 grid)
- **Scripts**:
  - `01_collection/download_himawari.py` — download raw NetCDF from JAXA FTP
  - `01_collection/process_himawari.py` — NetCDF to GeoTIFF, clip to Vietnam
  - `01_collection/extract_himawari_stations.py` — sample 5x5 grid at stations

### 4. GPM IMERG Rainfall (NASA PMM)
- **Data**: `data/gpm/raw/YYYY/MM/DD/*.zip` (half-hourly GIS zips)
- **Extracted**: `data/gpm/station_gis_extracted_v2/*.csv` (125 stations)
- **Scripts**:
  - `01_collection/download_gpm.py` — FTPS download from NASA PMM (Final + NRT fallback)
  - `01_collection/extract_gpm_stations.py` — extract station pixels from GIS zips (parallel)

### 5. TROPOMI + MODIS LST Daily (GEE)
- **Data**: `data/gee_exports/last-*.zip` (123 stations, daily NO2/SO2/CO/HCHO/LST)
- **Script**: `01_collection/extract_tropomi_lst_daily.js` — GEE Code Editor

### 6. Directional Climatology (GEE) — 8 compass directions x 3 distances
- **Data**: `data/stations/metadata/*_directional_123.csv` (7 products, 123 stations)
- **Scripts**:
  - `01_collection/gee_directional_all.js` — GEE Code Editor (7 export tasks)
  - `01_collection/_stations_123.js` — station coordinates helper

### 7. GHAP/ACAG PM2.5 Climatology (GEE)
- **Data**: `data/gee_exports/pm25-*.zip` (monthly/annual/daily)
- **Script**: `01_collection/extract_ghap_pm25.js` — GEE Code Editor

### 8. DEM / Topography
- **Data**: `data/dem/output_*.tif` (3 regional GeoTIFFs), `data/dem/station_dem_features.csv`
- **Script**: *Extracted in QGIS from SRTM 30m. Feature extraction in `02_processing/thesis_pipeline.py` Step 5.*

### 9. Building Density (Google Open Buildings)
- **Data**: `data/stations/metadata/station_building_density.csv` (123 stations)
- **Script**: `archive/scripts/extract_building_density.py`

### 10. MODIS MAIAC AOD (GEE) — supplementary
- **Script**: `01_collection/extract_maiac_aod.js` — GEE Code Editor

## Processing Pipeline

| Step | Script | Input | Output |
|------|--------|-------|--------|
| Unified build | `02_processing/build_unified.py` | PM2.5 + satellite + weather + static sources | `unified_thesis.csv` |
| PM2.5 QC | `02_processing/pm25_qc.py` | merged rows | row-level PM2.5 quality masks |
| QC validation | `02_processing/validate_pm25_qc_effect.py` | merged data + masks | QC consistency report |
| Data profile | `02_processing/data_profile.py` | merged data | dataset profile for thesis |
| Station feature table | `03_features/build_station_feature_table.py` | merged data + station metadata | station summaries for diagnostics |

## Experiment Scripts

| Experiment | Script | Purpose |
|------------|--------|---------|
| AOD-PM2.5 diagnostic | `03_features/aod_pm25_correlation_paper.py` | AOD signal, sparsity, and physical decoupling |
| Within-station ceiling | `03_features/within_station_predictability.py` | Temporal predictability when station identity is known |
| CTM/global products | `04_experiments/exp_satellite_products.py` | GEOS-CF and MERRA-2 baseline failures |
| Red River Delta final model | `04_experiments/exp_red_river_delta.py` | Regional XGBoost + RFSI/wind LOSO and external validation |
