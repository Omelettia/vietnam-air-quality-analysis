# Thesis Scripts — PM2.5 Prediction from Satellite Remote Sensing

## Data Sources & Collection Scripts

### 1. Ground-truth PM2.5 + Meteorology (Envisoft/TEDP)
- **Data**: `data/stations/historical_full_v2/*.csv` (123 stations)
- **Metadata**: `data/stations/metadata/envisoft_station_map.csv`
- **Scripts**:
  - `01_collection/get_stations.py` — fetch station list from TEDP API
  - `01_collection/fetch_id.py` — fetch record IDs per station
  - `01_collection/fetch_full_details.py` — download hourly observations (Stage 2)
  - `01_collection/rebuild_historical_v2.py` — parallel re-download with resume
  - `01_collection/detail_parser.py` — shared parser for Vietnamese API keys
  - `01_collection/crawler.py` — legacy Envisoft/OpenWeather crawler

### 2. ERA5 / OpenMeteo Weather (PBLH, wind, temperature)
- **Data**: `data/stations/weather/weather_*.csv` (125 stations)
- **Script**: `01_collection/fetch_weather.py` — OpenMeteo Historical API

### 3. Himawari-8 AOD (JAXA P-Tree FTP)
- **Data**: `data/station_aod_v3/L2/*.csv` (123 stations, hourly 5x5 grid)
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
| QC & audit | `02_processing/thesis_pipeline.py` | Raw station data | `unified_thesis_v1.csv` (40 stn) |
| Add RF/SSA | `02_processing/build_unified_v2.py` | v1 + Himawari L2 | `unified_thesis_v2.csv` (40 stn) |
| **123-station** | `02_processing/build_unified_v3.py` | All sources | `unified_thesis_v3.csv` (109 stn) |
| GEE format | `02_processing/convert_gee_directional.py` | GEE wide CSVs | `*_directional_123.csv` |
| Static features | `02_processing/build_static_features_123.py` | Directional CSVs | `station_*_features.csv` |

## Experiment Scripts

| Experiment | Script | Purpose |
|------------|--------|---------|
| Tier MoE | `analysis/exp_true_tier_moe_xgb.py` | Core model: MoE with tier grouping |
| Diverse streams | `analysis/exp_diverse_streams.py` | 5-stream ensemble |
| Full pipeline | `analysis/exp_himawari_noghap_diverse.py` | End-to-end: MoE+correct+diverse+kNN |
| Conformal | `04_experiments/conformal_trustmap.py` | Mondrian conformal prediction |
| External val | `04_experiments/exp_external_validation.py` | US Embassy/LCS validation |
| Journey | `04_experiments/experimental_journey/` | All intermediate experiments |
