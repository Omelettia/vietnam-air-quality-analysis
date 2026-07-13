# Red River Delta grid mapping scripts

This folder contains the scripts used to create the PM2.5 grid-map illustration
in the thesis defense version. The map is an inference grid, not a downscaled
satellite product.

## Workflow

1. `gee_map_export.js`
   Exports MODIS MAIAC AOD and TROPOMI trace-gas GeoTIFFs for the two map
   windows from Google Earth Engine.

2. `convert_gpm_imerg_grid.py`
   Converts raw half-hourly GPM IMERG GIS zips into multiband rain GeoTIFFs:
   `D:/map_data/gpm/gpm_rain_dec.tif` and `D:/map_data/gpm/gpm_rain_jul.tif`.

3. `fetch_met_grid.py`
   Fetches hourly ERA5/Open-Meteo meteorology on a 0.1-degree support grid for
   the map windows.

4. `build_grid_features.py`
   Builds `D:/map_data/grid/grid_unified_cols.csv`, one row per 0.02-degree grid
   cell and timestamp. Each cell center is sampled from the natural resolution of
   each input source. No global downscaling or forced common-resolution raster is
   applied.

5. `train_predict_maps.py`
   Retrains the final `delta_rfsi_wind` model on all 12 Red River Delta KK
   anchor stations and predicts PM2.5 on the grid.

6. `extract_anchor_obs.py`
   Extracts observed PM2.5 at anchor stations for overlay on the map.

7. `make_maps.py`
   Renders `defense_assets/fig_pm25_maps.png`, the four-panel map used in the
   thesis PDF and defense slides. The panels share one absolute PM2.5 color
   scale, with light contours and selected place labels added for readability.

## Interpretation

The grid spacing is 0.02 degrees, about 2.2 km in the Red River Delta. This is
the spacing of the prediction/display grid. Inputs remain multi-resolution:
GPM rain is 0.1 degrees, some satellite products are finer or coarser, and RFSI
features are computed from station anchors rather than raster pixels.

The four panels intentionally use a shared color scale. Summer panels therefore
look less contrasted when PM2.5 is low; this preserves absolute comparison
against the winter panels instead of stretching each panel independently.
