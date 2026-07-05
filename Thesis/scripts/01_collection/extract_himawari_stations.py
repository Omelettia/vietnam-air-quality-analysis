"""
Extract Himawari AOD values at monitoring station locations from GeoTIFFs.

Usage: python extract_himawari_stations.py <input.tif>

Reads a Vietnam-clipped Himawari GeoTIFF, samples the pixel value at each
station coordinate, and appends to per-station CSV files. Automatically
detects L2 (6 bands) vs L3 (15 bands) from the raster.

Original location: data/himawari/raw_scripts/extract_station_aod.py
"""
import sys
import os
import pandas as pd
import rasterio
import numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python extract_himawari_stations.py <input.tif>")
    sys.exit(1)

aod_file = sys.argv[1]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIONS_FILE = PROJECT_ROOT / "data" / "stations" / "metadata" / "envisoft_station_map.csv"
BASE_OUTPUT_DIR = PROJECT_ROOT / "data" / "station_aod"

filename = os.path.basename(aod_file)
is_l3 = "L3" in filename
product_type = "L3" if is_l3 else "L2"
output_dir = BASE_OUTPUT_DIR / product_type
output_dir.mkdir(parents=True, exist_ok=True)

parts = filename.split("_")
timestamp = parts[4] + "_" + parts[5]

stations = pd.read_csv(STATIONS_FILE)

with rasterio.open(aod_file) as src:
    if src.count == 6:
        cols = ["timestamp", "AOT", "Uncertainty", "AE", "QA_flag", "SSA", "RF"]
    else:
        cols = ["timestamp", "AOT_Merged", "AOT_Pure", "AOT_Merged_uncertainty",
                "AOT_Pure_uncertainty", "AE_Merged", "AE_Pure", "QA_flag_Merged",
                "QA_flag_Pure", "AOT_L2_Mean", "AOT_L2_SDV", "AOT_L2_Num",
                "AE_L2_Mean", "AE_L2_SDV", "AE_L2_Num"]

    data_bands = [src.read(i + 1) for i in range(src.count)]

    for _, row in stations.iterrows():
        station_id = str(row["id"])
        lon, lat = row["longitude"], row["latitude"]
        station_file = str(output_dir / f"{station_id}.csv")

        try:
            rowcol = src.index(lon, lat)
            pixel_values = [timestamp]

            for band in data_bands:
                val = band[rowcol[0], rowcol[1]]
                pixel_values.append(float(val) if not np.isnan(val) else None)

            new_data = pd.DataFrame([pixel_values], columns=cols)

            if os.path.exists(station_file):
                df_old = pd.read_csv(station_file)
                df_merged = pd.concat([df_old, new_data], ignore_index=True)
                df_merged.drop_duplicates(subset=['timestamp'], keep='last',
                                          inplace=True)
                df_merged.to_csv(station_file, index=False)
            else:
                new_data.to_csv(station_file, index=False)

        except Exception as e:
            print(f"Error extracting station {station_id}: {e}")

print(f"Done extracting {product_type} for timestamp {timestamp}")
