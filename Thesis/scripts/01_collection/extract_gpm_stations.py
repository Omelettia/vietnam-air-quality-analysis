"""
Extract station-level precipitation from downloaded GPM IMERG GIS zip files.

For each station, extracts the nearest 0.1° pixel value from IMERG GeoTIFFs
stored inside .zip files. Handles both Final (9 TIF bands) and Late/NRT
(6 TIF bands). Converts UTC timestamps to UTC+7 (Vietnam time).

Input:
  - GPM zip files: data/gpm/raw/YYYY/MM/DD/*.zip
  - Station metadata: data/stations/metadata/envisoft_station_map.csv

Output:
  - Per-station CSVs: data/gpm/station_gis_extracted_v2/{stationId}.csv
    Columns: stationId, datetime, precipitationCal, precipitationUncal,
             IRprecipitation, MWprecipitation, randomError, ...

Usage:
  python extract_gpm_stations.py [--input data/gpm/raw] [--output data/gpm/station_gis_extracted_v2]

Adapted from: github.com/sfatew/Air_Quality/blob/main/GIS/extract_stations_gis.py
"""
import os
import sys
import re
import zipfile
import argparse
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio required. Install with: pip install rasterio")
    sys.exit(1)

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

UTC_OFFSET = timedelta(hours=7)

FINAL_BANDS = [
    "precipitationCal", "precipitationUncal", "IRprecipitation",
    "MWprecipitation", "randomError", "HQprecipitation",
    "IRkalmanFilterWeight", "probabilityLiquidPrecipitation",
    "precipitationQualityIndex"
]
LATE_BANDS = [
    "precipitationCal", "precipitationUncal", "IRprecipitation",
    "HQprecipitation", "IRkalmanFilterWeight",
    "probabilityLiquidPrecipitation"
]


def load_stations(meta_path):
    df = pd.read_csv(meta_path, dtype={"stationId": str})
    stations = []
    for _, row in df.iterrows():
        stations.append({
            "stationId": str(row["stationId"]),
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
        })
    return stations


def parse_timestamp(filename):
    """Extract datetime from IMERG filename pattern."""
    m = re.search(r"(\d{8})-S(\d{6})", filename)
    if not m:
        return None
    date_str, time_str = m.group(1), m.group(2)
    utc_dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
    return utc_dt + UTC_OFFSET


def extract_zip(zip_path, stations):
    """Extract station values from all TIFs in a single zip file."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_names:
                return []

            band_names = FINAL_BANDS if len(tif_names) >= 9 else LATE_BANDS
            dt = parse_timestamp(os.path.basename(zip_path))
            if dt is None:
                return []

            band_data = {}
            for tif_name in tif_names:
                tif_bytes = zf.read(tif_name)
                with rasterio.open(rasterio.io.MemoryFile(tif_bytes).open()) as src:
                    data = src.read(1)
                    transform = src.transform
                    band_data[tif_name] = (data, transform)

            sorted_tifs = sorted(band_data.keys())
            results = []
            for stn in stations:
                row = {
                    "stationId": stn["stationId"],
                    "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Final" if len(tif_names) >= 9 else "Late",
                }
                for i, tif_name in enumerate(sorted_tifs):
                    if i >= len(band_names):
                        break
                    data, transform = band_data[tif_name]
                    col_idx, row_idx = ~transform * (stn["lon"], stn["lat"])
                    col_idx, row_idx = int(col_idx), int(row_idx)
                    if 0 <= row_idx < data.shape[0] and 0 <= col_idx < data.shape[1]:
                        val = float(data[row_idx, col_idx])
                        row[band_names[i]] = val if val != -9999.9 else np.nan
                    else:
                        row[band_names[i]] = np.nan
                results.append(row)
            return results
    except Exception as e:
        print(f"  Error processing {zip_path}: {e}")
        return []


def collect_zips(input_dir):
    """Recursively find all .zip files."""
    zips = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".zip"):
                zips.append(os.path.join(root, f))
    return sorted(zips)


def main():
    parser = argparse.ArgumentParser(description="Extract GPM station precipitation")
    parser.add_argument("--input", default="data/gpm/raw", help="GPM zip directory")
    parser.add_argument("--output", default="data/gpm/station_gis_extracted_v2",
                        help="Output directory for per-station CSVs")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    meta_path = os.path.join(repo_dir, "data", "stations", "metadata", "envisoft_station_map.csv")

    if not os.path.exists(meta_path):
        print(f"ERROR: Station metadata not found at {meta_path}")
        sys.exit(1)

    stations = load_stations(meta_path)
    print(f"Stations: {len(stations)}")

    input_dir = os.path.join(repo_dir, args.input)
    output_dir = os.path.join(repo_dir, args.output)
    os.makedirs(output_dir, exist_ok=True)

    zips = collect_zips(input_dir)
    print(f"ZIP files found: {len(zips)}")
    if not zips:
        print("No zip files found. Run download_gpm.py first.")
        return

    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_zip, z, stations): z for z in zips}
        done = 0
        for future in as_completed(futures):
            done += 1
            rows = future.result()
            all_results.extend(rows)
            if done % 500 == 0:
                print(f"  Processed {done}/{len(zips)} zips ({len(all_results)} rows)")

    print(f"\nTotal rows: {len(all_results)}")
    if not all_results:
        print("No data extracted.")
        return

    df = pd.DataFrame(all_results)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values(["stationId", "datetime"], inplace=True)

    # Deduplicate: keep Final over Late for same station+datetime
    source_priority = {"Final": 0, "Late": 1}
    df["_priority"] = df["source"].map(source_priority)
    df = df.sort_values(["stationId", "datetime", "_priority"])
    df = df.drop_duplicates(subset=["stationId", "datetime"], keep="first")
    df.drop(columns=["_priority"], inplace=True)

    for sid, grp in df.groupby("stationId"):
        out_path = os.path.join(output_dir, f"{sid}.csv")
        grp.to_csv(out_path, index=False)

    print(f"Saved per-station CSVs to {output_dir} ({df['stationId'].nunique()} stations)")


if __name__ == "__main__":
    main()
