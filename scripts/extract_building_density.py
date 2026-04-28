"""
Extract building density features from Google Open Buildings data.

For each station, count buildings and total building area within 1km and 3km
radius.  Processes ~28GB of CSV in chunks to stay within memory limits.

Input:  OpenBuildingData/*.csv  (8 files, ~28GB)
Output: data/stations/metadata/station_building_density.csv
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None,
                    help="Directory containing OpenBuildingData/ (default: REPO_DIR)")
parser.add_argument("--meta", default=None,
                    help="Path to envisoft_station_map.csv")
args = parser.parse_args()

DATA_DIR = args.data_dir or REPO_DIR
OUT_DIR = os.path.join(REPO_DIR, "data", "stations", "metadata")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load station metadata ───────────────────────────────────────────────────

meta_path = args.meta or os.path.join(DATA_DIR, "data/stations/metadata/envisoft_station_map.csv")
meta = pd.read_csv(meta_path, dtype={"stationId": str})
stations = meta[["stationId", "stationName", "latitude", "longitude"]].copy()
n_stations = len(stations)

stn_lat = stations["latitude"].values
stn_lon = stations["longitude"].values

print(f"Loaded {n_stations} stations")
print(f"  Lat range: {stn_lat.min():.2f} – {stn_lat.max():.2f}")
print(f"  Lon range: {stn_lon.min():.2f} – {stn_lon.max():.2f}")

# ─── Accumulators ─────────────────────────────────────────────────────────────

count_1km = np.zeros(n_stations, dtype=np.int64)
area_1km = np.zeros(n_stations, dtype=np.float64)
count_3km = np.zeros(n_stations, dtype=np.int64)
area_3km = np.zeros(n_stations, dtype=np.float64)

# ─── Haversine (vectorized, returns km) ───────────────────────────────────────

EARTH_R = 6371.0

def haversine_vec(lat1, lon1, lat2, lon2):
    """Haversine distance in km.  All inputs in degrees, broadcasting-safe."""
    rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
    rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

# ─── Bounding box for quick pre-filter ────────────────────────────────────────
# 3km ≈ 0.027° lat, up to ~0.035° lon at equator.  Use 0.03° for safety.

BOX_DEG = 0.03

# Vietnam rough bounding box for coarse pre-filter on each chunk
VN_LAT_MIN, VN_LAT_MAX = 8.0, 23.5
VN_LON_MIN, VN_LON_MAX = 102.0, 110.0

# ─── Process Open Buildings CSVs ─────────────────────────────────────────────

ob_pattern = os.path.join(DATA_DIR, "OpenBuildingData", "*.csv")
ob_files = sorted(glob.glob(ob_pattern))
if not ob_files:
    raise FileNotFoundError(f"No CSVs found at {ob_pattern}")

print(f"\nFound {len(ob_files)} Open Buildings files")
CHUNK_SIZE = 500_000
t0 = time.time()
total_rows = 0
total_matched = 0

for fpath in ob_files:
    fname = os.path.basename(fpath)
    file_rows = 0
    file_matched = 0
    ft0 = time.time()

    for chunk in pd.read_csv(fpath, usecols=["latitude", "longitude", "area_in_meters"],
                             chunksize=CHUNK_SIZE, dtype={"latitude": np.float64,
                                                          "longitude": np.float64,
                                                          "area_in_meters": np.float64}):
        file_rows += len(chunk)

        # Coarse Vietnam bounding box filter
        mask_vn = ((chunk["latitude"] >= VN_LAT_MIN) & (chunk["latitude"] <= VN_LAT_MAX) &
                   (chunk["longitude"] >= VN_LON_MIN) & (chunk["longitude"] <= VN_LON_MAX))
        chunk = chunk.loc[mask_vn]
        if chunk.empty:
            continue

        b_lat = chunk["latitude"].values
        b_lon = chunk["longitude"].values
        b_area = chunk["area_in_meters"].values

        for i in range(n_stations):
            # Tight bounding box around this station (±0.03°)
            lat_lo = stn_lat[i] - BOX_DEG
            lat_hi = stn_lat[i] + BOX_DEG
            lon_lo = stn_lon[i] - BOX_DEG
            lon_hi = stn_lon[i] + BOX_DEG

            box_mask = ((b_lat >= lat_lo) & (b_lat <= lat_hi) &
                        (b_lon >= lon_lo) & (b_lon <= lon_hi))
            n_box = box_mask.sum()
            if n_box == 0:
                continue

            dist = haversine_vec(stn_lat[i], stn_lon[i],
                                 b_lat[box_mask], b_lon[box_mask])
            areas = b_area[box_mask]

            m1 = dist <= 1.0
            m3 = dist <= 3.0

            count_1km[i] += m1.sum()
            area_1km[i] += areas[m1].sum()
            count_3km[i] += m3.sum()
            area_3km[i] += areas[m3].sum()
            file_matched += m3.sum()

    elapsed = time.time() - ft0
    total_rows += file_rows
    total_matched += file_matched
    print(f"  {fname}: {file_rows:,} rows, {file_matched:,} matched, {elapsed:.0f}s")

elapsed_total = time.time() - t0
print(f"\nTotal: {total_rows:,} rows processed, {total_matched:,} building-station matches")
print(f"Time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

# ─── Build output ─────────────────────────────────────────────────────────────

result = stations.copy()
result["building_count_1km"] = count_1km
result["building_area_1km"] = np.round(area_1km, 1)
result["building_count_3km"] = count_3km
result["building_area_3km"] = np.round(area_3km, 1)

out_path = os.path.join(OUT_DIR, "station_building_density.csv")
result.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ─── Summary ──────────────────────────────────────────────────────────────────

print("\n── Summary (1km radius) ──")
print(f"  Count: min={count_1km.min()}, median={int(np.median(count_1km))}, "
      f"max={count_1km.max()}")
print(f"  Area:  min={area_1km.min():.0f}, median={np.median(area_1km):.0f}, "
      f"max={area_1km.max():.0f} m²")

print("\n── Summary (3km radius) ──")
print(f"  Count: min={count_3km.min()}, median={int(np.median(count_3km))}, "
      f"max={count_3km.max()}")
print(f"  Area:  min={area_3km.min():.0f}, median={np.median(area_3km):.0f}, "
      f"max={area_3km.max():.0f} m²")

# Spot-check known stations
print("\n── Spot checks ──")
for name_frag, expected in [("ĐHBK", "high"), ("Trà Vinh", "low"),
                            ("Hà Nội", "high"), ("Sóc Trăng", "low")]:
    matches = result[result["stationName"].str.contains(name_frag, na=False)]
    for _, row in matches.head(2).iterrows():
        print(f"  {row['stationName'][:60]}")
        print(f"    1km: {row['building_count_1km']:,} buildings, "
              f"{row['building_area_1km']:,.0f} m²")
        print(f"    3km: {row['building_count_3km']:,} buildings, "
              f"{row['building_area_3km']:,.0f} m²")
