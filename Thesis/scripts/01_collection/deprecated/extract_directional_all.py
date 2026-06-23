"""
Extract directional climatology for ALL 123 stations via Google Earth Engine.

Products extracted (center + 8 directions × 3 distances = 25 points/station):
  1. TROPOMI NO2  — monthly mean (Jan-Dec), 2023-2024
  2. TROPOMI SO2  — monthly mean (Jan-Dec), 2023-2024
  3. TROPOMI CO   — monthly mean (Jan-Dec), 2023-2024
  4. TROPOMI HCHO — monthly mean (Jan-Dec), 2023-2024
  5. MODIS LST    — seasonal mean (DJF/MAM/JJA/SON), 2023-2024
  6. VIIRS NTL    — annual mean, 2023
  7. MODIS Fire   — seasonal count, 2023-2024

Usage:
  pip install earthengine-api
  earthengine authenticate          # first time only
  python extract_directional_all.py               # all products
  python extract_directional_all.py --product so2  # single product

Output: data/stations/metadata/<product>_directional_123.csv
"""

import ee
import csv
import math
import os
import sys
import io
import argparse
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser()
parser.add_argument("--product", default="all",
                    choices=["all", "no2", "so2", "co", "hcho", "lst", "ntl", "fire"],
                    help="Which product to extract (default: all)")
args = parser.parse_args()

ee.Initialize()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
META_DIR = os.path.join(REPO_DIR, "data", "stations", "metadata")
os.makedirs(META_DIR, exist_ok=True)

STATION_CSV = os.path.join(META_DIR, "envisoft_station_map.csv")

DIRECTIONS = {
    "N": 0, "NE": 45, "E": 90, "SE": 135,
    "S": 180, "SW": 225, "W": 270, "NW": 315,
}
DISTANCES_KM = [5, 10, 20]
MONTHS = list(range(1, 13))
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"


def offset_point(lat, lon, bearing_deg, dist_km):
    R = 6371.0
    d = dist_km / R
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) +
                     math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(lat1),
                              math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


# ── Load stations ──
import pandas as pd
stations_df = pd.read_csv(STATION_CSV, dtype={"stationId": str})
stations = []
for _, row in stations_df.iterrows():
    stations.append({
        "stationId": row["stationId"],
        "name": row["stationName"],
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"]),
    })
print(f"Loaded {len(stations)} stations")

# ── Build sampling points ──
sample_points = []
for stn in stations:
    sid = stn["stationId"]
    lat, lon = stn["lat"], stn["lon"]
    sample_points.append({
        "stationId": sid, "name": stn["name"],
        "direction": "C", "distance_km": 0, "lat": lat, "lon": lon,
    })
    for dir_name, bearing in DIRECTIONS.items():
        for dist in DISTANCES_KM:
            plat, plon = offset_point(lat, lon, bearing, dist)
            sample_points.append({
                "stationId": sid, "name": stn["name"],
                "direction": dir_name, "distance_km": dist,
                "lat": plat, "lon": plon,
            })

n_pts = len(sample_points)
print(f"Total sampling points: {n_pts} ({len(stations)} stations × 25 points)")

# ── Build GEE FeatureCollection ──
# GEE has a limit of ~5000 features per sampleRegions call.
# 123 stations × 25 = 3075, so we're fine in a single batch.
features = []
for i, pt in enumerate(sample_points):
    geom = ee.Geometry.Point([pt["lon"], pt["lat"]])
    props = {
        "point_id": i,
        "stationId": pt["stationId"],
        "direction": pt["direction"],
        "distance_km": pt["distance_km"],
    }
    features.append(ee.Feature(geom, props))

points_fc = ee.FeatureCollection(features)
print("Built GEE FeatureCollection")

# ── Lookup table for point metadata ──
pt_lookup = {i: pt for i, pt in enumerate(sample_points)}


def save_csv(rows, filename, fieldnames):
    out_path = os.path.join(META_DIR, filename)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path} ({len(rows)} rows)")


def sample_monthly_tropomi(collection_id, band, qa_band, qa_threshold, scale):
    """Extract monthly climatology for a TROPOMI product."""
    col = ee.ImageCollection(collection_id) \
        .filterDate(START_DATE, END_DATE)
    if qa_band:
        col = col.filter(ee.Filter.gt(qa_band, qa_threshold))

    all_results = []
    for month in MONTHS:
        t0 = time.time()
        monthly = col.filter(
            ee.Filter.calendarRange(month, month, "month")
        ).select(band)
        monthly_mean = monthly.mean()

        sampled = monthly_mean.sampleRegions(
            collection=points_fc, scale=scale, geometries=False,
        )
        sampled_list = sampled.getInfo()["features"]

        for feat in sampled_list:
            props = feat["properties"]
            all_results.append({
                "stationId": props["stationId"],
                "name": pt_lookup[props["point_id"]]["name"],
                "direction": props["direction"],
                "distance_km": props["distance_km"],
                "month": month,
                "mean": props.get(band, None),
            })
        elapsed = time.time() - t0
        print(f"    Month {month:2d}: {len(sampled_list)} samples ({elapsed:.1f}s)")

    return all_results


# ══════════════════════════════════════════════════════════════════════
#  PRODUCT EXTRACTORS
# ══════════════════════════════════════════════════════════════════════

def extract_no2():
    print("\n" + "=" * 60)
    print("TROPOMI NO2 — monthly climatology")
    print("=" * 60)
    rows = sample_monthly_tropomi(
        "COPERNICUS/S5P/OFFL/L3_NO2",
        "tropospheric_NO2_column_number_density",
        "QUALITY_FLAG", 0.75,
        scale=1113.2,
    )
    fields = ["stationId", "name", "direction", "distance_km", "month", "mean"]
    save_csv(rows, "no2_directional_123.csv", fields)


def extract_so2():
    print("\n" + "=" * 60)
    print("TROPOMI SO2 — monthly climatology")
    print("=" * 60)
    rows = sample_monthly_tropomi(
        "COPERNICUS/S5P/OFFL/L3_SO2",
        "SO2_column_number_density",
        "QUALITY_FLAG", 0.50,
        scale=1113.2,
    )
    fields = ["stationId", "name", "direction", "distance_km", "month", "mean"]
    save_csv(rows, "tropomi_so2_directional_123.csv", fields)


def extract_co():
    print("\n" + "=" * 60)
    print("TROPOMI CO — monthly climatology")
    print("=" * 60)
    rows = sample_monthly_tropomi(
        "COPERNICUS/S5P/OFFL/L3_CO",
        "CO_column_number_density",
        "QUALITY_FLAG", 0.50,
        scale=1113.2,
    )
    fields = ["stationId", "name", "direction", "distance_km", "month", "mean"]
    save_csv(rows, "tropomi_co_directional_123.csv", fields)


def extract_hcho():
    print("\n" + "=" * 60)
    print("TROPOMI HCHO — monthly climatology")
    print("=" * 60)
    rows = sample_monthly_tropomi(
        "COPERNICUS/S5P/OFFL/L3_HCHO",
        "tropospheric_HCHO_column_number_density",
        "QUALITY_FLAG", 0.50,
        scale=1113.2,
    )
    fields = ["stationId", "name", "direction", "distance_km", "month", "mean"]
    save_csv(rows, "tropomi_hcho_directional_123.csv", fields)


def extract_lst():
    print("\n" + "=" * 60)
    print("MODIS LST — seasonal day/night climatology")
    print("=" * 60)
    lst_day = ee.ImageCollection("MODIS/061/MOD11A1") \
        .filterDate(START_DATE, END_DATE) \
        .select("LST_Day_1km")
    lst_night = ee.ImageCollection("MODIS/061/MOD11A1") \
        .filterDate(START_DATE, END_DATE) \
        .select("LST_Night_1km")

    all_results = []
    for season_name, season_months in SEASONS.items():
        t0 = time.time()
        filt = ee.Filter.inList("system:index",
            lst_day.aggregate_array("system:index"))

        day_seasonal = lst_day.filter(
            ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in season_months])
        )
        night_seasonal = lst_night.filter(
            ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in season_months])
        )

        # MODIS LST is in Kelvin × 0.02 scale factor
        day_mean = day_seasonal.mean().multiply(0.02).subtract(273.15)
        night_mean = night_seasonal.mean().multiply(0.02).subtract(273.15)

        # Compute regional (Vietnam-wide) mean for anomaly
        vietnam = ee.Geometry.Rectangle([102, 8, 110, 24])
        day_regional = day_mean.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=vietnam,
            scale=1000, maxPixels=1e9
        ).getInfo().get("LST_Day_1km", 0)
        night_regional = night_mean.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=vietnam,
            scale=1000, maxPixels=1e9
        ).getInfo().get("LST_Night_1km", 0)

        sampled_day = day_mean.sampleRegions(
            collection=points_fc, scale=1000, geometries=False,
        )
        sampled_night = night_mean.sampleRegions(
            collection=points_fc, scale=1000, geometries=False,
        )

        day_vals = {f["properties"]["point_id"]: f["properties"].get("LST_Day_1km")
                    for f in sampled_day.getInfo()["features"]}
        night_vals = {f["properties"]["point_id"]: f["properties"].get("LST_Night_1km")
                      for f in sampled_night.getInfo()["features"]}

        for i, pt in enumerate(sample_points):
            d_val = day_vals.get(i)
            n_val = night_vals.get(i)
            all_results.append({
                "stationId": pt["stationId"],
                "direction": pt["direction"],
                "distance_km": pt["distance_km"],
                "season": season_name,
                "lst_day_mean": d_val,
                "lst_night_mean": n_val,
                "lst_day_anomaly": (d_val - day_regional) if d_val is not None else None,
                "lst_night_anomaly": (n_val - night_regional) if n_val is not None else None,
            })
        elapsed = time.time() - t0
        print(f"    {season_name}: day_regional={day_regional:.2f}°C, night_regional={night_regional:.2f}°C ({elapsed:.1f}s)")

    fields = ["stationId", "direction", "distance_km", "season",
              "lst_day_mean", "lst_night_mean", "lst_day_anomaly", "lst_night_anomaly"]
    save_csv(all_results, "lst_anomaly_directional_123.csv", fields)


def extract_ntl():
    print("\n" + "=" * 60)
    print("VIIRS Nightlights — annual mean")
    print("=" * 60)
    t0 = time.time()
    ntl = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") \
        .filterDate("2023-01-01", "2023-12-31") \
        .select("avg_rad")
    ntl_mean = ntl.mean()

    sampled = ntl_mean.sampleRegions(
        collection=points_fc, scale=463.83, geometries=False,
    )
    sampled_list = sampled.getInfo()["features"]

    all_results = []
    for feat in sampled_list:
        props = feat["properties"]
        all_results.append({
            "stationId": props["stationId"],
            "direction": props["direction"],
            "distance_km": props["distance_km"],
            "mean": props.get("avg_rad", None),
        })
    elapsed = time.time() - t0
    print(f"    {len(sampled_list)} samples ({elapsed:.1f}s)")

    fields = ["stationId", "direction", "distance_km", "mean"]
    save_csv(all_results, "nightlights_directional_123.csv", fields)


def extract_fire():
    print("\n" + "=" * 60)
    print("MODIS Fire — seasonal active fire counts")
    print("=" * 60)
    fires = ee.ImageCollection("MODIS/061/MCD64A1") \
        .filterDate(START_DATE, END_DATE) \
        .select("BurnDate")

    all_results = []
    for season_name, season_months in SEASONS.items():
        t0 = time.time()
        seasonal = fires.filter(
            ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in season_months])
        )
        # Count non-zero burn dates as fire occurrence
        fire_count = seasonal.map(lambda img: img.gt(0).selfMask()).sum()

        sampled = fire_count.sampleRegions(
            collection=points_fc, scale=500, geometries=False,
        )
        sampled_list = sampled.getInfo()["features"]

        for feat in sampled_list:
            props = feat["properties"]
            all_results.append({
                "stationId": props["stationId"],
                "direction": props["direction"],
                "distance_km": props["distance_km"],
                "season": season_name,
                "mean": props.get("BurnDate", 0),
            })
        elapsed = time.time() - t0
        print(f"    {season_name}: {len(sampled_list)} samples ({elapsed:.1f}s)")

    fields = ["stationId", "direction", "distance_km", "season", "mean"]
    save_csv(all_results, "fire_counts_directional_123.csv", fields)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

PRODUCTS = {
    "no2": extract_no2,
    "so2": extract_so2,
    "co": extract_co,
    "hcho": extract_hcho,
    "lst": extract_lst,
    "ntl": extract_ntl,
    "fire": extract_fire,
}

t_start = time.time()
if args.product == "all":
    for name, func in PRODUCTS.items():
        try:
            func()
        except Exception as e:
            print(f"\n  ERROR extracting {name}: {e}")
            print(f"  Skipping {name}, continuing with next product...\n")
else:
    PRODUCTS[args.product]()

elapsed = time.time() - t_start
print(f"\nDone — total time: {elapsed:.0f}s")
