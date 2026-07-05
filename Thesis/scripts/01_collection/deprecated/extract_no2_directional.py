"""
Extract directional NO2 climatology from Sentinel-5P via Google Earth Engine.

For each of 40 thesis stations, samples tropospheric NO2 column density at:
  - Center point (direction="C")
  - 8 directions (N, NE, E, SE, S, SW, W, NW) × 3 distances (5, 10, 20 km)
  = 25 points per station, 1000 points total

Computes monthly mean NO2 (Jan-Dec) across 2023-2024 with qa_value > 0.75.
Exports CSV to Google Drive.

Authentication:
  1. Install: pip install earthengine-api
  2. First run: earthengine authenticate
     (opens browser, follow prompts, paste token)
  3. Then run: python scripts/data/extract_no2_directional.py

If Python auth fails, use the JS version (extract_no2_directional.js)
in code.earthengine.google.com instead.

Output columns: stationId, direction, distance_km, month, no2_mean
Export filename on Drive: no2_directional_clim.csv
"""

import ee
import csv
import math
import os
import sys

ee.Initialize()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

STATION_CSV = os.path.join(REPO_DIR,
    "analysis/thesis_audit/station_selection_final.csv")

DIRECTIONS = {
    "N":  0,   "NE": 45,  "E":  90,  "SE": 135,
    "S":  180, "SW": 225, "W":  270, "NW": 315,
}
DISTANCES_KM = [5, 10, 20]
MONTHS = list(range(1, 13))
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
QA_THRESHOLD = 0.75

def offset_point(lat, lon, bearing_deg, dist_km):
    """Compute lat/lon offset by bearing and distance (Vincenty approx)."""
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

# Read stations
stations = []
with open(STATION_CSV, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        stations.append({
            "stationId": row["stationId"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        })

print(f"Loaded {len(stations)} stations")

# Build sampling points
sample_points = []
for stn in stations:
    sid = stn["stationId"]
    lat, lon = stn["lat"], stn["lon"]
    sample_points.append({
        "stationId": sid, "direction": "C", "distance_km": 0,
        "lat": lat, "lon": lon,
    })
    for dir_name, bearing in DIRECTIONS.items():
        for dist in DISTANCES_KM:
            plat, plon = offset_point(lat, lon, bearing, dist)
            sample_points.append({
                "stationId": sid, "direction": dir_name,
                "distance_km": dist, "lat": plat, "lon": plon,
            })

print(f"Total sampling points: {len(sample_points)}")

# Build GEE feature collection of points
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

# Process each month
no2_collection = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2") \
    .filterDate(START_DATE, END_DATE) \
    .filter(ee.Filter.gt("QUALITY_FLAG", QA_THRESHOLD))

all_results = []

for month in MONTHS:
    print(f"Processing month {month}...")

    monthly = no2_collection \
        .filter(ee.Filter.calendarRange(month, month, "month")) \
        .select("tropospheric_NO2_column_number_density")

    monthly_mean = monthly.mean()

    sampled = monthly_mean.sampleRegions(
        collection=points_fc,
        scale=1113.2,  # ~0.01° ≈ S5P native resolution
        geometries=False,
    )

    sampled_list = sampled.getInfo()["features"]

    for feat in sampled_list:
        props = feat["properties"]
        no2_val = props.get("tropospheric_NO2_column_number_density", None)
        all_results.append({
            "stationId": props["stationId"],
            "direction": props["direction"],
            "distance_km": props["distance_km"],
            "month": month,
            "no2_mean": no2_val,
        })

    print(f"  Month {month}: {len(sampled_list)} samples")

# Export locally
out_path = os.path.join(REPO_DIR, "data/stations/metadata/no2_directional_clim.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "stationId", "direction", "distance_km", "month", "no2_mean"])
    writer.writeheader()
    writer.writerows(all_results)

print(f"\nSaved: {out_path} ({len(all_results)} rows)")
print("Done!")
