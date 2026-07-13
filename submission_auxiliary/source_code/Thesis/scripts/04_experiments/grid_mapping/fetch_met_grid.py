# -*- coding: utf-8 -*-
"""Fetch Open-Meteo ERA5 archive hourly met on a 0.1-deg grid over the RRD box
for the two 3-day map windows. Save as parquet: D:/map_data/met/met_grid.csv"""
import requests, time
import pandas as pd
import os

MAP_DATA = os.environ.get("MAP_DATA", "D:/map_data")

OUT_DIR = MAP_DATA + "/met"
os.makedirs(OUT_DIR, exist_ok=True)

lats = [round(20.3 + 0.1 * i, 1) for i in range(11)]   # 20.3..21.3
lons = [round(105.5 + 0.1 * j, 1) for j in range(16)]  # 105.5..107.0
points = [(la, lo) for la in lats for lo in lons]       # 176 points

WINDOWS = [
    ("2025-12-06", "2025-12-09"),  # extra day for 12h/6h lags at 00 UTC
    ("2025-07-27", "2025-07-30"),
]
HOURLY = ("temperature_2m,relative_humidity_2m,surface_pressure,"
          "wind_speed_10m,wind_direction_10m,cloud_cover")

URL = "https://archive-api.open-meteo.com/v1/archive"
CHUNK = 40

frames = []
for (d0, d1) in WINDOWS:
    for k in range(0, len(points), CHUNK):
        batch = points[k:k + CHUNK]
        params = {
            "latitude": ",".join(str(p[0]) for p in batch),
            "longitude": ",".join(str(p[1]) for p in batch),
            "start_date": d0, "end_date": d1,
            "hourly": HOURLY, "timezone": "UTC",
            "wind_speed_unit": "ms",
        }
        for attempt in range(5):
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        js = r.json()
        if isinstance(js, dict):
            js = [js]
        for p, obj in zip(batch, js):
            h = obj["hourly"]
            df = pd.DataFrame(h)
            df["lat"] = p[0]; df["lon"] = p[1]
            frames.append(df)
        print(f"window {d0}: {min(k+CHUNK, len(points))}/{len(points)} points", flush=True)
        time.sleep(1.0)

met = pd.concat(frames, ignore_index=True)
met = met.rename(columns={"time": "ts_utc"})
met.to_csv(os.path.join(OUT_DIR, "met_grid.csv"), index=False)
print("SAVED", os.path.join(OUT_DIR, "met_grid.csv"), len(met), "rows,",
      met[['lat','lon']].drop_duplicates().shape[0], "points")
