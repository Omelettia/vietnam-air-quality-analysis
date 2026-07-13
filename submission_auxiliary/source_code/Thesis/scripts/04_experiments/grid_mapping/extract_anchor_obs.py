# -*- coding: utf-8 -*-
"""Extract anchor-station PM2.5 observations for the four map timestamps.

The 12 anchor IDs mirror DELTA_SIDS in ../exp_red_river_delta.py (kept inline
so the grid_mapping scripts stay standalone).
"""
import os
from pathlib import Path

import pandas as pd

MAP_DATA = os.environ.get("MAP_DATA", "D:/map_data")

ROOT = Path(__file__).resolve().parents[4]
IDS = [
    "28560877461938780203765592307",  # Hà Nội 556 Nguyễn Văn Cừ
    "28916504310234840885489983032",  # Bắc Ninh Thuận Thành
    "28916774462801800655608897080",  # Bắc Ninh Xuân Lâm
    "29196010501691076420299004774",  # Bắc Ninh Suối Hoa
    "29196021237696127337075448678",  # Bắc Ninh Cao Đức
    "29203727697074312726675247132",  # Thái Bình Thái Thọ
    "31388868531618872623864101418",  # Hải Dương
    "31388883344354363840031242796",  # Hà Nam
    "31390903576425084107499649578",  # Hà Nội ĐHBK
    "31390908889087377344742439468",  # Hà Nội Nhân Chính
    "31390921469766835629621918251",  # Hưng Yên
    "31390957404024291365397346858",  # Thái Bình TP
]
TS = ["2025-12-09 08:00:00", "2025-12-09 20:00:00",
      "2025-07-30 08:00:00", "2025-07-30 20:00:00"]

rows = []
for chunk in pd.read_csv(ROOT / "data" / "merged" / "unified_thesis.csv",
        usecols=["ts", "PM2.5", "stationId", "latitude", "longitude"],
        dtype={"stationId": str}, chunksize=2_000_000):
    c = chunk[chunk["ts"].isin(TS) & chunk["stationId"].isin(IDS)]
    if len(c):
        rows.append(c)
df = pd.concat(rows).dropna(subset=["PM2.5"])
df = df.rename(columns={"PM2.5": "pm25_obs", "latitude": "lat", "longitude": "lon"})
df["ts"] = pd.to_datetime(df["ts"])
os.makedirs(MAP_DATA + "/maps", exist_ok=True)
df[["ts", "stationId", "lat", "lon", "pm25_obs"]].to_csv(
    MAP_DATA + "/maps/anchor_obs.csv", index=False)
print("saved", len(df), "anchor obs rows")
