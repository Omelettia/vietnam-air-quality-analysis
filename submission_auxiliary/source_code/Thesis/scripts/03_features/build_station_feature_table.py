"""Build a station summary table (QC'd PM2.5 statistics + coordinates).

Intentionally narrow: station-level PM2.5 mean/std/p90/hours after the shared
row-level QC mask, joined with coordinates — for geography/anchor diagnostics.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd


def _repo_root():
    p = os.path.abspath(os.path.dirname(__file__))
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "data", "merged")):
            return p
        p = os.path.dirname(p)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


ROOT = _repo_root()
sys.path.insert(0, os.path.join(ROOT, "Thesis", "scripts", "02_processing"))
from pm25_qc import pm25_quality_masks  # noqa: E402

DATASET = os.path.join(ROOT, "data", "merged", "unified_thesis.csv")
META = os.path.join(ROOT, "Thesis", "results", "01_stations", "station_selection_final.csv")
OUT = os.path.join(ROOT, "analysis", "thesis_experiments", "station_feature_table.csv")

meta = pd.read_csv(META, dtype={"stationId": str})
thesis40 = set(meta["stationId"])

df = pd.read_csv(DATASET, dtype={"stationId": str}, usecols=["stationId", "ts", "PM2.5"])
df = df[df["stationId"].isin(thesis40)].copy()
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)

# Stronger row-level QC mask (same as the models) before computing station means.
masks = pm25_quality_masks(df)
df.loc[masks.any(axis=1), "PM2.5"] = np.nan
df = df.dropna(subset=["PM2.5"])

g = df.groupby("stationId")["PM2.5"]
stats = pd.DataFrame({
    "actual_mean": g.mean(),
    "actual_std": g.std(),
    "actual_p90": g.quantile(0.90),
    "n_hours": g.size(),
}).reset_index().rename(columns={"stationId": "station_id"})

coords = meta.rename(columns={"stationId": "station_id"})[
    ["station_id", "station_name", "lat", "lon"]]
table = stats.merge(coords, on="station_id", how="left")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
table.to_csv(OUT, index=False)
print(f"Wrote {OUT}: {len(table)} stations, {len(table.columns)} cols")
print(f"  cleaned PM2.5 station means (shared QC mask): "
      f"min={table['actual_mean'].min():.1f}, max={table['actual_mean'].max():.1f}")
print(table[["station_id", "actual_mean", "n_hours"]].head(3).to_string(index=False))
