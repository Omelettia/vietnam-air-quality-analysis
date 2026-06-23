"""Build the standalone station feature table from v4.

This is intentionally self-contained: it only needs the unified v4 dataset and the
40-station selection. It produces per-station summaries (mean/std/p90/n_hours after the
stronger PM2.5 QC mask) plus coordinates, which the kNN spatial prior in
`exp_diverse_knn_diagnostic.py` reads (station_id, lat, lon, actual_mean).

No dependency on any other experiment's outputs. Existing non-stat columns (if a prior
table exists) are preserved so downstream readers keep their schema.
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

V4 = os.path.join(ROOT, "data", "merged", "unified_thesis_v4.csv")
META = os.path.join(ROOT, "analysis", "thesis_audit", "station_selection_final.csv")
OUT = os.path.join(ROOT, "analysis", "experimental_shape_magnitude", "station_feature_table.csv")

meta = pd.read_csv(META, dtype={"stationId": str})
thesis40 = set(meta["stationId"])

df = pd.read_csv(V4, dtype={"stationId": str}, usecols=["stationId", "ts", "PM2.5"])
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

# Preserve any extra columns from an existing table (e.g. vestigial pred_* cols) so
# downstream readers that expect them don't break.
if os.path.exists(OUT):
    old = pd.read_csv(OUT, dtype={"station_id": str})
    extra = [c for c in old.columns if c not in table.columns and c != "station_id"]
    if extra:
        table = table.merge(old[["station_id", *extra]], on="station_id", how="left")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
table.to_csv(OUT, index=False)
print(f"Wrote {OUT}: {len(table)} stations, {len(table.columns)} cols")
print(f"  cleaned PM2.5 station means (v4 + stronger mask): "
      f"min={table['actual_mean'].min():.1f}, max={table['actual_mean'].max():.1f}")
print(table[["station_id", "actual_mean", "n_hours"]].head(3).to_string(index=False))
