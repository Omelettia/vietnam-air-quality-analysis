"""
Build static NO2 features from GEE directional climatology.

Reads:  data/stations/metadata/no2_directional_clim.csv
        (columns: stationId, name, direction, distance_km, year, month, mean)
        analysis/thesis_audit/station_selection_final.csv
        (maps long numeric stationId ↔ station_name with lat/lon)

Outputs: data/stations/metadata/station_no2_features.csv
         One row per station with static NO2 features (keyed by long numeric ID).

Features computed:
  - no2_center:          annual mean NO2 at station location
  - no2_clim_{N,NE,...}: mean NO2 per sector (averaged across distances + months)
  - no2_max_sector:      direction with highest mean NO2
  - no2_contrast:        max(sector) / min(sector)
  - no2_directionality:  std(sector means) / mean(sector means)
  - no2_clim_DJF_{N,...}: seasonal sector means (DJF, MAM, JJA, SON)
  - no2_center_DJF, ...: seasonal center means

Usage: python scripts/features/build_no2_features.py [--data-dir PATH]
"""

import argparse, os, sys, unicodedata
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = args.data_dir or REPO_DIR

INPUT_CSV = os.path.join(DATA_DIR,
    "data/stations/metadata/no2_directional_clim.csv")
STATION_CSV = os.path.join(DATA_DIR,
    "analysis/thesis_audit/station_selection_final.csv")
OUTPUT_CSV = os.path.join(DATA_DIR,
    "data/stations/metadata/station_no2_features.csv")

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
SEASONS = ["DJF", "MAM", "JJA", "SON"]

def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

def tokenize(s):
    return set(norm(s).replace("-", " ").replace(",", " ").split())

# =============================================================================
print("=" * 70)
print("BUILD NO2 FEATURES FROM DIRECTIONAL CLIMATOLOGY")
print("=" * 70)

df = pd.read_csv(INPUT_CSV, dtype={"stationId": str})
if "mean" in df.columns and "no2_mean" not in df.columns:
    df = df.rename(columns={"mean": "no2_mean"})
df = df.dropna(subset=["no2_mean"])
df["season"] = df["month"].map(SEASON_MAP)
print(f"Loaded NO2 clim: {len(df):,} rows, {df['stationId'].nunique()} stations")

# ── Map NO2 short IDs → long numeric IDs via name matching ──
meta = pd.read_csv(STATION_CSV, dtype={"stationId": str}, encoding="utf-8-sig")
print(f"Loaded station metadata: {len(meta)} stations")

no2_names = df.groupby("stationId")["name"].first() if "name" in df.columns else None
if no2_names is None:
    sys.exit("ERROR: NO2 CSV missing 'name' column — cannot map station IDs")

id_map = {}
for short_id, no2_name in no2_names.items():
    no2_tokens = tokenize(no2_name)
    best_score, best_long_id = -1, None
    for _, row in meta.iterrows():
        meta_tokens = tokenize(row["station_name"])
        score = len(no2_tokens & meta_tokens)
        if score > best_score:
            best_score = score
            best_long_id = row["stationId"]
    id_map[short_id] = best_long_id

mapped_long = set(id_map.values())
if len(mapped_long) < len(id_map):
    dupes = [v for v in mapped_long
             if list(id_map.values()).count(v) > 1]
    print(f"WARNING: duplicate long-ID matches: {dupes}")
    sys.exit("Fix mapping before continuing")

print(f"\nStation ID mapping ({len(id_map)} stations):")
for short_id, long_id in sorted(id_map.items()):
    meta_name = meta.loc[meta["stationId"] == long_id, "station_name"].iloc[0]
    no2_name = no2_names[short_id]
    print(f"  {short_id:20s} → {long_id[:12]}… | {no2_name} ↔ {meta_name[:50]}")

df["stationId"] = df["stationId"].map(id_map)
if df["stationId"].isna().any():
    print("ERROR: some NO2 stations could not be mapped")
    sys.exit(1)

station_ids = sorted(df["stationId"].unique())
rows = []

for sid in station_ids:
    sdf = df[df["stationId"] == sid]
    row = {"stationId": sid}

    # ── Center (annual) ──
    center = sdf[sdf["direction"] == "C"]["no2_mean"]
    row["no2_center"] = round(float(center.mean()), 6) if len(center) > 0 else np.nan

    # ── Center (seasonal) ──
    for szn in SEASONS:
        c_szn = sdf[(sdf["direction"] == "C") & (sdf["season"] == szn)]["no2_mean"]
        row[f"no2_center_{szn}"] = round(float(c_szn.mean()), 6) if len(c_szn) > 0 else np.nan

    # ── Sector means (annual, averaged across distances and months) ──
    sector_means = {}
    for d in DIRECTIONS:
        vals = sdf[(sdf["direction"] == d)]["no2_mean"]
        m = float(vals.mean()) if len(vals) > 0 else np.nan
        sector_means[d] = m
        row[f"no2_clim_{d}"] = round(m, 6) if not np.isnan(m) else np.nan

    # ── Sector means (seasonal) ──
    for szn in SEASONS:
        for d in DIRECTIONS:
            vals = sdf[(sdf["direction"] == d) & (sdf["season"] == szn)]["no2_mean"]
            m = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"no2_clim_{szn}_{d}"] = round(m, 6) if not np.isnan(m) else np.nan

    # ── Derived features ──
    valid_sectors = {k: v for k, v in sector_means.items() if not np.isnan(v)}
    if len(valid_sectors) >= 4:
        vals = np.array(list(valid_sectors.values()))
        row["no2_max_sector"] = max(valid_sectors, key=valid_sectors.get)
        row["no2_contrast"] = round(float(vals.max() / vals.min()), 4) \
            if vals.min() > 0 else np.nan
        row["no2_directionality"] = round(float(vals.std() / vals.mean()), 6) \
            if vals.mean() > 0 else np.nan
    else:
        row["no2_max_sector"] = ""
        row["no2_contrast"] = np.nan
        row["no2_directionality"] = np.nan

    rows.append(row)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\nSaved: {OUTPUT_CSV}")
print(f"  {len(out_df)} stations, {len(out_df.columns)} columns")

# Summary
print(f"\n  Annual features:")
print(f"    no2_center mean:          {out_df['no2_center'].mean():.6f}")
print(f"    no2_contrast mean:        {out_df['no2_contrast'].mean():.4f}")
print(f"    no2_directionality mean:  {out_df['no2_directionality'].mean():.6f}")

max_sectors = out_df["no2_max_sector"].value_counts()
print(f"\n  Max sector distribution:")
for s, c in max_sectors.items():
    print(f"    {s}: {c} stations")

print("\nDone!")
