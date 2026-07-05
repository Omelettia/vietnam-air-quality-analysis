"""
Build static MODIS AOD features from GEE directional climatology.

Reads:  data/stations/metadata/modis_aod_directional.csv
        (columns: stationId, direction, distance_km, year, month, mean)
        data/stations/metadata/no2_directional_clim.csv   (for shortId->longId mapping)
        analysis/thesis_audit/station_selection_final.csv  (long numeric IDs)

Outputs: data/stations/metadata/station_modis_aod_features.csv
         One row per station with static MODIS AOD features (keyed by long numeric ID).

Features computed:
  - maod_center:          annual mean MODIS AOD at station location
  - maod_clim_{N,NE,...}: mean AOD per sector (averaged across distances + months)
  - maod_contrast:        max(sector) / (min(sector) + 0.01)
  - maod_directionality:  std(sector means) / mean(sector means)
  - maod_max_nearby:      max AOD in any directional point
  - maod_DJF/MAM/JJA/SON: seasonal center AOD

Usage: python scripts/features/build_modis_aod_features.py [--data-dir PATH]
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

META_DIR = os.path.join(DATA_DIR, "data", "stations", "metadata")
INPUT_CSV = os.path.join(META_DIR, "modis_aod_directional.csv")
STATION_CSV = os.path.join(DATA_DIR,
    "analysis/thesis_audit/station_selection_final.csv")
NO2_CSV = os.path.join(META_DIR, "no2_directional_clim.csv")
OUTPUT_CSV = os.path.join(META_DIR, "station_modis_aod_features.csv")
MAPPING_AUDIT_CSV = os.path.join(DATA_DIR,
    "Thesis/results/06_data_quality/modis_station_id_mapping_audit.csv")

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
MIN_MATCH_SCORE = 3
SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def norm(s):
    return unicodedata.normalize("NFKD", str(s)).encode(
        "ascii", "ignore").decode().lower()


def tokenize(s):
    return set(norm(s).replace("-", " ").replace(",", " ").split())


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("BUILD MODIS AOD FEATURES FROM DIRECTIONAL CLIMATOLOGY")
print("=" * 70)

# ── shortId -> longId mapping via NO2 CSV name column ──
meta = pd.read_csv(STATION_CSV, dtype={"stationId": str}, encoding="utf-8-sig")
print(f"Station metadata: {len(meta)} stations")

no2_df = pd.read_csv(NO2_CSV, dtype={"stationId": str})
if "name" not in no2_df.columns:
    sys.exit("ERROR: NO2 CSV missing 'name' column — cannot build ID mapping")
no2_names = no2_df.groupby("stationId")["name"].first()
print(f"NO2 clim for ID mapping: {len(no2_names)} short IDs")

id_map = {}
mapping_audit = []
low_score_matches = []
for short_id, no2_name in no2_names.items():
    no2_tokens = tokenize(no2_name)
    best_score, best_long_id, best_meta_name = -1, None, ""
    for _, row in meta.iterrows():
        meta_tokens = tokenize(row["station_name"])
        score = len(no2_tokens & meta_tokens)
        if score > best_score:
            best_score = score
            best_long_id = row["stationId"]
            best_meta_name = row["station_name"]
    status = "ok" if best_score >= MIN_MATCH_SCORE else "low_score"
    mapping_audit.append({
        "short_id": short_id,
        "source_name": no2_name,
        "matched_stationId": best_long_id,
        "matched_station_name": best_meta_name,
        "token_overlap_score": best_score,
        "status": status,
    })
    if status == "low_score":
        low_score_matches.append((short_id, no2_name, best_long_id, best_score))
        continue
    id_map[short_id] = best_long_id

os.makedirs(os.path.dirname(MAPPING_AUDIT_CSV), exist_ok=True)
pd.DataFrame(mapping_audit).to_csv(
    MAPPING_AUDIT_CSV, index=False, encoding="utf-8-sig")

if low_score_matches:
    print(f"Saved mapping audit: {MAPPING_AUDIT_CSV}")
    print("ERROR: low-confidence MODIS station ID matches:")
    for short_id, no2_name, best_long_id, best_score in low_score_matches:
        print(f"  {short_id}: score={best_score}, best={best_long_id}, name={no2_name}")
    sys.exit("Fix/audit station ID mapping before continuing")

mapped_long = set(id_map.values())
if len(mapped_long) < len(id_map):
    dupes = [v for v in mapped_long if list(id_map.values()).count(v) > 1]
    sys.exit(f"ERROR: duplicate ID mappings: {dupes}")
print(f"Saved mapping audit: {MAPPING_AUDIT_CSV}")
print(f"Mapped {len(id_map)} short -> long IDs")

# ── Load MODIS AOD CSV ──
print(f"\n--- MODIS AOD: {INPUT_CSV} ---")
df = pd.read_csv(INPUT_CSV, dtype={"stationId": str})
if "mean" in df.columns and "maod_val" not in df.columns:
    df = df.rename(columns={"mean": "maod_val"})
df = df.dropna(subset=["maod_val"])
df["season"] = df["month"].map(SEASON_MAP)

modis_short_ids = set(df["stationId"].dropna().astype(str).unique())
missing_modis_ids = sorted(modis_short_ids - set(id_map))
if missing_modis_ids:
    print("ERROR: MODIS station IDs missing from audited mapping:")
    for sid in missing_modis_ids:
        print(f"  {sid}")
    sys.exit("Fix/audit MODIS station ID mapping before continuing")

df["stationId"] = df["stationId"].map(id_map)
df = df.dropna(subset=["stationId"])
print(f"  {len(df):,} rows, {df['stationId'].nunique()} stations")

# ── Compute features per station ──
station_ids = sorted(df["stationId"].unique())
print(f"\n--- Computing features for {len(station_ids)} stations ---")

rows = []
for sid in station_ids:
    sdf = df[df["stationId"] == sid]
    row = {"stationId": sid}

    # ── Center (annual) ──
    center = sdf[sdf["direction"] == "C"]["maod_val"]
    maod_center = float(center.mean()) if len(center) > 0 else np.nan
    row["maod_center"] = round(maod_center, 6) if not np.isnan(maod_center) else np.nan

    # ── Center (seasonal) ──
    for szn in SEASONS:
        c_szn = sdf[(sdf["direction"] == "C") & (sdf["season"] == szn)]["maod_val"]
        row[f"maod_{szn}"] = round(float(c_szn.mean()), 6) if len(c_szn) > 0 else np.nan

    # ── Sector means (annual, averaged across distances and months) ──
    sector_means = {}
    all_dir_vals = []
    for d in DIRECTIONS:
        vals = sdf[sdf["direction"] == d]["maod_val"]
        m = float(vals.mean()) if len(vals) > 0 else np.nan
        sector_means[d] = m
        row[f"maod_clim_{d}"] = round(m, 6) if not np.isnan(m) else np.nan
        if len(vals) > 0:
            all_dir_vals.extend(vals.tolist())

    # ── Derived features ──
    valid_sectors = {k: v for k, v in sector_means.items() if not np.isnan(v)}
    if len(valid_sectors) >= 4:
        sv = np.array(list(valid_sectors.values()))
        row["maod_contrast"] = round(float(sv.max() / (sv.min() + 0.01)), 4)
        row["maod_directionality"] = round(float(sv.std() / sv.mean()), 6) \
            if sv.mean() > 0 else np.nan
    else:
        row["maod_contrast"] = np.nan
        row["maod_directionality"] = np.nan

    row["maod_max_nearby"] = (round(float(max(all_dir_vals)), 6)
                              if all_dir_vals else np.nan)

    rows.append(row)

out_df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ── Summary ──
print(f"\nSaved: {OUTPUT_CSV}")
print(f"  {len(out_df)} stations, {len(out_df.columns)} columns")

print(f"\n  Feature summary:")
cols = [c for c in out_df.columns if c.startswith("maod_")]
print(f"    MODIS AOD: {len(cols)} features")

print(f"\n  Annual stats:")
print(f"    maod_center:          mean={out_df['maod_center'].mean():.6f}, "
      f"max={out_df['maod_center'].max():.6f}")
print(f"    maod_contrast:        mean={out_df['maod_contrast'].mean():.4f}")
print(f"    maod_directionality:  mean={out_df['maod_directionality'].mean():.6f}")
print(f"    maod_max_nearby:      mean={out_df['maod_max_nearby'].mean():.6f}")

print(f"\n  Seasonal center AOD:")
for szn in SEASONS:
    col = f"maod_{szn}"
    print(f"    {col}: mean={out_df[col].mean():.6f}")

print("\nDone!")
