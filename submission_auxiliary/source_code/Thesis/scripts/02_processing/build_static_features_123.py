"""
Generate station static satellite feature CSVs for 109+ stations.

Reads the _123 directional climatology files and produces:
  - station_no2_features.csv  (NO2 directional climatology)
  - station_emission_features.csv  (NTL, LST anomaly, fire)
  - station_all_satellite_features.csv  (SO2, CO, HCHO, FAOD)

These replicate the station-level static feature files used by the thesis
diagnostic and Red River Delta modelling scripts.
"""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
META_DIR = os.path.join(REPO_DIR, "data", "stations", "metadata")

SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def load_dir(filename, value_col="mean"):
    """Load _123 directional file, compute per-station + direction mean."""
    path = os.path.join(META_DIR, filename)
    raw = pd.read_csv(path, dtype={"stationId": str})
    clim = raw.groupby(["stationId", "direction"])[value_col].mean().reset_index()
    return clim


def load_dir_seasonal(filename, value_col="mean"):
    """Load _123 directional file with season, compute per-station+direction+season mean."""
    path = os.path.join(META_DIR, filename)
    raw = pd.read_csv(path, dtype={"stationId": str})
    clim = raw.groupby(["stationId", "direction", "season"])[value_col].mean().reset_index()
    return clim


def get_sector_value(clim, sid, direction, value_col="mean"):
    row = clim[(clim["stationId"] == sid) & (clim["direction"] == direction)]
    return float(row[value_col].iloc[0]) if len(row) > 0 else np.nan


def build_sector_features(clim, sid, prefix, value_col="mean"):
    """Build {prefix}_clim_{dir} for 8 directions."""
    result = {}
    for d in SECTOR_NAMES:
        result[f"{prefix}_clim_{d}"] = get_sector_value(clim, sid, d, value_col)
    return result


def build_seasonal_sector_features(clim_seasonal, sid, prefix, value_col="mean"):
    """Build {prefix}_clim_{season}_{dir} for 4 seasons × 8 directions."""
    result = {}
    for season in SEASONS:
        for d in SECTOR_NAMES:
            row = clim_seasonal[
                (clim_seasonal["stationId"] == sid) &
                (clim_seasonal["direction"] == d) &
                (clim_seasonal["season"] == season)
            ]
            result[f"{prefix}_clim_{season}_{d}"] = float(row[value_col].iloc[0]) if len(row) > 0 else np.nan
    return result


# ══════════════════════════════════════════════════════════════════════
#  LOAD ALL DIRECTIONAL DATA
# ══════════════════════════════════════════════════════════════════════

print("Loading directional climatology files...")

no2_clim = load_dir("no2_directional_clim_123.csv")
so2_clim = load_dir("tropomi_so2_directional_123.csv")
co_clim = load_dir("tropomi_co_directional_123.csv")
hcho_clim = load_dir("tropomi_hcho_directional_123.csv")
ntl_clim = load_dir("nightlights_directional_123.csv")
lst_clim = load_dir("lst_anomaly_directional_123.csv", "lst_anomaly")
fire_clim = load_dir_seasonal("fire_counts_directional_123.csv")

# Also load NO2 monthly for seasonal breakdown
no2_monthly_path = os.path.join(META_DIR, "no2_directional_clim_123.csv")
no2_monthly = pd.read_csv(no2_monthly_path, dtype={"stationId": str})
season_map = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
no2_monthly["season"] = no2_monthly["month"].map(season_map)
no2_seasonal = no2_monthly.groupby(["stationId", "direction", "season"])["mean"].mean().reset_index()

all_sids = sorted(no2_clim["stationId"].unique())
print(f"Stations: {len(all_sids)}")

# ══════════════════════════════════════════════════════════════════════
#  1. station_no2_features.csv
# ══════════════════════════════════════════════════════════════════════
print("\nBuilding station_no2_features.csv...")

no2_rows = []
for sid in all_sids:
    row = {"stationId": sid}
    center = get_sector_value(no2_clim, sid, "C")
    row["no2_center"] = center

    for season in SEASONS:
        s_row = no2_seasonal[
            (no2_seasonal["stationId"] == sid) &
            (no2_seasonal["direction"] == "C") &
            (no2_seasonal["season"] == season)
        ]
        row[f"no2_center_{season}"] = float(s_row["mean"].iloc[0]) if len(s_row) > 0 else np.nan

    row.update(build_sector_features(no2_clim, sid, "no2"))
    row.update(build_seasonal_sector_features(no2_seasonal, sid, "no2"))

    sectors = [row.get(f"no2_clim_{d}", np.nan) for d in SECTOR_NAMES]
    valid_sectors = [v for v in sectors if not np.isnan(v)]
    if valid_sectors and not np.isnan(center) and center > 0:
        max_val = max(valid_sectors)
        min_val = min(valid_sectors)
        max_dir = SECTOR_NAMES[sectors.index(max_val)] if max_val == max_val else ""
        row["no2_max_sector"] = max_dir
        row["no2_contrast"] = max_val / min_val if min_val > 0 else np.nan
        row["no2_directionality"] = np.std(valid_sectors) / np.mean(valid_sectors) if np.mean(valid_sectors) > 0 else 0
    else:
        row["no2_max_sector"] = ""
        row["no2_contrast"] = np.nan
        row["no2_directionality"] = np.nan

    no2_rows.append(row)

no2_df = pd.DataFrame(no2_rows)
no2_path = os.path.join(META_DIR, "station_no2_features.csv")
no2_df.to_csv(no2_path, index=False, encoding="utf-8-sig")
print(f"  Saved: {no2_path} ({len(no2_df)} stations, {len(no2_df.columns)} cols)")

# ══════════════════════════════════════════════════════════════════════
#  2. station_emission_features.csv
# ══════════════════════════════════════════════════════════════════════
print("\nBuilding station_emission_features.csv...")

emit_rows = []
for sid in all_sids:
    row = {"stationId": sid}

    # NTL
    ntl_c = get_sector_value(ntl_clim, sid, "C")
    row["ntl_center"] = ntl_c
    ntl_sectors = {}
    for d in SECTOR_NAMES:
        v = get_sector_value(ntl_clim, sid, d)
        row[f"ntl_clim_{d}"] = v
        ntl_sectors[d] = v

    valid_ntl = [v for v in ntl_sectors.values() if not np.isnan(v)]
    if valid_ntl and not np.isnan(ntl_c):
        row["ntl_contrast"] = max(valid_ntl) / min(valid_ntl) if min(valid_ntl) > 0 else np.nan
        row["ntl_max_nearby"] = max(valid_ntl)
        row["ntl_anomaly"] = ntl_c - np.mean(valid_ntl) + ntl_c
    else:
        row["ntl_contrast"] = np.nan
        row["ntl_max_nearby"] = np.nan
        row["ntl_anomaly"] = np.nan

    # LST anomaly
    lst_c = get_sector_value(lst_clim, sid, "C", "lst_anomaly")
    row["lst_anom_center"] = lst_c
    lst_sectors = {}
    for d in SECTOR_NAMES:
        v = get_sector_value(lst_clim, sid, d, "lst_anomaly")
        row[f"lst_anom_clim_{d}"] = v
        lst_sectors[d] = v

    valid_lst = [v for v in lst_sectors.values() if not np.isnan(v)]
    row["lst_anom_max_nearby"] = max(valid_lst) if valid_lst else np.nan

    # LST seasonal center
    lst_seasonal_path = os.path.join(META_DIR, "lst_anomaly_directional_123.csv")
    lst_raw = pd.read_csv(lst_seasonal_path, dtype={"stationId": str})
    for season in SEASONS:
        s_row = lst_raw[
            (lst_raw["stationId"] == sid) &
            (lst_raw["direction"] == "C") &
            (lst_raw["season"] == season)
        ]
        row[f"lst_anom_{season}"] = float(s_row["lst_anomaly"].iloc[0]) if len(s_row) > 0 else np.nan

    # Fire
    fire_annual = 0.0
    fire_max = 0.0
    for season in SEASONS:
        fire_season_val = get_sector_value(
            fire_clim[(fire_clim["season"] == season)].reset_index(drop=True),
            sid, "C"
        )
        row[f"fire_{season}"] = fire_season_val if not np.isnan(fire_season_val) else 0.0
        fire_annual += row[f"fire_{season}"]

        # Max across directions for this season
        for d in SECTOR_NAMES:
            dv = fire_clim[
                (fire_clim["stationId"] == sid) &
                (fire_clim["direction"] == d) &
                (fire_clim["season"] == season)
            ]["mean"]
            if len(dv) > 0:
                fire_max = max(fire_max, float(dv.iloc[0]))

    row["fire_annual"] = fire_annual
    row["fire_max_sector"] = fire_max

    emit_rows.append(row)

emit_df = pd.DataFrame(emit_rows)
emit_path = os.path.join(META_DIR, "station_emission_features.csv")
emit_df.to_csv(emit_path, index=False, encoding="utf-8-sig")
print(f"  Saved: {emit_path} ({len(emit_df)} stations, {len(emit_df.columns)} cols)")

# ══════════════════════════════════════════════════════════════════════
#  3. station_all_satellite_features.csv
# ══════════════════════════════════════════════════════════════════════
print("\nBuilding station_all_satellite_features.csv...")

sat_rows = []
for sid in all_sids:
    row = {"stationId": sid}

    # SO2
    so2_c = get_sector_value(so2_clim, sid, "C")
    row["so2_center"] = so2_c
    so2_sectors = {}
    for d in SECTOR_NAMES:
        v = get_sector_value(so2_clim, sid, d)
        row[f"so2_clim_{d}"] = v
        so2_sectors[d] = v
    valid_so2 = [v for v in so2_sectors.values() if not np.isnan(v)]
    if valid_so2 and not np.isnan(so2_c) and so2_c != 0:
        row["so2_contrast"] = max(valid_so2) / min(valid_so2) if min(valid_so2) != 0 else np.nan
        row["so2_directionality"] = np.std(valid_so2) / abs(np.mean(valid_so2)) if np.mean(valid_so2) != 0 else 0
    else:
        row["so2_contrast"] = np.nan
        row["so2_directionality"] = np.nan

    # CO
    co_c = get_sector_value(co_clim, sid, "C")
    row["co_center"] = co_c
    for d in SECTOR_NAMES:
        row[f"co_clim_{d}"] = get_sector_value(co_clim, sid, d)
    valid_co = [row[f"co_clim_{d}"] for d in SECTOR_NAMES if not np.isnan(row.get(f"co_clim_{d}", np.nan))]
    if valid_co and not np.isnan(co_c) and min(valid_co) != 0:
        row["co_contrast"] = max(valid_co) / min(valid_co)
    else:
        row["co_contrast"] = np.nan

    # HCHO
    hcho_c = get_sector_value(hcho_clim, sid, "C")
    row["hcho_center"] = hcho_c
    for d in SECTOR_NAMES:
        row[f"hcho_clim_{d}"] = get_sector_value(hcho_clim, sid, d)
    valid_hcho = [row[f"hcho_clim_{d}"] for d in SECTOR_NAMES if not np.isnan(row.get(f"hcho_clim_{d}", np.nan))]
    if valid_hcho and not np.isnan(hcho_c) and min(valid_hcho) != 0:
        row["hcho_contrast"] = max(valid_hcho) / min(valid_hcho)
    else:
        row["hcho_contrast"] = np.nan

    # FAOD columns stay NaN by design: fine-mode products came from Himawari, not\n# GEE; downstream code falls back to a neutral 0.5 where fmf_center is missing.
    row["faod_center"] = np.nan
    row["fmf_center"] = np.nan
    row["ae_center"] = np.nan
    for d in SECTOR_NAMES:
        row[f"faod_clim_{d}"] = np.nan
    row["faod_contrast"] = np.nan
    row["faod_directionality"] = np.nan

    sat_rows.append(row)

sat_df = pd.DataFrame(sat_rows)
sat_path = os.path.join(META_DIR, "station_all_satellite_features.csv")
sat_df.to_csv(sat_path, index=False, encoding="utf-8-sig")
print(f"  Saved: {sat_path} ({len(sat_df)} stations, {len(sat_df.columns)} cols)")

print("\nDone!")
