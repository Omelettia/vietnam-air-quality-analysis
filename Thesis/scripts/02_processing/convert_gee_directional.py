"""Convert GEE wide-format directional CSVs to long format for MoE script."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
GEE_DIR = os.path.join(REPO_DIR, "gee_directional_123")
META_DIR = os.path.join(REPO_DIR, "data", "stations", "metadata")

def convert_tropomi_monthly(filename, value_prefix, out_name):
    """Convert wide monthly (prefix_m01..m12) to long (stationId, direction, distance_km, year, month, mean)."""
    df = pd.read_csv(os.path.join(GEE_DIR, filename), dtype={"stationId": str})
    df = df.drop(columns=["system:index", ".geo"], errors="ignore")

    month_cols = [f"{value_prefix}_m{m:02d}" for m in range(1, 13)]
    existing = [c for c in month_cols if c in df.columns]

    rows = []
    for _, r in df.iterrows():
        for mc in existing:
            m = int(mc.split("_m")[1])
            val = r[mc]
            if pd.notna(val):
                rows.append({
                    "stationId": r["stationId"],
                    "direction": r["direction"],
                    "distance_km": r["distance_km"],
                    "year": 2023,
                    "month": m,
                    "mean": val,
                })
    out = pd.DataFrame(rows)
    out_path = os.path.join(META_DIR, out_name)
    out.to_csv(out_path, index=False)
    print(f"  {out_name}: {len(out)} rows, {out['stationId'].nunique()} stations")
    return out

def convert_lst_seasonal(filename):
    """Convert wide seasonal LST to long format with anomaly."""
    df = pd.read_csv(os.path.join(GEE_DIR, filename), dtype={"stationId": str})
    df = df.drop(columns=["system:index", ".geo"], errors="ignore")

    regional_day = {}
    regional_night = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        dc = f"lstDay_{season}"
        nc = f"lstNight_{season}"
        if dc in df.columns:
            regional_day[season] = df[dc].mean()
        if nc in df.columns:
            regional_night[season] = df[nc].mean()

    rows = []
    for _, r in df.iterrows():
        for season in ["DJF", "MAM", "JJA", "SON"]:
            dc = f"lstDay_{season}"
            nc = f"lstNight_{season}"
            d_val = r.get(dc)
            n_val = r.get(nc)
            lst_mean = np.nanmean([v for v in [d_val, n_val] if pd.notna(v)]) if pd.notna(d_val) or pd.notna(n_val) else np.nan
            lst_anom = (lst_mean - np.mean([regional_day.get(season, 0), regional_night.get(season, 0)])) if pd.notna(lst_mean) else np.nan
            if pd.notna(lst_mean):
                rows.append({
                    "stationId": r["stationId"],
                    "direction": r["direction"],
                    "distance_km": r["distance_km"],
                    "season": season,
                    "lst_mean": round(lst_mean, 4),
                    "lst_anomaly": round(lst_anom, 4),
                })
    out = pd.DataFrame(rows)
    out_path = os.path.join(META_DIR, "lst_anomaly_directional_123.csv")
    out.to_csv(out_path, index=False)
    print(f"  lst_anomaly_directional_123.csv: {len(out)} rows, {out['stationId'].nunique()} stations")

def convert_ntl(filename):
    """Convert NTL (already simple, just rename column)."""
    df = pd.read_csv(os.path.join(GEE_DIR, filename), dtype={"stationId": str})
    df = df.drop(columns=["system:index", ".geo"], errors="ignore")
    df = df.rename(columns={"ntl_mean": "mean"})
    out_path = os.path.join(META_DIR, "nightlights_directional_123.csv")
    df.to_csv(out_path, index=False)
    print(f"  nightlights_directional_123.csv: {len(df)} rows, {df['stationId'].nunique()} stations")

def convert_fire_seasonal(filename):
    """Convert wide seasonal fire to long format."""
    df = pd.read_csv(os.path.join(GEE_DIR, filename), dtype={"stationId": str})
    df = df.drop(columns=["system:index", ".geo"], errors="ignore")

    rows = []
    for _, r in df.iterrows():
        for season in ["DJF", "MAM", "JJA", "SON"]:
            col = f"fire_{season}"
            val = r.get(col, 0)
            rows.append({
                "stationId": r["stationId"],
                "direction": r["direction"],
                "distance_km": r["distance_km"],
                "season": season,
                "mean": val if pd.notna(val) else 0,
            })
    out = pd.DataFrame(rows)
    out_path = os.path.join(META_DIR, "fire_counts_directional_123.csv")
    out.to_csv(out_path, index=False)
    print(f"  fire_counts_directional_123.csv: {len(out)} rows, {out['stationId'].nunique()} stations")

print("Converting GEE directional exports to MoE-compatible format...")
print()

convert_tropomi_monthly("no2_directional_123.csv", "no2", "no2_directional_clim_123.csv")
convert_tropomi_monthly("so2_directional_123.csv", "so2", "tropomi_so2_directional_123.csv")
convert_tropomi_monthly("co_directional_123.csv", "co", "tropomi_co_directional_123.csv")
convert_tropomi_monthly("hcho_directional_123.csv", "hcho", "tropomi_hcho_directional_123.csv")
convert_lst_seasonal("lst_directional_123.csv")
convert_ntl("ntl_directional_123.csv")
convert_fire_seasonal("fire_directional_123.csv")

print("\nDone — files saved to data/stations/metadata/")
