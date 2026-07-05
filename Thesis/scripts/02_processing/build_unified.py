"""
Build unified_thesis.csv — final merged dataset for all stations.

Main processing choices:
  - No station-level exclusions.
  - Row-level QC via pm25_quality_masks (flatline, stuck-low, range checks).
  - Proper station_type label (LCS vs KK).
  - Relaxed PM2.5 coverage threshold (10%).

Merges: Envisoft hourly + OpenMeteo weather + Himawari AOD (L1+L2) + GPM rain + DEM.
Feature engineering: AOD spatial gradients, temporal lags, physics correction,
                     rain features, wind components, temporal encodings.

Usage: python Thesis/scripts/02_processing/build_unified.py
"""
import io, sys, os, warnings, glob, unicodedata, math, time, re
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(BASE)

QC_DIR = os.path.join(BASE, "Thesis", "scripts", "02_processing")
if QC_DIR not in sys.path:
    sys.path.insert(0, QC_DIR)
from pm25_qc import pm25_quality_masks

t0_total = time.time()

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

STUCK_TEMP_STATIONS = {
    "Hà Nam Công Viên Nam Cao - P.Quang Trung - TP. Phủ Lý (KK)",
    "Ninh Thuận Công viên (bến xe cũ) - Đ. Thống Nhất - P. Thanh Sơn - TP Phan Rang (KK)",
    "Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên (KK)",
}
QUEVO_NAN_WS = "Bắc Ninh UBND huyện Quế Võ, TT Phố Mới (KK)"

SENTINEL_VALUES = {-9999, -999, 9999, -9999.0, -999.0, 9999.0}
QC_RANGES = {
    "PM2.5": (0, 500), "PM10": (0, 1000), "Temperature": (-10, 50),
    "Humidity": (0, 100), "Pressure": (900, 1100),
    "Wind Speed": (0, 50), "Wind Direction": (0, 360),
    "NO2": (0, 500), "O3": (0, 500), "SO2": (0, 500), "CO": (0, 100),
}

MIN_PM25_COVERAGE = 0.10
GPM_IS_UTC = True

OUT_PATH = "data/merged/unified_thesis.csv"

# ══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════

def normalize(s):
    s = s.replace(":", "").replace("", "").replace("﻿", "").replace("﻿", "")
    s = " ".join(s.split())
    return unicodedata.normalize("NFC", s.strip())

def build_index(pattern, prefix_strip="", suffix_strip=".csv", space_char=" "):
    idx = {}
    for f in glob.glob(pattern):
        base = os.path.basename(f).replace(suffix_strip, "")
        if prefix_strip:
            base = base.replace(prefix_strip, "", 1)
        if space_char != " ":
            base = base.replace(space_char, " ")
        idx[normalize(base)] = f
    return idx

def fuzzy_get(idx, key):
    if key in idx:
        return idx[key]
    for k, v in idx.items():
        if key.startswith(k) or k.startswith(key):
            return v
    return None


# ══════════════════════════════════════════════════════════════════════
#  DATA INDEXES
# ══════════════════════════════════════════════════════════════════════

env_idx = {k: v for k, v in build_index("data/stations/historical_full/*.csv").items()
           if not v.endswith(".log")}
aod_idx = build_index("data/station_aod/L2/*.csv")
gpm_idx = build_index("data/gpm/station_gis_extracted_v2/*.csv", space_char="_")
wx_idx  = build_index("data/stations/weather/*.csv", prefix_strip="weather_")

metadata = pd.read_csv("data/stations/metadata/envisoft_station_map.csv",
                        dtype={"stationId": str})

print(f"Data indexes: env={len(env_idx)}, aod={len(aod_idx)}, gpm={len(gpm_idx)}, wx={len(wx_idx)}")

dem_lookup = {}
if os.path.exists("data/dem/station_dem_features.csv"):
    dem_feat = pd.read_csv("data/dem/station_dem_features.csv", dtype={"stationId": str})
    for _, r in dem_feat.iterrows():
        dem_lookup[str(r["stationId"])] = r
    print(f"DEM features: {len(dem_lookup)} stations")

L2_DIR = "data/himawari/L2"
PIXEL_POSITIONS = [
    "m2m2","m2m1","m2_0","m2p1","m2p2",
    "m1m2","m1m1","m1_0","m1p1","m1p2",
    "_0m2","_0m1","_0_0","_0p1","_0p2",
    "p1m2","p1m1","p1_0","p1p1","p1p2",
    "p2m2","p2m1","p2_0","p2p1","p2p2",
]
L2_SUMMARY = ["valid_count", "mean", "std", "center", "inner_mean", "outer_mean"]

l2_index = {}
if os.path.exists(L2_DIR):
    for f in os.listdir(L2_DIR):
        if not f.endswith(".csv"):
            continue
        try:
            peek = pd.read_csv(os.path.join(L2_DIR, f), usecols=["stationId"], nrows=1)
            sid = str(peek["stationId"].iloc[0])
            l2_index[sid] = os.path.join(L2_DIR, f)
        except Exception:
            pass
    print(f"L2 RF/SSA files: {len(l2_index)}")

AOT_PIXELS = [f"AOT_{p}" for p in PIXEL_POSITIONS]
AOT_SUMMARY = ["AOT_valid_count", "AOT_mean", "AOT_std", "AOT_center",
               "AOT_inner_count", "AOT_inner_mean", "AOT_outer_count", "AOT_outer_mean"]
AOT_META = ["Uncertainty", "AE", "QA_flag", "SSA", "RF"]


# ══════════════════════════════════════════════════════════════════════
#  LOADERS
# ══════════════════════════════════════════════════════════════════════

def load_envisoft(path, station_name):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].replace(list(SENTINEL_VALUES), np.nan)

    for col, (lo, hi) in QC_RANGES.items():
        if col in df.columns:
            df[col] = df[col].where(df[col].between(lo, hi))

    if station_name in STUCK_TEMP_STATIONS and "Temperature" in df.columns:
        df.loc[df["Temperature"] == 0.0, "Temperature"] = np.nan

    if QUEVO_NAN_WS in station_name and "Wind Speed" in df.columns:
        df["Wind Speed"] = np.nan

    for pc in ["PM2.5", "PM10"]:
        if pc in df.columns:
            df.loc[df[pc] < 0, pc] = np.nan

    df["_nn"] = df[num_cols].notna().sum(axis=1)
    df["ts_hour"] = df["ts"].dt.floor("h")
    df = df.sort_values(["ts_hour", "_nn"], ascending=[True, False])
    df = df.drop_duplicates("ts_hour", keep="first")
    df = df.drop(columns=["_nn"])

    keep = [c for c in ["PM2.5", "PM10", "Temperature", "Humidity", "Pressure",
                         "Wind Speed", "Wind Direction", "NO2", "O3", "SO2", "CO"]
            if c in df.columns]

    hourly = df.groupby("ts_hour")[keep].mean().reset_index()
    hourly = hourly.rename(columns={"ts_hour": "ts",
                                     "Wind Speed": "WS_local",
                                     "Wind Direction": "WD_local"})
    return hourly


def load_openmeteo(path):
    df = pd.read_csv(path)
    ts_str = str(df["Timestamp"].iloc[0])
    if "+00:00" in ts_str:
        df["ts"] = pd.to_datetime(df["Timestamp"], utc=True) + pd.Timedelta(hours=7)
        df["ts"] = df["ts"].dt.tz_localize(None)
    else:
        df["ts"] = pd.to_datetime(df["Timestamp"])
    df["ts"] = df["ts"].dt.floor("h")
    rename = {"Temperature": "Temperature_om", "Humidity": "Humidity_om",
              "Pressure": "Pressure_om", "Wind Speed": "WS_om",
              "Wind Direction": "WD_om", "PBLH": "PBLH", "Visibility": "Visibility"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["ts"] + [v for v in rename.values() if v in df.columns]
    return df[keep].drop_duplicates("ts", keep="first")


def load_aod(path):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["ts_hour"] = df["ts"].dt.floor("h")

    pixel_cols = [c for c in AOT_PIXELS if c in df.columns]
    summary_cols = [c for c in AOT_SUMMARY if c in df.columns]
    meta_cols = [c for c in AOT_META if c in df.columns]
    all_aod = pixel_cols + summary_cols + meta_cols

    agg_dict = {}
    for c in all_aod:
        if "count" in c.lower():
            agg_dict[c] = (c, "max")
        elif c == "QA_flag":
            agg_dict[c] = (c, "first")
        else:
            agg_dict[c] = (c, "mean")
    agg_dict["n_aod_obs"] = ("ts", "size")

    hourly = df.groupby("ts_hour").agg(**agg_dict).reset_index()
    hourly = hourly.rename(columns={"ts_hour": "ts"})
    if "AOT_center" in hourly.columns:
        hourly = hourly.rename(columns={"AOT_center": "AOT"})
    return hourly


def load_gpm(path):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"])
    if GPM_IS_UTC:
        df["ts"] = df["ts"] + pd.Timedelta(hours=7)
    df["ts_hour"] = df["ts"].dt.floor("h")
    hourly = df.groupby("ts_hour").agg(
        precip_mm=("total_accum_mm", "sum"),
        precip_rate=("total_rate_mmh", "mean"),
    ).reset_index().rename(columns={"ts_hour": "ts"})
    return hourly


def compute_gradients(df, band):
    north, south, west, east = [], [], [], []
    for p in PIXEL_POSITIONS:
        col = f"{band}_{p}"
        if col not in df.columns:
            continue
        row_idx = p[:2]
        col_idx = p[2:]
        if row_idx in ("m1", "m2"): north.append(col)
        if row_idx in ("p1", "p2"): south.append(col)
        if col_idx in ("m1", "m2"): west.append(col)
        if col_idx in ("p1", "p2"): east.append(col)

    if north and south:
        df[f"{band}_grad_ns"] = (df[south].mean(axis=1) - df[north].mean(axis=1)) / 4
    else:
        df[f"{band}_grad_ns"] = np.nan
    if east and west:
        df[f"{band}_grad_ew"] = (df[east].mean(axis=1) - df[west].mean(axis=1)) / 4
    else:
        df[f"{band}_grad_ew"] = np.nan
    df[f"{band}_grad_mag"] = np.sqrt(df[f"{band}_grad_ns"]**2 + df[f"{band}_grad_ew"]**2)
    inner_col = f"{band}_inner_mean"
    outer_col = f"{band}_outer_mean"
    if inner_col in df.columns and outer_col in df.columns:
        df[f"{band}_local_vs_regional"] = df[inner_col] - df[outer_col]
    else:
        df[f"{band}_local_vs_regional"] = np.nan
    return df


def load_l2_hourly(l2_path, station_id):
    need_pixel_cols = [f"RF_{p}" for p in PIXEL_POSITIONS] + [f"SSA_{p}" for p in PIXEL_POSITIONS]
    need_summary = (
        [f"RF_{s}" for s in L2_SUMMARY] + [f"SSA_{s}" for s in L2_SUMMARY] +
        ["RF_inner_count", "RF_outer_count", "SSA_inner_count", "SSA_outer_count"]
    )
    use_cols = ["timestamp", "stationId"] + need_pixel_cols + need_summary

    df = pd.read_csv(l2_path, usecols=lambda c: c in use_cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    compute_gradients(df, "RF")
    compute_gradients(df, "SSA")

    df["ts_hour"] = df["timestamp"].dt.floor("h")
    keep_cols = ["ts_hour"]
    for band in ("RF", "SSA"):
        keep_cols += [f"{band}_{s}" for s in L2_SUMMARY]
        keep_cols += [f"{band}_grad_ns", f"{band}_grad_ew", f"{band}_grad_mag", f"{band}_local_vs_regional"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    hourly = df[keep_cols].groupby("ts_hour").mean().reset_index()
    hourly["stationId"] = station_id
    hourly.rename(columns={"ts_hour": "ts"}, inplace=True)
    return hourly


# ══════════════════════════════════════════════════════════════════════
#  REGION CLASSIFIER
# ══════════════════════════════════════════════════════════════════════

def classify_region(name, lat):
    name_lower = name.lower()
    north_provinces = ["hà nội", "hải phòng", "hải dương", "bắc ninh", "bắc giang",
        "quảng ninh", "hưng yên", "hà nam", "nam định", "ninh bình",
        "thái nguyên", "thái bình", "phú thọ", "vĩnh phúc", "tuyên quang",
        "hà giang", "lào cai", "yên bái", "sơn la", "hòa bình", "lạng sơn",
        "cao bằng", "bắc kạn", "điện biên", "lai châu"]
    # Central = Bắc/Nam Trung Bộ coast + Tây Nguyên highlands. Listed explicitly
    # so coastal/highland provinces below lat 14 (Bình Định, Phú Yên, Khánh Hòa,
    # Ninh Thuận, Bình Thuận, Lâm Đồng, Đắk Lắk, Đắk Nông) are not swept into
    # "South" by the latitude fallback.
    central_provinces = ["thanh hóa", "nghệ an", "hà tĩnh", "quảng bình",
        "quảng trị", "thừa thiên huế", "huế", "đà nẵng", "quảng nam",
        "quảng ngãi", "bình định", "phú yên", "khánh hòa", "ninh thuận",
        "bình thuận", "kon tum", "gia lai", "đắk lắk", "đắk nông", "đăk lăk",
        "lâm đồng"]
    south_provinces = ["tp hồ chí minh", "hồ chí minh", "bình dương", "đồng nai",
        "bà rịa", "vũng tàu", "long an", "tây ninh", "bình phước",
        "cần thơ", "an giang", "kiên giang", "cà mau", "bạc liêu", "sóc trăng",
        "trà vinh", "bến tre", "vĩnh long", "đồng tháp", "tiền giang", "hậu giang"]
    for p in north_provinces:
        if p in name_lower:
            return "North"
    for p in central_provinces:
        if p in name_lower:
            return "Central"
    for p in south_provinces:
        if p in name_lower:
            return "South"
    if lat is not None and not np.isnan(lat):
        if lat > 19.5:
            return "North"
        elif lat < 14:
            return "South"
    return "Central"


# ══════════════════════════════════════════════════════════════════════
#  STEP 1: MERGE ALL DATA SOURCES — ALL STATIONS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 1: MERGE ALL DATA SOURCES — ALL STATIONS (no exclusions)")
print("=" * 80)

all_station_dfs = []
merge_log = []
n_total = len(metadata)
n_skip = 0

for idx, (_, row) in enumerate(metadata.iterrows()):
    sid = str(row["stationId"])
    name = row["stationName"]
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    norm = normalize(name)
    region = classify_region(name, lat)

    print(f"\n[{idx+1:3d}/{n_total}] {name[:55]}...", end="")

    # 1. Envisoft
    env_path = fuzzy_get(env_idx, norm)
    if not env_path:
        print(f" SKIP (no Envisoft file)")
        n_skip += 1
        continue
    base = load_envisoft(env_path, name)
    if len(base) == 0:
        print(f" SKIP (empty)")
        n_skip += 1
        continue

    # PM2.5 coverage check (relaxed — 10%)
    if "PM2.5" in base.columns:
        pm_cov = base["PM2.5"].notna().mean()
        if pm_cov < MIN_PM25_COVERAGE:
            print(f" SKIP (PM2.5 cov={pm_cov:.0%})")
            n_skip += 1
            continue
    else:
        print(f" SKIP (no PM2.5)")
        n_skip += 1
        continue

    n_env = len(base)
    print(f" env={n_env}h", end="")

    # 2. OpenMeteo weather
    wx_path = fuzzy_get(wx_idx, norm)
    if wx_path:
        om = load_openmeteo(wx_path)
        base = base.merge(om, on="ts", how="left")
        print(f" wx=Y", end="")
    else:
        print(f" wx=N", end="")

    # 3. Himawari AOD
    aod_path = fuzzy_get(aod_idx, norm)
    if aod_path:
        aod = load_aod(aod_path)
        base = base.merge(aod, on="ts", how="left")
        n_aot = base["AOT"].notna().sum() if "AOT" in base.columns else 0
        print(f" aod={100*n_aot/max(len(base),1):.0f}%", end="")
    else:
        print(f" aod=N", end="")

    # 4. GPM rain
    gpm_path = fuzzy_get(gpm_idx, norm)
    if gpm_path:
        gpm = load_gpm(gpm_path)
        base = base.merge(gpm, on="ts", how="left")
        print(f" gpm=Y", end="")
    else:
        print(f" gpm=N", end="")

    # 5. DEM (static)
    dem_row = dem_lookup.get(sid)
    if dem_row is not None:
        for dc in ["elevation_m", "slope_deg", "aspect_deg", "aspect_sin", "aspect_cos"]:
            base[dc] = dem_row.get(dc, np.nan)

    # 6. RF/SSA from L2
    l2_path = l2_index.get(sid)
    if l2_path:
        l2_h = load_l2_hourly(l2_path, sid)
        l2_h["ts"] = l2_h["ts"].astype(str)
        base["ts_str"] = base["ts"].astype(str)
        base = base.merge(l2_h.drop(columns=["stationId"]),
                         left_on="ts_str", right_on="ts", how="left",
                         suffixes=("", "_l2"))
        base.drop(columns=["ts_str", "ts_l2"], errors="ignore", inplace=True)
        print(f" l2=Y", end="")
    else:
        print(f" l2=N", end="")

    # Met source priority
    for ec, oc, final_name in [
        ("Temperature", "Temperature_om", "Temperature_final"),
        ("Humidity", "Humidity_om", "Humidity_final"),
        ("Pressure", "Pressure_om", "Pressure_final"),
    ]:
        if ec in base.columns and oc in base.columns:
            base[final_name] = base[ec].fillna(base[oc])
        elif ec in base.columns:
            base[final_name] = base[ec]
        elif oc in base.columns:
            base[final_name] = base[oc]
        else:
            base[final_name] = np.nan

    # Metadata columns
    base["station"] = name
    base["stationId"] = str(sid)
    base["region"] = region
    stype = "LCS" if "(LCS)" in name else "KK"
    base["station_type"] = stype
    base["latitude"] = lat
    base["longitude"] = lon

    all_station_dfs.append(base)
    merge_log.append({
        "station": name, "stationId": sid, "station_type": stype,
        "n_rows": len(base),
        "aot_pct": round(100 * base["AOT"].notna().mean(), 1) if "AOT" in base.columns else 0,
        "pm25_pct": round(100 * base["PM2.5"].notna().mean(), 1) if "PM2.5" in base.columns else 0,
    })

print(f"\n\n{'='*60}")
print(f"Merged {len(all_station_dfs)} stations ({n_skip} skipped)")

# Concatenate
unified = pd.concat(all_station_dfs, ignore_index=True)
unified = unified.sort_values(["station", "ts"]).reset_index(drop=True)
print(f"Combined: {len(unified):,} rows, {unified['station'].nunique()} stations")

# ══════════════════════════════════════════════════════════════════════
#  STEP 1b: ROW-LEVEL PM2.5 QC MASK
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 1b: PM2.5 QUALITY MASK (row-level)")
print("=" * 80)

qc_masks = pm25_quality_masks(unified)
n_flagged = qc_masks.any(axis=1).sum()
print(f"  Flagged rows: {n_flagged:,} / {len(unified):,} ({100*n_flagged/len(unified):.1f}%)")
for col in qc_masks.columns:
    n = qc_masks[col].sum()
    print(f"    {col:20s}: {n:>8,}")

unified.loc[qc_masks.any(axis=1), "PM2.5"] = np.nan
n_valid = unified["PM2.5"].notna().sum()
print(f"  PM2.5 valid after QC: {n_valid:,}")

# Per-station summary
stn_summary = unified.groupby(["stationId", "station"]).agg(
    n_rows=("PM2.5", "size"),
    n_valid=("PM2.5", "count"),
    pm25_mean=("PM2.5", "mean"),
).reset_index()
stn_summary["coverage"] = stn_summary["n_valid"] / stn_summary["n_rows"]
print(f"\n  Station QC summary:")
print(f"    Total stations: {len(stn_summary)}")
print(f"    With >50% PM2.5: {(stn_summary['coverage'] > 0.5).sum()}")
print(f"    With >20% PM2.5: {(stn_summary['coverage'] > 0.2).sum()}")
print(f"    With <10% PM2.5: {(stn_summary['coverage'] < 0.1).sum()}")

# PBLH climatology fill
if "PBLH" in unified.columns:
    print("\nPBLH climatology fill...")
    n_nan_before = unified["PBLH"].isna().sum()
    unified["_hour"] = unified["ts"].dt.hour
    pblh_clim = (unified.dropna(subset=["PBLH"])
                 .groupby(["station", "_hour"])["PBLH"].mean()
                 .reset_index().rename(columns={"PBLH": "PBLH_clim"}))
    unified = unified.merge(pblh_clim, on=["station", "_hour"], how="left")
    fill_mask = unified["PBLH"].isna() & unified["PBLH_clim"].notna()
    unified["PBLH"] = unified["PBLH"].fillna(unified["PBLH_clim"])
    unified["PBLH_source"] = np.where(fill_mask, "climatology", "ERA5")
    unified.loc[unified["PBLH"].isna(), "PBLH_source"] = "missing"
    n_nan_after = unified["PBLH"].isna().sum()
    print(f"  Filled {fill_mask.sum():,} PBLH gaps ({n_nan_before:,} -> {n_nan_after:,} NaN)")
    unified = unified.drop(columns=["_hour", "PBLH_clim"], errors="ignore")

# AOT_fine
if "AOT" in unified.columns and "RF_center" in unified.columns:
    unified["AOT_fine"] = unified["AOT"] * unified["RF_center"]


# ══════════════════════════════════════════════════════════════════════
#  STEP 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 80)

station_dfs = []
for station_name, sdf in unified.groupby("station"):
    sdf = sdf.sort_values("ts").reset_index(drop=True)

    # AOD Spatial Gradients
    north_px, south_px, west_px, east_px = [], [], [], []
    for c in AOT_PIXELS:
        if c not in sdf.columns:
            continue
        parts = c.replace("AOT_", "")
        if len(parts) >= 4:
            row_idx = parts[:2]
            col_idx = parts[2:]
            if row_idx in ("m1", "m2"): north_px.append(c)
            if row_idx in ("p1", "p2"): south_px.append(c)
            if col_idx in ("m1", "m2"): west_px.append(c)
            if col_idx in ("p1", "p2"): east_px.append(c)

    if north_px and south_px:
        sdf["AOT_grad_ns"] = (sdf[south_px].mean(axis=1) - sdf[north_px].mean(axis=1)) / 4
    else:
        sdf["AOT_grad_ns"] = np.nan
    if east_px and west_px:
        sdf["AOT_grad_ew"] = (sdf[east_px].mean(axis=1) - sdf[west_px].mean(axis=1)) / 4
    else:
        sdf["AOT_grad_ew"] = np.nan
    sdf["AOT_grad_mag"] = np.sqrt(sdf["AOT_grad_ns"]**2 + sdf["AOT_grad_ew"]**2)
    sdf["AOT_grad_dir"] = np.arctan2(sdf["AOT_grad_ew"], sdf["AOT_grad_ns"])
    if "AOT_inner_mean" in sdf.columns and "AOT_outer_mean" in sdf.columns:
        sdf["AOT_local_vs_regional"] = sdf["AOT_inner_mean"] - sdf["AOT_outer_mean"]
    else:
        sdf["AOT_local_vs_regional"] = np.nan
    if "AOT_std" in sdf.columns:
        sdf["AOT_spatial_std"] = sdf["AOT_std"]

    # AOD Temporal Features
    if "AOT" in sdf.columns:
        sdf["AOT_lag_1h"] = sdf["AOT"].shift(1)
        sdf["AOT_lag_3h"] = sdf["AOT"].shift(3)
        sdf["AOT_lag_6h"] = sdf["AOT"].shift(6)
        sdf["AOT_rolling_mean_6h"] = sdf["AOT"].rolling(6, min_periods=1).mean()
        sdf["AOT_rolling_mean_24h"] = sdf["AOT"].rolling(24, min_periods=1).mean()

        aot_valid = sdf["AOT"].notna()
        hrs_since_aot = np.full(len(sdf), np.nan)
        last_valid = -1
        for i in range(len(sdf)):
            if aot_valid.iloc[i]:
                last_valid = i
                hrs_since_aot[i] = 0
            elif last_valid >= 0:
                hrs_since_aot[i] = i - last_valid
        sdf["hours_since_valid_AOT"] = hrs_since_aot
        sdf["AOT_ffill_48h"] = sdf["AOT"].ffill(limit=48)

    # Physics Correction
    if "Humidity_final" in sdf.columns:
        rh = sdf["Humidity_final"] / 100.0
        rh_factor = (1 - rh).clip(lower=0.01) ** 0.6
        sdf["RH_factor"] = rh_factor
    if "AOT" in sdf.columns and "PBLH" in sdf.columns and "RH_factor" in sdf.columns:
        pblh_safe = sdf["PBLH"].clip(lower=50)
        sdf["AOD_physics"] = sdf["AOT"] * sdf["RH_factor"] / pblh_safe

    # Precipitation Features
    if "precip_mm" in sdf.columns:
        rain_mask = sdf["precip_mm"].fillna(0) > 0.1
        hrs_since_rain = np.full(len(sdf), np.nan)
        last_rain = -1
        for i in range(len(sdf)):
            if rain_mask.iloc[i]:
                last_rain = i
                hrs_since_rain[i] = 0
            elif last_rain >= 0:
                hrs_since_rain[i] = i - last_rain
        sdf["hrs_since_rain"] = hrs_since_rain
        sdf["rain_sum_24h"] = sdf["precip_mm"].fillna(0).rolling(24, min_periods=1).sum()
        sdf["rain_sum_48h"] = sdf["precip_mm"].fillna(0).rolling(48, min_periods=1).sum()

        sdf["date"] = sdf["ts"].dt.date
        daily_rain = sdf.groupby("date")["precip_mm"].sum().reset_index()
        daily_rain["rain_day"] = (daily_rain["precip_mm"] > 0.1).astype(int)
        daily_rain["rain_days_7d"] = daily_rain["rain_day"].rolling(7, min_periods=1).sum()
        consec = np.zeros(len(daily_rain))
        for i in range(len(daily_rain)):
            if daily_rain["rain_day"].iloc[i] == 0:
                consec[i] = consec[i-1] + 1 if i > 0 else 1
        daily_rain["consecutive_dry_days"] = consec
        sdf = sdf.merge(daily_rain[["date", "rain_days_7d", "consecutive_dry_days"]],
                         on="date", how="left")
        sdf = sdf.drop(columns=["date"], errors="ignore")

    # Wind Components
    if "WS_om" in sdf.columns and "WD_om" in sdf.columns:
        wd_rad = np.radians(sdf["WD_om"])
        sdf["wind_u"] = sdf["WS_om"] * np.sin(wd_rad)
        sdf["wind_v"] = sdf["WS_om"] * np.cos(wd_rad)
        sdf["wind_dir_sin"] = np.sin(wd_rad)
        sdf["wind_dir_cos"] = np.cos(wd_rad)

    if "WS_local" in sdf.columns and "WD_local" in sdf.columns:
        wd_local_rad = np.radians(sdf["WD_local"])
        sdf["wind_u_local"] = sdf["WS_local"] * np.sin(wd_local_rad)
        sdf["wind_v_local"] = sdf["WS_local"] * np.cos(wd_local_rad)
        sdf["wind_dir_sin_local"] = np.sin(wd_local_rad)
        sdf["wind_dir_cos_local"] = np.cos(wd_local_rad)

    if "PBLH" in sdf.columns and "WS_om" in sdf.columns:
        sdf["VC"] = sdf["PBLH"] * sdf["WS_om"]

    for col_name, src_col in [("dT_6h", "Temperature_final"), ("dRH_6h", "Humidity_final"),
                               ("dWS_6h", "WS_om"), ("dP_6h", "Pressure_final")]:
        if src_col in sdf.columns:
            sdf[col_name] = sdf[src_col] - sdf[src_col].shift(6)

    # Temporal Encodings
    hour = sdf["ts"].dt.hour
    month = sdf["ts"].dt.month
    doy = sdf["ts"].dt.dayofyear
    sdf["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    sdf["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    sdf["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    sdf["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    sdf["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365)
    sdf["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365)

    # DEM Interactions
    if "elevation_m" in sdf.columns and "PBLH" in sdf.columns:
        sdf["elev_x_PBLH"] = sdf["elevation_m"] * sdf["PBLH"]
    if "elevation_m" in sdf.columns:
        sdf["elev_x_hour_sin"] = sdf["elevation_m"] * sdf["hour_sin"]

    station_dfs.append(sdf)

print("Re-concatenating with features...")
unified = pd.concat(station_dfs, ignore_index=True)
unified = unified.sort_values(["station", "ts"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
print(f"\nSaving to {OUT_PATH}...")
unified.to_csv(OUT_PATH, index=False)
fsize = os.path.getsize(OUT_PATH) / 1e6
elapsed = time.time() - t0_total
print(f"\n{'='*60}")
print(f"DONE — {elapsed:.0f}s total")
print(f"  Stations: {unified['station'].nunique()}")
n_kk = unified[unified["station_type"] == "KK"]["stationId"].nunique()
n_lcs = unified[unified["station_type"] == "LCS"]["stationId"].nunique()
print(f"    KK: {n_kk}, LCS: {n_lcs}")
print(f"  Rows: {len(unified):,}")
print(f"  Columns: {len(unified.columns)}")
print(f"  PM2.5 valid: {unified['PM2.5'].notna().sum():,}")
print(f"  File size: {fsize:.0f} MB")
print(f"  Output: {OUT_PATH}")

# Merge log
log_path = "analysis/thesis_audit/merge_log.csv"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
log_df = pd.DataFrame(merge_log)
log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
print(f"\nMerge log: {log_path}")
