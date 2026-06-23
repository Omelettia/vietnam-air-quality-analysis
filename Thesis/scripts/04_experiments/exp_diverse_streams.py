"""
Diverse-stream experiment: train 5 genuinely different XGBoost models
with different feature subsets, then measure oracle ceiling vs deployable routing.

Streams:
  dispersion  — meteorology + terrain + temporal (no satellite, no gas, no RFSI)
  satellite   — Himawari AOD + RF/SSA + minimal met (no gas, no RFSI)
  emission    — TROPOMI gas + emission proxies + building (no AOD, no RFSI)
  spatial     — RFSI + basic met + temporal (no AOD, no gas)
  full        — everything combined

Uses gbtree (not DART) for stability. Himawari AOD only.
"""

import argparse, io, sys, os, warnings, time, glob, zipfile, unicodedata, math
from unicodedata import normalize as _unorm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None)
parser.add_argument("--resume", action="store_true",
                    help="Resume from saved per-fold checkpoints")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def _repo_root():
    """Walk up to repo root (dir containing data/merged) so this runs from anywhere."""
    p = SCRIPT_DIR
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "data", "merged")):
            return p
        p = os.path.dirname(p)
    return os.path.dirname(SCRIPT_DIR)
REPO_DIR = _repo_root()
DATA_DIR = args.data_dir or REPO_DIR
META_DIR = os.path.join(DATA_DIR, "data", "stations", "metadata")
OUT_DIR = os.path.join(REPO_DIR, "analysis", "experimental_shape_magnitude",
                       "diverse_streams")
os.makedirs(OUT_DIR, exist_ok=True)

QC_DIR = os.path.join(REPO_DIR, "Thesis", "scripts", "02_processing")
if QC_DIR not in sys.path:
    sys.path.insert(0, QC_DIR)
from pm25_qc import pm25_quality_masks

SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}

XGB_PARAMS = dict(
    booster="gbtree",
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=40,
    reg_alpha=0.1, reg_lambda=8.0, tree_method="hist",
    device="cuda", random_state=42, n_jobs=-1,
)

K_NN = 5

# ============================================================================
#  HELPERS
# ============================================================================
def ascii_norm(s):
    return _unorm("NFKD", str(s)).encode("ascii", "ignore").decode().lower()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))

def assign_tier(mean_pm):
    if mean_pm < 10: return "t0"
    elif mean_pm < 20: return "t1"
    elif mean_pm < 35: return "t2"
    return "t3"

def safe_r2(y, p):
    if len(y) < 3 or np.std(y) < 1e-9:
        return np.nan
    return float(r2_score(y, p))

def pm_class(x):
    if x < 10: return "low"
    if x < 20: return "moderate_low"
    if x < 35: return "moderate"
    return "high"

# ============================================================================
#  1. LOAD DATA
# ============================================================================
print("=" * 80)
print("DIVERSE STREAMS EXPERIMENT (5 feature-subset XGBoost models)")
print("=" * 80)
t0_wall = time.time()

df = pd.read_csv(os.path.join(DATA_DIR, "data/merged/unified_thesis_v4.csv"),
                 dtype={"stationId": str})  # v4 = definitive (all 40 stations, stronger mask)
# v4 holds all 121 stations; restrict to the 40 thesis stations for training.
_thesis40 = set(pd.read_csv(os.path.join(DATA_DIR,
    "Thesis/results/01_stations/station_selection_final.csv"),
    dtype={"stationId": str})["stationId"])
df = df[df["stationId"].isin(_thesis40)].copy()
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["month"] = df["ts"].dt.month
df["date"] = df["ts"].dt.date
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations")

meta_path = os.path.join(DATA_DIR,
    "Thesis/results/01_stations/station_selection_final.csv")
meta = pd.read_csv(meta_path, dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))
sid_lat = dict(zip(meta["stationId"], meta["lat"]))
sid_lon = dict(zip(meta["stationId"], meta["lon"]))
station_ids = sorted(df["stationId"].unique())
n_stn = len(station_ids)

y_all = df["PM2.5"].values
stationId_vals = df["stationId"].values

qc_masks = pm25_quality_masks(df)
df.loc[qc_masks.any(axis=1), "PM2.5"] = np.nan
y_all = df["PM2.5"].values
y_log = np.log1p(np.nan_to_num(y_all, nan=0.0))

station_pm_means = df.groupby("stationId")["PM2.5"].mean()
sid_tier = {s: assign_tier(station_pm_means[s]) for s in station_ids}
for t in ["t0", "t1", "t2", "t3"]:
    n_t = sum(1 for s in station_ids if sid_tier[s] == t)
    print(f"  {t}: {n_t} stations")

global_pm_mean = float(np.nanmean(y_all))
bm_global = np.log1p(global_pm_mean)
y_res = y_log - bm_global

# ============================================================================
#  2. SATELLITE FEATURES (TROPOMI + MODIS LST)
# ============================================================================
print("\n--- Loading satellite features ---")

zip_candidates = sorted(glob.glob(
    os.path.join(DATA_DIR, "data", "gee_exports", "last-*.zip")))
zip_path = zip_candidates[-1]
all_sat = []
with zipfile.ZipFile(zip_path) as z:
    for name in sorted(z.namelist()):
        if name.endswith(".csv"):
            with z.open(name) as f:
                all_sat.append(pd.read_csv(f, dtype={"stationId": str}))

sat_long = pd.concat(all_sat, ignore_index=True)
sat_wide = sat_long.pivot_table(
    index=["stationId", "date"], columns="variable",
    values="mean", aggfunc="first"
).reset_index()
sat_wide.columns.name = None
sat_wide["date"] = pd.to_datetime(sat_wide["date"])
sat_wide = sat_wide[sat_wide["stationId"].isin(set(station_ids))].copy()
sat_wide["month"] = sat_wide["date"].dt.month

CLIM_COLS = ["NO2", "SO2", "CO", "HCHO", "LST_terra_day", "LST_terra_night"]
for c in CLIM_COLS:
    if c not in sat_wide.columns:
        sat_wide[c] = np.nan

clim = sat_wide.groupby(["stationId", "month"])[CLIM_COLS].transform("mean")
sat_wide["so2_daily_anom"] = sat_wide["SO2"] - clim["SO2"]
sat_wide["co_daily_anom"] = sat_wide["CO"] - clim["CO"]
sat_wide["no2_daily_anom"] = sat_wide["NO2"] - clim["NO2"]
sat_wide["lst_day_anom"] = sat_wide["LST_terra_day"] - clim["LST_terra_day"]
sat_wide["lst_night_anom"] = sat_wide["LST_terra_night"] - clim["LST_terra_night"]

ANOM_RAW = ["so2_daily_anom", "co_daily_anom", "no2_daily_anom",
            "lst_day_anom", "lst_night_anom"]
sat_wide["date_merge"] = sat_wide["date"].dt.date
df = df.merge(sat_wide[["stationId", "date_merge"] + ANOM_RAW],
              left_on=["stationId", "date"],
              right_on=["stationId", "date_merge"], how="left")
df.drop(columns=["date_merge"], inplace=True)
print(f"  Daily anomalies merged")

# ============================================================================
#  3. BUILDING DENSITY
# ============================================================================
bld = pd.read_csv(os.path.join(META_DIR, "station_building_density.csv"),
                  dtype={"stationId": str})
BUILDING_COLS = ["building_count_1km", "building_area_1km",
                 "building_count_3km", "building_area_3km"]
bld_map = bld.set_index("stationId")[BUILDING_COLS]
df = df.merge(bld_map, left_on="stationId", right_index=True, how="left")
for col in BUILDING_COLS:
    df[col] = df[col].fillna(0)

# ============================================================================
#  4. MATCHED AOD FEATURES (Himawari)
# ============================================================================
print("\n--- Loading matched Himawari AOD features ---")
matched_static_path = os.path.join(META_DIR, "aod_source_matched_static.csv")
matched_temporal_path = os.path.join(META_DIR, "aod_source_matched_temporal.csv")
matched_static = pd.read_csv(matched_static_path, dtype={"stationId": str})
matched_static_map = matched_static.set_index("stationId")
matched_temporal = pd.read_csv(
    matched_temporal_path, dtype={"stationId": str}, parse_dates=["date"])
matched_temporal["date_merge"] = matched_temporal["date"].dt.date

prefix = "him"
temporal_rename = {
    f"{prefix}_aod_7d": "src_aod_7d",
    f"{prefix}_fmf_7d": "src_fmf_7d",
    f"{prefix}_fine_aod_7d": "src_fine_aod_7d",
}
keep_temporal = ["stationId", "date_merge"] + [
    c for c in temporal_rename if c in matched_temporal.columns]
matched_temporal = matched_temporal[keep_temporal].rename(columns=temporal_rename)
df = df.merge(matched_temporal, left_on=["stationId", "date"],
              right_on=["stationId", "date_merge"], how="left")
df.drop(columns=["date_merge"], inplace=True)

static_rename = {}
for suffix in (["aod_center", "aod_DJF", "aod_MAM", "aod_JJA", "aod_SON",
                "aod_contrast", "aod_directionality", "aod_max_nearby",
                "fmf_center", "fine_aod_center", "ae_center"] +
               [f"aod_clim_{d}" for d in SECTOR_NAMES] +
               [f"fine_aod_clim_{d}" for d in SECTOR_NAMES]):
    col = f"{prefix}_{suffix}"
    if col in matched_static_map.columns:
        static_rename[col] = f"src_{suffix}"
if static_rename:
    df = df.merge(
        matched_static_map[list(static_rename)].rename(columns=static_rename),
        left_on="stationId", right_index=True, how="left")

SOURCE_AOD_FEATURES = [
    "src_aod_7d", "src_fmf_7d", "src_fine_aod_7d",
    "src_aod_center", "src_fmf_center", "src_fine_aod_center",
    "src_ae_center", "src_aod_DJF", "src_aod_MAM", "src_aod_JJA",
    "src_aod_SON", "src_aod_contrast", "src_aod_directionality",
    "src_aod_max_nearby",
] + [f"src_aod_clim_{d}" for d in SECTOR_NAMES] + [
    f"src_fine_aod_clim_{d}" for d in SECTOR_NAMES
]
SOURCE_AOD_FEATURES = [f for f in SOURCE_AOD_FEATURES if f in df.columns]
print(f"  Matched AOD: {len(SOURCE_AOD_FEATURES)} features")

# ============================================================================
#  5. SATELLITE STATIC FEATURES (NO2, emission, NTL, LST)
# ============================================================================
print("\n--- Loading satellite static features ---")
no2_feat = pd.read_csv(os.path.join(META_DIR, "station_no2_features.csv"),
                       dtype={"stationId": str})
emit_feat = pd.read_csv(os.path.join(META_DIR, "station_emission_features.csv"),
                        dtype={"stationId": str})
new_sat_feat = pd.read_csv(os.path.join(META_DIR, "station_all_satellite_features.csv"),
                           dtype={"stationId": str})

no2_map = no2_feat.set_index("stationId")
emit_map = emit_feat.set_index("stationId")
new_sat_map = new_sat_feat.set_index("stationId")

NO2_STATIC_COLS = ["no2_center", "no2_contrast", "no2_directionality"]
NO2_SECTOR_COLS = [f"no2_clim_{d}" for d in SECTOR_NAMES]
df = df.merge(no2_map[NO2_STATIC_COLS + NO2_SECTOR_COLS],
              left_on="stationId", right_index=True, how="left")

NTL_SECTOR_COLS = [f"ntl_clim_{d}" for d in SECTOR_NAMES]
LST_SECTOR_COLS = [f"lst_anom_clim_{d}" for d in SECTOR_NAMES]
merge_emit = (["ntl_center"] + NTL_SECTOR_COLS +
              ["lst_anom_center"] + LST_SECTOR_COLS)
merge_emit = [c for c in merge_emit if c in emit_map.columns]
df = df.merge(emit_map[merge_emit],
              left_on="stationId", right_index=True, how="left")

merge_new = (["faod_center", "fmf_center", "ae_center", "so2_center"] +
             [f"faod_clim_{d}" for d in SECTOR_NAMES] +
             [f"so2_clim_{d}" for d in SECTOR_NAMES])
merge_new = [c for c in merge_new if c in new_sat_map.columns]
df = df.merge(new_sat_map[merge_new],
              left_on="stationId", right_index=True, how="left")

# ============================================================================
#  6. DIRECTIONAL CLIMATOLOGY (SO2, CO, HCHO, LST) + SMART_V1
# ============================================================================
print("\n--- Building directional features + smart_v1 ---")

def _norm_tok(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()

def _tokenize(s):
    return set(_norm_tok(s).replace("-", " ").replace(",", " ").split())

no2_clim_csv = os.path.join(META_DIR, "no2_directional_clim.csv")
no2_dir_raw = pd.read_csv(no2_clim_csv, dtype={"stationId": str})
no2_dir_names = no2_dir_raw.groupby("stationId")["name"].first()

id_map = {}
for short_id, no2_name in no2_dir_names.items():
    no2_tokens = _tokenize(no2_name)
    best_score, best_long_id = -1, None
    for _, row in meta.iterrows():
        meta_tokens = _tokenize(row["station_name"])
        score = len(no2_tokens & meta_tokens)
        if score > best_score:
            best_score = score
            best_long_id = row["stationId"]
    id_map[short_id] = best_long_id

def _load_dir_clim(csv_path, value_col="mean"):
    raw = pd.read_csv(csv_path, dtype={"stationId": str})
    raw["stationId"] = raw["stationId"].map(id_map)
    raw = raw.dropna(subset=["stationId"])
    clim_df = raw.groupby(["stationId", "direction"])[value_col].mean().reset_index()
    sectors, centers = {}, {}
    for sid in station_ids:
        sf = clim_df[clim_df["stationId"] == sid]
        sec = np.full(8, np.nan)
        for di, d in enumerate(SECTOR_NAMES):
            vals = sf[sf["direction"] == d][value_col]
            if len(vals) > 0: sec[di] = float(vals.iloc[0])
        sectors[sid] = sec
        cvals = sf[sf["direction"] == "C"][value_col]
        centers[sid] = float(cvals.iloc[0]) if len(cvals) > 0 else np.nan
    return sectors, centers

def _resolve_path(fn):
    p = os.path.join(META_DIR, fn)
    return p if os.path.exists(p) else os.path.join(DATA_DIR, fn)

station_so2_sectors, station_so2_centers = _load_dir_clim(
    _resolve_path("tropomi_so2_directional.csv"), "mean")
station_co_sectors, station_co_centers = _load_dir_clim(
    _resolve_path("tropomi_co_directional.csv"), "mean")
station_hcho_sectors, station_hcho_centers = _load_dir_clim(
    _resolve_path("tropomi_hcho_directional.csv"), "mean")
station_lstd_sectors, station_lstd_centers = _load_dir_clim(
    _resolve_path("lst_anomaly_directional.csv"), "lst_anomaly")

# Wind direction -> sector index + VC_inv
wd_from = np.degrees(np.arctan2(-df["wind_u"].values, -df["wind_v"].values)) % 360
sector_idx = ((wd_from + 22.5) / 45).astype(int) % 8
ws = np.sqrt(df["wind_u"].values**2 + df["wind_v"].values**2)
vc_inv = 1.0 / (df["PBLH"].clip(lower=50).values * ws.clip(min=0.1) + 1)

# Smart_v1 composite
all_no2_sec = np.array([no2_map.loc[s, [f"no2_clim_{d}" for d in SECTOR_NAMES]].values
                        if s in no2_map.index else np.full(8, np.nan)
                        for s in station_ids])
all_ntl_sec = np.array([emit_map.loc[s, [f"ntl_clim_{d}" for d in SECTOR_NAMES]].values
                        if s in emit_map.index else np.full(8, np.nan)
                        for s in station_ids])
all_lst_sec = np.array([emit_map.loc[s, [f"lst_anom_clim_{d}" for d in SECTOR_NAMES]].values
                        if s in emit_map.index else np.full(8, np.nan)
                        for s in station_ids])
no2_center_all = np.array([no2_map.loc[s, "no2_center"] if s in no2_map.index else np.nan
                           for s in station_ids])
ntl_center_all = np.array([emit_map.loc[s, "ntl_center"] if s in emit_map.index else np.nan
                           for s in station_ids])
lst_center_all = np.array([emit_map.loc[s, "lst_anom_center"] if s in emit_map.index else np.nan
                           for s in station_ids])

fmf_col = f"{prefix}_fmf_center"
fmf_center_all = np.array([
    matched_static_map.loc[s, fmf_col]
    if (s in matched_static_map.index and fmf_col in matched_static_map.columns)
    else np.nan for s in station_ids])

def _lohi(sec_arr, cen_arr):
    combined = np.concatenate([sec_arr.ravel(), cen_arr])
    return float(np.nanmin(combined)), float(np.nanmax(combined))

no2_lo, no2_hi = _lohi(all_no2_sec, no2_center_all)
ntl_lo, ntl_hi = _lohi(all_ntl_sec, ntl_center_all)
lst_lo, lst_hi = _lohi(all_lst_sec, lst_center_all)

def norm01(v, lo, hi):
    if hi - lo < 1e-12: return 0.0
    return float((v - lo) / (hi - lo)) if not np.isnan(v) else 0.0

station_smart_v1_sec, station_smart_v1_cen = {}, {}
for si, sid in enumerate(station_ids):
    fmf = fmf_center_all[si]
    if np.isnan(fmf): fmf = 0.5
    v1_sec = np.zeros(8)
    for di in range(8):
        no2_n = norm01(all_no2_sec[si, di], no2_lo, no2_hi)
        ntl_n = norm01(all_ntl_sec[si, di], ntl_lo, ntl_hi)
        lst_n = norm01(all_lst_sec[si, di], lst_lo, lst_hi)
        v1_sec[di] = no2_n * (1.0 + ntl_n) * (1.0 + lst_n) * fmf
    station_smart_v1_sec[sid] = v1_sec
    no2_cn = norm01(no2_center_all[si], no2_lo, no2_hi)
    ntl_cn = norm01(ntl_center_all[si], ntl_lo, ntl_hi)
    lst_cn = norm01(lst_center_all[si], lst_lo, lst_hi)
    station_smart_v1_cen[sid] = no2_cn * (1.0 + ntl_cn) * (1.0 + lst_cn) * fmf

df["smart_v1_center"] = np.array([station_smart_v1_cen[s] for s in stationId_vals])
smart_v1_upwind = np.zeros(len(df))
for sid in station_ids:
    mask = stationId_vals == sid
    if not mask.any(): continue
    idx = np.where(mask)[0]
    smart_v1_upwind[idx] = station_smart_v1_sec[sid][sector_idx[idx]]
df["smart_v1_upwind"] = smart_v1_upwind
s1_max = np.array([station_smart_v1_sec[s].max() for s in stationId_vals])
s1_min = np.array([station_smart_v1_sec[s].min() for s in stationId_vals])
df["smart_v1_max"] = s1_max
df["smart_v1_contrast"] = s1_max / (s1_min + 0.001)
df["smart_v1_upwind_x_VC_inv"] = smart_v1_upwind * vc_inv

# SO2, CO, HCHO standalone features
so2_upwind_vals = np.zeros(len(df))
co_upwind_vals = np.zeros(len(df))
lst_anom_upwind_vals = np.zeros(len(df))
for sid in station_ids:
    mask = stationId_vals == sid
    if not mask.any(): continue
    idx = np.where(mask)[0]
    so2_upwind_vals[idx] = station_so2_sectors[sid][sector_idx[idx]]
    co_upwind_vals[idx] = station_co_sectors[sid][sector_idx[idx]]
    lst_anom_upwind_vals[idx] = station_lstd_sectors[sid][sector_idx[idx]]

so2_upwind_vals = np.nan_to_num(so2_upwind_vals)
co_upwind_vals = np.nan_to_num(co_upwind_vals)
lst_anom_upwind_vals = np.nan_to_num(lst_anom_upwind_vals)

df["so2_upwind"] = so2_upwind_vals
df["co_upwind"] = co_upwind_vals
so2_cen_map = {sid: float(np.nan_to_num(station_so2_centers.get(sid, 0.0), nan=0.0))
               for sid in station_ids}
co_cen_map = {sid: float(np.nan_to_num(station_co_centers.get(sid, 0.0), nan=0.0))
              for sid in station_ids}
df["so2_center"] = df["stationId"].map(so2_cen_map).fillna(0.0)
df["co_center"] = df["stationId"].map(co_cen_map).fillna(0.0)

so2_contrast_map = {}
for sid in station_ids:
    sec = station_so2_sectors[sid]
    valid = sec[~np.isnan(sec)]
    so2_contrast_map[sid] = float(valid.max() / valid.mean()) if len(valid) > 0 and valid.mean() > 1e-12 else 1.0
df["so2_contrast"] = df["stationId"].map(so2_contrast_map).fillna(1.0)

hcho_cen_map = {sid: float(np.nan_to_num(station_hcho_centers.get(sid, 0.0), nan=0.0))
                for sid in station_ids}
df["hcho_center"] = df["stationId"].map(hcho_cen_map).fillna(0.0)

df["so2_upwind_x_VC_inv"] = so2_upwind_vals * vc_inv
df["lst_anom_upwind_x_VC_inv"] = lst_anom_upwind_vals * vc_inv

# Daily anomaly interactions
df["so2_anom_x_vc_inv"] = df["so2_daily_anom"].values * vc_inv
df["co_anom_x_vc_inv"] = df["co_daily_anom"].values * vc_inv
df["lst_anom_x_vc_inv"] = df["lst_day_anom"].values * vc_inv
ANOM_INTERACT = ["so2_anom_x_vc_inv", "co_anom_x_vc_inv", "lst_anom_x_vc_inv"]
DAILY_ANOM_ALL = ANOM_RAW + ANOM_INTERACT

# Fire upwind
fire_csv_path = _resolve_path("fire_counts_directional.csv")
fire_raw = pd.read_csv(fire_csv_path, dtype={"stationId": str})
fire_raw = fire_raw.rename(columns={"mean": "fire_val"})
fire_raw["stationId"] = fire_raw["stationId"].map(id_map)
fire_raw = fire_raw.dropna(subset=["stationId"])
station_fire_dir_season = {}
for sid in station_ids:
    sf = fire_raw[fire_raw["stationId"] == sid]
    lookup = {}
    for di, d in enumerate(SECTOR_NAMES):
        for szn in ["DJF", "MAM", "JJA", "SON"]:
            vals = sf[(sf["direction"] == d) & (sf["season"] == szn)]["fire_val"]
            lookup[(di, szn)] = float(vals.mean()) if len(vals) > 0 else 0.0
    for szn in ["DJF", "MAM", "JJA", "SON"]:
        vals = sf[(sf["direction"] == "C") & (sf["season"] == szn)]["fire_val"]
        lookup[(-1, szn)] = float(vals.mean()) if len(vals) > 0 else 0.0
    station_fire_dir_season[sid] = lookup

month_vals = df["month"].values
season_vals = np.array([SEASON_MAP[m] for m in month_vals])
fire_upwind = np.zeros(len(df))
for sid in station_ids:
    mask = stationId_vals == sid
    if not mask.any(): continue
    idx = np.where(mask)[0]
    lookup = station_fire_dir_season.get(sid, {})
    for i in idx:
        fire_upwind[i] = lookup.get((sector_idx[i], season_vals[i]), 0.0)
df["fire_upwind"] = fire_upwind

# Fill NaN for static features
fill_cols = (NO2_STATIC_COLS + NO2_SECTOR_COLS +
             ["ntl_center", "smart_v1_center", "smart_v1_upwind", "smart_v1_max",
              "smart_v1_contrast", "smart_v1_upwind_x_VC_inv",
              "so2_upwind", "so2_center", "so2_contrast",
              "co_upwind", "co_center", "hcho_center",
              "so2_upwind_x_VC_inv", "lst_anom_upwind_x_VC_inv", "fire_upwind"])
for c in set(fill_cols):
    if c in df.columns:
        df[c] = df[c].fillna(0)

# ============================================================================
#  7. OUTER AOD PHYSICS
# ============================================================================
aot_outer = df["AOT_outer_mean"].fillna(0).values
aot_center = df["AOT_ffill_48h"].fillna(0).values
pblh = df["PBLH"].fillna(200).values
rh_frac = (df["Humidity_final"] / 100.0).clip(0, 0.95).values
f_rh = 1.0 / (1.0 - rh_frac)
hours_since = df["hours_since_valid_AOT"].fillna(999).values

df["aod_outer_surface"] = aot_outer / (pblh + 100.0)
df["aod_outer_pm25"] = aot_outer / (pblh + 100.0) / f_rh
df["aod_outer_x_VC_inv"] = aot_outer * vc_inv
df["aod_outer_gradient"] = aot_outer - aot_center
is_real_aod = (hours_since == 0) & (aot_center > 0)
df["_outer_real"] = np.where(is_real_aod, aot_outer, np.nan)
df["_is_real_aod"] = is_real_aod
day_outer = df.groupby(["stationId", "date"]).agg(
    aod_outer_day_mean=("_outer_real", "mean"),
    _day_count=("_is_real_aod", "sum"),
).reset_index()
day_outer.loc[day_outer["_day_count"] == 0, "aod_outer_day_mean"] = np.nan
df = df.merge(day_outer[["stationId", "date", "aod_outer_day_mean"]],
              on=["stationId", "date"], how="left")
df.drop(columns=["_outer_real", "_is_real_aod"], inplace=True)

OUTER_ALL_EXTRA = ["aod_outer_surface", "aod_outer_pm25",
                   "aod_outer_x_VC_inv", "aod_outer_gradient",
                   "aod_outer_day_mean"]

# ============================================================================
#  8. WEATHER PERSISTENCE + TEMPORAL EXTRAS
# ============================================================================
df["dow_is_weekend"] = (df["ts"].dt.dayofweek >= 5).astype(float)
df['PBLH_min_24h'] = df.groupby('stationId')['PBLH'].transform(
    lambda x: x.rolling(24, min_periods=1).min())
df['VC_min_24h'] = df.groupby('stationId')['VC'].transform(
    lambda x: x.rolling(24, min_periods=1).min())
stag_col = ((df['PBLH'] < 500) & (df['WS_local'].fillna(0) < 2)).astype(float)
df['stagnation_hours_12h'] = stag_col.groupby(df['stationId']).rolling(
    12, min_periods=1).sum().reset_index(level=0, drop=True)
df['temp_diurnal_anomaly'] = df['Temperature_final'] - df.groupby(
    ['stationId', df['ts'].dt.month, df['ts'].dt.hour]
)['Temperature_final'].transform('mean')
df['day_of_year_sin'] = np.sin(2 * np.pi * df['ts'].dt.dayofyear / 365.25)

WEATHER_PERSIST = ['PBLH_min_24h', 'VC_min_24h', 'stagnation_hours_12h',
                   'temp_diurnal_anomaly']

# ============================================================================
#  9. RFSI SETUP
# ============================================================================
print("\n--- RFSI setup ---")
coords = {s: (sid_lat[s], sid_lon[s]) for s in station_ids}
sid_to_idx = {s: i for i, s in enumerate(station_ids)}
dist_full = np.zeros((n_stn, n_stn))
for i in range(n_stn):
    for j in range(i + 1, n_stn):
        d = haversine(*coords[station_ids[i]], *coords[station_ids[j]])
        dist_full[i, j] = d
        dist_full[j, i] = d

neighbor_order = {}
for i in range(n_stn):
    neighbor_order[i] = sorted(
        [(j, dist_full[i, j]) for j in range(n_stn) if j != i],
        key=lambda x: x[1])

pm25_wide = df.pivot_table(index="ts", columns="stationId",
                           values="PM2.5", aggfunc="first")
pm25_mat = pm25_wide.values
sid_cols = list(pm25_wide.columns)
sid_to_col = {s: i for i, s in enumerate(sid_cols)}
ts_to_row = pd.Series(range(len(pm25_wide)), index=pm25_wide.index)
df["ts_row"] = df["ts"].map(ts_to_row).astype(int).values

RFSI_LEAN = ["PM25_nn_idw", "PM25_nn1", "dist_nn1"]
RFSI_EXTRA = ["PM25_nn2", "PM25_nn3"]
LAG_FEATURES = ["PM25_nn1_lag1h", "PM25_nn1_lag3h", "PM25_nn1_lag6h"]
LAG_HOURS = [1, 3, 6]
n_ts = pm25_mat.shape[0]

def compute_rfsi(exclude_sid=None):
    n = len(df)
    pm_nn = np.full((n, K_NN), np.nan)
    d_nn = np.full((n, K_NN), np.nan)
    excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
    ts_row_vals = df["ts_row"].values
    for sid in station_ids:
        si = sid_to_idx[sid]
        mask = stationId_vals == sid
        if not mask.any(): continue
        ri = np.where(mask)[0]
        tr = ts_row_vals[ri]
        cands = [(j, d) for j, d in neighbor_order[si]
                 if excl is None or j != excl]
        if not cands: continue
        ccols = np.array([sid_to_col[station_ids[j]] for j, _ in cands])
        cdists = np.array([d for _, d in cands])
        nbr = pm25_mat[np.ix_(tr, ccols)]
        valid = ~np.isnan(nbr)
        cumv = np.cumsum(valid, axis=1)
        for k in range(K_NN):
            reached = cumv >= (k + 1)
            has = reached.any(axis=1)
            if not has.any(): break
            pos = np.argmax(reached, axis=1)
            ih = np.where(has)[0]
            pm_nn[ri[ih], k] = nbr[ih, pos[has]]
            d_nn[ri[ih], k] = cdists[pos[has]]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d_nn
        pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)
    return {"PM25_nn_idw": pm_idw, "PM25_nn1": pm_nn[:,0], "dist_nn1": d_nn[:,0],
            "PM25_nn2": pm_nn[:,1], "PM25_nn3": pm_nn[:,2]}

def compute_lagged_rfsi(exclude_sid=None):
    n = len(df)
    lags = {lh: np.full(n, np.nan) for lh in LAG_HOURS}
    excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
    ts_row_vals = df["ts_row"].values
    for sid in station_ids:
        si = sid_to_idx[sid]
        mask = stationId_vals == sid
        if not mask.any(): continue
        ri = np.where(mask)[0]
        tr = ts_row_vals[ri]
        cands = [(j, d) for j, d in neighbor_order[si]
                 if excl is None or j != excl]
        if not cands: continue
        nn1_col = sid_to_col[station_ids[cands[0][0]]]
        for lag_h in LAG_HOURS:
            tr_lag = tr - lag_h
            in_bounds = tr_lag >= 0
            tr_safe = np.clip(tr_lag, 0, n_ts - 1)
            vals = pm25_mat[tr_safe, nn1_col]
            vals[~in_bounds] = np.nan
            lags[lag_h][ri] = vals
    return {f"PM25_nn1_lag{lh}h": lags[lh] for lh in LAG_HOURS}

# ============================================================================
#  10. FEATURE SET DEFINITIONS
# ============================================================================
print("\n--- Feature set definitions ---")

MET_CORE = ["PBLH", "VC", "wind_u", "wind_v", "WS_local",
            "Temperature_final", "Humidity_final", "Pressure_final",
            "dT_6h", "dRH_6h", "rain_days_7d", "rain_sum_48h",
            "consecutive_dry_days", "hrs_since_rain", "RH_factor"]
TEMPORAL = ["hour_sin", "hour_cos", "month_sin", "month_cos",
            "day_of_year_cos", "day_of_year_sin", "dow_is_weekend"]

AOD_CORE = (["AOT_ffill_48h", "AOT_outer_mean", "AE", "RF",
             "hours_since_valid_AOT"] + SOURCE_AOD_FEATURES)
AOD_EXTENDED = AOD_CORE + OUTER_ALL_EXTRA + [
    "RF_center", "RF_mean", "SSA_center", "SSA_mean", "AOT_fine",
    "AOT_grad_mag", "AOT_local_vs_regional"]

SMART_EMISSION = ["smart_v1_center", "smart_v1_upwind", "smart_v1_max",
                  "smart_v1_contrast", "smart_v1_upwind_x_VC_inv"]
GAS_STANDALONE = ["so2_upwind", "so2_center", "so2_contrast",
                  "co_upwind", "co_center", "hcho_center",
                  "so2_upwind_x_VC_inv", "lst_anom_upwind_x_VC_inv"]
EMISSION_STATIC = (NO2_STATIC_COLS + NO2_SECTOR_COLS +
                   ["ntl_center"] + NTL_SECTOR_COLS +
                   ["lst_anom_center"] + LST_SECTOR_COLS)

BUILDING = ["building_area_1km", "building_count_3km"]
TERRAIN = ["elevation_m", "slope_deg"]

RFSI_ALL = RFSI_LEAN + RFSI_EXTRA + LAG_FEATURES

# -- The 5 diverse streams --
STREAMS = {
    "dispersion": (
        MET_CORE + WEATHER_PERSIST + TEMPORAL + TERRAIN + BUILDING +
        ["fire_upwind"]
    ),
    "satellite": (
        AOD_EXTENDED + ["PBLH", "VC", "RH_factor", "hours_since_rain"] +
        TEMPORAL[:5]
    ),
    "emission": (
        SMART_EMISSION + GAS_STANDALONE + DAILY_ANOM_ALL + EMISSION_STATIC +
        BUILDING + TERRAIN + ["PBLH", "VC", "fire_upwind"] +
        TEMPORAL
    ),
    "spatial": (
        RFSI_ALL + MET_CORE + TEMPORAL + TERRAIN
    ),
    "full": (
        MET_CORE + WEATHER_PERSIST + TEMPORAL + TERRAIN + BUILDING +
        AOD_EXTENDED + SMART_EMISSION + GAS_STANDALONE + DAILY_ANOM_ALL +
        EMISSION_STATIC + RFSI_ALL + ["fire_upwind"]
    ),
}

for name, feats in list(STREAMS.items()):
    feats_clean = sorted(set(f for f in feats if f in df.columns))
    STREAMS[name] = feats_clean
    STREAMS[f"raw_{name}"] = feats_clean
    print(f"  {name:<12s}: {len(feats_clean)} features")

STREAM_NAMES = list(STREAMS.keys())
print(f"  + 5 raw-space mirrors → {len(STREAM_NAMES)} total streams")

print(f"\nData loading complete ({time.time()-t0_wall:.0f}s)")

# ============================================================================
#  11. LOSO TRAINING
# ============================================================================
print(f"\n{'='*80}")
print(f"LOSO TRAINING: {len(STREAM_NAMES)} streams x {n_stn} folds")
print(f"{'='*80}")

oof_preds = {s: np.full(len(df), np.nan) for s in STREAM_NAMES}
fold_checkpoint = os.path.join(OUT_DIR, "fold_checkpoint.csv")
completed_folds = set()

if args.resume and os.path.exists(fold_checkpoint):
    ck = pd.read_csv(fold_checkpoint)
    completed_folds = set(zip(ck["stream"], ck["station_id"]))
    print(f"  Resuming: {len(completed_folds)} fold-stream pairs done")

    oof_path = os.path.join(OUT_DIR, "oof_predictions_partial.csv")
    if os.path.exists(oof_path):
        partial = pd.read_csv(oof_path, dtype={"stationId": str})
        for sn in STREAM_NAMES:
            col = f"pred_{sn}"
            if col in partial.columns:
                vals = partial[col].values
                mask = ~np.isnan(vals)
                oof_preds[sn][mask] = vals[mask]
        print(f"  Loaded partial OOF predictions")

checkpoint_rows = []
fold_times = []

for fi, held_sid in enumerate(station_ids):
    held_name = sid_name.get(held_sid, "?")
    fold_t0 = time.time()

    # Check which streams need training for this fold
    needed = [s for s in STREAM_NAMES if (s, held_sid) not in completed_folds]
    if not needed:
        continue

    # Masks
    held_mask = stationId_vals == held_sid
    train_mask = ~held_mask & ~np.isnan(y_all)
    test_idx = np.where(held_mask)[0]
    train_idx = np.where(train_mask)[0]

    if len(test_idx) == 0 or len(train_idx) == 0:
        continue

    y_train_log = y_res[train_idx]
    y_train_raw = y_all[train_idx].copy()
    y_train_raw = np.nan_to_num(y_train_raw, nan=0.0)

    # Compute RFSI for this fold (needed for spatial + full streams)
    needs_rfsi = any(s.replace("raw_", "") in ["spatial", "full"] for s in needed)
    if needs_rfsi:
        rfsi_vals = compute_rfsi(exclude_sid=held_sid)
        lag_vals = compute_lagged_rfsi(exclude_sid=held_sid)
        for col_name, vals in {**rfsi_vals, **lag_vals}.items():
            df[col_name] = vals

    for sn in needed:
        feats = STREAMS[sn]
        X_train = df.iloc[train_idx][feats].values.astype(np.float32)
        X_test = df.iloc[test_idx][feats].values.astype(np.float32)

        nan_mask_tr = np.isnan(X_train)
        if nan_mask_tr.any():
            col_medians = np.nanmedian(X_train, axis=0)
            for c in range(X_train.shape[1]):
                X_train[nan_mask_tr[:, c], c] = col_medians[c]
            nan_mask_te = np.isnan(X_test)
            for c in range(X_test.shape[1]):
                X_test[nan_mask_te[:, c], c] = col_medians[c]

        is_raw = sn.startswith("raw_")
        y_tr = y_train_raw if is_raw else y_train_log
        y_eval = y_all[test_idx].copy() if is_raw else y_res[test_idx]
        if is_raw:
            y_eval = np.nan_to_num(y_eval, nan=0.0)

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_tr,
                  eval_set=[(X_test, y_eval)],
                  verbose=False)

        raw_pred = model.predict(X_test)
        if is_raw:
            pred_pm = np.clip(raw_pred, 0.1, 300)
        else:
            pred_pm = np.expm1(raw_pred + bm_global).clip(0.1, 300)
        oof_preds[sn][test_idx] = pred_pm

        checkpoint_rows.append({"stream": sn, "station_id": held_sid})

    elapsed = time.time() - fold_t0
    fold_times.append(elapsed)
    avg = np.mean(fold_times)
    remaining = (n_stn - fi - 1) * avg / 60

    # Quick per-fold summary
    brief = []
    for sn in needed:
        y_test = y_all[test_idx]
        p_test = oof_preds[sn][test_idx]
        valid = ~np.isnan(y_test) & ~np.isnan(p_test)
        if valid.sum() >= 3:
            r2 = safe_r2(y_test[valid], p_test[valid])
            brief.append(f"{sn[:3]}={r2:+.2f}")
    print(f"  [{fi+1:2d}/{n_stn}] {held_name[:30]:<30s} "
          f"{elapsed:.0f}s  {' '.join(brief)}  "
          f"(~{remaining:.0f}m left)")

    # Save checkpoint every 5 folds
    if (fi + 1) % 5 == 0 or fi == n_stn - 1:
        ck_df = pd.DataFrame(checkpoint_rows)
        ck_df.to_csv(fold_checkpoint, index=False)
        partial_df = df[["stationId", "ts", "PM2.5"]].copy()
        for sn in STREAM_NAMES:
            partial_df[f"pred_{sn}"] = oof_preds[sn]
        partial_df.to_csv(os.path.join(OUT_DIR, "oof_predictions_partial.csv"),
                          index=False, encoding="utf-8-sig")

# ============================================================================
#  12. EVALUATION
# ============================================================================
print(f"\n{'='*80}")
print("EVALUATION")
print(f"{'='*80}")

results_df = df[["stationId", "ts", "PM2.5"]].copy()
for sn in STREAM_NAMES:
    results_df[f"pred_{sn}"] = oof_preds[sn]
results_df.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"),
                  index=False, encoding="utf-8-sig")

# Per-station metrics
valid_mask = ~np.isnan(y_all)

station_metrics = []
for sid in station_ids:
    sm = stationId_vals == sid
    sm_valid = sm & valid_mask
    if sm_valid.sum() < 10:
        continue
    y_s = y_all[sm_valid]
    actual_mean = float(np.mean(y_s))
    tier = sid_tier[sid]
    row = {"station_id": sid, "station_name": sid_name.get(sid, ""),
           "tier": tier, "actual_mean": actual_mean,
           "actual_class": pm_class(actual_mean), "n_rows": int(sm_valid.sum())}

    stream_r2s = {}
    for sn in STREAM_NAMES:
        p_s = oof_preds[sn][sm_valid]
        p_valid = ~np.isnan(p_s)
        if p_valid.sum() >= 3:
            r2 = safe_r2(y_s[p_valid], p_s[p_valid])
            pred_mean = float(np.mean(p_s[p_valid]))
        else:
            r2 = np.nan
            pred_mean = np.nan
        stream_r2s[sn] = r2
        row[f"r2_{sn}"] = r2
        row[f"pred_mean_{sn}"] = pred_mean

    # Oracle: best stream per station
    valid_r2s = {k: v for k, v in stream_r2s.items() if not np.isnan(v)}
    if valid_r2s:
        best_stream = max(valid_r2s, key=valid_r2s.get)
        row["oracle_stream"] = best_stream
        row["oracle_r2"] = valid_r2s[best_stream]
    else:
        row["oracle_stream"] = "full"
        row["oracle_r2"] = np.nan

    station_metrics.append(row)

met_df = pd.DataFrame(station_metrics)
met_df.to_csv(os.path.join(OUT_DIR, "station_metrics.csv"),
              index=False, encoding="utf-8-sig")

# Summary table
print(f"\n{'stream':<12s}  {'mean_stn':>8s}  {'med_stn':>8s}  {'pooled':>8s}  "
      f"{'pos':>4s}  {'h2nh':>5s}")
print("-" * 55)

summary_rows = []
for sn in STREAM_NAMES + ["oracle"]:
    if sn == "oracle":
        # Build oracle predictions
        oracle_pred = np.full(len(df), np.nan)
        for _, row in met_df.iterrows():
            sid = row["station_id"]
            best = row["oracle_stream"]
            sm = stationId_vals == sid
            oracle_pred[sm] = oof_preds[best][sm]
        pred_col = oracle_pred
    else:
        pred_col = oof_preds[sn]

    vm = valid_mask & ~np.isnan(pred_col)
    if vm.sum() < 10:
        continue

    y_v = y_all[vm]
    p_v = pred_col[vm]

    stn_r2s = []
    h2nh = 0
    for sid in station_ids:
        sm = (stationId_vals == sid) & vm
        if sm.sum() < 3: continue
        r2 = safe_r2(y_all[sm], pred_col[sm])
        stn_r2s.append(r2)
        am = float(np.mean(y_all[sm]))
        pm_ = float(np.mean(pred_col[sm]))
        if am >= 35 and pm_ < 35: h2nh += 1

    row = {
        "stream": sn,
        "mean_station_r2": float(np.nanmean(stn_r2s)),
        "median_station_r2": float(np.nanmedian(stn_r2s)),
        "pooled_r2": safe_r2(y_v, p_v),
        "positive_stations": sum(1 for x in stn_r2s if x > 0),
        "high_to_nonhigh": h2nh,
        "n_stations": len(stn_r2s),
    }
    summary_rows.append(row)
    print(f"  {sn:<12s}  {row['mean_station_r2']:>+8.4f}  "
          f"{row['median_station_r2']:>+8.4f}  {row['pooled_r2']:>8.4f}  "
          f"{row['positive_stations']:>4d}  {h2nh:>5d}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "summary.csv"),
                  index=False, encoding="utf-8-sig")

# Oracle stream distribution
print(f"\nOracle stream choices:")
for sn in STREAM_NAMES:
    n = int((met_df["oracle_stream"] == sn).sum())
    if n > 0:
        print(f"  {sn:<12s}: {n} stations")

# Selector gap
oracle_mean_r2 = float(met_df["oracle_r2"].mean())
full_mean_r2 = float(met_df["r2_full"].mean())
print(f"\nSelector gap: oracle={oracle_mean_r2:+.4f}, "
      f"full={full_mean_r2:+.4f}, "
      f"gap={oracle_mean_r2 - full_mean_r2:.4f}")

# Per-tier breakdown
print(f"\nPer-tier breakdown:")
print(f"  {'tier':<4s}  {'n':>3s}  " + "  ".join(f"{s[:5]:>7s}" for s in STREAM_NAMES) + f"  {'oracle':>7s}")
for t in ["t0", "t1", "t2", "t3"]:
    tm = met_df["tier"] == t
    if tm.sum() == 0: continue
    vals = [f"{met_df.loc[tm, f'r2_{s}'].mean():+.3f}" for s in STREAM_NAMES]
    orc = f"{met_df.loc[tm, 'oracle_r2'].mean():+.3f}"
    print(f"  {t:<4s}  {int(tm.sum()):>3d}  " + "  ".join(f"{v:>7s}" for v in vals) + f"  {orc:>7s}")

print(f"\nTotal time: {(time.time()-t0_wall)/60:.1f} minutes")
print(f"Output: {OUT_DIR}")
