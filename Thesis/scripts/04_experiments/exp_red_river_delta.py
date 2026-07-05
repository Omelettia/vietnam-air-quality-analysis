"""
Experiment: Red River Delta PM2.5 Prediction v5h.

v5h: Focus on 12 KK stations in the Red River Delta.
LOSO within 12 delta stations; regional leave-one-out mean as base margin.
Thesis result: can we predict PM2.5 variation within a known polluted region?

Row-level PM2.5 QC is applied through the final merged table and pm25_quality_masks.
Physics features from v5d retained.

Configs:
  - delta_bm: regional LOO mean BM + obs+physics features
  - delta_rfsi: regional LOO mean BM + obs+physics + RFSI
  - delta_rfsi_wind: regional LOO mean BM + obs+physics + wind-aware RFSI
  - oracle_bm: matched diagnostic ceiling (oracle BM + obs+physics + wind-aware RFSI)

Output: analysis/thesis_experiments/delta_v5h_test.csv
"""

import argparse, io, sys, os, warnings, time, glob, zipfile, unicodedata
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler  # kept for potential future use
from sklearn.cluster import KMeans  # used only for initial cluster identification

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None)
parser.add_argument(
    "--target-scale",
    choices=["log", "raw"],
    default="log",
    help="Model target scale. Default 'log' is the thesis canonical setting.",
)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = args.data_dir or REPO_DIR
OUT_DIR = os.path.join(REPO_DIR, "analysis", "thesis_experiments")
META_DIR = os.path.join(DATA_DIR, "data", "stations", "metadata")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_TAG = "delta_v5h" if args.target_scale == "log" else f"delta_v5h_{args.target_scale}"


def safe_to_csv(frame, path, label):
    try:
        frame.to_csv(path, index=False)
        print(f"\n  {label} saved: {path}")
    except PermissionError as e:
        print(f"\n  WARNING: could not save {label} to {path}: {e}")
        print("  Continuing; metrics above remain valid.")


QC_DIR = os.path.join(REPO_DIR, "Thesis", "scripts", "02_processing")
if QC_DIR not in sys.path:
    sys.path.insert(0, QC_DIR)
from pm25_qc import pm25_quality_masks

K_NN = 5
ROLL_DAYS = 30  # Window for temporal satellite statistics

SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}

XGB_BASE = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=50,
    reg_alpha=0.1, reg_lambda=10.0, tree_method="hist",
    device="cuda", random_state=42, n_jobs=-1,
)

CONFIGS = [
    "delta_bm",
    "delta_rfsi",
    "delta_rfsi_wind",
    "delta_rfsi_regime",
    "delta_rfsi_regime_wind",
    "delta_rfsi_near_bm",
    "delta_rfsi_blend_bm",
    "oracle_bm",
]
POST_HOC = []
FOLD_ABBREV = {
    "delta_bm": "DLT",
    "delta_rfsi": "D+R",
    "delta_rfsi_wind": "WND",
    "delta_rfsi_regime": "REG",
    "delta_rfsi_regime_wind": "R+W",
    "delta_rfsi_near_bm": "NNB",
    "delta_rfsi_blend_bm": "BLB",
    "oracle_bm": "ORC",
}

TIER_NAMES = ["t0", "t1", "t2", "t3"]
RFSI_FEATURES = ["PM25_nn_idw", "PM25_nn1", "PM25_nn2", "PM25_nn3"]
RFSI_WIND_FEATURES = RFSI_FEATURES + [
    "PM25_upwind_idw",
    "PM25_downwind_idw",
    "PM25_wind_spread",
    "PM25_neighbor_spread",
]


# =============================================================================
#  HELPERS
# =============================================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def bearing_degrees(lat1, lon1, lat2, lon2):
    y_b = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
    x_b = (np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) -
           np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) *
           np.cos(np.radians(lon2 - lon1)))
    return (np.degrees(np.arctan2(y_b, x_b)) + 360.0) % 360.0


def assign_tier(mean_pm):
    if mean_pm < 10:
        return "t0"
    elif mean_pm < 20:
        return "t1"
    elif mean_pm < 35:
        return "t2"
    return "t3"


def daily_aggregate(station_ids_arr, dates_arr, y_arr, pred_arr, min_hours=18):
    tmp = pd.DataFrame({
        "stationId": station_ids_arr,
        "date": dates_arr,
        "y": y_arr,
        "pred": pred_arr,
    })
    tmp = tmp.dropna(subset=["y", "pred"])
    if tmp.empty:
        return tmp
    daily = tmp.groupby(["stationId", "date"], as_index=False).agg(
        y=("y", "mean"),
        pred=("pred", "mean"),
        n_hours=("y", "size"),
    )
    return daily[daily["n_hours"] >= min_hours].reset_index(drop=True)


def daily_station_r2s(station_ids_arr, dates_arr, y_arr, pred_arr, min_hours=18):
    daily = daily_aggregate(station_ids_arr, dates_arr, y_arr, pred_arr, min_hours=min_hours)
    r2s = []
    for _, g in daily.groupby("stationId"):
        if len(g) >= 10 and g["y"].std() > 1e-9:
            r2s.append(r2_score(g["y"], g["pred"]))
    return daily, np.array(r2s)


def feature_group(feature_name):
    f = feature_name.lower()
    if feature_name == "RH_factor":
        return "meteorology"
    if feature_name == "aod_outer_pm25":
        return "aod_physics"
    if feature_name in RFSI_WIND_FEATURES or f.startswith("pm25_"):
        return "nearby_pm25_rfsi"
    if feature_name in SAT_AOD or feature_name in DAILY_SAT or feature_name in SAT_REGIME:
        if "no2" in f or "so2" in f or "co_" in f or "hcho" in f:
            return "tropomi_gases"
        return "aod_satellite"
    if feature_name in PHYSICS_FEATS:
        if "co" in f or "hcho" in f or "no2" in f or "so2" in f:
            return "gas_physics"
        if "aod" in f or "modis" in f:
            return "aod_physics"
        return "physics_interactions"
    if feature_name in MET or feature_name in PRECIP or feature_name in STABILITY:
        if "wind" in f or feature_name in {"VC", "WS_local"}:
            return "wind_dispersion"
        if "rain" in f or "dry" in f:
            return "rain_dryness"
        return "meteorology"
    if feature_name in TEMPORAL:
        return "time_cycle"
    if feature_name in REGIME_FEATS:
        return "urban_source_proxy"
    if feature_name in SPATIAL:
        return "static_spatial"
    if feature_name in SAT_REGIME_STN:
        return "station_satellite_summary"
    return "other"


# =============================================================================
#  LOAD DATA
# =============================================================================
print("=" * 80)
print("RED RIVER DELTA PM2.5 PREDICTION v5h")
print("=" * 80)

t0_start = time.time()

dataset_path = os.path.join(DATA_DIR, "data/merged/unified_thesis.csv")
df = pd.read_csv(dataset_path, dtype={"stationId": str})
# The merged table holds all 121 stations; restrict to the 40 thesis stations.
_thesis40 = set(pd.read_csv(os.path.join(DATA_DIR,
    "Thesis/results/01_stations/station_selection_final.csv"),
    dtype={"stationId": str})["stationId"])
df = df[df["stationId"].isin(_thesis40)].copy()
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["month"] = df["ts"].dt.month
df["date"] = df["ts"].dt.date
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations "
      f"({time.time()-t0_start:.1f}s)")

# --- Station-level audit notes ---
# These non-regional stations were flagged in exploratory sensor checks. They
# are not removed from the canonical 40-station national diagnostics. The Delta
# experiment below checks whether any flagged station lies inside the region.
STATIONS_REMOVE = {
    "31616865099255512061948816121",  # Da Nang Pham Hung: 12% zeros, 986 flat runs, mean=6.2 vs city ~20
    "30991938797551443885460120607",  # Soc Trang: 3.7% zeros, 206 flat runs, mean=6.7 vs GHAP ~15
    "29098319146067624969113973428",  # Tra Vinh Dong Hai: mean=5.7 vs GHAP ~14
}
STATIONS_FLAG = {
    "28602897318711027016899843809",  # Quang Ninh Nha may tuyen than: placement inside water-sprayed compound
    "31651502905690497791503780869",  # Thai Nguyen: source-impacted (TISCO steelworks)
}

# --- Delta focus: keep only geographic Red River Delta stations ---
DELTA_SIDS = {
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
}
n_before_delta = len(df)
df = df[df["stationId"].isin(DELTA_SIDS)].reset_index(drop=True)
print(f"Delta focus: {n_before_delta:,} → {len(df):,} rows "
      f"({df['stationId'].nunique()} delta stations)")

delta_remove = sorted(set(DELTA_SIDS) & STATIONS_REMOVE)
if delta_remove:
    n_before_quality = len(df)
    df = df[~df["stationId"].isin(delta_remove)].reset_index(drop=True)
    print(f"Delta station audit: removed {n_before_quality - len(df):,} rows "
          f"from {len(delta_remove)} flagged regional stations")
else:
    print("Delta station audit: no regional stations removed; QC is row-level.")

meta = pd.read_csv(os.path.join(DATA_DIR,
                    "Thesis/results/01_stations/station_selection_final.csv"),
                    dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))
sid_region = dict(zip(meta["stationId"], meta["region"]))
sid_lat = dict(zip(meta["stationId"], meta["lat"]))
sid_lon = dict(zip(meta["stationId"], meta["lon"]))
station_ids = sorted(df["stationId"].unique())
n_stn = len(station_ids)

# --- External validation targets (identified early so satellite rolling tables
#     can be computed at their own coordinates, exactly like the KK stations) ---
DELTA_BOX = (20.3, 21.3, 105.5, 107.0)
lcs_meta = pd.read_csv(os.path.join(REPO_DIR,
    "Thesis/results/01_stations/station_selection_lcs.csv"), dtype={"station_id": str})
lcs_passed = lcs_meta[lcs_meta["lcs_flag"] == "pass"].copy()
env_map_all = pd.read_csv(os.path.join(META_DIR, "envisoft_station_map.csv"),
                          dtype={"stationId": str})
env_coord = env_map_all.set_index("stationId")[["latitude", "longitude"]]

val_stations = []
for _, row in lcs_passed.iterrows():
    sid = row["station_id"]
    if sid not in env_coord.index:
        continue
    lat, lon = env_coord.loc[sid, "latitude"], env_coord.loc[sid, "longitude"]
    if DELTA_BOX[0] <= lat <= DELTA_BOX[1] and DELTA_BOX[2] <= lon <= DELTA_BOX[3]:
        val_stations.append({"sid": sid, "name": row["station_name"],
                             "lat": lat, "lon": lon, "type": "LCS"})
val_stations.append({"sid": "US_EMBASSY_HAN", "name": "US Embassy Hanoi",
                     "lat": 21.0219, "lon": 105.8188, "type": "Embassy"})
external_sids = {v["sid"] for v in val_stations}
print(f"External validation targets: {len(val_stations)} "
      f"({sum(1 for v in val_stations if v['type']=='LCS')} LCS + 1 Embassy)")

TARGET = "PM2.5"
y_all = df[TARGET].values
stationId_vals = df["stationId"].values


def target_transform(pm_values):
    pm_values = np.nan_to_num(pm_values, nan=0.0)
    if args.target_scale == "log":
        return np.log1p(pm_values)
    return pm_values.astype(float)


def target_inverse(model_values):
    if args.target_scale == "log":
        return np.expm1(model_values)
    return model_values


def bm_from_pm_mean(pm_mean):
    if args.target_scale == "log":
        return np.log1p(pm_mean)
    return float(pm_mean)

# --- PM2.5 quality filter ---
qc_masks = pm25_quality_masks(df)
n_filtered = int(qc_masks.any(axis=1).sum())
df.loc[qc_masks.any(axis=1), 'PM2.5'] = np.nan
print(
    "PM2.5 quality filter: "
    f"{n_filtered} rows ({100*n_filtered/len(df):.1f}%) "
    f"[zero/neg={int(qc_masks['zero_or_negative'].sum())}, "
    f"flat={int(qc_masks['flatline'].sum())}, "
    f"stuck_low={int(qc_masks['stuck_low'].sum())}, "
    f"high={int(qc_masks['too_high'].sum())}]"
)
y_all = df['PM2.5'].values
y_model = target_transform(y_all)

station_pm_means = df.groupby("stationId")["PM2.5"].mean()
sid_tier = {s: assign_tier(station_pm_means[s]) for s in station_ids}
for t in TIER_NAMES:
    sids_t = [s for s in station_ids if sid_tier[s] == t]
    print(f"  {t}: {len(sids_t)} stations")

global_pm_mean = float(station_pm_means.mean())
bm_global = bm_from_pm_mean(global_pm_mean)

# =============================================================================
#  LOAD GEE DAILY EXPORT + COMPUTE TEMPORAL STATISTICS
# =============================================================================
print(f"\n  Loading GEE daily export...")

zip_candidates = sorted(glob.glob(os.path.join(DATA_DIR, "data", "gee_exports", "last-*.zip")))
if not zip_candidates:
    print("ERROR: No GEE export zip found"); sys.exit(1)
zip_path = zip_candidates[-1]

all_sat = []
with zipfile.ZipFile(zip_path) as z:
    for name in sorted(z.namelist()):
        if not name.endswith(".csv"):
            continue
        with z.open(name) as f:
            chunk = pd.read_csv(f, dtype={"stationId": str})
        all_sat.append(chunk)

sat_long = pd.concat(all_sat, ignore_index=True)
sat_wide = sat_long.pivot_table(
    index=["stationId", "date"], columns="variable",
    values="mean", aggfunc="first"
).reset_index()
sat_wide.columns.name = None
sat_wide["date"] = pd.to_datetime(sat_wide["date"])

model_sids = set(station_ids)
# Keep KK model stations plus external validation targets so the per-station
# rolling/anomaly statistics are computed at each target's own coordinates.
# Per-station groupby keeps target statistics fully separate from KK stations.
sat_keep_sids = model_sids | external_sids
sat_wide = sat_wide[sat_wide["stationId"].isin(sat_keep_sids)].copy()
sat_wide["month"] = sat_wide["date"].dt.month

CLIM_COLS = ["NO2", "SO2", "CO", "HCHO"]
for c in CLIM_COLS:
    if c not in sat_wide.columns:
        sat_wide[c] = np.nan

# Daily anomalies
clim = sat_wide.groupby(["stationId", "month"])[CLIM_COLS].transform("mean")
sat_wide["so2_daily_anom"] = sat_wide["SO2"] - clim["SO2"]
sat_wide["co_daily_anom"] = sat_wide["CO"] - clim["CO"]
sat_wide["no2_daily_anom"] = sat_wide["NO2"] - clim["NO2"]
sat_wide["hcho_daily_anom"] = sat_wide["HCHO"] - clim["HCHO"]

ANOM_COLS = ["so2_daily_anom", "co_daily_anom", "no2_daily_anom",
             "hcho_daily_anom"]

# --- Rolling temporal statistics of TROPOMI (30-day window) ---
print(f"  Computing {ROLL_DAYS}-day rolling TROPOMI statistics...")
sat_wide = sat_wide.sort_values(["stationId", "date"])

tropomi_roll_feats = []
for sid, grp in sat_wide.groupby("stationId"):
    grp = grp.set_index("date").sort_index()
    row_dates = grp.index

    feats_df = pd.DataFrame(index=row_dates)
    feats_df["stationId"] = sid

    # HCHO rolling stats
    hcho = grp["HCHO"]
    feats_df["hcho_30d_mean"] = hcho.rolling(f"{ROLL_DAYS}D", min_periods=5).mean()
    feats_df["hcho_30d_p90"] = hcho.rolling(f"{ROLL_DAYS}D", min_periods=5).quantile(0.9)
    hcho_std = hcho.rolling(f"{ROLL_DAYS}D", min_periods=5).std()
    hcho_mean = hcho.rolling(f"{ROLL_DAYS}D", min_periods=5).mean()
    feats_df["hcho_30d_cv"] = hcho_std / (hcho_mean.abs() + 1e-12)

    # CO rolling stats
    co = grp["CO"]
    feats_df["co_30d_mean"] = co.rolling(f"{ROLL_DAYS}D", min_periods=5).mean()
    feats_df["co_30d_std"] = co.rolling(f"{ROLL_DAYS}D", min_periods=5).std()
    co_q75 = co.rolling(f"{ROLL_DAYS}D", min_periods=5).quantile(0.75)
    co_q25 = co.rolling(f"{ROLL_DAYS}D", min_periods=5).quantile(0.25)
    feats_df["co_30d_iqr"] = co_q75 - co_q25

    feats_df = feats_df.reset_index().rename(columns={"date": "date_trop"})
    tropomi_roll_feats.append(feats_df)

tropomi_roll = pd.concat(tropomi_roll_feats, ignore_index=True)
tropomi_roll["date_merge"] = tropomi_roll["date_trop"].dt.date

TROPOMI_ROLL_COLS = ["hcho_30d_mean", "hcho_30d_p90", "hcho_30d_cv",
                     "co_30d_mean", "co_30d_std", "co_30d_iqr"]

print(f"  TROPOMI rolling features: {len(TROPOMI_ROLL_COLS)} features")

# Prepare merge keys once; the merges themselves are factored into
# merge_satellite_features() so the same satellite enrichment is applied
# identically to the KK model rows and the external validation rows.
sat_wide["date_merge"] = sat_wide["date"].dt.date


def merge_satellite_features(d):
    """Attach GEE daily anomalies, TROPOMI rolling, MODIS daily and MODIS
    rolling features to a dataframe with ['stationId', 'date'] columns.

    Uses the module-level sat_wide / tropomi_roll / modis_tf / modis_roll
    tables, which already cover the KK model stations and the external
    validation targets. This is the single source of truth for satellite
    feature construction, so KK and external rows are enriched identically.
    """
    d = d.merge(sat_wide[["stationId", "date_merge"] + ANOM_COLS],
                left_on=["stationId", "date"],
                right_on=["stationId", "date_merge"], how="left")
    d.drop(columns=["date_merge"], inplace=True)

    d = d.merge(tropomi_roll[["stationId", "date_merge"] + TROPOMI_ROLL_COLS],
                left_on=["stationId", "date"],
                right_on=["stationId", "date_merge"], how="left")
    d.drop(columns=["date_merge"], inplace=True)

    d = d.merge(
        modis_tf[["stationId", "date_m", "modis_aod_7d", "modis_fine_aod_7d"]]
        .rename(columns={"date_m": "date"}),
        on=["stationId", "date"], how="left")

    d = d.merge(modis_roll[["stationId", "date_m"] + MODIS_ROLL_COLS],
                left_on=["stationId", "date"],
                right_on=["stationId", "date_m"], how="left")
    d.drop(columns=["date_m"], inplace=True)
    return d

# =============================================================================
#  LOAD MODIS TEMPORAL FEATURES + ROLLING STATISTICS
# =============================================================================
print("  Loading MODIS temporal features...")
modis_path = os.path.join(META_DIR, "modis_temporal_features.csv")
modis_tf = pd.read_csv(modis_path, dtype={"stationId": str})
modis_tf["date"] = pd.to_datetime(modis_tf["date"])

# Compute rolling AOD statistics (30-day)
print(f"  Computing {ROLL_DAYS}-day rolling MODIS AOD statistics...")
modis_tf = modis_tf.sort_values(["stationId", "date"])

modis_roll_feats = []
for sid, grp in modis_tf.groupby("stationId"):
    grp = grp.set_index("date").sort_index()
    aod = grp["modis_aod_7d"]

    feats_df = pd.DataFrame(index=grp.index)
    feats_df["stationId"] = sid
    feats_df["aod_30d_mean"] = aod.rolling(f"{ROLL_DAYS}D", min_periods=10).mean()
    feats_df["aod_30d_std"] = aod.rolling(f"{ROLL_DAYS}D", min_periods=10).std()
    aod_q75 = aod.rolling(f"{ROLL_DAYS}D", min_periods=10).quantile(0.75)
    aod_q25 = aod.rolling(f"{ROLL_DAYS}D", min_periods=10).quantile(0.25)
    feats_df["aod_30d_iqr"] = aod_q75 - aod_q25
    feats_df["aod_30d_p90"] = aod.rolling(f"{ROLL_DAYS}D", min_periods=10).quantile(0.9)
    aod_std = feats_df["aod_30d_std"]
    aod_mean = feats_df["aod_30d_mean"]
    feats_df["aod_30d_cv"] = aod_std / (aod_mean + 1e-9)

    feats_df = feats_df.reset_index()
    modis_roll_feats.append(feats_df)

modis_roll = pd.concat(modis_roll_feats, ignore_index=True)
modis_roll["date_m"] = modis_roll["date"].dt.date

MODIS_ROLL_COLS = ["aod_30d_mean", "aod_30d_std", "aod_30d_iqr",
                   "aod_30d_p90", "aod_30d_cv"]

# MODIS daily merge key (rolling key date_m already set above)
modis_tf["date_m"] = modis_tf["date"].dt.date

# --- Attach all satellite features to the KK model rows (single source of
#     truth; the same function is reused for the external targets later) ---
df = merge_satellite_features(df)

n_total = len(df)
n_with_hcho = df["hcho_daily_anom"].notna().sum()
n_with_roll = df["hcho_30d_mean"].notna().sum()
n_modis = df["modis_aod_7d"].notna().sum()
n_modis_roll = df["aod_30d_mean"].notna().sum()
print(f"  GEE merged: HCHO anom {n_with_hcho:,}/{n_total:,}, "
      f"rolling stats {n_with_roll:,}/{n_total:,}")
print(f"  MODIS merged: {n_modis:,}/{n_total:,} daily, "
      f"{n_modis_roll:,}/{n_total:,} rolling")

# --- Station-level aggregates of rolling satellite features ---
print("  Computing station-level satellite regime fingerprints...")
ALL_ROLL_COLS = MODIS_ROLL_COLS + TROPOMI_ROLL_COLS
STN_AGG_COLS = []
for col in ALL_ROLL_COLS:
    stn_col = col + "_stn"
    df[stn_col] = df.groupby("stationId")[col].transform("mean")
    STN_AGG_COLS.append(stn_col)
n_stn_agg = df[STN_AGG_COLS[0]].notna().sum()
print(f"  Station-level regime features: {len(STN_AGG_COLS)} "
      f"({n_stn_agg:,}/{n_total:,} non-NaN)")

# =============================================================================
#  BUILDING DENSITY
# =============================================================================
bld_path = os.path.join(META_DIR, "station_building_density.csv")
bld = pd.read_csv(bld_path, dtype={"stationId": str})
bld_map = bld.set_index("stationId")[["building_area_1km"]]
df = df.merge(bld_map, left_on="stationId", right_index=True, how="left")
df["building_area_1km"] = df["building_area_1km"].fillna(0)

# =============================================================================
#  RFSI
# =============================================================================
print("  RFSI setup...")

coords = {s: (sid_lat[s], sid_lon[s]) for s in station_ids}
sid_to_idx = {s: i for i, s in enumerate(station_ids)}
dist_full = np.zeros((n_stn, n_stn))
bearing_full = np.zeros((n_stn, n_stn))
for i in range(n_stn):
    for j in range(i + 1, n_stn):
        lat1, lon1 = coords[station_ids[i]]
        lat2, lon2 = coords[station_ids[j]]
        d = haversine(lat1, lon1, lat2, lon2)
        dist_full[i, j] = d
        dist_full[j, i] = d
        # Bearing from receptor i to candidate/source j. Upwind alignment uses
        # the meteorological wind-from direction at the receptor row.
        y_b = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
        x_b = (np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) -
               np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) *
               np.cos(np.radians(lon2 - lon1)))
        b_ij = (np.degrees(np.arctan2(y_b, x_b)) + 360.0) % 360.0
        b_ji = (b_ij + 180.0) % 360.0
        bearing_full[i, j] = b_ij
        bearing_full[j, i] = b_ji

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


def compute_rfsi(exclude_sid=None, K=5):
    n = len(df)
    pm_nn = np.full((n, K), np.nan)
    d_nn = np.full((n, K), np.nan)
    upwind_idw = np.full(n, np.nan)
    downwind_idw = np.full(n, np.nan)
    wind_spread = np.full(n, np.nan)
    neighbor_spread = np.full(n, np.nan)
    excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
    ts_row_vals = df["ts_row"].values
    wind_from = (np.degrees(np.arctan2(-df["wind_u"].values, -df["wind_v"].values)) + 360.0) % 360.0
    for sid in station_ids:
        si = sid_to_idx[sid]
        mask = stationId_vals == sid
        if not mask.any():
            continue
        ri = np.where(mask)[0]
        tr = ts_row_vals[ri]
        cands = [(j, d) for j, d in neighbor_order[si]
                 if excl is None or j != excl]
        if not cands:
            continue
        ccols = np.array([sid_to_col[station_ids[j]] for j, _ in cands])
        cdists = np.array([d for _, d in cands])
        cbear = np.array([bearing_full[si, j] for j, _ in cands])
        nbr = pm25_mat[np.ix_(tr, ccols)]
        valid = ~np.isnan(nbr)
        cumv = np.cumsum(valid, axis=1)
        for k in range(K):
            reached = cumv >= (k + 1)
            has = reached.any(axis=1)
            if not has.any():
                break
            pos = np.argmax(reached, axis=1)
            ih = np.where(has)[0]
            pm_nn[ri[ih], k] = nbr[ih, pos[has]]
            d_nn[ri[ih], k] = cdists[pos[has]]
        # Wind-aware summaries over all candidate stations available at each row.
        row_from = wind_from[ri]
        diff_up = np.abs(((cbear[None, :] - row_from[:, None] + 180.0) % 360.0) - 180.0)
        diff_down = np.abs(((cbear[None, :] - ((row_from[:, None] + 180.0) % 360.0) + 180.0) % 360.0) - 180.0)
        align_up = np.clip(np.cos(np.radians(diff_up)), 0.0, 1.0)
        align_down = np.clip(np.cos(np.radians(diff_down)), 0.0, 1.0)
        dist_w = 1.0 / np.maximum(cdists, 0.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            wu = align_up * dist_w[None, :] * valid
            wd = align_down * dist_w[None, :] * valid
            upwind_idw[ri] = np.nansum(nbr * wu, axis=1) / np.nansum(wu, axis=1)
            downwind_idw[ri] = np.nansum(nbr * wd, axis=1) / np.nansum(wd, axis=1)
            wind_spread[ri] = upwind_idw[ri] - downwind_idw[ri]
            neighbor_spread[ri] = np.nanmax(nbr, axis=1) - np.nanmin(nbr, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d_nn
        pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)
    return {"PM25_nn_idw": pm_idw, "PM25_nn1": pm_nn[:, 0],
            "PM25_nn2": pm_nn[:, 1], "PM25_nn3": pm_nn[:, 2],
            "PM25_upwind_idw": upwind_idw,
            "PM25_downwind_idw": downwind_idw,
            "PM25_wind_spread": wind_spread,
            "PM25_neighbor_spread": neighbor_spread}


def compute_external_rfsi(v_lat, v_lon, val_ts_idx, wind_u, wind_v, K=5):
    """RFSI for a validation site, matching the internal timestamp logic.

    For each timestamp, nearest-neighbor features use the closest K anchor
    stations that have valid PM2.5 at that timestamp. This mirrors
    compute_rfsi(), instead of fixing the same K geographic anchors for all
    timestamps.
    """
    kk_dists = []
    for sid in station_ids:
        kk_lat, kk_lon = kk_coords[sid]
        kk_dists.append((
            sid,
            haversine(v_lat, v_lon, kk_lat, kk_lon),
            bearing_degrees(v_lat, v_lon, kk_lat, kk_lon),
        ))
    kk_dists.sort(key=lambda x: x[1])

    pm_all = np.full((len(val_ts_idx), len(kk_dists)), np.nan)
    d_all = np.array([d for _, d, _ in kk_dists])
    b_all = np.array([b for _, _, b in kk_dists])
    for k, (kk_sid, _, _) in enumerate(kk_dists):
        if kk_sid in pm25_wide.columns:
            pm_all[:, k] = pm25_wide[kk_sid].reindex(val_ts_idx).values

    valid = ~np.isnan(pm_all)
    cumv = np.cumsum(valid, axis=1)
    pm_nn = np.full((len(val_ts_idx), K), np.nan)
    d_nn = np.full((len(val_ts_idx), K), np.nan)
    for k in range(K):
        reached = cumv >= (k + 1)
        has = reached.any(axis=1)
        if not has.any():
            break
        pos = np.argmax(reached, axis=1)
        ih = np.where(has)[0]
        pm_nn[ih, k] = pm_all[ih, pos[has]]
        d_nn[ih, k] = d_all[pos[has]]

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d_nn
        pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)

    wind_from = (np.degrees(np.arctan2(-wind_u, -wind_v)) + 360.0) % 360.0
    diff_up = np.abs(((b_all[None, :] - wind_from[:, None] + 180.0) % 360.0) - 180.0)
    diff_down = np.abs(
        ((b_all[None, :] - ((wind_from[:, None] + 180.0) % 360.0) + 180.0) % 360.0) -
        180.0
    )
    align_up = np.clip(np.cos(np.radians(diff_up)), 0.0, 1.0)
    align_down = np.clip(np.cos(np.radians(diff_down)), 0.0, 1.0)
    dist_w = 1.0 / np.maximum(d_all, 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        wu = align_up * dist_w[None, :] * valid
        wd = align_down * dist_w[None, :] * valid
        upwind = np.nansum(pm_all * wu, axis=1) / np.nansum(wu, axis=1)
        downwind = np.nansum(pm_all * wd, axis=1) / np.nansum(wd, axis=1)
        wind_spread = upwind - downwind
        neighbor_spread = np.nanmax(pm_all, axis=1) - np.nanmin(pm_all, axis=1)

    return np.column_stack([
        pm_idw, pm_nn[:, 0], pm_nn[:, 1], pm_nn[:, 2],
        upwind, downwind, wind_spread, neighbor_spread,
    ])


# =============================================================================
#  LOAD SATELLITE FEATURES + BUILD SMART_V1
# =============================================================================
print("  Loading satellite feature CSVs...")

no2_feat = pd.read_csv(os.path.join(META_DIR, "station_no2_features.csv"),
                       dtype={"stationId": str})
emit_feat = pd.read_csv(os.path.join(META_DIR, "station_emission_features.csv"),
                        dtype={"stationId": str})
new_sat_feat = pd.read_csv(os.path.join(META_DIR,
                            "station_all_satellite_features.csv"),
                           dtype={"stationId": str})

no2_map = no2_feat.set_index("stationId")
emit_map = emit_feat.set_index("stationId")
new_sat_map = new_sat_feat.set_index("stationId")

NO2_SECTOR_COLS = [f"no2_clim_{d}" for d in SECTOR_NAMES]
df = df.merge(no2_map[["no2_center"] + NO2_SECTOR_COLS],
              left_on="stationId", right_index=True, how="left")

merge_emit = (["ntl_center"] + [f"ntl_clim_{d}" for d in SECTOR_NAMES] +
              ["lst_anom_center"] + [f"lst_anom_clim_{d}" for d in SECTOR_NAMES])
merge_emit = [c for c in merge_emit if c in emit_map.columns]
df = df.merge(emit_map[merge_emit],
              left_on="stationId", right_index=True, how="left")

merge_new = ["fmf_center"]
merge_new = [c for c in merge_new if c in new_sat_map.columns]
df = df.merge(new_sat_map[merge_new],
              left_on="stationId", right_index=True, how="left")

# --- Directional climatology ---
print("  Loading directional climatology...")


def _norm_tok(s):
    return unicodedata.normalize("NFKD", str(s)).encode(
        "ascii", "ignore").decode().lower()


def _tokenize(s):
    return set(_norm_tok(s).replace("-", " ").replace(",", " ").split())


no2_clim_csv = os.path.join(META_DIR, "no2_directional_clim.csv")
no2_dir_raw = pd.read_csv(no2_clim_csv, dtype={"stationId": str})
no2_dir_names = no2_dir_raw.groupby("stationId")["name"].first()

id_map = {}
audit_csv = os.path.join(REPO_DIR, "Thesis", "results", "06_data_quality",
                         "no2_station_id_mapping_audit.csv")
if os.path.exists(audit_csv):
    audit = pd.read_csv(audit_csv, dtype=str)
    audit = audit[audit["status"].eq("ok")]
    id_map = dict(zip(audit["short_id"], audit["matched_stationId"]))
    print(f"    Using audited station ID mapping: {audit_csv}")
else:
    print("    WARNING: audited NO2 station ID mapping not found; rebuilding with minimum token score")
    low_score_matches = []
    skipped_non_model = []
    for short_id, no2_name in no2_dir_names.items():
        no2_tokens = _tokenize(no2_name)
        best_score, best_long_id = -1, None
        for _, row in meta.iterrows():
            meta_tokens = _tokenize(row["station_name"])
            score = len(no2_tokens & meta_tokens)
            if score > best_score:
                best_score = score
                best_long_id = row["stationId"]
        if best_score < 3 and best_long_id not in model_sids:
            skipped_non_model.append((short_id, no2_name, best_long_id, best_score))
            continue
        if best_score < 3:
            low_score_matches.append((short_id, no2_name, best_long_id, best_score))
            continue
        id_map[short_id] = best_long_id
    if low_score_matches:
        print("ERROR: low-confidence directional station ID matches:")
        for short_id, no2_name, best_long_id, best_score in low_score_matches:
            print(f"  {short_id}: score={best_score}, best={best_long_id}, name={no2_name}")
        sys.exit("Fix/audit station ID mapping before continuing")
    if skipped_non_model:
        print(f"    skipped {len(skipped_non_model)} low-confidence non-Delta directional mappings")

missing_model_sids = sorted(set(model_sids) - set(id_map.values()))
if missing_model_sids:
    print("ERROR: directional climatology is missing mapped Delta station IDs:")
    for sid in missing_model_sids:
        print(f"  {sid}")
    sys.exit("Fix/audit station ID mapping before continuing")


def _load_dir_clim(csv_path, value_col="mean"):
    raw = pd.read_csv(csv_path, dtype={"stationId": str})
    raw_short_ids = set(raw["stationId"].dropna().astype(str).unique())
    missing = sorted(raw_short_ids - set(id_map))
    if missing:
        raw = raw[~raw["stationId"].isin(missing)].copy()
    raw["stationId"] = raw["stationId"].map(id_map)
    raw = raw.dropna(subset=["stationId"])
    clim_df = raw.groupby(["stationId", "direction"])[value_col].mean() \
                  .reset_index()
    sectors, centers = {}, {}
    for sid in station_ids:
        sf = clim_df[clim_df["stationId"] == sid]
        sec = np.full(8, np.nan)
        for di, d in enumerate(SECTOR_NAMES):
            vals = sf[sf["direction"] == d][value_col]
            if len(vals) > 0:
                sec[di] = float(vals.iloc[0])
        sectors[sid] = sec
        cvals = sf[sf["direction"] == "C"][value_col]
        centers[sid] = float(cvals.iloc[0]) if len(cvals) > 0 else np.nan
    return sectors, centers


def _resolve_path(filename):
    p = os.path.join(META_DIR, filename)
    if not os.path.exists(p):
        p = os.path.join(DATA_DIR, filename)
    return p


station_so2_sectors, _ = _load_dir_clim(
    _resolve_path("tropomi_so2_directional.csv"), "mean")
station_co_sectors, _ = _load_dir_clim(
    _resolve_path("tropomi_co_directional.csv"), "mean")

# --- Build smart_v1 ---
print("  Building smart_v1 + sector features...")

station_no2_sectors, station_ntl_sectors, station_lst_sectors = {}, {}, {}
for sid in station_ids:
    if sid in no2_map.index:
        station_no2_sectors[sid] = np.array(
            [no2_map.loc[sid, f"no2_clim_{d}"] for d in SECTOR_NAMES], dtype=float)
    else:
        station_no2_sectors[sid] = np.full(8, np.nan)
    if sid in emit_map.index:
        station_ntl_sectors[sid] = np.array(
            [emit_map.loc[sid, f"ntl_clim_{d}"] for d in SECTOR_NAMES], dtype=float)
        station_lst_sectors[sid] = np.array(
            [emit_map.loc[sid, f"lst_anom_clim_{d}"] for d in SECTOR_NAMES], dtype=float)
    else:
        station_ntl_sectors[sid] = np.full(8, np.nan)
        station_lst_sectors[sid] = np.full(8, np.nan)

wd_from = np.degrees(np.arctan2(-df["wind_u"].values, -df["wind_v"].values)) % 360
sector_idx = ((wd_from + 22.5) / 45).astype(int) % 8
month_vals = df["month"].values
season_vals = np.array([SEASON_MAP[m] for m in month_vals])

all_no2_sec = np.array([station_no2_sectors[s] for s in station_ids])
all_ntl_sec = np.array([station_ntl_sectors[s] for s in station_ids])
all_lst_sec = np.array([station_lst_sectors[s] for s in station_ids])

no2_center_all = np.array([
    no2_map.loc[s, "no2_center"] if s in no2_map.index else np.nan
    for s in station_ids])
ntl_center_all = np.array([
    emit_map.loc[s, "ntl_center"] if s in emit_map.index else np.nan
    for s in station_ids])
lst_center_all = np.array([
    emit_map.loc[s, "lst_anom_center"] if s in emit_map.index else np.nan
    for s in station_ids])
fmf_center_all = np.array([
    new_sat_map.loc[s, "fmf_center"] if s in new_sat_map.index else np.nan
    for s in station_ids])


def _lohi(sec_arr, center_arr):
    combined = np.concatenate([sec_arr.ravel(), center_arr])
    return float(np.nanmin(combined)), float(np.nanmax(combined))


no2_lo, no2_hi = _lohi(all_no2_sec, no2_center_all)
ntl_lo, ntl_hi = _lohi(all_ntl_sec, ntl_center_all)
lst_lo, lst_hi = _lohi(all_lst_sec, lst_center_all)


def norm01(v, lo, hi):
    if hi - lo < 1e-12:
        return 0.0
    return float((v - lo) / (hi - lo)) if not np.isnan(v) else 0.0


station_smart_v1_sec, station_smart_v1_cen = {}, {}
for si, sid in enumerate(station_ids):
    fmf = fmf_center_all[si]
    if np.isnan(fmf):
        fmf = 0.5
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
    if not mask.any():
        continue
    idx = np.where(mask)[0]
    smart_v1_upwind[idx] = station_smart_v1_sec[sid][sector_idx[idx]]
df["smart_v1_upwind"] = smart_v1_upwind

# --- SO2 / CO upwind ---
so2_upwind_vals = np.zeros(len(df))
co_upwind_vals = np.zeros(len(df))
for sid in station_ids:
    mask = stationId_vals == sid
    if not mask.any():
        continue
    idx = np.where(mask)[0]
    so2_upwind_vals[idx] = station_so2_sectors[sid][sector_idx[idx]]
    co_upwind_vals[idx] = station_co_sectors[sid][sector_idx[idx]]

so2_upwind_vals = np.nan_to_num(so2_upwind_vals, nan=0.0)
co_upwind_vals = np.nan_to_num(co_upwind_vals, nan=0.0)
df["so2_upwind"] = so2_upwind_vals
df["co_upwind"] = co_upwind_vals

# --- Fire upwind ---
fire_csv_path = os.path.join(META_DIR, "fire_counts_directional.csv")
if not os.path.exists(fire_csv_path):
    fire_csv_path = os.path.join(DATA_DIR, "fire_counts_directional.csv")

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
    station_fire_dir_season[sid] = lookup

fire_upwind = np.zeros(len(df))
for sid in station_ids:
    mask = stationId_vals == sid
    if not mask.any():
        continue
    idx = np.where(mask)[0]
    lookup = station_fire_dir_season.get(sid, {})
    for i in idx:
        fire_upwind[i] = lookup.get((sector_idx[i], season_vals[i]), 0.0)
df["fire_upwind"] = fire_upwind

for c in ["no2_center", "ntl_center", "smart_v1_center", "smart_v1_upwind",
          "so2_upwind", "co_upwind", "fire_upwind"]:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# =============================================================================
#  FEATURE CLEANING
# =============================================================================
print(f"\n{'='*80}")
print("FEATURE CLEANING")
print(f"{'='*80}")


def derive_obs_features(d, label="rows"):
    """Feature cleaning + derived/physics features for the FEAT_OBS set.

    Applied identically to the KK model rows and to the external validation
    rows so both share one source of truth. Does not build the emission/regime
    interactions (so2_upwind_x_VC_inv, urban_*), which are not part of FEAT_OBS.
    """
    rh_bad = d['Humidity_final'] < 5
    d.loc[rh_bad, 'Humidity_final'] = np.nan
    prs_bad = (d['Pressure_final'] < 950) | (d['Pressure_final'] > 1040)
    d.loc[prs_bad, 'Pressure_final'] = np.nan
    temp_bad = (d['Temperature_final'] < 0) | (d['Temperature_final'] > 50)
    d.loc[temp_bad, 'Temperature_final'] = np.nan
    pblh_bad = (d['PBLH'] < 0) | (d['PBLH'] > 6000)
    d.loc[pblh_bad, 'PBLH'] = np.nan
    ae_bad = d['AE'].abs() > 5
    d.loc[ae_bad, 'AE'] = np.nan
    aot_bad = d['AOT_ffill_48h'] > 5
    d.loc[aot_bad, 'AOT_ffill_48h'] = np.nan
    aot2_bad = d['AOT_outer_mean'] > 5
    d.loc[aot2_bad, 'AOT_outer_mean'] = np.nan
    ws_suspect = (d['WS_local'] == 0) & (np.sqrt(d['wind_u']**2 + d['wind_v']**2) > 1.0)
    d.loc[ws_suspect, 'WS_local'] = np.nan

    any_cleaned = rh_bad | prs_bad | temp_bad | pblh_bad | ws_suspect
    print(f"  Feature cleaning ({label}): {int(any_cleaned.sum()):>7,} rows "
          f"({100*any_cleaned.sum()/max(len(d), 1):.1f}%)")

    ws_arr = np.sqrt(d["wind_u"].values**2 + d["wind_v"].values**2)
    pblh_f = d["PBLH"].fillna(200).values

    d["RH_factor"] = 1.0 / (1.0 - (d["Humidity_final"] / 100.0).clip(upper=0.95))
    d["VC"] = pblh_f * np.clip(ws_arr, 0.1, None)

    aot_outer_c = d["AOT_outer_mean"].fillna(0).values
    rh_frac_c = (d["Humidity_final"] / 100.0).clip(0, 0.95).values
    f_rh_c = 1.0 / (1.0 - rh_frac_c)
    d["aod_outer_pm25"] = aot_outer_c / (pblh_f + 100.0) / f_rh_c

    d['PBLH_min_24h'] = d.groupby('stationId')['PBLH'].transform(
        lambda x: x.rolling(24, min_periods=1).min())
    stag_col = ((d['PBLH'] < 500) & (d['WS_local'].fillna(0) < 2)).astype(float)
    d['stagnation_hours_12h'] = stag_col.groupby(d['stationId']).rolling(
        12, min_periods=1).sum().reset_index(level=0, drop=True)

    pblh_km = d["PBLH"].fillna(200).clip(lower=50) / 1000.0
    rh_safe = d["RH_factor"].fillna(1.0).clip(lower=1.0)

    d["aod_surface"]    = d["AOT_inner_mean"].fillna(0) / (pblh_km + 0.1)
    d["aod_dry"]        = d["AOT_inner_mean"].fillna(0) / rh_safe
    d["co_surface"]     = d["co_30d_mean"].fillna(0) / (pblh_km + 0.1)
    d["hcho_surface"]   = d["hcho_30d_mean"].fillna(0) / (pblh_km + 0.1)
    d["no2_surface"]    = d["no2_daily_anom"].fillna(0) / (pblh_km + 0.1)
    d["so2_surface"]    = d["so2_daily_anom"].fillna(0) / (pblh_km + 0.1)
    d["combustion_aod"] = d["co_30d_mean"].fillna(0) * d["AOT_fine"].fillna(0)
    d["secondary_form"] = d["hcho_30d_mean"].fillna(0) * (d["Humidity_final"].fillna(50) / 100)
    d["modis_surface"]  = d["modis_aod_7d"].fillna(0) / (pblh_km + 0.1)
    d["stagnant_aod"]   = d["AOT_inner_mean"].fillna(0) * d["stagnation_hours_12h"].fillna(0)
    d["stagnant_co"]    = d["co_30d_mean"].fillna(0) * d["stagnation_hours_12h"].fillna(0)
    d["aod_anomaly"]    = d["AOT_inner_mean"].fillna(0) - d["aod_30d_mean"].fillna(0)
    return d


df = derive_obs_features(df, label="KK model rows")

# Emission interaction (FEAT_BASE only; needs the KK-only directional regime
# features, so it is built here rather than inside derive_obs_features).
_ws_arr = np.sqrt(df["wind_u"].values**2 + df["wind_v"].values**2)
_pblh_f = df["PBLH"].fillna(200).values
_vc_inv_clean = 1.0 / (np.clip(_pblh_f, 50, None) * np.clip(_ws_arr, 0.1, None) + 1)
df["so2_upwind_x_VC_inv"] = df["so2_upwind"].values * _vc_inv_clean

# pblh_km kept for the regime interactions below (urban_pblh_inv).
pblh_km = df["PBLH"].fillna(200).clip(lower=50) / 1000.0


def _norm01_col(name):
    s = pd.to_numeric(df[name], errors="coerce").astype(float)
    lo, hi = np.nanquantile(s, [0.05, 0.95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return ((s - lo) / (hi - lo)).clip(0, 1).fillna(0)


# Observable regional regime descriptors. These are not learned station IDs:
# they are map-available proxies for urban density and combustion/industrial
# influence, used as smooth interactions inside the regional model.
building_n = np.log1p(pd.to_numeric(df["building_area_1km"], errors="coerce").fillna(0))
building_n = (building_n - building_n.quantile(0.05)) / max(
    building_n.quantile(0.95) - building_n.quantile(0.05), 1e-6)
building_n = building_n.clip(0, 1).fillna(0)
no2_n = _norm01_col("no2_center") if "no2_center" in df.columns else pd.Series(0.0, index=df.index)
ntl_n = _norm01_col("ntl_center") if "ntl_center" in df.columns else pd.Series(0.0, index=df.index)
smart_n = _norm01_col("smart_v1_center") if "smart_v1_center" in df.columns else pd.Series(0.0, index=df.index)
df["urban_score"] = (0.45 * building_n + 0.35 * ntl_n + 0.20 * no2_n).clip(0, 1)
df["industrial_score"] = (0.55 * smart_n + 0.30 * no2_n + 0.15 * building_n).clip(0, 1)
df["peri_rural_score"] = (1.0 - df["urban_score"]).clip(0, 1)
df["urban_aod"] = df["urban_score"] * df["AOT_inner_mean"].fillna(0)
df["urban_stagnation"] = df["urban_score"] * df["stagnation_hours_12h"].fillna(0)
df["urban_pblh_inv"] = df["urban_score"] / (pblh_km + 0.1)
df["industrial_vent"] = df["industrial_score"] / (df["VC"].fillna(df["VC"].median()).clip(lower=0.1))

print("  Derived features recomputed.")

# =============================================================================
#  FEATURE SETS
# =============================================================================

# A. Satellite AOD (9)
SAT_AOD = ["AOT_ffill_48h", "AOT_outer_mean", "AOT_inner_mean", "AOT_fine",
           "RF", "AE", "AOT_spatial_std", "AOT_rolling_mean_24h",
           "hours_since_valid_AOT"]

# B. Daily satellite (6)
DAILY_SAT = ["modis_aod_7d", "modis_fine_aod_7d",
             "no2_daily_anom", "co_daily_anom", "so2_daily_anom",
             "hcho_daily_anom"]

# C. Meteorology (10)
MET = ["PBLH", "VC", "wind_u", "wind_v", "WS_local",
       "Temperature_final", "Humidity_final", "Pressure_final",
       "dT_6h", "dRH_6h"]

# D. Precipitation (3)
PRECIP = ["rain_days_7d", "consecutive_dry_days", "hrs_since_rain"]

# E. Temporal (4)
TEMPORAL = ["hour_sin", "hour_cos", "month_sin", "month_cos"]

# F. Wind-emission interaction (5)
EMISSION = ["smart_v1_center", "smart_v1_upwind",
            "so2_upwind_x_VC_inv", "co_upwind", "fire_upwind"]

# G. Spatial context (5)
SPATIAL = ["building_area_1km", "elevation_m", "latitude"]

# H. Atmospheric stability (2)
STABILITY = ["PBLH_min_24h", "stagnation_hours_12h"]

# I. Satellite temporal regime indicators — daily rolling (11)
SAT_REGIME = MODIS_ROLL_COLS + TROPOMI_ROLL_COLS

# J. NEW: Station-level regime fingerprints — stable aggregates (11)
SAT_REGIME_STN = STN_AGG_COLS

FEAT_BASE = [f for f in (SAT_AOD + DAILY_SAT + MET + PRECIP + TEMPORAL +
              EMISSION + SPATIAL + STABILITY + SAT_REGIME + SAT_REGIME_STN)
             if f in df.columns]

# K. Physics-informed interactions (v5d) — observation-level (12)
PHYSICS_FEATS = ["aod_surface", "aod_dry", "co_surface", "hcho_surface",
                 "no2_surface", "so2_surface", "combustion_aod", "secondary_form",
                 "modis_surface", "stagnant_aod", "stagnant_co", "aod_anomaly"]

# M. Observable regional regime features (urban/industrial/peri-rural)
REGIME_FEATS = ["building_area_1km", "no2_center", "ntl_center", "smart_v1_center",
                "urban_score", "industrial_score", "peri_rural_score",
                "urban_aod", "urban_stagnation", "urban_pblh_inv",
                "industrial_vent"]

# L. Observation-only features (vary per hour/day, no station identity)
OBS_DERIVED = ["RH_factor", "aod_outer_pm25"]
FEAT_OBS = [f for f in (SAT_AOD + DAILY_SAT + MET + PRECIP + TEMPORAL +
            STABILITY + SAT_REGIME + OBS_DERIVED + PHYSICS_FEATS) if f in df.columns]
FEAT_OBS_RFSI = FEAT_OBS + RFSI_FEATURES
FEAT_OBS_WIND_RFSI = FEAT_OBS + RFSI_WIND_FEATURES
FEAT_OBS_REGIME = [f for f in (FEAT_OBS + REGIME_FEATS) if f in df.columns]
FEAT_OBS_REGIME_RFSI = FEAT_OBS_REGIME + RFSI_FEATURES
FEAT_OBS_REGIME_WIND_RFSI = FEAT_OBS_REGIME + RFSI_WIND_FEATURES

FEAT_BASE_PHYS = [f for f in (FEAT_BASE + PHYSICS_FEATS) if f in df.columns]
FEAT_BASE_PHYS = list(dict.fromkeys(FEAT_BASE_PHYS))
FEAT_ALL = FEAT_BASE_PHYS + RFSI_FEATURES

n_base = len(FEAT_BASE_PHYS)
n_obs = len(FEAT_OBS)
n_phys = sum(1 for f in PHYSICS_FEATS if f in df.columns)
n_feat = len(FEAT_ALL)
print(f"\n  Features: all={n_base}f + RFSI={len(RFSI_FEATURES)}f = {n_feat}f")
print(f"  Obs+physics: {n_obs}f (incl {n_phys} physics features, no station-level)")

base_arr = df[FEAT_BASE_PHYS].values.astype(np.float32)
base_arr = np.nan_to_num(base_arr, nan=np.nan)

obs_arr = df[FEAT_OBS].values.astype(np.float32)
obs_arr = np.nan_to_num(obs_arr, nan=np.nan)

# Monotonicity constraints
MONO_DICT = {
    "smart_v1_upwind": 1, "so2_upwind_x_VC_inv": 1,
    "co_upwind": 1, "fire_upwind": 1,
    "VC": -1, "PBLH": -1, "WS_local": -1,
    "PM25_nn_idw": 1, "PM25_nn1": 1,
    "PM25_nn2": 1, "PM25_nn3": 1,
    "PM25_upwind_idw": 1, "PM25_downwind_idw": 1,
    "modis_aod_7d": 1, "AOT_ffill_48h": 1,
    "aod_30d_mean": 1, "aod_30d_p90": 1,
    "hcho_30d_mean": 1, "co_30d_mean": 1,
    "aod_30d_mean_stn": 1, "aod_30d_p90_stn": 1,
    "hcho_30d_mean_stn": 1, "co_30d_mean_stn": 1,
    "aod_surface": 1, "aod_dry": 1, "co_surface": 1, "hcho_surface": 1,
    "combustion_aod": 1, "modis_surface": 1,
    "stagnant_aod": 1, "stagnant_co": 1,
}

mono_str = tuple(MONO_DICT.get(f, 0) for f in FEAT_ALL)
mono_str_base = tuple(MONO_DICT.get(f, 0) for f in FEAT_BASE_PHYS)
mono_str_obs = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS)
mono_str_obs_rfsi = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_RFSI)
mono_str_obs_wind_rfsi = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_WIND_RFSI)
mono_str_obs_regime_rfsi = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_REGIME_RFSI)
mono_str_obs_regime_wind_rfsi = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_REGIME_WIND_RFSI)

CONFIG_SETUP = {
    "delta_bm":                ("obs", "none",  "regional"),
    "delta_rfsi":              ("obs", "basic", "regional"),
    "delta_rfsi_wind":         ("obs", "wind",  "regional"),
    "delta_rfsi_regime":       ("regime", "basic", "regional"),
    "delta_rfsi_regime_wind":  ("regime", "wind",  "regional"),
    "delta_rfsi_near_bm":      ("obs", "basic", "nearest"),
    "delta_rfsi_blend_bm":     ("obs", "basic", "blend"),
    "oracle_bm":               ("obs", "wind", "oracle"),
}

sid_to_int = {s: i for i, s in enumerate(station_ids)}
row_sid_idx = np.array([sid_to_int[s] for s in stationId_vals])

# =============================================================================
#  BASE MARGINS
# =============================================================================
print(f"\n{'='*80}")
print("BASE MARGIN SETUP")
print(f"{'='*80}")

station_monthly_pm = df.groupby(['stationId', 'month'])['PM2.5'].mean()

bm_oracle = np.full(len(df), bm_global)
for sid in station_ids:
    sid_mask = stationId_vals == sid
    months = df['month'].values[sid_mask]
    for m in np.unique(months):
        try:
            val = station_monthly_pm.loc[(sid, int(m))]
        except KeyError:
            val = np.nan
        if np.isnan(val):
            val = global_pm_mean
        m_mask = sid_mask & (df['month'].values == m)
        bm_oracle[m_mask] = bm_from_pm_mean(val)

print(f"  Target scale: {args.target_scale}")
print(f"  Global BM:   {bm_global:.4f} ({args.target_scale} scale of {global_pm_mean:.1f} µg/m³)")
print(f"  Oracle BM:   mean={bm_oracle.mean():.4f}, std={bm_oracle.std():.4f}")

# Regional LOO BM diagnostic
print(f"\n  Delta regional BM (LOO on {n_stn} stations):")
regional_mean = float(station_pm_means.mean())
print(f"    Regional mean: {regional_mean:.1f} µg/m³")
reg_pred_loo = np.zeros(n_stn)
for si, sid in enumerate(station_ids):
    peers = [s for s in station_ids if s != sid]
    reg_pred_loo[si] = float(np.mean([station_pm_means[s] for s in peers]))
stn_actual = np.array([float(station_pm_means[sid]) for sid in station_ids])
reg_loo_r2 = r2_score(stn_actual, reg_pred_loo)
reg_loo_rmse = np.sqrt(np.mean((stn_actual - reg_pred_loo)**2))
print(f"    LOO R²: {reg_loo_r2:.3f}, RMSE: {reg_loo_rmse:.1f} µg/m³")
print(f"    {'Station':<35s} {'tier':>4s} {'actual':>7s} {'pred':>7s} {'err':>7s}")
for si, sid in enumerate(station_ids):
    actual = stn_actual[si]
    pred = reg_pred_loo[si]
    err = pred - actual
    print(f"      {sid_name.get(sid, sid)[:35]:35s} "
          f"{sid_tier[sid]:>4s} "
          f"{actual:7.1f} {pred:7.1f} {err:+7.1f}")

bm_map = {
    "oracle_bm": bm_oracle,
}

# =============================================================================
#  LOSO (12 delta folds)
# =============================================================================
n_configs = len(CONFIGS)
print(f"\n{'='*80}")
print(f"LOSO: {n_stn} folds, {n_configs} configs")
print(f"{'='*80}\n")

pred_all = {c: np.full(len(df), np.nan) for c in CONFIGS}

for fold_i, held_sid in enumerate(station_ids):
    nm = sid_name.get(held_sid, held_sid)[:35]
    held_tier = sid_tier[held_sid]
    mask_test = stationId_vals == held_sid
    n_test = mask_test.sum()
    if n_test < 10:
        print(f"  [{fold_i+1:2d}/{n_stn}] {nm:35s} | SKIP (n={n_test})")
        continue

    t_fold = time.time()
    pm_val = float(station_pm_means[held_sid])

    # --- RFSI (exclude held-out) ---
    rfsi_fold = compute_rfsi(exclude_sid=held_sid, K=K_NN)
    rfsi_arr = np.column_stack([rfsi_fold[c] for c in RFSI_FEATURES])
    rfsi_wind_arr = np.column_stack([rfsi_fold[c] for c in RFSI_WIND_FEATURES])

    test_idx = np.where(mask_test)[0]
    train_mask = (stationId_vals != held_sid) & ~np.isnan(y_all)
    train_idx = np.where(train_mask)[0]

    # Regional LOO BM: mean PM2.5 of all other delta stations
    train_sids = [s for s in station_ids if s != held_sid]
    regional_loo_mean = float(np.mean([station_pm_means[s] for s in train_sids]))
    nearest_sids = sorted(
        train_sids,
        key=lambda s: dist_full[sid_to_idx[held_sid], sid_to_idx[s]]
    )[: min(3, len(train_sids))]
    nearest_dist = np.array([
        max(dist_full[sid_to_idx[held_sid], sid_to_idx[s]], 0.5)
        for s in nearest_sids
    ], dtype=float)
    nearest_w = 1.0 / nearest_dist
    nearest_w = nearest_w / nearest_w.sum()
    nearest_mean = float(np.dot(nearest_w, [station_pm_means[s] for s in nearest_sids]))
    blend_mean = 0.55 * regional_loo_mean + 0.45 * nearest_mean
    regional_bm_val = bm_from_pm_mean(regional_loo_mean)
    nearest_bm_val = bm_from_pm_mean(nearest_mean)
    blend_bm_val = bm_from_pm_mean(blend_mean)

    # Train separate model per config
    for cname in CONFIGS:
        feat_set, rfsi_mode, bm_type = CONFIG_SETUP[cname]
        if feat_set == "obs":
            X_base_cfg = obs_arr
            mc = mono_str_obs
        elif feat_set == "regime":
            X_base_cfg = df[FEAT_OBS_REGIME].values.astype(np.float32)
            mc = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_REGIME)
        else:
            X_base_cfg = base_arr
            mc = mono_str_base

        if rfsi_mode == "basic":
            X_cfg = np.hstack([X_base_cfg, rfsi_arr])
            if feat_set == "obs":
                mc = mono_str_obs_rfsi
            elif feat_set == "regime":
                mc = mono_str_obs_regime_rfsi
            else:
                mc = mono_str
        elif rfsi_mode == "wind":
            X_cfg = np.hstack([X_base_cfg, rfsi_wind_arr])
            if feat_set == "obs":
                mc = mono_str_obs_wind_rfsi
            elif feat_set == "regime":
                mc = mono_str_obs_regime_wind_rfsi
            else:
                mc = tuple(MONO_DICT.get(f, 0) for f in (FEAT_BASE_PHYS + RFSI_WIND_FEATURES))
        else:
            X_cfg = X_base_cfg

        if bm_type == "regional":
            bm_train = np.full(len(train_idx), regional_bm_val)
            bm_test = np.full(n_test, regional_bm_val)
        elif bm_type == "nearest":
            bm_train = np.full(len(train_idx), nearest_bm_val)
            bm_test = np.full(n_test, nearest_bm_val)
        elif bm_type == "blend":
            bm_train = np.full(len(train_idx), blend_bm_val)
            bm_test = np.full(n_test, blend_bm_val)
        elif bm_type == "oracle":
            bm_train = bm_map["oracle_bm"][train_idx]
            bm_test = bm_map["oracle_bm"][mask_test]
        else:
            bm_train = np.full(len(train_idx), bm_global)
            bm_test = np.full(n_test, bm_global)

        y_tr = y_model[train_idx] - bm_train
        params = {**XGB_BASE, "monotone_constraints": mc}
        m = xgb.XGBRegressor(**params)
        m.fit(X_cfg[train_idx], y_tr)
        pred_res = m.predict(X_cfg[test_idx])
        pred_all[cname][mask_test] = np.clip(
            target_inverse(pred_res + bm_test), 0, None)

    # --- Fold summary ---
    fold_time = time.time() - t_fold
    remaining = fold_time * (n_stn - fold_i - 1)

    valid_y = ~np.isnan(y_all[mask_test])
    fold_r2s = {}
    for cname in CONFIGS:
        p = pred_all[cname][mask_test]
        v = valid_y & ~np.isnan(p)
        if v.sum() > 0:
            fold_r2s[cname] = r2_score(y_all[mask_test][v], p[v])
        else:
            fold_r2s[cname] = np.nan

    best = max(CONFIGS, key=lambda c: fold_r2s.get(c, -999))

    print(f"  [{fold_i+1:2d}/{n_stn}] {nm:35s} {held_tier} "
          f"pm={pm_val:5.1f} rbm={regional_loo_mean:5.1f} | "
          f"DLT={fold_r2s['delta_bm']:+.3f} "
          f"D+R={fold_r2s['delta_rfsi']:+.3f} "
          f"R+W={fold_r2s['delta_rfsi_regime_wind']:+.3f} "
          f"ORC={fold_r2s['oracle_bm']:+.3f} "
          f"best={FOLD_ABBREV.get(best, best[:3].upper())} "
          f"({fold_time:.0f}s, ETA {remaining/60:.0f}m)")

loso_time = time.time() - t0_start
print(f"\n  LOSO done: {loso_time:.0f}s")

# =============================================================================
#  RESULTS
# =============================================================================
print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")

ALL_CONFIGS = CONFIGS

# --- Pooled R2 ---
print(f"\n  Pooled R2_hourly:")
print(f"  {'Config':<18s} {'R2_h':>8s} {'RMSE':>7s} {'MAE':>7s} {'Bias':>8s}")
print("  " + "-" * 55)
for cname in ALL_CONFIGS:
    p = pred_all[cname]
    valid = ~np.isnan(p) & ~np.isnan(y_all)
    r2 = r2_score(y_all[valid], p[valid])
    rmse = np.sqrt(mean_squared_error(y_all[valid], p[valid]))
    mae = mean_absolute_error(y_all[valid], p[valid])
    bias = float(p[valid].mean() - y_all[valid].mean())
    print(f"  {cname:<18s} {r2:+8.4f} {rmse:7.2f} {mae:7.2f} {bias:+8.3f}")

# --- Per-station mean R2 ---
print(f"\n  Per-station mean R2:")
print(f"  {'Config':<18s} {'mean_R2':>8s} {'median':>8s} {'%>0':>6s}")
print("  " + "-" * 45)
for cname in ALL_CONFIGS:
    stn_r2s = []
    for sid in station_ids:
        mask = stationId_vals == sid
        p = pred_all[cname][mask]
        valid = ~np.isnan(p) & ~np.isnan(y_all[mask])
        if valid.sum() >= 10:
            stn_r2s.append(r2_score(y_all[mask][valid], p[valid]))
    arr = np.array(stn_r2s)
    pct_pos = 100 * (arr > 0).mean()
    print(f"  {cname:<18s} {arr.mean():+8.4f} {np.median(arr):+8.4f} "
          f"{pct_pos:5.0f}%")

# --- Daily aggregation ---
print(f"\n  Daily station-day R2 (>=18 hourly observations/day):")
print(f"  {'Config':<18s} {'R2_day':>8s} {'RMSE':>7s} {'mean_R2':>8s} "
      f"{'median':>8s} {'%>0':>6s} {'n_days':>7s}")
print("  " + "-" * 72)
date_vals = pd.to_datetime(df["date"]).dt.date.values
for cname in ALL_CONFIGS:
    daily, daily_r2s = daily_station_r2s(
        stationId_vals, date_vals, y_all, pred_all[cname], min_hours=18)
    if len(daily) == 0 or len(daily_r2s) == 0:
        continue
    pooled_day_r2 = r2_score(daily["y"], daily["pred"])
    pooled_day_rmse = np.sqrt(mean_squared_error(daily["y"], daily["pred"]))
    pct_pos = 100 * (daily_r2s > 0).mean()
    print(f"  {cname:<18s} {pooled_day_r2:+8.4f} {pooled_day_rmse:7.2f} "
          f"{daily_r2s.mean():+8.4f} {np.median(daily_r2s):+8.4f} "
          f"{pct_pos:5.0f}% {len(daily):7d}")

# --- Per-station by tier ---
print(f"\n  By tier (mean R2):")
print(f"  {'Config':<18s} {'t0':>8s} {'t1':>8s} {'t2':>8s} {'t3':>8s}")
print("  " + "-" * 52)
for cname in ALL_CONFIGS:
    tier_r2 = {}
    for t in TIER_NAMES:
        t_sids = [s for s in station_ids if sid_tier[s] == t]
        r2s = []
        for sid in t_sids:
            mask = stationId_vals == sid
            p = pred_all[cname][mask]
            valid = ~np.isnan(p) & ~np.isnan(y_all[mask])
            if valid.sum() >= 10:
                r2s.append(r2_score(y_all[mask][valid], p[valid]))
        tier_r2[t] = np.mean(r2s) if r2s else np.nan
    print(f"  {cname:<18s} {tier_r2['t0']:+8.4f} {tier_r2['t1']:+8.4f} "
          f"{tier_r2['t2']:+8.4f} {tier_r2['t3']:+8.4f}")

# =============================================================================
#  OUTPUT CSV
# =============================================================================
rows_out = []
for cname in ALL_CONFIGS:
    for sid in station_ids:
        mask = stationId_vals == sid
        p = pred_all[cname][mask]
        valid = ~np.isnan(p) & ~np.isnan(y_all[mask])
        if valid.sum() < 10:
            continue
        r2_h = r2_score(y_all[mask][valid], p[valid])
        rmse = np.sqrt(mean_squared_error(y_all[mask][valid], p[valid]))
        mae_v = mean_absolute_error(y_all[mask][valid], p[valid])
        bias_v = float(p[valid].mean() - y_all[mask][valid].mean())

        rows_out.append({
            "config": cname,
            "station_id": sid,
            "station_name": sid_name.get(sid, sid),
            "region": "Red River Delta",
            "tier": sid_tier[sid],
            "pm25_mean": float(station_pm_means[sid]),
            "n_hours": int(valid.sum()),
            "r2_hourly": round(r2_h, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae_v, 2),
            "bias": round(bias_v, 3),
        })

out_df = pd.DataFrame(rows_out)
out_path = os.path.join(OUT_DIR, f"{OUT_TAG}_test.csv")
safe_to_csv(out_df, out_path, "Results")

# =============================================================================
#  EXTERNAL VALIDATION: US Embassy + LCS
# =============================================================================
print(f"\n{'='*80}")
print("EXTERNAL VALIDATION: US Embassy + Delta LCS")
print(f"{'='*80}")

print(f"  Validation stations: {len(val_stations)} "
      f"({sum(1 for v in val_stations if v['type']=='LCS')} LCS + 1 Embassy)")

# --- KK coordinates (RFSI anchor network) ---
kk_coords = {sid: (sid_lat[sid], sid_lon[sid]) for sid in station_ids}

# --- Build the external feature table at each target's OWN coordinates ---
# Load the LCS/Embassy rows straight from the unified merged table (features
# already extracted at their own coordinates) and run them through the exact
# same satellite + derived-feature pipeline used for the KK model rows. The KK
# anchor network, training set, base margin and RFSI matrices are untouched.
print("  Building external feature table at target coordinates...")
ext_df = pd.read_csv(dataset_path, dtype={"stationId": str})
ext_df = ext_df[ext_df["stationId"].isin(external_sids)].reset_index(drop=True)
ext_df["ts"] = pd.to_datetime(ext_df["ts"])
ext_df["month"] = ext_df["ts"].dt.month
ext_df["date"] = ext_df["ts"].dt.date

# PM2.5 row-level QC (same masks as the KK pipeline); QC'd PM2.5 is the target.
ext_qc = pm25_quality_masks(ext_df)
_qc_tmp = ext_qc.copy()
_qc_tmp["stationId"] = ext_df["stationId"].values
ext_qc_counts = _qc_tmp.groupby("stationId")[list(ext_qc.columns)].sum().astype(int)
ext_df.loc[ext_qc.any(axis=1), "PM2.5"] = np.nan

ext_df = merge_satellite_features(ext_df)
if "AOT_fine" not in ext_df.columns:
    ext_df["AOT_fine"] = ext_df.get("AOT", np.nan) * ext_df.get("RF_center", np.nan)
ext_df = derive_obs_features(ext_df, label="external targets")
print(f"  External feature table: {len(ext_df):,} rows, "
      f"{ext_df['stationId'].nunique()} stations")

# --- Train final model on all 12 KK delta stations ---
rfsi_full = compute_rfsi(exclude_sid=None, K=K_NN)
rfsi_arr_full = np.column_stack([rfsi_full[c] for c in RFSI_FEATURES])
rfsi_wind_arr_full = np.column_stack([rfsi_full[c] for c in RFSI_WIND_FEATURES])
regional_all_mean = float(station_pm_means.mean())
bm_final = bm_from_pm_mean(regional_all_mean)

valid_train = ~np.isnan(y_all)
train_idx_all = np.where(valid_train)[0]
X_train_final = np.hstack([obs_arr, rfsi_wind_arr_full])[train_idx_all]
y_train_final = y_model[train_idx_all] - bm_final

m_final = xgb.XGBRegressor(**{**XGB_BASE, "monotone_constraints": mono_str_obs_wind_rfsi})
m_final.fit(X_train_final, y_train_final)
print(f"  Final delta_rfsi_wind model trained on {len(train_idx_all):,} rows "
      f"from {n_stn} KK stations")

# --- Final all-station feature gain for thesis figures/tables ---
print("\n  Final all-station grouped feature gain:")
final_gain_rows = []
final_feature_rows = []
final_importance_specs = [
    ("delta_bm", FEAT_OBS, obs_arr, mono_str_obs, None),
    ("delta_rfsi", FEAT_OBS_RFSI, np.hstack([obs_arr, rfsi_arr_full]),
     mono_str_obs_rfsi, None),
    ("delta_rfsi_wind", FEAT_OBS_WIND_RFSI, np.hstack([obs_arr, rfsi_wind_arr_full]),
     mono_str_obs_wind_rfsi, m_final),
]

for model_name, feat_names, x_full, mono_constraints, fitted_model in final_importance_specs:
    if fitted_model is None:
        fitted_model = xgb.XGBRegressor(**{
            **XGB_BASE,
            "monotone_constraints": mono_constraints,
        })
        fitted_model.fit(x_full[train_idx_all], y_train_final)

    imp = fitted_model.feature_importances_
    group_df = pd.DataFrame({
        "model": model_name,
        "feature": feat_names,
        "gain": imp,
        "feature_group": [feature_group(f) for f in feat_names],
    })
    total_gain = float(group_df["gain"].sum())
    group_gain = (
        group_df.groupby(["model", "feature_group"], as_index=False)["gain"]
        .sum()
        .sort_values("gain", ascending=False)
    )
    group_gain["gain_share_percent"] = np.where(
        total_gain > 0,
        100.0 * group_gain["gain"] / total_gain,
        0.0,
    )
    final_gain_rows.append(group_gain)
    final_feature_rows.append(group_df)

    print(f"    {model_name}:")
    for _, row in group_gain.iterrows():
        print(f"      {row['feature_group']:<26s} "
              f"gain={row['gain']:.4f} share={row['gain_share_percent']:5.1f}%")

final_group_gain_df = pd.concat(final_gain_rows, ignore_index=True)
final_feature_gain_df = pd.concat(final_feature_rows, ignore_index=True)
safe_to_csv(
    final_group_gain_df,
    os.path.join(OUT_DIR, f"{OUT_TAG}_final_feature_gain_by_group.csv"),
    "Final grouped feature gain",
)
safe_to_csv(
    final_feature_gain_df,
    os.path.join(OUT_DIR, f"{OUT_TAG}_final_feature_gain_by_feature.csv"),
    "Final feature gain",
)

# --- Per-target enriched feature index (own coordinates) ---
ext_by_sid = {}
for sid, g in ext_df.groupby("stationId"):
    ext_by_sid[sid] = g.sort_values("ts").drop_duplicates("ts", keep="first")


def _predict_target(feat_rows, v_lat, v_lon, ts_idx, wind_u, wind_v):
    """Predict PM2.5 for a set of target-feature rows + RFSI from KK anchors."""
    x_obs = feat_rows[FEAT_OBS].to_numpy(np.float32)
    rfsi = compute_external_rfsi(v_lat, v_lon, ts_idx, wind_u, wind_v, K=K_NN)
    x_val = np.hstack([x_obs, rfsi])
    pred = m_final.predict(x_val) + bm_final
    return np.clip(target_inverse(pred), 0, None)


# --- Process each validation station (own-coordinate features) ---
val_results = []
val_skips = []
val_pred_rows = []

for vi, vst in enumerate(val_stations):
    v_sid, v_lat, v_lon = vst["sid"], vst["lat"], vst["lon"]
    v_name = vst["name"]

    tdf = ext_by_sid.get(v_sid)
    if tdf is None:
        val_skips.append({"station": v_name, "type": vst["type"], "sid": v_sid,
                          "reason": "missing_in_unified",
                          "detail": "no rows in unified_thesis.csv"})
        continue

    # Own-coordinate rows: target PM2.5 valid AND the target's own
    # meteorology is available at that timestamp.
    own = tdf[tdf["PM2.5"].notna() & tdf["Temperature_final"].notna()].copy()
    own = own.sort_values("ts").reset_index(drop=True)
    if len(own) < 100:
        val_skips.append({"station": v_name, "type": vst["type"], "sid": v_sid,
                          "reason": "too_few_own_feature_hours",
                          "detail": f"{len(own)} hours with own PM2.5 + meteorology"})
        continue

    nearest_kk = min(station_ids,
                     key=lambda s: haversine(v_lat, v_lon, kk_coords[s][0], kk_coords[s][1]))
    nearest_dist = haversine(v_lat, v_lon, kk_coords[nearest_kk][0], kk_coords[nearest_kk][1])

    own_ts_idx = pd.to_datetime(own["ts"].values)
    pred_pm = _predict_target(own, v_lat, v_lon, own_ts_idx,
                              own["wind_u"].to_numpy(), own["wind_v"].to_numpy())

    y_val = own["PM2.5"].to_numpy()
    valid_both = ~np.isnan(pred_pm) & ~np.isnan(y_val)
    if valid_both.sum() < 50:
        val_skips.append({"station": v_name, "type": vst["type"], "sid": v_sid,
                          "reason": "too_few_valid_predictions",
                          "detail": f"{valid_both.sum()} rows after prediction/y filters"})
        continue

    r2_val = r2_score(y_val[valid_both], pred_pm[valid_both])
    rmse_val = np.sqrt(mean_squared_error(y_val[valid_both], pred_pm[valid_both]))
    mae_val = mean_absolute_error(y_val[valid_both], pred_pm[valid_both])
    bias_val = float(pred_pm[valid_both].mean() - y_val[valid_both].mean())

    val_pred_rows.append(pd.DataFrame({
        "station": v_name, "type": vst["type"], "sid": v_sid,
        "ts": own["ts"].values[valid_both],
        "y_true": y_val[valid_both], "y_pred": pred_pm[valid_both],
        "nearest_kk": sid_name.get(nearest_kk, nearest_kk),
        "kk_dist_km": nearest_dist,
    }))

    daily_val = pd.DataFrame({
        "date": own["ts"].dt.date.values[valid_both],
        "y": y_val[valid_both], "pred": pred_pm[valid_both],
    }).groupby("date", as_index=False).agg(
        y=("y", "mean"), pred=("pred", "mean"), n_hours=("y", "size"))
    daily_val = daily_val[daily_val["n_hours"] >= 18]
    if len(daily_val) >= 10 and daily_val["y"].std() > 1e-9:
        r2_day_val = r2_score(daily_val["y"], daily_val["pred"])
        rmse_day_val = np.sqrt(mean_squared_error(daily_val["y"], daily_val["pred"]))
        mae_day_val = mean_absolute_error(daily_val["y"], daily_val["pred"])
    else:
        r2_day_val = rmse_day_val = mae_day_val = np.nan

    qc_row = (ext_qc_counts.loc[v_sid] if v_sid in ext_qc_counts.index
              else pd.Series(0, index=ext_qc.columns))
    val_results.append({
        "station": v_name, "type": vst["type"], "sid": v_sid,
        "lat": v_lat, "lon": v_lon,
        "nearest_kk": sid_name.get(nearest_kk, nearest_kk)[:30],
        "kk_dist_km": round(nearest_dist, 1),
        "own_feature_hours": int(len(own)),
        "pm25_mean": round(float(y_val[valid_both].mean()), 1),
        "n_hours": int(valid_both.sum()),
        "qc_removed": int(qc_row.sum()),
        "qc_zero_or_negative": int(qc_row["zero_or_negative"]),
        "qc_too_high": int(qc_row["too_high"]),
        "qc_flatline": int(qc_row["flatline"]),
        "qc_stuck_low": int(qc_row["stuck_low"]),
        "r2": round(r2_val, 4), "rmse": round(rmse_val, 2),
        "mae": round(mae_val, 2), "bias": round(bias_val, 2),
        "n_days": int(len(daily_val)),
        "r2_daily": round(r2_day_val, 4) if np.isfinite(r2_day_val) else np.nan,
        "rmse_daily": round(rmse_day_val, 2) if np.isfinite(rmse_day_val) else np.nan,
        "mae_daily": round(mae_day_val, 2) if np.isfinite(mae_day_val) else np.nan,
    })

    if (vi + 1) % 10 == 0 or vi == len(val_stations) - 1:
        print(f"    [{vi+1}/{len(val_stations)}] processed...")

# --- Print LCS results ---
print(f"\n  External validation results ({len(val_results)} stations):")
print(f"  {'Station':<45s} {'type':>6s} {'dist':>5s} {'pm25':>5s} {'n':>5s} "
      f"{'R2':>7s} {'RMSE':>6s} {'MAE':>5s} {'bias':>6s}")
print("  " + "-" * 100)

lcs_r2s = []
for vr in sorted(val_results, key=lambda x: x["r2"], reverse=True):
    print(f"  {vr['station'][:45]:<45s} {vr['type']:>6s} {vr['kk_dist_km']:5.1f} "
          f"{vr['pm25_mean']:5.1f} {vr['n_hours']:5d} "
          f"{vr['r2']:+7.3f} {vr['rmse']:6.1f} {vr['mae']:5.1f} {vr['bias']:+6.1f}")
    if vr["type"] == "LCS":
        lcs_r2s.append(vr["r2"])

if lcs_r2s:
    arr_lcs = np.array(lcs_r2s)
    print(f"\n  LCS summary: n={len(arr_lcs)}, "
          f"mean R²={arr_lcs.mean():+.3f}, median={np.median(arr_lcs):+.3f}, "
          f"pct>0={100*(arr_lcs>0).mean():.0f}%")
lcs_daily = np.array([
    vr["r2_daily"] for vr in val_results
    if vr["type"] == "LCS" and np.isfinite(vr.get("r2_daily", np.nan))
])
if len(lcs_daily):
    print(f"  LCS daily summary: n={len(lcs_daily)}, "
          f"mean R2={lcs_daily.mean():+.3f}, "
          f"median={np.median(lcs_daily):+.3f}, "
          f"pct>0={100*(lcs_daily>0).mean():.0f}%")

emb = [vr for vr in val_results if vr["type"] == "Embassy"]
if emb:
    print(f"  US Embassy: R²={emb[0]['r2']:+.3f}, RMSE={emb[0]['rmse']:.1f}, "
          f"n={emb[0]['n_hours']} hours")
    if np.isfinite(emb[0].get("r2_daily", np.nan)):
        print(f"  US Embassy daily: R2={emb[0]['r2_daily']:+.3f}, "
              f"RMSE={emb[0]['rmse_daily']:.1f}, n={emb[0]['n_days']} days")

# Save validation results
if val_skips:
    print(f"\n  Validation skips ({len(val_skips)} stations):")
    for sk in val_skips:
        print(f"    - {sk['station'][:55]}: {sk['reason']} ({sk['detail']})")

val_df = pd.DataFrame(val_results)
val_path = os.path.join(OUT_DIR, f"{OUT_TAG}_lcs_validation.csv")
safe_to_csv(val_df, val_path, "Validation")
skip_df = pd.DataFrame(val_skips)
skip_path = os.path.join(OUT_DIR, f"{OUT_TAG}_lcs_validation_skips.csv")
safe_to_csv(skip_df, skip_path, "Validation skip log")
pred_df = pd.concat(val_pred_rows, ignore_index=True) if val_pred_rows else pd.DataFrame()
pred_path = os.path.join(OUT_DIR, f"{OUT_TAG}_lcs_validation_predictions.csv")
safe_to_csv(pred_df, pred_path, "Validation predictions")
print(f"  Total time: {time.time()-t0_start:.0f}s")
