"""
Experiment: Red River Delta regional PM2.5 model.

LOSO within the 12 Red River Delta KK stations. The model target is
log1p(PM2.5), the same target as the national LOSO diagnostic. The regional
leave-one-out mean is reported only as a naive benchmark. Thesis question:
can PM2.5 variation be predicted at a held-out location inside a known
polluted region when concurrent anchor stations are available?

Row-level PM2.5 QC is applied through pm25_quality_masks on the merged table.

Configs (the five rows of the thesis internal-LOSO table):
  - delta_obs: obs+physics features
  - delta_rfsi: obs+physics + RFSI
  - delta_rfsi_wind: obs+physics + wind-aware RFSI (final)
  - delta_rfsi_regime / delta_rfsi_regime_wind: + observable regime controls

Output: analysis/thesis_experiments/red_river_delta_*.csv
"""

import argparse, io, sys, os, warnings, time
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
parser.add_argument(
    "--target-scale",
    choices=["log", "raw"],
    default="log",
    help="Model target scale. Default 'log' is the thesis canonical setting.",
)
parser.add_argument(
    "--xgb-preset",
    choices=["default", "kfold"],
    default="default",
    help=(
        "'kfold' swaps the tree parameters for the strong known-station config "
        "of exp_random_sample_kfold.py (800 trees, depth 8, light regularization) "
        "as a capacity control; target scale and configs stay the "
        "same. Outputs are tagged _kfoldcfg."
    ),
)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = args.data_dir or REPO_DIR
OUT_DIR = os.path.join(REPO_DIR, "analysis", "thesis_experiments")
META_DIR = os.path.join(DATA_DIR, "data", "stations", "metadata")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_TAG = ("red_river_delta" if args.target_scale == "log"
           else f"red_river_delta_{args.target_scale}")
if args.xgb_preset == "kfold":
    OUT_TAG += "_kfoldcfg"


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

FEATURE_DIR = os.path.join(REPO_DIR, "Thesis", "scripts", "03_features")
if FEATURE_DIR not in sys.path:
    sys.path.insert(0, FEATURE_DIR)
from regional_feature_pipeline import (
    DAILY_SAT,
    MET,
    MODIS_ROLL_COLS,
    MONO_DICT,
    OBS_DERIVED,
    PHYSICS_FEATS,
    PRECIP,
    REGIONAL_SOURCE_COLUMNS,
    REGIME_FEATS,
    SAT_AOD,
    SAT_REGIME,
    STABILITY,
    TEMPORAL,
    TROPOMI_ROLL_COLS,
    attach_regime_features,
    prepare_observation_features,
    read_unified_stations,
    require_enriched_unified,
)

K_NN = 5

XGB_BASE = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=50,
    reg_alpha=0.1, reg_lambda=10.0, tree_method="hist",
    device="cuda", random_state=42, n_jobs=-1,
)
if args.xgb_preset == "kfold":
    XGB_BASE.update(
        n_estimators=800, max_depth=8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.3, reg_lambda=1.5,
    )
    print("XGB preset: kfold (800 trees, depth 8) — capacity control run")

CONFIGS = [
    "delta_obs",
    "delta_rfsi",
    "delta_rfsi_wind",
    "delta_rfsi_regime",
    "delta_rfsi_regime_wind",
]
FOLD_ABBREV = {
    "delta_obs": "OBS",
    "delta_rfsi": "D+R",
    "delta_rfsi_wind": "WND",
    "delta_rfsi_regime": "REG",
    "delta_rfsi_regime_wind": "R+W",
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
    return "other"


# =============================================================================
#  LOAD DATA
# =============================================================================
print("=" * 80)
print("RED RIVER DELTA REGIONAL PM2.5 MODEL")
print("=" * 80)

t0_start = time.time()

# Geographic scope of the final regional model.
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
dataset_path = os.path.join(DATA_DIR, "data/merged/unified_thesis.csv")
df = read_unified_stations(
    dataset_path, DELTA_SIDS, usecols=REGIONAL_SOURCE_COLUMNS
)
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["month"] = df["ts"].dt.month
df["date"] = df["ts"].dt.date
print(f"Loaded Delta scope: {len(df):,} rows, {df['stationId'].nunique()} stations "
      f"({time.time()-t0_start:.1f}s)")

# No station-level removals: the thesis keeps all stations and relies on the
# shared row-level PM2.5 QC masks (pm25_quality_masks) applied below.
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

# The unified table carries every non-fold-dependent GEE/TROPOMI, MODIS and
# static source-context feature.  RFSI is computed per fold in this script
# because it must exclude the held-out station of each LOSO fold.
require_enriched_unified(df)
print(
    "Unified regional enrichment: "
    f"HCHO rolling={df['hcho_30d_mean'].notna().sum():,}/{len(df):,}, "
    f"MODIS daily={df['modis_aod_7d'].notna().sum():,}/{len(df):,}"
)

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
#  MODEL-SPECIFIC FEATURE PREPARATION
# =============================================================================
# Source-level GEE/MODIS/static fields come from unified_thesis.csv.
# This section only cleans them and adds deterministic interactions.
df = prepare_observation_features(df, label="KK model rows")
df = attach_regime_features(df)
print("  Regional observation/regime interactions prepared.")

# =============================================================================
#  FEATURE SETS
# =============================================================================

# Feature names and their construction are defined in regional_feature_pipeline.py.
# Only the fold-dependent RFSI features are defined in this experiment.
FEAT_OBS = [f for f in (SAT_AOD + DAILY_SAT + MET + PRECIP + TEMPORAL +
            STABILITY + SAT_REGIME + OBS_DERIVED + PHYSICS_FEATS) if f in df.columns]
FEAT_OBS_RFSI = FEAT_OBS + RFSI_FEATURES
FEAT_OBS_WIND_RFSI = FEAT_OBS + RFSI_WIND_FEATURES
FEAT_OBS_REGIME = [f for f in (FEAT_OBS + REGIME_FEATS) if f in df.columns]
FEAT_OBS_REGIME_RFSI = FEAT_OBS_REGIME + RFSI_FEATURES
FEAT_OBS_REGIME_WIND_RFSI = FEAT_OBS_REGIME + RFSI_WIND_FEATURES

n_obs = len(FEAT_OBS)
n_phys = sum(1 for f in PHYSICS_FEATS if f in df.columns)
print(
    f"\n  Final feature policy: {n_obs} observation/physics features "
    f"+ {len(RFSI_WIND_FEATURES)} fold-specific RFSI = "
    f"{n_obs + len(RFSI_WIND_FEATURES)} features"
)
print(f"  Physics interactions: {n_phys}; no station identity in FEAT_OBS")

obs_arr = df[FEAT_OBS].values.astype(np.float32)
obs_arr = np.nan_to_num(obs_arr, nan=np.nan)

# Monotonicity constraints come from the shared pipeline MONO_DICT.
mono_str_obs = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS)
mono_str_obs_rfsi = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_RFSI)
mono_str_obs_wind_rfsi = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_WIND_RFSI)
mono_str_obs_regime_rfsi = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_REGIME_RFSI)
mono_str_obs_regime_wind_rfsi = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_REGIME_WIND_RFSI)

CONFIG_SETUP = {
    "delta_obs":               ("obs", "none"),
    "delta_rfsi":              ("obs", "basic"),
    "delta_rfsi_wind":         ("obs", "wind"),
    "delta_rfsi_regime":       ("regime", "basic"),
    "delta_rfsi_regime_wind":  ("regime", "wind"),
}

sid_to_int = {s: i for i, s in enumerate(station_ids)}
row_sid_idx = np.array([sid_to_int[s] for s in stationId_vals])

# =============================================================================
#  TARGET AND NAIVE-BENCHMARK SETUP
# =============================================================================
print(f"\n{'='*80}")
print("TARGET SETUP")
print(f"{'='*80}")
print(f"  Target scale: {args.target_scale}")

# Regional LOO mean diagnostic (naive benchmark)
print(f"\n  Delta regional mean (LOO on {n_stn} stations):")
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

# =============================================================================
#  LOSO (12 delta folds)
# =============================================================================
n_configs = len(CONFIGS)
print(f"\n{'='*80}")
print(f"LOSO: {n_stn} folds, {n_configs} configs")
print(f"{'='*80}\n")

pred_all = {c: np.full(len(df), np.nan) for c in CONFIGS}
regional_mean_oof = np.full(len(df), np.nan)

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
    regional_mean_oof[mask_test] = regional_loo_mean

    # Train separate model per config
    for cname in CONFIGS:
        feat_set, rfsi_mode = CONFIG_SETUP[cname]
        if feat_set == "obs":
            X_base_cfg = obs_arr
            mc = mono_str_obs
        elif feat_set == "regime":
            X_base_cfg = df[FEAT_OBS_REGIME].values.astype(np.float32)
            mc = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_REGIME)
        else:
            raise ValueError(f"Unknown feature set: {feat_set}")

        if rfsi_mode == "basic":
            X_cfg = np.hstack([X_base_cfg, rfsi_arr])
            if feat_set == "obs":
                mc = mono_str_obs_rfsi
            else:
                mc = mono_str_obs_regime_rfsi
        elif rfsi_mode == "wind":
            X_cfg = np.hstack([X_base_cfg, rfsi_wind_arr])
            if feat_set == "obs":
                mc = mono_str_obs_wind_rfsi
            else:
                mc = mono_str_obs_regime_wind_rfsi
        else:
            X_cfg = X_base_cfg

        params = {**XGB_BASE, "monotone_constraints": mc}
        m = xgb.XGBRegressor(**params)
        m.fit(X_cfg[train_idx], y_model[train_idx])
        pred_scaled = m.predict(X_cfg[test_idx])
        pred_all[cname][mask_test] = np.clip(
            target_inverse(pred_scaled), 0, None)

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
          f"OBS={fold_r2s['delta_obs']:+.3f} "
          f"D+R={fold_r2s['delta_rfsi']:+.3f} "
          f"R+W={fold_r2s['delta_rfsi_regime_wind']:+.3f} "
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

# Naive location-agnostic benchmark: each held station receives the mean of
# the other regional stations. It is not used as an XGBoost target offset.
baseline_valid = ~np.isnan(regional_mean_oof) & ~np.isnan(y_all)
baseline_r2 = r2_score(y_all[baseline_valid], regional_mean_oof[baseline_valid])
baseline_rmse = np.sqrt(mean_squared_error(
    y_all[baseline_valid], regional_mean_oof[baseline_valid]))
baseline_mae = mean_absolute_error(
    y_all[baseline_valid], regional_mean_oof[baseline_valid])
baseline_bias = float(
    regional_mean_oof[baseline_valid].mean() - y_all[baseline_valid].mean())
print("\n  Regional-mean naive OOF benchmark:")
print(f"    R2={baseline_r2:+.4f}, RMSE={baseline_rmse:.2f}, "
      f"MAE={baseline_mae:.2f}, bias={baseline_bias:+.3f}")

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
# anchor network, training set and RFSI matrices are untouched.
print("  Building external feature table at target coordinates...")
ext_df = read_unified_stations(
    dataset_path, external_sids, usecols=REGIONAL_SOURCE_COLUMNS
)
ext_df["ts"] = pd.to_datetime(ext_df["ts"])
ext_df["month"] = ext_df["ts"].dt.month
ext_df["date"] = ext_df["ts"].dt.date

# PM2.5 row-level QC (same masks as the KK pipeline); QC'd PM2.5 is the target.
ext_qc = pm25_quality_masks(ext_df)
_qc_tmp = ext_qc.copy()
_qc_tmp["stationId"] = ext_df["stationId"].values
ext_qc_counts = _qc_tmp.groupby("stationId")[list(ext_qc.columns)].sum().astype(int)
ext_df.loc[ext_qc.any(axis=1), "PM2.5"] = np.nan

require_enriched_unified(ext_df)
ext_df = prepare_observation_features(ext_df, label="external targets")
print(f"  External feature table: {len(ext_df):,} rows, "
      f"{ext_df['stationId'].nunique()} stations")

# --- Train final model on all 12 KK delta stations ---
rfsi_full = compute_rfsi(exclude_sid=None, K=K_NN)
rfsi_arr_full = np.column_stack([rfsi_full[c] for c in RFSI_FEATURES])
rfsi_wind_arr_full = np.column_stack([rfsi_full[c] for c in RFSI_WIND_FEATURES])
valid_train = ~np.isnan(y_all)
train_idx_all = np.where(valid_train)[0]
X_train_final = np.hstack([obs_arr, rfsi_wind_arr_full])[train_idx_all]
y_train_final = y_model[train_idx_all]

m_final = xgb.XGBRegressor(**{**XGB_BASE, "monotone_constraints": mono_str_obs_wind_rfsi})
m_final.fit(X_train_final, y_train_final)
print(f"  Final delta_rfsi_wind model trained on {len(train_idx_all):,} rows "
      f"from {n_stn} KK stations")

# --- Final all-station feature gain for thesis figures/tables ---
print("\n  Final all-station grouped feature gain:")
final_gain_rows = []
final_feature_rows = []
final_importance_specs = [
    ("delta_obs", FEAT_OBS, obs_arr, mono_str_obs, None),
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
    pred = m_final.predict(x_val)
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
