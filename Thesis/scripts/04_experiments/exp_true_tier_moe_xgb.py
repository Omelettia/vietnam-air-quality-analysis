"""
Experiment: Operationally valid tier assignment.

Current T4F uses the station's true PM2.5 mean for tier assignment — but we
don't have that at new locations. This tests operationally valid alternatives:

  oracle_t4f:      true PM2.5 tier (reference, operationally INVALID)
  no_t4f:          all stations, no tiers (operationally valid, lower bound)
  ghap_t4f:        tier assigned from GHAP monthly climatology mean
  twophase_t4f:    Phase 1 all-station model → predicted PM2.5 → assign tier → Phase 2
  twophase_knn10:  Phase 1 → predicted PM2.5 → KNN-10 by predicted mean → Phase 2
  twophase_knn15:  Phase 1 → predicted PM2.5 → KNN-15 by predicted mean → Phase 2

Output: analysis/thesis_experiments/tier_operational_test.csv
"""

import argparse, io, sys, os, warnings, time, glob, zipfile, unicodedata
from unicodedata import normalize as _unorm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None)
parser.add_argument(
    "--configs",
    default="oracle_t4f,no_t4f,no_t4f_raw,no_t4f_blend,no_t4f_gated,true_tier_moe_expert,true_tier_moe_40,true_tier_moe_guarded",
    help=(
        "Comma-separated configs to run."
    ),
)
parser.add_argument(
    "--out-prefix",
    default="true_tier_moe_xgb",
    help="Output filename prefix under analysis/experimental_shape_magnitude/true_tier_moe_xgb.",
)
parser.add_argument(
    "--save-oof",
    action="store_true",
    help="Also save per-hour out-of-fold predictions for time-series plots.",
)
parser.add_argument("--n-estimators", type=int, default=500)
parser.add_argument("--device", default="cuda")
parser.add_argument("--quick-stations", type=int, default=0)
parser.add_argument(
    "--source-profile",
    default="full",
    choices=[
        "full",
        "no_directional",
        "core_meteo_aod_rfsi",
        "no_static_gas_urban",
        "no_tropomi_gas",
        "no_tropomi_gas_no_aod",
        "no_urban_building",
        "no_aod",
        "no_rfsi",
    ],
    help="Remove whole source families from the hourly XGB feature set.",
)
parser.add_argument(
    "--aod-mode",
    choices=["dual", "himawari", "modis", "combined"],
    default="dual",
    help=(
        "AOD source handling: dual=current dynamic AOT plus MODIS fine-mode "
        "inside smart_v1; himawari=dynamic AOT plus Himawari RF/FM features; "
        "modis=matched MODIS AOD/FM/fine-AOD only; combined=single fused "
        "MODIS+Himawari AOD/FM representation."
    ),
)
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
OUT_DIR = os.path.join(REPO_DIR, "analysis", "experimental_shape_magnitude", "true_tier_moe_xgb")
META_DIR = os.path.join(DATA_DIR, "data", "stations", "metadata")
os.makedirs(OUT_DIR, exist_ok=True)

QC_DIR = os.path.join(REPO_DIR, "Thesis", "scripts", "02_processing")
if QC_DIR not in sys.path:
    sys.path.insert(0, QC_DIR)
from pm25_qc import pm25_quality_masks

K_NN = 5
MIN_TIER_STATIONS = 3
TIERS = ["t0", "t1", "t2", "t3"]

SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}

XGB_BASE = dict(
    n_estimators=args.n_estimators, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=50,
    reg_alpha=0.1, reg_lambda=10.0, tree_method="hist",
    device=args.device, random_state=42, n_jobs=-1,
)

VALID_CONFIGS = {
    "oracle_t4f", "no_t4f", "no_t4f_raw", "no_t4f_blend", "no_t4f_gated",
    "ghap_t4f", "twophase_t4f", "twophase_knn10", "twophase_knn15",
    "true_tier_moe_expert", "true_tier_moe_40", "true_tier_moe_guarded",
    "true_tier_hard_pred", "tierexpert_t0", "tierexpert_t1", "tierexpert_t2", "tierexpert_t3",
    "sel_ridge", "sel_extra",
}
CONFIGS = [c.strip() for c in args.configs.split(",") if c.strip()]
unknown = sorted(set(CONFIGS).difference(VALID_CONFIGS))
if unknown:
    raise ValueError(f"Unknown configs: {unknown}")
FOLD_ABBREV = {"oracle_t4f": "ORC", "no_t4f": "ALL", "ghap_t4f": "GHP",
               "no_t4f_raw": "RAW", "no_t4f_blend": "BLD",
               "no_t4f_gated": "GAT", "twophase_t4f": "2P4",
               "twophase_knn10": "2K10", "twophase_knn15": "2K15",
               "true_tier_moe_expert": "MOE",
               "true_tier_moe_40": "M40",
               "true_tier_moe_guarded": "MGD",
               "true_tier_hard_pred": "HRD",
               "tierexpert_t0": "EX0", "tierexpert_t1": "EX1",
               "tierexpert_t2": "EX2", "tierexpert_t3": "EX3",
               "sel_ridge": "SRD", "sel_extra": "SEX"}


def assign_tier(mean_pm):
    if mean_pm < 10:   return "t0"
    elif mean_pm < 20: return "t1"
    elif mean_pm < 35: return "t2"
    return "t3"


def tier_class(tier):
    return {"t0": "low", "t1": "moderate_low", "t2": "moderate", "t3": "high"}[tier]


def severity_tier(tier):
    return {"t0": 0, "t1": 1, "t2": 2, "t3": 3}[tier]


def load_observable_tier_prior(repo_dir):
    prior_path = os.path.join(
        repo_dir, "analysis", "experimental_shape_magnitude", "soft_tier_moe",
        "soft_tier_prior_station_table.csv",
    )
    if not os.path.exists(prior_path):
        print("  WARNING: soft tier prior not found; using phase-1 fallback probabilities")
        return {}
    prior = pd.read_csv(prior_path, dtype={"station_id": str})
    out = {}
    for _, row in prior.iterrows():
        vals = np.array([row.get(f"p_{t}", 0.0) for t in TIERS], dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        vals = vals / vals.sum() if vals.sum() > 0 else np.full(4, 0.25)
        out[row["station_id"]] = vals
    print(f"  Loaded observable tier prior: {len(out)} stations")
    return out


def fallback_tier_probs(pred_tier):
    vals = np.full(4, 0.05, dtype=float)
    vals[TIERS.index(pred_tier)] = 0.85
    return vals / vals.sum()


def station_alpha_guarded(probs, base_mean, expert_mean):
    p_high = float(probs[TIERS.index("t3")])
    pmax = float(np.max(probs))
    confidence = np.clip((pmax - 0.25) / 0.75, 0.0, 1.0)
    alpha = 0.15 + 0.45 * confidence
    if base_mean < 35.0 <= expert_mean and p_high < 0.75:
        alpha = min(alpha, 0.05)
    if base_mean >= 35.0 > expert_mean and p_high >= 0.35:
        alpha = min(alpha, 0.05)
    if abs(expert_mean - base_mean) > 15 and p_high < 0.70:
        alpha *= 0.65
    return float(np.clip(alpha, 0.0, 0.65))


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


# =============================================================================
#  LOAD DATA (same pipeline as other experiments)
# =============================================================================
print("=" * 80)
print("TIER OPERATIONAL EXPERIMENT  (operationally valid tier assignment)")
print("=" * 80)
print(f"AOD mode: {args.aod_mode}")

t0 = time.time()

# Prefer v4 (121 stations: keeps all 40 thesis stations, the 3 ex-broken sensors
# cleaned by the stronger row-level QC mask rather than dropped). v3/v2 are legacy
# datasets that excluded those 3 stations at the dataset-build level.
_v4_path = os.path.join(DATA_DIR, "data/merged/unified_thesis_v4.csv")
_v3_path = os.path.join(DATA_DIR, "data/merged/unified_thesis_v3.csv")
_v2_path = os.path.join(DATA_DIR, "data/merged/unified_thesis_v2.csv")
_data_path = next((p for p in (_v4_path, _v3_path, _v2_path) if os.path.exists(p)), _v2_path)
df = pd.read_csv(_data_path, dtype={"stationId": str})
print(f"  Dataset: {os.path.basename(_data_path)}")
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["month"] = df["ts"].dt.month
df["date"] = df["ts"].dt.date
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations "
      f"({time.time()-t0:.1f}s)")

_meta_path = os.path.join(DATA_DIR, "analysis/thesis_audit/station_selection_final.csv")
if os.path.exists(_meta_path) and "station" not in df.columns:
    meta = pd.read_csv(_meta_path, dtype={"stationId": str})
    sid_name = dict(zip(meta["stationId"], meta["station_name"]))
    sid_region = dict(zip(meta["stationId"], meta["region"]))
    sid_lat = dict(zip(meta["stationId"], meta["lat"]))
    sid_lon = dict(zip(meta["stationId"], meta["lon"]))
else:
    _meta_df = df.groupby("stationId").first().reset_index()
    sid_name = dict(zip(_meta_df["stationId"], _meta_df.get("station", _meta_df["stationId"])))
    sid_region = dict(zip(_meta_df["stationId"], _meta_df.get("region", "Unknown")))
    sid_lat = dict(zip(_meta_df["stationId"], _meta_df.get("latitude", np.nan)))
    sid_lon = dict(zip(_meta_df["stationId"], _meta_df.get("longitude", np.nan)))
    meta = _meta_df.rename(columns={"station": "station_name", "latitude": "lat", "longitude": "lon"})
station_ids = sorted(df["stationId"].unique())
if args.quick_stations:
    station_ids = station_ids[:args.quick_stations]
    df = df[df["stationId"].isin(station_ids)].copy().reset_index(drop=True)
    print(f"  QUICK MODE: {len(station_ids)} stations")
n_stn = len(station_ids)

TARGET = "PM2.5"
y_all = df[TARGET].values
stationId_vals = df["stationId"].values

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
y_log = np.log1p(np.nan_to_num(y_all, nan=0.0))

station_pm_means = df.groupby("stationId")["PM2.5"].mean()
sid_tier = {s: assign_tier(station_pm_means[s]) for s in station_ids}
for t in ["t0", "t1", "t2", "t3"]:
    print(f"  {t}: {len([s for s in station_ids if sid_tier[s] == t])} stations")

# =============================================================================
#  LOAD GHAP
# =============================================================================
ghap_zip_candidates = sorted(glob.glob(os.path.join(DATA_DIR, "data", "gee_exports", "pm25-*.zip")))
if ghap_zip_candidates:
    with zipfile.ZipFile(ghap_zip_candidates[-1]) as z:
        with z.open("pm25/ghap_monthly_climatology.csv") as f:
            ghap_mc = pd.read_csv(f, dtype={"stationId": str})
    ghap_mc = ghap_mc.rename(columns={"mean": "ghap_clim"})
    ghap_station_mean = ghap_mc.groupby("stationId")["ghap_clim"].mean()
    sid_ghap_tier = {}
    for sid in station_ids:
        if sid in ghap_station_mean.index and not np.isnan(ghap_station_mean[sid]):
            sid_ghap_tier[sid] = assign_tier(ghap_station_mean[sid])
        else:
            sid_ghap_tier[sid] = assign_tier(station_pm_means[sid])
    print(f"\n  GHAP tier assignment:")
    for t in ["t0", "t1", "t2", "t3"]:
        ghap_sids = [s for s in station_ids if sid_ghap_tier[s] == t]
        true_sids = [s for s in station_ids if sid_tier[s] == t]
        print(f"    {t}: GHAP={len(ghap_sids)} (true={len(true_sids)})")
    match = sum(1 for s in station_ids if sid_tier[s] == sid_ghap_tier[s])
    print(f"    Concordance: {match}/{n_stn} ({100*match/n_stn:.0f}%)")
else:
    print("  WARNING: No GHAP zip found, ghap_t4f will fall back to oracle")
    sid_ghap_tier = dict(sid_tier)

tier_prior_probs = load_observable_tier_prior(REPO_DIR)

# =============================================================================
#  GEE DAILY EXPORT
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
merge_cols = ["stationId", "date_merge"] + ANOM_RAW
df = df.merge(sat_wide[merge_cols], left_on=["stationId", "date"],
              right_on=["stationId", "date_merge"], how="left")
df.drop(columns=["date_merge"], inplace=True)
print(f"  GEE merged: SO2 anom coverage {df['so2_daily_anom'].notna().sum():,}/{len(df):,}")

# =============================================================================
#  BUILDING DENSITY
# =============================================================================
bld_path = os.path.join(META_DIR, "station_building_density.csv")
bld = pd.read_csv(bld_path, dtype={"stationId": str})
BUILDING_COLS = ["building_count_1km", "building_area_1km",
                 "building_count_3km", "building_area_3km"]
bld_map = bld.set_index("stationId")[BUILDING_COLS]
df = df.merge(bld_map, left_on="stationId", right_index=True, how="left")
for col in BUILDING_COLS:
    df[col] = df[col].fillna(0)

# =============================================================================
#  RFSI
# =============================================================================
print("  RFSI setup...")

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
n_ts = pm25_mat.shape[0]

RFSI_LEAN = ["PM25_nn_idw", "PM25_nn1", "dist_nn1"]
RFSI_EXTRA = ["PM25_nn2", "PM25_nn3"]
LAG_FEATURES = ["PM25_nn1_lag1h", "PM25_nn1_lag3h", "PM25_nn1_lag6h"]
LAG_HOURS = [1, 3, 6]


def compute_rfsi(exclude_sid=None, K=5):
    n = len(df)
    pm_nn = np.full((n, K), np.nan)
    d_nn = np.full((n, K), np.nan)
    excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
    ts_row_vals = df["ts_row"].values
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
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d_nn
        pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)
    return {"PM25_nn_idw": pm_idw, "PM25_nn1": pm_nn[:, 0],
            "dist_nn1": d_nn[:, 0], "PM25_nn2": pm_nn[:, 1],
            "PM25_nn3": pm_nn[:, 2]}


def compute_lagged_rfsi(exclude_sid=None):
    n = len(df)
    lags = {lh: np.full(n, np.nan) for lh in LAG_HOURS}
    excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
    ts_row_vals = df["ts_row"].values
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
        nn1_col = sid_to_col[station_ids[cands[0][0]]]
        for lag_h in LAG_HOURS:
            tr_lag = tr - lag_h
            in_bounds = tr_lag >= 0
            tr_safe = np.clip(tr_lag, 0, n_ts - 1)
            vals = pm25_mat[tr_safe, nn1_col]
            vals[~in_bounds] = np.nan
            lags[lag_h][ri] = vals
    return {f"PM25_nn1_lag{lh}h": lags[lh] for lh in LAG_HOURS}


# =============================================================================
#  SATELLITE FEATURES
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

matched_static_path = os.path.join(META_DIR, "aod_source_matched_static.csv")
matched_temporal_path = os.path.join(META_DIR, "aod_source_matched_temporal.csv")
matched_static_map = pd.DataFrame()
SOURCE_AOD_FEATURES = []
if args.aod_mode in {"himawari", "modis", "combined"}:
    if not (os.path.exists(matched_static_path) and os.path.exists(matched_temporal_path)):
        raise FileNotFoundError(
            "Matched AOD feature files are missing. Run analysis/build_matched_aod_features.py first."
        )
    matched_static = pd.read_csv(matched_static_path, dtype={"stationId": str})
    matched_static_map = matched_static.set_index("stationId")
    matched_temporal = pd.read_csv(
        matched_temporal_path, dtype={"stationId": str}, parse_dates=["date"]
    )
    matched_temporal["date_merge"] = matched_temporal["date"].dt.date
    prefix = {"himawari": "him", "modis": "modis", "combined": "combined"}[args.aod_mode]

    temporal_rename = {
        f"{prefix}_aod_7d": "src_aod_7d",
        f"{prefix}_fmf_7d": "src_fmf_7d",
        f"{prefix}_fine_aod_7d": "src_fine_aod_7d",
    }
    keep_temporal = ["stationId", "date_merge"] + [
        c for c in temporal_rename if c in matched_temporal.columns
    ]
    matched_temporal = matched_temporal[keep_temporal].rename(columns=temporal_rename)
    df = df.merge(
        matched_temporal,
        left_on=["stationId", "date"],
        right_on=["stationId", "date_merge"],
        how="left",
    )
    df.drop(columns=["date_merge"], inplace=True)

    static_rename = {}
    for suffix in [
        "aod_center", "aod_DJF", "aod_MAM", "aod_JJA", "aod_SON",
        "aod_contrast", "aod_directionality", "aod_max_nearby",
        "fmf_center", "fine_aod_center", "ae_center",
    ] + [f"aod_clim_{d}" for d in SECTOR_NAMES] + [f"fine_aod_clim_{d}" for d in SECTOR_NAMES]:
        col = f"{prefix}_{suffix}"
        if col in matched_static_map.columns:
            static_rename[col] = f"src_{suffix}"
    if static_rename:
        df = df.merge(
            matched_static_map[list(static_rename)].rename(columns=static_rename),
            left_on="stationId",
            right_index=True,
            how="left",
        )

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
    print(f"  Matched AOD source merged ({args.aod_mode}): {len(SOURCE_AOD_FEATURES)} features")

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

# Directional climatology
print("  Loading directional climatology...")


def _norm_tok(s):
    return unicodedata.normalize("NFKD", str(s)).encode(
        "ascii", "ignore").decode().lower()


def _tokenize(s):
    return set(_norm_tok(s).replace("-", " ").replace(",", " ").split())


def _resolve_dir_path(basename):
    """Prefer _123 version of directional file, fall back to original."""
    stem, ext = os.path.splitext(basename)
    p123 = os.path.join(META_DIR, f"{stem}_123{ext}")
    if os.path.exists(p123):
        return p123, True
    p = os.path.join(META_DIR, basename)
    if os.path.exists(p):
        return p, False
    p = os.path.join(DATA_DIR, basename)
    return p, False

# Build id_map for legacy (40-station short-name) directional files
no2_clim_123_path = os.path.join(META_DIR, "no2_directional_clim_123.csv")
no2_clim_legacy_path = os.path.join(META_DIR, "no2_directional_clim.csv")
_use_123 = os.path.exists(no2_clim_123_path)

id_map = {}
if not _use_123 and os.path.exists(no2_clim_legacy_path):
    no2_dir_raw = pd.read_csv(no2_clim_legacy_path, dtype={"stationId": str})
    if "name" in no2_dir_raw.columns:
        no2_dir_names = no2_dir_raw.groupby("stationId")["name"].first()
        for short_id, no2_name in no2_dir_names.items():
            no2_tokens = _tokenize(no2_name)
            best_score, best_long_id = -1, None
            for _, row in meta.iterrows():
                meta_tokens = _tokenize(row.get("station_name", row.get("station", "")))
                score = len(no2_tokens & meta_tokens)
                if score > best_score:
                    best_score = score
                    best_long_id = row["stationId"]
            id_map[short_id] = best_long_id


def _load_dir_clim(csv_path, value_col="mean", needs_id_map=False):
    raw = pd.read_csv(csv_path, dtype={"stationId": str})
    if needs_id_map and id_map:
        raw["stationId"] = raw["stationId"].map(id_map)
        raw = raw.dropna(subset=["stationId"])
    clim_df = raw.groupby(["stationId", "direction"])[value_col].mean().reset_index()
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


_so2_path, _so2_is123 = _resolve_dir_path("tropomi_so2_directional.csv")
station_so2_sectors, station_so2_centers = _load_dir_clim(
    _so2_path, "mean", needs_id_map=not _so2_is123)
_co_path, _co_is123 = _resolve_dir_path("tropomi_co_directional.csv")
station_co_sectors, station_co_centers = _load_dir_clim(
    _co_path, "mean", needs_id_map=not _co_is123)
_hcho_path, _hcho_is123 = _resolve_dir_path("tropomi_hcho_directional.csv")
station_hcho_sectors, station_hcho_centers = _load_dir_clim(
    _hcho_path, "mean", needs_id_map=not _hcho_is123)
_lst_path, _lst_is123 = _resolve_dir_path("lst_anomaly_directional.csv")
station_lstd_sectors, station_lstd_centers = _load_dir_clim(
    _lst_path, "lst_anomaly", needs_id_map=not _lst_is123)
print(f"  Directional files: {'123-station' if _use_123 else 'legacy 40-station'}")

# =============================================================================
#  SECTOR + SMART_V1
# =============================================================================
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
ws = np.sqrt(df["wind_u"].values**2 + df["wind_v"].values**2)
vc_inv = 1.0 / (df["PBLH"].clip(lower=50).values * ws.clip(min=0.1) + 1)
month_vals = df["month"].values
season_vals = np.array([SEASON_MAP[m] for m in month_vals])

all_no2_sec = np.array([station_no2_sectors[s] for s in station_ids])
all_ntl_sec = np.array([station_ntl_sectors[s] for s in station_ids])
all_lst_sec = np.array([station_lst_sectors[s] for s in station_ids])

no2_center_all = np.array([
    no2_map.loc[s, "no2_center"] if s in no2_map.index else np.nan for s in station_ids])
ntl_center_all = np.array([
    emit_map.loc[s, "ntl_center"] if s in emit_map.index else np.nan for s in station_ids])
lst_center_all = np.array([
    emit_map.loc[s, "lst_anom_center"] if s in emit_map.index else np.nan for s in station_ids])
if args.aod_mode == "dual":
    fmf_center_all = np.array([
        new_sat_map.loc[s, "fmf_center"] if s in new_sat_map.index else np.nan
        for s in station_ids])
else:
    prefix = {"himawari": "him", "modis": "modis", "combined": "combined"}[args.aod_mode]
    fmf_col = f"{prefix}_fmf_center"
    fmf_center_all = np.array([
        matched_static_map.loc[s, fmf_col]
        if (s in matched_static_map.index and fmf_col in matched_static_map.columns)
        else np.nan
        for s in station_ids])


def _lohi(sec_arr, center_arr):
    combined = np.concatenate([sec_arr.ravel(), center_arr])
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

# SO2/CO/HCHO/LST standalone
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

so2_upwind_vals = np.nan_to_num(so2_upwind_vals, nan=0.0)
co_upwind_vals = np.nan_to_num(co_upwind_vals, nan=0.0)
lst_anom_upwind_vals = np.nan_to_num(lst_anom_upwind_vals, nan=0.0)

df["so2_upwind"] = so2_upwind_vals
df["co_upwind"] = co_upwind_vals
so2_cen_map = {sid: float(np.nan_to_num(station_so2_centers.get(sid, 0.0), nan=0.0))
               for sid in station_ids}
df["so2_center"] = df["stationId"].map(so2_cen_map).fillna(0.0)
co_cen_map = {sid: float(np.nan_to_num(station_co_centers.get(sid, 0.0), nan=0.0))
              for sid in station_ids}
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
_fire_path, _fire_is123 = _resolve_dir_path("fire_counts_directional.csv")
fire_raw = pd.read_csv(_fire_path, dtype={"stationId": str})
fire_raw = fire_raw.rename(columns={"mean": "fire_val"})
if not _fire_is123 and id_map:
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

fire_upwind = np.zeros(len(df))
for sid in station_ids:
    mask = stationId_vals == sid
    if not mask.any(): continue
    idx = np.where(mask)[0]
    lookup = station_fire_dir_season.get(sid, {})
    for i in idx:
        fire_upwind[i] = lookup.get((sector_idx[i], season_vals[i]), 0.0)
df["fire_upwind"] = fire_upwind

fill_cols = (NO2_STATIC_COLS + NO2_SECTOR_COLS +
             ["ntl_center", "smart_v1_center", "smart_v1_upwind", "smart_v1_max",
              "smart_v1_contrast", "smart_v1_upwind_x_VC_inv",
              "so2_upwind", "so2_center", "so2_contrast",
              "co_upwind", "co_center", "hcho_center",
              "so2_upwind_x_VC_inv", "lst_anom_upwind_x_VC_inv", "fire_upwind"])
for c in set(fill_cols):
    if c in df.columns:
        df[c] = df[c].fillna(0)

# Outer AOD
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
    _day_count=("_is_real_aod", "sum")).reset_index()
day_outer.loc[day_outer["_day_count"] == 0, "aod_outer_day_mean"] = np.nan
df = df.merge(day_outer[["stationId", "date", "aod_outer_day_mean"]],
              on=["stationId", "date"], how="left")
df.drop(columns=["_outer_real", "_is_real_aod"], inplace=True)

# Temporal
df["dow_is_weekend"] = (df["ts"].dt.dayofweek >= 5).astype(float)
df.sort_values(["stationId", "ts"], inplace=True)
df.reset_index(drop=True, inplace=True)
stationId_vals = df["stationId"].values

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
TEMPORAL_EXTRA = ['day_of_year_sin']

# =============================================================================
#  FEATURE CLEANING
# =============================================================================
print(f"\n{'='*80}")
print("FEATURE CLEANING")
print(f"{'='*80}")
n_total = len(df)

rh_bad = df['Humidity_final'] < 5
df.loc[rh_bad, 'Humidity_final'] = np.nan
prs_bad = (df['Pressure_final'] < 950) | (df['Pressure_final'] > 1040)
df.loc[prs_bad, 'Pressure_final'] = np.nan
temp_bad = (df['Temperature_final'] < 0) | (df['Temperature_final'] > 50)
df.loc[temp_bad, 'Temperature_final'] = np.nan
pblh_bad = (df['PBLH'] < 0) | (df['PBLH'] > 6000)
df.loc[pblh_bad, 'PBLH'] = np.nan
ae_bad = df['AE'].abs() > 5
df.loc[ae_bad, 'AE'] = np.nan
aot_bad = df['AOT_ffill_48h'] > 5
df.loc[aot_bad, 'AOT_ffill_48h'] = np.nan
aot2_bad = df['AOT_outer_mean'] > 5
df.loc[aot2_bad, 'AOT_outer_mean'] = np.nan
ws_suspect = (df['WS_local'] == 0) & (np.sqrt(df['wind_u']**2 + df['wind_v']**2) > 1.0)
df.loc[ws_suspect, 'WS_local'] = np.nan

any_cleaned = rh_bad | prs_bad | temp_bad | pblh_bad | ws_suspect
print(f"  Cleaned: {any_cleaned.sum():,} rows ({100*any_cleaned.sum()/n_total:.1f}%)")

ws_arr = np.sqrt(df["wind_u"].values**2 + df["wind_v"].values**2)
pblh_f = df["PBLH"].fillna(200).values
vc_inv_clean = 1.0 / (np.clip(pblh_f, 50, None) * np.clip(ws_arr, 0.1, None) + 1)

df["RH_factor"] = 1.0 / (1.0 - (df["Humidity_final"] / 100.0).clip(upper=0.95))
df["VC"] = pblh_f * np.clip(ws_arr, 0.1, None)
df["smart_v1_upwind_x_VC_inv"] = smart_v1_upwind * vc_inv_clean
df["so2_upwind_x_VC_inv"] = so2_upwind_vals * vc_inv_clean
df["lst_anom_upwind_x_VC_inv"] = lst_anom_upwind_vals * vc_inv_clean
df["so2_anom_x_vc_inv"] = df["so2_daily_anom"].values * vc_inv_clean
df["co_anom_x_vc_inv"] = df["co_daily_anom"].values * vc_inv_clean
df["lst_anom_x_vc_inv"] = df["lst_day_anom"].values * vc_inv_clean

aot_outer_c = df["AOT_outer_mean"].fillna(0).values
rh_frac_c = (df["Humidity_final"] / 100.0).clip(0, 0.95).values
f_rh_c = 1.0 / (1.0 - rh_frac_c)
df["aod_outer_surface"] = aot_outer_c / (pblh_f + 100.0)
df["aod_outer_pm25"] = aot_outer_c / (pblh_f + 100.0) / f_rh_c
df["aod_outer_x_VC_inv"] = aot_outer_c * vc_inv_clean

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

print("  Derived features recomputed.")

# =============================================================================
#  FEATURE SET (66f)
# =============================================================================
MET_CORE = ["PBLH", "VC", "wind_u", "wind_v", "WS_local",
            "Temperature_final", "Humidity_final", "Pressure_final",
            "dT_6h", "dRH_6h", "rain_days_7d", "rain_sum_48h",
            "consecutive_dry_days", "hrs_since_rain", "RH_factor"]
TEMPORAL_LEAN = ["hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_cos"]
AOD_CORE = ["AOT_ffill_48h", "AOT_outer_mean", "AE", "RF", "hours_since_valid_AOT"]
if args.aod_mode in {"modis", "combined"}:
    AOD_CORE = SOURCE_AOD_FEATURES
elif args.aod_mode == "himawari":
    AOD_CORE = AOD_CORE + SOURCE_AOD_FEATURES
BUILDING_LEAN = ["building_area_1km"]
SMART_V1_EMISSION = ["smart_v1_center", "smart_v1_upwind", "smart_v1_max",
                     "smart_v1_contrast", "smart_v1_upwind_x_VC_inv"]
SO2CO_STANDALONE = ["so2_upwind", "so2_center", "so2_contrast",
                    "co_upwind", "co_center", "hcho_center",
                    "so2_upwind_x_VC_inv", "lst_anom_upwind_x_VC_inv"]
OUTER_ALL_EXTRA = ["aod_outer_surface", "aod_outer_pm25",
                   "aod_outer_x_VC_inv", "aod_outer_gradient", "aod_outer_day_mean"]
if args.aod_mode in {"modis", "combined"}:
    OUTER_ALL_EXTRA = []

BASE_FULL = [f for f in MET_CORE + TEMPORAL_LEAN + AOD_CORE + BUILDING_LEAN if f in df.columns]
FL_BASE = [f for f in BASE_FULL + SMART_V1_EMISSION + OUTER_ALL_EXTRA + SO2CO_STANDALONE if f in df.columns]
DA_BASE = [f for f in FL_BASE + DAILY_ANOM_ALL + WEATHER_PERSIST + TEMPORAL_EXTRA if f in df.columns]
FEAT_FULL = DA_BASE + RFSI_LEAN + RFSI_EXTRA + LAG_FEATURES + ["dow_is_weekend"]

SOURCE_GROUPS = {
    "meteo": set(MET_CORE + WEATHER_PERSIST),
    "time": set(TEMPORAL_LEAN + TEMPORAL_EXTRA + ["dow_is_weekend"]),
    "aod": set(AOD_CORE + OUTER_ALL_EXTRA),
    "urban_building": set(BUILDING_LEAN + ["fire_upwind"]),
    "tropomi_gas": set(SMART_V1_EMISSION + SO2CO_STANDALONE + DAILY_ANOM_ALL),
    "rfsi": set(RFSI_LEAN + RFSI_EXTRA + LAG_FEATURES),
}


def source_profile_features(profile):
    keep = list(FEAT_FULL)
    if profile == "full":
        return keep
    if profile == "no_directional":
        blocked = set(SMART_V1_EMISSION + SO2CO_STANDALONE + ["fire_upwind"])
        return [f for f in keep if f not in blocked]
    if profile == "core_meteo_aod_rfsi":
        allowed = SOURCE_GROUPS["meteo"] | SOURCE_GROUPS["time"] | SOURCE_GROUPS["aod"] | SOURCE_GROUPS["rfsi"]
        return [f for f in keep if f in allowed]
    if profile == "no_static_gas_urban":
        blocked = SOURCE_GROUPS["tropomi_gas"] | SOURCE_GROUPS["urban_building"]
        return [f for f in keep if f not in blocked]
    if profile == "no_tropomi_gas":
        return [f for f in keep if f not in SOURCE_GROUPS["tropomi_gas"]]
    if profile == "no_tropomi_gas_no_aod":
        blocked = SOURCE_GROUPS["tropomi_gas"] | SOURCE_GROUPS["aod"]
        return [f for f in keep if f not in blocked]
    if profile == "no_urban_building":
        return [f for f in keep if f not in SOURCE_GROUPS["urban_building"]]
    if profile == "no_aod":
        return [f for f in keep if f not in SOURCE_GROUPS["aod"]]
    if profile == "no_rfsi":
        return [f for f in keep if f not in SOURCE_GROUPS["rfsi"]]
    raise ValueError(profile)


FEAT_ALL = source_profile_features(args.source_profile)

n_feat = len(FEAT_ALL)

MONO_DICT = {
    "smart_v1_upwind": 1, "smart_v1_upwind_x_VC_inv": 1,
    "so2_upwind": 1, "so2_center": 1, "so2_upwind_x_VC_inv": 1,
    "co_upwind": 1, "co_center": 1, "hcho_center": 1,
    "lst_anom_upwind_x_VC_inv": 1,
    "VC": -1, "PBLH": -1, "WS_local": -1, "PM25_nn_idw": 1,
    "so2_daily_anom": 1, "co_daily_anom": 1, "no2_daily_anom": 1,
    "so2_anom_x_vc_inv": 1, "co_anom_x_vc_inv": 1,
    "PM25_nn1_lag1h": 1, "PM25_nn1_lag3h": 1, "PM25_nn1_lag6h": 1,
    "PM25_nn2": 1, "PM25_nn3": 1,
}
mono_full = tuple(MONO_DICT.get(f, 0) for f in FEAT_ALL)

pipeline_time = time.time() - t0
print(f"\n  Pipeline built in {pipeline_time:.0f}s")
print(f"  Source profile: {args.source_profile}")
print(f"  Features: {n_feat}f")
print(f"  Feature list: {', '.join(FEAT_ALL)}")

# =============================================================================
#  TARGETS
# =============================================================================
y_all = df['PM2.5'].values
y_log = np.log1p(np.nan_to_num(y_all, nan=0.0))
stationId_vals = df["stationId"].values

global_pm_mean = float(np.nanmean(y_all))
bm_global = np.log1p(global_pm_mean)
y_res = y_log - bm_global

BASE_FEATURE_SET = set(DA_BASE)
RFSI_LEAN_SET = set(RFSI_LEAN)
RFSI_EXTRA_SET = set(RFSI_EXTRA)
LAG_FEATURE_SET = set(LAG_FEATURES)

sid_to_int = {s: i for i, s in enumerate(station_ids)}
row_sid_idx = np.array([sid_to_int[s] for s in stationId_vals])

print(f"\n  Global PM2.5 mean: {global_pm_mean:.2f}")

# =============================================================================
#  LOSO — TWO-PHASE
# =============================================================================
print(f"\n{'='*80}")
print(f"LOSO: {n_stn} folds, hist booster, {n_feat}f, depth=4")
print(f"  Configs: {' / '.join(CONFIGS)}")
print(f"{'='*80}\n")

pred_all = {c: np.full(len(df), np.nan) for c in CONFIGS}
params_base = {**XGB_BASE, "monotone_constraints": mono_full}

phase1_predicted_means = {}

for fold_i, held_sid in enumerate(station_ids):
    nm = sid_name.get(held_sid, held_sid)[:35]
    held_tier_true = sid_tier[held_sid]
    held_tier_ghap = sid_ghap_tier[held_sid]
    mask_test = stationId_vals == held_sid
    n_test = mask_test.sum()
    if n_test < 10:
        print(f"  [{fold_i+1:2d}/{n_stn}] {nm:35s} | SKIP (n={n_test})")
        continue

    t_fold = time.time()
    pm_val = float(station_pm_means[held_sid])

    rfsi_fold = compute_rfsi(exclude_sid=held_sid, K=K_NN)
    lag_fold = compute_lagged_rfsi(exclude_sid=held_sid)

    x_parts = []
    for feat in FEAT_ALL:
        if feat in rfsi_fold:
            x_parts.append(rfsi_fold[feat])
        elif feat in lag_fold:
            x_parts.append(lag_fold[feat])
        elif feat == "dow_is_weekend":
            x_parts.append(df["dow_is_weekend"].values)
        else:
            x_parts.append(df[feat].values)
    X_all = np.column_stack(x_parts)

    test_idx = np.where(mask_test)[0]
    valid_y = ~np.isnan(y_all[mask_test])
    X_te = pd.DataFrame(X_all[test_idx], columns=FEAT_ALL)

    others = [s for s in station_ids if s != held_sid]
    fold_r2s = []

    # --- Phase 1: no_t4f (all stations) — also provides predictions for two-phase ---
    train_mask_all = (stationId_vals != held_sid) & ~np.isnan(y_all)
    X_tr_all = pd.DataFrame(X_all[np.where(train_mask_all)[0]], columns=FEAT_ALL)
    y_tr_all = y_res[train_mask_all]
    m_all = xgb.XGBRegressor(**params_base)
    m_all.fit(X_tr_all, y_tr_all)

    pred_res_all = m_all.predict(X_te)
    pred_phase1 = np.clip(np.expm1(pred_res_all + bm_global), 0, None)
    pred_mean_phase1 = float(np.nanmean(pred_phase1))
    phase1_predicted_means[held_sid] = pred_mean_phase1
    pred_tier_phase1 = assign_tier(pred_mean_phase1)

    # --- Selector streams: Ridge and ExtraTrees on raw features ---
    needs_selectors = any(c in CONFIGS for c in ["sel_ridge", "sel_extra"])
    pred_sel_ridge = None
    pred_sel_extra = None
    if needs_selectors:
        X_tr_sel = X_all[np.where(train_mask_all)[0]]
        y_tr_sel = y_res[train_mask_all]
        X_te_sel = X_all[test_idx]

        if "sel_ridge" in CONFIGS:
            pipe_ridge = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("reg", Ridge(alpha=10.0)),
            ])
            pipe_ridge.fit(X_tr_sel, y_tr_sel)
            pred_sel_ridge = np.clip(
                np.expm1(pipe_ridge.predict(X_te_sel) + bm_global), 0, None)

        if "sel_extra" in CONFIGS:
            pipe_extra = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("reg", ExtraTreesRegressor(
                    n_estimators=100, max_depth=4, min_samples_leaf=20,
                    random_state=44, n_jobs=-1)),
            ])
            pipe_extra.fit(X_tr_sel, y_tr_sel)
            pred_sel_extra = np.clip(
                np.expm1(pipe_extra.predict(X_te_sel) + bm_global), 0, None)

    needs_raw_general = any(c in CONFIGS for c in [
        "no_t4f_raw", "no_t4f_blend", "no_t4f_gated",
        "true_tier_moe_40", "true_tier_moe_guarded",
    ])
    pred_raw_general = None
    if needs_raw_general:
        y_tr_raw = y_all[train_mask_all]
        raw_weights = np.ones_like(y_tr_raw, dtype=float)
        raw_weights += (y_tr_raw >= 50).astype(float)
        raw_weights += 2.0 * (y_tr_raw >= 75).astype(float)
        raw_weights += 3.0 * (y_tr_raw >= 100).astype(float)
        raw_params = {
            **XGB_BASE,
            "booster": "gbtree",
            "objective": "reg:squarederror",
            "monotone_constraints": mono_full,
            "reg_lambda": 20.0,
        }
        m_raw = xgb.XGBRegressor(**raw_params)
        m_raw.fit(X_tr_all, y_tr_raw, sample_weight=raw_weights)
        pred_raw_general = np.clip(m_raw.predict(X_te), 0, None)

    needs_true_tier = any(c in CONFIGS for c in [
        "oracle_t4f", "true_tier_moe_expert", "true_tier_moe_40",
        "true_tier_moe_guarded", "true_tier_hard_pred",
        "tierexpert_t0", "tierexpert_t1", "tierexpert_t2", "tierexpert_t3",
    ])
    tier_expert_preds = {}
    if needs_true_tier:
        for tier in TIERS:
            same_tier = [s for s in others if sid_tier[s] == tier]
            train_sids = same_tier if len(same_tier) >= MIN_TIER_STATIONS else others
            train_mask = np.isin(stationId_vals, train_sids) & ~np.isnan(y_all)
            X_tr = pd.DataFrame(X_all[np.where(train_mask)[0]], columns=FEAT_ALL)
            y_tr = y_res[train_mask]
            m = xgb.XGBRegressor(**params_base)
            m.fit(X_tr, y_tr)
            tier_expert_preds[tier] = np.clip(np.expm1(m.predict(X_te) + bm_global), 0, None)

        probs = tier_prior_probs.get(held_sid)
        if probs is None:
            probs = fallback_tier_probs(pred_tier_phase1)
        probs = np.asarray(probs, dtype=float)
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(4, 0.25)
        tier_stack = np.vstack([tier_expert_preds[tier] for tier in TIERS])
        pred_true_tier_moe = np.clip(np.dot(probs, tier_stack), 0, None)
        hard_tier = TIERS[int(np.argmax(probs))]
        pred_true_tier_hard = tier_expert_preds[hard_tier].copy()

        if pred_raw_general is None:
            pred_base_gated = pred_phase1.copy()
        else:
            raw_weight = 0.55 if pred_mean_phase1 >= 20 else 0.15
            pred_base_gated = (1.0 - raw_weight) * pred_phase1 + raw_weight * pred_raw_general
        alpha_guarded = station_alpha_guarded(
            probs,
            float(np.nanmean(pred_base_gated)),
            float(np.nanmean(pred_true_tier_moe)),
        )
        pred_true_tier_40 = np.clip(0.60 * pred_base_gated + 0.40 * pred_true_tier_moe, 0, None)
        pred_true_tier_guarded = np.clip(
            (1.0 - alpha_guarded) * pred_base_gated + alpha_guarded * pred_true_tier_moe,
            0,
            None,
        )

    for cname in CONFIGS:
        if cname == "no_t4f":
            pred = pred_phase1.copy()

        elif cname == "no_t4f_raw":
            pred = pred_raw_general.copy()

        elif cname == "no_t4f_blend":
            pred = 0.70 * pred_phase1 + 0.30 * pred_raw_general

        elif cname == "no_t4f_gated":
            raw_weight = 0.55 if pred_mean_phase1 >= 20 else 0.15
            pred = (1.0 - raw_weight) * pred_phase1 + raw_weight * pred_raw_general

        elif cname == "true_tier_moe_expert":
            pred = pred_true_tier_moe.copy()

        elif cname == "true_tier_moe_40":
            pred = pred_true_tier_40.copy()

        elif cname == "true_tier_moe_guarded":
            pred = pred_true_tier_guarded.copy()

        elif cname == "true_tier_hard_pred":
            pred = pred_true_tier_hard.copy()

        elif cname.startswith("tierexpert_"):
            tier = cname.replace("tierexpert_", "")
            pred = tier_expert_preds[tier].copy()

        elif cname == "oracle_t4f":
            if tier_expert_preds:
                pred = tier_expert_preds[held_tier_true].copy()
            else:
                same_tier = [s for s in others if sid_tier[s] == held_tier_true]
                if len(same_tier) < MIN_TIER_STATIONS:
                    train_sids = others
                else:
                    train_sids = same_tier
                train_mask = np.isin(stationId_vals, train_sids) & ~np.isnan(y_all)
                X_tr = pd.DataFrame(X_all[np.where(train_mask)[0]], columns=FEAT_ALL)
                y_tr = y_res[train_mask]
                m = xgb.XGBRegressor(**params_base)
                m.fit(X_tr, y_tr)
                pred = np.clip(np.expm1(m.predict(X_te) + bm_global), 0, None)

        elif cname == "ghap_t4f":
            same_tier = [s for s in others if sid_ghap_tier[s] == held_tier_ghap]
            if len(same_tier) < MIN_TIER_STATIONS:
                train_sids = others
            else:
                train_sids = same_tier
            train_mask = np.isin(stationId_vals, train_sids) & ~np.isnan(y_all)
            X_tr = pd.DataFrame(X_all[np.where(train_mask)[0]], columns=FEAT_ALL)
            y_tr = y_res[train_mask]
            m = xgb.XGBRegressor(**params_base)
            m.fit(X_tr, y_tr)
            pred = np.clip(np.expm1(m.predict(X_te) + bm_global), 0, None)

        elif cname == "twophase_t4f":
            same_tier = [s for s in others if sid_tier[s] == pred_tier_phase1]
            if len(same_tier) < MIN_TIER_STATIONS:
                train_sids = others
            else:
                train_sids = same_tier
            train_mask = np.isin(stationId_vals, train_sids) & ~np.isnan(y_all)
            X_tr = pd.DataFrame(X_all[np.where(train_mask)[0]], columns=FEAT_ALL)
            y_tr = y_res[train_mask]
            m = xgb.XGBRegressor(**params_base)
            m.fit(X_tr, y_tr)
            pred = np.clip(np.expm1(m.predict(X_te) + bm_global), 0, None)

        elif cname.startswith("twophase_knn"):
            k_val = int(cname.split("knn")[1])
            pm_dists = [(s, abs(station_pm_means[s] - pred_mean_phase1)) for s in others]
            pm_dists.sort(key=lambda x: x[1])
            train_sids = [s for s, _ in pm_dists[:k_val]]
            train_mask = np.isin(stationId_vals, train_sids) & ~np.isnan(y_all)
            X_tr = pd.DataFrame(X_all[np.where(train_mask)[0]], columns=FEAT_ALL)
            y_tr = y_res[train_mask]
            m = xgb.XGBRegressor(**params_base)
            m.fit(X_tr, y_tr)
            pred = np.clip(np.expm1(m.predict(X_te) + bm_global), 0, None)

        elif cname == "sel_ridge":
            pred = pred_sel_ridge.copy()

        elif cname == "sel_extra":
            pred = pred_sel_extra.copy()

        pred_all[cname][mask_test] = pred
        r2 = r2_score(y_all[mask_test][valid_y], pred[valid_y]) if valid_y.sum() > 0 else np.nan
        fold_r2s.append(r2)

    fold_time = time.time() - t_fold
    remaining = fold_time * (n_stn - fold_i - 1)

    tier_match = "Y" if pred_tier_phase1 == held_tier_true else "N"
    r2_strs = " ".join(f"{FOLD_ABBREV[c]}={fold_r2s[i]:+.3f}"
                       for i, c in enumerate(CONFIGS))
    print(f"  [{fold_i+1:2d}/{n_stn}] {nm:35s} {held_tier_true} pm={pm_val:5.1f} "
          f"p1={pred_mean_phase1:5.1f}({pred_tier_phase1}{tier_match}) | "
          f"{r2_strs} ({fold_time:.0f}s, ETA {remaining/60:.0f}m)")

loso_time = time.time() - t0
print(f"\n  LOSO done: {loso_time:.0f}s")

# =============================================================================
#  PHASE 1 TIER CONCORDANCE
# =============================================================================
print(f"\n{'='*80}")
print("PHASE 1 TIER CONCORDANCE")
print(f"{'='*80}")

match_p1 = sum(1 for s in station_ids
               if assign_tier(phase1_predicted_means.get(s, 0)) == sid_tier[s])
match_ghap = sum(1 for s in station_ids if sid_ghap_tier[s] == sid_tier[s])
print(f"  Phase 1 prediction: {match_p1}/{n_stn} ({100*match_p1/n_stn:.0f}%)")
print(f"  GHAP:               {match_ghap}/{n_stn} ({100*match_ghap/n_stn:.0f}%)")

print(f"\n  {'Station':<35s} {'true':>5s} {'p1_pm':>6s} {'p1_t':>4s} {'ghap_t':>6s} {'ok_p1':>5s} {'ok_g':>5s}")
print("  " + "-" * 70)
for sid in sorted(station_ids, key=lambda s: station_pm_means[s]):
    nm = sid_name.get(sid, sid)[:34].encode('ascii', 'replace').decode()
    true_pm = station_pm_means[sid]
    tt = sid_tier[sid]
    p1_pm = phase1_predicted_means.get(sid, 0)
    p1_t = assign_tier(p1_pm)
    gt = sid_ghap_tier[sid]
    ok_p1 = "Y" if p1_t == tt else "N"
    ok_g = "Y" if gt == tt else "N"
    print(f"  {nm:<35s} {true_pm:5.1f} {p1_pm:6.1f} {p1_t:>4s} {gt:>6s} {ok_p1:>5s} {ok_g:>5s}")

# =============================================================================
#  RESULTS
# =============================================================================
print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")

print(f"\n  Overall R2_hourly (pooled):")
print(f"  {'Config':<18s} {'R2_h':>8s} {'RMSE':>7s} {'MAE':>7s} {'Bias':>8s} {'vs ORC':>8s}  note")
print("  " + "-" * 75)
r2_ref = None
for cname in CONFIGS:
    p = pred_all[cname]
    valid = ~np.isnan(p) & ~np.isnan(y_all)
    r2 = r2_score(y_all[valid], p[valid])
    rmse = np.sqrt(mean_squared_error(y_all[valid], p[valid]))
    mae = mean_absolute_error(y_all[valid], p[valid])
    bias = float(p[valid].mean() - y_all[valid].mean())
    if cname == "oracle_t4f":
        r2_ref = r2
    delta = r2 - r2_ref if r2_ref is not None else 0.0
    note = ""
    if cname == "oracle_t4f": note = "operationally INVALID"
    elif cname == "no_t4f": note = "valid, lower bound"
    else: note = "operationally valid"
    print(f"  {cname:<18s} {r2:+8.4f} {rmse:7.2f} {mae:7.2f} {bias:+8.3f} {delta:+8.4f}  {note}")

# Per-tier
print(f"\n  Per-tier R2_hourly:")
print(f"  {'Tier':<5s}", end="")
for c in CONFIGS:
    print(f" {FOLD_ABBREV[c]:>6s}", end="")
print()
print("  " + "-" * (5 + 7 * len(CONFIGS)))
for tier in ["t0", "t1", "t2", "t3"]:
    tier_sids = [s for s in station_ids if sid_tier[s] == tier]
    tier_m = np.isin(stationId_vals, tier_sids)
    print(f"  {tier:<5s}", end="")
    for cname in CONFIGS:
        p = pred_all[cname]
        valid = ~np.isnan(p) & ~np.isnan(y_all) & tier_m
        v = r2_score(y_all[valid], p[valid]) if valid.sum() >= 10 else np.nan
        print(f" {v:+6.3f}" if not np.isnan(v) else "    N/A", end="")
    print()

# Per-station
print(f"\n  Per-station R2:")
print(f"  {'Station':<28s} {'T':>2s} {'pm':>5s}", end="")
for c in CONFIGS:
    print(f" {FOLD_ABBREV[c]:>6s}", end="")
print()
print("  " + "-" * (28 + 2 + 5 + 7 * len(CONFIGS)))

station_results = []
for sid in station_ids:
    mask = stationId_vals == sid
    if mask.sum() < 10: continue
    r2s = {}
    for cname in CONFIGS:
        p = pred_all[cname][mask]
        valid = ~np.isnan(p) & ~np.isnan(y_all[mask])
        r2s[cname] = r2_score(y_all[mask][valid], p[valid]) if valid.sum() >= 10 else np.nan
    station_results.append((sid, r2s))

for sid, r2s in station_results:
    nm = sid_name.get(sid, sid)[:27]
    tier = sid_tier[sid]
    pm = float(station_pm_means[sid])
    print(f"  {nm:<28s} {tier:>2s} {pm:5.1f}", end="")
    for c in CONFIGS:
        v = r2s.get(c, np.nan)
        print(f" {v:+6.3f}" if not np.isnan(v) else "    N/A", end="")
    print()

# =============================================================================
#  SAVE
# =============================================================================
print(f"\n{'='*80}")
print("SAVING")
print(f"{'='*80}")

all_rows = []
for cname in CONFIGS:
    for sid, r2s_dict in station_results:
        mask = stationId_vals == sid
        p = pred_all[cname][mask]
        valid = ~np.isnan(p) & ~np.isnan(y_all[mask])
        ya = y_all[mask][valid]
        pp = p[valid]
        if len(ya) < 10: continue
        r2 = r2_score(ya, pp)
        rmse = np.sqrt(mean_squared_error(ya, pp))
        mae = mean_absolute_error(ya, pp)
        bias = float(pp.mean() - ya.mean())
        p1_pm = phase1_predicted_means.get(sid, np.nan)
        all_rows.append(dict(
            config=cname, station_id=sid,
            station_name=sid_name.get(sid, sid),
            region=sid_region.get(sid, "?"),
            tier=sid_tier[sid],
            pm25_mean=round(float(station_pm_means[sid]), 2),
            phase1_pred_mean=round(p1_pm, 2) if not np.isnan(p1_pm) else None,
            phase1_tier=assign_tier(p1_pm) if not np.isnan(p1_pm) else None,
            ghap_tier=sid_ghap_tier[sid],
            n_hours=int(valid.sum()),
            r2_hourly=round(r2, 4),
            rmse=round(rmse, 2),
            mae=round(mae, 2),
            bias=round(bias, 3),
        ))

out_df = pd.DataFrame(all_rows)
csv_path = os.path.join(OUT_DIR, f"{args.out_prefix}.csv")
out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Saved: {csv_path}")
print(f"  {len(out_df)} rows ({len(CONFIGS)} configs x {len(station_results)} stations)")

if args.save_oof:
    oof_path = os.path.join(OUT_DIR, f"{args.out_prefix}_oof.csv")
    if os.path.exists(oof_path):
        os.remove(oof_path)
    first = True
    for cname in CONFIGS:
        p = pred_all[cname]
        valid = ~np.isnan(p) & ~np.isnan(y_all)
        oof = pd.DataFrame({
            "config": cname,
            "station_id": stationId_vals[valid],
            "tier": [sid_tier[s] for s in stationId_vals[valid]],
            "ts": df.loc[valid, "ts"].astype(str).values,
            "y_true": y_all[valid],
            "y_pred": p[valid],
        })
        oof["residual"] = oof["y_pred"] - oof["y_true"]
        oof.to_csv(
            oof_path,
            index=False,
            encoding="utf-8-sig",
            mode="w" if first else "a",
            header=first,
        )
        first = False
    print(f"Saved OOF predictions: {oof_path}")

print(f"\nDONE — total time: {time.time()-t0:.0f}s")
