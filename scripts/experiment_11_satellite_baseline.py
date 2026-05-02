"""
Experiment 11: Satellite-Only vs Full Stage 1 Baselines in Full Prediction

Stage 1 baselines (Ridge LOO on per-station summary features):
  S1_full (Best-7): mean_PM25_nn_idw, AOT_valid_frac, mean_WS, mean_VC,
                     rain_freq, slope_deg, mean_AOT_grad_mag   (LOO R²≈0.70)
  S1_sat  (Best-7): mean_AOT_outer_mean, mean_AOT_inner_mean,
                     mean_AOT_grad_mag, latitude,
                     mean_SSA_inner_mean_clean, mean_SSA_grad_mag_clean,
                     mean_SSA_local_vs_regional_clean   (LOO R²≈0.60)

Configs:
  V1 — S1_full baseline as feature + B1 features (single-stage XGB)
  V2 — S1_sat  baseline as feature + B1 features (single-stage XGB)
  V3 — S1_full two-stage: target = PM2.5 - baseline, Stage 2 met+AOD+RFSI
  V4 — S1_sat  two-stage: same but satellite-only baseline
  V5 — S1_full baseline as feature + met+AOD+RFSI+buildings (no geography)

Output:
  analysis/thesis_experiments/experiment_11_satellite_baseline.md
  analysis/thesis_experiments/loso_per_station_exp11.csv
  analysis/thesis_experiments/kfold_exp11.csv
  analysis/thesis_experiments/stage1_diagnostics_exp11.csv
"""

import argparse, io, sys, os, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = args.data_dir or REPO_DIR
OUT_DIR = os.path.join(REPO_DIR, "analysis", "thesis_experiments")
os.makedirs(OUT_DIR, exist_ok=True)

K_NN = 5

XGB_PARAMS = dict(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    device="cuda", random_state=42, n_jobs=-1,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("EXPERIMENT 11: SATELLITE-ONLY VS FULL STAGE 1 BASELINES")
print("=" * 80)

t0 = time.time()
df = pd.read_csv(os.path.join(DATA_DIR, "data/merged/unified_thesis_v2.csv"),
                 dtype={"stationId": str})
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["month"] = df["ts"].dt.month
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations "
      f"({time.time()-t0:.1f}s)")

meta = pd.read_csv(os.path.join(DATA_DIR,
                    "analysis/thesis_audit/station_selection_final.csv"),
                    dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))
sid_region = dict(zip(meta["stationId"], meta["region"]))
sid_lat = dict(zip(meta["stationId"], meta["lat"]))
sid_lon = dict(zip(meta["stationId"], meta["lon"]))
station_ids = sorted(df["stationId"].unique())
n_stn = len(station_ids)
sid_to_idx = {s: i for i, s in enumerate(station_ids)}

TARGET = "PM2.5"
y_all = df[TARGET].values
stationId_vals = df["stationId"].values

# ═══════════════════════════════════════════════════════════════════════════════
#  JOIN BUILDING DENSITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Building density ---")
bld_path = os.path.join(DATA_DIR, "data/stations/metadata/station_building_density.csv")
if not os.path.exists(bld_path):
    bld_path = os.path.join(REPO_DIR, "data/stations/metadata/station_building_density.csv")
bld = pd.read_csv(bld_path, dtype={"stationId": str})
BUILDING_COLS = ["building_count_1km", "building_area_1km",
                 "building_count_3km", "building_area_3km"]
bld_map = bld.set_index("stationId")[BUILDING_COLS]
df = df.merge(bld_map, left_on="stationId", right_index=True, how="left")
for col in BUILDING_COLS:
    df[col] = df[col].fillna(0)
print(f"Building density loaded: {len(bld)} stations")

# ═══════════════════════════════════════════════════════════════════════════════
#  JOIN ACAG CLIMATOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- ACAG satellite climatology ---")
acag_path = os.path.join(DATA_DIR, "data/acag/acag_station_climatology.csv")
if not os.path.exists(acag_path):
    acag_path = os.path.join(REPO_DIR, "data/acag/acag_station_climatology.csv")
acag = pd.read_csv(acag_path, dtype={"stationId": str})
acag_monthly_cols_map = {m: f"ACAG_monthly_clim_{m:02d}" for m in range(1, 13)}
acag_long = acag.melt(
    id_vars=["stationId"], value_vars=list(acag_monthly_cols_map.values()),
    var_name="month_col", value_name="ACAG_monthly")
acag_long["month"] = acag_long["month_col"].str.extract(r"(\d{2})$").astype(int)
acag_long = acag_long[["stationId", "month", "ACAG_monthly"]]
df = df.merge(acag_long, on=["stationId", "month"], how="left")
n_acag = df["ACAG_monthly"].notna().sum()
print(f"ACAG joined: {n_acag:,}/{len(df):,} rows ({n_acag/len(df)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
#  STATION DISTANCES + RFSI
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Station distance matrix ---")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

coords = {s: (sid_lat[s], sid_lon[s]) for s in station_ids}
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

# ═══════════════════════════════════════════════════════════════════════════════
#  PM2.5 WIDE MATRIX + RFSI
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- PM2.5 wide matrix ---")
pm25_wide = df.pivot_table(index="ts", columns="stationId",
                           values="PM2.5", aggfunc="first")
pm25_mat = pm25_wide.values
sid_cols = list(pm25_wide.columns)
sid_to_col = {s: i for i, s in enumerate(sid_cols)}
ts_to_row = pd.Series(range(len(pm25_wide)), index=pm25_wide.index)
df["ts_row"] = df["ts"].map(ts_to_row).astype(int).values
print(f"Shape: {pm25_mat.shape[0]:,} timestamps x {pm25_mat.shape[1]} stations")

RFSI_COLS = ([f"PM25_nn{k+1}" for k in range(K_NN)] +
             [f"dist_nn{k+1}" for k in range(K_NN)] +
             ["n_neighbors_available", "PM25_nn_mean", "PM25_nn_idw"])

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

    n_avail = np.sum(~np.isnan(pm_nn), axis=1).astype(int)
    pm_mean = np.nanmean(pm_nn, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d_nn
        pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)

    out = {}
    for k in range(K):
        out[f"PM25_nn{k+1}"] = pm_nn[:, k]
        out[f"dist_nn{k+1}"] = d_nn[:, k]
    out["n_neighbors_available"] = n_avail
    out["PM25_nn_mean"] = pm_mean
    out["PM25_nn_idw"] = pm_idw
    return out

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Stage 1 definitions ---")

S1_FULL_FEATS = ["mean_PM25_nn_idw", "mean_AOT_valid_frac", "mean_WS",
                 "mean_VC", "rain_freq", "slope_deg", "mean_AOT_grad_mag"]

S1_SAT_FEATS = ["mean_AOT_outer_mean", "mean_AOT_inner_mean",
                "mean_AOT_grad_mag", "latitude",
                "mean_SSA_inner_mean_clean", "mean_SSA_grad_mag_clean",
                "mean_SSA_local_vs_regional_clean"]

def compute_stage1_features(rfsi_vals, s1_type="full"):
    """Compute per-station summary features for Stage 1."""
    rows = []
    for sid in station_ids:
        sdf_mask = stationId_vals == sid
        sdf = df.loc[sdf_mask]
        n_rows = sdf_mask.sum()
        if n_rows == 0:
            rows.append({})
            continue

        row = {}

        # IDW from RFSI
        idw_vals = rfsi_vals["PM25_nn_idw"][sdf_mask]
        row["mean_PM25_nn_idw"] = np.nanmean(idw_vals)

        # Met averages
        row["mean_WS"] = sdf["WS_local"].mean() if "WS_local" in sdf.columns else np.nan
        row["mean_VC"] = sdf["VC"].mean() if "VC" in sdf.columns else np.nan
        precip = sdf["precip_mm"].values if "precip_mm" in sdf.columns else np.zeros(n_rows)
        row["rain_freq"] = (precip > 0.1).sum() / n_rows

        # AOT averages
        if "AOT_valid_count" in sdf.columns:
            aot_vc = sdf["AOT_valid_count"].values
            total_obs = n_rows
            row["mean_AOT_valid_frac"] = np.nansum(aot_vc > 0) / total_obs
        else:
            row["mean_AOT_valid_frac"] = np.nan

        if "AOT_grad_mag" in sdf.columns:
            row["mean_AOT_grad_mag"] = sdf["AOT_grad_mag"].mean()
        else:
            row["mean_AOT_grad_mag"] = np.nan

        if "AOT_inner_mean" in sdf.columns:
            row["mean_AOT_inner_mean"] = sdf["AOT_inner_mean"].mean()
        else:
            row["mean_AOT_inner_mean"] = np.nan

        if "AOT_outer_mean" in sdf.columns:
            row["mean_AOT_outer_mean"] = sdf["AOT_outer_mean"].mean()
        else:
            row["mean_AOT_outer_mean"] = np.nan

        # Static
        row["slope_deg"] = sdf["slope_deg"].iloc[0] if "slope_deg" in sdf.columns else np.nan
        row["latitude"] = sid_lat.get(sid, np.nan)

        # SSA cleaned (filter values > 1 before averaging)
        for ssa_col, out_name in [
            ("SSA_inner_mean", "mean_SSA_inner_mean_clean"),
            ("SSA_grad_mag", "mean_SSA_grad_mag_clean"),
            ("SSA_local_vs_regional", "mean_SSA_local_vs_regional_clean"),
        ]:
            if ssa_col in sdf.columns:
                vals = sdf[ssa_col].values.copy()
                if ssa_col == "SSA_inner_mean":
                    vals[~((vals >= 0) & (vals <= 1.1))] = np.nan
                elif ssa_col == "SSA_grad_mag":
                    vals[np.abs(vals) > 0.5] = np.nan
                elif ssa_col == "SSA_local_vs_regional":
                    vals[np.abs(vals) > 0.5] = np.nan
                row[out_name] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan
            else:
                row[out_name] = np.nan

        rows.append(row)

    return pd.DataFrame(rows, index=station_ids)


def ridge_predict(X_train, y_train, X_test, alpha=1.0):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_tr, y_train)
    return ridge.predict(X_te)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 LOO DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Stage 1: LOO diagnostics ---")

print("Computing global RFSI ...")
t1 = time.time()
rfsi_global = compute_rfsi(exclude_sid=None, K=K_NN)
for col in RFSI_COLS:
    df[col] = rfsi_global[col]
print(f"Done ({time.time()-t1:.1f}s)")

station_pm_means = df.groupby("stationId")["PM2.5"].mean()
y_stage1 = station_pm_means.reindex(station_ids).values

s1_df = compute_stage1_features(rfsi_global)

for label, feats in [("S1_full", S1_FULL_FEATS), ("S1_sat", S1_SAT_FEATS)]:
    X_s1 = s1_df[feats].values
    loo_preds = np.zeros(n_stn)
    for i in range(n_stn):
        tr = np.concatenate([np.arange(0, i), np.arange(i+1, n_stn)])
        loo_preds[i] = ridge_predict(X_s1[tr], y_stage1[tr], X_s1[i:i+1])[0]
    r2 = r2_score(y_stage1, loo_preds)
    mae = mean_absolute_error(y_stage1, loo_preds)
    rmse = np.sqrt(mean_squared_error(y_stage1, loo_preds))
    print(f"  {label}: LOO R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════════

TEMPORAL = ["hour_sin", "hour_cos", "month_sin", "month_cos",
            "day_of_year_sin", "day_of_year_cos"]
MET = [
    "Temperature_final", "Humidity_final", "Pressure_final",
    "PBLH", "VC", "RH_factor",
    "wind_u", "wind_v", "wind_dir_sin", "wind_dir_cos",
    "WS_local", "wind_u_local", "wind_v_local",
    "wind_dir_sin_local", "wind_dir_cos_local",
    "dT_6h", "dRH_6h", "dWS_6h", "dP_6h",
    "precip_mm", "hrs_since_rain", "rain_sum_24h", "rain_sum_48h",
    "rain_days_7d", "consecutive_dry_days",
]
AOD = [
    "AOT", "AOT_mean", "AOT_inner_mean", "AOT_outer_mean",
    "RF", "SSA", "Uncertainty", "AE",
    "AOT_valid_count", "AOD_physics", "AOT_spatial_std", "AOT_local_vs_regional",
    "AOT_ffill_48h", "hours_since_valid_AOT",
    "AOT_lag_1h", "AOT_lag_3h", "AOT_lag_6h",
    "AOT_rolling_mean_6h", "AOT_rolling_mean_24h",
    "AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_grad_dir",
]
GEO = ["latitude", "longitude", "elevation_m", "slope_deg",
       "aspect_sin", "aspect_cos", "elev_x_PBLH", "elev_x_hour_sin"]

# B1 features = MET + AOD + GEO + TEMPORAL + RFSI + BUILDINGS
B1_FEATS = MET + AOD + GEO + TEMPORAL + RFSI_COLS + BUILDING_COLS

# V1: S1_full baseline as feature + B1 (single-stage)
FEATS_V1 = B1_FEATS + ["baseline_full"]

# V2: S1_sat baseline as feature + B1 (single-stage)
FEATS_V2 = B1_FEATS + ["baseline_sat"]

# V3: S1_full two-stage — met+AOD+RFSI+temporal (no geography, no buildings)
FEATS_V3 = MET + AOD + TEMPORAL + RFSI_COLS

# V4: S1_sat two-stage — same features
FEATS_V4 = MET + AOD + TEMPORAL + RFSI_COLS

# V5: S1_full baseline as feature + met+AOD+RFSI+buildings (no geography)
FEATS_V5 = MET + AOD + TEMPORAL + RFSI_COLS + BUILDING_COLS + ["baseline_full"]

CONFIGS = {
    "V1": {"feats": FEATS_V1, "stage": "single", "baseline": "full",
           "desc": "S1_full baseline + B1 features (single-stage)"},
    "V2": {"feats": FEATS_V2, "stage": "single", "baseline": "sat",
           "desc": "S1_sat baseline + B1 features (single-stage)"},
    "V3": {"feats": FEATS_V3, "stage": "two", "baseline": "full",
           "desc": "S1_full two-stage (met+AOD+RFSI, no geography)"},
    "V4": {"feats": FEATS_V4, "stage": "two", "baseline": "sat",
           "desc": "S1_sat two-stage (met+AOD+RFSI, no geography)"},
    "V5": {"feats": FEATS_V5, "stage": "single", "baseline": "full",
           "desc": "S1_full baseline + met+AOD+RFSI+buildings (no geo)"},
}
CONFIG_ORDER = ["V1", "V2", "V3", "V4", "V5"]

# Filter to available columns
for cname in CONFIG_ORDER:
    feats = CONFIGS[cname]["feats"]
    avail = [f for f in feats
             if f in df.columns or f in RFSI_COLS
             or f in ("baseline_full", "baseline_sat")]
    missing = [f for f in feats if f not in avail]
    if missing:
        print(f"WARNING: {cname} missing: {missing}")
    CONFIGS[cname]["feats"] = avail

for cn in CONFIG_ORDER:
    print(f"  {cn}: {len(CONFIGS[cn]['feats'])} features — {CONFIGS[cn]['desc']}")

# ═══════════════════════════════════════════════════════════════════════════════
#  KFOLD (global RFSI, global Stage 1)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("KFOLD 5-FOLD CV")
print(f"{'='*80}")

sid_mean_map = station_pm_means.to_dict()

# For KFold, use actual station mean as baseline (no LOSO leakage in random folds)
baselines_full_kf = df["stationId"].map(sid_mean_map).values
baselines_sat_kf = baselines_full_kf.copy()
df["baseline_full"] = baselines_full_kf
df["baseline_sat"] = baselines_sat_kf
residuals_full_kf = y_all - baselines_full_kf

kf_results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for cname in CONFIG_ORDER:
    cfg = CONFIGS[cname]
    feats = cfg["feats"]
    X = df[feats]

    if cfg["stage"] == "single":
        target_kf = y_all
    else:
        target_kf = residuals_full_kf

    print(f"\n--- {cname} ({len(feats)} features) ---")
    t1 = time.time()
    folds = []
    for _, (tr, va) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[tr], target_kf[tr])
        pred = m.predict(X.iloc[va])
        if cfg["stage"] == "two":
            pred = pred + baselines_full_kf[va]
        folds.append(dict(
            r2=r2_score(y_all[va], pred),
            rmse=np.sqrt(mean_squared_error(y_all[va], pred)),
            mae=mean_absolute_error(y_all[va], pred)))
    r2m = np.mean([f["r2"] for f in folds])
    rmsem = np.mean([f["rmse"] for f in folds])
    maem = np.mean([f["mae"] for f in folds])
    print(f"  R²={r2m:.4f}  RMSE={rmsem:.2f}  MAE={maem:.2f} ({time.time()-t1:.0f}s)")
    kf_results[cname] = dict(r2=round(r2m, 4), rmse=round(rmsem, 2), mae=round(maem, 2))

df.drop(columns=RFSI_COLS + ["baseline_full", "baseline_sat"], inplace=True,
        errors="ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOSO
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"LOSO CV ({n_stn} stations)")
print(f"{'='*80}")

all_base_feats = sorted(set(
    f for cn in CONFIG_ORDER for f in CONFIGS[cn]["feats"]
    if f not in RFSI_COLS and f not in ("baseline_full", "baseline_sat")))
base_arr = df[all_base_feats].values
base_col_map = {f: i for i, f in enumerate(all_base_feats)}
rfsi_col_map = {f: i for i, f in enumerate(RFSI_COLS)}

loso_results = {cn: [] for cn in CONFIG_ORDER}

for fold_i, held_sid in enumerate(station_ids):
    nm = sid_name.get(held_sid, held_sid)[:45]
    rg = sid_region.get(held_sid, "?")
    mask_test = stationId_vals == held_sid
    n_test = mask_test.sum()
    if n_test < 10:
        print(f"  [{fold_i+1:2d}/{n_stn}] {nm:45s} | SKIP (n={n_test})")
        continue

    mask_train = ~mask_test
    y_test = y_all[mask_test]
    y_train = y_all[mask_train]

    t_fold = time.time()

    # ── Recompute RFSI excluding held-out station ──
    train_sids = [s for s in station_ids if s != held_sid]
    held_idx = station_ids.index(held_sid)

    rfsi_fold = compute_rfsi(exclude_sid=held_sid, K=K_NN)
    rfsi_arr = np.column_stack([rfsi_fold[c] for c in RFSI_COLS])

    # ── Stage 1: compute summary features with LOSO RFSI ──
    s1_fold = compute_stage1_features(rfsi_fold)
    train_means = df.loc[mask_train].groupby("stationId")["PM2.5"].mean()
    y_s1_tr = np.array([train_means[s] for s in train_sids])
    s1_train_idx = [i for i in range(n_stn) if station_ids[i] != held_sid]

    # S1_full baseline
    X_s1_full = s1_fold[S1_FULL_FEATS].values
    held_baseline_full = ridge_predict(
        X_s1_full[s1_train_idx], y_s1_tr, X_s1_full[held_idx:held_idx+1])[0]

    # S1_sat baseline
    X_s1_sat = s1_fold[S1_SAT_FEATS].values
    held_baseline_sat = ridge_predict(
        X_s1_sat[s1_train_idx], y_s1_tr, X_s1_sat[held_idx:held_idx+1])[0]

    # Training baselines: actual station mean (clean residuals)
    train_baselines = np.array([train_means[s] for s in stationId_vals[mask_train]])
    train_residuals_full = y_train - train_baselines
    train_residuals_sat = y_train - train_baselines

    parts = [f"[{fold_i+1:2d}/{n_stn}] {nm:45s} |"
             f" full={held_baseline_full:5.1f}"
             f" sat={held_baseline_sat:5.1f}"
             f" (actual={station_pm_means[held_sid]:5.1f}) |"]

    for cname in CONFIG_ORDER:
        cfg = CONFIGS[cname]
        feats = cfg["feats"]
        baseline_type = cfg["baseline"]

        b_feats = [f for f in feats
                   if f not in RFSI_COLS and f not in ("baseline_full", "baseline_sat")]
        r_feats = [f for f in feats if f in RFSI_COLS]
        has_bl_full = "baseline_full" in feats
        has_bl_sat = "baseline_sat" in feats

        b_idx = [base_col_map[f] for f in b_feats] if b_feats else []
        r_idx = [rfsi_col_map[f] for f in r_feats] if r_feats else []

        arrays = []
        if b_idx:
            arrays.append(base_arr[:, b_idx])
        if r_idx:
            arrays.append(rfsi_arr[:, r_idx])

        if has_bl_full:
            bp = np.zeros(len(df))
            for s in train_sids:
                bp[stationId_vals == s] = train_means[s]
            bp[mask_test] = held_baseline_full
            arrays.append(bp.reshape(-1, 1))

        if has_bl_sat:
            bp = np.zeros(len(df))
            for s in train_sids:
                bp[stationId_vals == s] = train_means[s]
            bp[mask_test] = held_baseline_sat
            arrays.append(bp.reshape(-1, 1))

        X_all = np.hstack(arrays)
        X_tr = X_all[mask_train]
        X_te = X_all[mask_test]

        held_baseline = held_baseline_full if baseline_type == "full" else held_baseline_sat

        if cfg["stage"] == "single":
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X_tr, y_train)
            pm_pred = m.predict(X_te)
        else:
            train_resid = train_residuals_full if baseline_type == "full" else train_residuals_sat
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X_tr, train_resid)
            resid_pred = m.predict(X_te)
            pm_pred = resid_pred + held_baseline

        r2 = r2_score(y_test, pm_pred)
        rmse = np.sqrt(mean_squared_error(y_test, pm_pred))
        mae = mean_absolute_error(y_test, pm_pred)

        loso_results[cname].append(dict(
            station_id=held_sid,
            station_name=sid_name.get(held_sid, held_sid),
            region=rg, n_rows=n_test,
            r2=round(r2, 4), rmse=round(rmse, 2), mae=round(mae, 2),
            baseline_full=round(held_baseline_full, 2),
            baseline_sat=round(held_baseline_sat, 2),
            stage1_err_full=round(held_baseline_full - station_pm_means[held_sid], 2),
            stage1_err_sat=round(held_baseline_sat - station_pm_means[held_sid], 2)))

        parts.append(f"{cname}={r2:+.3f}")

    fold_time = time.time() - t_fold
    remaining = fold_time * (n_stn - fold_i - 1)
    print(f"  {' '.join(parts)}  [{rg}] ({fold_time:.0f}s, ETA {remaining/60:.0f}m)")

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE RAW RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

all_loso_rows = []
for cfg in CONFIG_ORDER:
    for r in loso_results[cfg]:
        all_loso_rows.append({"config": cfg, **r})
pd.DataFrame(all_loso_rows).to_csv(
    os.path.join(OUT_DIR, "loso_per_station_exp11.csv"),
    index=False, encoding="utf-8-sig")

pd.DataFrame([{"config": cfg, **kf_results[cfg]} for cfg in CONFIG_ORDER]).to_csv(
    os.path.join(OUT_DIR, "kfold_exp11.csv"),
    index=False, encoding="utf-8-sig")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD PREVIOUS RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
prev_loso = {}
prev_kf = {}
for exp_csv, exp_kf in [
    ("loso_per_station_exp07.csv", "kfold_exp07.csv"),
    ("loso_per_station_exp08.csv", "kfold_exp08.csv"),
    ("loso_per_station_exp10.csv", "kfold_exp10.csv"),
]:
    lp = os.path.join(OUT_DIR, exp_csv)
    kp = os.path.join(OUT_DIR, exp_kf)
    if os.path.exists(lp):
        tmp = pd.read_csv(lp, dtype={"station_id": str})
        for cfg in tmp["config"].unique():
            sub = tmp[tmp["config"] == cfg]
            prev_loso[cfg] = sub.to_dict("records")
    if os.path.exists(kp):
        tmp = pd.read_csv(kp)
        for _, row in tmp.iterrows():
            prev_kf[row["config"]] = dict(r2=row["r2"], rmse=row["rmse"], mae=row["mae"])

def loso_summary(results):
    rdf = pd.DataFrame(results)
    v = rdf.dropna(subset=["r2"])
    return dict(
        mean_r2=round(v["r2"].mean(), 4),
        median_r2=round(v["r2"].median(), 4),
        wmean_r2=round((v["r2"]*v["n_rows"]).sum()/v["n_rows"].sum(), 4),
        mean_rmse=round(v["rmse"].mean(), 2),
        mean_mae=round(v["mae"].mean(), 2),
        neg_count=int((v["r2"] < 0).sum()))

sums = {c: loso_summary(loso_results[c]) for c in CONFIG_ORDER}
prev_sums = {c: loso_summary(recs) for c, recs in prev_loso.items()}

b1_r2 = {r["station_id"]: r["r2"] for r in prev_loso.get("B1", [])}
t2_r2 = {r["station_id"]: r["r2"] for r in prev_loso.get("T2", [])}

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
diag_rows = []
for r in loso_results["V1"]:
    diag_rows.append({
        "station_id": r["station_id"],
        "station_name": r["station_name"],
        "region": r["region"],
        "pm25_actual": station_pm_means.get(r["station_id"], np.nan),
        "baseline_full": r["baseline_full"],
        "baseline_sat": r["baseline_sat"],
        "err_full": r["stage1_err_full"],
        "err_sat": r["stage1_err_sat"],
    })
diag_df = pd.DataFrame(diag_rows).sort_values("err_full", key=abs, ascending=False)
diag_df.to_csv(os.path.join(OUT_DIR, "stage1_diagnostics_exp11.csv"),
               index=False, encoding="utf-8-sig")

s1_full_r2 = r2_score(diag_df["pm25_actual"], diag_df["baseline_full"])
s1_sat_r2 = r2_score(diag_df["pm25_actual"], diag_df["baseline_sat"])
s1_full_mae = mean_absolute_error(diag_df["pm25_actual"], diag_df["baseline_full"])
s1_sat_mae = mean_absolute_error(diag_df["pm25_actual"], diag_df["baseline_sat"])
print(f"\nStage 1 LOSO diagnostics:")
print(f"  S1_full: R²={s1_full_r2:.4f}, MAE={s1_full_mae:.2f}")
print(f"  S1_sat:  R²={s1_sat_r2:.4f}, MAE={s1_sat_mae:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("GENERATING REPORT")
print(f"{'='*80}")

rpt = []
rpt.append("# Experiment 11: Satellite-Only vs Full Stage 1 Baselines\n")
rpt.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rpt.append(f"**Dataset:** unified_thesis_v2.csv — {len(df):,} rows, {n_stn} stations")
rpt.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, "
           f"lr=0.05, device=cuda\n")

rpt.append("## Stage 1 Baselines\n")
rpt.append("| Baseline | Features | LOSO R² | LOSO MAE |")
rpt.append("|----------|---------|:---:|:---:|")
rpt.append(f"| S1_full (Best-7) | {', '.join(S1_FULL_FEATS)} | {s1_full_r2:.4f} | {s1_full_mae:.2f} |")
rpt.append(f"| S1_sat (Best-7) | {', '.join(S1_SAT_FEATS)} | {s1_sat_r2:.4f} | {s1_sat_mae:.2f} |")

rpt.append("\n## Configs\n")
for cn in CONFIG_ORDER:
    cfg = CONFIGS[cn]
    rpt.append(f"- **{cn}**: {cfg['desc']}")

rpt.append("\n## Comparison Table\n")
rpt.append("| Config | Description | KFold R² | LOSO R² (mean) | "
           "LOSO R² (median) | LOSO MAE | Neg Stations |")
rpt.append("|--------|-------------|:---:|:---:|:---:|:---:|:---:|")

ref_cfgs = [
    ("B1", "Exp07: Full+RFSI+buildings"),
    ("T2", "Exp10: B1 + Ridge baseline"),
]
for cn, desc in ref_cfgs:
    if cn in prev_sums and cn in prev_kf:
        s = prev_sums[cn]
        kf = prev_kf[cn]
        rpt.append(f"| {cn} (ref) | {desc} | {kf['r2']:.4f} | "
                   f"{s['mean_r2']:.4f} | {s['median_r2']:.4f} | "
                   f"{s.get('mean_mae','—')} | {s['neg_count']} |")

for cn in CONFIG_ORDER:
    s = sums[cn]
    kf = kf_results[cn]
    desc = CONFIGS[cn]["desc"]
    rpt.append(f"| **{cn}** | {desc} | {kf['r2']:.4f} | "
               f"**{s['mean_r2']:.4f}** | {s['median_r2']:.4f} | "
               f"{s['mean_mae']:.2f} | {s['neg_count']} |")

# Per-station table for best config
best_cfg = max(CONFIG_ORDER, key=lambda c: sums[c]["mean_r2"])
best_df = pd.DataFrame(loso_results[best_cfg]).sort_values("r2", ascending=True)

rpt.append(f"\n## Per-Station LOSO: {best_cfg}\n")
rpt.append(f"| Station | Region | B1 R² | T2 R² | {best_cfg} R² | "
           f"Δ vs B1 | S1_full err | S1_sat err |")
rpt.append("|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|")

for _, r in best_df.iterrows():
    sid = r["station_id"]
    nm2 = str(r["station_name"])[:40]
    b1v = b1_r2.get(sid, np.nan)
    t2v = t2_r2.get(sid, np.nan)
    bv = r["r2"]

    b1_s = f"{b1v:.4f}" if pd.notna(b1v) else "—"
    t2_s = f"{t2v:.4f}" if pd.notna(t2v) else "—"
    delta = f"{bv - b1v:+.4f}" if pd.notna(b1v) else "—"

    rpt.append(f"| {nm2} | {r['region']} | {b1_s} | {t2_s} | "
               f"{bv:.4f} | {delta} | {r['stage1_err_full']:+.1f} | "
               f"{r['stage1_err_sat']:+.1f} |")

rpt.append("\n## Summary\n")
for cn in CONFIG_ORDER:
    s = sums[cn]
    rpt.append(f"- **{cn}**: LOSO R²={s['mean_r2']:.4f} "
               f"(median={s['median_r2']:.4f}), MAE={s['mean_mae']:.2f}, "
               f"{s['neg_count']} negative stations")

b1_mean = prev_sums.get("B1", {}).get("mean_r2")
t2_mean = prev_sums.get("T2", {}).get("mean_r2")
best_r2 = sums[best_cfg]["mean_r2"]
if b1_mean is not None:
    rpt.append(f"\n**Best ({best_cfg}) vs B1:** {best_r2 - b1_mean:+.4f}")
if t2_mean is not None:
    rpt.append(f"**Best ({best_cfg}) vs T2 (Exp10):** {best_r2 - t2_mean:+.4f}")

report_path = os.path.join(OUT_DIR, "experiment_11_satellite_baseline.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))

print(f"\nReport: {report_path}")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
