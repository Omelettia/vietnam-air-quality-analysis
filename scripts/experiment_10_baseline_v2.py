"""
Experiment 10: Improved Two-Stage with LOSO-Safe Baseline

Stage 1 — Ridge regression on 6 per-station summary features computed from
          the 39 training stations (mean_PM25_nn_idw, mean_PBLH, mean_VC,
          rain_freq, slope_deg, building_area_3km).
          Predicts station mean PM2.5 for the held-out station.

Configs:
  T1 — Two-stage: Ridge baseline + XGBoost residual.
       Stage 2 features: met + AOD + RFSI + temporal. NO geography, no buildings.
       Training rows use actual station mean as baseline (clean residuals).
  T2 — Single-stage: B1 features + Ridge-predicted baseline as extra feature.
       Target = PM2.5. Model uses baseline however it wants.
  T3 — Two-stage + buildings + ACAG_monthly in Stage 2.

Output:
  analysis/thesis_experiments/experiment_10_baseline_v2.md
  analysis/thesis_experiments/loso_per_station_exp10.csv
  analysis/thesis_experiments/kfold_exp10.csv
  analysis/thesis_experiments/stage1_diagnostics_exp10.csv
"""

import argparse, io, sys, os, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut
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
print("EXPERIMENT 10: IMPROVED TWO-STAGE WITH LOSO-SAFE BASELINE")
print("=" * 80)

t0 = time.time()
df = pd.read_csv(os.path.join(DATA_DIR, "data/merged/unified_thesis_v1.csv"),
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
#  PM2.5 WIDE MATRIX
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

# ═══════════════════════════════════════════════════════════════════════════════
#  RFSI COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
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
#  STAGE 1: PER-STATION SUMMARY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Stage 1: LOSO-safe station summary features ---")

STAGE1_FEATS = ["mean_PM25_nn_idw", "mean_PBLH", "mean_VC",
                "rain_freq", "slope_deg", "building_area_3km"]

station_pm_means = df.groupby("stationId")["PM2.5"].mean()

def compute_stage1_features(station_df, sid_list, rfsi_vals=None):
    """Compute per-station summary features from a dataframe."""
    rows = []
    for sid in sid_list:
        sdf = station_df[station_df["stationId"] == sid]
        n_rows = len(sdf)
        if n_rows == 0:
            rows.append({f: np.nan for f in STAGE1_FEATS})
            continue

        idw_col = "PM25_nn_idw"
        if rfsi_vals is not None and idw_col in rfsi_vals:
            mask = stationId_vals == sid
            idw_series = pd.Series(rfsi_vals[idw_col][mask])
        elif idw_col in sdf.columns:
            idw_series = sdf[idw_col]
        else:
            idw_series = pd.Series([np.nan] * n_rows)

        precip = sdf["precip_mm"]
        rows.append({
            "mean_PM25_nn_idw": idw_series.mean(),
            "mean_PBLH": sdf["PBLH"].mean(),
            "mean_VC": sdf["VC"].mean(),
            "rain_freq": (precip > 0.1).sum() / n_rows if n_rows > 0 else 0,
            "slope_deg": sdf["slope_deg"].iloc[0] if "slope_deg" in sdf.columns else np.nan,
            "building_area_3km": sdf["building_area_3km"].iloc[0] if "building_area_3km" in sdf.columns else 0,
        })
    return pd.DataFrame(rows, index=sid_list)

# Compute global RFSI for KFold and global Stage 1
print("Computing global RFSI ...")
t1 = time.time()
rfsi_global = compute_rfsi(exclude_sid=None, K=K_NN)
for col in RFSI_COLS:
    df[col] = rfsi_global[col]
print(f"Done ({time.time()-t1:.1f}s)")

stage1_df = compute_stage1_features(df, station_ids)
X_stage1_all = stage1_df[STAGE1_FEATS].values
y_stage1_all = station_pm_means.reindex(station_ids).values

print(f"Stage 1 data: {n_stn} stations, {len(STAGE1_FEATS)} features")
print(f"PM2.5 means: min={y_stage1_all.min():.1f}, max={y_stage1_all.max():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 LOO DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Stage 1: Leave-One-Out diagnostic ---")

loo_preds = np.zeros(n_stn)
for i in range(n_stn):
    tr = np.concatenate([np.arange(0, i), np.arange(i+1, n_stn)])
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_stage1_all[tr])
    X_te = scaler.transform(X_stage1_all[i:i+1])
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_stage1_all[tr])
    loo_preds[i] = ridge.predict(X_te)[0]

stage1_loo_r2 = r2_score(y_stage1_all, loo_preds)
stage1_loo_rmse = np.sqrt(mean_squared_error(y_stage1_all, loo_preds))
stage1_loo_mae = mean_absolute_error(y_stage1_all, loo_preds)
print(f"Stage 1 LOO: R²={stage1_loo_r2:.4f}, RMSE={stage1_loo_rmse:.2f}, "
      f"MAE={stage1_loo_mae:.2f}")

stage1_diag = pd.DataFrame({
    "station_id": station_ids,
    "station_name": [sid_name.get(s, s) for s in station_ids],
    "region": [sid_region.get(s, "?") for s in station_ids],
    "pm25_actual_mean": y_stage1_all,
    "pm25_predicted_mean": np.round(loo_preds, 2),
    "error": np.round(loo_preds - y_stage1_all, 2),
    "abs_error": np.round(np.abs(loo_preds - y_stage1_all), 2),
}).sort_values("abs_error", ascending=False)
stage1_diag.to_csv(os.path.join(OUT_DIR, "stage1_diagnostics_exp10.csv"),
                    index=False, encoding="utf-8-sig")

print("\nWorst predictions:")
for _, r in stage1_diag.head(10).iterrows():
    nm = str(r["station_name"])[:45]
    print(f"  {nm:45s} | actual={r['pm25_actual_mean']:5.1f} | "
          f"pred={r['pm25_predicted_mean']:5.1f} | err={r['error']:+6.1f}")

# Full model coefficients
scaler_full = StandardScaler()
X_s1_scaled = scaler_full.fit_transform(X_stage1_all)
ridge_full = Ridge(alpha=1.0)
ridge_full.fit(X_s1_scaled, y_stage1_all)
print("\nRidge coefficients (standardized):")
for fname, coef in zip(STAGE1_FEATS, ridge_full.coef_):
    print(f"  {fname:25s}: {coef:+.3f}")
print(f"  {'intercept':25s}: {ridge_full.intercept_:.3f}")

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

# T1: met + AOD + RFSI + temporal (NO geography, NO buildings)
FEATURES_T1 = MET + AOD + TEMPORAL + RFSI_COLS

# T2: B1 features + baseline_pred (single-stage)
FEATURES_T2_BASE = MET + AOD + GEO + TEMPORAL + RFSI_COLS + BUILDING_COLS
FEATURES_T2 = FEATURES_T2_BASE + ["baseline_pred"]

# T3: T1 + buildings + ACAG_monthly
FEATURES_T3 = MET + AOD + TEMPORAL + RFSI_COLS + BUILDING_COLS + ["ACAG_monthly"]

CONFIGS = {"T1": FEATURES_T1, "T2": FEATURES_T2, "T3": FEATURES_T3}
CONFIG_ORDER = ["T1", "T2", "T3"]

for cname in CONFIG_ORDER:
    feats = CONFIGS[cname]
    avail = [f for f in feats if f in df.columns or f in RFSI_COLS or f == "baseline_pred"]
    missing = [f for f in feats if f not in avail]
    if missing:
        print(f"WARNING: {cname} missing: {missing}")
    CONFIGS[cname] = avail

for cn in CONFIG_ORDER:
    print(f"  {cn}: {len(CONFIGS[cn])} features")

# ═══════════════════════════════════════════════════════════════════════════════
#  KFOLD (global RFSI)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("KFOLD 5-FOLD CV")
print(f"{'='*80}")

sid_mean_map = station_pm_means.to_dict()
baselines_kf = df["stationId"].map(sid_mean_map).values
residuals_kf = y_all - baselines_kf

# For T2 KFold, baseline_pred = actual station mean (no leakage in random KFold)
df["baseline_pred"] = baselines_kf

kf_results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for cname in CONFIG_ORDER:
    feats = CONFIGS[cname]
    if cname == "T2":
        X = df[feats]
        target_kf = y_all
    else:
        X = df[feats]
        target_kf = residuals_kf

    print(f"\n--- {cname} ({len(feats)} features) ---")
    t1 = time.time()
    folds = []
    for _, (tr, va) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[tr], target_kf[tr])
        pred = m.predict(X.iloc[va])
        if cname != "T2":
            pred = pred + baselines_kf[va]
        folds.append(dict(
            r2=r2_score(y_all[va], pred),
            rmse=np.sqrt(mean_squared_error(y_all[va], pred)),
            mae=mean_absolute_error(y_all[va], pred)))
    r2m = np.mean([f["r2"] for f in folds])
    rmsem = np.mean([f["rmse"] for f in folds])
    maem = np.mean([f["mae"] for f in folds])
    print(f"  R²={r2m:.4f}  RMSE={rmsem:.2f}  MAE={maem:.2f} ({time.time()-t1:.0f}s)")
    kf_results[cname] = dict(r2=round(r2m, 4), rmse=round(rmsem, 2), mae=round(maem, 2))

df.drop(columns=RFSI_COLS + ["baseline_pred"], inplace=True, errors="ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOSO
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"LOSO CV ({n_stn} stations)")
print(f"{'='*80}")

all_base_feats = sorted(set(
    f for cn in CONFIG_ORDER for f in CONFIGS[cn]
    if f not in RFSI_COLS and f != "baseline_pred"))
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

    # ── Stage 1: compute summary features from 39 training stations ──
    train_sids = [s for s in station_ids if s != held_sid]
    held_idx = station_ids.index(held_sid)

    # Recompute RFSI excluding held-out station
    rfsi_fold = compute_rfsi(exclude_sid=held_sid, K=K_NN)
    rfsi_arr = np.column_stack([rfsi_fold[c] for c in RFSI_COLS])

    # Compute Stage 1 features for all stations using LOSO RFSI
    s1_features = []
    for sid in station_ids:
        sdf_mask = stationId_vals == sid
        sdf = df.loc[sdf_mask]
        n_rows = sdf_mask.sum()
        if n_rows == 0:
            s1_features.append({f: np.nan for f in STAGE1_FEATS})
            continue
        idw_vals = rfsi_fold["PM25_nn_idw"][sdf_mask]
        precip = sdf["precip_mm"].values
        s1_features.append({
            "mean_PM25_nn_idw": np.nanmean(idw_vals),
            "mean_PBLH": sdf["PBLH"].mean(),
            "mean_VC": sdf["VC"].mean(),
            "rain_freq": (precip > 0.1).sum() / n_rows,
            "slope_deg": sdf["slope_deg"].iloc[0],
            "building_area_3km": sdf["building_area_3km"].iloc[0],
        })
    s1_df = pd.DataFrame(s1_features, index=station_ids)
    X_s1 = s1_df[STAGE1_FEATS].values

    # Train station means from training data
    train_means = df.loc[mask_train].groupby("stationId")["PM2.5"].mean()
    y_s1_tr = np.array([train_means[s] for s in train_sids])

    s1_train_idx = [i for i in range(n_stn) if station_ids[i] != held_sid]

    scaler_s1 = StandardScaler()
    X_s1_tr = scaler_s1.fit_transform(X_s1[s1_train_idx])
    X_s1_te = scaler_s1.transform(X_s1[held_idx:held_idx+1])

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_s1_tr, y_s1_tr)
    held_baseline = ridge.predict(X_s1_te)[0]

    # Training residuals: actual station mean as baseline (clean)
    train_baselines = np.array([train_means[s] for s in stationId_vals[mask_train]])
    train_residuals = y_train - train_baselines

    parts = [f"[{fold_i+1:2d}/{n_stn}] {nm:45s} |"
             f" baseline={held_baseline:5.1f}"
             f" (actual={station_pm_means[held_sid]:5.1f},"
             f" err={held_baseline - station_pm_means[held_sid]:+5.1f}) |"]

    for cname in CONFIG_ORDER:
        feats = CONFIGS[cname]
        b_feats = [f for f in feats if f not in RFSI_COLS and f != "baseline_pred"]
        r_feats = [f for f in feats if f in RFSI_COLS]
        has_baseline_feat = "baseline_pred" in feats

        b_idx = [base_col_map[f] for f in b_feats] if b_feats else []
        r_idx = [rfsi_col_map[f] for f in r_feats] if r_feats else []

        arrays = []
        if b_idx:
            arrays.append(base_arr[:, b_idx])
        if r_idx:
            arrays.append(rfsi_arr[:, r_idx])

        if has_baseline_feat:
            # T2: add baseline_pred as extra column
            # Training: use actual station mean; test: use Ridge prediction
            bp = np.zeros(len(df))
            for s in train_sids:
                bp[stationId_vals == s] = train_means[s]
            bp[mask_test] = held_baseline
            arrays.append(bp.reshape(-1, 1))

        X_all = np.hstack(arrays)
        X_tr = X_all[mask_train]
        X_te = X_all[mask_test]

        if cname == "T2":
            # Single-stage: target = PM2.5
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X_tr, y_train)
            pm_pred = m.predict(X_te)
        else:
            # Two-stage: target = residual
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X_tr, train_residuals)
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
            stage1_baseline=round(held_baseline, 2),
            stage1_error=round(held_baseline - station_pm_means[held_sid], 2)))

        parts.append(f"{cname}={r2:+.3f}")

    fold_time = time.time() - t_fold
    remaining = fold_time * (n_stn - fold_i - 1)
    print(f"  {' '.join(parts)}  [{rg}] ({fold_time:.0f}s, ETA {remaining/60:.0f}m)")

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

all_loso_rows = []
for cfg in CONFIG_ORDER:
    for r in loso_results[cfg]:
        all_loso_rows.append({"config": cfg, **r})
pd.DataFrame(all_loso_rows).to_csv(
    os.path.join(OUT_DIR, "loso_per_station_exp10.csv"),
    index=False, encoding="utf-8-sig")

pd.DataFrame([{"config": cfg, **kf_results[cfg]} for cfg in CONFIG_ORDER]).to_csv(
    os.path.join(OUT_DIR, "kfold_exp10.csv"),
    index=False, encoding="utf-8-sig")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD PREVIOUS RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
prev_loso = {}
prev_kf = {}
for exp_csv, exp_kf in [
    ("loso_per_station_exp07.csv", "kfold_exp07.csv"),
    ("loso_per_station_exp08.csv", "kfold_exp08.csv"),
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
        neg_count=int((v["r2"] < 0).sum()))

sums = {c: loso_summary(loso_results[c]) for c in CONFIG_ORDER}
prev_sums = {}
for c, recs in prev_loso.items():
    prev_sums[c] = loso_summary(recs)

b1_r2 = {}
if "B1" in prev_loso:
    b1_r2 = {r["station_id"]: r["r2"] for r in prev_loso["B1"]}
s1_r2 = {}
if "S1" in prev_loso:
    s1_r2 = {r["station_id"]: r["r2"] for r in prev_loso["S1"]}

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("GENERATING REPORT")
print(f"{'='*80}")

rpt = []
rpt.append("# Experiment 10: Improved Two-Stage with LOSO-Safe Baseline\n")
rpt.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rpt.append(f"**Dataset:** {len(df):,} rows, {n_stn} stations")
rpt.append(f"**Stage 1:** Ridge on 6 LOSO-safe features (α=1.0)")
rpt.append(f"**Stage 1 LOO R²:** {stage1_loo_r2:.4f}")
rpt.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, "
           f"lr=0.05, device=cuda\n")

rpt.append("## Stage 1 Features\n")
rpt.append("| Feature | Coefficient (standardized) |")
rpt.append("|---------|:---:|")
for fname, coef in zip(STAGE1_FEATS, ridge_full.coef_):
    rpt.append(f"| {fname} | {coef:+.3f} |")

rpt.append("\n## Comparison Table\n")
rpt.append("| Config | Description | KFold R² | LOSO R² (mean) | "
           "LOSO R² (median) | Neg Stations |")
rpt.append("|--------|-------------|:---:|:---:|:---:|:---:|")

ref_cfgs = [
    ("B1 (Exp07)", "Full+RFSI+buildings"),
    ("S1 (Exp08)", "Two-stage: static Ridge + XGB"),
]
for label, desc in ref_cfgs:
    cn = label.split(" ")[0]
    if cn in prev_sums and cn in prev_kf:
        s = prev_sums[cn]
        kf = prev_kf[cn]
        rpt.append(f"| {label} | {desc} | {kf['r2']:.4f} | "
                   f"{s['mean_r2']:.4f} | {s['median_r2']:.4f} | {s['neg_count']} |")

descs = {
    "T1": "Two-stage: LOSO-safe Ridge + XGB (met+AOD+RFSI)",
    "T2": "B1 + Ridge baseline as feature",
    "T3": "Two-stage + buildings + ACAG_monthly",
}
for cn in CONFIG_ORDER:
    s = sums[cn]
    kf = kf_results[cn]
    bold = "**" if s["mean_r2"] == max(sums[c]["mean_r2"] for c in CONFIG_ORDER) else ""
    rpt.append(f"| {bold}{cn}{bold} | {bold}{descs[cn]}{bold} | {kf['r2']:.4f} | "
               f"{bold}{s['mean_r2']:.4f}{bold} | {s['median_r2']:.4f} | {s['neg_count']} |")

# Per-station table for best config
best_cfg = max(CONFIG_ORDER, key=lambda c: sums[c]["mean_r2"])
best_df = pd.DataFrame(loso_results[best_cfg]).sort_values("r2", ascending=True)

rpt.append(f"\n## Per-Station LOSO: {best_cfg} vs B1 vs S1\n")
rpt.append(f"| Station | Region | B1 R² | S1 R² | {best_cfg} R² | "
           f"Δ vs B1 | Stage1 err | {best_cfg} RMSE |")
rpt.append("|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|")

for _, r in best_df.iterrows():
    sid = r["station_id"]
    nm = str(r["station_name"])[:40]
    b1v = b1_r2.get(sid, np.nan)
    s1v = s1_r2.get(sid, np.nan)
    bv = r["r2"]
    s1_err = r.get("stage1_error", np.nan)

    b1_s = f"{b1v:.4f}" if pd.notna(b1v) else "—"
    s1_s = f"{s1v:.4f}" if pd.notna(s1v) else "—"
    delta = f"{bv - b1v:+.4f}" if pd.notna(b1v) else "—"
    s1_err_s = f"{s1_err:+.1f}" if pd.notna(s1_err) else "—"

    rpt.append(f"| {nm} | {r['region']} | {b1_s} | {s1_s} | "
               f"{bv:.4f} | {delta} | {s1_err_s} | {r['rmse']:.1f} |")

# Summary
rpt.append("\n## Summary\n")
for cn in CONFIG_ORDER:
    s = sums[cn]
    rpt.append(f"- **{cn}**: LOSO R²={s['mean_r2']:.4f} (median={s['median_r2']:.4f}), "
               f"RMSE={s['mean_rmse']:.2f}, {s['neg_count']} negative stations")

b1_mean = prev_sums.get("B1", {}).get("mean_r2")
s1_mean = prev_sums.get("S1", {}).get("mean_r2")
best_t = max(CONFIG_ORDER, key=lambda c: sums[c]["mean_r2"])
best_r2 = sums[best_t]["mean_r2"]
if b1_mean is not None:
    rpt.append(f"\n**Best config ({best_t}) vs B1:** {best_r2 - b1_mean:+.4f}")
if s1_mean is not None:
    rpt.append(f"**Best config ({best_t}) vs S1 (Exp08):** {best_r2 - s1_mean:+.4f}")

report_path = os.path.join(OUT_DIR, "experiment_10_baseline_v2.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))

print(f"\nReport: {report_path}")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
