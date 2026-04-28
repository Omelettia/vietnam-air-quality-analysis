"""
Experiment 08: Two-Stage Architecture

Stage 1 — Ridge regression predicts station-mean PM2.5 from static features
          (building density, elevation, slope, lat, lon).  N=39 per LOSO fold.
Stage 2 — XGBoost predicts the residual (PM2.5 - baseline) from met + AOD + RFSI.
          NO geographic or building features — Stage 1 handles the spatial baseline.

Configs:
  S1 — Ridge baseline + XGBoost residual (met + AOD + RFSI + temporal)
  S2 — Ridge baseline + XGBoost residual (met + RFSI + temporal, no AOD)
  S3 — Ridge baseline only (predict station mean for every hour, no XGBoost)

Output:
  analysis/thesis_experiments/experiment_08_twostage.md
  analysis/thesis_experiments/loso_per_station_exp08.csv
  analysis/thesis_experiments/kfold_exp08.csv
  analysis/thesis_experiments/stage1_diagnostics_exp08.csv
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
parser.add_argument("--data-dir", default=None,
                    help="Base directory containing data/ and analysis/ folders")
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
print("EXPERIMENT 08: TWO-STAGE ARCHITECTURE")
print("=" * 80)

t0 = time.time()
df = pd.read_csv(os.path.join(DATA_DIR, "data/merged/unified_thesis_v1.csv"),
                  dtype={"stationId": str})
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
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

# ═══════════════════════════════════════════════════════════════════════════════
#  JOIN BUILDING DENSITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Building density features ---")

bld_path = os.path.join(DATA_DIR, "data/stations/metadata/station_building_density.csv")
if not os.path.exists(bld_path):
    bld_path = os.path.join(REPO_DIR, "data/stations/metadata/station_building_density.csv")
bld = pd.read_csv(bld_path, dtype={"stationId": str})
BUILDING_COLS = ["building_count_1km", "building_area_1km",
                 "building_count_3km", "building_area_3km"]

bld_map = bld.set_index("stationId")[BUILDING_COLS]
print(f"Building density loaded: {len(bld)} stations")

# ═══════════════════════════════════════════════════════════════════════════════
#  STATION DISTANCES + RFSI SETUP
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

nn1 = [neighbor_order[i][0][1] for i in range(n_stn)]
print(f"NN1 distances: min={min(nn1):.0f}km, median={np.median(nn1):.0f}km, "
      f"max={max(nn1):.0f}km")

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
print(f"Non-NaN: {(~np.isnan(pm25_mat)).mean():.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
#  RFSI COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

RFSI_COLS = ([f"PM25_nn{k+1}" for k in range(K_NN)] +
             [f"dist_nn{k+1}" for k in range(K_NN)] +
             ["n_neighbors_available", "PM25_nn_mean", "PM25_nn_idw"])


def compute_rfsi(exclude_sid=None, K=5):
    """Compute RFSI features for every row in df."""
    n = len(df)
    pm_nn = np.full((n, K), np.nan)
    d_nn = np.full((n, K), np.nan)

    excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
    stationId_vals = df["stationId"].values
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
#  STAGE 1 SETUP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Stage 1: Station-mean prediction setup ---")

STAGE1_FEATS = ["building_count_1km", "building_area_1km",
                "building_count_3km", "building_area_3km",
                "elevation_m", "slope_deg", "latitude", "longitude"]

station_static = pd.DataFrame({"stationId": station_ids})
station_static["elevation_m"] = station_static["stationId"].map(
    df.groupby("stationId")["elevation_m"].first())
station_static["slope_deg"] = station_static["stationId"].map(
    df.groupby("stationId")["slope_deg"].first())
station_static["latitude"] = station_static["stationId"].map(sid_lat)
station_static["longitude"] = station_static["stationId"].map(sid_lon)

for col in BUILDING_COLS:
    mapping = bld.set_index("stationId")[col]
    station_static[col] = station_static["stationId"].map(mapping).fillna(0)

station_pm_means = df.groupby("stationId")["PM2.5"].mean()
station_static["pm25_mean"] = station_static["stationId"].map(station_pm_means)

X_stage1_all = station_static[STAGE1_FEATS].values
y_stage1_all = station_static["pm25_mean"].values
stage1_sids = station_static["stationId"].values

print(f"Stage 1 data: {len(station_static)} stations, {len(STAGE1_FEATS)} features")
print(f"PM2.5 means: min={y_stage1_all.min():.1f}, max={y_stage1_all.max():.1f}, "
      f"std={y_stage1_all.std():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 LOO DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Stage 1: Leave-One-Out diagnostic ---")

loo = LeaveOneOut()
loo_preds = np.zeros(n_stn)
loo_actuals = y_stage1_all.copy()

for train_idx, test_idx in loo.split(X_stage1_all):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_stage1_all[train_idx])
    X_te = scaler.transform(X_stage1_all[test_idx])
    y_tr = y_stage1_all[train_idx]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    loo_preds[test_idx] = ridge.predict(X_te)

stage1_loo_r2 = r2_score(loo_actuals, loo_preds)
stage1_loo_rmse = np.sqrt(mean_squared_error(loo_actuals, loo_preds))
stage1_loo_mae = mean_absolute_error(loo_actuals, loo_preds)

print(f"Stage 1 LOO: R²={stage1_loo_r2:.4f}, RMSE={stage1_loo_rmse:.2f}, "
      f"MAE={stage1_loo_mae:.2f}")

stage1_diag = pd.DataFrame({
    "station_id": stage1_sids,
    "station_name": [sid_name.get(s, s) for s in stage1_sids],
    "region": [sid_region.get(s, "?") for s in stage1_sids],
    "pm25_actual_mean": loo_actuals,
    "pm25_predicted_mean": np.round(loo_preds, 2),
    "error": np.round(loo_preds - loo_actuals, 2),
    "abs_error": np.round(np.abs(loo_preds - loo_actuals), 2),
}).sort_values("abs_error", ascending=False)
stage1_diag.to_csv(os.path.join(OUT_DIR, "stage1_diagnostics_exp08.csv"),
                    index=False, encoding="utf-8-sig")

print("\nStage 1 LOO — worst predictions:")
for _, r in stage1_diag.head(10).iterrows():
    nm = r["station_name"][:45]
    print(f"  {nm:45s} | actual={r['pm25_actual_mean']:5.1f} | "
          f"pred={r['pm25_predicted_mean']:5.1f} | "
          f"err={r['error']:+6.1f}")

# Also fit a full model to inspect coefficients
scaler_full = StandardScaler()
X_s1_scaled = scaler_full.fit_transform(X_stage1_all)
ridge_full = Ridge(alpha=1.0)
ridge_full.fit(X_s1_scaled, y_stage1_all)
print("\nStage 1 Ridge coefficients (standardized):")
for fname, coef in zip(STAGE1_FEATS, ridge_full.coef_):
    print(f"  {fname:25s}: {coef:+.3f}")
print(f"  {'intercept':25s}: {ridge_full.intercept_:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 FEATURE SETS
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

# S1: met + AOD + RFSI + temporal
FEATURES_S1 = MET + AOD + TEMPORAL + RFSI_COLS
# S2: met + RFSI + temporal (no AOD)
FEATURES_S2 = MET + TEMPORAL + RFSI_COLS
# S3: Ridge-only — no Stage 2 features needed

CONFIGS = {"S1": FEATURES_S1, "S2": FEATURES_S2}
CONFIG_ORDER = ["S1", "S2", "S3"]

for cname in ["S1", "S2"]:
    feats = CONFIGS[cname]
    missing = [f for f in feats if f not in df.columns and f not in RFSI_COLS]
    if missing:
        print(f"WARNING: {cname} missing columns: {missing}")
        CONFIGS[cname] = [f for f in feats if f in df.columns or f in RFSI_COLS]

for cn in ["S1", "S2"]:
    nb = len([f for f in CONFIGS[cn] if f not in RFSI_COLS])
    nr = len([f for f in CONFIGS[cn] if f in RFSI_COLS])
    print(f"  {cn}: {len(CONFIGS[cn])} features ({nb} base + {nr} RFSI)")
print(f"  S3: 0 features (Ridge baseline only)")

# ═══════════════════════════════════════════════════════════════════════════════
#  KFOLD (global RFSI, S1 and S2 only — S3 has no KFold equivalent)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("KFOLD 5-FOLD CV (Stage 2 residual prediction, global RFSI)")
print(f"{'='*80}")

print("\nComputing global RFSI features ...")
t1 = time.time()
rfsi_global = compute_rfsi(exclude_sid=None, K=K_NN)
print(f"Done ({time.time()-t1:.1f}s)")

for col in RFSI_COLS:
    df[col] = rfsi_global[col]

# For KFold: use each station's actual mean as baseline (no leakage issue in
# random KFold since station appears in both train and test splits)
sid_mean_map = station_pm_means.to_dict()
baselines_kf = df["stationId"].map(sid_mean_map).values
residuals_kf = y_all - baselines_kf

kf_results = {}
for cname in ["S1", "S2"]:
    feats = CONFIGS[cname]
    X = df[feats]
    print(f"\n--- {cname} ({len(feats)} features, target=residual) ---")
    t1 = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for _, (tr, va) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[tr], residuals_kf[tr])
        resid_pred = m.predict(X.iloc[va])
        pm_pred = resid_pred + baselines_kf[va]
        folds.append(dict(
            r2=r2_score(y_all[va], pm_pred),
            rmse=np.sqrt(mean_squared_error(y_all[va], pm_pred)),
            mae=mean_absolute_error(y_all[va], pm_pred)))
    r2m = np.mean([f["r2"] for f in folds])
    rmsem = np.mean([f["rmse"] for f in folds])
    maem = np.mean([f["mae"] for f in folds])
    print(f"  R²={r2m:.4f}  RMSE={rmsem:.2f}  MAE={maem:.2f} "
          f"({time.time()-t1:.0f}s)")
    kf_results[cname] = dict(r2=round(r2m, 4), rmse=round(rmsem, 2),
                              mae=round(maem, 2))

# S3 KFold: baseline only (no temporal variation, so KFold = LOO on means)
kf_results["S3"] = dict(r2=round(stage1_loo_r2, 4),
                          rmse=round(stage1_loo_rmse, 2),
                          mae=round(stage1_loo_mae, 2))

df.drop(columns=RFSI_COLS, inplace=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOSO
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"LOSO CV ({n_stn} stations)")
print(f"{'='*80}")

# Pre-compute base feature arrays for Stage 2
all_base = sorted(set(f for cn in ["S1", "S2"]
                       for f in CONFIGS[cn] if f not in RFSI_COLS))
base_arr = df[all_base].values
base_col_map = {f: i for i, f in enumerate(all_base)}
rfsi_col_map = {f: i for i, f in enumerate(RFSI_COLS)}

stationId_vals = df["stationId"].values

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

    # ── Stage 1: predict held-out station's baseline ──
    # Train station means (excluding held-out)
    train_sids = [s for s in station_ids if s != held_sid]
    held_idx = list(stage1_sids).index(held_sid)

    s1_train_idx = [i for i in range(n_stn) if stage1_sids[i] != held_sid]
    s1_test_idx = [held_idx]

    scaler_s1 = StandardScaler()
    X_s1_tr = scaler_s1.fit_transform(X_stage1_all[s1_train_idx])
    X_s1_te = scaler_s1.transform(X_stage1_all[s1_test_idx])

    # Recompute training station means from training data only
    train_means = df.loc[mask_train].groupby("stationId")["PM2.5"].mean()
    y_s1_tr = np.array([train_means[s] for s in train_sids])

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_s1_tr, y_s1_tr)
    held_baseline = ridge.predict(X_s1_te)[0]

    # ── Training residuals: use each station's ACTUAL mean ──
    train_baselines = np.array([train_means[s] for s in stationId_vals[mask_train]])
    train_residuals = y_train - train_baselines

    # ── RFSI features (exclude held-out station) ──
    rfsi_fold = compute_rfsi(exclude_sid=held_sid, K=K_NN)
    rfsi_arr = np.column_stack([rfsi_fold[c] for c in RFSI_COLS])

    parts = [f"[{fold_i+1:2d}/{n_stn}] {nm:45s} |"
             f" baseline={held_baseline:5.1f} (actual={station_pm_means[held_sid]:5.1f},"
             f" err={held_baseline - station_pm_means[held_sid]:+5.1f}) |"]

    # ── S3: Ridge-only ──
    pm_pred_s3 = np.full(n_test, held_baseline)
    r2_s3 = r2_score(y_test, pm_pred_s3)
    rmse_s3 = np.sqrt(mean_squared_error(y_test, pm_pred_s3))
    mae_s3 = mean_absolute_error(y_test, pm_pred_s3)
    loso_results["S3"].append(dict(
        station_id=held_sid,
        station_name=sid_name.get(held_sid, held_sid),
        region=rg, n_rows=n_test,
        r2=round(r2_s3, 4), rmse=round(rmse_s3, 2), mae=round(mae_s3, 2),
        stage1_baseline=round(held_baseline, 2),
        stage1_error=round(held_baseline - station_pm_means[held_sid], 2)))
    parts.append(f"S3={r2_s3:+.3f}")

    # ── S1, S2: two-stage ──
    for cname in ["S1", "S2"]:
        feats = CONFIGS[cname]
        b_feats = [f for f in feats if f not in RFSI_COLS]
        r_feats = [f for f in feats if f in RFSI_COLS]

        b_idx = [base_col_map[f] for f in b_feats] if b_feats else []
        r_idx = [rfsi_col_map[f] for f in r_feats] if r_feats else []

        arrays = []
        if b_idx:
            arrays.append(base_arr[:, b_idx])
        if r_idx:
            arrays.append(rfsi_arr[:, r_idx])
        X_all = np.hstack(arrays)

        X_tr = X_all[mask_train]
        X_te = X_all[mask_test]

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
    print(f"  {' '.join(parts)}  [{rg}]  "
          f"({fold_time:.0f}s, ETA {remaining/60:.0f}m)")

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE (best config by LOSO R²)
# ═══════════════════════════════════════════════════════════════════════════════

best_cfg = max(["S1", "S2"],
               key=lambda c: np.mean([r["r2"] for r in loso_results[c]]))
best_loso = np.mean([r["r2"] for r in loso_results[best_cfg]])

print(f"\n{'='*80}")
print(f"FEATURE IMPORTANCE — Config {best_cfg} "
      f"(best LOSO R²={best_loso:.4f})")
print(f"{'='*80}")

for col in RFSI_COLS:
    df[col] = rfsi_global[col]

feats_best = CONFIGS[best_cfg]
X_full = df[feats_best]

# Compute residuals using actual station means
resid_full = y_all - df["stationId"].map(sid_mean_map).values

model_full = xgb.XGBRegressor(**XGB_PARAMS)
model_full.fit(X_full, resid_full)

importance = model_full.get_booster().get_score(importance_type="gain")
imp_df = pd.DataFrame(
    [{"feature": k, "gain": v} for k, v in importance.items()]
).sort_values("gain", ascending=False).reset_index(drop=True)
feat_map = {f"f{i}": name for i, name in enumerate(feats_best)}
imp_df["feature"] = imp_df["feature"].map(lambda x: feat_map.get(x, x))
imp_df["rank"] = range(1, len(imp_df) + 1)
imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance_exp08.csv"),
              index=False, encoding="utf-8-sig")

print("\nTop 20 features by gain:")
for _, r in imp_df.head(20).iterrows():
    tag = " *RFSI*" if r["feature"] in RFSI_COLS else ""
    print(f"  {r['rank']:2d}. {r['feature']:30s} gain={r['gain']:.0f}{tag}")

df.drop(columns=RFSI_COLS, inplace=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

all_loso_rows = []
for cfg in CONFIG_ORDER:
    for r in loso_results[cfg]:
        all_loso_rows.append({"config": cfg, **r})
pd.DataFrame(all_loso_rows).to_csv(
    os.path.join(OUT_DIR, "loso_per_station_exp08.csv"),
    index=False, encoding="utf-8-sig")

pd.DataFrame([{"config": cfg, **kf_results[cfg]} for cfg in CONFIG_ORDER
               if cfg in kf_results]).to_csv(
    os.path.join(OUT_DIR, "kfold_exp08.csv"),
    index=False, encoding="utf-8-sig")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD PREVIOUS RESULTS FOR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

prev_loso = {}
prev_kf = {}

for exp_csv, exp_kf in [
    ("loso_per_station_exp04_all.csv", "kfold_exp04.csv"),
    ("loso_per_station_exp07.csv", "kfold_exp07.csv"),
]:
    lp = os.path.join(OUT_DIR, exp_csv)
    kp = os.path.join(OUT_DIR, exp_kf)
    if os.path.exists(lp):
        tmp = pd.read_csv(lp, dtype={"station_id": str})
        for cfg in tmp["config"].unique():
            sub = tmp[tmp["config"] == cfg].drop(columns=["config"])
            prev_loso[cfg] = sub.to_dict("records")
    if os.path.exists(kp):
        tmp = pd.read_csv(kp)
        for _, row in tmp.iterrows():
            prev_kf[row["config"]] = dict(r2=row["r2"], rmse=row["rmse"],
                                           mae=row["mae"])

c_r2 = {}
loso_c_path = os.path.join(OUT_DIR, "loso_per_station_config_c.csv")
if os.path.exists(loso_c_path):
    loso_c = pd.read_csv(loso_c_path, dtype={"station_id": str})
    c_r2 = dict(zip(loso_c["station_id"], loso_c["r2"]))

k2_r2 = {}
if "K2" in prev_loso:
    k2_r2 = {r["station_id"]: r["r2"] for r in prev_loso["K2"]}

b1_r2 = {}
if "B1" in prev_loso:
    b1_r2 = {r["station_id"]: r["r2"] for r in prev_loso["B1"]}

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("GENERATING REPORT")
print(f"{'='*80}")


def loso_summary(results):
    rdf = pd.DataFrame(results)
    v = rdf.dropna(subset=["r2"])
    by_region = {}
    for rg in ["North", "Central", "South", "Unknown"]:
        sub = v[v["region"] == rg]
        if len(sub):
            by_region[rg] = dict(
                n=len(sub),
                mean_r2=round(sub["r2"].mean(), 4),
                median_r2=round(sub["r2"].median(), 4),
                mean_rmse=round(sub["rmse"].mean(), 1))
    return dict(
        mean_r2=round(v["r2"].mean(), 4),
        median_r2=round(v["r2"].median(), 4),
        wmean_r2=round((v["r2"]*v["n_rows"]).sum()/v["n_rows"].sum(), 4),
        mean_rmse=round(v["rmse"].mean(), 2),
        neg_count=int((v["r2"] < 0).sum()),
        by_region=by_region)


sums = {c: loso_summary(loso_results[c]) for c in CONFIG_ORDER}

prev_sums = {}
for c, recs in prev_loso.items():
    prev_sums[c] = loso_summary(recs)

rpt = []
rpt.append("# Experiment 08: Two-Stage Architecture\n")
rpt.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rpt.append(f"**Dataset:** {len(df):,} rows, {n_stn} stations")
rpt.append(f"**Stage 1:** Ridge regression on station means "
           f"({len(STAGE1_FEATS)} static features, α=1.0)")
rpt.append(f"**Stage 2:** XGBoost on residuals (met + AOD + RFSI, "
           f"no geographic features)")
rpt.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, "
           f"lr=0.05, device=cuda")
rpt.append(f"**RFSI:** K={K_NN} nearest neighbors\n")

# ── Stage 1 diagnostic ──
rpt.append("## Stage 1: Station-Mean Prediction (Ridge LOO)\n")
rpt.append(f"- **LOO R²:** {stage1_loo_r2:.4f}")
rpt.append(f"- **LOO RMSE:** {stage1_loo_rmse:.2f} µg/m³")
rpt.append(f"- **LOO MAE:** {stage1_loo_mae:.2f} µg/m³\n")

rpt.append("### Ridge Coefficients (standardized)\n")
rpt.append("| Feature | Coefficient |")
rpt.append("|---------|------------|")
coef_order = sorted(zip(STAGE1_FEATS, ridge_full.coef_),
                    key=lambda x: abs(x[1]), reverse=True)
for fname, coef in coef_order:
    rpt.append(f"| {fname} | {coef:+.3f} |")
rpt.append(f"| intercept | {ridge_full.intercept_:.3f} |")

rpt.append("\n### Station-Level Predictions (sorted by |error|)\n")
rpt.append("| Station | Region | Actual Mean | Predicted | Error |")
rpt.append("|---------|--------|------------|-----------|-------|")
for _, r in stage1_diag.iterrows():
    nm = str(r["station_name"])[:45]
    rpt.append(f"| {nm} | {r['region']} | {r['pm25_actual_mean']:.1f} | "
               f"{r['pm25_predicted_mean']:.1f} | {r['error']:+.1f} |")

# ── Comparison table ──
rpt.append("\n## Comparison Table\n")
rpt.append("| Config | Description | Features | KFold R² | LOSO R² (mean) | "
           "LOSO R² (median) | Neg Stations | Gap |")
rpt.append("|--------|-------------|----------|----------|----------------|"
           "------------------|--------------|-----|")

refs = [
    ("C (Exp01)", "Absolute baseline", 62, 0.7262, -0.4953, -0.0004, 20, 1.2215),
    ("E (Exp02)", "Oracle anomaly",    55, 0.6926,  0.2252,  0.2640,  7, 0.4674),
]
for cfg, desc, nf, kf_r2, lm, lmed, neg, gap in refs:
    rpt.append(f"| {cfg} | {desc} | {nf} | {kf_r2:.4f} | {lm:.4f} | "
               f"{lmed:.4f} | {neg} | {gap:.4f} |")

prev_descs = {
    "K2": ("K2 (Exp04)", "Full + RFSI"),
    "B1": ("B1 (Exp07)", "K2 + buildings"),
}
for cn, (label, desc) in prev_descs.items():
    if cn in prev_sums and cn in prev_kf:
        s = prev_sums[cn]
        kf = prev_kf[cn]
        gap = round(kf["r2"] - s["mean_r2"], 4)
        rpt.append(f"| {label} | {desc} | — | {kf['r2']:.4f} | "
                   f"{s['mean_r2']:.4f} | {s['median_r2']:.4f} | "
                   f"{s['neg_count']} | {gap:.4f} |")

descs = {
    "S1": "Two-stage: Ridge + XGB (met+AOD+RFSI)",
    "S2": "Two-stage: Ridge + XGB (met+RFSI)",
    "S3": "Ridge baseline only (no temporal)",
}
for cn in CONFIG_ORDER:
    nf = len(CONFIGS.get(cn, [])) if cn != "S3" else 8
    kf = kf_results.get(cn, {})
    s = sums[cn]
    kf_r2 = kf.get("r2", s["mean_r2"])
    gap = round(kf_r2 - s["mean_r2"], 4)
    bold = "**" if cn == "S1" else ""
    rpt.append(f"| {bold}{cn}{bold} | {bold}{descs[cn]}{bold} | {nf} | "
               f"{kf_r2:.4f} | {bold}{s['mean_r2']:.4f}{bold} | "
               f"{s['median_r2']:.4f} | {s['neg_count']} | {gap:.4f} |")

# ── Per-station LOSO (best config) ──
best_loso_df = pd.DataFrame(loso_results[best_cfg]).sort_values(
    "r2", ascending=True)

rpt.append(f"\n## Per-Station LOSO: {best_cfg} vs B1 vs K2 vs C\n")
rpt.append(f"| Station | Region | C R² | K2 R² | B1 R² | {best_cfg} R² | "
           f"Δ vs B1 | S1 baseline err | {best_cfg} RMSE |")
rpt.append("|---------|--------|------|-------|-------|-------|---------|"
           "----------------|--------|")

for _, r in best_loso_df.iterrows():
    nm = str(r["station_name"])[:45]
    cv = c_r2.get(r["station_id"], np.nan)
    k2v = k2_r2.get(r["station_id"], np.nan)
    b1v = b1_r2.get(r["station_id"], np.nan)
    bv = r["r2"]
    s1_err = r.get("stage1_error", np.nan)

    cv_s = f"{cv:.4f}" if pd.notna(cv) else "—"
    k2v_s = f"{k2v:.4f}" if pd.notna(k2v) else "—"
    b1v_s = f"{b1v:.4f}" if pd.notna(b1v) else "—"

    if pd.notna(b1v):
        delta = bv - b1v
        flag = " ✓" if delta > 0 else ""
        delta_s = f"{delta:+.4f}{flag}"
    else:
        delta_s = "—"

    s1_err_s = f"{s1_err:+.1f}" if pd.notna(s1_err) else "—"

    rpt.append(f"| {nm} | {r['region']} | {cv_s} | {k2v_s} | {b1v_s} | "
               f"{bv:.4f} | {delta_s} | {s1_err_s} | {r['rmse']:.1f} |")

# ── Regional breakdown ──
rpt.append("\n## Regional Breakdown\n")
all_cfgs = []
for cn in ["K2", "B1"]:
    if cn in prev_sums:
        all_cfgs.append(cn)
all_cfgs += CONFIG_ORDER
hdr = "| Region | C R² | " + " | ".join(f"{c} R²" for c in all_cfgs) + " |"
sep = "|--------|------|" + "|".join(["------"] * len(all_cfgs)) + "|"
rpt.append(hdr)
rpt.append(sep)

c_reg = {"North": 0.0458, "Central": -0.0211, "South": -2.1908}
for rg in ["North", "Central", "South"]:
    cv = c_reg.get(rg, 0)
    vals = []
    for cn in all_cfgs:
        src = prev_sums if cn in prev_sums else sums
        if cn in src:
            v = src[cn]["by_region"].get(rg, {}).get("mean_r2")
            vals.append(f"{v:.4f}" if v is not None else "—")
        else:
            vals.append("—")
    rpt.append(f"| {rg} | {cv:.4f} | " + " | ".join(vals) + " |")

# ── Feature importance ──
rpt.append(f"\n## Feature Importance (Config {best_cfg} Stage 2, top 20)\n")
rpt.append("| Rank | Feature | Gain | Type |")
rpt.append("|------|---------|------|------|")
for _, r in imp_df.head(20).iterrows():
    tag = "RFSI" if r["feature"] in RFSI_COLS else ""
    rpt.append(f"| {r['rank']} | {r['feature']} | {r['gain']:.0f} | {tag} |")

# ── Analysis ──
rpt.append("\n## Analysis\n")

rpt.append("### 1. Stage 1 quality — can Ridge predict station means?\n")
rpt.append(f"- Stage 1 LOO R² = {stage1_loo_r2:.4f}")
rpt.append(f"- Stage 1 LOO RMSE = {stage1_loo_rmse:.2f} µg/m³")
if stage1_loo_r2 > 0.5:
    rpt.append("- Ridge captures >50% of inter-station variance — strong baseline")
elif stage1_loo_r2 > 0.3:
    rpt.append("- Ridge captures 30-50% — moderate baseline, room for improvement")
else:
    rpt.append("- Ridge captures <30% — weak baseline, Stage 1 is the bottleneck")

rpt.append(f"\n### 2. Two-stage vs single-stage (S1 vs B1 vs K2)\n")
s1_loso = sums["S1"]["mean_r2"]
b1_loso = prev_sums.get("B1", {}).get("mean_r2")
k2_loso = prev_sums.get("K2", {}).get("mean_r2")
rpt.append(f"- K2 (single-stage, full+RFSI): LOSO R² = "
           f"{k2_loso:.4f}" if k2_loso is not None else "- K2: N/A")
rpt.append(f"- B1 (single-stage, +buildings): LOSO R² = "
           f"{b1_loso:.4f}" if b1_loso is not None else "- B1: N/A")
rpt.append(f"- S1 (two-stage): LOSO R² = {s1_loso:.4f}")
if b1_loso is not None:
    rpt.append(f"- Delta S1 vs B1: {s1_loso - b1_loso:+.4f}")

rpt.append(f"\n### 3. Does AOD help in Stage 2? (S1 vs S2)\n")
s2_loso = sums["S2"]["mean_r2"]
rpt.append(f"- S1 (met+AOD+RFSI): LOSO R² = {s1_loso:.4f}")
rpt.append(f"- S2 (met+RFSI):     LOSO R² = {s2_loso:.4f}")
rpt.append(f"- AOD effect: {s1_loso - s2_loso:+.4f}")

rpt.append(f"\n### 4. S3 Ridge-only — how much does the baseline alone explain?\n")
s3_loso = sums["S3"]["mean_r2"]
rpt.append(f"- S3 LOSO R² = {s3_loso:.4f} (predicting station mean for every hour)")
rpt.append(f"- S1 LOSO R² = {s1_loso:.4f}")
rpt.append(f"- Stage 2 adds {s1_loso - s3_loso:+.4f} R² on top of the baseline")
if s3_loso > 0:
    rpt.append(f"- Even the baseline alone achieves positive LOSO R²!")

rpt.append(f"\n### 5. KFold-LOSO gap\n")
for cn in CONFIG_ORDER:
    kfr = kf_results.get(cn, {}).get("r2", sums[cn]["mean_r2"])
    lr = sums[cn]["mean_r2"]
    rpt.append(f"- {cn}: KFold={kfr:.4f}, LOSO={lr:.4f}, gap={kfr - lr:.4f}")
rpt.append(f"- Baseline (C) gap: 1.2215")
rpt.append(f"- Oracle (E) gap: 0.4674")

rpt.append(f"\n### 6. Stations where Stage 1 error determines outcome\n")
s1_recs = {r["station_id"]: r for r in loso_results[best_cfg]}
rpt.append("Stations with |Stage 1 error| > 10 µg/m³:\n")
for _, r in stage1_diag.iterrows():
    if abs(r["error"]) > 10:
        sid = r["station_id"]
        loso_r2 = s1_recs.get(sid, {}).get("r2", np.nan)
        rpt.append(f"- {str(r['station_name'])[:45]}: "
                   f"baseline err={r['error']:+.1f}, "
                   f"LOSO R²={loso_r2:.4f}" if pd.notna(loso_r2) else "")

report_path = os.path.join(OUT_DIR, "experiment_08_twostage.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))

print(f"\nReport: {report_path}")
print(f"LOSO: {os.path.join(OUT_DIR, 'loso_per_station_exp08.csv')}")
print(f"KFold: {os.path.join(OUT_DIR, 'kfold_exp08.csv')}")
print(f"Stage 1 diagnostics: {os.path.join(OUT_DIR, 'stage1_diagnostics_exp08.csv')}")
print(f"Feature importance: {os.path.join(OUT_DIR, 'feature_importance_exp08.csv')}")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
