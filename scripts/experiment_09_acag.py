"""
Experiment 09: ACAG Satellite Climatology as Spatial Baseline

ACAG V6.GL.02.04 provides satellite-derived PM2.5 at 0.1° monthly resolution
(2020-2023 mean).  It is NOT from ground stations → no leakage in LOSO.

Configs:
  A1 — B1 (full+RFSI+buildings, best at -0.020) + ACAG_annual_mean + ACAG_monthly
  A2 — Two-stage: ACAG_monthly as baseline (no Ridge).
        Target = PM2.5 - ACAG_monthly.  Stage 2 = XGBoost on met+AOD+RFSI.
  A3 — Two-stage: same as A2 + building density in Stage 2.

Output:
  analysis/thesis_experiments/experiment_09_acag.md
  analysis/thesis_experiments/loso_per_station_exp09.csv
  analysis/thesis_experiments/kfold_exp09.csv
  analysis/thesis_experiments/feature_importance_exp09.csv
"""

import argparse, io, sys, os, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
print("EXPERIMENT 09: ACAG SATELLITE CLIMATOLOGY")
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
station_pm_means = df.groupby("stationId")["PM2.5"].mean()

# ═══════════════════════════════════════════════════════════════════════════════
#  JOIN ACAG CLIMATOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- ACAG satellite climatology ---")

acag_path = os.path.join(DATA_DIR, "data/acag/acag_station_climatology.csv")
if not os.path.exists(acag_path):
    acag_path = os.path.join(REPO_DIR, "data/acag/acag_station_climatology.csv")
acag = pd.read_csv(acag_path, dtype={"stationId": str})
print(f"ACAG CSV: {len(acag)} stations")

acag_annual = acag.set_index("stationId")["ACAG_annual_mean"]
acag_monthly_cols = {m: f"ACAG_monthly_clim_{m:02d}" for m in range(1, 13)}
acag_monthly_map = {}
for m in range(1, 13):
    col = acag_monthly_cols[m]
    acag_monthly_map[m] = acag.set_index("stationId")[col]

df["ACAG_annual_mean"] = df["stationId"].map(acag_annual)

acag_long = acag.melt(
    id_vars=["stationId"], value_vars=list(acag_monthly_cols.values()),
    var_name="month_col", value_name="ACAG_monthly")
acag_long["month"] = acag_long["month_col"].str.extract(r"(\d{2})$").astype(int)
acag_long = acag_long[["stationId", "month", "ACAG_monthly"]]
df = df.merge(acag_long, on=["stationId", "month"], how="left")

n_acag = df["ACAG_monthly"].notna().sum()
print(f"ACAG joined: {n_acag:,}/{len(df):,} rows "
      f"({n_acag/len(df)*100:.1f}%) have ACAG values")

acag_corr = df[["PM2.5", "ACAG_annual_mean", "ACAG_monthly"]].corr()
print(f"Correlation with PM2.5:")
print(f"  ACAG_annual_mean: {acag_corr.loc['PM2.5','ACAG_annual_mean']:.4f}")
print(f"  ACAG_monthly:     {acag_corr.loc['PM2.5','ACAG_monthly']:.4f}")

# Per-station comparison: ACAG annual vs actual station mean
print("\nACAG annual vs station mean PM2.5:")
for sid in station_ids[:5]:
    actual = station_pm_means[sid]
    acag_val = acag_annual.get(sid, np.nan)
    nm = sid_name.get(sid, sid)[:40]
    print(f"  {nm:40s} | actual={actual:5.1f} | ACAG={acag_val:5.1f} | "
          f"diff={acag_val-actual:+5.1f}")
print("  ...")

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
for col in BUILDING_COLS:
    df[col] = df["stationId"].map(bld_map[col]).fillna(0)
print(f"Building density joined: {len(bld)} stations")

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

ACAG_FEATS = ["ACAG_annual_mean", "ACAG_monthly"]

# A1: B1 + ACAG (single-stage XGBoost, target = PM2.5)
FEATURES_A1 = MET + AOD + GEO + TEMPORAL + RFSI_COLS + BUILDING_COLS + ACAG_FEATS

# A2 Stage 2: met + AOD + RFSI + temporal (no geography, no buildings)
FEATURES_A2_S2 = MET + AOD + TEMPORAL + RFSI_COLS

# A3 Stage 2: met + AOD + RFSI + temporal + buildings (no geography)
FEATURES_A3_S2 = MET + AOD + TEMPORAL + RFSI_COLS + BUILDING_COLS

CONFIGS = {"A1": FEATURES_A1, "A2": FEATURES_A2_S2, "A3": FEATURES_A3_S2}
CONFIG_ORDER = ["A1", "A2", "A3"]

for cname, feats in list(CONFIGS.items()):
    missing = [f for f in feats if f not in df.columns and f not in RFSI_COLS]
    if missing:
        print(f"WARNING: {cname} missing columns: {missing}")
        CONFIGS[cname] = [f for f in feats if f in df.columns or f in RFSI_COLS]

for cn in CONFIG_ORDER:
    nb = len([f for f in CONFIGS[cn] if f not in RFSI_COLS])
    nr = len([f for f in CONFIGS[cn] if f in RFSI_COLS])
    desc = "single-stage" if cn == "A1" else "two-stage Stage 2"
    print(f"  {cn} ({desc}): {len(CONFIGS[cn])} features ({nb} base + {nr} RFSI)")

# ═══════════════════════════════════════════════════════════════════════════════
#  KFOLD
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("KFOLD 5-FOLD CV")
print(f"{'='*80}")

print("\nComputing global RFSI features ...")
t1 = time.time()
rfsi_global = compute_rfsi(exclude_sid=None, K=K_NN)
print(f"Done ({time.time()-t1:.1f}s)")

for col in RFSI_COLS:
    df[col] = rfsi_global[col]

kf_results = {}

# A1: single-stage, target = PM2.5
feats_a1 = CONFIGS["A1"]
X_a1 = df[feats_a1]
print(f"\n--- A1 ({len(feats_a1)} features, target=PM2.5) ---")
t1 = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = []
for _, (tr, va) in enumerate(kf.split(X_a1)):
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_a1.iloc[tr], y_all[tr])
    p = m.predict(X_a1.iloc[va])
    folds.append(dict(r2=r2_score(y_all[va], p),
                      rmse=np.sqrt(mean_squared_error(y_all[va], p)),
                      mae=mean_absolute_error(y_all[va], p)))
r2m = np.mean([f["r2"] for f in folds])
rmsem = np.mean([f["rmse"] for f in folds])
maem = np.mean([f["mae"] for f in folds])
print(f"  R²={r2m:.4f}  RMSE={rmsem:.2f}  MAE={maem:.2f} ({time.time()-t1:.0f}s)")
kf_results["A1"] = dict(r2=round(r2m, 4), rmse=round(rmsem, 2), mae=round(maem, 2))

# A2, A3: two-stage KFold — use actual ACAG_monthly as baseline (same for all folds)
acag_baselines_kf = df["ACAG_monthly"].values
residuals_kf = y_all - acag_baselines_kf

for cname in ["A2", "A3"]:
    feats = CONFIGS[cname]
    X = df[feats]
    print(f"\n--- {cname} ({len(feats)} features, target=residual vs ACAG_monthly) ---")
    t1 = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for _, (tr, va) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[tr], residuals_kf[tr])
        resid_pred = m.predict(X.iloc[va])
        pm_pred = resid_pred + acag_baselines_kf[va]
        folds.append(dict(r2=r2_score(y_all[va], pm_pred),
                          rmse=np.sqrt(mean_squared_error(y_all[va], pm_pred)),
                          mae=mean_absolute_error(y_all[va], pm_pred)))
    r2m = np.mean([f["r2"] for f in folds])
    rmsem = np.mean([f["rmse"] for f in folds])
    maem = np.mean([f["mae"] for f in folds])
    print(f"  R²={r2m:.4f}  RMSE={rmsem:.2f}  MAE={maem:.2f} ({time.time()-t1:.0f}s)")
    kf_results[cname] = dict(r2=round(r2m, 4), rmse=round(rmsem, 2),
                              mae=round(maem, 2))

df.drop(columns=RFSI_COLS, inplace=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOSO
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"LOSO CV ({n_stn} stations)")
print(f"{'='*80}")

# Pre-compute base feature arrays
all_base_a1 = sorted(set(f for f in CONFIGS["A1"] if f not in RFSI_COLS))
all_base_s2 = sorted(set(f for cn in ["A2", "A3"]
                          for f in CONFIGS[cn] if f not in RFSI_COLS))
all_base = sorted(set(all_base_a1 + all_base_s2))
base_arr = df[all_base].values
base_col_map = {f: i for i, f in enumerate(all_base)}
rfsi_col_map = {f: i for i, f in enumerate(RFSI_COLS)}

stationId_vals = df["stationId"].values
acag_monthly_vals = df["ACAG_monthly"].values

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

    # ACAG baseline for held-out station (same for A2, A3)
    acag_baseline_test = acag_monthly_vals[mask_test]
    acag_baseline_train = acag_monthly_vals[mask_train]
    held_acag_annual = float(df.loc[mask_test, "ACAG_annual_mean"].iloc[0])

    # Training residuals for two-stage configs
    train_residuals = y_train - acag_baseline_train

    # RFSI (exclude held-out)
    rfsi_fold = compute_rfsi(exclude_sid=held_sid, K=K_NN)
    rfsi_arr = np.column_stack([rfsi_fold[c] for c in RFSI_COLS])

    parts = [f"[{fold_i+1:2d}/{n_stn}] {nm:45s} |"
             f" ACAG_ann={held_acag_annual:5.1f}"
             f" (actual={station_pm_means[held_sid]:5.1f},"
             f" diff={held_acag_annual - station_pm_means[held_sid]:+5.1f}) |"]

    # ── A1: single-stage ──
    feats = CONFIGS["A1"]
    b_feats = [f for f in feats if f not in RFSI_COLS]
    r_feats = [f for f in feats if f in RFSI_COLS]
    b_idx = [base_col_map[f] for f in b_feats]
    r_idx = [rfsi_col_map[f] for f in r_feats]

    arrays = [base_arr[:, b_idx], rfsi_arr[:, r_idx]]
    X_all_a1 = np.hstack(arrays)

    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_all_a1[mask_train], y_train)
    pm_pred = m.predict(X_all_a1[mask_test])

    r2 = r2_score(y_test, pm_pred)
    rmse = np.sqrt(mean_squared_error(y_test, pm_pred))
    mae = mean_absolute_error(y_test, pm_pred)
    loso_results["A1"].append(dict(
        station_id=held_sid, station_name=sid_name.get(held_sid, held_sid),
        region=rg, n_rows=n_test,
        r2=round(r2, 4), rmse=round(rmse, 2), mae=round(mae, 2),
        acag_annual=round(held_acag_annual, 2),
        acag_vs_actual=round(held_acag_annual - station_pm_means[held_sid], 2)))
    parts.append(f"A1={r2:+.3f}")

    # ── A2, A3: two-stage ──
    for cname in ["A2", "A3"]:
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
        X_all_s2 = np.hstack(arrays)

        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_all_s2[mask_train], train_residuals)
        resid_pred = m.predict(X_all_s2[mask_test])
        pm_pred = resid_pred + acag_baseline_test

        r2 = r2_score(y_test, pm_pred)
        rmse = np.sqrt(mean_squared_error(y_test, pm_pred))
        mae = mean_absolute_error(y_test, pm_pred)
        loso_results[cname].append(dict(
            station_id=held_sid, station_name=sid_name.get(held_sid, held_sid),
            region=rg, n_rows=n_test,
            r2=round(r2, 4), rmse=round(rmse, 2), mae=round(mae, 2),
            acag_annual=round(held_acag_annual, 2),
            acag_vs_actual=round(held_acag_annual - station_pm_means[held_sid], 2)))
        parts.append(f"{cname}={r2:+.3f}")

    fold_time = time.time() - t_fold
    remaining = fold_time * (n_stn - fold_i - 1)
    print(f"  {' '.join(parts)}  [{rg}]  "
          f"({fold_time:.0f}s, ETA {remaining/60:.0f}m)")

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE (best config)
# ═══════════════════════════════════════════════════════════════════════════════

best_cfg = max(CONFIG_ORDER,
               key=lambda c: np.mean([r["r2"] for r in loso_results[c]]))
best_loso = np.mean([r["r2"] for r in loso_results[best_cfg]])

print(f"\n{'='*80}")
print(f"FEATURE IMPORTANCE — Config {best_cfg} (LOSO R²={best_loso:.4f})")
print(f"{'='*80}")

for col in RFSI_COLS:
    df[col] = rfsi_global[col]

feats_best = CONFIGS[best_cfg]
X_full = df[feats_best]

if best_cfg == "A1":
    target_fi = y_all
else:
    target_fi = y_all - acag_monthly_vals

model_full = xgb.XGBRegressor(**XGB_PARAMS)
model_full.fit(X_full, target_fi)

importance = model_full.get_booster().get_score(importance_type="gain")
imp_df = pd.DataFrame(
    [{"feature": k, "gain": v} for k, v in importance.items()]
).sort_values("gain", ascending=False).reset_index(drop=True)
feat_map = {f"f{i}": name for i, name in enumerate(feats_best)}
imp_df["feature"] = imp_df["feature"].map(lambda x: feat_map.get(x, x))
imp_df["rank"] = range(1, len(imp_df) + 1)
imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance_exp09.csv"),
              index=False, encoding="utf-8-sig")

print("\nTop 20 features by gain:")
for _, r in imp_df.head(20).iterrows():
    tag = " *RFSI*" if r["feature"] in RFSI_COLS else ""
    tag += " *ACAG*" if "ACAG" in r["feature"] else ""
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
    os.path.join(OUT_DIR, "loso_per_station_exp09.csv"),
    index=False, encoding="utf-8-sig")

pd.DataFrame([{"config": cfg, **kf_results[cfg]} for cfg in CONFIG_ORDER]).to_csv(
    os.path.join(OUT_DIR, "kfold_exp09.csv"),
    index=False, encoding="utf-8-sig")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD PREVIOUS RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

prev_loso = {}
prev_kf = {}

for exp_csv, exp_kf in [
    ("loso_per_station_exp04_all.csv", "kfold_exp04.csv"),
    ("loso_per_station_exp07.csv", "kfold_exp07.csv"),
    ("loso_per_station_exp08.csv", "kfold_exp08.csv"),
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

b1_r2 = {}
if "B1" in prev_loso:
    b1_r2 = {r["station_id"]: r["r2"] for r in prev_loso["B1"]}

k2_r2 = {}
if "K2" in prev_loso:
    k2_r2 = {r["station_id"]: r["r2"] for r in prev_loso["K2"]}

s1_r2 = {}
if "S1" in prev_loso:
    s1_r2 = {r["station_id"]: r["r2"] for r in prev_loso["S1"]}

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
rpt.append("# Experiment 09: ACAG Satellite Climatology\n")
rpt.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rpt.append(f"**Dataset:** {len(df):,} rows, {n_stn} stations")
rpt.append(f"**ACAG:** V6.GL.02.04, 0.1° monthly AS, 2020-2023 climatology")
rpt.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, "
           f"lr=0.05, device=cuda")
rpt.append(f"**RFSI:** K={K_NN} nearest neighbors\n")

rpt.append("## ACAG Diagnostics\n")
rpt.append(f"- Correlation ACAG_annual vs PM2.5: "
           f"{acag_corr.loc['PM2.5','ACAG_annual_mean']:.4f}")
rpt.append(f"- Correlation ACAG_monthly vs PM2.5: "
           f"{acag_corr.loc['PM2.5','ACAG_monthly']:.4f}")

acag_r2_annual = r2_score(
    [station_pm_means[s] for s in station_ids],
    [acag_annual.get(s, np.nan) for s in station_ids])
rpt.append(f"- ACAG annual mean vs station mean R² (across {n_stn} stations): "
           f"{acag_r2_annual:.4f}")
rpt.append(f"  (Compare to Ridge LOO R² from Exp08 which was ~0.3-0.5)")

acag_abs_err = [abs(acag_annual.get(s, np.nan) - station_pm_means[s])
                for s in station_ids]
rpt.append(f"- ACAG vs station mean |error|: median={np.nanmedian(acag_abs_err):.1f}, "
           f"max={np.nanmax(acag_abs_err):.1f} µg/m³\n")

# ── Comparison table ──
rpt.append("## Comparison Table\n")
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
    "S1": ("S1 (Exp08)", "Two-stage Ridge+XGB"),
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
    "A1": "B1 + ACAG features (single-stage)",
    "A2": "ACAG baseline + XGB residual (met+AOD+RFSI)",
    "A3": "ACAG baseline + XGB residual (+buildings)",
}
for cn in CONFIG_ORDER:
    nf = len(CONFIGS[cn])
    kf = kf_results[cn]
    s = sums[cn]
    gap = round(kf["r2"] - s["mean_r2"], 4)
    bold = "**" if cn == best_cfg else ""
    rpt.append(f"| {bold}{cn}{bold} | {bold}{descs[cn]}{bold} | {nf} | "
               f"{kf['r2']:.4f} | {bold}{s['mean_r2']:.4f}{bold} | "
               f"{s['median_r2']:.4f} | {s['neg_count']} | {gap:.4f} |")

# ── Per-station LOSO ──
best_loso_df = pd.DataFrame(loso_results[best_cfg]).sort_values(
    "r2", ascending=True)

rpt.append(f"\n## Per-Station LOSO: {best_cfg} vs B1 vs S1\n")
rpt.append(f"| Station | Region | B1 R² | S1 R² | {best_cfg} R² | "
           f"Δ vs B1 | ACAG diff | {best_cfg} RMSE |")
rpt.append("|---------|--------|-------|-------|-------|---------|"
           "-----------|--------|")

for _, r in best_loso_df.iterrows():
    nm = str(r["station_name"])[:45]
    b1v = b1_r2.get(r["station_id"], np.nan)
    s1v = s1_r2.get(r["station_id"], np.nan)
    bv = r["r2"]
    acag_diff = r.get("acag_vs_actual", np.nan)

    b1v_s = f"{b1v:.4f}" if pd.notna(b1v) else "—"
    s1v_s = f"{s1v:.4f}" if pd.notna(s1v) else "—"

    if pd.notna(b1v):
        delta = bv - b1v
        flag = " ✓" if delta > 0 else ""
        delta_s = f"{delta:+.4f}{flag}"
    else:
        delta_s = "—"

    acag_s = f"{acag_diff:+.1f}" if pd.notna(acag_diff) else "—"

    rpt.append(f"| {nm} | {r['region']} | {b1v_s} | {s1v_s} | "
               f"{bv:.4f} | {delta_s} | {acag_s} | {r['rmse']:.1f} |")

# ── Regional breakdown ──
rpt.append("\n## Regional Breakdown\n")
all_cfgs = []
for cn in ["K2", "B1", "S1"]:
    if cn in prev_sums:
        all_cfgs.append(cn)
all_cfgs += CONFIG_ORDER
hdr = "| Region | " + " | ".join(f"{c} R²" for c in all_cfgs) + " |"
sep = "|--------|" + "|".join(["------"] * len(all_cfgs)) + "|"
rpt.append(hdr)
rpt.append(sep)

for rg in ["North", "Central", "South"]:
    vals = []
    for cn in all_cfgs:
        src = prev_sums if cn in prev_sums else sums
        if cn in src:
            v = src[cn]["by_region"].get(rg, {}).get("mean_r2")
            vals.append(f"{v:.4f}" if v is not None else "—")
        else:
            vals.append("—")
    rpt.append(f"| {rg} | " + " | ".join(vals) + " |")

# ── Feature importance ──
rpt.append(f"\n## Feature Importance (Config {best_cfg}, top 20)\n")
rpt.append("| Rank | Feature | Gain | Type |")
rpt.append("|------|---------|------|------|")
for _, r in imp_df.head(20).iterrows():
    tag = "RFSI" if r["feature"] in RFSI_COLS else ""
    if "ACAG" in str(r["feature"]):
        tag = "ACAG"
    if r["feature"] in BUILDING_COLS:
        tag = "building"
    rpt.append(f"| {r['rank']} | {r['feature']} | {r['gain']:.0f} | {tag} |")

# ── Analysis ──
rpt.append("\n## Analysis\n")

a1_loso = sums["A1"]["mean_r2"]
a2_loso = sums["A2"]["mean_r2"]
a3_loso = sums["A3"]["mean_r2"]
b1_loso_mean = prev_sums.get("B1", {}).get("mean_r2")
s1_loso_mean = prev_sums.get("S1", {}).get("mean_r2")

rpt.append("### 1. ACAG quality as spatial baseline\n")
rpt.append(f"- ACAG annual vs station mean R² = {acag_r2_annual:.4f}")
rpt.append(f"- Median |error| = {np.nanmedian(acag_abs_err):.1f} µg/m³")
rpt.append("- ACAG is satellite-derived (van Donkelaar et al.) — no ground station "
           "leakage in LOSO")
rpt.append("- This is a key advantage over Ridge baseline which uses station-derived "
           "building density\n")

rpt.append("### 2. A1: Does adding ACAG to B1 help?\n")
rpt.append(f"- B1 LOSO R² = {b1_loso_mean:.4f}" if b1_loso_mean else "- B1: N/A")
rpt.append(f"- A1 LOSO R² = {a1_loso:.4f}")
if b1_loso_mean:
    rpt.append(f"- Delta: {a1_loso - b1_loso_mean:+.4f}")

rpt.append(f"\n### 3. A2 vs A3: Do buildings help Stage 2?\n")
rpt.append(f"- A2 (met+AOD+RFSI): LOSO R² = {a2_loso:.4f}")
rpt.append(f"- A3 (+buildings):    LOSO R² = {a3_loso:.4f}")
rpt.append(f"- Building effect: {a3_loso - a2_loso:+.4f}")

rpt.append(f"\n### 4. ACAG two-stage vs Ridge two-stage (A2 vs S1)\n")
rpt.append(f"- S1 (Ridge baseline): LOSO R² = "
           f"{s1_loso_mean:.4f}" if s1_loso_mean else "- S1: N/A")
rpt.append(f"- A2 (ACAG baseline):  LOSO R² = {a2_loso:.4f}")
if s1_loso_mean:
    rpt.append(f"- Delta: {a2_loso - s1_loso_mean:+.4f}")
    rpt.append("- ACAG uses external satellite data, Ridge uses station-derived "
               "static features")

rpt.append(f"\n### 5. Progression summary\n")
rpt.append("| Experiment | Config | LOSO R² | Key addition |")
rpt.append("|------------|--------|---------|--------------|")
rpt.append("| Exp01 | C | -0.4953 | Absolute baseline |")
rpt.append("| Exp02 | E | +0.2252 | Oracle anomaly (ceiling) |")
if "K2" in prev_sums:
    rpt.append(f"| Exp04 | K2 | {prev_sums['K2']['mean_r2']:.4f} | RFSI neighbors |")
if b1_loso_mean:
    rpt.append(f"| Exp07 | B1 | {b1_loso_mean:.4f} | Building density |")
if s1_loso_mean:
    rpt.append(f"| Exp08 | S1 | {s1_loso_mean:.4f} | Two-stage Ridge |")
rpt.append(f"| Exp09 | {best_cfg} | {sums[best_cfg]['mean_r2']:.4f} | "
           f"ACAG satellite climatology |")
rpt.append(f"| — | Oracle | +0.2252 | Perfect per-station mean |")

rpt.append(f"\n### 6. KFold-LOSO gap\n")
for cn in CONFIG_ORDER:
    kfr = kf_results[cn]["r2"]
    lr = sums[cn]["mean_r2"]
    rpt.append(f"- {cn}: KFold={kfr:.4f}, LOSO={lr:.4f}, gap={kfr - lr:.4f}")

report_path = os.path.join(OUT_DIR, "experiment_09_acag.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))

print(f"\nReport: {report_path}")
print(f"LOSO: {os.path.join(OUT_DIR, 'loso_per_station_exp09.csv')}")
print(f"KFold: {os.path.join(OUT_DIR, 'kfold_exp09.csv')}")
print(f"Feature importance: {os.path.join(OUT_DIR, 'feature_importance_exp09.csv')}")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
