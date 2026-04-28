"""
Experiment 04: RFSI Nearest-Station Features

Configs:
  K1 — RFSI + temporal (spatial interpolation baseline)
  K2 — Full Config C + RFSI (kitchen-sink)
  K3 — Met + AOD + RFSI, no geography
  K4 — Met + RFSI, no AOD, no geography
  K5 — Minimal physics (16 research-backed features)
  K6 — K5 + constrained XGBoost (monotonic + shallow)

Supports incremental runs: already-computed configs are loaded from CSV,
only new configs are executed.  Results accumulate across runs.

Output:
  analysis/thesis_experiments/experiment_04_rfsi.md
  analysis/thesis_experiments/loso_per_station_exp04_all.csv
  analysis/thesis_experiments/kfold_exp04.csv
  analysis/thesis_experiments/feature_importance_exp04.csv
  analysis/thesis_experiments/station_distances.csv
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
print("EXPERIMENT 04: RFSI NEAREST-STATION FEATURES")
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
#  STATION DISTANCES
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

dist_rows = []
for i in range(n_stn):
    for rank, j in enumerate(
            sorted(range(n_stn), key=lambda x: dist_full[i, x] if x != i else 1e9), 1):
        if j == i:
            continue
        dist_rows.append(dict(
            station_id=station_ids[i],
            station_name=sid_name[station_ids[i]],
            neighbor_rank=rank,
            neighbor_id=station_ids[j],
            neighbor_name=sid_name[station_ids[j]],
            distance_km=round(dist_full[i, j], 2)))
pd.DataFrame(dist_rows).to_csv(
    os.path.join(OUT_DIR, "station_distances.csv"),
    index=False, encoding="utf-8-sig")

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

# K5/K6: minimal physics feature set
FEATURES_K5 = [
    "PM25_nn_idw", "PM25_nn_mean", "dist_nn1", "dist_nn2", "dist_nn3",
    "PBLH", "WS_om", "Temperature_final", "Humidity_final", "Pressure_final",
    "VC", "AOD_physics", "AOT", "RF", "precip_mm",
    "month_sin", "month_cos",
]

# Monotonic constraints for K6 (same feature order as FEATURES_K5):
#   RFSI (5): unconstrained
#   PBLH↓  WS_om↓  Temp 0  Hum 0  Pres 0
#   VC↓  AOD_phys 0  AOT↑  RF↑  precip↓
#   month_sin 0  month_cos 0
K6_CONSTRAINTS = (0, 0, 0, 0, 0,
                  -1, -1, 0, 0, 0,
                  -1, 0, 1, 1, -1,
                  0, 0)

K6_XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=20,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    device="cuda", random_state=42, n_jobs=-1,
    monotone_constraints=K6_CONSTRAINTS,
)

CONFIGS = {
    "K1": TEMPORAL + RFSI_COLS,
    "K2": MET + AOD + GEO + TEMPORAL + RFSI_COLS,
    "K3": MET + AOD + TEMPORAL + RFSI_COLS,
    "K4": MET + TEMPORAL + RFSI_COLS,
    "K5": FEATURES_K5,
    "K6": FEATURES_K5,
}

CONFIG_ORDER = ["K1", "K2", "K3", "K4", "K5", "K6"]

CONFIG_PARAMS = {c: XGB_PARAMS for c in CONFIG_ORDER}
CONFIG_PARAMS["K6"] = K6_XGB_PARAMS

for cname, feats in list(CONFIGS.items()):
    missing = [f for f in feats if f not in df.columns and f not in RFSI_COLS]
    if missing:
        print(f"WARNING: {cname} missing columns: {missing}")
        CONFIGS[cname] = [f for f in feats if f in df.columns or f in RFSI_COLS]

for cn in CONFIG_ORDER:
    nb = len([f for f in CONFIGS[cn] if f not in RFSI_COLS])
    nr = len([f for f in CONFIGS[cn] if f in RFSI_COLS])
    print(f"  {cn}: {len(CONFIGS[cn])} features ({nb} base + {nr} RFSI)")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD SAVED RESULTS (incremental run support)
# ═══════════════════════════════════════════════════════════════════════════════

LOSO_ALL_CSV = os.path.join(OUT_DIR, "loso_per_station_exp04_all.csv")
KFOLD_CSV = os.path.join(OUT_DIR, "kfold_exp04.csv")

saved_loso = {}
saved_kf = {}

if os.path.exists(LOSO_ALL_CSV):
    tmp = pd.read_csv(LOSO_ALL_CSV, dtype={"station_id": str})
    for cfg in tmp["config"].unique():
        sub = tmp[tmp["config"] == cfg].drop(columns=["config"])
        saved_loso[cfg] = sub.to_dict("records")

if os.path.exists(KFOLD_CSV):
    tmp = pd.read_csv(KFOLD_CSV)
    for _, row in tmp.iterrows():
        saved_kf[row["config"]] = dict(r2=row["r2"], rmse=row["rmse"],
                                        mae=row["mae"])

configs_to_run = [c for c in CONFIG_ORDER if c not in saved_loso]
configs_loaded = [c for c in CONFIG_ORDER if c in saved_loso]

print(f"\nLoaded from CSV: {configs_loaded or '(none)'}")
print(f"Will run: {configs_to_run or '(none — all cached)'}")

if not configs_to_run:
    print("All configs already computed. Skipping to report.")

# ═══════════════════════════════════════════════════════════════════════════════
#  KFOLD (global RFSI, new configs only)
# ═══════════════════════════════════════════════════════════════════════════════

kf_results = dict(saved_kf)
loso_results = dict(saved_loso)
rfsi_global = None

if configs_to_run:
    print(f"\n{'='*80}")
    print("KFOLD 5-FOLD CV")
    print(f"{'='*80}")

    print("\nComputing global RFSI features ...")
    t1 = time.time()
    rfsi_global = compute_rfsi(exclude_sid=None, K=K_NN)
    print(f"Done ({time.time()-t1:.1f}s)")

    for col in RFSI_COLS:
        df[col] = rfsi_global[col]

    nn1_nan = np.isnan(rfsi_global["PM25_nn1"]).sum()
    nn5_nan = np.isnan(rfsi_global["PM25_nn5"]).sum()
    print(f"PM25_nn1 NaN: {nn1_nan:,} ({nn1_nan/len(df)*100:.1f}%)")
    print(f"PM25_nn5 NaN: {nn5_nan:,} ({nn5_nan/len(df)*100:.1f}%)")

    for cname in configs_to_run:
        feats = CONFIGS[cname]
        params = CONFIG_PARAMS[cname]
        X = df[feats]
        print(f"\n--- {cname} ({len(feats)} features) ---")
        t1 = time.time()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = []
        for _, (tr, va) in enumerate(kf.split(X)):
            m = xgb.XGBRegressor(**params)
            m.fit(X.iloc[tr], y_all[tr])
            p = m.predict(X.iloc[va])
            folds.append(dict(
                r2=r2_score(y_all[va], p),
                rmse=np.sqrt(mean_squared_error(y_all[va], p)),
                mae=mean_absolute_error(y_all[va], p)))
        r2m = np.mean([f["r2"] for f in folds])
        rmsem = np.mean([f["rmse"] for f in folds])
        maem = np.mean([f["mae"] for f in folds])
        print(f"  R²={r2m:.4f}  RMSE={rmsem:.2f}  MAE={maem:.2f} "
              f"({time.time()-t1:.0f}s)")
        kf_results[cname] = dict(r2=round(r2m, 4), rmse=round(rmsem, 2),
                                  mae=round(maem, 2))

    df.drop(columns=RFSI_COLS, inplace=True)

    # ═══════════════════════════════════════════════════════════════════════════
    #  LOSO (per-fold RFSI, new configs only)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"LOSO CV ({n_stn} stations x {len(configs_to_run)} configs: "
          f"{configs_to_run})")
    print(f"{'='*80}")

    all_base = sorted(set(f for cn in configs_to_run
                           for f in CONFIGS[cn] if f not in RFSI_COLS))
    base_arr = df[all_base].values
    base_col_map = {f: i for i, f in enumerate(all_base)}
    rfsi_col_map = {f: i for i, f in enumerate(RFSI_COLS)}

    for cn in configs_to_run:
        loso_results[cn] = []

    for fold_i, held_sid in enumerate(station_ids):
        nm = sid_name.get(held_sid, held_sid)[:45]
        rg = sid_region.get(held_sid, "?")
        mask_test = df["stationId"].values == held_sid
        n_test = mask_test.sum()
        if n_test < 10:
            print(f"  [{fold_i+1:2d}/{n_stn}] {nm:45s} | SKIP (n={n_test})")
            continue

        mask_train = ~mask_test
        y_test = y_all[mask_test]
        y_train = y_all[mask_train]

        t_fold = time.time()
        rfsi_fold = compute_rfsi(exclude_sid=held_sid, K=K_NN)
        rfsi_arr = np.column_stack([rfsi_fold[c] for c in RFSI_COLS])

        parts = [f"[{fold_i+1:2d}/{n_stn}] {nm:45s} |"]

        for cname in configs_to_run:
            feats = CONFIGS[cname]
            params = CONFIG_PARAMS[cname]
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

            m = xgb.XGBRegressor(**params)
            m.fit(X_tr, y_train)
            p = m.predict(X_te)

            r2 = r2_score(y_test, p)
            rmse = np.sqrt(mean_squared_error(y_test, p))
            mae = mean_absolute_error(y_test, p)

            loso_results[cname].append(dict(
                station_id=held_sid,
                station_name=sid_name.get(held_sid, held_sid),
                region=rg, n_rows=n_test,
                r2=round(r2, 4), rmse=round(rmse, 2), mae=round(mae, 2)))

            parts.append(f"{cname}={r2:+.3f}")

        fold_time = time.time() - t_fold
        remaining = fold_time * (n_stn - fold_i - 1)
        print(f"  {' '.join(parts)}  [{rg}]  "
              f"({fold_time:.0f}s, ETA {remaining/60:.0f}m)")

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE (best config by LOSO R²)
# ═══════════════════════════════════════════════════════════════════════════════

best_cfg = max(CONFIG_ORDER,
               key=lambda c: np.mean([r["r2"] for r in loso_results[c]])
               if c in loso_results else -999)
best_loso = np.mean([r["r2"] for r in loso_results[best_cfg]])

print(f"\n{'='*80}")
print(f"FEATURE IMPORTANCE — Config {best_cfg} "
      f"(best LOSO R²={best_loso:.4f})")
print(f"{'='*80}")

if rfsi_global is None:
    rfsi_global = compute_rfsi(exclude_sid=None, K=K_NN)

for col in RFSI_COLS:
    df[col] = rfsi_global[col]

feats_best = CONFIGS[best_cfg]
params_best = CONFIG_PARAMS[best_cfg]
X_full = df[feats_best]
model_full = xgb.XGBRegressor(**params_best)
model_full.fit(X_full, y_all)

importance = model_full.get_booster().get_score(importance_type="gain")
imp_df = pd.DataFrame(
    [{"feature": k, "gain": v} for k, v in importance.items()]
).sort_values("gain", ascending=False).reset_index(drop=True)
feat_map = {f"f{i}": name for i, name in enumerate(feats_best)}
imp_df["feature"] = imp_df["feature"].map(lambda x: feat_map.get(x, x))
imp_df["rank"] = range(1, len(imp_df) + 1)
imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance_exp04.csv"),
              index=False, encoding="utf-8-sig")

print("\nTop 20 features by gain:")
for _, r in imp_df.head(20).iterrows():
    tag = " *RFSI*" if r["feature"] in RFSI_COLS else ""
    print(f"  {r['rank']:2d}. {r['feature']:30s} gain={r['gain']:.0f}{tag}")

df.drop(columns=RFSI_COLS, inplace=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE ALL RESULTS (incremental)
# ═══════════════════════════════════════════════════════════════════════════════

all_loso_rows = []
for cfg in CONFIG_ORDER:
    if cfg not in loso_results:
        continue
    for r in loso_results[cfg]:
        all_loso_rows.append({"config": cfg, **r})
pd.DataFrame(all_loso_rows).to_csv(LOSO_ALL_CSV, index=False,
                                    encoding="utf-8-sig")

# Also save best-config-only for backward compat
best_loso_df = pd.DataFrame(loso_results[best_cfg]).sort_values(
    "r2", ascending=True)
best_loso_df.to_csv(os.path.join(OUT_DIR, "loso_per_station_exp04.csv"),
                     index=False, encoding="utf-8-sig")

all_kf_rows = [{"config": cfg, **kf_results[cfg]}
                for cfg in CONFIG_ORDER if cfg in kf_results]
pd.DataFrame(all_kf_rows).to_csv(KFOLD_CSV, index=False, encoding="utf-8-sig")

loso_c = pd.read_csv(os.path.join(DATA_DIR,
                      "analysis/thesis_experiments/loso_per_station_config_c.csv"),
                      dtype={"station_id": str})
c_r2 = dict(zip(loso_c["station_id"], loso_c["r2"]))

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


sums = {c: loso_summary(loso_results[c]) for c in CONFIG_ORDER
        if c in loso_results}

rpt = []
rpt.append("# Experiment 04: RFSI Nearest-Station Features\n")
rpt.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rpt.append(f"**Dataset:** {len(df):,} rows, {n_stn} stations")
rpt.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, "
           f"lr=0.05, device=cuda")
rpt.append(f"**K6 overrides:** max_depth=5, colsample_bytree=0.6, "
           f"min_child_weight=20, monotonic constraints")
rpt.append(f"**RFSI:** K={K_NN} nearest neighbors, haversine distances\n")

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

descs = {"K1": "RFSI + temporal", "K2": "Full + RFSI",
         "K3": "Met+AOD+RFSI (no geo)", "K4": "Met+RFSI (no AOD)",
         "K5": "Minimal physics", "K6": "K5 + constrained XGB"}
for cn in CONFIG_ORDER:
    if cn not in sums or cn not in kf_results:
        continue
    nf = len(CONFIGS[cn])
    kf = kf_results[cn]
    s = sums[cn]
    gap = round(kf["r2"] - s["mean_r2"], 4)
    rpt.append(f"| {cn} | {descs[cn]} | {nf} | {kf['r2']:.4f} | "
               f"{s['mean_r2']:.4f} | {s['median_r2']:.4f} | "
               f"{s['neg_count']} | {gap:.4f} |")

# ── Per-station LOSO (best config vs Config C) ──
rpt.append(f"\n## Per-Station LOSO: Config {best_cfg} vs Config C\n")
rpt.append(f"| Station | Region | C R² | {best_cfg} R² | "
           f"Delta | {best_cfg} RMSE |")
rpt.append("|---------|--------|------|------|-------|--------|")

for _, r in best_loso_df.iterrows():
    nm = str(r["station_name"])[:50]
    cv = c_r2.get(r["station_id"], np.nan)
    bv = r["r2"]
    if pd.notna(cv):
        delta = bv - cv
        flag = " ✓" if delta > 0 else ""
        rpt.append(f"| {nm} | {r['region']} | {cv:.4f} | {bv:.4f} | "
                   f"{delta:+.4f}{flag} | {r['rmse']:.1f} |")
    else:
        rpt.append(f"| {nm} | {r['region']} | — | {bv:.4f} | — | "
                   f"{r['rmse']:.1f} |")

# ── Regional breakdown ──
avail = [cn for cn in CONFIG_ORDER if cn in sums]
rpt.append("\n## Regional Breakdown\n")
hdr = "| Region | C LOSO R² | " + " | ".join(f"{c} R²" for c in avail) + " |"
sep = "|--------|-----------|" + "|".join(["----------"] * len(avail)) + "|"
rpt.append(hdr)
rpt.append(sep)

c_reg = {"North": 0.0458, "Central": -0.0211, "South": -2.1908}
for rg in ["North", "Central", "South"]:
    cv = c_reg.get(rg, 0)
    vals = []
    for cn in avail:
        v = sums[cn]["by_region"].get(rg, {}).get("mean_r2")
        vals.append(f"{v:.4f}" if v is not None else "—")
    rpt.append(f"| {rg} | {cv:.4f} | " + " | ".join(vals) + " |")

# ── Feature importance ──
rpt.append(f"\n## Feature Importance (Config {best_cfg}, top 20)\n")
rpt.append("| Rank | Feature | Gain | RFSI? |")
rpt.append("|------|---------|------|-------|")
for _, r in imp_df.head(20).iterrows():
    tag = "yes" if r["feature"] in RFSI_COLS else ""
    rpt.append(f"| {r['rank']} | {r['feature']} | {r['gain']:.0f} | {tag} |")

rpt.append(f"\n### RFSI Feature Rankings\n")
rpt.append("| Feature | Rank | Gain |")
rpt.append("|---------|------|------|")
for col in RFSI_COLS:
    match = imp_df[imp_df["feature"] == col]
    if len(match):
        r = match.iloc[0]
        rpt.append(f"| {col} | {r['rank']} | {r['gain']:.0f} |")
    else:
        rpt.append(f"| {col} | not used | 0 |")

# ── Analysis ──
rpt.append("\n## Analysis\n")

rpt.append("### 1. Does RFSI alone (K1) beat the baseline?\n")
if "K1" in sums:
    k1 = sums["K1"]["mean_r2"]
    rpt.append(f"- Config C baseline: LOSO R² = -0.4953")
    rpt.append(f"- Config K1 (RFSI only): LOSO R² = {k1:.4f}")
    if k1 > -0.4953:
        rpt.append(f"- RFSI alone beats the baseline by {k1 - (-0.4953):+.4f}")
    else:
        rpt.append(f"- RFSI alone does not beat the baseline")
    rpt.append(f"- Oracle ceiling (Config E): LOSO R² = 0.2252")

rpt.append(f"\n### 2. Does Full + RFSI (K2) set a new best?\n")
if "K2" in sums:
    k2 = sums["K2"]["mean_r2"]
    rpt.append(f"- Config K2: LOSO R² = {k2:.4f}")
    if k2 > 0.2252:
        rpt.append(f"- New best — surpasses oracle anomaly (0.2252)")
    elif k2 > -0.4953:
        rpt.append(f"- Better than baseline but below oracle (0.2252)")
    else:
        rpt.append(f"- Does not beat baseline")

rpt.append(f"\n### 3. K5 minimal physics vs K2 kitchen-sink\n")
if "K5" in sums:
    k5 = sums["K5"]["mean_r2"]
    k2v = sums.get("K2", {}).get("mean_r2", "N/A")
    rpt.append(f"- K2 (75 features): LOSO R² = {k2v}")
    rpt.append(f"- K5 (17 features): LOSO R² = {k5:.4f}")
    if isinstance(k2v, float):
        rpt.append(f"- Delta: {k5 - k2v:+.4f}")

rpt.append(f"\n### 4. Do monotonic constraints help? (K6 vs K5)\n")
if "K5" in sums and "K6" in sums:
    k5 = sums["K5"]["mean_r2"]
    k6 = sums["K6"]["mean_r2"]
    rpt.append(f"- K5 (unconstrained): LOSO R² = {k5:.4f}, "
               f"neg={sums['K5']['neg_count']}")
    rpt.append(f"- K6 (constrained):   LOSO R² = {k6:.4f}, "
               f"neg={sums['K6']['neg_count']}")
    rpt.append(f"- Delta: {k6 - k5:+.4f}")
    if k6 > k5:
        rpt.append(f"- Constraints help — regularization improves generalization")
    else:
        rpt.append(f"- Constraints do not help")

rpt.append(f"\n### 5. KFold-LOSO gap\n")
for cn in CONFIG_ORDER:
    if cn in kf_results and cn in sums:
        kfr = kf_results[cn]["r2"]
        lr = sums[cn]["mean_r2"]
        rpt.append(f"- {cn}: KFold={kfr:.4f}, LOSO={lr:.4f}, "
                   f"gap={kfr - lr:.4f}")
rpt.append(f"- Baseline (C) gap: 1.2215")

report_path = os.path.join(OUT_DIR, "experiment_04_rfsi.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))

print(f"\nReport: {report_path}")
print(f"LOSO all configs: {LOSO_ALL_CSV}")
print(f"KFold summary: {KFOLD_CSV}")
print(f"Feature importance: {os.path.join(OUT_DIR, 'feature_importance_exp04.csv')}")
print(f"Station distances: {os.path.join(OUT_DIR, 'station_distances.csv')}")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
