"""
Experiment 07: Building Density as Spatial Baseline Proxy

The architecture review identified station-mean PM2.5 (driven by local emissions
and urbanization) as the strongest predictor of LOSO failure (Spearman r=0.60).
Building density from Google Open Buildings is our proxy for that missing signal.

Configs:
  B1 — K2 (full+RFSI) + building density (4 features)
  B2 — K3 (met+AOD+RFSI, no lat/lon/elevation/slope) + building density
  B3 — Minimal: 3 RFSI + building density + 10 physics + 2 temporal

Output:
  analysis/thesis_experiments/experiment_07_buildings.md
  analysis/thesis_experiments/loso_per_station_exp07.csv
  analysis/thesis_experiments/kfold_exp07.csv
  analysis/thesis_experiments/feature_importance_exp07.csv
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
print("EXPERIMENT 07: BUILDING DENSITY AS SPATIAL BASELINE PROXY")
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
bld = pd.read_csv(bld_path, dtype={"stationId": str})
BUILDING_COLS = ["building_count_1km", "building_area_1km",
                 "building_count_3km", "building_area_3km"]

bld_map = bld.set_index("stationId")[BUILDING_COLS]
n_before = len(df)
df = df.merge(bld_map, left_on="stationId", right_index=True, how="left")
assert len(df) == n_before, "Merge changed row count"

matched = df[BUILDING_COLS[0]].notna().sum()
print(f"Matched {matched:,}/{n_before:,} rows with building density")
if matched < n_before:
    missing_sids = df.loc[df[BUILDING_COLS[0]].isna(), "stationId"].unique()
    print(f"WARNING: {len(missing_sids)} stations missing building data: "
          f"{missing_sids[:5]}")
    for col in BUILDING_COLS:
        df[col] = df[col].fillna(0)

bld_stats = df.groupby("stationId")[BUILDING_COLS[0]].first()
print(f"Building count 1km: min={bld_stats.min():.0f}, "
      f"median={bld_stats.median():.0f}, max={bld_stats.max():.0f}")

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

# B1: K2 (full+RFSI) + buildings
FEATURES_B1 = MET + AOD + GEO + TEMPORAL + RFSI_COLS + BUILDING_COLS

# B2: K3 (met+AOD+RFSI, no geo) + buildings
FEATURES_B2 = MET + AOD + TEMPORAL + RFSI_COLS + BUILDING_COLS

# B3: minimal RFSI + buildings + physics + temporal
FEATURES_B3 = (["PM25_nn_idw", "PM25_nn_mean", "dist_nn1"] +
               BUILDING_COLS +
               ["PBLH", "WS_om", "Temperature_final", "Humidity_final",
                "AOT", "RF", "VC", "precip_mm",
                "month_sin", "month_cos"])

CONFIGS = {"B1": FEATURES_B1, "B2": FEATURES_B2, "B3": FEATURES_B3}
CONFIG_ORDER = ["B1", "B2", "B3"]

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
#  LOAD PREVIOUS RESULTS (Exp04 K1-K6 for comparison)
# ═══════════════════════════════════════════════════════════════════════════════

prev_loso = {}
prev_kf = {}

loso04_path = os.path.join(OUT_DIR, "loso_per_station_exp04_all.csv")
kf04_path = os.path.join(OUT_DIR, "kfold_exp04.csv")

if os.path.exists(loso04_path):
    tmp = pd.read_csv(loso04_path, dtype={"station_id": str})
    for cfg in tmp["config"].unique():
        sub = tmp[tmp["config"] == cfg].drop(columns=["config"])
        prev_loso[cfg] = sub.to_dict("records")
    print(f"\nLoaded Exp04 LOSO: {list(prev_loso.keys())}")

if os.path.exists(kf04_path):
    tmp = pd.read_csv(kf04_path)
    for _, row in tmp.iterrows():
        prev_kf[row["config"]] = dict(r2=row["r2"], rmse=row["rmse"],
                                       mae=row["mae"])
    print(f"Loaded Exp04 KFold: {list(prev_kf.keys())}")

# Config C reference
loso_c_path = os.path.join(DATA_DIR,
                            "analysis/thesis_experiments/loso_per_station_config_c.csv")
c_r2 = {}
if os.path.exists(loso_c_path):
    loso_c = pd.read_csv(loso_c_path, dtype={"station_id": str})
    c_r2 = dict(zip(loso_c["station_id"], loso_c["r2"]))

# ═══════════════════════════════════════════════════════════════════════════════
#  KFOLD (global RFSI)
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

nn1_nan = np.isnan(rfsi_global["PM25_nn1"]).sum()
print(f"PM25_nn1 NaN: {nn1_nan:,} ({nn1_nan/len(df)*100:.1f}%)")

kf_results = {}

for cname in CONFIG_ORDER:
    feats = CONFIGS[cname]
    X = df[feats]
    print(f"\n--- {cname} ({len(feats)} features) ---")
    t1 = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for _, (tr, va) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(**XGB_PARAMS)
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

# ═══════════════════════════════════════════════════════════════════════════════
#  LOSO (per-fold RFSI)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"LOSO CV ({n_stn} stations x {len(CONFIG_ORDER)} configs)")
print(f"{'='*80}")

all_base = sorted(set(f for cn in CONFIG_ORDER
                       for f in CONFIGS[cn] if f not in RFSI_COLS))
base_arr = df[all_base].values
base_col_map = {f: i for i, f in enumerate(all_base)}
rfsi_col_map = {f: i for i, f in enumerate(RFSI_COLS)}

loso_results = {cn: [] for cn in CONFIG_ORDER}

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

    for cname in CONFIG_ORDER:
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
model_full = xgb.XGBRegressor(**XGB_PARAMS)
model_full.fit(X_full, y_all)

importance = model_full.get_booster().get_score(importance_type="gain")
imp_df = pd.DataFrame(
    [{"feature": k, "gain": v} for k, v in importance.items()]
).sort_values("gain", ascending=False).reset_index(drop=True)
feat_map = {f"f{i}": name for i, name in enumerate(feats_best)}
imp_df["feature"] = imp_df["feature"].map(lambda x: feat_map.get(x, x))
imp_df["rank"] = range(1, len(imp_df) + 1)
imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance_exp07.csv"),
              index=False, encoding="utf-8-sig")

print("\nTop 20 features by gain:")
for _, r in imp_df.head(20).iterrows():
    tag = ""
    if r["feature"] in RFSI_COLS:
        tag = " *RFSI*"
    elif r["feature"] in BUILDING_COLS:
        tag = " *BUILDING*"
    print(f"  {r['rank']:2d}. {r['feature']:30s} gain={r['gain']:.0f}{tag}")

print("\nBuilding feature rankings:")
for col in BUILDING_COLS:
    match = imp_df[imp_df["feature"] == col]
    if len(match):
        r = match.iloc[0]
        print(f"  {col}: rank={r['rank']}, gain={r['gain']:.0f}")
    else:
        print(f"  {col}: not used")

df.drop(columns=RFSI_COLS, inplace=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

all_loso_rows = []
for cfg in CONFIG_ORDER:
    for r in loso_results[cfg]:
        all_loso_rows.append({"config": cfg, **r})
pd.DataFrame(all_loso_rows).to_csv(
    os.path.join(OUT_DIR, "loso_per_station_exp07.csv"),
    index=False, encoding="utf-8-sig")

pd.DataFrame([{"config": cfg, **kf_results[cfg]} for cfg in CONFIG_ORDER]).to_csv(
    os.path.join(OUT_DIR, "kfold_exp07.csv"),
    index=False, encoding="utf-8-sig")

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

# Also compute summaries for loaded exp04 configs
prev_sums = {}
for c, recs in prev_loso.items():
    prev_sums[c] = loso_summary(recs)

rpt = []
rpt.append("# Experiment 07: Building Density as Spatial Baseline Proxy\n")
rpt.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rpt.append(f"**Dataset:** {len(df):,} rows, {n_stn} stations")
rpt.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, "
           f"lr=0.05, device=cuda")
rpt.append(f"**RFSI:** K={K_NN} nearest neighbors")
rpt.append(f"**Building density:** Google Open Buildings, 1km and 3km radius\n")

rpt.append("## Hypothesis\n")
rpt.append("The architecture review identified station-mean PM2.5 (driven by "
           "local emissions and urbanization) as the strongest predictor of LOSO "
           "failure (Spearman r=0.60). Building density from Google Open Buildings "
           "serves as a proxy for urbanization intensity — the missing spatial "
           "baseline signal that the model needs to distinguish clean rural "
           "stations from polluted urban ones without memorizing station identity.\n")

# ── Building density summary ──
rpt.append("## Building Density Statistics\n")
rpt.append("| Station | Region | PM2.5 mean | Bldg count 1km | Bldg area 1km (m²) | "
           "Bldg count 3km |")
rpt.append("|---------|--------|-----------|----------------|---------------------|"
           "----------------|")

stn_pm = df.groupby("stationId")["PM2.5"].mean()
bld_join = bld.set_index("stationId")
for sid in sorted(station_ids, key=lambda s: stn_pm.get(s, 0)):
    nm = sid_name.get(sid, sid)[:45]
    rg = sid_region.get(sid, "?")
    pm = stn_pm.get(sid, np.nan)
    if sid in bld_join.index:
        bc1 = bld_join.loc[sid, "building_count_1km"]
        ba1 = bld_join.loc[sid, "building_area_1km"]
        bc3 = bld_join.loc[sid, "building_count_3km"]
    else:
        bc1 = ba1 = bc3 = 0
    rpt.append(f"| {nm} | {rg} | {pm:.1f} | {bc1:,.0f} | {ba1:,.0f} | {bc3:,.0f} |")

# ── Comparison table ──
rpt.append("\n## Comparison Table (all configs)\n")
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

# Exp04 configs
exp04_descs = {"K1": "RFSI + temporal", "K2": "Full + RFSI",
               "K3": "Met+AOD+RFSI (no geo)", "K4": "Met+RFSI (no AOD)",
               "K5": "Minimal physics", "K6": "K5 + constrained XGB"}
for cn in ["K1", "K2", "K3", "K4", "K5", "K6"]:
    if cn not in prev_sums or cn not in prev_kf:
        continue
    s = prev_sums[cn]
    kf = prev_kf[cn]
    gap = round(kf["r2"] - s["mean_r2"], 4)
    rpt.append(f"| {cn} (Exp04) | {exp04_descs[cn]} | — | {kf['r2']:.4f} | "
               f"{s['mean_r2']:.4f} | {s['median_r2']:.4f} | "
               f"{s['neg_count']} | {gap:.4f} |")

# This experiment's configs
descs = {"B1": "K2 + buildings", "B2": "K3 + buildings (no geo)",
         "B3": "Minimal RFSI + buildings + physics"}
for cn in CONFIG_ORDER:
    nf = len(CONFIGS[cn])
    kf = kf_results[cn]
    s = sums[cn]
    gap = round(kf["r2"] - s["mean_r2"], 4)
    rpt.append(f"| **{cn}** | **{descs[cn]}** | {nf} | {kf['r2']:.4f} | "
               f"**{s['mean_r2']:.4f}** | {s['median_r2']:.4f} | "
               f"{s['neg_count']} | {gap:.4f} |")

# ── Per-station LOSO (best config vs Config C and K2) ──
best_loso_df = pd.DataFrame(loso_results[best_cfg]).sort_values(
    "r2", ascending=True)

# Get K2 LOSO for comparison
k2_r2 = {}
if "K2" in prev_loso:
    k2_r2 = {r["station_id"]: r["r2"] for r in prev_loso["K2"]}

rpt.append(f"\n## Per-Station LOSO: Config {best_cfg} vs K2 vs Config C\n")
rpt.append(f"| Station | Region | C R² | K2 R² | {best_cfg} R² | "
           f"Δ vs K2 | {best_cfg} RMSE |")
rpt.append("|---------|--------|------|-------|-------|---------|--------|")

for _, r in best_loso_df.iterrows():
    nm = str(r["station_name"])[:45]
    cv = c_r2.get(r["station_id"], np.nan)
    k2v = k2_r2.get(r["station_id"], np.nan)
    bv = r["r2"]
    cv_s = f"{cv:.4f}" if pd.notna(cv) else "—"
    k2v_s = f"{k2v:.4f}" if pd.notna(k2v) else "—"
    if pd.notna(k2v):
        delta = bv - k2v
        flag = " ✓" if delta > 0 else ""
        delta_s = f"{delta:+.4f}{flag}"
    else:
        delta_s = "—"
    rpt.append(f"| {nm} | {r['region']} | {cv_s} | {k2v_s} | {bv:.4f} | "
               f"{delta_s} | {r['rmse']:.1f} |")

# ── Regional breakdown ──
rpt.append("\n## Regional Breakdown\n")
all_cfgs = []
if "K2" in prev_sums:
    all_cfgs.append("K2")
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
rpt.append(f"\n## Feature Importance (Config {best_cfg}, top 20)\n")
rpt.append("| Rank | Feature | Gain | Type |")
rpt.append("|------|---------|------|------|")
for _, r in imp_df.head(20).iterrows():
    if r["feature"] in BUILDING_COLS:
        tag = "BUILDING"
    elif r["feature"] in RFSI_COLS:
        tag = "RFSI"
    else:
        tag = ""
    rpt.append(f"| {r['rank']} | {r['feature']} | {r['gain']:.0f} | {tag} |")

rpt.append(f"\n### Building Feature Rankings\n")
rpt.append("| Feature | Rank | Gain |")
rpt.append("|---------|------|------|")
for col in BUILDING_COLS:
    match = imp_df[imp_df["feature"] == col]
    if len(match):
        r = match.iloc[0]
        rpt.append(f"| {col} | {r['rank']} | {r['gain']:.0f} |")
    else:
        rpt.append(f"| {col} | not used | 0 |")

# ── Analysis ──
rpt.append("\n## Analysis\n")

rpt.append("### 1. Do buildings help the kitchen-sink model? (B1 vs K2)\n")
k2_loso = prev_sums.get("K2", {}).get("mean_r2")
b1_loso = sums["B1"]["mean_r2"]
if k2_loso is not None:
    rpt.append(f"- K2 (full+RFSI, no buildings): LOSO R² = {k2_loso:.4f}")
    rpt.append(f"- B1 (K2 + buildings): LOSO R² = {b1_loso:.4f}")
    rpt.append(f"- Delta: {b1_loso - k2_loso:+.4f}")
    if b1_loso > k2_loso:
        rpt.append(f"- Buildings improve the full model")
    else:
        rpt.append(f"- Buildings do not help when lat/lon is already present "
                   f"(geographic proxies already capture urbanization gradient)")

rpt.append("\n### 2. Can buildings replace geography? (B2 vs K3 vs K2)\n")
k3_loso = prev_sums.get("K3", {}).get("mean_r2")
b2_loso = sums["B2"]["mean_r2"]
if k3_loso is not None and k2_loso is not None:
    rpt.append(f"- K2 (with geo): LOSO R² = {k2_loso:.4f}")
    rpt.append(f"- K3 (no geo, no buildings): LOSO R² = {k3_loso:.4f}")
    rpt.append(f"- B2 (no geo, with buildings): LOSO R² = {b2_loso:.4f}")
    rpt.append(f"- Buildings recover {b2_loso - k3_loso:+.4f} of the geo gap")
    if b2_loso > k2_loso:
        rpt.append(f"- B2 surpasses K2 — buildings are a better spatial signal "
                   f"than lat/lon for LOSO")

rpt.append("\n### 3. Minimal model with buildings (B3)\n")
b3_loso = sums["B3"]["mean_r2"]
rpt.append(f"- B3 (17 features): LOSO R² = {b3_loso:.4f}, "
           f"neg={sums['B3']['neg_count']}")
rpt.append(f"- Oracle ceiling (Config E): LOSO R² = 0.2252")
if b3_loso > 0:
    rpt.append(f"- First positive LOSO R² achieved!")

rpt.append("\n### 4. KFold-LOSO gap (identity leakage diagnostic)\n")
for cn in CONFIG_ORDER:
    kfr = kf_results[cn]["r2"]
    lr = sums[cn]["mean_r2"]
    rpt.append(f"- {cn}: KFold={kfr:.4f}, LOSO={lr:.4f}, gap={kfr - lr:.4f}")
rpt.append(f"- Baseline (C) gap: 1.2215")
rpt.append(f"- Oracle (E) gap: 0.4674")

rpt.append("\n### 5. Impact on disaster stations\n")
disasters = ["Trà Vinh Đông Hải", "Đà Nẵng Phạm Hùng", "Sóc Trăng",
             "Tây Ninh Trảng Bàng"]
best_recs = {r["station_name"]: r for r in loso_results[best_cfg]}
for d in disasters:
    matches = [r for name, r in best_recs.items() if d in name]
    if matches:
        r = matches[0]
        cv = c_r2.get(r["station_id"], np.nan)
        k2v = k2_r2.get(r["station_id"], np.nan)
        nm = r["station_name"][:45]
        parts = [f"- {nm}: {best_cfg}={r['r2']:.4f}"]
        if pd.notna(k2v):
            parts.append(f"K2={k2v:.4f}")
        if pd.notna(cv):
            parts.append(f"C={cv:.4f}")
        bld_row = bld_join.loc[r["station_id"]] if r["station_id"] in bld_join.index else None
        if bld_row is not None:
            parts.append(f"(bldg_1km={bld_row['building_count_1km']:,.0f})")
        rpt.append(", ".join(parts))

rpt.append("\n### 6. Correlation: building density vs station PM2.5 mean\n")
stn_bld = pd.DataFrame({
    "stationId": station_ids,
    "pm25_mean": [stn_pm.get(s, np.nan) for s in station_ids],
    "bldg_1km": [bld_join.loc[s, "building_count_1km"]
                 if s in bld_join.index else 0 for s in station_ids],
    "bldg_3km": [bld_join.loc[s, "building_count_3km"]
                 if s in bld_join.index else 0 for s in station_ids],
})
from scipy import stats
r1, p1 = stats.spearmanr(stn_bld["pm25_mean"].dropna(),
                          stn_bld.loc[stn_bld["pm25_mean"].notna(), "bldg_1km"])
r3, p3 = stats.spearmanr(stn_bld["pm25_mean"].dropna(),
                          stn_bld.loc[stn_bld["pm25_mean"].notna(), "bldg_3km"])
rpt.append(f"- Spearman(PM2.5 mean, building_count_1km) = {r1:.3f} (p={p1:.4f})")
rpt.append(f"- Spearman(PM2.5 mean, building_count_3km) = {r3:.3f} (p={p3:.4f})")
if abs(r1) > 0.3:
    rpt.append(f"- Building density is a meaningful proxy for station PM2.5 level")
else:
    rpt.append(f"- Building density is a weak proxy for station PM2.5 level")

report_path = os.path.join(OUT_DIR, "experiment_07_buildings.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))

print(f"\nReport: {report_path}")
print(f"LOSO: {os.path.join(OUT_DIR, 'loso_per_station_exp07.csv')}")
print(f"KFold: {os.path.join(OUT_DIR, 'kfold_exp07.csv')}")
print(f"Feature importance: {os.path.join(OUT_DIR, 'feature_importance_exp07.csv')}")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
