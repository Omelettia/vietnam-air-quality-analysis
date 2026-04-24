"""
Experiment 05: Daily Aggregation Test

Tests whether switching from hourly to daily resolution fixes LOSO.
Aggregates unified_thesis_v1.csv to daily means, rebuilds RFSI on the
daily PM2.5 matrix, runs D_K1 (RFSI+temporal) and D_K2 (full+RFSI).

Output:
  analysis/thesis_experiments/experiment_05_daily.md
  analysis/thesis_experiments/loso_per_station_exp05.csv
  analysis/thesis_experiments/feature_importance_exp05.csv
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
#  LOAD & AGGREGATE TO DAILY
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("EXPERIMENT 05: DAILY AGGREGATION TEST")
print("=" * 80)

t0 = time.time()
raw = pd.read_csv(os.path.join(DATA_DIR, "data/merged/unified_thesis_v1.csv"),
                  dtype={"stationId": str})
raw = raw.dropna(subset=["PM2.5"]).reset_index(drop=True)
raw["ts"] = pd.to_datetime(raw["ts"])
print(f"Loaded hourly: {len(raw):,} rows, {raw['stationId'].nunique()} stations "
      f"({time.time()-t0:.1f}s)")

meta = pd.read_csv(os.path.join(DATA_DIR,
                    "analysis/thesis_audit/station_selection_final.csv"),
                    dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))
sid_region = dict(zip(meta["stationId"], meta["region"]))
sid_lat = dict(zip(meta["stationId"], meta["lat"]))
sid_lon = dict(zip(meta["stationId"], meta["lon"]))

# ── Build aggregation rules ──
raw["date"] = raw["ts"].dt.date

MEAN_COLS = [
    "PM2.5",
    "Temperature_final", "Humidity_final", "Pressure_final",
    "PBLH", "VC", "RH_factor",
    "wind_u", "wind_v", "wind_dir_sin", "wind_dir_cos",
    "WS_local", "wind_u_local", "wind_v_local",
    "wind_dir_sin_local", "wind_dir_cos_local",
    "dT_6h", "dRH_6h", "dWS_6h", "dP_6h",
    "rain_sum_24h", "rain_sum_48h", "rain_days_7d", "consecutive_dry_days",
    "AOT", "AOT_mean", "AOT_inner_mean", "AOT_outer_mean",
    "RF", "SSA", "Uncertainty", "AE",
    "AOT_valid_count", "AOD_physics", "AOT_spatial_std", "AOT_local_vs_regional",
    "AOT_ffill_48h", "hours_since_valid_AOT",
    "AOT_lag_1h", "AOT_lag_3h", "AOT_lag_6h",
    "AOT_rolling_mean_6h", "AOT_rolling_mean_24h",
    "AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_grad_dir",
    "elev_x_PBLH",
]

agg = {}
for c in MEAN_COLS:
    if c in raw.columns:
        agg[c] = "mean"
if "precip_mm" in raw.columns:
    agg["precip_mm"] = "sum"
if "hrs_since_rain" in raw.columns:
    agg["hrs_since_rain"] = "min"

FIRST_COLS = ["latitude", "longitude", "elevation_m", "slope_deg",
              "aspect_sin", "aspect_cos",
              "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos"]
for c in FIRST_COLS:
    if c in raw.columns:
        agg[c] = "first"

print(f"Aggregating {len(agg)} columns to daily ...")
t1 = time.time()
df = raw.groupby(["stationId", "date"]).agg(agg).reset_index()
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])
print(f"Daily dataset: {len(df):,} rows, {df['stationId'].nunique()} stations "
      f"({time.time()-t1:.1f}s)")
print(f"Compression: {len(raw):,} hourly → {len(df):,} daily "
      f"({len(raw)/len(df):.1f}x)")

station_ids = sorted(df["stationId"].unique())
n_stn = len(station_ids)
sid_to_idx = {s: i for i, s in enumerate(station_ids)}

TARGET = "PM2.5"
y_all = df[TARGET].values
print(f"Daily PM2.5: mean={y_all.mean():.1f}, std={y_all.std():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  STATION DISTANCES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Station distances ---")


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
#  DAILY PM2.5 WIDE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Daily PM2.5 wide matrix ---")
pm25_wide = df.pivot_table(index="date", columns="stationId",
                           values="PM2.5", aggfunc="first")
pm25_mat = pm25_wide.values
sid_cols = list(pm25_wide.columns)
sid_to_col = {s: i for i, s in enumerate(sid_cols)}

date_to_row = pd.Series(range(len(pm25_wide)), index=pm25_wide.index)
df["date_row"] = df["date"].map(date_to_row).astype(int).values

print(f"Shape: {pm25_mat.shape[0]:,} dates x {pm25_mat.shape[1]} stations")
print(f"Non-NaN: {(~np.isnan(pm25_mat)).mean():.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
#  RFSI COMPUTATION (daily)
# ═══════════════════════════════════════════════════════════════════════════════

RFSI_COLS = ([f"PM25_nn{k+1}" for k in range(K_NN)] +
             [f"dist_nn{k+1}" for k in range(K_NN)] +
             ["n_neighbors_available", "PM25_nn_mean", "PM25_nn_idw"])


def compute_rfsi(exclude_sid=None, K=5):
    """Compute daily RFSI features for every row in df."""
    n = len(df)
    pm_nn = np.full((n, K), np.nan)
    d_nn = np.full((n, K), np.nan)

    excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
    sid_vals = df["stationId"].values
    dr_vals = df["date_row"].values

    for sid in station_ids:
        si = sid_to_idx[sid]
        mask = sid_vals == sid
        if not mask.any():
            continue
        ri = np.where(mask)[0]
        dr = dr_vals[ri]

        cands = [(j, d) for j, d in neighbor_order[si]
                 if excl is None or j != excl]
        if not cands:
            continue

        ccols = np.array([sid_to_col[station_ids[j]] for j, _ in cands])
        cdists = np.array([d for _, d in cands])

        nbr = pm25_mat[np.ix_(dr, ccols)]
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
#  FEATURE SETS (daily — no hour_sin/cos, no elev_x_hour_sin)
# ═══════════════════════════════════════════════════════════════════════════════

TEMPORAL_D = ["month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos"]

MET_D = [
    "Temperature_final", "Humidity_final", "Pressure_final",
    "PBLH", "VC", "RH_factor",
    "wind_u", "wind_v", "wind_dir_sin", "wind_dir_cos",
    "WS_local", "wind_u_local", "wind_v_local",
    "wind_dir_sin_local", "wind_dir_cos_local",
    "dT_6h", "dRH_6h", "dWS_6h", "dP_6h",
    "precip_mm", "hrs_since_rain", "rain_sum_24h", "rain_sum_48h",
    "rain_days_7d", "consecutive_dry_days",
]

AOD_D = [
    "AOT", "AOT_mean", "AOT_inner_mean", "AOT_outer_mean",
    "RF", "SSA", "Uncertainty", "AE",
    "AOT_valid_count", "AOD_physics", "AOT_spatial_std", "AOT_local_vs_regional",
    "AOT_ffill_48h", "hours_since_valid_AOT",
    "AOT_lag_1h", "AOT_lag_3h", "AOT_lag_6h",
    "AOT_rolling_mean_6h", "AOT_rolling_mean_24h",
    "AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_grad_dir",
]

GEO_D = ["latitude", "longitude", "elevation_m", "slope_deg",
         "aspect_sin", "aspect_cos", "elev_x_PBLH"]

CONFIGS = {
    "D_K1": TEMPORAL_D + RFSI_COLS,
    "D_K2": MET_D + AOD_D + GEO_D + TEMPORAL_D + RFSI_COLS,
}

for cname, feats in list(CONFIGS.items()):
    missing = [f for f in feats if f not in df.columns and f not in RFSI_COLS]
    if missing:
        print(f"WARNING: {cname} missing columns: {missing}")
        CONFIGS[cname] = [f for f in feats if f in df.columns or f in RFSI_COLS]

for cname in CONFIGS:
    nb = len([f for f in CONFIGS[cname] if f not in RFSI_COLS])
    nr = len([f for f in CONFIGS[cname] if f in RFSI_COLS])
    print(f"  {cname}: {len(CONFIGS[cname])} features ({nb} base + {nr} RFSI)")

# ═══════════════════════════════════════════════════════════════════════════════
#  KFOLD (global RFSI)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("KFOLD 5-FOLD CV")
print(f"{'='*80}")

print("\nComputing global daily RFSI features ...")
t1 = time.time()
rfsi_global = compute_rfsi(exclude_sid=None, K=K_NN)
print(f"Done ({time.time()-t1:.1f}s)")

for col in RFSI_COLS:
    df[col] = rfsi_global[col]

nn1_nan = np.isnan(rfsi_global["PM25_nn1"]).sum()
nn5_nan = np.isnan(rfsi_global["PM25_nn5"]).sum()
print(f"PM25_nn1 NaN: {nn1_nan:,} ({nn1_nan/len(df)*100:.1f}%)")
print(f"PM25_nn5 NaN: {nn5_nan:,} ({nn5_nan/len(df)*100:.1f}%)")
print(f"PM25_nn_idw mean: {np.nanmean(rfsi_global['PM25_nn_idw']):.1f}")

kf_results = {}
for cname in CONFIGS:
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
    print(f"  R²={r2m:.4f}  RMSE={rmsem:.2f}  MAE={maem:.2f} ({time.time()-t1:.0f}s)")
    kf_results[cname] = dict(r2=round(r2m, 4), rmse=round(rmsem, 2),
                              mae=round(maem, 2))

df.drop(columns=RFSI_COLS, inplace=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOSO (per-fold RFSI)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"LOSO CV ({n_stn} stations x {len(CONFIGS)} configs)")
print(f"{'='*80}")

all_base = sorted(set(f for fs in CONFIGS.values()
                       for f in fs if f not in RFSI_COLS))
base_arr = df[all_base].values
base_col_map = {f: i for i, f in enumerate(all_base)}
rfsi_col_map = {f: i for i, f in enumerate(RFSI_COLS)}

loso_results = {c: [] for c in CONFIGS}

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

    for cname in CONFIGS:
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

best_cfg = max(CONFIGS, key=lambda c: np.mean([r["r2"] for r in loso_results[c]]))
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
imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance_exp05.csv"),
              index=False, encoding="utf-8-sig")

print("\nTop 20 features by gain:")
for _, r in imp_df.head(20).iterrows():
    tag = " *RFSI*" if r["feature"] in RFSI_COLS else ""
    print(f"  {r['rank']:2d}. {r['feature']:30s} gain={r['gain']:.0f}{tag}")

df.drop(columns=RFSI_COLS, inplace=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE PER-STATION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

best_loso_df = pd.DataFrame(loso_results[best_cfg]).sort_values(
    "r2", ascending=True)
best_loso_df.to_csv(os.path.join(OUT_DIR, "loso_per_station_exp05.csv"),
                     index=False, encoding="utf-8-sig")

# Load hourly exp04 K2 per-station results if available
exp04_path = os.path.join(OUT_DIR, "loso_per_station_exp04.csv")
if os.path.exists(exp04_path):
    exp04_df = pd.read_csv(exp04_path, dtype={"station_id": str})
    hourly_r2 = dict(zip(exp04_df["station_id"], exp04_df["r2"]))
else:
    hourly_r2 = {}

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


sums = {c: loso_summary(loso_results[c]) for c in CONFIGS}

rpt = []
rpt.append("# Experiment 05: Daily Aggregation Test\n")
rpt.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rpt.append(f"**Hourly dataset:** {len(raw):,} rows")
rpt.append(f"**Daily dataset:** {len(df):,} rows, {n_stn} stations")
rpt.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, "
           f"lr=0.05, device=cuda")
rpt.append(f"**RFSI:** K={K_NN} nearest neighbors (daily PM2.5)\n")

# ── Comparison table: hourly vs daily ──
rpt.append("## Comparison Table — Hourly vs Daily\n")
rpt.append("| Config | Resolution | Features | KFold R² | LOSO R² (mean) | "
           "LOSO R² (median) | Neg Stations | Gap |")
rpt.append("|--------|------------|----------|----------|----------------|"
           "------------------|--------------|-----|")

refs = [
    ("K1 (Exp04)", "Hourly", 19, 0.7855, -0.1520, -0.0030, "?", "?"),
    ("K2 (Exp04)", "Hourly", 75, 0.8099, -0.1140, 0.0480, "?", "?"),
    ("E (Exp02)",  "Hourly", 55, 0.6926,  0.2252, 0.2640, 7, 0.4674),
]
for cfg, res, nf, kf_r2, lm, lmed, neg, gap in refs:
    neg_s = str(neg)
    gap_s = f"{gap:.4f}" if isinstance(gap, float) else str(gap)
    lm_s = f"{lm:.4f}" if isinstance(lm, float) else str(lm)
    lmed_s = f"{lmed:.4f}" if isinstance(lmed, float) else str(lmed)
    rpt.append(f"| {cfg} | {res} | {nf} | {kf_r2:.4f} | {lm_s} | "
               f"{lmed_s} | {neg_s} | {gap_s} |")

for cn in CONFIGS:
    nf = len(CONFIGS[cn])
    kf = kf_results[cn]
    s = sums[cn]
    gap = round(kf["r2"] - s["mean_r2"], 4)
    rpt.append(f"| {cn} | Daily | {nf} | {kf['r2']:.4f} | "
               f"{s['mean_r2']:.4f} | {s['median_r2']:.4f} | "
               f"{s['neg_count']} | {gap:.4f} |")

# ── Per-station LOSO ──
rpt.append(f"\n## Per-Station LOSO: Daily {best_cfg} vs Hourly K2\n")
if hourly_r2:
    rpt.append(f"| Station | Region | Hourly K2 R² | Daily {best_cfg} R² | "
               f"Delta | Daily RMSE |")
    rpt.append("|---------|--------|--------------|------------|-------|------------|")
else:
    rpt.append(f"| Station | Region | Daily {best_cfg} R² | Daily RMSE |")
    rpt.append("|---------|--------|------------|------------|")

for _, r in best_loso_df.iterrows():
    nm = str(r["station_name"])[:50]
    bv = r["r2"]
    if hourly_r2:
        hv = hourly_r2.get(r["station_id"], np.nan)
        if pd.notna(hv):
            delta = bv - hv
            flag = " ✓" if delta > 0 else ""
            rpt.append(f"| {nm} | {r['region']} | {hv:.4f} | {bv:.4f} | "
                       f"{delta:+.4f}{flag} | {r['rmse']:.1f} |")
        else:
            rpt.append(f"| {nm} | {r['region']} | — | {bv:.4f} | — | "
                       f"{r['rmse']:.1f} |")
    else:
        rpt.append(f"| {nm} | {r['region']} | {bv:.4f} | {r['rmse']:.1f} |")

# ── Regional breakdown ──
rpt.append("\n## Regional Breakdown\n")
rpt.append("| Region | " + " | ".join(f"{c} R²" for c in CONFIGS) + " |")
rpt.append("|--------|" + "|".join(["----------"] * len(CONFIGS)) + "|")

for rg in ["North", "Central", "South"]:
    vals = []
    for cn in CONFIGS:
        v = sums[cn]["by_region"].get(rg, {}).get("mean_r2")
        vals.append(f"{v:.4f}" if v is not None else "—")
    rpt.append(f"| {rg} | " + " | ".join(vals) + " |")

# ── Feature importance ──
rpt.append(f"\n## Feature Importance (Config {best_cfg}, top 20)\n")
rpt.append("| Rank | Feature | Gain | RFSI? |")
rpt.append("|------|---------|------|-------|")
for _, r in imp_df.head(20).iterrows():
    tag = "yes" if r["feature"] in RFSI_COLS else ""
    rpt.append(f"| {r['rank']} | {r['feature']} | {r['gain']:.0f} | {tag} |")

# ── Key question ──
rpt.append("\n## Key Question: Does daily resolution fix LOSO?\n")

dk2 = sums["D_K2"]["mean_r2"]
rpt.append(f"- Hourly K2 (Exp04): LOSO R² = -0.1140")
rpt.append(f"- Daily D_K2:        LOSO R² = {dk2:.4f}")

if dk2 > 0.2:
    rpt.append(f"\n**YES** — daily resolution dramatically improves LOSO "
               f"(R² > +0.2). Hourly noise was the bottleneck.")
elif dk2 > 0:
    rpt.append(f"\n**PARTIAL** — daily helps (LOSO R² > 0) but doesn't reach "
               f"oracle level (0.225). Resolution is part of the problem.")
else:
    rpt.append(f"\n**NO** — daily resolution does not fix LOSO. "
               f"The problem is deeper: sparse network + Vietnam heterogeneity.")

rpt.append(f"\n- Daily D_K1 (RFSI only): LOSO R² = {sums['D_K1']['mean_r2']:.4f}")
rpt.append(f"- Oracle ceiling (Exp02): LOSO R² = 0.2252")

report_path = os.path.join(OUT_DIR, "experiment_05_daily.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))

print(f"\nReport: {report_path}")
print(f"LOSO per-station: {os.path.join(OUT_DIR, 'loso_per_station_exp05.csv')}")
print(f"Feature importance: {os.path.join(OUT_DIR, 'feature_importance_exp05.csv')}")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
