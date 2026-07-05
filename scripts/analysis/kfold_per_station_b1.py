"""
Per-station KFold R² for B1 config.

For each of 5 KFold splits, compute R² separately per station on that
station's test rows, then average across folds.  Compare with LOSO R²
to quantify the spatial-transfer gap per station.

Output: analysis/thesis_experiments/kfold_per_station_b1.csv
"""

import argparse, io, sys, os, warnings, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
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
#  LOAD
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PER-STATION KFOLD R² FOR B1")
print("=" * 70)

t0 = time.time()
df = pd.read_csv(os.path.join(DATA_DIR, "data/merged/unified_thesis_v1.csv"),
                  dtype={"stationId": str})
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations")

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

y_all = df["PM2.5"].values
stationId_vals = df["stationId"].values

# ── Building density ──
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

# ── Station distances + RFSI ──
print("Computing distances ...")

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

nn1_dist = {station_ids[i]: neighbor_order[i][0][1] for i in range(n_stn)}

# ── PM2.5 wide matrix ──
print("Building PM2.5 wide matrix ...")
pm25_wide = df.pivot_table(index="ts", columns="stationId",
                           values="PM2.5", aggfunc="first")
pm25_mat = pm25_wide.values
sid_cols = list(pm25_wide.columns)
sid_to_col = {s: i for i, s in enumerate(sid_cols)}

ts_to_row = pd.Series(range(len(pm25_wide)), index=pm25_wide.index)
df["ts_row"] = df["ts"].map(ts_to_row).astype(int).values

RFSI_COLS = ([f"PM25_nn{k+1}" for k in range(K_NN)] +
             [f"dist_nn{k+1}" for k in range(K_NN)] +
             ["n_neighbors_available", "PM25_nn_mean", "PM25_nn_idw"])

print("Computing global RFSI ...")
t1 = time.time()
n = len(df)
pm_nn = np.full((n, K_NN), np.nan)
d_nn = np.full((n, K_NN), np.nan)
ts_row_vals = df["ts_row"].values

for sid in station_ids:
    si = sid_to_idx[sid]
    mask = stationId_vals == sid
    if not mask.any():
        continue
    ri = np.where(mask)[0]
    tr = ts_row_vals[ri]
    cands = neighbor_order[si]
    ccols = np.array([sid_to_col[station_ids[j]] for j, _ in cands])
    cdists = np.array([d for _, d in cands])
    nbr = pm25_mat[np.ix_(tr, ccols)]
    valid = ~np.isnan(nbr)
    cumv = np.cumsum(valid, axis=1)
    for k in range(K_NN):
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

for k in range(K_NN):
    df[f"PM25_nn{k+1}"] = pm_nn[:, k]
    df[f"dist_nn{k+1}"] = d_nn[:, k]
df["n_neighbors_available"] = n_avail
df["PM25_nn_mean"] = pm_mean
df["PM25_nn_idw"] = pm_idw
print(f"RFSI done ({time.time()-t1:.1f}s)")

# ── B1 feature set (same as experiment_07) ──
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

FEATURES_B1 = MET + AOD + GEO + TEMPORAL + RFSI_COLS + BUILDING_COLS
FEATURES_B1 = [f for f in FEATURES_B1 if f in df.columns]
print(f"B1: {len(FEATURES_B1)} features")

X = df[FEATURES_B1].values

# ── Load LOSO results for comparison ──
loso_path = os.path.join(OUT_DIR, "loso_per_station_exp07.csv")
loso_df = pd.read_csv(loso_path, dtype={"station_id": str})
loso_b1 = loso_df[loso_df["config"] == "B1"]
loso_r2 = dict(zip(loso_b1["station_id"], loso_b1["r2"]))

# Station PM2.5 means
pm_means = df.groupby("stationId")["PM2.5"].mean().to_dict()

# ═══════════════════════════════════════════════════════════════════════════════
#  KFOLD — per-station R²
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning 5-fold KFold ...")
t1 = time.time()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Collect per-station predictions across all folds
pred_all = np.full(len(df), np.nan)

for fold_i, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"  Fold {fold_i+1}/5 ... ", end="", flush=True)
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X[tr_idx], y_all[tr_idx])
    pred_all[va_idx] = m.predict(X[va_idx])
    print(f"done")

print(f"KFold done ({time.time()-t1:.0f}s)")

# ── Compute per-station KFold R² ──
rows = []
for sid in station_ids:
    mask = stationId_vals == sid
    y_true = y_all[mask]
    y_pred = pred_all[mask]
    valid = ~np.isnan(y_pred)
    if valid.sum() < 20:
        continue
    kf_r2 = r2_score(y_true[valid], y_pred[valid])
    lr2 = loso_r2.get(sid, np.nan)
    gap = kf_r2 - lr2 if pd.notna(lr2) else np.nan
    rows.append({
        "station_id": sid,
        "station_name": sid_name.get(sid, sid),
        "region": sid_region.get(sid, "?"),
        "kfold_r2": round(kf_r2, 4),
        "loso_r2": round(lr2, 4) if pd.notna(lr2) else np.nan,
        "gap": round(gap, 4) if pd.notna(gap) else np.nan,
        "pm25_mean": round(pm_means.get(sid, np.nan), 1),
        "nn1_km": round(nn1_dist.get(sid, np.nan), 0),
    })

result = pd.DataFrame(rows).sort_values("gap", ascending=False)

# ── Save ──
out_path = os.path.join(OUT_DIR, "kfold_per_station_b1.csv")
result.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_path}")

# ── Print ──
print(f"\n{'='*95}")
print(f"  {'Station':<50s} {'KFold':>7s} {'LOSO':>7s} {'Gap':>7s} {'PM2.5':>6s} {'NN1':>5s}")
print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*5}")
for _, r in result.iterrows():
    nm = str(r["station_name"])[:50]
    print(f"  {nm:<50s} {r['kfold_r2']:+.4f} {r['loso_r2']:+.4f} "
          f"{r['gap']:+.4f} {r['pm25_mean']:6.1f} {r['nn1_km']:5.0f}")

print(f"\n{'='*95}")
print("Summary:")
agg_kf = result["kfold_r2"].mean()
agg_loso = result["loso_r2"].mean()
print(f"  Mean KFold R² = {agg_kf:.4f}")
print(f"  Mean LOSO  R² = {agg_loso:.4f}")
print(f"  Mean gap      = {result['gap'].mean():.4f}")
print(f"  Median gap    = {result['gap'].median():.4f}")

print(f"\nTop 10 largest gaps (most identity leakage):")
for _, r in result.head(10).iterrows():
    nm = str(r["station_name"])[:45]
    print(f"  {nm:<45s} gap={r['gap']:+.4f}  KF={r['kfold_r2']:+.4f}  "
          f"LOSO={r['loso_r2']:+.4f}  PM2.5={r['pm25_mean']:.1f}")

print(f"\nBottom 5 smallest gaps (spatial transfer works):")
for _, r in result.tail(5).iterrows():
    nm = str(r["station_name"])[:45]
    print(f"  {nm:<45s} gap={r['gap']:+.4f}  KF={r['kfold_r2']:+.4f}  "
          f"LOSO={r['loso_r2']:+.4f}  PM2.5={r['pm25_mean']:.1f}")

print(f"\nDone — {time.time()-t0:.0f}s total")
