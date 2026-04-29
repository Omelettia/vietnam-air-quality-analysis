"""
Per-station aggregate features → predict station mean PM2.5 with simple models.

Computes ~30 station-level features from unified_thesis_v1.csv + metadata,
then runs LOO-CV with Ridge, ElasticNet, RandomForest, and best-5-subset Ridge.

Output:
  analysis/thesis_experiments/station_baseline_features.csv
  analysis/thesis_experiments/baseline_model_analysis.md
"""

import os, sys, io, warnings, time, itertools
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(REPO_DIR, "analysis", "thesis_experiments")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("STATION BASELINE FEATURES ANALYSIS")
print("=" * 70)
t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(REPO_DIR, "data/merged/unified_thesis_v1.csv"),
                 dtype={"stationId": str})
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["hour"] = df["ts"].dt.hour
df["month"] = df["ts"].dt.month
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations")

meta = pd.read_csv(os.path.join(REPO_DIR,
                    "analysis/thesis_audit/station_selection_final.csv"),
                    dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))
sid_region = dict(zip(meta["stationId"], meta["region"]))
station_ids = sorted(df["stationId"].unique())
n_stn = len(station_ids)

bld = pd.read_csv(os.path.join(REPO_DIR,
                   "data/stations/metadata/station_building_density.csv"),
                   dtype={"stationId": str})
bld_map = bld.set_index("stationId")[["building_count_3km", "building_area_3km"]]

acag = pd.read_csv(os.path.join(REPO_DIR, "data/acag/acag_station_climatology.csv"),
                    dtype={"stationId": str})
acag_map = acag.set_index("stationId")

loso = pd.read_csv(os.path.join(OUT_DIR, "loso_per_station_exp07.csv"),
                    dtype={"station_id": str})
loso_b1 = loso[loso["config"] == "B1"]
loso_r2 = dict(zip(loso_b1["station_id"], loso_b1["r2"]))

# ═══════════════════════════════════════════════════════════════════════════════
#  STATION DISTANCES + NN1
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing station distances ...")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

sid_lat = dict(zip(meta["stationId"], meta["lat"]))
sid_lon = dict(zip(meta["stationId"], meta["lon"]))
coords = {s: (sid_lat[s], sid_lon[s]) for s in station_ids}

sid_to_idx = {s: i for i, s in enumerate(station_ids)}
dist_full = np.zeros((n_stn, n_stn))
for i in range(n_stn):
    for j in range(i + 1, n_stn):
        d = haversine(*coords[station_ids[i]], *coords[station_ids[j]])
        dist_full[i, j] = d
        dist_full[j, i] = d

nn1_idx = {}
nn1_dist = {}
for i in range(n_stn):
    dists = [(j, dist_full[i, j]) for j in range(n_stn) if j != i]
    nearest = min(dists, key=lambda x: x[1])
    nn1_idx[station_ids[i]] = station_ids[nearest[0]]
    nn1_dist[station_ids[i]] = nearest[1]

# ═══════════════════════════════════════════════════════════════════════════════
#  PM2.5 WIDE MATRIX → NN1 temporal correlation + RFSI IDW
# ═══════════════════════════════════════════════════════════════════════════════
print("Building PM2.5 wide matrix ...")
pm25_wide = df.pivot_table(index="ts", columns="stationId",
                           values="PM2.5", aggfunc="first")

nn1_corr = {}
for sid in station_ids:
    nn_sid = nn1_idx[sid]
    if sid in pm25_wide.columns and nn_sid in pm25_wide.columns:
        s1 = pm25_wide[sid]
        s2 = pm25_wide[nn_sid]
        valid = s1.notna() & s2.notna()
        if valid.sum() >= 50:
            r, _ = sp_stats.pearsonr(s1[valid], s2[valid])
            nn1_corr[sid] = r
        else:
            nn1_corr[sid] = np.nan
    else:
        nn1_corr[sid] = np.nan

print("Computing RFSI IDW per row ...")
K_NN = 5
pm25_mat = pm25_wide.values
sid_cols = list(pm25_wide.columns)
sid_to_col = {s: i for i, s in enumerate(sid_cols)}
ts_to_row = pd.Series(range(len(pm25_wide)), index=pm25_wide.index)
df["ts_row"] = df["ts"].map(ts_to_row).astype(int).values

neighbor_order = {}
for i in range(n_stn):
    neighbor_order[i] = sorted(
        [(j, dist_full[i, j]) for j in range(n_stn) if j != i],
        key=lambda x: x[1])

stationId_vals = df["stationId"].values
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

with np.errstate(divide="ignore", invalid="ignore"):
    w = 1.0 / d_nn
    pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)
df["PM25_nn_idw"] = pm_idw
print("RFSI IDW done")

# ── Neighbor-based seasonal & diurnal features (LOSO-safe) ──
print("Computing neighbor seasonal amp & diurnal range ...")
WINTER = [11, 12, 1, 2]
SUMMER = [5, 6, 7, 8]
df["date"] = df["ts"].dt.date

nbr_seasonal_amp = {}
nbr_diurnal_range = {}
for sid in station_ids:
    sdf = df[df["stationId"] == sid]
    idw = sdf["PM25_nn_idw"]
    months = sdf["month"]
    w_mean = idw[months.isin(WINTER)].mean()
    s_mean = idw[months.isin(SUMMER)].mean()
    nbr_seasonal_amp[sid] = (w_mean / s_mean) if (pd.notna(s_mean) and s_mean > 0) else np.nan

    daily = sdf.groupby("date")["PM25_nn_idw"].agg(["max", "min"])
    daily["range"] = daily["max"] - daily["min"]
    nbr_diurnal_range[sid] = daily["range"].mean() if len(daily) > 0 else np.nan

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPUTE PER-STATION FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing per-station features ...")
rows = []
for sid in station_ids:
    sdf = df[df["stationId"] == sid]
    pm = sdf["PM2.5"]
    n_rows = len(sdf)

    # --- Target ---
    pm25_mean = pm.mean()

    # --- Urbanization ---
    bc3 = bld_map.loc[sid, "building_count_3km"] if sid in bld_map.index else 0
    ba3 = bld_map.loc[sid, "building_area_3km"] if sid in bld_map.index else 0
    acag_ann = acag_map.loc[sid, "ACAG_annual_mean"] if sid in acag_map.index else np.nan

    # --- Atmospheric averages ---
    aot = sdf["AOT"]
    aot_valid = aot.notna()
    mean_AOT = aot.mean()
    median_AOT = aot.median()
    AOT_valid_frac = aot_valid.sum() / n_rows if n_rows > 0 else 0

    mean_AOD_physics = sdf["AOD_physics"].mean()
    aot_pblh = sdf["AOT"] / sdf["PBLH"]
    mean_AOD_PBLH_ratio = aot_pblh.mean()
    mean_PBLH = sdf["PBLH"].mean()
    mean_WS = sdf["WS_om"].mean() if "WS_om" in sdf.columns else np.nan
    mean_VC = sdf["VC"].mean()
    mean_Temp = sdf["Temperature_final"].mean()
    mean_Humidity = sdf["Humidity_final"].mean()
    mean_Pressure = sdf["Pressure_final"].mean()

    precip = sdf["precip_mm"]
    rain_freq = (precip > 0.1).sum() / n_rows if n_rows > 0 else 0

    # --- Atmospheric variability ---
    AOT_std = aot.std()
    AOT_iqr = aot.quantile(0.75) - aot.quantile(0.25) if aot_valid.sum() > 10 else np.nan
    AOT_p95 = aot.quantile(0.95)

    hourly_means = sdf.groupby("hour")["PM2.5"].mean()
    diurnal_range = hourly_means.max() - hourly_means.min() if len(hourly_means) > 1 else 0

    monthly_means = sdf.groupby("month")["PM2.5"].mean()
    seasonal_amp = monthly_means.max() - monthly_means.min() if len(monthly_means) > 1 else 0

    # --- Spatial context (RFSI IDW) ---
    idw = sdf["PM25_nn_idw"]
    mean_idw = idw.mean()
    std_idw = idw.std()
    iqr_idw = idw.quantile(0.75) - idw.quantile(0.25) if idw.notna().sum() > 10 else np.nan
    nn1_km = nn1_dist[sid]
    nn1_r = nn1_corr.get(sid, np.nan)

    # --- Geography ---
    elev = sdf["elevation_m"].iloc[0] if "elevation_m" in sdf.columns else np.nan
    lat = sdf["latitude"].iloc[0]
    lon = sdf["longitude"].iloc[0]
    slope = sdf["slope_deg"].iloc[0] if "slope_deg" in sdf.columns else np.nan

    rows.append({
        "station_id": sid,
        "station_name": sid_name.get(sid, sid),
        "region": sid_region.get(sid, "?"),
        "n_rows": n_rows,
        "pm25_mean": round(pm25_mean, 2),
        "loso_r2": loso_r2.get(sid, np.nan),
        # Urbanization
        "building_count_3km": bc3,
        "building_area_3km": ba3,
        "ACAG_annual_mean": round(acag_ann, 2) if pd.notna(acag_ann) else np.nan,
        # Atmospheric averages
        "mean_AOT": round(mean_AOT, 4) if pd.notna(mean_AOT) else np.nan,
        "median_AOT": round(median_AOT, 4) if pd.notna(median_AOT) else np.nan,
        "AOT_valid_frac": round(AOT_valid_frac, 4),
        "mean_AOD_physics": round(mean_AOD_physics, 6) if pd.notna(mean_AOD_physics) else np.nan,
        "mean_AOD_PBLH_ratio": round(mean_AOD_PBLH_ratio, 6) if pd.notna(mean_AOD_PBLH_ratio) else np.nan,
        "mean_PBLH": round(mean_PBLH, 1),
        "mean_WS": round(mean_WS, 2) if pd.notna(mean_WS) else np.nan,
        "mean_VC": round(mean_VC, 0),
        "mean_Temp": round(mean_Temp, 2),
        "mean_Humidity": round(mean_Humidity, 2),
        "mean_Pressure": round(mean_Pressure, 1),
        "rain_freq": round(rain_freq, 4),
        # Atmospheric variability
        "AOT_std": round(AOT_std, 4) if pd.notna(AOT_std) else np.nan,
        "AOT_iqr": round(AOT_iqr, 4) if pd.notna(AOT_iqr) else np.nan,
        "AOT_p95": round(AOT_p95, 4) if pd.notna(AOT_p95) else np.nan,
        "diurnal_pm25_range": round(diurnal_range, 2),
        "seasonal_pm25_amp": round(seasonal_amp, 2),
        # Spatial context
        "mean_PM25_nn_idw": round(mean_idw, 2) if pd.notna(mean_idw) else np.nan,
        "std_PM25_nn_idw": round(std_idw, 2) if pd.notna(std_idw) else np.nan,
        "iqr_PM25_nn_idw": round(iqr_idw, 2) if pd.notna(iqr_idw) else np.nan,
        "nn1_km": round(nn1_km, 1),
        "nn1_corr": round(nn1_r, 4) if pd.notna(nn1_r) else np.nan,
        "nbr_seasonal_amp": round(nbr_seasonal_amp[sid], 4) if pd.notna(nbr_seasonal_amp[sid]) else np.nan,
        "nbr_diurnal_range": round(nbr_diurnal_range[sid], 2) if pd.notna(nbr_diurnal_range[sid]) else np.nan,
        # Geography
        "elevation_m": round(elev, 1) if pd.notna(elev) else np.nan,
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "slope_deg": round(slope, 3) if pd.notna(slope) else np.nan,
    })

feat_df = pd.DataFrame(rows)
feat_path = os.path.join(OUT_DIR, "station_baseline_features.csv")
feat_df.to_csv(feat_path, index=False, encoding="utf-8-sig")
print(f"Saved features: {feat_path}")
print(f"  {len(feat_df)} stations, {len(feat_df.columns)} columns")

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS: CORRELATIONS + MODELS
# ═══════════════════════════════════════════════════════════════════════════════
LEAKY_FEATURES = {"diurnal_pm25_range", "seasonal_pm25_amp", "nn1_corr"}

ALL_PRED_COLS = [
    "building_count_3km", "building_area_3km", "ACAG_annual_mean",
    "mean_AOT", "median_AOT", "AOT_valid_frac", "mean_AOD_physics",
    "mean_AOD_PBLH_ratio",
    "mean_PBLH", "mean_WS", "mean_VC",
    "mean_Temp", "mean_Humidity", "mean_Pressure", "rain_freq",
    "AOT_std", "AOT_iqr", "AOT_p95",
    "diurnal_pm25_range", "seasonal_pm25_amp",
    "mean_PM25_nn_idw", "std_PM25_nn_idw", "iqr_PM25_nn_idw",
    "nn1_km", "nn1_corr",
    "nbr_seasonal_amp", "nbr_diurnal_range",
    "elevation_m", "latitude", "longitude", "slope_deg",
]

LOSO_SAFE_COLS = [
    "mean_PM25_nn_idw", "std_PM25_nn_idw", "iqr_PM25_nn_idw",
    "ACAG_annual_mean", "median_AOT", "AOT_std", "AOT_p95",
    "AOT_valid_frac", "mean_AOD_PBLH_ratio",
    "mean_PBLH", "mean_WS", "mean_VC",
    "mean_Temp", "mean_Humidity", "mean_Pressure", "rain_freq",
    "building_count_3km", "building_area_3km",
    "nbr_seasonal_amp", "nbr_diurnal_range",
    "elevation_m", "latitude", "longitude", "slope_deg",
]

target = feat_df["pm25_mean"].values
loso_target = feat_df["loso_r2"].values

# Drop features with too many NaNs
def filter_usable(cols, df, min_valid=30):
    out = []
    for c in cols:
        if c not in df.columns:
            print(f"  Missing column: {c}")
            continue
        nna = df[c].notna().sum()
        if nna >= min_valid:
            out.append(c)
        else:
            print(f"  Dropping {c}: only {nna} non-NaN values")
    return out

usable = filter_usable(ALL_PRED_COLS, feat_df)
usable_safe = filter_usable(LOSO_SAFE_COLS, feat_df)
print(f"\nAll predictor features: {len(usable)}")
print(f"LOSO-safe features: {len(usable_safe)}")

# --- Correlation with PM2.5 mean and LOSO R² ---
print(f"\n{'='*90}")
print(f"  {'Feature':<27s} {'Leaky':>5s} {'Pearson→PM2.5':>14s} {'Spearman→PM2.5':>15s} "
      f"{'Pearson→R²':>12s} {'Spearman→R²':>13s}")
print(f"  {'-'*27} {'-'*5} {'-'*14} {'-'*15} {'-'*12} {'-'*13}")

corr_rows = []
for c in usable:
    vals = feat_df[c].values
    valid = ~np.isnan(vals) & ~np.isnan(target)
    if valid.sum() < 10:
        continue
    pr, _ = sp_stats.pearsonr(vals[valid], target[valid])
    sr, _ = sp_stats.spearmanr(vals[valid], target[valid])

    valid_r2 = valid & ~np.isnan(loso_target)
    if valid_r2.sum() >= 10:
        pr2, _ = sp_stats.pearsonr(vals[valid_r2], loso_target[valid_r2])
        sr2, _ = sp_stats.spearmanr(vals[valid_r2], loso_target[valid_r2])
    else:
        pr2, sr2 = np.nan, np.nan

    leaky = "LEAK" if c in LEAKY_FEATURES else ""
    print(f"  {c:<27s} {leaky:>5s} {pr:+.4f}         {sr:+.4f}          "
          f"{pr2:+.4f}       {sr2:+.4f}")
    corr_rows.append({"feature": c, "leaky": c in LEAKY_FEATURES,
                       "pearson_pm25": round(pr, 4), "spearman_pm25": round(sr, 4),
                       "pearson_r2": round(pr2, 4), "spearman_r2": round(sr2, 4)})

corr_df = pd.DataFrame(corr_rows)

# --- Prepare X matrix (impute NaN with median, then scale) ---
X_raw = feat_df[usable].values.copy()
medians = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    nans = np.isnan(X_raw[:, j])
    X_raw[nans, j] = medians[j]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
y = target.copy()
n = len(y)

def loo_cv(model_cls, X, y, **kwargs):
    preds = np.zeros(n)
    for i in range(n):
        tr = np.concatenate([np.arange(0, i), np.arange(i+1, n)])
        m = model_cls(**kwargs)
        m.fit(X[tr], y[tr])
        preds[i] = m.predict(X[i:i+1])[0]
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    return r2, mae, preds

def ridge_loo_analytical(X, y, alpha=1.0):
    """Analytical Ridge LOO via hat matrix with unpenalized intercept."""
    n, p = X.shape
    X_aug = np.column_stack([np.ones(n), X])
    penalty = alpha * np.eye(p + 1)
    penalty[0, 0] = 0.0
    XtX = X_aug.T @ X_aug + penalty
    H = X_aug @ np.linalg.solve(XtX, X_aug.T)
    y_hat = H @ y
    diag_H = np.diag(H)
    loo_preds = (y_hat - diag_H * y) / (1 - diag_H)
    return r2_score(y, loo_preds), mean_absolute_error(y, loo_preds), loo_preds

# --- Ridge LOO-CV ---
print(f"\n{'='*70}")
print("Ridge LOO-CV (alpha=1.0) ...")
ridge_r2, ridge_mae, ridge_preds = loo_cv(Ridge, X_scaled, y, alpha=1.0)
print(f"  R² = {ridge_r2:.4f}, MAE = {ridge_mae:.2f}")

# --- ElasticNet LOO-CV ---
print("ElasticNet LOO-CV (alpha=0.1, l1_ratio=0.5) ...")
en_r2, en_mae, en_preds = loo_cv(ElasticNet, X_scaled, y,
                                   alpha=0.1, l1_ratio=0.5, max_iter=5000)
print(f"  R² = {en_r2:.4f}, MAE = {en_mae:.2f}")

# --- Random Forest LOO-CV ---
print("Random Forest LOO-CV (100 trees) ...")
rf_r2, rf_mae, rf_preds = loo_cv(RandomForestRegressor, X_raw, y,
                                   n_estimators=100, max_depth=5,
                                   random_state=42, n_jobs=-1)
print(f"  R² = {rf_r2:.4f}, MAE = {rf_mae:.2f}")

# --- Verify analytical formula matches iterative ---
a_r2, a_mae, _ = ridge_loo_analytical(X_scaled, y, alpha=1.0)
print(f"  Analytical Ridge LOO check: R²={a_r2:.4f} (iterative={ridge_r2:.4f})")

# --- Best-5-subset Ridge LOO-CV (analytical hat-matrix formula) ---
print(f"\nSearching best 5-feature subset for Ridge LOO-CV (analytical) ...")
best_r2 = -999
best_combo = None
best_preds = None
combos = list(itertools.combinations(range(len(usable)), 5))
print(f"  Testing {len(combos):,} combinations ...")
t1 = time.time()

for combo in combos:
    cols = list(combo)
    Xsub = X_scaled[:, cols]
    r2, mae, preds = ridge_loo_analytical(Xsub, y, alpha=1.0)
    if r2 > best_r2:
        best_r2 = r2
        best_combo = combo
        best_preds = preds.copy()

best_names = [usable[i] for i in best_combo]
best_mae = mean_absolute_error(y, best_preds)
print(f"  Best 5 features: {best_names}")
print(f"  R² = {best_r2:.4f}, MAE = {best_mae:.2f}")
print(f"  Search took {time.time()-t1:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOSO-SAFE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("LOSO-SAFE FEATURES ONLY")
print(f"{'='*70}")

X_safe_raw = feat_df[usable_safe].values.copy()
med_safe = np.nanmedian(X_safe_raw, axis=0)
for j in range(X_safe_raw.shape[1]):
    nans = np.isnan(X_safe_raw[:, j])
    X_safe_raw[nans, j] = med_safe[j]

scaler_safe = StandardScaler()
X_safe = scaler_safe.fit_transform(X_safe_raw)

print(f"Ridge LOO-CV ({len(usable_safe)} LOSO-safe features) ...")
safe_ridge_r2, safe_ridge_mae, _ = loo_cv(Ridge, X_safe, y, alpha=1.0)
print(f"  R² = {safe_ridge_r2:.4f}, MAE = {safe_ridge_mae:.2f}")

print(f"ElasticNet LOO-CV ({len(usable_safe)} LOSO-safe features) ...")
safe_en_r2, safe_en_mae, _ = loo_cv(ElasticNet, X_safe, y,
                                      alpha=0.1, l1_ratio=0.5, max_iter=5000)
print(f"  R² = {safe_en_r2:.4f}, MAE = {safe_en_mae:.2f}")

print(f"Random Forest LOO-CV ({len(usable_safe)} LOSO-safe features) ...")
safe_rf_r2, safe_rf_mae, _ = loo_cv(RandomForestRegressor, X_safe_raw, y,
                                      n_estimators=100, max_depth=5,
                                      random_state=42, n_jobs=-1)
print(f"  R² = {safe_rf_r2:.4f}, MAE = {safe_rf_mae:.2f}")

# Best 5-subset Ridge (LOSO-safe only)
print(f"\nSearching best 5-feature subset (LOSO-safe) ...")
safe_best_r2 = -999
safe_best_combo = None
safe_best_preds = None
safe_combos = list(itertools.combinations(range(len(usable_safe)), 5))
print(f"  Testing {len(safe_combos):,} combinations ...")
t1 = time.time()
for combo in safe_combos:
    cols = list(combo)
    Xsub = X_safe[:, cols]
    r2, mae, preds = ridge_loo_analytical(Xsub, y, alpha=1.0)
    if r2 > safe_best_r2:
        safe_best_r2 = r2
        safe_best_combo = combo
        safe_best_preds = preds.copy()

safe_best_names = [usable_safe[i] for i in safe_best_combo]
safe_best_mae = mean_absolute_error(y, safe_best_preds)
print(f"  Best 5 features: {safe_best_names}")
print(f"  R² = {safe_best_r2:.4f}, MAE = {safe_best_mae:.2f}")
print(f"  Search took {time.time()-t1:.0f}s")

# --- Ridge LOO for LOSO R² as target (LOSO-safe features) ---
print(f"\nRidge LOO-CV predicting LOSO R² (LOSO-safe) ...")
y_r2 = loso_target.copy()
valid_r2 = ~np.isnan(y_r2)
if valid_r2.sum() >= 30:
    Xr2 = X_safe[valid_r2]
    yr2 = y_r2[valid_r2]
    nr2 = len(yr2)
    preds_r2 = np.zeros(nr2)
    for i in range(nr2):
        tr = np.concatenate([np.arange(0, i), np.arange(i+1, nr2)])
        m = Ridge(alpha=1.0)
        m.fit(Xr2[tr], yr2[tr])
        preds_r2[i] = m.predict(Xr2[i:i+1])[0]
    r2_r2 = r2_score(yr2, preds_r2)
    mae_r2 = mean_absolute_error(yr2, preds_r2)
    print(f"  R² = {r2_r2:.4f}, MAE = {mae_r2:.4f}")
else:
    r2_r2, mae_r2 = np.nan, np.nan
    print("  Not enough valid LOSO R² values")

# ═══════════════════════════════════════════════════════════════════════════════
#  WRITE MARKDOWN REPORT
# ═══════════════════════════════════════════════════════════════════════════════
md_path = os.path.join(OUT_DIR, "baseline_model_analysis.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Station Baseline Features — Model Analysis\n\n")
    f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"**Stations**: {n_stn}\n")
    f.write(f"**All predictor features**: {len(usable)}\n")
    f.write(f"**LOSO-safe features**: {len(usable_safe)}\n\n")

    f.write("## Feature Correlations\n\n")
    f.write("| Feature | Leaky? | Pearson→PM2.5 | Spearman→PM2.5 | Pearson→R² | Spearman→R² |\n")
    f.write("|---------|:---:|:---:|:---:|:---:|:---:|\n")
    for _, cr in corr_df.sort_values("spearman_pm25", ascending=False,
                                      key=abs).iterrows():
        leak = "LEAK" if cr["leaky"] else ""
        f.write(f"| {cr['feature']} | {leak} | {cr['pearson_pm25']:+.4f} | "
                f"{cr['spearman_pm25']:+.4f} | {cr['pearson_r2']:+.4f} | "
                f"{cr['spearman_r2']:+.4f} |\n")

    f.write("\n## LOO-CV Results — All Features (target = station mean PM2.5)\n\n")
    f.write("| Model | R² | MAE (µg/m³) |\n")
    f.write("|-------|:---:|:---:|\n")
    f.write(f"| Ridge (all {len(usable)} features) | {ridge_r2:.4f} | {ridge_mae:.2f} |\n")
    f.write(f"| ElasticNet (all {len(usable)} features) | {en_r2:.4f} | {en_mae:.2f} |\n")
    f.write(f"| Random Forest (all {len(usable)} features) | {rf_r2:.4f} | {rf_mae:.2f} |\n")
    f.write(f"| Ridge (best 5) | {best_r2:.4f} | {best_mae:.2f} |\n")
    f.write(f"\n**Best 5-feature subset**: {', '.join(best_names)}\n\n")

    f.write("## LOO-CV Results — LOSO-Safe Features Only\n\n")
    f.write("These features do NOT use the station's own PM2.5, so they are "
            "valid as a Stage 1 spatial baseline in LOSO evaluation.\n\n")
    f.write("| Model | R² | MAE (µg/m³) |\n")
    f.write("|-------|:---:|:---:|\n")
    f.write(f"| Ridge ({len(usable_safe)} features) | {safe_ridge_r2:.4f} | {safe_ridge_mae:.2f} |\n")
    f.write(f"| ElasticNet ({len(usable_safe)} features) | {safe_en_r2:.4f} | {safe_en_mae:.2f} |\n")
    f.write(f"| Random Forest ({len(usable_safe)} features) | {safe_rf_r2:.4f} | {safe_rf_mae:.2f} |\n")
    f.write(f"| Ridge (best 5 LOSO-safe) | {safe_best_r2:.4f} | {safe_best_mae:.2f} |\n")
    f.write(f"\n**Best 5 LOSO-safe features**: {', '.join(safe_best_names)}\n\n")

    f.write("## LOO-CV predicting LOSO R² (LOSO-safe features)\n\n")
    f.write("| Model | R² | MAE |\n")
    f.write("|-------|:---:|:---:|\n")
    if pd.notna(r2_r2):
        f.write(f"| Ridge ({len(usable_safe)} features) | {r2_r2:.4f} | {mae_r2:.4f} |\n")
    else:
        f.write("| Ridge | N/A | N/A |\n")

    f.write("\n## Interpretation\n\n")
    f.write("Features flagged **LEAK** use the station's own PM2.5 data "
            "(diurnal_pm25_range, seasonal_pm25_amp, nn1_corr). "
            "These inflate the all-features LOO R² but are unavailable "
            "in LOSO evaluation.\n\n")
    f.write("The LOSO-safe LOO R² represents the real ceiling for a "
            "Stage 1 spatial baseline that predicts station mean PM2.5 "
            "without using the target station's own observations.\n")

print(f"\nSaved report: {md_path}")
print(f"\nDone — {time.time()-t0:.0f}s total")
