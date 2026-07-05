"""
Experiment 02: Anomaly-Based XGBoost + Feature Ablation

Configs:
  D — Config C minus geographic identity features (diagnose leakage)
  E — Oracle anomaly: station's own climatology (upper bound)
  F — Cross-validated anomaly: K-nearest neighbor climatology (K=3, K=5)
  G — Anomaly + terrain features (no lat/lon)

Output:
  analysis/thesis_experiments/experiment_02_anomaly.md
  analysis/thesis_experiments/loso_per_station_config_f.csv
  analysis/thesis_experiments/feature_importance_config_f.csv
  analysis/thesis_experiments/climatology_comparison.csv
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

BASE = args.data_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
os.makedirs("analysis/thesis_experiments", exist_ok=True)

XGB_PARAMS = dict(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    random_state=42, n_jobs=-1,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA & COMPUTE ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("EXPERIMENT 02: ANOMALY-BASED XGBOOST")
print("=" * 80)

t0 = time.time()
df = pd.read_csv("data/merged/unified_thesis_v1.csv", dtype={"stationId": str})
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["month"] = df["ts"].dt.month
df["hour"] = df["ts"].dt.hour
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations ({time.time()-t0:.1f}s)")

meta = pd.read_csv("analysis/thesis_audit/station_selection_final.csv",
                    dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))
sid_region = dict(zip(meta["stationId"], meta["region"]))
sid_lat = dict(zip(meta["stationId"], meta["lat"]))
sid_lon = dict(zip(meta["stationId"], meta["lon"]))
station_ids = sorted(df["stationId"].unique())

# Climatology: station × month × hour means
print("Computing climatologies...")
clim_df = df.groupby(["stationId", "month", "hour"], as_index=False).agg({
    "PM2.5": "mean", "Temperature_final": "mean", "Humidity_final": "mean",
    "Pressure_final": "mean", "WS_om": "mean", "PBLH": "mean",
})
clim_df.rename(columns={"PM2.5": "PM25_clim", "Temperature_final": "_Tc",
    "Humidity_final": "_Hc", "Pressure_final": "_Pc",
    "WS_om": "_WSc", "PBLH": "_PBLHc"}, inplace=True)

df = df.merge(clim_df, on=["stationId", "month", "hour"], how="left")
df["PM25_anom"] = df["PM2.5"] - df["PM25_clim"]
df["Temperature_anom"] = df["Temperature_final"] - df["_Tc"]
df["Humidity_anom"] = df["Humidity_final"] - df["_Hc"]
df["Pressure_anom"] = df["Pressure_final"] - df["_Pc"]
df["WS_anom"] = df["WS_om"] - df["_WSc"]
df["PBLH_anom"] = df["PBLH"] - df["_PBLHc"]
df.drop(columns=["_Tc", "_Hc", "_Pc", "_WSc", "_PBLHc"], inplace=True)

pm25_clim_lookup = {}
for sid in station_ids:
    sub = clim_df[clim_df["stationId"] == sid]
    pm25_clim_lookup[sid] = dict(zip(zip(sub["month"], sub["hour"]), sub["PM25_clim"]))

print(f"PM25_anomaly: mean={df['PM25_anom'].mean():.2f}, std={df['PM25_anom'].std():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  NEAREST NEIGHBORS
# ═══════════════════════════════════════════════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

coords = {sid: (sid_lat[sid], sid_lon[sid]) for sid in station_ids}
nn_table = {}
for sid in station_ids:
    la, lo = coords[sid]
    dists = sorted([(o, haversine(la, lo, *coords[o]))
                    for o in station_ids if o != sid], key=lambda x: x[1])
    nn_table[sid] = dists

nn1 = [nn_table[s][0][1] for s in station_ids]
print(f"NN1 distances: min={min(nn1):.0f}km, median={np.median(nn1):.0f}km, max={max(nn1):.0f}km")

far_stations = [(sid_name.get(s, s)[:30], nn_table[s][0][1])
                for s in station_ids if nn_table[s][0][1] > 100]
if far_stations:
    print(f"Stations >100km from nearest neighbor:")
    for nm, d in far_stations:
        print(f"  {nm:30s} {d:.0f}km")

def get_neighbor_clim(sid, K):
    neighbors = nn_table[sid][:K]
    all_keys = set()
    for s, _ in neighbors:
        all_keys.update(pm25_clim_lookup.get(s, {}).keys())
    result = {}
    for key in all_keys:
        vals = [pm25_clim_lookup[s][key] for s, _ in neighbors
                if key in pm25_clim_lookup.get(s, {})]
        if vals:
            result[key] = np.mean(vals)
    return result

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════════
FEATURES_C = [
    "Temperature_final", "Humidity_final", "Pressure_final", "PBLH", "VC", "RH_factor",
    "wind_u", "wind_v", "wind_dir_sin", "wind_dir_cos",
    "WS_local", "wind_u_local", "wind_v_local", "wind_dir_sin_local", "wind_dir_cos_local",
    "dT_6h", "dRH_6h", "dWS_6h", "dP_6h",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
    "precip_mm", "hrs_since_rain", "rain_sum_24h", "rain_sum_48h",
    "rain_days_7d", "consecutive_dry_days",
    "latitude", "longitude", "elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
    "elev_x_PBLH", "elev_x_hour_sin",
    "AOT", "AOT_mean", "AOT_inner_mean", "AOT_outer_mean",
    "RF", "SSA", "Uncertainty", "AE", "AOT_valid_count",
    "AOD_physics", "AOT_spatial_std", "AOT_local_vs_regional",
    "AOT_ffill_48h", "hours_since_valid_AOT",
    "AOT_lag_1h", "AOT_lag_3h", "AOT_lag_6h",
    "AOT_rolling_mean_6h", "AOT_rolling_mean_24h",
    "AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_grad_dir",
]

GEO = {"latitude", "longitude", "elevation_m", "slope_deg",
       "aspect_sin", "aspect_cos", "elev_x_PBLH", "elev_x_hour_sin"}

FEATURES_D = [f for f in FEATURES_C if f not in GEO]

FEATURES_EF = [
    "Temperature_anom", "Humidity_anom", "Pressure_anom", "WS_anom", "PBLH_anom",
    "VC", "RH_factor",
    "wind_u", "wind_v", "wind_dir_sin", "wind_dir_cos",
    "WS_local", "wind_u_local", "wind_v_local", "wind_dir_sin_local", "wind_dir_cos_local",
    "dT_6h", "dRH_6h", "dWS_6h", "dP_6h",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
    "precip_mm", "hrs_since_rain", "rain_sum_24h", "rain_sum_48h",
    "rain_days_7d", "consecutive_dry_days",
    "AOT", "AOT_mean", "AOT_inner_mean", "AOT_outer_mean",
    "RF", "SSA", "Uncertainty", "AE", "AOT_valid_count",
    "AOD_physics", "AOT_spatial_std", "AOT_local_vs_regional",
    "AOT_ffill_48h", "hours_since_valid_AOT",
    "AOT_lag_1h", "AOT_lag_3h", "AOT_lag_6h",
    "AOT_rolling_mean_6h", "AOT_rolling_mean_24h",
    "AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_grad_dir",
]

FEATURES_G = FEATURES_EF + ["slope_deg", "aspect_sin", "aspect_cos",
                             "elev_x_PBLH", "elev_x_hour_sin"]

for name, feats in [("D", FEATURES_D), ("E/F", FEATURES_EF), ("G", FEATURES_G)]:
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"WARNING: {name} missing: {missing}")
        if name == "D":
            FEATURES_D = [f for f in FEATURES_D if f in df.columns]
        elif name == "E/F":
            FEATURES_EF = [f for f in FEATURES_EF if f in df.columns]
            FEATURES_G = FEATURES_EF + [f for f in
                ["slope_deg", "aspect_sin", "aspect_cos", "elev_x_PBLH", "elev_x_hour_sin"]
                if f in df.columns]

print(f"Features: D={len(FEATURES_D)}, E/F={len(FEATURES_EF)}, G={len(FEATURES_G)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  CV FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
y_abs_all = df["PM2.5"].values
y_anom_all = df["PM25_anom"].values
clim_own_all = df["PM25_clim"].values


def run_kfold(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    out = []
    for _, (tr, va) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[va])
        out.append(dict(r2=r2_score(y[va], p),
                        rmse=np.sqrt(mean_squared_error(y[va], p)),
                        mae=mean_absolute_error(y[va], p)))
    return out


def run_kfold_anomaly(X, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    out = []
    for _, (tr, va) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[tr], y_anom_all[tr])
        ap = m.predict(X.iloc[va])
        absp = ap + clim_own_all[va]
        out.append(dict(
            r2_anom=r2_score(y_anom_all[va], ap),
            r2_abs=r2_score(y_abs_all[va], absp),
            rmse=np.sqrt(mean_squared_error(y_abs_all[va], absp)),
            mae=mean_absolute_error(y_abs_all[va], absp),
        ))
    return out


def run_loso_standard(features, label=""):
    results = []
    t_start = time.time()
    for i, sid in enumerate(station_ids):
        mask = df["stationId"] == sid
        n = mask.sum()
        nm = sid_name.get(sid, sid)[:45]
        rg = sid_region.get(sid, "?")
        if n < 10:
            continue
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(df.loc[~mask, features], df.loc[~mask, "PM2.5"].values)
        p = m.predict(df.loc[mask, features])
        yt = df.loc[mask, "PM2.5"].values
        r2 = r2_score(yt, p)
        rmse = np.sqrt(mean_squared_error(yt, p))
        mae = mean_absolute_error(yt, p)
        print(f"  [{i+1:2d}/{len(station_ids)}] {nm:45s} | "
              f"R²={r2:+.3f}  RMSE={rmse:5.1f}  n={n:5d}  [{rg}]")
        results.append(dict(station_id=sid, station_name=sid_name.get(sid, sid),
                            region=rg, n_rows=n,
                            r2=round(r2, 4), rmse=round(rmse, 2), mae=round(mae, 2)))
    print(f"  Time: {time.time()-t_start:.0f}s")
    return results


def run_loso_anomaly(features, mode="oracle", K=3, label=""):
    results = []
    t_start = time.time()
    for i, sid in enumerate(station_ids):
        mask = df["stationId"] == sid
        n = mask.sum()
        nm = sid_name.get(sid, sid)[:45]
        rg = sid_region.get(sid, "?")
        if n < 10:
            continue

        X_tr = df.loc[~mask, features]
        y_tr = df.loc[~mask, "PM25_anom"].values

        test = df[mask].reset_index(drop=True)
        y_test_abs = test["PM2.5"].values

        if mode == "oracle":
            tc = test["PM25_clim"].values
        else:
            nc = get_neighbor_clim(sid, K)
            tc = np.array([nc.get((mo, hr), np.nan)
                           for mo, hr in zip(test["month"], test["hour"])])

        ta = y_test_abs - tc
        ok = ~np.isnan(ta) & ~np.isnan(tc)
        nv = int(ok.sum())
        if nv < 10:
            print(f"  [{i+1:2d}/{len(station_ids)}] {nm:45s} | SKIP (n={nv})")
            continue

        X_te = test.loc[ok, features]

        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr, y_tr)
        ap = m.predict(X_te)

        r2a = r2_score(ta[ok], ap)
        absp = ap + tc[ok]
        r2 = r2_score(y_test_abs[ok], absp)
        rmse = np.sqrt(mean_squared_error(y_test_abs[ok], absp))
        mae = mean_absolute_error(y_test_abs[ok], absp)
        nd = nn_table[sid][0][1]

        print(f"  [{i+1:2d}/{len(station_ids)}] {nm:45s} | "
              f"R²={r2:+.3f}  R²_a={r2a:+.3f}  RMSE={rmse:5.1f}  n={nv:5d}  [{rg}]")

        results.append(dict(
            station_id=sid, station_name=sid_name.get(sid, sid),
            region=rg, n_rows=nv,
            r2_abs=round(r2, 4), r2_anom=round(r2a, 4),
            rmse_abs=round(rmse, 2), mae_abs=round(mae, 2),
            nn_dist_km=round(nd, 1),
        ))
    print(f"  Time: {time.time()-t_start:.0f}s")
    return results


def summarize(results, r2_col="r2", rmse_col="rmse"):
    rdf = pd.DataFrame(results)
    v = rdf.dropna(subset=[r2_col])
    by_region = {}
    for rg in ["North", "Central", "South", "Unknown"]:
        sub = v[v["region"] == rg]
        if len(sub):
            by_region[rg] = dict(n=len(sub), mean_r2=round(sub[r2_col].mean(), 4),
                                 median_r2=round(sub[r2_col].median(), 4),
                                 mean_rmse=round(sub[rmse_col].mean(), 1))
    return dict(
        mean_r2=round(v[r2_col].mean(), 4),
        median_r2=round(v[r2_col].median(), 4),
        wmean_r2=round((v[r2_col] * v["n_rows"]).sum() / v["n_rows"].sum(), 4),
        mean_rmse=round(v[rmse_col].mean(), 2),
        neg_count=int((v[r2_col] < 0).sum()),
        by_region=by_region, df=rdf,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════
ALL = {}

# ── Config D ──
print(f"\n{'='*80}")
print(f"CONFIG D: DROP GEOGRAPHY ({len(FEATURES_D)} features)")
print(f"{'='*80}")
t1 = time.time()
kf_d = run_kfold(df[FEATURES_D], y_abs_all)
kf_d_r2 = np.mean([m["r2"] for m in kf_d])
kf_d_rmse = np.mean([m["rmse"] for m in kf_d])
kf_d_mae = np.mean([m["mae"] for m in kf_d])
print(f"  KFold: R²={kf_d_r2:.4f}  RMSE={kf_d_rmse:.2f}  MAE={kf_d_mae:.2f} ({time.time()-t1:.0f}s)")

print(f"\n  --- LOSO ---")
loso_d = run_loso_standard(FEATURES_D, "D")
sum_d = summarize(loso_d)
ALL["D"] = dict(nf=len(FEATURES_D), kf_r2=round(kf_d_r2, 4), kf_rmse=round(kf_d_rmse, 2),
                kf_mae=round(kf_d_mae, 2), loso=sum_d)
print(f"  LOSO mean R²={sum_d['mean_r2']:.4f}  median={sum_d['median_r2']:.4f}  "
      f"neg={sum_d['neg_count']}")

# ── Config E/F KFold (shared — same features, same target for KFold) ──
print(f"\n{'='*80}")
print(f"CONFIG E/F KFOLD ({len(FEATURES_EF)} anomaly features)")
print(f"{'='*80}")
t1 = time.time()
kf_ef = run_kfold_anomaly(df[FEATURES_EF])
kf_ef_r2a = np.mean([m["r2_anom"] for m in kf_ef])
kf_ef_r2 = np.mean([m["r2_abs"] for m in kf_ef])
kf_ef_rmse = np.mean([m["rmse"] for m in kf_ef])
kf_ef_mae = np.mean([m["mae"] for m in kf_ef])
print(f"  KFold: R²_abs={kf_ef_r2:.4f}  R²_anom={kf_ef_r2a:.4f}  "
      f"RMSE={kf_ef_rmse:.2f} ({time.time()-t1:.0f}s)")

# ── Config E LOSO (oracle) ──
print(f"\n{'='*80}")
print(f"CONFIG E: ORACLE ANOMALY LOSO ({len(FEATURES_EF)} features)")
print(f"{'='*80}")
loso_e = run_loso_anomaly(FEATURES_EF, mode="oracle", label="E")
sum_e = summarize(loso_e, "r2_abs", "rmse_abs")
ALL["E"] = dict(nf=len(FEATURES_EF), kf_r2=round(kf_ef_r2, 4), kf_rmse=round(kf_ef_rmse, 2),
                kf_mae=round(kf_ef_mae, 2), kf_r2a=round(kf_ef_r2a, 4), loso=sum_e)
print(f"  LOSO mean R²={sum_e['mean_r2']:.4f}  median={sum_e['median_r2']:.4f}  "
      f"neg={sum_e['neg_count']}")

# ── Config F K=3 LOSO ──
print(f"\n{'='*80}")
print(f"CONFIG F (K=3): NEIGHBOR ANOMALY LOSO ({len(FEATURES_EF)} features)")
print(f"{'='*80}")
loso_f3 = run_loso_anomaly(FEATURES_EF, mode="neighbor", K=3, label="F3")
sum_f3 = summarize(loso_f3, "r2_abs", "rmse_abs")
ALL["F3"] = dict(nf=len(FEATURES_EF), kf_r2=round(kf_ef_r2, 4), kf_rmse=round(kf_ef_rmse, 2),
                 kf_mae=round(kf_ef_mae, 2), loso=sum_f3)
print(f"  LOSO mean R²={sum_f3['mean_r2']:.4f}  median={sum_f3['median_r2']:.4f}  "
      f"neg={sum_f3['neg_count']}")

# ── Config F K=5 LOSO ──
print(f"\n{'='*80}")
print(f"CONFIG F (K=5): NEIGHBOR ANOMALY LOSO ({len(FEATURES_EF)} features)")
print(f"{'='*80}")
loso_f5 = run_loso_anomaly(FEATURES_EF, mode="neighbor", K=5, label="F5")
sum_f5 = summarize(loso_f5, "r2_abs", "rmse_abs")
ALL["F5"] = dict(nf=len(FEATURES_EF), kf_r2=round(kf_ef_r2, 4), kf_rmse=round(kf_ef_rmse, 2),
                 kf_mae=round(kf_ef_mae, 2), loso=sum_f5)
print(f"  LOSO mean R²={sum_f5['mean_r2']:.4f}  median={sum_f5['median_r2']:.4f}  "
      f"neg={sum_f5['neg_count']}")

# ── Config G KFold + LOSO ──
print(f"\n{'='*80}")
print(f"CONFIG G: ANOMALY + TERRAIN ({len(FEATURES_G)} features)")
print(f"{'='*80}")
t1 = time.time()
kf_g = run_kfold_anomaly(df[FEATURES_G])
kf_g_r2 = np.mean([m["r2_abs"] for m in kf_g])
kf_g_rmse = np.mean([m["rmse"] for m in kf_g])
kf_g_mae = np.mean([m["mae"] for m in kf_g])
kf_g_r2a = np.mean([m["r2_anom"] for m in kf_g])
print(f"  KFold: R²_abs={kf_g_r2:.4f}  R²_anom={kf_g_r2a:.4f}  "
      f"RMSE={kf_g_rmse:.2f} ({time.time()-t1:.0f}s)")

print(f"\n  --- LOSO (K=3 neighbors) ---")
loso_g = run_loso_anomaly(FEATURES_G, mode="neighbor", K=3, label="G")
sum_g = summarize(loso_g, "r2_abs", "rmse_abs")
ALL["G"] = dict(nf=len(FEATURES_G), kf_r2=round(kf_g_r2, 4), kf_rmse=round(kf_g_rmse, 2),
                kf_mae=round(kf_g_mae, 2), kf_r2a=round(kf_g_r2a, 4), loso=sum_g)
print(f"  LOSO mean R²={sum_g['mean_r2']:.4f}  median={sum_g['median_r2']:.4f}  "
      f"neg={sum_g['neg_count']}")

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE (Config F — full training on anomaly target)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("FEATURE IMPORTANCE (Config F — full training)")
print(f"{'='*80}")

model_f = xgb.XGBRegressor(**XGB_PARAMS)
model_f.fit(df[FEATURES_EF], y_anom_all)

importance = model_f.get_booster().get_score(importance_type="gain")
imp_df = pd.DataFrame([{"feature": k, "gain": v} for k, v in importance.items()]
                       ).sort_values("gain", ascending=False).reset_index(drop=True)
feat_map = {f"f{i}": name for i, name in enumerate(FEATURES_EF)}
imp_df["feature"] = imp_df["feature"].map(lambda x: feat_map.get(x, x))
imp_df["rank"] = range(1, len(imp_df) + 1)
imp_df.to_csv("analysis/thesis_experiments/feature_importance_config_f.csv",
              index=False, encoding="utf-8-sig")

print(f"\nTop 20 features by gain:")
for _, r in imp_df.head(20).iterrows():
    print(f"  {r['rank']:2d}. {r['feature']:30s} gain={r['gain']:.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  CLIMATOLOGY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("CLIMATOLOGY COMPARISON")
print(f"{'='*80}")

clim_comp = []
for sid in station_ids:
    own = pm25_clim_lookup.get(sid, {})
    own_mean = np.mean(list(own.values())) if own else np.nan
    for K in [3, 5]:
        nc = get_neighbor_clim(sid, K)
        nc_mean = np.mean(list(nc.values())) if nc else np.nan
        if K == 3:
            nc3_mean = nc_mean
        else:
            nc5_mean = nc_mean

    clim_comp.append(dict(
        station_id=sid, station_name=sid_name.get(sid, sid),
        region=sid_region.get(sid, "?"),
        own_clim_mean=round(own_mean, 2),
        neighbor_clim_mean_K3=round(nc3_mean, 2),
        neighbor_clim_mean_K5=round(nc5_mean, 2),
        diff_K3=round(own_mean - nc3_mean, 2),
        diff_K5=round(own_mean - nc5_mean, 2),
        nn1_dist_km=round(nn_table[sid][0][1], 1),
    ))

clim_comp_df = pd.DataFrame(clim_comp)
clim_comp_df.to_csv("analysis/thesis_experiments/climatology_comparison.csv",
                     index=False, encoding="utf-8-sig")

print(f"\nClimatology mismatch (own - neighbor K=3):")
print(f"  Mean abs diff: {clim_comp_df['diff_K3'].abs().mean():.2f} µg/m³")
print(f"  Max abs diff:  {clim_comp_df['diff_K3'].abs().max():.2f} µg/m³")
worst = clim_comp_df.loc[clim_comp_df["diff_K3"].abs().idxmax()]
print(f"  Worst station: {worst['station_name'][:40]} (diff={worst['diff_K3']:.2f})")

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE PER-STATION RESULTS (Config F K=3)
# ═══════════════════════════════════════════════════════════════════════════════
loso_f_df = pd.DataFrame(loso_f3).sort_values("r2_abs", ascending=True)
loso_f_df.to_csv("analysis/thesis_experiments/loso_per_station_config_f.csv",
                 index=False, encoding="utf-8-sig")

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("GENERATING REPORT")
print(f"{'='*80}")

# Load Config C results for comparison
loso_c_df = pd.read_csv("analysis/thesis_experiments/loso_per_station_config_c.csv",
                         dtype={"station_id": str})

rpt = []
rpt.append("# Experiment 02: Anomaly-Based XGBoost + Feature Ablation\n")
rpt.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rpt.append(f"**Dataset:** {len(df):,} rows, {len(station_ids)} stations")
rpt.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, lr=0.05\n")

# Comparison table
rpt.append("## Comparison Table (all R² on absolute PM2.5)\n")
rpt.append("| Config | Description | Features | KFold R² | LOSO R² (mean) | "
           "LOSO R² (median) | LOSO RMSE | Neg R² | Gap |")
rpt.append("|--------|-------------|----------|----------|----------------|"
           "------------------|-----------|--------|-----|")

rows = [
    ("C (Exp01)", "Full baseline", 62, 0.7262, -0.4953, -0.0004, 18.13, 20, 1.2215),
    ("D", "Drop geography", ALL["D"]["nf"], ALL["D"]["kf_r2"],
     ALL["D"]["loso"]["mean_r2"], ALL["D"]["loso"]["median_r2"],
     ALL["D"]["loso"]["mean_rmse"], ALL["D"]["loso"]["neg_count"],
     round(ALL["D"]["kf_r2"] - ALL["D"]["loso"]["mean_r2"], 4)),
    ("E", "Oracle anomaly", ALL["E"]["nf"], ALL["E"]["kf_r2"],
     ALL["E"]["loso"]["mean_r2"], ALL["E"]["loso"]["median_r2"],
     ALL["E"]["loso"]["mean_rmse"], ALL["E"]["loso"]["neg_count"],
     round(ALL["E"]["kf_r2"] - ALL["E"]["loso"]["mean_r2"], 4)),
    ("F (K=3)", "Neighbor anomaly K=3", ALL["F3"]["nf"], ALL["F3"]["kf_r2"],
     ALL["F3"]["loso"]["mean_r2"], ALL["F3"]["loso"]["median_r2"],
     ALL["F3"]["loso"]["mean_rmse"], ALL["F3"]["loso"]["neg_count"],
     round(ALL["F3"]["kf_r2"] - ALL["F3"]["loso"]["mean_r2"], 4)),
    ("F (K=5)", "Neighbor anomaly K=5", ALL["F5"]["nf"], ALL["F5"]["kf_r2"],
     ALL["F5"]["loso"]["mean_r2"], ALL["F5"]["loso"]["median_r2"],
     ALL["F5"]["loso"]["mean_rmse"], ALL["F5"]["loso"]["neg_count"],
     round(ALL["F5"]["kf_r2"] - ALL["F5"]["loso"]["mean_r2"], 4)),
    ("G", "Anomaly + terrain", ALL["G"]["nf"], ALL["G"]["kf_r2"],
     ALL["G"]["loso"]["mean_r2"], ALL["G"]["loso"]["median_r2"],
     ALL["G"]["loso"]["mean_rmse"], ALL["G"]["loso"]["neg_count"],
     round(ALL["G"]["kf_r2"] - ALL["G"]["loso"]["mean_r2"], 4)),
]
for cfg, desc, nf, kf, lm, lmed, lrmse, neg, gap in rows:
    rpt.append(f"| {cfg} | {desc} | {nf} | {kf:.4f} | {lm:.4f} | "
               f"{lmed:.4f} | {lrmse:.2f} | {neg} | {gap:.4f} |")

# Per-station Config F vs Config C
rpt.append("\n## Per-Station LOSO: Config F (K=3) vs Config C\n")
rpt.append("| Station | Region | C R² | F R² | Δ R² | F RMSE |")
rpt.append("|---------|--------|------|------|------|--------|")

merged = pd.DataFrame(loso_f3).merge(
    loso_c_df[["station_id", "r2"]].rename(columns={"r2": "r2_C"}),
    on="station_id", how="left"
).sort_values("r2_abs", ascending=True)

for _, r in merged.iterrows():
    nm = str(r["station_name"])[:50]
    c_r2 = r.get("r2_C", np.nan)
    f_r2 = r["r2_abs"]
    delta = f_r2 - c_r2 if pd.notna(c_r2) else np.nan
    flag = " ✓" if pd.notna(delta) and delta > 0 else ""
    rpt.append(f"| {nm} | {r['region']} | "
               f"{c_r2:.4f} | {f_r2:.4f} | "
               f"{delta:+.4f}{flag} | {r['rmse_abs']:.1f} |"
               if pd.notna(c_r2) else
               f"| {nm} | {r['region']} | — | {f_r2:.4f} | — | {r['rmse_abs']:.1f} |")

# Regional comparison
rpt.append("\n## Regional Comparison\n")
rpt.append("| Region | C LOSO R² | D LOSO R² | E LOSO R² | F(K=3) LOSO R² | G LOSO R² |")
rpt.append("|--------|-----------|-----------|-----------|----------------|-----------|")

c_by_region = {"North": 0.0458, "Central": -0.0211, "South": -2.1908}
for rg in ["North", "Central", "South"]:
    c_val = c_by_region.get(rg, "—")
    d_val = ALL["D"]["loso"]["by_region"].get(rg, {}).get("mean_r2", "—")
    e_val = ALL["E"]["loso"]["by_region"].get(rg, {}).get("mean_r2", "—")
    f_val = ALL["F3"]["loso"]["by_region"].get(rg, {}).get("mean_r2", "—")
    g_val = ALL["G"]["loso"]["by_region"].get(rg, {}).get("mean_r2", "—")
    rpt.append(f"| {rg} | {c_val:.4f} | {d_val:.4f} | {e_val:.4f} | "
               f"{f_val:.4f} | {g_val:.4f} |"
               if isinstance(d_val, float) else
               f"| {rg} | {c_val} | {d_val} | {e_val} | {f_val} | {g_val} |")

# Feature importance
rpt.append("\n## Feature Importance (Config F, top 20)\n")
rpt.append("| Rank | Feature | Gain |")
rpt.append("|------|---------|------|")
for _, r in imp_df.head(20).iterrows():
    rpt.append(f"| {r['rank']} | {r['feature']} | {r['gain']:.0f} |")

# Anomaly-space KFold R²
rpt.append("\n## Anomaly-Space KFold R²\n")
rpt.append(f"- Config E/F: R²_anom = {kf_ef_r2a:.4f} (absolute R² = {kf_ef_r2:.4f})")
rpt.append(f"- Config G:   R²_anom = {kf_g_r2a:.4f} (absolute R² = {kf_g_r2:.4f})")

# Analysis
rpt.append("\n## Analysis\n")

rpt.append("### 1. How much did dropping geography help? (D vs C)\n")
d_loso = ALL["D"]["loso"]["mean_r2"]
d_gap = round(ALL["D"]["kf_r2"] - d_loso, 4)
rpt.append(f"- Config C: KFold=0.7262, LOSO=-0.4953, gap=1.2215")
rpt.append(f"- Config D: KFold={ALL['D']['kf_r2']:.4f}, LOSO={d_loso:.4f}, gap={d_gap:.4f}")
delta_gap = round(1.2215 - d_gap, 4)
if d_loso > -0.4953:
    rpt.append(f"- LOSO improved by {d_loso - (-0.4953):+.4f}, "
               f"gap reduced by {delta_gap:.4f}")
else:
    rpt.append(f"- LOSO did not improve — geography wasn't the only leakage source")

rpt.append("\n### 2. Oracle ceiling (Config E)\n")
e_loso = ALL["E"]["loso"]["mean_r2"]
rpt.append(f"- Oracle LOSO R² = {e_loso:.4f}")
rpt.append(f"- This is the ceiling with perfect climatology — "
           f"{'promising' if e_loso > 0.3 else 'limited'} signal in anomalies")

rpt.append("\n### 3. Neighbor climatology accuracy (F vs E)\n")
f3_loso = ALL["F3"]["loso"]["mean_r2"]
f5_loso = ALL["F5"]["loso"]["mean_r2"]
rpt.append(f"- Oracle (E): LOSO R² = {e_loso:.4f}")
rpt.append(f"- Neighbor K=3 (F): LOSO R² = {f3_loso:.4f}")
rpt.append(f"- Neighbor K=5 (F): LOSO R² = {f5_loso:.4f}")
best_k = "K=3" if f3_loso >= f5_loso else "K=5"
rpt.append(f"- Best K: {best_k}")
rpt.append(f"- Mean |own - neighbor K=3| climatology diff: "
           f"{clim_comp_df['diff_K3'].abs().mean():.2f} µg/m³")

rpt.append("\n### 4. Does terrain add value after removing identity? (G vs F)\n")
g_loso = ALL["G"]["loso"]["mean_r2"]
rpt.append(f"- Config F (K=3): LOSO R² = {f3_loso:.4f}")
rpt.append(f"- Config G:       LOSO R² = {g_loso:.4f}")
delta = g_loso - f3_loso
rpt.append(f"- Terrain {'helps' if delta > 0 else 'hurts'} by {delta:+.4f}")

rpt.append("\n### 5. South stations\n")
f3_south = ALL["F3"]["loso"]["by_region"].get("South", {})
rpt.append(f"- Config C South: mean R² = -2.1908")
rpt.append(f"- Config F South: mean R² = {f3_south.get('mean_r2', '—')}")
south_stations = merged[merged["region"] == "South"].sort_values("r2_abs")
for _, r in south_stations.iterrows():
    nm = str(r["station_name"])[:40]
    c_r2 = r.get("r2_C", np.nan)
    f_r2 = r["r2_abs"]
    rpt.append(f"  - {nm}: C={c_r2:.3f} → F={f_r2:.3f}")

rpt.append("\n### 6. Best K for neighbor climatology\n")
rpt.append(f"- K=3: mean LOSO R² = {f3_loso:.4f}, neg stations = {sum_f3['neg_count']}")
rpt.append(f"- K=5: mean LOSO R² = {f5_loso:.4f}, neg stations = {sum_f5['neg_count']}")

report_path = "analysis/thesis_experiments/experiment_02_anomaly.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))

print(f"\nReport saved: {report_path}")
print(f"LOSO per-station: analysis/thesis_experiments/loso_per_station_config_f.csv")
print(f"Feature importance: analysis/thesis_experiments/feature_importance_config_f.csv")
print(f"Climatology: analysis/thesis_experiments/climatology_comparison.csv")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
