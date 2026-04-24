"""
Experiment 01: XGBoost PM2.5 Baseline — 3 feature configs, KFold + LOSO evaluation.

Configs:
  A — Met-only (no satellite)
  B — Met + AOD
  C — Full (Met + AOD + spatial gradients)

Output:
  analysis/thesis_experiments/experiment_01_baseline.md
  analysis/thesis_experiments/loso_per_station_config_c.csv
  analysis/thesis_experiments/feature_importance_config_c.csv
  models/xgb_config_c_full.json
"""

import argparse, io, sys, os, warnings, time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None,
                    help="Base directory containing data/ and analysis/ folders")
args = parser.parse_args()

BASE = args.data_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
os.makedirs("analysis/thesis_experiments", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
#  STEP 1: FEATURE SETS
# ══════════════════════════════════════════════════════════════════════

FEATURES_A = [
    # Met
    "Temperature_final", "Humidity_final", "Pressure_final",
    "PBLH", "VC", "RH_factor",
    # Wind (OpenMeteo)
    "wind_u", "wind_v", "wind_dir_sin", "wind_dir_cos",
    # Wind (local)
    "WS_local", "wind_u_local", "wind_v_local", "wind_dir_sin_local", "wind_dir_cos_local",
    # Met derivatives
    "dT_6h", "dRH_6h", "dWS_6h", "dP_6h",
    # Temporal
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
    # Precipitation
    "precip_mm", "hrs_since_rain", "rain_sum_24h", "rain_sum_48h",
    "rain_days_7d", "consecutive_dry_days",
    # Spatial/static
    "latitude", "longitude",
    "elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
    # Interactions
    "elev_x_PBLH", "elev_x_hour_sin",
]

FEATURES_B = FEATURES_A + [
    # AOD core
    "AOT", "AOT_mean", "AOT_inner_mean", "AOT_outer_mean",
    "RF", "SSA", "Uncertainty", "AE",
    "AOT_valid_count",
    "AOD_physics",
    "AOT_spatial_std",
    "AOT_local_vs_regional",
    # AOD temporal
    "AOT_ffill_48h", "hours_since_valid_AOT",
    "AOT_lag_1h", "AOT_lag_3h", "AOT_lag_6h",
    "AOT_rolling_mean_6h", "AOT_rolling_mean_24h",
]

FEATURES_C = FEATURES_B + [
    "AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_grad_dir",
]

CONFIGS = {
    "A": FEATURES_A,
    "B": FEATURES_B,
    "C": FEATURES_C,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

# ══════════════════════════════════════════════════════════════════════
#  STEP 2: DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

t0 = time.time()
df = pd.read_csv("data/merged/unified_thesis_v1.csv",
                  dtype={"stationId": str})
print(f"Loaded: {len(df):,} rows × {len(df.columns)} cols in {time.time()-t0:.1f}s")

# Drop rows without PM2.5
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
print(f"After dropping NaN PM2.5: {len(df):,} rows")

# Integer-encode stations for LOSO indexing
station_ids = sorted(df["stationId"].unique())
sid_to_int = {s: i for i, s in enumerate(station_ids)}
df["station_idx"] = df["stationId"].map(sid_to_int)

# Load station metadata for region mapping
station_meta = pd.read_csv("analysis/thesis_audit/station_selection_final.csv",
                            dtype={"stationId": str})
sid_to_region = dict(zip(station_meta["stationId"], station_meta["region"]))
sid_to_name = dict(zip(station_meta["stationId"], station_meta["station_name"]))

TARGET = "PM2.5"
y_all = df[TARGET].values

# Verify all feature columns exist
for config_name, features in CONFIGS.items():
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"WARNING: Config {config_name} missing columns: {missing}")
        CONFIGS[config_name] = [f for f in features if f in df.columns]

print(f"\nFeature counts: A={len(CONFIGS['A'])}, B={len(CONFIGS['B'])}, C={len(CONFIGS['C'])}")
print(f"Stations: {len(station_ids)}")
print(f"PM2.5 stats: mean={y_all.mean():.1f}, std={y_all.std():.1f}, "
      f"median={np.median(y_all):.1f}")

# ══════════════════════════════════════════════════════════════════════
#  STEP 3-4: TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════

def run_kfold(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[val_idx])
        r2 = r2_score(y[val_idx], preds)
        rmse = np.sqrt(mean_squared_error(y[val_idx], preds))
        mae = mean_absolute_error(y[val_idx], preds)
        fold_metrics.append({"fold": fold, "r2": r2, "rmse": rmse, "mae": mae})
    return fold_metrics


def run_loso(X, y, station_indices, station_ids_list, config_name=""):
    results = []
    unique_stations = sorted(set(station_indices))

    for i, held_out_idx in enumerate(unique_stations):
        mask_test = station_indices == held_out_idx
        mask_train = ~mask_test
        n_test = mask_test.sum()
        n_train = mask_train.sum()

        sid = station_ids_list[held_out_idx]
        sname = sid_to_name.get(sid, sid)[:45]
        region = sid_to_region.get(sid, "?")

        if n_test < 10:
            print(f"  [{i+1:2d}/{len(unique_stations)}] {sname:45s} | SKIP (n={n_test})")
            continue

        try:
            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X.iloc[mask_train], y[mask_train])
            preds = model.predict(X.iloc[mask_test])
            r2 = r2_score(y[mask_test], preds)
            rmse = np.sqrt(mean_squared_error(y[mask_test], preds))
            mae = mean_absolute_error(y[mask_test], preds)

            print(f"  [{i+1:2d}/{len(unique_stations)}] {sname:45s} | "
                  f"R²={r2:+.3f}  RMSE={rmse:5.1f}  n={n_test:5d}  [{region}]")

            results.append({
                "station_id": sid,
                "station_name": sid_to_name.get(sid, sid),
                "region": region,
                "n_rows": n_test,
                "r2": round(r2, 4),
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
            })
        except Exception as e:
            print(f"  [{i+1:2d}/{len(unique_stations)}] {sname:45s} | ERROR: {e}")
            results.append({
                "station_id": sid, "station_name": sid_to_name.get(sid, sid),
                "region": region, "n_rows": n_test,
                "r2": np.nan, "rmse": np.nan, "mae": np.nan,
            })

    return results


# Store all results
all_results = {}

for config_name in ["A", "B", "C"]:
    features = CONFIGS[config_name]
    X = df[features].copy()

    print(f"\n{'='*80}")
    print(f"CONFIG {config_name}: {len(features)} features")
    print(f"{'='*80}")

    # KFold
    print(f"\n--- KFold 5-fold CV ---")
    t1 = time.time()
    kf_results = run_kfold(X, y_all)
    kf_time = time.time() - t1
    kf_r2 = np.mean([m["r2"] for m in kf_results])
    kf_rmse = np.mean([m["rmse"] for m in kf_results])
    kf_mae = np.mean([m["mae"] for m in kf_results])
    print(f"  Mean R²={kf_r2:.4f}, RMSE={kf_rmse:.2f}, MAE={kf_mae:.2f} ({kf_time:.0f}s)")

    # LOSO
    print(f"\n--- LOSO CV ({len(station_ids)} stations) ---")
    t2 = time.time()
    loso_results = run_loso(X, y_all, df["station_idx"].values,
                             station_ids, config_name)
    loso_time = time.time() - t2

    loso_df = pd.DataFrame(loso_results)
    valid_loso = loso_df.dropna(subset=["r2"])
    loso_mean_r2 = valid_loso["r2"].mean()
    loso_median_r2 = valid_loso["r2"].median()
    loso_wmean_r2 = (valid_loso["r2"] * valid_loso["n_rows"]).sum() / valid_loso["n_rows"].sum()
    loso_mean_rmse = valid_loso["rmse"].mean()
    loso_mean_mae = valid_loso["mae"].mean()
    neg_r2 = (valid_loso["r2"] < 0).sum()

    print(f"\n  LOSO Summary:")
    print(f"    Mean R²:     {loso_mean_r2:.4f}")
    print(f"    Median R²:   {loso_median_r2:.4f}")
    print(f"    Weighted R²: {loso_wmean_r2:.4f}")
    print(f"    Mean RMSE:   {loso_mean_rmse:.2f}")
    print(f"    Mean MAE:    {loso_mean_mae:.2f}")
    print(f"    Neg R² stns: {neg_r2}")
    print(f"    Time: {loso_time:.0f}s")

    # Region breakdown
    print(f"\n  By region:")
    for region in ["North", "Central", "South"]:
        sub = valid_loso[valid_loso["region"] == region]
        if len(sub) > 0:
            print(f"    {region:8s}: n={len(sub):2d}, mean R²={sub['r2'].mean():.4f}, "
                  f"median R²={sub['r2'].median():.4f}")

    all_results[config_name] = {
        "n_features": len(features),
        "kf_r2": round(kf_r2, 4),
        "kf_rmse": round(kf_rmse, 2),
        "kf_mae": round(kf_mae, 2),
        "loso_mean_r2": round(loso_mean_r2, 4),
        "loso_median_r2": round(loso_median_r2, 4),
        "loso_wmean_r2": round(loso_wmean_r2, 4),
        "loso_mean_rmse": round(loso_mean_rmse, 2),
        "loso_mean_mae": round(loso_mean_mae, 2),
        "loso_neg_r2_count": neg_r2,
        "loso_df": loso_df,
        "kf_results": kf_results,
    }

# ══════════════════════════════════════════════════════════════════════
#  STEP 5: FEATURE IMPORTANCE (Config C)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("STEP 5: FEATURE IMPORTANCE (Config C — full training)")
print(f"{'='*80}")

features_c = CONFIGS["C"]
X_full = df[features_c].copy()

# Train on all data for feature importance + model saving
model_full = xgb.XGBRegressor(**XGB_PARAMS)
model_full.fit(X_full, y_all)

# Built-in importance (gain)
importance = model_full.get_booster().get_score(importance_type="gain")
imp_df = pd.DataFrame([
    {"feature": k, "importance_gain": v}
    for k, v in importance.items()
]).sort_values("importance_gain", ascending=False).reset_index(drop=True)

# Map feature indices to names (xgboost uses f0, f1, etc. sometimes)
feat_name_map = {f"f{i}": name for i, name in enumerate(features_c)}
imp_df["feature"] = imp_df["feature"].map(lambda x: feat_name_map.get(x, x))
imp_df["rank"] = range(1, len(imp_df) + 1)

imp_df.to_csv("analysis/thesis_experiments/feature_importance_config_c.csv",
              index=False, encoding="utf-8-sig")

print(f"\nTop 20 features by gain:")
for _, r in imp_df.head(20).iterrows():
    marker = ""
    new_feats = {"AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_grad_dir",
                 "elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
                 "WS_local", "wind_u_local", "wind_v_local",
                 "AOD_physics", "RF", "AOT_local_vs_regional", "AOT_spatial_std"}
    if r["feature"] in new_feats:
        marker = " ★"
    print(f"  {r['rank']:2d}. {r['feature']:30s} gain={r['importance_gain']:.0f}{marker}")

# Save model
model_full.save_model("models/xgb_config_c_full.json")
print(f"\nModel saved: models/xgb_config_c_full.json")

# ══════════════════════════════════════════════════════════════════════
#  STEP 6: SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("STEP 6: RESULTS SUMMARY")
print(f"{'='*80}")

# Save LOSO per-station for Config C
loso_c = all_results["C"]["loso_df"].sort_values("r2", ascending=True)
loso_c.to_csv("analysis/thesis_experiments/loso_per_station_config_c.csv",
              index=False, encoding="utf-8-sig")

# Write report
report = []
report.append("# Experiment 01: XGBoost PM2.5 Baseline\n")
report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
report.append(f"**Dataset:** unified_thesis_v1.csv — {len(df):,} rows (PM2.5 non-null), "
              f"{len(station_ids)} stations")
report.append(f"**Date range:** {df['ts'].min()} to {df['ts'].max()}")
report.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, lr=0.05\n")

report.append("## Comparison Table\n")
report.append("| Config | Features | KFold R² | KFold RMSE | KFold MAE | LOSO R² (mean) | "
              "LOSO R² (median) | LOSO R² (weighted) | LOSO RMSE | Gap (KF-LOSO) |")
report.append("|--------|----------|----------|------------|-----------|----------------|"
              "------------------|---------------------|-----------|---------------|")

for cfg in ["A", "B", "C"]:
    r = all_results[cfg]
    gap = round(r["kf_r2"] - r["loso_mean_r2"], 4)
    report.append(f"| {cfg} ({'met-only' if cfg=='A' else 'met+AOD' if cfg=='B' else 'full'}) "
                  f"| {r['n_features']} "
                  f"| {r['kf_r2']:.4f} | {r['kf_rmse']:.2f} | {r['kf_mae']:.2f} "
                  f"| {r['loso_mean_r2']:.4f} | {r['loso_median_r2']:.4f} "
                  f"| {r['loso_wmean_r2']:.4f} | {r['loso_mean_rmse']:.2f} "
                  f"| {gap:.4f} |")

report.append(f"| Previous (14-stn) | 32 | 0.8130 | — | — | 0.2090 | — | — | — | 0.6040 |")

report.append("\n## Per-Station LOSO Results (Config C, sorted worst → best)\n")
report.append("| Station | Region | n | R² | RMSE | MAE |")
report.append("|---------|--------|---|-----|------|-----|")
for _, r in loso_c.iterrows():
    sname = str(r["station_name"])[:50]
    flag = " ⚠" if r["r2"] < 0 else ""
    report.append(f"| {sname} | {r['region']} | {r['n_rows']:,} "
                  f"| {r['r2']:.4f}{flag} | {r['rmse']:.1f} | {r['mae']:.1f} |")

neg_stations = loso_c[loso_c["r2"] < 0]
report.append(f"\n**Stations with negative R²:** {len(neg_stations)}")
if len(neg_stations) > 0:
    for _, r in neg_stations.iterrows():
        report.append(f"- {r['station_name'][:50]} (R²={r['r2']:.4f}, region={r['region']})")

report.append("\n## LOSO by Region (Config C)\n")
report.append("| Region | Stations | Mean R² | Median R² | Mean RMSE |")
report.append("|--------|----------|---------|-----------|-----------|")
valid_c = loso_c.dropna(subset=["r2"])
for region in ["North", "Central", "South", "Unknown"]:
    sub = valid_c[valid_c["region"] == region]
    if len(sub) > 0:
        report.append(f"| {region} | {len(sub)} | {sub['r2'].mean():.4f} "
                      f"| {sub['r2'].median():.4f} | {sub['rmse'].mean():.1f} |")

report.append("\n## Feature Importance (Config C, top 20)\n")
report.append("| Rank | Feature | Gain | New? |")
report.append("|------|---------|------|------|")
new_feats = {"AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_grad_dir",
             "elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
             "WS_local", "wind_u_local", "wind_v_local",
             "AOD_physics", "RF", "AOT_local_vs_regional", "AOT_spatial_std"}
for _, r in imp_df.head(20).iterrows():
    is_new = "yes" if r["feature"] in new_feats else ""
    report.append(f"| {r['rank']} | {r['feature']} | {r['importance_gain']:.0f} | {is_new} |")

# Where do key new features rank?
report.append("\n### Key New Feature Rankings\n")
key_new = ["AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag",
           "elevation_m", "slope_deg", "WS_local",
           "AOD_physics", "RF", "AOT_local_vs_regional"]
report.append("| Feature | Rank | Gain |")
report.append("|---------|------|------|")
for fn in key_new:
    match = imp_df[imp_df["feature"] == fn]
    if len(match) > 0:
        r = match.iloc[0]
        report.append(f"| {fn} | {r['rank']} | {r['importance_gain']:.0f} |")
    else:
        report.append(f"| {fn} | not used | 0 |")

report.append("\n## Key Questions Answered\n")
report.append("### 1. Did going from 14 → 40 stations close the LOSO gap?\n")
prev_gap = 0.604
new_gap = round(all_results["C"]["kf_r2"] - all_results["C"]["loso_mean_r2"], 4)
report.append(f"- Previous (14 stations): KFold=0.813, LOSO=0.209, gap=0.604")
report.append(f"- Current (40 stations): KFold={all_results['C']['kf_r2']:.4f}, "
              f"LOSO={all_results['C']['loso_mean_r2']:.4f}, gap={new_gap:.4f}")
if new_gap < prev_gap:
    report.append(f"- **Gap reduced by {prev_gap - new_gap:.3f}** — "
                  f"{'substantial' if (prev_gap - new_gap) > 0.1 else 'modest'} improvement in spatial generalization")
else:
    report.append(f"- Gap did not improve — need more diverse training stations")

report.append("\n### 2. Does adding AOD features (B vs A) help LOSO, or only KFold?\n")
a_loso = all_results["A"]["loso_mean_r2"]
b_loso = all_results["B"]["loso_mean_r2"]
a_kf = all_results["A"]["kf_r2"]
b_kf = all_results["B"]["kf_r2"]
report.append(f"- KFold: A={a_kf:.4f} → B={b_kf:.4f} (Δ={b_kf-a_kf:+.4f})")
report.append(f"- LOSO:  A={a_loso:.4f} → B={b_loso:.4f} (Δ={b_loso-a_loso:+.4f})")
if b_loso > a_loso:
    report.append("- AOD helps BOTH KFold and LOSO — genuine spatial signal")
else:
    report.append("- AOD helps KFold but not LOSO — may be overfitting to station-specific AOD patterns")

report.append("\n### 3. Do spatial gradients (C vs B) add anything?\n")
b_loso2 = all_results["B"]["loso_mean_r2"]
c_loso = all_results["C"]["loso_mean_r2"]
b_kf2 = all_results["B"]["kf_r2"]
c_kf = all_results["C"]["kf_r2"]
report.append(f"- KFold: B={b_kf2:.4f} → C={c_kf:.4f} (Δ={c_kf-b_kf2:+.4f})")
report.append(f"- LOSO:  B={b_loso2:.4f} → C={c_loso:.4f} (Δ={c_loso-b_loso2:+.4f})")

report.append("\n### 4. Which region generalizes best/worst?\n")
for region in ["North", "Central", "South"]:
    sub = valid_c[valid_c["region"] == region]
    if len(sub) > 0:
        report.append(f"- {region}: mean LOSO R²={sub['r2'].mean():.4f} "
                      f"(n={len(sub)} stations)")

report.append("\n### 5. Feature importance — do new features matter?\n")
top10_names = set(imp_df.head(10)["feature"].values)
new_in_top10 = top10_names.intersection(new_feats)
report.append(f"- New features in top 10: {len(new_in_top10)} — {sorted(new_in_top10) if new_in_top10 else 'none'}")

report_path = "analysis/thesis_experiments/experiment_01_baseline.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"\nReport saved: {report_path}")
print(f"LOSO results: analysis/thesis_experiments/loso_per_station_config_c.csv")
print(f"Feature importance: analysis/thesis_experiments/feature_importance_config_c.csv")
print(f"Model: models/xgb_config_c_full.json")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
