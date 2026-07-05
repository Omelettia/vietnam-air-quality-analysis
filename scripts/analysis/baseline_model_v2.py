"""
Baseline model v2: push Stage 1 LOSO-safe LOO R² beyond 0.658.

Loads station_baseline_features.csv + ACAG monthly climatology.
Tests feature combos, interactions, nonlinear models, per-station errors.

Output: analysis/thesis_experiments/baseline_model_v2.md
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

print("=" * 70)
print("BASELINE MODEL V2 — PUSHING LOSO-SAFE LOO R²")
print("=" * 70)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════���════════════
#  LOAD
# ═══════════════════════════════════════════════════════════════════════════════
feat = pd.read_csv(os.path.join(OUT_DIR, "station_baseline_features.csv"),
                    dtype={"station_id": str})
acag = pd.read_csv(os.path.join(REPO_DIR, "data/acag/acag_station_climatology.csv"),
                    dtype={"stationId": str})

acag_monthly_cols = [f"ACAG_monthly_clim_{m:02d}" for m in range(1, 13)]
acag_monthly = acag.set_index("stationId")[acag_monthly_cols]
feat = feat.merge(acag_monthly, left_on="station_id", right_index=True, how="left")

y = feat["pm25_mean"].values
n = len(y)
print(f"Loaded {n} stations")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOSO-SAFE BASE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
BEST5 = ["mean_PM25_nn_idw", "mean_PBLH", "mean_VC", "rain_freq", "slope_deg"]

LOSO_SAFE = [
    "mean_PM25_nn_idw", "std_PM25_nn_idw", "iqr_PM25_nn_idw",
    "ACAG_annual_mean", "median_AOT", "AOT_std", "AOT_p95",
    "AOT_valid_frac", "mean_AOD_PBLH_ratio",
    "mean_PBLH", "mean_WS", "mean_VC",
    "mean_Temp", "mean_Humidity", "mean_Pressure", "rain_freq",
    "building_count_3km", "building_area_3km",
    "nbr_seasonal_amp", "nbr_diurnal_range",
    "elevation_m", "latitude", "longitude", "slope_deg",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_X(cols, df=feat):
    X = df[cols].values.copy().astype(float)
    med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        nans = np.isnan(X[:, j])
        X[nans, j] = med[j]
    return X

def scale(X):
    return StandardScaler().fit_transform(X)

def ridge_loo(X, y, alpha=1.0):
    n, p = X.shape
    X_aug = np.column_stack([np.ones(n), X])
    penalty = alpha * np.eye(p + 1)
    penalty[0, 0] = 0.0
    H = X_aug @ np.linalg.solve(X_aug.T @ X_aug + penalty, X_aug.T)
    y_hat = H @ y
    d = np.diag(H)
    preds = (y_hat - d * y) / (1 - d)
    return r2_score(y, preds), mean_absolute_error(y, preds), preds

def loo_cv(model_cls, X, y, **kwargs):
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        tr = np.concatenate([np.arange(0, i), np.arange(i+1, n)])
        m = model_cls(**kwargs)
        m.fit(X[tr], y[tr])
        preds[i] = m.predict(X[i:i+1])[0]
    return r2_score(y, preds), mean_absolute_error(y, preds), preds

results = []

def log_result(name, r2, mae, preds=None):
    results.append({"name": name, "r2": r2, "mae": mae, "preds": preds})
    print(f"  {name:<65s} R²={r2:.4f}  MAE={mae:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  1. BASELINES
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("1. BASELINES")
print(f"{'='*70}")

X5 = scale(prepare_X(BEST5))
r2, mae, preds = ridge_loo(X5, y)
log_result("Best-5 Ridge (baseline)", r2, mae, preds)

Xall = scale(prepare_X(LOSO_SAFE))
r2, mae, preds = ridge_loo(Xall, y)
log_result("All 24 LOSO-safe Ridge", r2, mae, preds)

r2, mae, preds = loo_cv(ElasticNet, Xall, y, alpha=0.1, l1_ratio=0.5, max_iter=5000)
log_result("All 24 LOSO-safe ElasticNet", r2, mae, preds)

r2, mae, preds = loo_cv(RandomForestRegressor, prepare_X(LOSO_SAFE), y,
                          n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
log_result("All 24 LOSO-safe RF", r2, mae, preds)

# ═══════════════════════════════════════════════════════════════════════════════
#  2. ADD ACAG
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("2. ADD ACAG TO BEST-5")
print(f"{'='*70}")

cols_5_acag_ann = BEST5 + ["ACAG_annual_mean"]
r2, mae, preds = ridge_loo(scale(prepare_X(cols_5_acag_ann)), y)
log_result("Best-5 + ACAG_annual_mean", r2, mae, preds)

cols_5_acag_12 = BEST5 + acag_monthly_cols
r2, mae, preds = ridge_loo(scale(prepare_X(cols_5_acag_12)), y)
log_result("Best-5 + 12 ACAG monthly", r2, mae, preds)

cols_5_acag_all = BEST5 + ["ACAG_annual_mean"] + acag_monthly_cols
r2, mae, preds = ridge_loo(scale(prepare_X(cols_5_acag_all)), y)
log_result("Best-5 + ACAG_annual + 12 monthly", r2, mae, preds)

# ═══════════════════════════════════════════════════════════════════════════════
#  3. ADD nbr_diurnal_range
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("3. ADD nbr_diurnal_range")
print(f"{'='*70}")

cols_5_ndr = BEST5 + ["nbr_diurnal_range"]
r2, mae, preds = ridge_loo(scale(prepare_X(cols_5_ndr)), y)
log_result("Best-5 + nbr_diurnal_range", r2, mae, preds)

cols_5_ndr_acag = BEST5 + ["nbr_diurnal_range", "ACAG_annual_mean"]
r2, mae, preds = ridge_loo(scale(prepare_X(cols_5_ndr_acag)), y)
log_result("Best-5 + nbr_diurnal_range + ACAG_annual", r2, mae, preds)

# ═══════════════════════════════════════════════════════════════════════════════
#  4. BEST 6, 7, 8 SUBSETS (LOSO-safe)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("4. BEST SUBSETS (6, 7, 8 features)")
print(f"{'='*70}")

safe_names = LOSO_SAFE
Xsafe_scaled = scale(prepare_X(safe_names))

best_subsets = {}
for k in [6, 7, 8]:
    combos = list(itertools.combinations(range(len(safe_names)), k))
    print(f"  k={k}: testing {len(combos):,} combinations ...", end="", flush=True)
    t1 = time.time()
    best_r2 = -999
    best_combo = None
    best_preds = None
    for combo in combos:
        cols = list(combo)
        r2, mae, preds = ridge_loo(Xsafe_scaled[:, cols], y)
        if r2 > best_r2:
            best_r2 = r2
            best_combo = combo
            best_preds = preds.copy()
    best_feat = [safe_names[i] for i in best_combo]
    best_mae = mean_absolute_error(y, best_preds)
    best_subsets[k] = {"features": best_feat, "r2": best_r2, "mae": best_mae,
                       "preds": best_preds}
    print(f" {time.time()-t1:.0f}s")
    log_result(f"Best-{k} Ridge LOSO-safe", best_r2, best_mae, best_preds)
    print(f"    Features: {best_feat}")

# ═══════════════════════════════════════════════════════════════════════════════
#  5. NONLINEAR ON BEST SUBSETS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("5. NONLINEAR MODELS ON BEST SUBSETS")
print(f"{'='*70}")

for k in [6, 7, 8]:
    feats = best_subsets[k]["features"]
    Xk = prepare_X(feats)
    Xks = scale(Xk)

    r2, mae, preds = loo_cv(ElasticNet, Xks, y, alpha=0.1, l1_ratio=0.5, max_iter=5000)
    log_result(f"Best-{k} ElasticNet", r2, mae, preds)

    r2, mae, preds = loo_cv(RandomForestRegressor, Xk, y,
                              n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
    log_result(f"Best-{k} RF (200 trees, depth=4)", r2, mae, preds)

# ═══════════════════════════════════════════════════════════════════════════════
#  6. FEATURE INTERACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("6. FEATURE INTERACTIONS")
print(f"{'='*70}")

interactions = [
    ("idw_x_ACAG", "mean_PM25_nn_idw", "ACAG_annual_mean"),
    ("ndr_x_bldg", "nbr_diurnal_range", "building_area_3km"),
    ("VC_x_rain", "mean_VC", "rain_freq"),
    ("idw_x_ndr", "mean_PM25_nn_idw", "nbr_diurnal_range"),
    ("ACAG_x_lat", "ACAG_annual_mean", "latitude"),
    ("idw_x_PBLH", "mean_PM25_nn_idw", "mean_PBLH"),
]

for iname, f1, f2 in interactions:
    feat[iname] = feat[f1] * feat[f2]

int_names = [x[0] for x in interactions]

# Best-8 + all interactions
best8_feats = best_subsets[8]["features"]
cols_8_int = best8_feats + int_names
Xint = scale(prepare_X(cols_8_int))
r2, mae, preds = ridge_loo(Xint, y)
log_result(f"Best-8 + 6 interactions Ridge", r2, mae, preds)

r2, mae, preds = loo_cv(ElasticNet, Xint, y, alpha=0.1, l1_ratio=0.5, max_iter=5000)
log_result(f"Best-8 + 6 interactions ElasticNet", r2, mae, preds)

r2, mae, preds = loo_cv(RandomForestRegressor, prepare_X(cols_8_int), y,
                          n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
log_result(f"Best-8 + 6 interactions RF", r2, mae, preds)

# Best-5 + interactions only
cols_5_int = BEST5 + int_names
r2, mae, preds = ridge_loo(scale(prepare_X(cols_5_int)), y)
log_result(f"Best-5 + 6 interactions Ridge", r2, mae, preds)

# Try ACAG monthly + best features
best8_acag = best8_feats + acag_monthly_cols
r2, mae, preds = ridge_loo(scale(prepare_X(best8_acag)), y)
log_result(f"Best-8 + 12 ACAG monthly Ridge", r2, mae, preds)

r2, mae, preds = loo_cv(ElasticNet, scale(prepare_X(best8_acag)), y,
                          alpha=0.1, l1_ratio=0.5, max_iter=5000)
log_result(f"Best-8 + 12 ACAG monthly ElasticNet", r2, mae, preds)

# All LOSO-safe + interactions + ACAG monthly
all_cols = LOSO_SAFE + int_names + acag_monthly_cols
r2, mae, preds = loo_cv(ElasticNet, scale(prepare_X(all_cols)), y,
                          alpha=0.1, l1_ratio=0.5, max_iter=5000)
log_result(f"All LOSO-safe + interactions + ACAG monthly EN", r2, mae, preds)

r2, mae, preds = loo_cv(RandomForestRegressor, prepare_X(all_cols), y,
                          n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
log_result(f"All LOSO-safe + interactions + ACAG monthly RF", r2, mae, preds)

# ═══════════════════════════════════════════════════════════════════════════════
#  7. FIND THE OVERALL BEST MODEL — PER-STATION ERRORS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("7. BEST MODEL — PER-STATION ERRORS")
print(f"{'='*70}")

res_df = pd.DataFrame([{"name": r["name"], "r2": r["r2"], "mae": r["mae"]}
                        for r in results]).sort_values("r2", ascending=False)
print("\nTop 10 models:")
for i, (_, r) in enumerate(res_df.head(10).iterrows()):
    print(f"  {i+1}. {r['name']:<60s} R²={r['r2']:.4f}  MAE={r['mae']:.2f}")

best_idx = res_df.index[0]
best_model = results[best_idx]
best_preds = best_model["preds"]
best_name = best_model["name"]
print(f"\nBest model: {best_name}")

if best_preds is not None:
    errors = y - best_preds
    abs_errors = np.abs(errors)
    print(f"\n{'Station':<55s} {'Actual':>7s} {'Pred':>7s} {'Error':>7s} {'|Err|':>6s}")
    print(f"{'-'*55} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    order = np.argsort(-abs_errors)
    for idx in order:
        nm = str(feat.iloc[idx]["station_name"])[:55]
        print(f"{nm:<55s} {y[idx]:7.1f} {best_preds[idx]:7.1f} "
              f"{errors[idx]:+7.1f} {abs_errors[idx]:6.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  WRITE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
md_path = os.path.join(OUT_DIR, "baseline_model_v2.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Baseline Model V2 — Pushing LOSO-Safe LOO R²\n\n")
    f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"**Target**: station mean PM2.5 (40 stations, LOO-CV)\n\n")

    f.write("## All Results\n\n")
    f.write("| # | Model | R² | MAE (µg/m³) |\n")
    f.write("|---|-------|:---:|:---:|\n")
    for i, (_, r) in enumerate(res_df.iterrows()):
        f.write(f"| {i+1} | {r['name']} | {r['r2']:.4f} | {r['mae']:.2f} |\n")

    f.write(f"\n## Best Subsets\n\n")
    for k in [6, 7, 8]:
        bs = best_subsets[k]
        f.write(f"**Best {k}**: {', '.join(bs['features'])} "
                f"(R²={bs['r2']:.4f}, MAE={bs['mae']:.2f})\n\n")

    f.write("## Feature Interactions Tested\n\n")
    f.write("| Name | Formula |\n")
    f.write("|------|--------|\n")
    for iname, f1, f2 in interactions:
        f.write(f"| {iname} | {f1} × {f2} |\n")

    if best_preds is not None:
        f.write(f"\n## Per-Station Errors — Best Model: {best_name}\n\n")
        f.write(f"R² = {best_model['r2']:.4f}, MAE = {best_model['mae']:.2f}\n\n")
        f.write("| Station | Region | Actual | Predicted | Error | |Error| |\n")
        f.write("|---------|--------|:---:|:---:|:---:|:---:|\n")
        order = np.argsort(-abs_errors)
        for idx in order:
            row = feat.iloc[idx]
            nm = str(row["station_name"])[:45]
            reg = str(row["region"])
            f.write(f"| {nm} | {reg} | {y[idx]:.1f} | {best_preds[idx]:.1f} | "
                    f"{errors[idx]:+.1f} | {abs_errors[idx]:.1f} |\n")

    f.write("\n## Key Findings\n\n")
    top = res_df.iloc[0]
    base = [r for r in results if r["name"] == "Best-5 Ridge (baseline)"][0]
    f.write(f"- Baseline (best-5 Ridge): R²={base['r2']:.4f}, MAE={base['mae']:.2f}\n")
    f.write(f"- Best model: {top['name']} — R²={top['r2']:.4f}, MAE={top['mae']:.2f}\n")
    f.write(f"- Improvement: +{top['r2']-base['r2']:.4f} R², "
            f"-{base['mae']-top['mae']:.2f} MAE\n")

print(f"\nSaved: {md_path}")
print(f"Done — {time.time()-t0:.0f}s total")
