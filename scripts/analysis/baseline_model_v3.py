"""
Baseline model v3: AOD spatial + RF fine-mode features.

Adds per-station means of AOD spatial features and RF (fine-mode ratio)
to the LOSO-safe feature pool. Exhaustive best-k Ridge LOO-CV search.
If new subset beats R²=0.695, runs full two-stage LOSO pipeline.

Output: analysis/thesis_experiments/baseline_model_v3.md
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
print("BASELINE MODEL V3 — AOD SPATIAL + RF FINE-MODE FEATURES")
print("=" * 70)
t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD FEATURES CSV + COMPUTE NEW FEATURES FROM UNIFIED DATASET
# ═══════════════════════════════════════════════════════════════════════════════
feat = pd.read_csv(os.path.join(OUT_DIR, "station_baseline_features.csv"),
                    dtype={"station_id": str})

EXTRA_COLS = ["AOT_inner_mean", "AOT_outer_mean", "AOT_spatial_std",
              "AOT_local_vs_regional", "AOT_grad_mag", "RF",
              "RF_inner_mean", "RF_outer_mean", "AOT"]

print("Loading unified dataset for new features ...")
t1 = time.time()
want_cols = ["stationId", "PM2.5"] + EXTRA_COLS
df_raw = pd.read_csv(os.path.join(REPO_DIR, "data/merged/unified_thesis_v1.csv"),
                      dtype={"stationId": str}, usecols=lambda c: c in want_cols)
df_raw = df_raw.dropna(subset=["PM2.5"]).reset_index(drop=True)
available = [c for c in EXTRA_COLS if c in df_raw.columns]
missing = [c for c in EXTRA_COLS if c not in df_raw.columns]
print(f"Loaded: {len(df_raw):,} rows ({time.time()-t1:.1f}s)")
print(f"Available: {available}")
if missing:
    print(f"Missing (not in unified dataset): {missing}")

new_feat_cols = []
for col in available:
    mean_col = f"mean_{col}"
    mapping = df_raw.groupby("stationId")[col].mean()
    feat[mean_col] = feat["station_id"].map(mapping)
    nna = feat[mean_col].notna().sum()
    new_feat_cols.append(mean_col)
    print(f"  {mean_col}: {nna}/40 non-NaN")

# Derived: AOT_fine_clim = mean_AOT × mean_RF
if "mean_AOT" in feat.columns and "mean_RF" in feat.columns:
    feat["AOT_fine_clim"] = feat["mean_AOT"] * feat["mean_RF"]
    new_feat_cols.append("AOT_fine_clim")
    nna = feat["AOT_fine_clim"].notna().sum()
    print(f"  AOT_fine_clim (mean_AOT × mean_RF): {nna}/40 non-NaN")

# Save updated CSV
feat.to_csv(os.path.join(OUT_DIR, "station_baseline_features.csv"),
            index=False, encoding="utf-8-sig")
print(f"Updated CSV: {len(feat.columns)} columns")

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE POOL
# ═══════════════════════════════════════════════════════════════════════════════
y = feat["pm25_mean"].values
n = len(y)

PREV_LOSO_SAFE = [
    "mean_PM25_nn_idw", "std_PM25_nn_idw", "iqr_PM25_nn_idw",
    "ACAG_annual_mean", "median_AOT", "AOT_std", "AOT_p95",
    "AOT_valid_frac", "mean_AOD_PBLH_ratio",
    "mean_PBLH", "mean_WS", "mean_VC",
    "mean_Temp", "mean_Humidity", "mean_Pressure", "rain_freq",
    "building_count_3km", "building_area_3km",
    "nbr_seasonal_amp", "nbr_diurnal_range",
    "elevation_m", "latitude", "longitude", "slope_deg",
]

FULL_POOL = PREV_LOSO_SAFE + new_feat_cols
# Deduplicate (mean_AOT_inner_mean etc. may already be from v3_aod run)
FULL_POOL = list(dict.fromkeys(FULL_POOL))

usable = []
for c in FULL_POOL:
    if c not in feat.columns:
        continue
    nna = feat[c].notna().sum()
    if nna >= 30:
        usable.append(c)
    else:
        print(f"  Dropping {c}: only {nna} non-NaN")
print(f"\nUsable features: {len(usable)} ({len(usable)-24} new vs v2)")

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_X(cols):
    X = feat[cols].values.copy().astype(float)
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
def log(name, r2, mae, preds=None):
    results.append({"name": name, "r2": r2, "mae": mae, "preds": preds})
    print(f"  {name:<60s} R²={r2:.4f}  MAE={mae:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CORRELATIONS WITH STATION MEAN PM2.5")
print(f"{'='*70}")

print(f"\n  {'Feature':<30s} {'Pearson':>8s} {'Spearman':>9s} {'New?':>5s}")
print(f"  {'-'*30} {'-'*8} {'-'*9} {'-'*5}")

corr_rows = []
for c in usable:
    vals = feat[c].values
    valid = ~np.isnan(vals) & ~np.isnan(y)
    if valid.sum() < 10:
        continue
    pr, _ = sp_stats.pearsonr(vals[valid], y[valid])
    sr, _ = sp_stats.spearmanr(vals[valid], y[valid])
    is_new = c in new_feat_cols
    tag = "NEW" if is_new else ""
    print(f"  {c:<30s} {pr:+.4f}   {sr:+.4f}   {tag}")
    corr_rows.append({"feature": c, "pearson": round(pr, 4),
                       "spearman": round(sr, 4), "new": is_new})

corr_df = pd.DataFrame(corr_rows).sort_values("spearman", ascending=False, key=abs)

# ═══════════════════════════════════════════════════════════════════════════════
#  BASELINES
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BASELINES")
print(f"{'='*70}")

V2_BEST5 = ["mean_PM25_nn_idw", "mean_PBLH", "mean_VC", "rain_freq", "slope_deg"]
r2, mae, p = ridge_loo(scale(prepare_X(V2_BEST5)), y)
log("v2 Best-5 Ridge", r2, mae, p)

V3A_BEST7 = ["mean_PM25_nn_idw", "AOT_p95", "AOT_valid_frac", "mean_WS",
              "mean_VC", "slope_deg", "mean_AOT_grad_mag"]
if all(c in feat.columns for c in V3A_BEST7):
    r2, mae, p = ridge_loo(scale(prepare_X(V3A_BEST7)), y)
    log("v3a Best-7 Ridge (AOD grad)", r2, mae, p)

r2, mae, p = ridge_loo(scale(prepare_X(usable)), y)
log(f"All {len(usable)} features Ridge", r2, mae, p)

r2, mae, p = loo_cv(RandomForestRegressor, prepare_X(usable), y,
                      n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
log(f"All {len(usable)} features RF", r2, mae, p)

# ═══════════════════════════════════════════════════════════════════════════════
#  BEST SUBSETS k=5,6,7,8
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BEST SUBSETS (k=5,6,7,8)")
print(f"{'='*70}")

Xall_scaled = scale(prepare_X(usable))
best_subsets = {}

for k in [5, 6, 7, 8]:
    combos = list(itertools.combinations(range(len(usable)), k))
    print(f"  k={k}: {len(combos):,} combinations ...", end="", flush=True)
    t1 = time.time()
    best_r2 = -999
    best_combo = None
    best_preds = None
    for combo in combos:
        cols = list(combo)
        r2, mae, preds = ridge_loo(Xall_scaled[:, cols], y)
        if r2 > best_r2:
            best_r2 = r2
            best_combo = combo
            best_preds = preds.copy()
    best_feat = [usable[i] for i in best_combo]
    best_mae = mean_absolute_error(y, best_preds)
    best_subsets[k] = {"features": best_feat, "r2": best_r2, "mae": best_mae,
                       "preds": best_preds}
    elapsed = time.time() - t1
    print(f" {elapsed:.0f}s")
    has_new = [f for f in best_feat if f in new_feat_cols]
    log(f"Best-{k} Ridge", best_r2, best_mae, best_preds)
    print(f"    Features: {best_feat}")
    if has_new:
        print(f"    New features: {has_new}")

# ═══════════════════════════════════════════════════════════════════════════════
#  RF + ELASTICNET ON BEST SUBSETS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("NONLINEAR ON BEST SUBSETS")
print(f"{'='*70}")

for k in [6, 7, 8]:
    feats = best_subsets[k]["features"]
    Xk = prepare_X(feats)
    r2, mae, p = loo_cv(ElasticNet, scale(Xk), y,
                          alpha=0.1, l1_ratio=0.5, max_iter=5000)
    log(f"Best-{k} ElasticNet", r2, mae, p)
    r2, mae, p = loo_cv(RandomForestRegressor, Xk, y,
                          n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
    log(f"Best-{k} RF", r2, mae, p)

# ═══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL NEW FEATURE ADDITIONS TO v2 BEST-5
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("v2 BEST-5 + INDIVIDUAL NEW FEATURES")
print(f"{'='*70}")

for nf in new_feat_cols:
    if nf not in feat.columns or feat[nf].notna().sum() < 30:
        continue
    cols = V2_BEST5 + [nf]
    r2, mae, p = ridge_loo(scale(prepare_X(cols)), y)
    log(f"Best-5 + {nf}", r2, mae, p)

# ═══════════════════════════════════════════════════════════════════════════════
#  PER-STATION ERRORS FOR BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("RANKINGS + PER-STATION ERRORS")
print(f"{'='*70}")

res_df = pd.DataFrame([{"name": r["name"], "r2": r["r2"], "mae": r["mae"]}
                        for r in results]).sort_values("r2", ascending=False)

print("\nTop 15 models:")
for i, (_, r) in enumerate(res_df.head(15).iterrows()):
    print(f"  {i+1:2d}. {r['name']:<55s} R²={r['r2']:.4f}  MAE={r['mae']:.2f}")

best_idx = res_df.index[0]
best_model = results[best_idx]
best_preds = best_model["preds"]

if best_preds is not None:
    errors = y - best_preds
    abs_errors = np.abs(errors)
    print(f"\nPer-station errors ({best_model['name']}):")
    print(f"  {'Station':<50s} {'Actual':>6s} {'Pred':>6s} {'Err':>7s}")
    order = np.argsort(-abs_errors)
    for idx in order[:15]:
        nm = str(feat.iloc[idx]["station_name"])[:50]
        print(f"  {nm:<50s} {y[idx]:6.1f} {best_preds[idx]:6.1f} {errors[idx]:+7.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  WRITE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
md_path = os.path.join(OUT_DIR, "baseline_model_v3.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Baseline Model V3 — AOD Spatial + RF Fine-Mode Features\n\n")
    f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"**Target**: station mean PM2.5 (40 stations, LOO-CV)\n")
    f.write(f"**Feature pool**: {len(usable)} LOSO-safe features "
            f"({len(usable)-24} new vs v2)\n\n")

    if missing:
        f.write(f"**Note**: {', '.join(missing)} not in unified dataset "
                f"(only center-pixel RF available, 14.9% coverage)\n\n")

    f.write("## Feature Correlations with Station Mean PM2.5\n\n")
    f.write("| Feature | Pearson | Spearman | New? |\n")
    f.write("|---------|:---:|:---:|:---:|\n")
    for _, cr in corr_df.iterrows():
        tag = "NEW" if cr["new"] else ""
        f.write(f"| {cr['feature']} | {cr['pearson']:+.4f} | "
                f"{cr['spearman']:+.4f} | {tag} |\n")

    f.write("\n## All Results\n\n")
    f.write("| # | Model | R² | MAE |\n")
    f.write("|---|-------|:---:|:---:|\n")
    for i, (_, r) in enumerate(res_df.iterrows()):
        f.write(f"| {i+1} | {r['name']} | {r['r2']:.4f} | {r['mae']:.2f} |\n")

    f.write("\n## Best Subsets\n\n")
    for k in [5, 6, 7, 8]:
        bs = best_subsets[k]
        has_new = [x for x in bs["features"] if x in new_feat_cols]
        new_tag = f" (new: {', '.join(has_new)})" if has_new else ""
        f.write(f"**Best-{k}**: {', '.join(bs['features'])}{new_tag}\n")
        f.write(f"  R²={bs['r2']:.4f}, MAE={bs['mae']:.2f}\n\n")

    f.write("## v2 Best-5 + Individual New Features\n\n")
    f.write("| Added Feature | R² | Δ vs Best-5 |\n")
    f.write("|---------------|:---:|:---:|\n")
    base_r2 = [r for r in results if r["name"] == "v2 Best-5 Ridge"][0]["r2"]
    for nf in new_feat_cols:
        match = [r for r in results if r["name"] == f"Best-5 + {nf}"]
        if match:
            r = match[0]
            f.write(f"| {nf} | {r['r2']:.4f} | {r['r2']-base_r2:+.4f} |\n")

    if best_preds is not None:
        f.write(f"\n## Per-Station Errors — {best_model['name']}\n\n")
        f.write("| Station | Region | Actual | Predicted | Error |\n")
        f.write("|---------|--------|:---:|:---:|:---:|\n")
        order = np.argsort(-abs_errors)
        for idx in order:
            row = feat.iloc[idx]
            nm = str(row["station_name"])[:45]
            f.write(f"| {nm} | {row['region']} | {y[idx]:.1f} | "
                    f"{best_preds[idx]:.1f} | {errors[idx]:+.1f} |\n")

    f.write("\n## Conclusion\n\n")
    best = res_df.iloc[0]
    f.write(f"- v2 ceiling (Best-5 Ridge): R²=0.658\n")
    f.write(f"- v3 best model: **{best['name']}** (R²={best['r2']:.4f})\n")
    f.write(f"- Improvement: {best['r2']-0.658:+.4f}\n")

print(f"\nSaved: {md_path}")
print(f"Done — {time.time()-t0:.0f}s total")
