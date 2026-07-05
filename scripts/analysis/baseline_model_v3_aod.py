"""
Baseline model v3: Add AOD spatial features to LOSO-safe set.

Computes per-station means of AOT_inner_mean, AOT_outer_mean,
AOT_spatial_std, AOT_local_vs_regional, AOT_grad_mag from unified dataset.
Reruns best-subset search (k=5..8) with Ridge and RF LOO-CV.

Output: analysis/thesis_experiments/baseline_model_v3_aod.md
        analysis/thesis_experiments/station_baseline_features.csv (updated)
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
print("BASELINE MODEL V3 — AOD SPATIAL FEATURES")
print("=" * 70)
t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD + COMPUTE AOD SPATIAL AVERAGES
# ═══════════════════════════════════════════════════════════════════════════════
feat = pd.read_csv(os.path.join(OUT_DIR, "station_baseline_features.csv"),
                    dtype={"station_id": str})

AOD_SPATIAL_COLS = ["AOT_inner_mean", "AOT_outer_mean", "AOT_spatial_std",
                    "AOT_local_vs_regional", "AOT_grad_mag"]

print("Loading unified dataset for AOD spatial features ...")
t1 = time.time()
df = pd.read_csv(os.path.join(REPO_DIR, "data/merged/unified_thesis_v1.csv"),
                  dtype={"stationId": str},
                  usecols=["stationId", "PM2.5"] + AOD_SPATIAL_COLS)
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
print(f"Loaded: {len(df):,} rows ({time.time()-t1:.1f}s)")

aod_agg = df.groupby("stationId")[AOD_SPATIAL_COLS].agg(["mean", "std"])
aod_agg.columns = [f"{col}_{stat}" for col, stat in aod_agg.columns]

new_feat_cols = []
for col in AOD_SPATIAL_COLS:
    mean_col = f"mean_{col}"
    new_feat_cols.append(mean_col)
    mapping = aod_agg[f"{col}_mean"]
    feat[mean_col] = feat["station_id"].map(mapping)
    nna = feat[mean_col].notna().sum()
    print(f"  {mean_col}: {nna}/40 non-NaN, "
          f"mean={feat[mean_col].mean():.4f}, std={feat[mean_col].std():.4f}")

# Also add AOT_valid_frac weighted versions? No, keep it simple - just means.

# Update the CSV
feat.to_csv(os.path.join(OUT_DIR, "station_baseline_features.csv"),
            index=False, encoding="utf-8-sig")
print(f"Updated station_baseline_features.csv: {len(feat.columns)} columns")

# ═══════════════════════════════════════════════════════════════════════════════
#  SETUP
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

LOSO_SAFE_V3 = PREV_LOSO_SAFE + new_feat_cols

# Filter usable
usable = []
for c in LOSO_SAFE_V3:
    if c not in feat.columns:
        print(f"  Missing: {c}")
        continue
    nna = feat[c].notna().sum()
    if nna >= 30:
        usable.append(c)
    else:
        print(f"  Dropping {c}: only {nna} non-NaN")
print(f"\nUsable LOSO-safe features: {len(usable)} (was 24, now +{len(usable)-24})")

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

# ═══════════════════════════════════════════════════════════════════════════════
#  CORRELATIONS — NEW FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("NEW AOD SPATIAL FEATURE CORRELATIONS")
print(f"{'='*70}")

print(f"\n  {'Feature':<30s} {'Pearson→PM2.5':>14s} {'Spearman→PM2.5':>15s}")
print(f"  {'-'*30} {'-'*14} {'-'*15}")

corr_rows = []
for c in usable:
    vals = feat[c].values
    valid = ~np.isnan(vals) & ~np.isnan(y)
    if valid.sum() < 10:
        continue
    pr, _ = sp_stats.pearsonr(vals[valid], y[valid])
    sr, _ = sp_stats.spearmanr(vals[valid], y[valid])
    is_new = c in new_feat_cols
    marker = " ** NEW" if is_new else ""
    if is_new or abs(sr) > 0.5:
        print(f"  {c:<30s} {pr:+.4f}         {sr:+.4f}{marker}")
    corr_rows.append({"feature": c, "pearson": round(pr, 4),
                       "spearman": round(sr, 4), "new": is_new})

corr_df = pd.DataFrame(corr_rows).sort_values("spearman", ascending=False, key=abs)

# ═══════════════════════════════════════════════════════════════════════════════
#  BASELINES — v2 best-5 and all-24
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BASELINES")
print(f"{'='*70}")

results = []

def log(name, r2, mae, preds=None):
    results.append({"name": name, "r2": r2, "mae": mae, "preds": preds})
    print(f"  {name:<60s} R²={r2:.4f}  MAE={mae:.2f}")

BEST5_V2 = ["mean_PM25_nn_idw", "mean_PBLH", "mean_VC", "rain_freq", "slope_deg"]
r2, mae, p = ridge_loo(scale(prepare_X(BEST5_V2)), y)
log("v2 Best-5 Ridge (baseline)", r2, mae, p)

r2, mae, p = ridge_loo(scale(prepare_X(PREV_LOSO_SAFE)), y)
log("v2 All 24 LOSO-safe Ridge", r2, mae, p)

r2, mae, p = ridge_loo(scale(prepare_X(usable)), y)
log(f"v3 All {len(usable)} LOSO-safe Ridge", r2, mae, p)

r2, mae, p = loo_cv(ElasticNet, scale(prepare_X(usable)), y,
                      alpha=0.1, l1_ratio=0.5, max_iter=5000)
log(f"v3 All {len(usable)} LOSO-safe ElasticNet", r2, mae, p)

r2, mae, p = loo_cv(RandomForestRegressor, prepare_X(usable), y,
                      n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
log(f"v3 All {len(usable)} LOSO-safe RF", r2, mae, p)

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
        print(f"    NEW AOD features in subset: {has_new}")

# ═══════════════════════════════════════════════════════════════════════════════
#  RF + ELASTICNET ON BEST SUBSETS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("NONLINEAR ON BEST SUBSETS")
print(f"{'='*70}")

for k in [5, 6, 7, 8]:
    feats = best_subsets[k]["features"]
    Xk = prepare_X(feats)

    r2, mae, p = loo_cv(ElasticNet, scale(Xk), y, alpha=0.1, l1_ratio=0.5, max_iter=5000)
    log(f"Best-{k} ElasticNet", r2, mae, p)

    r2, mae, p = loo_cv(RandomForestRegressor, Xk, y,
                          n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
    log(f"Best-{k} RF", r2, mae, p)

# ═══════════════════════════════════════════════════════════════════════════════
#  DIRECT TEST: ADD EACH NEW FEATURE TO BEST-5
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ADD EACH NEW AOD FEATURE TO v2 BEST-5")
print(f"{'='*70}")

for nf in new_feat_cols:
    cols = BEST5_V2 + [nf]
    r2, mae, p = ridge_loo(scale(prepare_X(cols)), y)
    log(f"Best-5 + {nf}", r2, mae, p)

# All 5 new AOD together
cols_all_new = BEST5_V2 + new_feat_cols
r2, mae, p = ridge_loo(scale(prepare_X(cols_all_new)), y)
log(f"Best-5 + all 5 AOD spatial", r2, mae, p)

r2, mae, p = loo_cv(RandomForestRegressor, prepare_X(cols_all_new), y,
                      n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
log(f"Best-5 + all 5 AOD spatial RF", r2, mae, p)

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
    order = np.argsort(-abs_errors)
    for idx in order[:10]:
        nm = str(feat.iloc[idx]["station_name"])[:50]
        print(f"  {nm:<50s} actual={y[idx]:5.1f} pred={best_preds[idx]:5.1f} "
              f"err={errors[idx]:+6.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  WRITE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
md_path = os.path.join(OUT_DIR, "baseline_model_v3_aod.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Baseline Model V3 — AOD Spatial Features\n\n")
    f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"**Target**: station mean PM2.5 (40 stations, LOO-CV)\n")
    f.write(f"**New features**: per-station means of {', '.join(AOD_SPATIAL_COLS)}\n\n")

    f.write("## New Feature Correlations with Station Mean PM2.5\n\n")
    f.write("| Feature | Pearson | Spearman | New? |\n")
    f.write("|---------|:---:|:---:|:---:|\n")
    for _, cr in corr_df.iterrows():
        tag = "NEW" if cr["new"] else ""
        f.write(f"| {cr['feature']} | {cr['pearson']:+.4f} | "
                f"{cr['spearman']:+.4f} | {tag} |\n")

    f.write("\n## All Results (sorted by R²)\n\n")
    f.write("| # | Model | R² | MAE (µg/m³) |\n")
    f.write("|---|-------|:---:|:---:|\n")
    for i, (_, r) in enumerate(res_df.iterrows()):
        f.write(f"| {i+1} | {r['name']} | {r['r2']:.4f} | {r['mae']:.2f} |\n")

    f.write("\n## Best Subsets\n\n")
    for k in [5, 6, 7, 8]:
        bs = best_subsets[k]
        has_new = [x for x in bs["features"] if x in new_feat_cols]
        new_tag = f" (new: {', '.join(has_new)})" if has_new else " (no new AOD features)"
        f.write(f"**Best-{k}**: {', '.join(bs['features'])}{new_tag}\n")
        f.write(f"  Ridge R²={bs['r2']:.4f}, MAE={bs['mae']:.2f}\n\n")

    f.write("## v2 Best-5 + Individual AOD Features\n\n")
    f.write("| Added Feature | R² | MAE | Δ R² vs Best-5 |\n")
    f.write("|---------------|:---:|:---:|:---:|\n")
    base_r2 = [r for r in results if r["name"] == "v2 Best-5 Ridge (baseline)"][0]["r2"]
    for nf in new_feat_cols:
        match = [r for r in results if r["name"] == f"Best-5 + {nf}"]
        if match:
            r = match[0]
            f.write(f"| {nf} | {r['r2']:.4f} | {r['mae']:.2f} | "
                    f"{r['r2']-base_r2:+.4f} |\n")

    f.write(f"\n## Conclusion\n\n")
    best = res_df.iloc[0]
    f.write(f"Best model: **{best['name']}** (R²={best['r2']:.4f}, MAE={best['mae']:.2f})\n\n")
    f.write(f"v2 baseline (Best-5 Ridge): R²={base_r2:.4f}\n")
    f.write(f"Improvement: {best['r2']-base_r2:+.4f} R²\n")

print(f"\nSaved: {md_path}")
print(f"Done — {time.time()-t0:.0f}s total")
