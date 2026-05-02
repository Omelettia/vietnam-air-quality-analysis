"""
Baseline model v3b: Test RF and SSA spatial features in Stage 1 baseline.

Adds per-station means of RF_mean, RF_std, RF_center, RF_inner_mean,
RF_outer_mean, RF_grad_mag, RF_local_vs_regional, SSA_mean, SSA_std,
SSA_center, SSA_inner_mean, SSA_outer_mean, SSA_grad_mag,
SSA_local_vs_regional, AOT_fine from unified_thesis_v2.csv.

Reports Spearman correlations, then runs exhaustive Ridge LOO-CV
best-k search (k=5..7, conditionally k=8).
"""

import io, sys, os, warnings, time, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(REPO_DIR, "analysis", "thesis_experiments")

V2_PATH = os.path.join(REPO_DIR, "data", "merged", "unified_thesis_v2.csv")
FEAT_PATH = os.path.join(OUT_DIR, "station_baseline_features.csv")

print("=" * 70)
print("BASELINE MODEL V3b — RF & SSA SPATIAL FEATURES")
print("=" * 70)
t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD + COMPUTE NEW STATION-LEVEL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

NEW_HOURLY_COLS = [
    "RF_mean", "RF_std", "RF_center", "RF_inner_mean", "RF_outer_mean",
    "RF_grad_mag", "RF_local_vs_regional",
    "SSA_mean", "SSA_std", "SSA_center", "SSA_inner_mean", "SSA_outer_mean",
    "SSA_grad_mag", "SSA_local_vs_regional",
    "AOT_fine",
]

print(f"Loading unified v2: {V2_PATH}")
use_cols = ["stationId"] + NEW_HOURLY_COLS
df = pd.read_csv(V2_PATH, usecols=use_cols)
print(f"  Loaded: {len(df):,} rows ({time.time()-t0:.1f}s)")

station_means = df.groupby("stationId")[NEW_HOURLY_COLS].mean()
station_means.columns = ["mean_" + c for c in station_means.columns]
station_means = station_means.reset_index()
station_means["stationId"] = station_means["stationId"].astype(str)

new_feat_cols = list(station_means.columns[1:])
print(f"  New station-level features: {len(new_feat_cols)}")
for c in new_feat_cols:
    nn = station_means[c].notna().sum()
    mu = station_means[c].mean()
    sd = station_means[c].std()
    print(f"    {c}: {nn}/40 non-NaN, mean={mu:.4f}, std={sd:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  MERGE WITH EXISTING FEATURE TABLE
# ═══════════════════════════════════════════════════════════════════════════════

feat = pd.read_csv(FEAT_PATH)
feat["station_id"] = feat["station_id"].astype(str)
feat = feat.merge(station_means, left_on="station_id", right_on="stationId", how="left")
feat.drop(columns=["stationId"], inplace=True)
feat.to_csv(FEAT_PATH, index=False)
print(f"\nUpdated {FEAT_PATH}: {feat.shape[1]} columns")

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
    "mean_AOT_inner_mean", "mean_AOT_outer_mean",
    "mean_AOT_spatial_std", "mean_AOT_local_vs_regional",
    "mean_AOT_grad_mag",
]

LOSO_SAFE_V3B = PREV_LOSO_SAFE + new_feat_cols

usable = []
for c in LOSO_SAFE_V3B:
    if c not in feat.columns:
        print(f"  Missing: {c}")
        continue
    nna = feat[c].notna().sum()
    if nna >= 30:
        usable.append(c)
    else:
        print(f"  Dropping {c}: only {nna} non-NaN")
print(f"\nUsable LOSO-safe features: {len(usable)} (was 29, now +{len(usable)-29})")


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


results = []
def log(name, r2, mae, preds):
    results.append({"name": name, "r2": r2, "mae": mae})
    print(f"  {name:60s} R²={r2:.4f}  MAE={mae:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SPEARMAN CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("NEW RF/SSA FEATURE CORRELATIONS WITH STATION MEAN PM2.5")
print(f"{'='*70}")

corr_rows = []
for c in new_feat_cols:
    vals = feat[c].values
    mask = ~np.isnan(vals) & ~np.isnan(y)
    if mask.sum() < 10:
        continue
    pr, _ = pearsonr(vals[mask], y[mask])
    sr, _ = spearmanr(vals[mask], y[mask])
    corr_rows.append({"feature": c, "pearson": pr, "spearman": sr, "n": mask.sum()})

ref_feat = "mean_AOT_grad_mag"
if ref_feat in feat.columns:
    vals = feat[ref_feat].values
    mask = ~np.isnan(vals) & ~np.isnan(y)
    pr, _ = pearsonr(vals[mask], y[mask])
    sr, _ = spearmanr(vals[mask], y[mask])
    corr_rows.append({"feature": f"{ref_feat} (reference)", "pearson": pr, "spearman": sr, "n": mask.sum()})

corr_df = pd.DataFrame(corr_rows).sort_values("spearman", ascending=False, key=abs)
print(f"\n  {'Feature':40s} {'Pearson':>10s} {'Spearman':>10s}  {'n':>4s}")
print(f"  {'-'*40} {'-'*10} {'-'*10}  {'-'*4}")
for _, row in corr_df.iterrows():
    print(f"  {row['feature']:40s} {row['pearson']:+10.4f} {row['spearman']:+10.4f}  {row['n']:4.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("BASELINES")
print(f"{'='*70}")

V3_BEST8 = ["mean_PM25_nn_idw", "AOT_p95", "AOT_valid_frac", "mean_WS",
            "mean_VC", "mean_Temp", "slope_deg", "mean_AOT_grad_mag"]
r2, mae, p = ridge_loo(scale(prepare_X(V3_BEST8)), y)
log("v3 Best-8 Ridge (baseline)", r2, mae, p)

# ═══════════════════════════════════════════════════════════════════════════════
#  BEST SUBSETS k=5,6,7 (conditionally k=8)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("BEST SUBSETS (k=5,6,7)")
print(f"{'='*70}")

Xall_scaled = scale(prepare_X(usable))
best_subsets = {}
any_new_enters = False
any_beats_704 = False

for k in [5, 6, 7]:
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
        print(f"    NEW RF/SSA features in subset: {has_new}")
        any_new_enters = True
    if best_r2 > 0.704:
        any_beats_704 = True

# Conditionally run k=8
if any_new_enters and any_beats_704:
    print(f"\n  New feature entered AND R² > 0.704 → running k=8 ...")
    k = 8
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
        print(f"    NEW RF/SSA features in subset: {has_new}")
else:
    reasons = []
    if not any_new_enters:
        reasons.append("no new feature entered any best subset")
    if not any_beats_704:
        reasons.append("no R² > 0.704")
    print(f"\n  Skipping k=8: {'; '.join(reasons)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

sorted_results = sorted(results, key=lambda x: x["r2"], reverse=True)
for i, r in enumerate(sorted_results):
    print(f"  {i+1:2d}. {r['name']:60s} R²={r['r2']:.4f}  MAE={r['mae']:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════════

md_path = os.path.join(OUT_DIR, "baseline_model_v3_aod.md")
with open(md_path, "r", encoding="utf-8") as f:
    existing = f.read()

lines = ["\n\n---\n"]
lines.append("## V3b — RF & SSA Spatial Features\n")
lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n")
lines.append(f"**New features tested**: {len(new_feat_cols)} station-level RF/SSA/AOT_fine means\n")

lines.append("\n### Spearman Correlations with Station Mean PM2.5\n")
lines.append("| Feature | Pearson | Spearman |")
lines.append("|---------|:---:|:---:|")
for _, row in corr_df.iterrows():
    lines.append(f"| {row['feature']} | {row['pearson']:+.4f} | {row['spearman']:+.4f} |")

lines.append("\n### Best Subsets\n")
for k in sorted(best_subsets.keys()):
    bs = best_subsets[k]
    has_new = [f for f in bs["features"] if f in new_feat_cols]
    new_str = f" (new: {', '.join(has_new)})" if has_new else ""
    lines.append(f"**Best-{k}**: {', '.join(bs['features'])}{new_str}")
    lines.append(f"  Ridge R²={bs['r2']:.4f}, MAE={bs['mae']:.2f}\n")

if not any_new_enters:
    lines.append("**Result**: No new RF/SSA feature entered any best subset. The v3 Best-8 (R²=0.704) remains the ceiling.\n")
elif not any_beats_704:
    lines.append("**Result**: New features entered but did not beat v3 Best-8 R²=0.704.\n")
else:
    lines.append("**Result**: New features entered AND beat v3 Best-8 R²=0.704.\n")

with open(md_path, "w", encoding="utf-8") as f:
    f.write(existing + "\n".join(lines))

print(f"\nAppended to: {md_path}")
elapsed = time.time() - t0
print(f"Done — {elapsed:.0f}s total")
