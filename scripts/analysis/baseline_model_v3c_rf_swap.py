"""
Baseline model v3c: RF swap test + satellite-only ceiling.

1. Swap IDW → RF features in v3 Best-8
2. Add RF features on top of v3 Best-8
3. Fix SSA (filter values > 1), recompute station means
4. Satellite-only best-subset search (no ground-station-derived features)
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

t0 = time.time()
print("=" * 70)
print("BASELINE MODEL V3c — RF SWAP + SATELLITE-ONLY CEILING")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

feat = pd.read_csv(FEAT_PATH)
y = feat["pm25_mean"].values
n = len(y)

def prepare_X(cols, df=None):
    if df is None:
        df = feat
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

V3_BEST8 = ["mean_PM25_nn_idw", "AOT_p95", "AOT_valid_frac", "mean_WS",
            "mean_VC", "mean_Temp", "slope_deg", "mean_AOT_grad_mag"]

r2_base, mae_base, _ = ridge_loo(scale(prepare_X(V3_BEST8)), y)
print(f"\nv3 Best-8 baseline: R²={r2_base:.4f}  MAE={mae_base:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  1. SWAP TEST: replace IDW with RF features
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("1. SWAP TEST — replace mean_PM25_nn_idw with RF feature")
print(f"{'='*70}")

RF_SWAP = ["mean_RF_mean", "mean_RF_inner_mean", "mean_RF_outer_mean",
           "mean_RF_center", "mean_AOT_fine", "mean_RF_grad_mag"]

print(f"\n  {'Swap':40s} {'R²':>8s} {'MAE':>8s} {'ΔR²':>8s}")
print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'v3 Best-8 (with IDW)':40s} {r2_base:8.4f} {mae_base:8.2f} {'---':>8s}")
for rf in RF_SWAP:
    swapped = [rf if f == "mean_PM25_nn_idw" else f for f in V3_BEST8]
    r2, mae, _ = ridge_loo(scale(prepare_X(swapped)), y)
    print(f"  IDW → {rf:33s} {r2:8.4f} {mae:8.2f} {r2-r2_base:+8.4f}")

# Also try removing IDW entirely (7 features)
no_idw = [f for f in V3_BEST8 if f != "mean_PM25_nn_idw"]
r2, mae, _ = ridge_loo(scale(prepare_X(no_idw)), y)
print(f"  {'Drop IDW entirely (7 feats)':40s} {r2:8.4f} {mae:8.2f} {r2-r2_base:+8.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  2. ADD TEST: keep IDW, add RF on top
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("2. ADD TEST — keep IDW in Best-8, add RF features")
print(f"{'='*70}")

print(f"\n  {'Added feature':40s} {'R²':>8s} {'MAE':>8s} {'ΔR²':>8s}")
print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'v3 Best-8 (baseline)':40s} {r2_base:8.4f} {mae_base:8.2f} {'---':>8s}")
for rf in RF_SWAP:
    combo = V3_BEST8 + [rf]
    r2, mae, _ = ridge_loo(scale(prepare_X(combo)), y)
    print(f"  + {rf:37s} {r2:8.4f} {mae:8.2f} {r2-r2_base:+8.4f}")

# All RF features at once
all_rf = ["mean_RF_mean", "mean_RF_std", "mean_RF_center",
          "mean_RF_inner_mean", "mean_RF_outer_mean",
          "mean_RF_grad_mag", "mean_RF_local_vs_regional", "mean_AOT_fine"]
combo_all = V3_BEST8 + all_rf
r2, mae, _ = ridge_loo(scale(prepare_X(combo_all)), y)
print(f"  + {'all 8 RF features':37s} {r2:8.4f} {mae:8.2f} {r2-r2_base:+8.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  3. FIX SSA — filter values > 1, recompute station means
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("3. FIX SSA — recompute with only valid values (0-1)")
print(f"{'='*70}")

print(f"\nLoading unified v2 for SSA fix ...")
ssa_cols = ["stationId", "SSA_mean", "SSA_std", "SSA_center",
            "SSA_inner_mean", "SSA_outer_mean",
            "SSA_grad_mag", "SSA_local_vs_regional"]
df = pd.read_csv(V2_PATH, usecols=ssa_cols)
print(f"  Loaded {len(df):,} rows")

before_nn = {c: df[c].notna().sum() for c in ssa_cols[1:]}
for c in ssa_cols[1:]:
    if "valid_count" in c or "grad" in c or "local" in c:
        continue
    df.loc[~df[c].between(0, 1.1), c] = np.nan

# Recompute grad_mag and local_vs_regional from cleaned values
# grad depends on pixel-level data we don't have here, so just filter
# implausible gradient values too (if inner/outer are 0-1, grad should be small)
df.loc[df["SSA_grad_mag"].abs() > 0.5, "SSA_grad_mag"] = np.nan
df.loc[df["SSA_local_vs_regional"].abs() > 0.5, "SSA_local_vs_regional"] = np.nan

after_nn = {c: df[c].notna().sum() for c in ssa_cols[1:]}
print(f"\n  SSA coverage before/after filtering:")
for c in ssa_cols[1:]:
    print(f"    {c:30s}: {before_nn[c]:>8,} → {after_nn[c]:>8,} ({100*after_nn[c]/len(df):.1f}%)")

ssa_station = df.groupby("stationId")[ssa_cols[1:]].mean().reset_index()
ssa_station.columns = ["stationId"] + ["mean_" + c + "_clean" for c in ssa_cols[1:]]
ssa_station["stationId"] = ssa_station["stationId"].astype(str)

feat["station_id"] = feat["station_id"].astype(str)
# Drop old clean columns if exist
old_clean = [c for c in feat.columns if c.endswith("_clean")]
if old_clean:
    feat.drop(columns=old_clean, inplace=True)
feat = feat.merge(ssa_station, left_on="station_id", right_on="stationId", how="left")
feat.drop(columns=["stationId"], inplace=True)

print(f"\n  Cleaned SSA station-level stats:")
ssa_clean_feats = [c for c in feat.columns if "_clean" in c]
for c in ssa_clean_feats:
    nn = feat[c].notna().sum()
    vals = feat[c].dropna()
    if len(vals) > 0:
        print(f"    {c:40s}: {nn}/40 non-NaN, mean={vals.mean():.4f}, std={vals.std():.4f}")
    else:
        print(f"    {c:40s}: {nn}/40 non-NaN")

# SSA correlations with PM2.5
print(f"\n  Cleaned SSA correlations:")
for c in ssa_clean_feats:
    vals = feat[c].values
    mask = ~np.isnan(vals) & ~np.isnan(y)
    if mask.sum() < 10:
        print(f"    {c:40s}: too few valid ({mask.sum()})")
        continue
    sr, _ = spearmanr(vals[mask], y[mask])
    pr, _ = pearsonr(vals[mask], y[mask])
    print(f"    {c:40s}: Pearson={pr:+.4f}  Spearman={sr:+.4f}")

# Test cleaned SSA in model
print(f"\n  Add cleaned SSA to Best-8:")
for c in ssa_clean_feats:
    if feat[c].notna().sum() < 30:
        continue
    combo = V3_BEST8 + [c]
    r2, mae, _ = ridge_loo(scale(prepare_X(combo)), y)
    print(f"    + {c:36s} R²={r2:.4f}  MAE={mae:.2f}  ΔR²={r2-r2_base:+.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  4. SATELLITE-ONLY BEST SUBSETS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("4. SATELLITE-ONLY BEST SUBSETS (no ground-station features)")
print(f"{'='*70}")

GROUND_DERIVED = {
    "mean_PM25_nn_idw", "std_PM25_nn_idw", "iqr_PM25_nn_idw",
    "nbr_diurnal_range", "nbr_seasonal_amp",
}

SAT_POOL = [
    # AOT spatial
    "ACAG_annual_mean", "median_AOT", "AOT_std", "AOT_p95",
    "AOT_valid_frac", "mean_AOD_PBLH_ratio",
    "mean_AOT_inner_mean", "mean_AOT_outer_mean",
    "mean_AOT_spatial_std", "mean_AOT_local_vs_regional",
    "mean_AOT_grad_mag",
    # RF spatial
    "mean_RF_mean", "mean_RF_std", "mean_RF_center",
    "mean_RF_inner_mean", "mean_RF_outer_mean",
    "mean_RF_grad_mag", "mean_RF_local_vs_regional",
    "mean_AOT_fine",
    # Met (reanalysis, available everywhere)
    "mean_PBLH", "mean_WS", "mean_VC",
    "mean_Temp", "mean_Humidity", "mean_Pressure", "rain_freq",
    # Static
    "building_count_3km", "building_area_3km",
    "elevation_m", "latitude", "longitude", "slope_deg",
]

# Add cleaned SSA if available
for c in ssa_clean_feats:
    if feat[c].notna().sum() >= 30:
        SAT_POOL.append(c)

SAT_POOL = [c for c in SAT_POOL if c in feat.columns and feat[c].notna().sum() >= 30]
print(f"\n  Satellite-only feature pool: {len(SAT_POOL)} features")
print(f"  Excluded ground-derived: {GROUND_DERIVED}")

Xall_sat = scale(prepare_X(SAT_POOL))
best_subsets = {}

for k in [5, 6, 7, 8]:
    combos = list(itertools.combinations(range(len(SAT_POOL)), k))
    print(f"\n  k={k}: {len(combos):,} combinations ...", end="", flush=True)
    t1 = time.time()
    best_r2 = -999
    best_combo = None
    best_preds = None
    for combo in combos:
        r2, mae, preds = ridge_loo(Xall_sat[:, list(combo)], y)
        if r2 > best_r2:
            best_r2 = r2
            best_combo = combo
            best_preds = preds.copy()
    best_feat = [SAT_POOL[i] for i in best_combo]
    best_mae = mean_absolute_error(y, best_preds)
    best_subsets[k] = {"features": best_feat, "r2": best_r2, "mae": best_mae}
    elapsed = time.time() - t1
    rf_in = [f for f in best_feat if "RF" in f or "fine" in f.lower()]
    ssa_in = [f for f in best_feat if "SSA" in f or "clean" in f]
    print(f" {elapsed:.0f}s")
    print(f"    Best-{k} Ridge  R²={best_r2:.4f}  MAE={best_mae:.2f}")
    print(f"    Features: {best_feat}")
    if rf_in:
        print(f"    RF features: {rf_in}")
    if ssa_in:
        print(f"    SSA features: {ssa_in}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

print(f"\n  v3 Best-8 (with IDW):              R²={r2_base:.4f}")
for k in sorted(best_subsets.keys()):
    bs = best_subsets[k]
    print(f"  Satellite-only Best-{k}:             R²={bs['r2']:.4f}  (gap={bs['r2']-r2_base:+.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════════

md_path = os.path.join(OUT_DIR, "baseline_model_v3_aod.md")
with open(md_path, "r", encoding="utf-8") as f:
    existing = f.read()

lines = ["\n\n---\n"]
lines.append("## V3c — RF Swap + Satellite-Only Ceiling\n")
lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n")

lines.append("\n### 1. Swap IDW → RF in v3 Best-8\n")
lines.append("| Configuration | R² | MAE | ΔR² |")
lines.append("|---|:---:|:---:|:---:|")
lines.append(f"| v3 Best-8 (with IDW) | {r2_base:.4f} | {mae_base:.2f} | — |")

lines.append("\n### 4. Satellite-Only Best Subsets\n")
lines.append("| k | R² | MAE | Features |")
lines.append("|---|:---:|:---:|---|")
for k in sorted(best_subsets.keys()):
    bs = best_subsets[k]
    lines.append(f"| {k} | {bs['r2']:.4f} | {bs['mae']:.2f} | {', '.join(bs['features'])} |")

with open(md_path, "w", encoding="utf-8") as f:
    f.write(existing + "\n".join(lines))
print(f"\nAppended to: {md_path}")

feat.to_csv(FEAT_PATH, index=False)
print(f"Updated: {FEAT_PATH}")

elapsed = time.time() - t0
print(f"\nDone — {elapsed:.0f}s total")
