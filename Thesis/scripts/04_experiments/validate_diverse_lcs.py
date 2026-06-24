"""
External LCS validation of the diverse-streams pipeline.

Uses unified_thesis_v4.csv (121 stations, row-level QC).
Trains diverse XGB models on ALL thesis stations (no LOSO).
Evaluates on LCS stations (external) and extra KK stations.

Also runs internal LOSO on thesis stations for comparison.
"""
import argparse, io, sys, os, warnings, time, glob, zipfile, math
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from unicodedata import normalize as _unorm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--skip-loso", action="store_true",
                    help="Skip internal LOSO, only run external validation")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def _repo_root():
    p = SCRIPT_DIR
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "data", "merged")):
            return p
        p = os.path.dirname(p)
    return os.path.dirname(SCRIPT_DIR)
REPO_DIR = _repo_root()
DATA_DIR = REPO_DIR
META_DIR = os.path.join(DATA_DIR, "data", "stations", "metadata")
OUT_DIR = os.path.join(REPO_DIR, "analysis", "thesis_audit")
os.makedirs(OUT_DIR, exist_ok=True)

QC_DIR = os.path.join(REPO_DIR, "Thesis", "scripts", "02_processing")
if QC_DIR not in sys.path:
    sys.path.insert(0, QC_DIR)
from pm25_qc import pm25_quality_masks

SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}

XGB_PARAMS = dict(
    booster="gbtree",
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=40,
    reg_alpha=0.1, reg_lambda=8.0, tree_method="hist",
    device="cuda", random_state=42, n_jobs=-1,
)

K_NN = 5
KNN_SELECT_K = 3
KM_SCALE = 60.0
LAG_HOURS = [1, 3, 6]  # for the (parity-only) thesis-anchored lagged RFSI functions

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))

def safe_r2(y, p):
    if len(y) < 3 or np.std(y) < 1e-9:
        return np.nan
    return float(r2_score(y, p))

# ============================================================================
#  1. LOAD DATA
# ============================================================================
print("=" * 80)
print("DIVERSE STREAMS — EXTERNAL LCS VALIDATION (unified_thesis_v4)")
print("=" * 80)
t0 = time.time()

df = pd.read_csv(os.path.join(DATA_DIR, "data/merged/unified_thesis_v4.csv"),
                 dtype={"stationId": str})
df["ts"] = pd.to_datetime(df["ts"])
df["month"] = df["ts"].dt.month
df["date"] = df["ts"].dt.date
print(f"Loaded v4: {len(df):,} rows, {df['stationId'].nunique()} stations")

thesis_meta = pd.read_csv(
    os.path.join(DATA_DIR, "Thesis/results/01_stations/station_selection_final.csv"),
    dtype={"stationId": str})
THESIS_SIDS = set(thesis_meta["stationId"])

env_meta = pd.read_csv(os.path.join(META_DIR, "envisoft_station_map.csv"),
                        dtype={"stationId": str})
sid_lat = dict(zip(env_meta["stationId"], env_meta["latitude"]))
sid_lon = dict(zip(env_meta["stationId"], env_meta["longitude"]))

all_sids = sorted(df["stationId"].unique())
thesis_sids = sorted([s for s in all_sids if s in THESIS_SIDS])
lcs_sids = sorted([s for s in all_sids
                    if s not in THESIS_SIDS and
                    df.loc[df["stationId"]==s, "station_type"].iloc[0] == "LCS"])
extra_kk_sids = sorted([s for s in all_sids
                         if s not in THESIS_SIDS and s not in set(lcs_sids)])

print(f"  Thesis: {len(thesis_sids)}, LCS: {len(lcs_sids)}, Extra KK: {len(extra_kk_sids)}")

stationId_vals = df["stationId"].values

# ============================================================================
#  2-9. FEATURE PIPELINE (shared single source of truth = diverse_features.py)
# ============================================================================
from diverse_features import build_diverse_features
_meta_all = pd.DataFrame({
    "stationId": all_sids,
    "lat": [sid_lat.get(s, np.nan) for s in all_sids],
    "lon": [sid_lon.get(s, np.nan) for s in all_sids],
})
df, STREAMS, _crf, _clrf = build_diverse_features(
    df, _meta_all, all_sids, data_dir=DATA_DIR, meta_dir=META_DIR, k_nn=K_NN)
STREAM_NAMES = list(STREAMS.keys())
print(f"  Feature pipeline built (shared module): {len(STREAM_NAMES)} streams")
# NOTE: RFSI is filtered out of every stream by the STREAMS df-column filter (it is
# added per-fold AFTER the filter) -- same as the trainer. The thesis-anchored RFSI
# functions below are kept for parity but do not enter any stream.

# Targets / row-aligned arrays (build_diverse_features preserves row order).
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
stationId_vals = df["stationId"].values
y_all = df["PM2.5"].values
global_pm_mean = float(np.nanmean(y_all))
bm_global = np.log1p(global_pm_mean)
y_log = np.log1p(y_all)
y_res = y_log - bm_global
print(f"After dropping PM2.5 NaN: {len(df):,} rows, {df['stationId'].nunique()} stations")

# ============================================================================
#  10. RFSI SETUP (thesis stations provide PM2.5)
# ============================================================================
print("\n--- RFSI setup (thesis stations as reference) ---")

thesis_coords = {s: (sid_lat[s], sid_lon[s]) for s in thesis_sids}
thesis_idx_map = {s: i for i, s in enumerate(thesis_sids)}
n_thesis = len(thesis_sids)

pm25_wide = df[df["stationId"].isin(set(thesis_sids))].pivot_table(
    index="ts", columns="stationId", values="PM2.5", aggfunc="first")
pm25_mat = pm25_wide.values
sid_cols = list(pm25_wide.columns)
sid_to_col = {s: i for i, s in enumerate(sid_cols)}
ts_to_row = pd.Series(range(len(pm25_wide)), index=pm25_wide.index)
df["ts_row"] = df["ts"].map(ts_to_row).values
n_ts = pm25_mat.shape[0]

def compute_rfsi_for_stations(target_sids, exclude_sid=None):
    n = len(df)
    pm_nn = np.full((n, K_NN), np.nan)
    d_nn = np.full((n, K_NN), np.nan)

    for sid in target_sids:
        mask = stationId_vals == sid
        if not mask.any(): continue
        ri = np.where(mask)[0]
        tr = df["ts_row"].values[ri]

        lat_s, lon_s = sid_lat[sid], sid_lon[sid]
        dists = []
        for ts in thesis_sids:
            if ts == exclude_sid:
                continue
            d = haversine(lat_s, lon_s, sid_lat[ts], sid_lon[ts])
            if ts in sid_to_col:
                dists.append((sid_to_col[ts], d))
        dists.sort(key=lambda x: x[1])

        if not dists: continue
        ccols = np.array([c for c, _ in dists])
        cdists = np.array([d for _, d in dists])

        valid_tr = ~np.isnan(tr)
        tr_int = np.where(valid_tr, tr.astype(int), 0)
        tr_int = np.clip(tr_int, 0, n_ts - 1)

        nbr = pm25_mat[np.ix_(tr_int, ccols)]
        nbr[~valid_tr[:, None].repeat(len(ccols), axis=1)] = np.nan

        valid = ~np.isnan(nbr)
        cumv = np.cumsum(valid, axis=1)
        for k in range(K_NN):
            reached = cumv >= (k + 1)
            has = reached.any(axis=1)
            if not has.any(): break
            pos = np.argmax(reached, axis=1)
            ih = np.where(has)[0]
            pm_nn[ri[ih], k] = nbr[ih, pos[has]]
            d_nn[ri[ih], k] = cdists[pos[has]]

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d_nn
        pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)
    return {"PM25_nn_idw": pm_idw, "PM25_nn1": pm_nn[:,0], "dist_nn1": d_nn[:,0],
            "PM25_nn2": pm_nn[:,1], "PM25_nn3": pm_nn[:,2]}

def compute_lagged_rfsi_for_stations(target_sids, exclude_sid=None):
    n = len(df)
    lags = {lh: np.full(n, np.nan) for lh in LAG_HOURS}

    for sid in target_sids:
        mask = stationId_vals == sid
        if not mask.any(): continue
        ri = np.where(mask)[0]
        tr = df["ts_row"].values[ri]

        lat_s, lon_s = sid_lat[sid], sid_lon[sid]
        dists = []
        for ts in thesis_sids:
            if ts == exclude_sid:
                continue
            d = haversine(lat_s, lon_s, sid_lat[ts], sid_lon[ts])
            if ts in sid_to_col:
                dists.append((sid_to_col[ts], d))
        dists.sort(key=lambda x: x[1])
        if not dists: continue

        nn1_col = dists[0][0]
        for lag_h in LAG_HOURS:
            tr_lag = tr - lag_h
            valid = ~np.isnan(tr_lag)
            tr_safe = np.clip(np.where(valid, tr_lag, 0).astype(int), 0, n_ts - 1)
            in_bounds = valid & (tr_lag >= 0)
            vals = pm25_mat[tr_safe, nn1_col]
            vals[~in_bounds] = np.nan
            lags[lag_h][ri] = vals

    return {f"PM25_nn1_lag{lh}h": lags[lh] for lh in LAG_HOURS}


# ============================================================================
#  11. SPATIAL PRIOR
# ============================================================================
print("Computing spatial prior...")
spatial_prior = np.full(len(df), np.nan)
thesis_pm_means = df[df["stationId"].isin(set(thesis_sids))].groupby("stationId")["PM2.5"].mean()

for sid in all_sids:
    mask = stationId_vals == sid
    if not mask.any(): continue
    lat_s, lon_s = sid_lat[sid], sid_lon[sid]
    dists = {}
    for ts in thesis_sids:
        if ts == sid: continue
        dists[ts] = haversine(lat_s, lon_s, sid_lat[ts], sid_lon[ts])
    nearest = sorted(dists.items(), key=lambda x: x[1])[:10]
    weights = np.array([np.exp(-(d/KM_SCALE)**2) for _, d in nearest])
    means = np.array([thesis_pm_means.get(s, np.nan) for s, _ in nearest])
    valid = ~np.isnan(means)
    if valid.sum() > 0:
        sp = np.average(means[valid], weights=weights[valid])
    else:
        sp = global_pm_mean
    spatial_prior[mask] = sp

df["spatial_prior"] = spatial_prior


# ============================================================================
#  12. TRAIN DIVERSE MODELS ON ALL THESIS DATA
# ============================================================================
print(f"\n{'='*80}")
print("TRAINING DIVERSE MODELS ON ALL THESIS STATIONS")
print(f"{'='*80}")

thesis_mask = df["stationId"].isin(set(thesis_sids))
df_thesis = df[thesis_mask].copy()

# Compute RFSI for thesis (using all thesis stations, no exclusion)
print("  Computing RFSI for thesis (full)...")
rfsi_out = compute_rfsi_for_stations(thesis_sids)
lag_out = compute_lagged_rfsi_for_stations(thesis_sids)
for k, v in {**rfsi_out, **lag_out}.items():
    df[k] = v

y_thesis = df_thesis["PM2.5"].values
y_thesis_log = np.log1p(y_thesis)
y_thesis_res = y_thesis_log - bm_global

models = {}
for stream_name in STREAM_NAMES:
    feats = STREAMS[stream_name]
    is_raw = stream_name.startswith("raw_")
    X_train = df.loc[thesis_mask, feats].values
    y_train = y_thesis if is_raw else y_thesis_res

    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_train, y_train)
    models[stream_name] = m
    print(f"  {stream_name:<16s}: {len(feats)} features — trained")

print(f"  {len(models)} models trained ({time.time()-t0:.0f}s)")


# ============================================================================
#  13. INTERNAL LOSO (thesis stations)
# ============================================================================
if not args.skip_loso:
    print(f"\n{'='*80}")
    print(f"INTERNAL LOSO ({len(thesis_sids)} thesis stations)")
    print(f"{'='*80}")

    oof_preds = {s: np.full(len(df), np.nan) for s in STREAM_NAMES}
    thesis_indices = np.where(thesis_mask)[0]

    for fi, leave_sid in enumerate(thesis_sids):
        t_fold = time.time()
        train_mask_fold = thesis_mask & (df["stationId"] != leave_sid)
        test_mask_fold = df["stationId"] == leave_sid

        # Recompute RFSI excluding leave-out station
        rfsi_fold = compute_rfsi_for_stations([leave_sid], exclude_sid=leave_sid)
        lag_fold = compute_lagged_rfsi_for_stations([leave_sid], exclude_sid=leave_sid)
        for k, v in {**rfsi_fold, **lag_fold}.items():
            test_idx = np.where(test_mask_fold)[0]
            df.loc[test_mask_fold, k] = v[test_idx]

        for stream_name in STREAM_NAMES:
            feats = STREAMS[stream_name]
            is_raw = stream_name.startswith("raw_")
            X_tr = df.loc[train_mask_fold, feats].values
            X_te = df.loc[test_mask_fold, feats].values
            y_tr = y_all[train_mask_fold] if is_raw else y_res[train_mask_fold]

            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            oof_preds[stream_name][test_mask_fold] = pred

        elapsed = time.time() - t_fold
        if (fi + 1) % 5 == 0 or fi == 0:
            print(f"  Fold {fi+1:2d}/{len(thesis_sids)} ({leave_sid[:8]}...) {elapsed:.1f}s")

    # Evaluate LOSO
    print("\n  LOSO Results (thesis stations):")
    print(f"  {'Stream':<20s} {'mean_r2':>8s} {'pool_r2':>8s}")
    print("  " + "-" * 40)

    for stream_name in STREAM_NAMES:
        preds = oof_preds[stream_name]
        is_raw = stream_name.startswith("raw_")
        per_stn_r2 = []
        all_y, all_p = [], []
        for sid in thesis_sids:
            mask = stationId_vals == sid
            y_s = y_all[mask]
            p_raw = preds[mask]
            valid = ~np.isnan(p_raw) & ~np.isnan(y_s)
            if valid.sum() < 10: continue
            if not is_raw:
                p_pm = np.expm1(p_raw + bm_global)
            else:
                p_pm = p_raw
            r2_s = safe_r2(y_s[valid], p_pm[valid])
            if not np.isnan(r2_s):
                per_stn_r2.append(r2_s)
            all_y.extend(y_s[valid])
            all_p.extend(p_pm[valid])

        mean_r2 = np.mean(per_stn_r2) if per_stn_r2 else np.nan
        pool_r2 = safe_r2(np.array(all_y), np.array(all_p)) if all_y else np.nan
        print(f"  {stream_name:<20s} {mean_r2:+.4f}   {pool_r2:+.4f}")


# ============================================================================
#  14. EXTERNAL LCS VALIDATION
# ============================================================================
print(f"\n{'='*80}")
print(f"EXTERNAL VALIDATION ({len(lcs_sids)} LCS stations)")
print(f"{'='*80}")

# Compute RFSI for LCS using all thesis stations
print("  Computing RFSI for LCS stations...")
rfsi_lcs = compute_rfsi_for_stations(lcs_sids)
lag_lcs = compute_lagged_rfsi_for_stations(lcs_sids)
for k, v in {**rfsi_lcs, **lag_lcs}.items():
    lcs_mask_all = df["stationId"].isin(set(lcs_sids))
    lcs_idx = np.where(lcs_mask_all)[0]
    df.loc[lcs_mask_all, k] = v[lcs_idx]

# Generate predictions
lcs_mask = df["stationId"].isin(set(lcs_sids))
print(f"  LCS rows: {lcs_mask.sum():,}")

lcs_preds = {}
for stream_name in STREAM_NAMES:
    feats = STREAMS[stream_name]
    is_raw = stream_name.startswith("raw_")
    X_lcs = df.loc[lcs_mask, feats].values
    pred = models[stream_name].predict(X_lcs)
    if not is_raw:
        pred_pm = np.expm1(pred + bm_global)
    else:
        pred_pm = pred
    lcs_preds[stream_name] = pred_pm

# Per-stream evaluation
print(f"\n  Per-stream LCS metrics:")
print(f"  {'Stream':<24s} {'mean_r2':>8s} {'pool_r2':>8s} {'pos':>4s}")
print("  " + "-" * 50)

y_lcs = df.loc[lcs_mask, "PM2.5"].values
sid_lcs = df.loc[lcs_mask, "stationId"].values
stream_station_r2 = {}

for stream_name in STREAM_NAMES:
    pred_pm = lcs_preds[stream_name]
    per_stn = []
    all_y, all_p = [], []
    for sid in lcs_sids:
        m = sid_lcs == sid
        y_s, p_s = y_lcs[m], pred_pm[m]
        valid = ~np.isnan(y_s) & ~np.isnan(p_s)
        if valid.sum() < 10: continue
        r2_s = safe_r2(y_s[valid], p_s[valid])
        per_stn.append((sid, r2_s))
        all_y.extend(y_s[valid])
        all_p.extend(p_s[valid])

    mean_r2 = np.mean([r for _, r in per_stn if not np.isnan(r)]) if per_stn else np.nan
    pool_r2 = safe_r2(np.array(all_y), np.array(all_p)) if all_y else np.nan
    n_pos = sum(1 for _, r in per_stn if r > 0)
    print(f"  {stream_name:<24s} {mean_r2:+.4f}   {pool_r2:+.4f}   {n_pos:>3d}")
    stream_station_r2[stream_name] = dict(per_stn)


# ============================================================================
#  15. KNN SELECTOR + BLEND
# ============================================================================
print(f"\n--- kNN stream selector (k={KNN_SELECT_K}) ---")

# Build thesis OOF R² per stream (from LOSO if available, otherwise use train R²)
thesis_stream_r2 = {}
if not args.skip_loso:
    for stream_name in STREAM_NAMES:
        preds = oof_preds[stream_name]
        is_raw = stream_name.startswith("raw_")
        for sid in thesis_sids:
            mask = stationId_vals == sid
            y_s = y_all[mask]
            p_raw = preds[mask]
            valid = ~np.isnan(p_raw) & ~np.isnan(y_s)
            if valid.sum() < 10: continue
            if not is_raw:
                p_pm = np.expm1(p_raw + bm_global)
            else:
                p_pm = p_raw
            r2_s = safe_r2(y_s[valid], p_pm[valid])
            thesis_stream_r2[(sid, stream_name)] = r2_s if not np.isnan(r2_s) else -1.0
else:
    for stream_name in STREAM_NAMES:
        feats = STREAMS[stream_name]
        is_raw = stream_name.startswith("raw_")
        for sid in thesis_sids:
            mask_s = thesis_mask & (df["stationId"] == sid)
            X_s = df.loc[mask_s, feats].values
            y_s = y_all[mask_s] if is_raw else y_res[mask_s]
            pred = models[stream_name].predict(X_s)
            if not is_raw:
                pred_pm = np.expm1(pred + bm_global)
                y_pm = y_all[mask_s]
            else:
                pred_pm, y_pm = pred, y_s
            r2_s = safe_r2(y_pm, pred_pm)
            thesis_stream_r2[(sid, stream_name)] = r2_s if not np.isnan(r2_s) else -1.0

# For each LCS station, find k nearest thesis stations and pick best stream
lcs_selected_stream = {}
for sid in lcs_sids:
    lat_s, lon_s = sid_lat[sid], sid_lon[sid]
    dists = [(ts, haversine(lat_s, lon_s, sid_lat[ts], sid_lon[ts]))
             for ts in thesis_sids]
    dists.sort(key=lambda x: x[1])
    nearest_k = dists[:KNN_SELECT_K]

    best_stream, best_r2 = STREAM_NAMES[0], -999
    for stream_name in STREAM_NAMES:
        avg_r2 = np.mean([thesis_stream_r2.get((ts, stream_name), -1.0)
                          for ts, _ in nearest_k])
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_stream = stream_name
    lcs_selected_stream[sid] = best_stream

# Build kNN-selected predictions
knn_selected_pred = np.full(lcs_mask.sum(), np.nan)
sp_vals = df.loc[lcs_mask, "spatial_prior"].values
sid_lcs_arr = sid_lcs

for sid in lcs_sids:
    m = sid_lcs_arr == sid
    stream = lcs_selected_stream[sid]
    knn_selected_pred[m] = lcs_preds[stream][m]

# Blend variants
print(f"\n  Evaluation variants:")
print(f"  {'Variant':<40s} {'mean_r2':>8s} {'pool_r2':>8s} {'pos':>4s}")
print("  " + "-" * 65)

def eval_variant(name, pred_arr):
    per_stn = []
    all_y, all_p = [], []
    for sid in lcs_sids:
        m = sid_lcs_arr == sid
        y_s = y_lcs[m]
        p_s = pred_arr[m]
        valid = ~np.isnan(y_s) & ~np.isnan(p_s)
        if valid.sum() < 10: continue
        r2_s = safe_r2(y_s[valid], p_s[valid])
        per_stn.append(r2_s)
        all_y.extend(y_s[valid])
        all_p.extend(p_s[valid])
    mean_r2 = np.mean([r for r in per_stn if not np.isnan(r)]) if per_stn else np.nan
    pool_r2 = safe_r2(np.array(all_y), np.array(all_p)) if all_y else np.nan
    n_pos = sum(1 for r in per_stn if r > 0)
    print(f"  {name:<40s} {mean_r2:+.4f}   {pool_r2:+.4f}   {n_pos:>3d}")
    return mean_r2, pool_r2

eval_variant("knn_selected_raw", knn_selected_pred)

for alpha in [0.35, 0.45, 0.55, 0.65]:
    blend = alpha * knn_selected_pred + (1 - alpha) * sp_vals
    eval_variant(f"blend_a{int(alpha*100)}", blend)

# Spatial prior + shift baseline
shift_vals = [30, 45, 60]
for s in shift_vals:
    shifted = sp_vals + s
    eval_variant(f"spatial_prior_shift_{s}", shifted)

# Stream selection breakdown
from collections import Counter
sel_counts = Counter(lcs_selected_stream.values())
print(f"\n  kNN selection breakdown:")
for stream, cnt in sel_counts.most_common():
    print(f"    {stream:<24s}: {cnt} stations")

# ============================================================================
#  DONE
# ============================================================================
elapsed = time.time() - t0
print(f"\n{'='*80}")
print(f"DONE — {elapsed:.0f}s total")
print(f"{'='*80}")
