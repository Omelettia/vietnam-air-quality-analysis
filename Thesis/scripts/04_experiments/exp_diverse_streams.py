"""
Diverse-stream experiment: train 5 genuinely different XGBoost models
with different feature subsets, then measure oracle ceiling vs deployable routing.

Streams:
  dispersion  — meteorology + terrain + temporal (no satellite, no gas, no RFSI)
  satellite   — Himawari AOD + RF/SSA + minimal met (no gas, no RFSI)
  emission    — TROPOMI gas + emission proxies + building (no AOD, no RFSI)
  spatial     — RFSI + basic met + temporal (no AOD, no gas)
  full        — everything combined

Uses gbtree (not DART) for stability. Himawari AOD only.
"""

import argparse, io, sys, os, warnings, time, glob, zipfile, unicodedata, math
from unicodedata import normalize as _unorm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None)
parser.add_argument("--resume", action="store_true",
                    help="Resume from saved per-fold checkpoints")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def _repo_root():
    """Walk up to repo root (dir containing data/merged) so this runs from anywhere."""
    p = SCRIPT_DIR
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "data", "merged")):
            return p
        p = os.path.dirname(p)
    return os.path.dirname(SCRIPT_DIR)
REPO_DIR = _repo_root()
DATA_DIR = args.data_dir or REPO_DIR
META_DIR = os.path.join(DATA_DIR, "data", "stations", "metadata")
OUT_DIR = os.path.join(REPO_DIR, "analysis", "experimental_shape_magnitude",
                       "diverse_streams")
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

# ============================================================================
#  HELPERS
# ============================================================================
def ascii_norm(s):
    return _unorm("NFKD", str(s)).encode("ascii", "ignore").decode().lower()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))

def assign_tier(mean_pm):
    if mean_pm < 10: return "t0"
    elif mean_pm < 20: return "t1"
    elif mean_pm < 35: return "t2"
    return "t3"

def safe_r2(y, p):
    if len(y) < 3 or np.std(y) < 1e-9:
        return np.nan
    return float(r2_score(y, p))

def pm_class(x):
    if x < 10: return "low"
    if x < 20: return "moderate_low"
    if x < 35: return "moderate"
    return "high"

# ============================================================================
#  1. LOAD DATA
# ============================================================================
print("=" * 80)
print("DIVERSE STREAMS EXPERIMENT (5 feature-subset XGBoost models)")
print("=" * 80)
t0_wall = time.time()

df = pd.read_csv(os.path.join(DATA_DIR, "data/merged/unified_thesis_v4.csv"),
                 dtype={"stationId": str})  # v4 = definitive (all 40 stations, stronger mask)
# v4 holds all 121 stations; restrict to the 40 thesis stations for training.
_thesis40 = set(pd.read_csv(os.path.join(DATA_DIR,
    "Thesis/results/01_stations/station_selection_final.csv"),
    dtype={"stationId": str})["stationId"])
df = df[df["stationId"].isin(_thesis40)].copy()
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["month"] = df["ts"].dt.month
df["date"] = df["ts"].dt.date
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations")

meta_path = os.path.join(DATA_DIR,
    "Thesis/results/01_stations/station_selection_final.csv")
meta = pd.read_csv(meta_path, dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))
sid_lat = dict(zip(meta["stationId"], meta["lat"]))
sid_lon = dict(zip(meta["stationId"], meta["lon"]))
station_ids = sorted(df["stationId"].unique())
n_stn = len(station_ids)

y_all = df["PM2.5"].values
stationId_vals = df["stationId"].values

qc_masks = pm25_quality_masks(df)
df.loc[qc_masks.any(axis=1), "PM2.5"] = np.nan
y_all = df["PM2.5"].values
y_log = np.log1p(np.nan_to_num(y_all, nan=0.0))

station_pm_means = df.groupby("stationId")["PM2.5"].mean()
sid_tier = {s: assign_tier(station_pm_means[s]) for s in station_ids}
for t in ["t0", "t1", "t2", "t3"]:
    n_t = sum(1 for s in station_ids if sid_tier[s] == t)
    print(f"  {t}: {n_t} stations")

global_pm_mean = float(np.nanmean(y_all))
bm_global = np.log1p(global_pm_mean)
y_res = y_log - bm_global

# ============================================================================
#  2-10. FEATURE PIPELINE (shared single source of truth)
# ============================================================================
from diverse_features import build_diverse_features
df, STREAMS, compute_rfsi, compute_lagged_rfsi = build_diverse_features(
    df, meta, station_ids, data_dir=DATA_DIR, meta_dir=META_DIR, k_nn=K_NN)
# build_diverse_features preserves row order; refresh row-aligned arrays.
stationId_vals = df["stationId"].values
y_all = df["PM2.5"].values
STREAM_NAMES = list(STREAMS.keys())
print(f"  Feature pipeline built: {len(STREAM_NAMES)} streams")
print(f"Data loading complete ({time.time()-t0_wall:.0f}s)")

# ============================================================================
#  11. LOSO TRAINING
# ============================================================================
print(f"\n{'='*80}")
print(f"LOSO TRAINING: {len(STREAM_NAMES)} streams x {n_stn} folds")
print(f"{'='*80}")

oof_preds = {s: np.full(len(df), np.nan) for s in STREAM_NAMES}
fold_checkpoint = os.path.join(OUT_DIR, "fold_checkpoint.csv")
completed_folds = set()

if args.resume and os.path.exists(fold_checkpoint):
    ck = pd.read_csv(fold_checkpoint)
    completed_folds = set(zip(ck["stream"], ck["station_id"]))
    print(f"  Resuming: {len(completed_folds)} fold-stream pairs done")

    oof_path = os.path.join(OUT_DIR, "oof_predictions_partial.csv")
    if os.path.exists(oof_path):
        partial = pd.read_csv(oof_path, dtype={"stationId": str})
        for sn in STREAM_NAMES:
            col = f"pred_{sn}"
            if col in partial.columns:
                vals = partial[col].values
                mask = ~np.isnan(vals)
                oof_preds[sn][mask] = vals[mask]
        print(f"  Loaded partial OOF predictions")

checkpoint_rows = []
fold_times = []

for fi, held_sid in enumerate(station_ids):
    held_name = sid_name.get(held_sid, "?")
    fold_t0 = time.time()

    # Check which streams need training for this fold
    needed = [s for s in STREAM_NAMES if (s, held_sid) not in completed_folds]
    if not needed:
        continue

    # Masks
    held_mask = stationId_vals == held_sid
    train_mask = ~held_mask & ~np.isnan(y_all)
    test_idx = np.where(held_mask)[0]
    train_idx = np.where(train_mask)[0]

    if len(test_idx) == 0 or len(train_idx) == 0:
        continue

    y_train_log = y_res[train_idx]
    y_train_raw = y_all[train_idx].copy()
    y_train_raw = np.nan_to_num(y_train_raw, nan=0.0)

    # Compute RFSI for this fold (needed for spatial + full streams)
    needs_rfsi = any(s.replace("raw_", "") in ["spatial", "full"] for s in needed)
    if needs_rfsi:
        rfsi_vals = compute_rfsi(exclude_sid=held_sid)
        lag_vals = compute_lagged_rfsi(exclude_sid=held_sid)
        for col_name, vals in {**rfsi_vals, **lag_vals}.items():
            df[col_name] = vals

    for sn in needed:
        feats = STREAMS[sn]
        X_train = df.iloc[train_idx][feats].values.astype(np.float32)
        X_test = df.iloc[test_idx][feats].values.astype(np.float32)

        nan_mask_tr = np.isnan(X_train)
        if nan_mask_tr.any():
            col_medians = np.nanmedian(X_train, axis=0)
            for c in range(X_train.shape[1]):
                X_train[nan_mask_tr[:, c], c] = col_medians[c]
            nan_mask_te = np.isnan(X_test)
            for c in range(X_test.shape[1]):
                X_test[nan_mask_te[:, c], c] = col_medians[c]

        is_raw = sn.startswith("raw_")
        y_tr = y_train_raw if is_raw else y_train_log
        y_eval = y_all[test_idx].copy() if is_raw else y_res[test_idx]
        if is_raw:
            y_eval = np.nan_to_num(y_eval, nan=0.0)

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_tr,
                  eval_set=[(X_test, y_eval)],
                  verbose=False)

        raw_pred = model.predict(X_test)
        if is_raw:
            pred_pm = np.clip(raw_pred, 0.1, 300)
        else:
            pred_pm = np.expm1(raw_pred + bm_global).clip(0.1, 300)
        oof_preds[sn][test_idx] = pred_pm

        checkpoint_rows.append({"stream": sn, "station_id": held_sid})

    elapsed = time.time() - fold_t0
    fold_times.append(elapsed)
    avg = np.mean(fold_times)
    remaining = (n_stn - fi - 1) * avg / 60

    # Quick per-fold summary
    brief = []
    for sn in needed:
        y_test = y_all[test_idx]
        p_test = oof_preds[sn][test_idx]
        valid = ~np.isnan(y_test) & ~np.isnan(p_test)
        if valid.sum() >= 3:
            r2 = safe_r2(y_test[valid], p_test[valid])
            brief.append(f"{sn[:3]}={r2:+.2f}")
    print(f"  [{fi+1:2d}/{n_stn}] {held_name[:30]:<30s} "
          f"{elapsed:.0f}s  {' '.join(brief)}  "
          f"(~{remaining:.0f}m left)")

    # Save checkpoint every 5 folds
    if (fi + 1) % 5 == 0 or fi == n_stn - 1:
        ck_df = pd.DataFrame(checkpoint_rows)
        ck_df.to_csv(fold_checkpoint, index=False)
        partial_df = df[["stationId", "ts", "PM2.5"]].copy()
        for sn in STREAM_NAMES:
            partial_df[f"pred_{sn}"] = oof_preds[sn]
        partial_df.to_csv(os.path.join(OUT_DIR, "oof_predictions_partial.csv"),
                          index=False, encoding="utf-8-sig")

# ============================================================================
#  12. EVALUATION
# ============================================================================
print(f"\n{'='*80}")
print("EVALUATION")
print(f"{'='*80}")

results_df = df[["stationId", "ts", "PM2.5"]].copy()
for sn in STREAM_NAMES:
    results_df[f"pred_{sn}"] = oof_preds[sn]
results_df.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"),
                  index=False, encoding="utf-8-sig")

# Per-station metrics
valid_mask = ~np.isnan(y_all)

station_metrics = []
for sid in station_ids:
    sm = stationId_vals == sid
    sm_valid = sm & valid_mask
    if sm_valid.sum() < 10:
        continue
    y_s = y_all[sm_valid]
    actual_mean = float(np.mean(y_s))
    tier = sid_tier[sid]
    row = {"station_id": sid, "station_name": sid_name.get(sid, ""),
           "tier": tier, "actual_mean": actual_mean,
           "actual_class": pm_class(actual_mean), "n_rows": int(sm_valid.sum())}

    stream_r2s = {}
    for sn in STREAM_NAMES:
        p_s = oof_preds[sn][sm_valid]
        p_valid = ~np.isnan(p_s)
        if p_valid.sum() >= 3:
            r2 = safe_r2(y_s[p_valid], p_s[p_valid])
            pred_mean = float(np.mean(p_s[p_valid]))
        else:
            r2 = np.nan
            pred_mean = np.nan
        stream_r2s[sn] = r2
        row[f"r2_{sn}"] = r2
        row[f"pred_mean_{sn}"] = pred_mean

    # Oracle: best stream per station
    valid_r2s = {k: v for k, v in stream_r2s.items() if not np.isnan(v)}
    if valid_r2s:
        best_stream = max(valid_r2s, key=valid_r2s.get)
        row["oracle_stream"] = best_stream
        row["oracle_r2"] = valid_r2s[best_stream]
    else:
        row["oracle_stream"] = "full"
        row["oracle_r2"] = np.nan

    station_metrics.append(row)

met_df = pd.DataFrame(station_metrics)
met_df.to_csv(os.path.join(OUT_DIR, "station_metrics.csv"),
              index=False, encoding="utf-8-sig")

# Summary table
print(f"\n{'stream':<12s}  {'mean_stn':>8s}  {'med_stn':>8s}  {'pooled':>8s}  "
      f"{'pos':>4s}  {'h2nh':>5s}")
print("-" * 55)

summary_rows = []
for sn in STREAM_NAMES + ["oracle"]:
    if sn == "oracle":
        # Build oracle predictions
        oracle_pred = np.full(len(df), np.nan)
        for _, row in met_df.iterrows():
            sid = row["station_id"]
            best = row["oracle_stream"]
            sm = stationId_vals == sid
            oracle_pred[sm] = oof_preds[best][sm]
        pred_col = oracle_pred
    else:
        pred_col = oof_preds[sn]

    vm = valid_mask & ~np.isnan(pred_col)
    if vm.sum() < 10:
        continue

    y_v = y_all[vm]
    p_v = pred_col[vm]

    stn_r2s = []
    h2nh = 0
    for sid in station_ids:
        sm = (stationId_vals == sid) & vm
        if sm.sum() < 3: continue
        r2 = safe_r2(y_all[sm], pred_col[sm])
        stn_r2s.append(r2)
        am = float(np.mean(y_all[sm]))
        pm_ = float(np.mean(pred_col[sm]))
        if am >= 35 and pm_ < 35: h2nh += 1

    row = {
        "stream": sn,
        "mean_station_r2": float(np.nanmean(stn_r2s)),
        "median_station_r2": float(np.nanmedian(stn_r2s)),
        "pooled_r2": safe_r2(y_v, p_v),
        "positive_stations": sum(1 for x in stn_r2s if x > 0),
        "high_to_nonhigh": h2nh,
        "n_stations": len(stn_r2s),
    }
    summary_rows.append(row)
    print(f"  {sn:<12s}  {row['mean_station_r2']:>+8.4f}  "
          f"{row['median_station_r2']:>+8.4f}  {row['pooled_r2']:>8.4f}  "
          f"{row['positive_stations']:>4d}  {h2nh:>5d}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "summary.csv"),
                  index=False, encoding="utf-8-sig")

# Oracle stream distribution
print(f"\nOracle stream choices:")
for sn in STREAM_NAMES:
    n = int((met_df["oracle_stream"] == sn).sum())
    if n > 0:
        print(f"  {sn:<12s}: {n} stations")

# Selector gap
oracle_mean_r2 = float(met_df["oracle_r2"].mean())
full_mean_r2 = float(met_df["r2_full"].mean())
print(f"\nSelector gap: oracle={oracle_mean_r2:+.4f}, "
      f"full={full_mean_r2:+.4f}, "
      f"gap={oracle_mean_r2 - full_mean_r2:.4f}")

# Per-tier breakdown
print(f"\nPer-tier breakdown:")
print(f"  {'tier':<4s}  {'n':>3s}  " + "  ".join(f"{s[:5]:>7s}" for s in STREAM_NAMES) + f"  {'oracle':>7s}")
for t in ["t0", "t1", "t2", "t3"]:
    tm = met_df["tier"] == t
    if tm.sum() == 0: continue
    vals = [f"{met_df.loc[tm, f'r2_{s}'].mean():+.3f}" for s in STREAM_NAMES]
    orc = f"{met_df.loc[tm, 'oracle_r2'].mean():+.3f}"
    print(f"  {t:<4s}  {int(tm.sum()):>3d}  " + "  ".join(f"{v:>7s}" for v in vals) + f"  {orc:>7s}")

print(f"\nTotal time: {(time.time()-t0_wall)/60:.1f} minutes")
print(f"Output: {OUT_DIR}")
