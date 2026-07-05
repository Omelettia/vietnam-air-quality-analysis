"""
Experiment 12c: Metadata clusters + oracle baseline.

Reuses KM5 cluster assignments from experiment_12.
Oracle baseline: held-out station's actual mean PM2.5 (perfect knowledge).

Configs:
  O1: Per-cluster XGBoost + oracle baseline
  O2: All stations + cluster_id + oracle baseline
  O3: All stations + oracle baseline, no cluster info
  O4: All stations, no cluster, no baseline (pure B1 reference)

Key question: does O1 beat O3? If yes, metadata clustering adds value
on top of a good baseline. If O3 ≈ O1, the model can learn regimes
from features alone.
"""

import argparse, io, sys, os, warnings, time
from datetime import datetime
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
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = args.data_dir or REPO_DIR
OUT_DIR = os.path.join(REPO_DIR, "analysis", "thesis_experiments")
os.makedirs(OUT_DIR, exist_ok=True)

K_NN = 5

XGB_PARAMS = dict(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    device="cuda", random_state=42, n_jobs=-1,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("EXPERIMENT 12c: METADATA CLUSTERS + ORACLE BASELINE")
print("=" * 80)

t0 = time.time()
df = pd.read_csv(os.path.join(DATA_DIR, "data/merged/unified_thesis_v2.csv"),
                 dtype={"stationId": str})
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["month"] = df["ts"].dt.month
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations "
      f"({time.time()-t0:.1f}s)")

meta = pd.read_csv(os.path.join(DATA_DIR,
                    "analysis/thesis_audit/station_selection_final.csv"),
                    dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))
sid_region = dict(zip(meta["stationId"], meta["region"]))
sid_lat = dict(zip(meta["stationId"], meta["lat"]))
sid_lon = dict(zip(meta["stationId"], meta["lon"]))
station_ids = sorted(df["stationId"].unique())
n_stn = len(station_ids)
sid_to_idx = {s: i for i, s in enumerate(station_ids)}

TARGET = "PM2.5"
y_all = df[TARGET].values
stationId_vals = df["stationId"].values

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD CLUSTER ASSIGNMENTS FROM EXP12
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Loading cluster assignments from experiment 12 ---")
clust_path = os.path.join(OUT_DIR, "cluster_assignments_exp12.csv")
if not os.path.exists(clust_path):
    clust_path = os.path.join(DATA_DIR, "analysis/thesis_experiments/cluster_assignments_exp12.csv")
clust_df = pd.read_csv(clust_path, dtype={"station_id": str})

if "cluster_KM5" in clust_df.columns:
    clust_col = "cluster_KM5"
elif "cluster" in clust_df.columns:
    clust_col = "cluster"
else:
    raise ValueError(f"No cluster column found. Columns: {list(clust_df.columns)}")

sid_cluster = dict(zip(clust_df["station_id"].astype(str),
                       clust_df[clust_col].astype(int)))
n_clusters = len(set(sid_cluster.values()))
print(f"Loaded {len(sid_cluster)} station assignments, {n_clusters} clusters "
      f"(column: {clust_col})")

for c in range(n_clusters):
    sids_c = [s for s in station_ids if sid_cluster.get(s) == c]
    pm_c = [df.loc[stationId_vals == s, "PM2.5"].mean() for s in sids_c]
    print(f"  Cluster {c}: {len(sids_c)} stations, "
          f"PM2.5 mean={np.mean(pm_c):.1f}, std={np.std(pm_c):.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  JOIN BUILDING DENSITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Building density ---")
bld_path = os.path.join(DATA_DIR, "data/stations/metadata/station_building_density.csv")
if not os.path.exists(bld_path):
    bld_path = os.path.join(REPO_DIR, "data/stations/metadata/station_building_density.csv")
bld = pd.read_csv(bld_path, dtype={"stationId": str})
BUILDING_COLS = ["building_count_1km", "building_area_1km",
                 "building_count_3km", "building_area_3km"]
bld_map = bld.set_index("stationId")[BUILDING_COLS]
df = df.merge(bld_map, left_on="stationId", right_index=True, how="left")
for col in BUILDING_COLS:
    df[col] = df[col].fillna(0)
print(f"Building density loaded: {len(bld)} stations")

# ═══════════════════════════════════════════════════════════════════════════════
#  STATION DISTANCES + RFSI
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Station distance matrix ---")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

coords = {s: (sid_lat[s], sid_lon[s]) for s in station_ids}
dist_full = np.zeros((n_stn, n_stn))
for i in range(n_stn):
    for j in range(i + 1, n_stn):
        d = haversine(*coords[station_ids[i]], *coords[station_ids[j]])
        dist_full[i, j] = d
        dist_full[j, i] = d

neighbor_order = {}
for i in range(n_stn):
    neighbor_order[i] = sorted(
        [(j, dist_full[i, j]) for j in range(n_stn) if j != i],
        key=lambda x: x[1])

# ═══════════════════════════════════════════════════════════════════════════════
#  PM2.5 WIDE MATRIX + RFSI
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- PM2.5 wide matrix ---")
pm25_wide = df.pivot_table(index="ts", columns="stationId",
                           values="PM2.5", aggfunc="first")
pm25_mat = pm25_wide.values
sid_cols = list(pm25_wide.columns)
sid_to_col = {s: i for i, s in enumerate(sid_cols)}
ts_to_row = pd.Series(range(len(pm25_wide)), index=pm25_wide.index)
df["ts_row"] = df["ts"].map(ts_to_row).astype(int).values
print(f"Shape: {pm25_mat.shape[0]:,} timestamps x {pm25_mat.shape[1]} stations")

RFSI_COLS = ([f"PM25_nn{k+1}" for k in range(K_NN)] +
             [f"dist_nn{k+1}" for k in range(K_NN)] +
             ["n_neighbors_available", "PM25_nn_mean", "PM25_nn_idw"])

def compute_rfsi(exclude_sid=None, K=5):
    n = len(df)
    pm_nn = np.full((n, K), np.nan)
    d_nn = np.full((n, K), np.nan)
    excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
    ts_row_vals = df["ts_row"].values

    for sid in station_ids:
        si = sid_to_idx[sid]
        mask = stationId_vals == sid
        if not mask.any():
            continue
        ri = np.where(mask)[0]
        tr = ts_row_vals[ri]
        cands = [(j, d) for j, d in neighbor_order[si]
                 if excl is None or j != excl]
        if not cands:
            continue
        ccols = np.array([sid_to_col[station_ids[j]] for j, _ in cands])
        cdists = np.array([d for _, d in cands])
        nbr = pm25_mat[np.ix_(tr, ccols)]
        valid = ~np.isnan(nbr)
        cumv = np.cumsum(valid, axis=1)
        for k in range(K):
            reached = cumv >= (k + 1)
            has = reached.any(axis=1)
            if not has.any():
                break
            pos = np.argmax(reached, axis=1)
            ih = np.where(has)[0]
            pm_nn[ri[ih], k] = nbr[ih, pos[has]]
            d_nn[ri[ih], k] = cdists[pos[has]]

    n_avail = np.sum(~np.isnan(pm_nn), axis=1).astype(int)
    pm_mean = np.nanmean(pm_nn, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d_nn
        pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)

    out = {}
    for k in range(K):
        out[f"PM25_nn{k+1}"] = pm_nn[:, k]
        out[f"dist_nn{k+1}"] = d_nn[:, k]
    out["n_neighbors_available"] = n_avail
    out["PM25_nn_mean"] = pm_mean
    out["PM25_nn_idw"] = pm_idw
    return out

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════════

TEMPORAL = ["hour_sin", "hour_cos", "month_sin", "month_cos",
            "day_of_year_sin", "day_of_year_cos"]
MET = [
    "Temperature_final", "Humidity_final", "Pressure_final",
    "PBLH", "VC", "RH_factor",
    "wind_u", "wind_v", "wind_dir_sin", "wind_dir_cos",
    "WS_local", "wind_u_local", "wind_v_local",
    "wind_dir_sin_local", "wind_dir_cos_local",
    "dT_6h", "dRH_6h", "dWS_6h", "dP_6h",
    "precip_mm", "hrs_since_rain", "rain_sum_24h", "rain_sum_48h",
    "rain_days_7d", "consecutive_dry_days",
]
AOD = [
    "AOT", "AOT_mean", "AOT_inner_mean", "AOT_outer_mean",
    "RF", "SSA", "Uncertainty", "AE",
    "AOT_valid_count", "AOD_physics", "AOT_spatial_std", "AOT_local_vs_regional",
    "AOT_ffill_48h", "hours_since_valid_AOT",
    "AOT_lag_1h", "AOT_lag_3h", "AOT_lag_6h",
    "AOT_rolling_mean_6h", "AOT_rolling_mean_24h",
    "AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_grad_dir",
]
GEO = ["latitude", "longitude", "elevation_m", "slope_deg",
       "aspect_sin", "aspect_cos", "elev_x_PBLH", "elev_x_hour_sin"]

B1_FEATS = MET + AOD + GEO + TEMPORAL + RFSI_COLS + BUILDING_COLS

B1_avail = [f for f in B1_FEATS
            if f in df.columns or f in RFSI_COLS]
missing = [f for f in B1_FEATS if f not in B1_avail]
if missing:
    print(f"WARNING: B1 missing columns: {missing}")
B1_FEATS = B1_avail

print(f"\n  B1 features: {len(B1_FEATS)}")

# Precompute station mean PM2.5 (used for oracle baseline)
station_pm_means = df.groupby("stationId")["PM2.5"].mean()

# Drop global RFSI before LOSO — recomputed per fold
df.drop(columns=[c for c in RFSI_COLS if c in df.columns],
        inplace=True, errors="ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOSO: ORACLE BASELINE CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("LOSO: ORACLE BASELINE CONFIGS")
print(f"{'='*80}")
print("""
  O1: Per-cluster XGBoost + oracle baseline
  O2: All stations + cluster_id + oracle baseline
  O3: All stations + oracle baseline, no cluster
  O4: All stations, no cluster, no baseline (pure B1)
""")

# Pre-extract base features for fast LOSO assembly
all_base_feats = sorted(set(
    f for f in B1_FEATS if f not in RFSI_COLS))
base_arr = df[all_base_feats].values
base_col_map = {f: i for i, f in enumerate(all_base_feats)}
rfsi_col_map = {f: i for i, f in enumerate(RFSI_COLS)}

b_feats = [f for f in B1_FEATS if f not in RFSI_COLS]
r_feats = [f for f in B1_FEATS if f in RFSI_COLS]
b_idx = [base_col_map[f] for f in b_feats]
r_idx = [rfsi_col_map[f] for f in r_feats]

CONFIG_ORDER = ["O1", "O2", "O3", "O4"]
loso_results = {cn: [] for cn in CONFIG_ORDER}

for fold_i, held_sid in enumerate(station_ids):
    nm = sid_name.get(held_sid, held_sid)[:45]
    rg = sid_region.get(held_sid, "?")
    held_cluster = sid_cluster.get(held_sid, -1)
    mask_test = stationId_vals == held_sid
    n_test = mask_test.sum()
    if n_test < 10:
        print(f"  [{fold_i+1:2d}/{n_stn}] {nm:45s} | SKIP (n={n_test})")
        continue

    y_test = y_all[mask_test]
    t_fold = time.time()

    # ── RFSI excluding held-out ──
    train_sids = [s for s in station_ids if s != held_sid]
    rfsi_fold = compute_rfsi(exclude_sid=held_sid, K=K_NN)
    rfsi_arr = np.column_stack([rfsi_fold[c] for c in RFSI_COLS])

    # ── Oracle baseline: actual station mean PM2.5 ──
    oracle_val = station_pm_means[held_sid]
    train_means = df.loc[~mask_test].groupby("stationId")["PM2.5"].mean()

    # ── Assemble B1 feature matrix (no baseline) ──
    X_b1 = np.hstack([base_arr[:, b_idx], rfsi_arr[:, r_idx]])

    # ── Baseline column: oracle for held-out, actual mean for train ──
    bl_col = np.zeros(len(df))
    for s in train_sids:
        bl_col[stationId_vals == s] = train_means[s]
    bl_col[mask_test] = oracle_val

    # ── X with baseline appended ──
    X_with_bl = np.hstack([X_b1, bl_col.reshape(-1, 1)])

    # ── Cluster column ──
    cluster_col = np.array([sid_cluster.get(s, -1) for s in stationId_vals])

    X_te_b1 = X_b1[mask_test]
    X_te_bl = X_with_bl[mask_test]

    parts = [f"[{fold_i+1:2d}/{n_stn}] {nm:45s} | cl={held_cluster}"
             f" oracle={oracle_val:5.1f} |"]

    for cname in CONFIG_ORDER:
        if cname == "O1":
            same_cluster_sids = set(
                s for s in train_sids if sid_cluster.get(s) == held_cluster)
            mask_train = np.array([
                (stationId_vals[i] in same_cluster_sids)
                for i in range(len(df))], dtype=bool)
            n_train_stations = len(same_cluster_sids)
            X_tr = X_with_bl[mask_train]
            y_train = y_all[mask_train]
            X_te = X_te_bl

        elif cname == "O2":
            mask_train = ~mask_test
            X_tr = np.hstack([X_with_bl[mask_train],
                              cluster_col[mask_train].reshape(-1, 1)])
            X_te = np.hstack([X_te_bl,
                              np.full((n_test, 1), held_cluster)])
            y_train = y_all[mask_train]

        elif cname == "O3":
            mask_train = ~mask_test
            X_tr = X_with_bl[mask_train]
            y_train = y_all[mask_train]
            X_te = X_te_bl

        elif cname == "O4":
            mask_train = ~mask_test
            X_tr = X_b1[mask_train]
            y_train = y_all[mask_train]
            X_te = X_te_b1

        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr, y_train)
        pm_pred = m.predict(X_te)

        r2 = r2_score(y_test, pm_pred)
        rmse = np.sqrt(mean_squared_error(y_test, pm_pred))
        mae = mean_absolute_error(y_test, pm_pred)

        extra = {}
        if cname == "O1":
            extra["n_train_stations"] = n_train_stations

        loso_results[cname].append(dict(
            station_id=held_sid,
            station_name=sid_name.get(held_sid, held_sid),
            region=rg, n_rows=n_test, cluster=held_cluster,
            r2=round(r2, 4), rmse=round(rmse, 2), mae=round(mae, 2),
            oracle_baseline=round(oracle_val, 2),
            **extra))

        parts.append(f"{cname}={r2:+.3f}")

    fold_time = time.time() - t_fold
    remaining = fold_time * (n_stn - fold_i - 1)
    print(f"  {' '.join(parts)}  [{rg}] ({fold_time:.0f}s, ETA {remaining/60:.0f}m)")

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE RAW RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
all_loso_rows = []
for cfg in CONFIG_ORDER:
    for r in loso_results[cfg]:
        all_loso_rows.append({"config": cfg, **r})
pd.DataFrame(all_loso_rows).to_csv(
    os.path.join(OUT_DIR, "loso_per_station_exp12c.csv"),
    index=False, encoding="utf-8-sig")
print(f"\nSaved: loso_per_station_exp12c.csv")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD PREVIOUS RESULTS FOR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
prev_loso = {}
for exp_csv in ["loso_per_station_exp07.csv",
                "loso_per_station_exp11.csv",
                "loso_per_station_exp12.csv"]:
    lp = os.path.join(OUT_DIR, exp_csv)
    if os.path.exists(lp):
        tmp = pd.read_csv(lp, dtype={"station_id": str})
        for cfg in tmp["config"].unique():
            prev_loso[cfg] = tmp[tmp["config"] == cfg].to_dict("records")
        print(f"Loaded previous: {exp_csv} ({len(tmp['config'].unique())} configs)")

def loso_summary(results):
    rdf = pd.DataFrame(results)
    v = rdf.dropna(subset=["r2"])
    return dict(
        mean_r2=round(v["r2"].mean(), 4),
        median_r2=round(v["r2"].median(), 4),
        wmean_r2=round((v["r2"]*v["n_rows"]).sum()/v["n_rows"].sum(), 4),
        mean_rmse=round(v["rmse"].mean(), 2),
        mean_mae=round(v["mae"].mean(), 2),
        neg_count=int((v["r2"] < 0).sum()))

sums = {c: loso_summary(loso_results[c]) for c in CONFIG_ORDER}
prev_sums = {c: loso_summary(recs) for c, recs in prev_loso.items()}

# ═══════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

for cn in CONFIG_ORDER:
    s = sums[cn]
    print(f"  {cn}: R²={s['mean_r2']:.4f} (median={s['median_r2']:.4f}), "
          f"MAE={s['mean_mae']:.2f}, neg={s['neg_count']}")

for cn in ["B1", "V2", "C1", "C2", "C3"]:
    if cn in prev_sums:
        s = prev_sums[cn]
        print(f"  {cn} (ref): R²={s['mean_r2']:.4f} (median={s['median_r2']:.4f}), "
              f"MAE={s['mean_mae']:.2f}")

print(f"\n  Key comparison:")
print(f"    O1 (cluster + oracle) vs O3 (no cluster + oracle): "
      f"ΔR²={sums['O1']['mean_r2'] - sums['O3']['mean_r2']:+.4f}")
print(f"    O3 (oracle) vs O4 (no baseline): "
      f"ΔR²={sums['O3']['mean_r2'] - sums['O4']['mean_r2']:+.4f}")
if "C1" in prev_sums:
    print(f"    O1 (oracle) vs C1 (sat baseline): "
          f"ΔR²={sums['O1']['mean_r2'] - prev_sums['C1']['mean_r2']:+.4f}")
if "C3" in prev_sums:
    print(f"    O3 (oracle) vs C3 (sat baseline): "
          f"ΔR²={sums['O3']['mean_r2'] - prev_sums['C3']['mean_r2']:+.4f}")

# Per-cluster breakdown
print(f"\n  Per-cluster breakdown:")
for cname in CONFIG_ORDER:
    rdf = pd.DataFrame(loso_results[cname])
    parts = []
    for c in range(n_clusters):
        c_df = rdf[rdf["cluster"] == c]
        if len(c_df) > 0:
            parts.append(f"cl{c}={c_df['r2'].mean():.4f}({len(c_df)})")
    print(f"    {cname}: {', '.join(parts)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("GENERATING REPORT")
print(f"{'='*80}")

rpt = []
rpt.append("# Experiment 12c: Metadata Clusters + Oracle Baseline\n")
rpt.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rpt.append(f"**Dataset:** unified_thesis_v2.csv — {len(df):,} rows, {n_stn} stations")
rpt.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, "
           f"lr=0.05, device=cuda")
rpt.append(f"**Clusters:** experiment_12 KM5 ({n_clusters} clusters)\n")

rpt.append("## Design\n")
rpt.append("Oracle baseline: for each held-out station, `baseline_oracle` = the station's "
           "actual mean PM2.5 (computed from its own data). This is deliberately cheating — "
           "it tests the ceiling: how well can the model do if it knows the station's mean "
           "pollution level perfectly?\n")
rpt.append("| Config | Description |")
rpt.append("|--------|-------------|")
rpt.append("| O1 | Per-cluster XGBoost + oracle baseline |")
rpt.append("| O2 | All stations + cluster_id feature + oracle baseline |")
rpt.append("| O3 | All stations + oracle baseline, no cluster info |")
rpt.append("| O4 | All stations, no cluster, no baseline (pure B1) |\n")

rpt.append("## Results\n")
rpt.append("| Config | Description | LOSO R² (mean) | LOSO R² (median) | "
           "LOSO MAE | Neg Stations |")
rpt.append("|--------|-------------|:---:|:---:|:---:|:---:|")

ref_cfgs = [
    ("B1", "Exp07: B1 reference"),
    ("V2", "Exp11: B1 + sat baseline"),
    ("C1", "Exp12: per-cluster + sat baseline"),
    ("C2", "Exp12: all + cluster_id + sat baseline"),
    ("C3", "Exp12: all + sat baseline"),
]
for cn, desc in ref_cfgs:
    if cn in prev_sums:
        s = prev_sums[cn]
        rpt.append(f"| {cn} (ref) | {desc} | {s['mean_r2']:.4f} | "
                   f"{s['median_r2']:.4f} | {s['mean_mae']:.2f} | {s['neg_count']} |")

descs = {
    "O1": "Per-cluster + oracle baseline",
    "O2": "All + cluster_id + oracle baseline",
    "O3": "All + oracle baseline",
    "O4": "All, no cluster, no baseline (B1)",
}
for cn in CONFIG_ORDER:
    s = sums[cn]
    rpt.append(f"| **{cn}** | {descs[cn]} | **{s['mean_r2']:.4f}** | "
               f"{s['median_r2']:.4f} | {s['mean_mae']:.2f} | {s['neg_count']} |")

rpt.append(f"\n### Per-Cluster Breakdown\n")
for cname in CONFIG_ORDER:
    rdf = pd.DataFrame(loso_results[cname])
    rpt.append(f"**{cname}** ({descs[cname]}):\n")
    rpt.append("| Cluster | Stations | Mean R² | Mean MAE |")
    rpt.append("|:---:|:---:|:---:|:---:|")
    for c in range(n_clusters):
        c_df = rdf[rdf["cluster"] == c]
        if len(c_df) == 0:
            continue
        rpt.append(f"| {c} | {len(c_df)} | {c_df['r2'].mean():.4f} | "
                   f"{c_df['mae'].mean():.2f} |")
    rpt.append("")

# Per-station comparison table
rpt.append("### Per-Station Comparison\n")
o3_r2 = {r["station_id"]: r["r2"] for r in loso_results["O3"]}
o4_r2 = {r["station_id"]: r["r2"] for r in loso_results["O4"]}
b1_r2 = {r["station_id"]: r["r2"] for r in prev_loso.get("B1", [])}
c1_r2 = {r["station_id"]: r["r2"] for r in prev_loso.get("C1", [])}

rpt.append("| Station | Region | Cl | B1 R² | C1 R² | O1 R² | O3 R² | O4 R² |")
rpt.append("|---------|--------|:--:|:---:|:---:|:---:|:---:|:---:|")
o1_df = pd.DataFrame(loso_results["O1"]).sort_values("r2", ascending=True)
for _, r in o1_df.iterrows():
    sid = r["station_id"]
    nm2 = str(r["station_name"])[:35]
    b1v = b1_r2.get(sid, np.nan)
    c1v = c1_r2.get(sid, np.nan)
    o3v = o3_r2.get(sid, np.nan)
    o4v = o4_r2.get(sid, np.nan)
    rpt.append(f"| {nm2} | {r['region']} | {r['cluster']} | "
               f"{b1v:.4f} | "
               f"{f'{c1v:.4f}' if pd.notna(c1v) else '—'} | "
               f"{r['r2']:.4f} | "
               f"{f'{o3v:.4f}' if pd.notna(o3v) else '—'} | "
               f"{f'{o4v:.4f}' if pd.notna(o4v) else '—'} |")

rpt.append("\n## Key Comparisons\n")
rpt.append(f"- **O1 vs O3** (does clustering help on top of oracle baseline?): "
           f"ΔR² = {sums['O1']['mean_r2'] - sums['O3']['mean_r2']:+.4f}")
rpt.append(f"- **O3 vs O4** (oracle baseline value): "
           f"ΔR² = {sums['O3']['mean_r2'] - sums['O4']['mean_r2']:+.4f}")
if "C1" in prev_sums:
    rpt.append(f"- **O1 vs C1** (oracle vs satellite baseline, per-cluster): "
               f"ΔR² = {sums['O1']['mean_r2'] - prev_sums['C1']['mean_r2']:+.4f}")
if "C3" in prev_sums:
    rpt.append(f"- **O3 vs C3** (oracle vs satellite baseline, all stations): "
               f"ΔR² = {sums['O3']['mean_r2'] - prev_sums['C3']['mean_r2']:+.4f}")

rpt.append("\n## Interpretation\n")
o1_r2_val = sums["O1"]["mean_r2"]
o3_r2_val = sums["O3"]["mean_r2"]
diff = o1_r2_val - o3_r2_val
if abs(diff) < 0.01:
    rpt.append(f"O1 ≈ O3 (ΔR²={diff:+.4f}): metadata clustering does NOT add meaningful "
               f"value on top of a good station-level baseline. The XGBoost model can "
               f"implicitly learn pollution regimes from the hourly features alone, "
               f"once it knows the station's mean pollution level.")
elif diff > 0.01:
    rpt.append(f"O1 > O3 (ΔR²={diff:+.4f}): metadata clustering adds value even on top "
               f"of a perfect baseline. Per-cluster training captures regime-specific "
               f"patterns that a single global model misses.")
else:
    rpt.append(f"O1 < O3 (ΔR²={diff:+.4f}): clustering hurts — per-cluster training "
               f"reduces training data without adding signal. A single global model "
               f"with the oracle baseline performs better.")

report_path = os.path.join(OUT_DIR, "experiment_12c_oracle_baseline.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))

print(f"\nReport: {report_path}")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
