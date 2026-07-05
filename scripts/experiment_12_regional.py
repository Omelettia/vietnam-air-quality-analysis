"""
Experiment 12: Regional Clustering + Per-Cluster LOSO

Part 1 — Natural station clustering using LOSO-safe summary features.
  K-means k=2,3,4,5 and hierarchical. Pick k with most coherent
  within-cluster PM2.5 distributions.

Part 2 — Per-cluster LOSO prediction.
  C1: Per-cluster XGBoost (train only on same-cluster stations)
  C2: All-station XGBoost with cluster_id as categorical feature
  C3: B1 reference (all-station, no cluster info)
  All use B1 features + S1_sat baseline.

Output:
  analysis/thesis_experiments/experiment_12_regional.md
  analysis/thesis_experiments/loso_per_station_exp12.csv
  analysis/thesis_experiments/cluster_assignments_exp12.csv
"""

import argparse, io, sys, os, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

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
print("EXPERIMENT 12: REGIONAL CLUSTERING + PER-CLUSTER LOSO")
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
#  JOIN BUILDING DENSITY
# ═══════���═══════════════════════════════════════════════════════════════════════
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
#  STAGE 1 SATELLITE BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

S1_SAT_FEATS = ["mean_AOT_outer_mean", "mean_AOT_inner_mean",
                "mean_AOT_grad_mag", "latitude",
                "mean_SSA_inner_mean_clean", "mean_SSA_grad_mag_clean",
                "mean_SSA_local_vs_regional_clean"]

def compute_stage1_features(rfsi_vals):
    rows = []
    for sid in station_ids:
        sdf_mask = stationId_vals == sid
        sdf = df.loc[sdf_mask]
        n_rows = sdf_mask.sum()
        if n_rows == 0:
            rows.append({})
            continue

        row = {}
        idw_vals = rfsi_vals["PM25_nn_idw"][sdf_mask]
        row["mean_PM25_nn_idw"] = np.nanmean(idw_vals)
        row["mean_WS"] = sdf["WS_local"].mean() if "WS_local" in sdf.columns else np.nan
        row["mean_VC"] = sdf["VC"].mean() if "VC" in sdf.columns else np.nan
        precip = sdf["precip_mm"].values if "precip_mm" in sdf.columns else np.zeros(n_rows)
        row["rain_freq"] = (precip > 0.1).sum() / n_rows
        row["mean_PBLH"] = sdf["PBLH"].mean() if "PBLH" in sdf.columns else np.nan
        row["mean_Temp"] = sdf["Temperature_final"].mean() if "Temperature_final" in sdf.columns else np.nan

        if "AOT_valid_count" in sdf.columns:
            row["mean_AOT_valid_frac"] = np.nansum(sdf["AOT_valid_count"].values > 0) / n_rows
        else:
            row["mean_AOT_valid_frac"] = np.nan

        for col, out in [("AOT_grad_mag", "mean_AOT_grad_mag"),
                         ("AOT_inner_mean", "mean_AOT_inner_mean"),
                         ("AOT_outer_mean", "mean_AOT_outer_mean")]:
            row[out] = sdf[col].mean() if col in sdf.columns else np.nan

        row["slope_deg"] = sdf["slope_deg"].iloc[0] if "slope_deg" in sdf.columns else np.nan
        row["latitude"] = sid_lat.get(sid, np.nan)
        row["longitude"] = sid_lon.get(sid, np.nan)
        row["elevation_m"] = sdf["elevation_m"].iloc[0] if "elevation_m" in sdf.columns else np.nan
        row["building_area_3km"] = sdf["building_area_3km"].iloc[0] if "building_area_3km" in sdf.columns else 0

        for ssa_col, out_name in [
            ("SSA_inner_mean", "mean_SSA_inner_mean_clean"),
            ("SSA_grad_mag", "mean_SSA_grad_mag_clean"),
            ("SSA_local_vs_regional", "mean_SSA_local_vs_regional_clean"),
        ]:
            if ssa_col in sdf.columns:
                vals = sdf[ssa_col].values.copy()
                if ssa_col == "SSA_inner_mean":
                    vals[~((vals >= 0) & (vals <= 1.1))] = np.nan
                else:
                    vals[np.abs(vals) > 0.5] = np.nan
                row[out_name] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan
            else:
                row[out_name] = np.nan

        rows.append(row)
    return pd.DataFrame(rows, index=station_ids)


def ridge_predict(X_train, y_train, X_test, alpha=1.0):
    X_tr = X_train.copy()
    X_te = X_test.copy()
    med = np.nanmedian(X_tr, axis=0)
    for j in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, j]), j] = med[j]
        X_te[np.isnan(X_te[:, j]), j] = med[j]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_tr, y_train)
    return ridge.predict(X_te)


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPUTE STATION SUMMARIES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Computing station summaries ---")
print("Computing global RFSI ...")
t1 = time.time()
rfsi_global = compute_rfsi(exclude_sid=None, K=K_NN)
for col in RFSI_COLS:
    df[col] = rfsi_global[col]
print(f"Done ({time.time()-t1:.1f}s)")

station_pm_means = df.groupby("stationId")["PM2.5"].mean()
y_stage1 = station_pm_means.reindex(station_ids).values

s1_df = compute_stage1_features(rfsi_global)

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1: CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("PART 1: NATURAL STATION CLUSTERING")
print(f"{'='*80}")

CLUSTER_FEATS = [
    "mean_AOT_outer_mean", "mean_AOT_inner_mean",
    "mean_SSA_inner_mean_clean", "mean_SSA_grad_mag_clean",
    "mean_SSA_local_vs_regional_clean",
    "latitude", "building_area_3km", "elevation_m",
    "mean_PBLH", "mean_VC", "rain_freq", "mean_Temp",
]

X_clust = s1_df[CLUSTER_FEATS].values.copy()
med = np.nanmedian(X_clust, axis=0)
for j in range(X_clust.shape[1]):
    X_clust[np.isnan(X_clust[:, j]), j] = med[j]

scaler_clust = StandardScaler()
X_clust_scaled = scaler_clust.fit_transform(X_clust)

print(f"\nClustering features ({len(CLUSTER_FEATS)}):")
for f in CLUSTER_FEATS:
    print(f"  {f}")

cluster_results = {}

# K-Means k=2..5
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_clust_scaled)
    sil = silhouette_score(X_clust_scaled, labels) if k > 1 else 0

    # Within-cluster PM2.5 coherence
    cluster_pm_std = []
    for c in range(k):
        c_mask = labels == c
        c_pm = y_stage1[c_mask]
        cluster_pm_std.append(np.std(c_pm))

    mean_within_std = np.mean(cluster_pm_std)
    total_std = np.std(y_stage1)
    coherence = 1 - mean_within_std / total_std

    cluster_results[f"KM{k}"] = {
        "labels": labels.copy(), "k": k, "method": "KMeans",
        "silhouette": sil, "coherence": coherence,
        "mean_within_std": mean_within_std,
    }
    print(f"\n  K-Means k={k}: silhouette={sil:.3f}, "
          f"PM2.5 coherence={coherence:.3f} "
          f"(within-std={mean_within_std:.1f} vs total-std={total_std:.1f})")

    for c in range(k):
        c_mask = labels == c
        c_sids = [station_ids[i] for i in range(n_stn) if c_mask[i]]
        c_pm = y_stage1[c_mask]
        print(f"    Cluster {c} (n={c_mask.sum()}): "
              f"PM2.5 mean={c_pm.mean():.1f} std={c_pm.std():.1f} "
              f"range=[{c_pm.min():.1f}, {c_pm.max():.1f}]")

# Hierarchical (Ward) k=2..5
Z = linkage(X_clust_scaled, method="ward")
for k in [2, 3, 4, 5]:
    labels = fcluster(Z, k, criterion="maxclust") - 1
    sil = silhouette_score(X_clust_scaled, labels)

    cluster_pm_std = []
    for c in range(k):
        c_mask = labels == c
        c_pm = y_stage1[c_mask]
        cluster_pm_std.append(np.std(c_pm))

    mean_within_std = np.mean(cluster_pm_std)
    total_std = np.std(y_stage1)
    coherence = 1 - mean_within_std / total_std

    cluster_results[f"HC{k}"] = {
        "labels": labels.copy(), "k": k, "method": "Hierarchical",
        "silhouette": sil, "coherence": coherence,
        "mean_within_std": mean_within_std,
    }
    print(f"\n  Hierarchical k={k}: silhouette={sil:.3f}, "
          f"PM2.5 coherence={coherence:.3f} "
          f"(within-std={mean_within_std:.1f} vs total-std={total_std:.1f})")

    for c in range(k):
        c_mask = labels == c
        c_pm = y_stage1[c_mask]
        print(f"    Cluster {c} (n={c_mask.sum()}): "
              f"PM2.5 mean={c_pm.mean():.1f} std={c_pm.std():.1f} "
              f"range=[{c_pm.min():.1f}, {c_pm.max():.1f}]")

# Pick best: highest coherence
best_key = max(cluster_results, key=lambda x: cluster_results[x]["coherence"])
best_clust = cluster_results[best_key]
best_labels = best_clust["labels"]
best_k = best_clust["k"]

print(f"\n{'='*60}")
print(f"Best clustering: {best_key} (coherence={best_clust['coherence']:.3f}, "
      f"silhouette={best_clust['silhouette']:.3f})")
print(f"{'='*60}")

# Print station map
print(f"\n  Station cluster assignments ({best_key}):\n")
print(f"  {'Station':50s} {'Cluster':>8s} {'PM2.5':>8s} {'Region':>10s}")
print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*10}")
for i, sid in enumerate(station_ids):
    nm = sid_name.get(sid, sid)[:50]
    rg = sid_region.get(sid, "?")
    print(f"  {nm:50s} {best_labels[i]:>8d} {y_stage1[i]:>8.1f} {rg:>10s}")

# Save cluster assignments
clust_df = pd.DataFrame({
    "station_id": station_ids,
    "station_name": [sid_name.get(s, s) for s in station_ids],
    "region": [sid_region.get(s, "?") for s in station_ids],
    "pm25_mean": y_stage1,
    "cluster": best_labels,
})
for key, cr in cluster_results.items():
    clust_df[f"cluster_{key}"] = cr["labels"]
clust_df.to_csv(os.path.join(OUT_DIR, "cluster_assignments_exp12.csv"),
                index=False, encoding="utf-8-sig")

sid_cluster = dict(zip(station_ids, best_labels))

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
V2_FEATS = B1_FEATS + ["baseline_sat"]

V2_avail = [f for f in V2_FEATS
            if f in df.columns or f in RFSI_COLS or f == "baseline_sat"]
missing = [f for f in V2_FEATS if f not in V2_avail]
if missing:
    print(f"WARNING: V2 missing columns: {missing}")
V2_FEATS = V2_avail

print(f"\n  V2 features: {len(V2_FEATS)} (B1 + satellite baseline)")

# Drop global RFSI before LOSO
df.drop(columns=RFSI_COLS, inplace=True, errors="ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2: PER-CLUSTER LOSO
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"PART 2: PER-CLUSTER LOSO ({best_key}, k={best_k})")
print(f"{'='*80}")

# Pre-extract base features for fast LOSO assembly
all_base_feats = sorted(set(
    f for f in V2_FEATS if f not in RFSI_COLS and f != "baseline_sat"))
base_arr = df[all_base_feats].values
base_col_map = {f: i for i, f in enumerate(all_base_feats)}
rfsi_col_map = {f: i for i, f in enumerate(RFSI_COLS)}

b_feats = [f for f in V2_FEATS if f not in RFSI_COLS and f != "baseline_sat"]
r_feats = [f for f in V2_FEATS if f in RFSI_COLS]
b_idx = [base_col_map[f] for f in b_feats]
r_idx = [rfsi_col_map[f] for f in r_feats]

CONFIG_ORDER = ["C1", "C2", "C3"]
loso_results = {cn: [] for cn in CONFIG_ORDER}

for fold_i, held_sid in enumerate(station_ids):
    nm = sid_name.get(held_sid, held_sid)[:45]
    rg = sid_region.get(held_sid, "?")
    held_cluster = sid_cluster[held_sid]
    mask_test = stationId_vals == held_sid
    n_test = mask_test.sum()
    if n_test < 10:
        print(f"  [{fold_i+1:2d}/{n_stn}] {nm:45s} | SKIP (n={n_test})")
        continue

    y_test = y_all[mask_test]
    t_fold = time.time()

    # ── RFSI excluding held-out ──
    train_sids = [s for s in station_ids if s != held_sid]
    held_idx = station_ids.index(held_sid)
    rfsi_fold = compute_rfsi(exclude_sid=held_sid, K=K_NN)
    rfsi_arr = np.column_stack([rfsi_fold[c] for c in RFSI_COLS])

    # ── Stage 1 satellite baseline ──
    s1_fold = compute_stage1_features(rfsi_fold)
    train_means = df.loc[~mask_test].groupby("stationId")["PM2.5"].mean()
    y_s1_tr = np.array([train_means[s] for s in train_sids])
    s1_train_idx = [i for i in range(n_stn) if station_ids[i] != held_sid]

    X_s1_sat = s1_fold[S1_SAT_FEATS].values
    held_baseline_sat = ridge_predict(
        X_s1_sat[s1_train_idx], y_s1_tr, X_s1_sat[held_idx:held_idx+1])[0]

    # ── Assemble feature matrix ──
    arrays = [base_arr[:, b_idx], rfsi_arr[:, r_idx]]

    # baseline_sat column
    bp = np.zeros(len(df))
    for s in train_sids:
        bp[stationId_vals == s] = train_means[s]
    bp[mask_test] = held_baseline_sat
    arrays.append(bp.reshape(-1, 1))

    X_all = np.hstack(arrays)
    X_te = X_all[mask_test]

    # cluster_id column for C2
    cluster_col = np.array([sid_cluster[s] for s in stationId_vals])

    parts = [f"[{fold_i+1:2d}/{n_stn}] {nm:45s} | cl={held_cluster}"
             f" bl_sat={held_baseline_sat:5.1f}"
             f" (actual={station_pm_means[held_sid]:5.1f}) |"]

    for cname in CONFIG_ORDER:
        if cname == "C1":
            # Per-cluster: train only on same-cluster stations
            same_cluster_sids = set(
                s for s in train_sids if sid_cluster[s] == held_cluster)
            mask_train = np.array([
                (stationId_vals[i] in same_cluster_sids)
                for i in range(len(df))], dtype=bool)
            n_train_stations = len(same_cluster_sids)
            X_tr = X_all[mask_train]
            y_train = y_all[mask_train]

        elif cname == "C2":
            # All stations + cluster_id as feature
            mask_train = ~mask_test
            X_tr = np.hstack([X_all[mask_train], cluster_col[mask_train].reshape(-1, 1)])
            X_te_c2 = np.hstack([X_te, np.full((n_test, 1), held_cluster)])
            y_train = y_all[mask_train]

        elif cname == "C3":
            # Reference: all stations, no cluster info
            mask_train = ~mask_test
            X_tr = X_all[mask_train]
            y_train = y_all[mask_train]

        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr, y_train)

        if cname == "C2":
            pm_pred = m.predict(X_te_c2)
        else:
            pm_pred = m.predict(X_te)

        r2 = r2_score(y_test, pm_pred)
        rmse = np.sqrt(mean_squared_error(y_test, pm_pred))
        mae = mean_absolute_error(y_test, pm_pred)

        extra = {}
        if cname == "C1":
            extra["n_train_stations"] = n_train_stations

        loso_results[cname].append(dict(
            station_id=held_sid,
            station_name=sid_name.get(held_sid, held_sid),
            region=rg, n_rows=n_test, cluster=held_cluster,
            r2=round(r2, 4), rmse=round(rmse, 2), mae=round(mae, 2),
            baseline_sat=round(held_baseline_sat, 2),
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
    os.path.join(OUT_DIR, "loso_per_station_exp12.csv"),
    index=False, encoding="utf-8-sig")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD PREVIOUS RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
prev_loso = {}
for exp_csv in ["loso_per_station_exp07.csv", "loso_per_station_exp11.csv"]:
    lp = os.path.join(OUT_DIR, exp_csv)
    if os.path.exists(lp):
        tmp = pd.read_csv(lp, dtype={"station_id": str})
        for cfg in tmp["config"].unique():
            prev_loso[cfg] = tmp[tmp["config"] == cfg].to_dict("records")

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

b1_r2 = {r["station_id"]: r["r2"] for r in prev_loso.get("B1", [])}
v2_r2 = {r["station_id"]: r["r2"] for r in prev_loso.get("V2", [])}

# Per-cluster summaries
print(f"\n{'='*80}")
print("PER-CLUSTER LOSO SUMMARY")
print(f"{'='*80}")
for cname in CONFIG_ORDER:
    rdf = pd.DataFrame(loso_results[cname])
    print(f"\n  {cname}: overall R²={sums[cname]['mean_r2']:.4f}")
    for c in range(best_k):
        c_df = rdf[rdf["cluster"] == c]
        if len(c_df) == 0:
            continue
        c_r2 = c_df["r2"].mean()
        c_mae = c_df["mae"].mean()
        print(f"    Cluster {c} (n_stations={len(c_df)}): "
              f"R²={c_r2:.4f}, MAE={c_mae:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("GENERATING REPORT")
print(f"{'='*80}")

rpt = []
rpt.append("# Experiment 12: Regional Clustering + Per-Cluster LOSO\n")
rpt.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rpt.append(f"**Dataset:** unified_thesis_v2.csv — {len(df):,} rows, {n_stn} stations")
rpt.append(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, "
           f"lr=0.05, device=cuda\n")

rpt.append("## Part 1: Clustering Results\n")
rpt.append(f"Clustering features ({len(CLUSTER_FEATS)}): {', '.join(CLUSTER_FEATS)}\n")
rpt.append("| Method | k | Silhouette | PM2.5 Coherence | Within-Cluster Std |")
rpt.append("|--------|---|:---:|:---:|:---:|")
for key in sorted(cluster_results.keys()):
    cr = cluster_results[key]
    bold = "**" if key == best_key else ""
    rpt.append(f"| {bold}{cr['method']} k={cr['k']}{bold} | {cr['k']} | "
               f"{cr['silhouette']:.3f} | {bold}{cr['coherence']:.3f}{bold} | "
               f"{cr['mean_within_std']:.1f} |")

rpt.append(f"\n**Selected:** {best_key} (highest PM2.5 coherence)\n")

rpt.append(f"### Cluster Assignments ({best_key})\n")
for c in range(best_k):
    c_mask = best_labels == c
    c_pm = y_stage1[c_mask]
    rpt.append(f"**Cluster {c}** (n={c_mask.sum()}, "
               f"PM2.5 mean={c_pm.mean():.1f}, std={c_pm.std():.1f}):\n")
    for i in range(n_stn):
        if best_labels[i] == c:
            nm = sid_name.get(station_ids[i], station_ids[i])[:50]
            rg = sid_region.get(station_ids[i], "?")
            rpt.append(f"- {nm} ({rg}, PM2.5={y_stage1[i]:.1f})")
    rpt.append("")

rpt.append("## Part 2: LOSO Results\n")
rpt.append("| Config | Description | LOSO R² (mean) | LOSO R² (median) | "
           "LOSO MAE | Neg Stations |")
rpt.append("|--------|-------------|:---:|:---:|:---:|:---:|")

ref_cfgs = [("B1", "Exp07 reference"), ("V2", "Exp11: B1 + sat baseline")]
for cn, desc in ref_cfgs:
    if cn in prev_sums:
        s = prev_sums[cn]
        rpt.append(f"| {cn} (ref) | {desc} | {s['mean_r2']:.4f} | "
                   f"{s['median_r2']:.4f} | {s['mean_mae']:.2f} | {s['neg_count']} |")

descs = {
    "C1": f"Per-cluster LOSO ({best_key})",
    "C2": f"All stations + cluster_id feature ({best_key})",
    "C3": "All stations, no cluster (V2 reference)",
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
    for c in range(best_k):
        c_df = rdf[rdf["cluster"] == c]
        if len(c_df) == 0:
            continue
        rpt.append(f"| {c} | {len(c_df)} | {c_df['r2'].mean():.4f} | "
                   f"{c_df['mae'].mean():.2f} |")
    rpt.append("")

# Per-station table
best_cfg = max(CONFIG_ORDER, key=lambda c: sums[c]["mean_r2"])
best_df = pd.DataFrame(loso_results[best_cfg]).sort_values("r2", ascending=True)

rpt.append(f"\n### Per-Station: {best_cfg}\n")
rpt.append(f"| Station | Region | Cluster | B1 R² | {best_cfg} R² | Δ vs B1 |")
rpt.append("|---------|--------|:---:|:---:|:---:|:---:|")
for _, r in best_df.iterrows():
    sid = r["station_id"]
    nm2 = str(r["station_name"])[:40]
    b1v = b1_r2.get(sid, np.nan)
    bv = r["r2"]
    b1_s = f"{b1v:.4f}" if pd.notna(b1v) else "—"
    delta = f"{bv - b1v:+.4f}" if pd.notna(b1v) else "—"
    rpt.append(f"| {nm2} | {r['region']} | {r['cluster']} | "
               f"{b1_s} | {bv:.4f} | {delta} |")

rpt.append("\n## Summary\n")
for cn in CONFIG_ORDER:
    s = sums[cn]
    rpt.append(f"- **{cn}**: LOSO R²={s['mean_r2']:.4f} "
               f"(median={s['median_r2']:.4f}), MAE={s['mean_mae']:.2f}, "
               f"{s['neg_count']} negative stations")

b1_mean = prev_sums.get("B1", {}).get("mean_r2")
if b1_mean is not None:
    best_r2 = sums[best_cfg]["mean_r2"]
    rpt.append(f"\n**Best ({best_cfg}) vs B1:** {best_r2 - b1_mean:+.4f}")

report_path = os.path.join(OUT_DIR, "experiment_12_regional.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))

print(f"\nReport: {report_path}")
print(f"\nDONE — total time: {time.time()-t0:.0f}s")
