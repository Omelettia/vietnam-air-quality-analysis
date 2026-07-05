"""
Experiment 03: Regional Daily Anomaly XGBoost

Instead of subtracting station-level climatology (unavailable at unseen stations),
subtract the regional daily mean PM2.5 across other stations.

Configs:
  H1 — National mean (date × hour) anomaly, no geography
  H2 — Region-specific mean (date × hour) anomaly
  H3 — National mean (date only) anomaly — preserves diurnal cycle in target
  H4 — Best of H1-H3 + terrain features (no lat/lon)

Output:
  analysis/thesis_experiments/experiment_03_regional_anomaly.md
  analysis/thesis_experiments/loso_per_station_exp03.csv
  analysis/thesis_experiments/feature_importance_exp03.csv
"""

import argparse, io, sys, os, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None,
                    help="Base directory containing data/ and analysis/ folders")
args = parser.parse_args()

BASE = args.data_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
os.makedirs("analysis/thesis_experiments", exist_ok=True)

XGB_PARAMS = dict(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    device="cuda",
    random_state=42, n_jobs=-1,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("EXPERIMENT 03: REGIONAL DAILY ANOMALY XGBOOST")
print("=" * 80)

t0 = time.time()
df = pd.read_csv("data/merged/unified_thesis_v1.csv", dtype={"stationId": str})
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["hour"] = df["ts"].dt.hour
df["date"] = df["ts"].dt.date
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations ({time.time()-t0:.1f}s)")

meta = pd.read_csv("analysis/thesis_audit/station_selection_final.csv",
                    dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))
sid_region = dict(zip(meta["stationId"], meta["region"]))
station_ids = sorted(df["stationId"].unique())

df["region_adj"] = df["region"].replace("Unknown", "Central")
sid_region_adj = {s: ("Central" if sid_region.get(s) == "Unknown" else sid_region.get(s, "Unknown"))
                  for s in station_ids}

region_counts = {}
for s in station_ids:
    r = sid_region_adj[s]
    region_counts[r] = region_counts.get(r, 0) + 1
print(f"Region counts (adj): {region_counts}")

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
MET_BASE = ["Temperature_final", "Humidity_final", "Pressure_final", "WS_om", "PBLH"]
MET_ANOM = ["Temperature_regional_anom", "Humidity_regional_anom",
            "Pressure_regional_anom", "WS_regional_anom", "PBLH_regional_anom"]

RAW_FEATURES = [
    "AOT", "AOT_mean", "AOT_inner_mean", "AOT_outer_mean", "RF", "SSA",
    "Uncertainty", "AE", "AOT_valid_count", "AOD_physics", "AOT_spatial_std",
    "AOT_local_vs_regional", "AOT_ffill_48h", "hours_since_valid_AOT",
    "AOT_lag_1h", "AOT_lag_3h", "AOT_lag_6h", "AOT_rolling_mean_6h",
    "AOT_rolling_mean_24h", "AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_grad_dir",
    "precip_mm", "hrs_since_rain", "rain_sum_24h", "rain_sum_48h",
    "rain_days_7d", "consecutive_dry_days",
    "wind_u", "wind_v", "wind_dir_sin", "wind_dir_cos",
    "WS_local", "wind_u_local", "wind_v_local", "wind_dir_sin_local", "wind_dir_cos_local",
    "VC",
    "dT_6h", "dRH_6h", "dWS_6h", "dP_6h",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
]

TERRAIN = ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
           "elev_x_PBLH", "elev_x_hour_sin"]

for lst_name, lst in [("RAW", RAW_FEATURES), ("TERRAIN", TERRAIN)]:
    miss = [f for f in lst if f not in df.columns]
    if miss:
        print(f"WARNING: {lst_name} missing: {miss}")
RAW_FEATURES = [f for f in RAW_FEATURES if f in df.columns]
TERRAIN = [f for f in TERRAIN if f in df.columns]

FEATURES_H = MET_ANOM + RAW_FEATURES
FEATURES_H4 = FEATURES_H + TERRAIN
print(f"Features: H={len(FEATURES_H)}, H4={len(FEATURES_H4)}")

AC = ["PM2.5"] + MET_BASE

# ═══════════════════════════════════════════════════════════════════════════════
#  PRECOMPUTE SUMS/COUNTS FOR EFFICIENT LOSO
# ═══════════════════════════════════════════════════════════════════════════════
print("Precomputing regional sums...")
t1 = time.time()

nat_dh_sum = df.groupby(["date", "hour"])[AC].sum()
nat_dh_cnt = df.groupby(["date", "hour"])[AC].count()

reg_dh_sum = df.groupby(["region_adj", "date", "hour"])[AC].sum()
reg_dh_cnt = df.groupby(["region_adj", "date", "hour"])[AC].count()

nat_d_sum = df.groupby(["date"])[AC].sum()
nat_d_cnt = df.groupby(["date"])[AC].count()

stn_dh_sum = {}
stn_dh_cnt = {}
stn_d_sum = {}
stn_d_cnt = {}
for sid in station_ids:
    sub = df[df["stationId"] == sid]
    stn_dh_sum[sid] = sub.groupby(["date", "hour"])[AC].sum()
    stn_dh_cnt[sid] = sub.groupby(["date", "hour"])[AC].count()
    stn_d_sum[sid] = sub.groupby(["date"])[AC].sum()
    stn_d_cnt[sid] = sub.groupby(["date"])[AC].count()

unique_dh = nat_dh_sum.index
dh_to_pos = {dh: i for i, dh in enumerate(unique_dh)}
row_dh_pos = np.array([dh_to_pos[(d, h)]
                        for d, h in zip(df["date"].values, df["hour"].values)])

unique_d = nat_d_sum.index
d_to_pos = {d: i for i, d in enumerate(unique_d)}
row_d_pos = np.array([d_to_pos[d] for d in df["date"].values])

pm25_arr = df["PM2.5"].values.copy()
station_arr = df["stationId"].values
region_adj_arr = df["region_adj"].values
met_raw = {f: df[f].values.copy() for f in MET_BASE}
raw_X = {f: df[f].values for f in RAW_FEATURES + TERRAIN if f in df.columns}

print(f"Precomputation done ({time.time()-t1:.1f}s)")
print(f"Unique (date, hour): {len(unique_dh):,} | Unique dates: {len(unique_d):,}")

# ═══════════════════════════════════════════════════════════════════════════════
#  REGIONAL MEAN FUNCTIONS — return {col: array} mapped to ALL rows
# ═══════════════════════════════════════════════════════════════════════════════
REGIONS = ["North", "Central", "South"]

def rm_h1_mapped(sid):
    s = stn_dh_sum[sid].reindex(nat_dh_sum.index, fill_value=0)
    c = stn_dh_cnt[sid].reindex(nat_dh_cnt.index, fill_value=0)
    rm = (nat_dh_sum - s) / (nat_dh_cnt - c)
    return {col: rm[col].values[row_dh_pos] for col in AC}

def rm_h2_mapped(sid):
    region = sid_region_adj[sid]
    result = {col: np.full(len(df), np.nan) for col in AC}
    for r in REGIONS:
        r_mask = region_adj_arr == r
        if not r_mask.any():
            continue
        if r == region and region_counts.get(r, 0) <= 2:
            nat = rm_h1_mapped(sid)
            for col in AC:
                result[col][r_mask] = nat[col][r_mask]
            continue
        if r == region:
            r_sum = reg_dh_sum.loc[r]
            r_cnt = reg_dh_cnt.loc[r]
            s = stn_dh_sum[sid].reindex(r_sum.index, fill_value=0)
            c = stn_dh_cnt[sid].reindex(r_cnt.index, fill_value=0)
            rm_r = (r_sum - s) / (r_cnt - c)
        else:
            rm_r = reg_dh_sum.loc[r] / reg_dh_cnt.loc[r]
        rm_r_full = rm_r.reindex(unique_dh)
        for col in AC:
            result[col][r_mask] = rm_r_full[col].values[row_dh_pos[r_mask]]
    return result

def rm_h3_mapped(sid):
    s = stn_d_sum[sid].reindex(nat_d_sum.index, fill_value=0)
    c = stn_d_cnt[sid].reindex(nat_d_cnt.index, fill_value=0)
    rm = (nat_d_sum - s) / (nat_d_cnt - c)
    return {col: rm[col].values[row_d_pos] for col in AC}

# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
def build_X(features, anom_dict, indices):
    cols = []
    for f in features:
        if f in anom_dict:
            cols.append(anom_dict[f][indices])
        elif f in raw_X:
            cols.append(raw_X[f][indices])
        else:
            cols.append(np.full(len(indices), np.nan))
    return np.column_stack(cols)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOSO
# ═══════════════════════════════════════════════════════════════════════════════
def run_loso(features, rm_func, label=""):
    results = []
    t_start = time.time()
    for i, sid in enumerate(station_ids):
        rm = rm_func(sid)
        target = pm25_arr - rm["PM2.5"]
        anom = {a: met_raw[b] - rm[b] for b, a in zip(MET_BASE, MET_ANOM)}

        train_mask = station_arr != sid
        valid = ~np.isnan(target)
        train_idx = np.where(train_mask & valid)[0]
        test_idx = np.where(~train_mask & valid)[0]

        if len(test_idx) == 0:
            print(f"  [{i+1:2d}/40] {sid_name.get(sid, sid)[:45]:45s} | SKIPPED")
            continue

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(build_X(features, anom, train_idx), target[train_idx])

        pred_anom = model.predict(build_X(features, anom, test_idx))
        pred_abs = pred_anom + rm["PM2.5"][test_idx]
        y_abs = pm25_arr[test_idx]

        r2 = r2_score(y_abs, pred_abs)
        r2_a = r2_score(target[test_idx], pred_anom)
        rmse = np.sqrt(mean_squared_error(y_abs, pred_abs))
        mae = mean_absolute_error(y_abs, pred_abs)

        region = sid_region.get(sid, "?")
        name = sid_name.get(sid, sid)[:45]
        print(f"  [{i+1:2d}/40] {name:45s} | R²={r2:+.3f}  R²_a={r2_a:+.3f}  "
              f"RMSE={rmse:5.1f}  n={len(test_idx):5d}  [{region}]")

        results.append(dict(station_id=sid, station_name=sid_name.get(sid, sid),
                            region=region, n_rows=len(test_idx),
                            r2=round(r2, 4), r2_anom=round(r2_a, 4),
                            rmse=round(rmse, 2), mae=round(mae, 2)))

    elapsed = time.time() - t_start
    r2s = [r["r2"] for r in results]
    mean_r2, median_r2 = np.mean(r2s), np.median(r2s)
    neg = sum(1 for v in r2s if v < 0)
    mean_rmse = np.mean([r["rmse"] for r in results])
    print(f"  Time: {elapsed:.0f}s")
    print(f"  LOSO mean R²={mean_r2:.4f}  median={median_r2:.4f}  neg={neg}")
    return results, mean_r2, median_r2, neg, mean_rmse

# ═══════════════════════════════════════════════════════════════════════════════
#  KFOLD
# ═══════════════════════════════════════════════════════════════════════════════
def run_kfold(features, mode, n_splits=5, label=""):
    t_start = time.time()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_at, all_ap, all_nt, all_np_ = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_sub = df.iloc[train_idx]
        if mode == "national_dh":
            rm_s = train_sub.groupby(["date", "hour"])[AC].mean()
            rm_full = rm_s.reindex(unique_dh)
            rm_vals = {col: rm_full[col].values[row_dh_pos] for col in AC}
        elif mode == "regional_dh":
            rm_s = train_sub.groupby(["region_adj", "date", "hour"])[AC].mean()
            rm_vals = {col: np.full(len(df), np.nan) for col in AC}
            for r in REGIONS:
                r_mask = region_adj_arr == r
                if not r_mask.any():
                    continue
                try:
                    rm_r = rm_s.loc[r].reindex(unique_dh)
                except KeyError:
                    continue
                for col in AC:
                    rm_vals[col][r_mask] = rm_r[col].values[row_dh_pos[r_mask]]
        elif mode == "national_d":
            rm_s = train_sub.groupby(["date"])[AC].mean()
            rm_full = rm_s.reindex(unique_d)
            rm_vals = {col: rm_full[col].values[row_d_pos] for col in AC}

        target = pm25_arr - rm_vals["PM2.5"]
        anom = {a: met_raw[b] - rm_vals[b] for b, a in zip(MET_BASE, MET_ANOM)}
        valid = ~np.isnan(target)
        ti = np.intersect1d(train_idx, np.where(valid)[0])
        vi = np.intersect1d(test_idx, np.where(valid)[0])
        if len(vi) == 0:
            continue

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(build_X(features, anom, ti), target[ti])
        pred_anom = model.predict(build_X(features, anom, vi))
        pred_abs = pred_anom + rm_vals["PM2.5"][vi]

        all_at.extend(pm25_arr[vi])
        all_ap.extend(pred_abs)
        all_nt.extend(target[vi])
        all_np_.extend(pred_anom)

    r2_abs = r2_score(all_at, all_ap)
    r2_anom = r2_score(all_nt, all_np_)
    rmse = np.sqrt(mean_squared_error(all_at, all_ap))
    mae = mean_absolute_error(all_at, all_ap)
    elapsed = time.time() - t_start
    print(f"  KFold: R²_abs={r2_abs:.4f}  R²_anom={r2_anom:.4f}  RMSE={rmse:.2f} ({elapsed:.0f}s)")
    return r2_abs, r2_anom, rmse, mae

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN H1, H2, H3
# ═══════════════════════════════════════════════════════════════════════════════
configs = {}

for key, desc, mode, rm_func in [
    ("H1", "NATIONAL DAY+HOUR ANOMALY", "national_dh", rm_h1_mapped),
    ("H2", "REGION-SPECIFIC DAY+HOUR ANOMALY", "regional_dh", rm_h2_mapped),
    ("H3", "NATIONAL DAILY ANOMALY (no hour)", "national_d", rm_h3_mapped),
]:
    print(f"\n{'='*80}\nCONFIG {key}: {desc}\n{'='*80}")
    kf = run_kfold(FEATURES_H, mode, label=key)
    print("\n  --- LOSO ---")
    loso_res, m_r2, md_r2, neg, m_rmse = run_loso(FEATURES_H, rm_func, label=key)
    configs[key] = dict(kf_r2=kf[0], kf_r2_anom=kf[1], kf_rmse=kf[2], kf_mae=kf[3],
                        loso_r2=m_r2, loso_med=md_r2, neg=neg, loso_rmse=m_rmse,
                        loso_results=loso_res, mode=mode, rm_func=rm_func)

best_key = max(["H1", "H2", "H3"], key=lambda k: configs[k]["loso_r2"])
print(f"\nBest base config: {best_key} (LOSO R²={configs[best_key]['loso_r2']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN H4: BEST + TERRAIN
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}\nCONFIG H4: {best_key} + TERRAIN ({len(FEATURES_H4)} features)\n{'='*80}")
kf = run_kfold(FEATURES_H4, configs[best_key]["mode"], label="H4")
print("\n  --- LOSO ---")
loso_res, m_r2, md_r2, neg, m_rmse = run_loso(FEATURES_H4, configs[best_key]["rm_func"], label="H4")
configs["H4"] = dict(kf_r2=kf[0], kf_r2_anom=kf[1], kf_rmse=kf[2], kf_mae=kf[3],
                     loso_r2=m_r2, loso_med=md_r2, neg=neg, loso_rmse=m_rmse,
                     loso_results=loso_res, mode=configs[best_key]["mode"],
                     rm_func=configs[best_key]["rm_func"])

overall_best = max(configs.keys(), key=lambda k: configs[k]["loso_r2"])

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}\nFEATURE IMPORTANCE ({overall_best} — full training)\n{'='*80}")

best_mode = configs[overall_best]["mode"]
if best_mode == "national_dh":
    rm_all = {col: (nat_dh_sum[col] / nat_dh_cnt[col]).values[row_dh_pos] for col in AC}
elif best_mode == "regional_dh":
    rm_all = {col: np.full(len(df), np.nan) for col in AC}
    for r in REGIONS:
        r_mask = region_adj_arr == r
        try:
            rm_r = (reg_dh_sum.loc[r] / reg_dh_cnt.loc[r]).reindex(unique_dh)
        except KeyError:
            continue
        for col in AC:
            rm_all[col][r_mask] = rm_r[col].values[row_dh_pos[r_mask]]
elif best_mode == "national_d":
    rm_all = {col: (nat_d_sum[col] / nat_d_cnt[col]).values[row_d_pos] for col in AC}

target_all = pm25_arr - rm_all["PM2.5"]
anom_all = {a: met_raw[b] - rm_all[b] for b, a in zip(MET_BASE, MET_ANOM)}
valid_all = np.where(~np.isnan(target_all))[0]

best_features = FEATURES_H4 if overall_best == "H4" else FEATURES_H
print(f"Training on {len(valid_all):,} rows, {len(best_features)} features")

model_full = xgb.XGBRegressor(**XGB_PARAMS)
model_full.fit(build_X(best_features, anom_all, valid_all), target_all[valid_all])

gains = model_full.get_booster().get_score(importance_type="total_gain")
fi = sorted([(best_features[int(k.replace("f", ""))], v) for k, v in gains.items()],
            key=lambda x: -x[1])

print("\nTop 20 features by gain:")
for rank, (feat, gain) in enumerate(fi[:20], 1):
    print(f"  {rank:3d}. {feat:35s} gain={gain:.0f}")

pd.DataFrame([(f, g, r) for r, (f, g) in enumerate(fi, 1)],
             columns=["feature", "gain", "rank"]).to_csv(
    "analysis/thesis_experiments/feature_importance_exp03.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE PER-STATION CSV
# ═══════════════════════════════════════════════════════════════════════════════
best_loso = configs[overall_best]["loso_results"]
pd.DataFrame(best_loso).sort_values("r2").to_csv(
    "analysis/thesis_experiments/loso_per_station_exp03.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}\nGENERATING REPORT\n{'='*80}")

try:
    prev_c = pd.read_csv("analysis/thesis_experiments/loso_per_station_config_c.csv",
                         dtype={"station_id": str})
    c_r2 = dict(zip(prev_c["station_id"], prev_c["r2"]))
except Exception:
    c_r2 = {}

try:
    prev_f = pd.read_csv("analysis/thesis_experiments/loso_per_station_config_f.csv",
                         dtype={"station_id": str})
    f_r2 = dict(zip(prev_f["station_id"], prev_f["r2"]))
except Exception:
    f_r2 = {}

ref = {
    "C":    dict(kf=0.7262, loso=-0.4953, med=-0.0004, neg=20, gap=1.2215),
    "E":    dict(kf=0.6926, loso=0.2252,  med=0.2640,  neg=7,  gap=0.4674),
    "F_K5": dict(kf=0.6926, loso=-0.3030, med=-0.0364, neg=22, gap=0.9956),
}

L = []
def w(s=""):
    L.append(s)

now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
w(f"# Experiment 03: Regional Daily Anomaly XGBoost")
w()
w(f"**Date:** {now_str}")
w(f"**Dataset:** {len(df):,} rows, {len(station_ids)} stations")
w(f"**XGBoost:** v{xgb.__version__}, n_estimators=500, max_depth=7, lr=0.05")
w(f"**Best base config:** {best_key} | Overall best: {overall_best}")
w()

w("## Comparison Table (all R² on absolute PM2.5)")
w()
w("| Config | Description | Features | KFold R² | LOSO R² (mean) | LOSO R² (median) | Neg R² | Gap |")
w("|--------|-------------|----------|----------|----------------|------------------|--------|-----|")
for rk, rd, rnf in [("C (Exp01)", "Full baseline", 62),
                     ("E (Exp02)", "Oracle station anomaly", 55),
                     ("F K=5 (Exp02)", "Neighbor station anomaly", 55)]:
    tag = rk.split()[0]
    r = ref.get(tag, ref.get(tag.replace(" ", "_"), ref.get("F_K5")))
    if "F" in rk:
        r = ref["F_K5"]
    elif "E" in rk:
        r = ref["E"]
    else:
        r = ref["C"]
    w(f"| {rk} | {rd} | {rnf} | {r['kf']:.4f} | {r['loso']:.4f} | {r['med']:.4f} | {r['neg']} | {r['gap']:.4f} |")

for key in ["H1", "H2", "H3", "H4"]:
    c = configs[key]
    desc = {"H1": "National day+hour anomaly", "H2": "Regional day+hour anomaly",
            "H3": "National daily anomaly", "H4": f"{best_key} + terrain"}[key]
    nf = len(FEATURES_H4) if key == "H4" else len(FEATURES_H)
    gap = c["kf_r2"] - c["loso_r2"]
    w(f"| {key} | {desc} | {nf} | {c['kf_r2']:.4f} | {c['loso_r2']:.4f} | {c['loso_med']:.4f} | {c['neg']} | {gap:.4f} |")
w()

w(f"## Per-Station LOSO: {overall_best} vs Config C vs Config F(K=3)")
w()
w(f"| Station | Region | C R² | F R² | {overall_best} R² | Δ(C) | RMSE |")
w("|---------|--------|------|------|--------|------|------|")
for res in sorted(best_loso, key=lambda x: x["r2"]):
    sid = res["station_id"]
    cr = c_r2.get(sid, float("nan"))
    fr = f_r2.get(sid, float("nan"))
    hr = res["r2"]
    delta = hr - cr if not np.isnan(cr) else float("nan")
    mark = " ✓" if not np.isnan(delta) and delta > 0 else ""
    name = res["station_name"][:50]
    cr_s = f"{cr:.4f}" if not np.isnan(cr) else "—"
    fr_s = f"{fr:.4f}" if not np.isnan(fr) else "—"
    d_s = f"{delta:+.4f}{mark}" if not np.isnan(delta) else "—"
    w(f"| {name} | {res['region']} | {cr_s} | {fr_s} | {hr:.4f} | {d_s} | {res['rmse']:.1f} |")
w()

w("## Regional Comparison")
w()
w(f"| Region | Config C | Config F(K=3) | {overall_best} | Oracle E |")
w("|--------|---------|---------------|------|----------|")
e_reg = {"North": 0.2313, "Central": 0.2501, "South": 0.1888}
for region in ["North", "Central", "South"]:
    stns = [r for r in best_loso
            if r["region"] == region or (region == "Central" and r["region"] == "Unknown")]
    sids = [r["station_id"] for r in stns]
    c_vals = [c_r2[s] for s in sids if s in c_r2]
    f_vals = [f_r2[s] for s in sids if s in f_r2]
    h_vals = [r["r2"] for r in stns]
    cr = np.mean(c_vals) if c_vals else float("nan")
    fr = np.mean(f_vals) if f_vals else float("nan")
    hr = np.mean(h_vals) if h_vals else float("nan")
    er = e_reg.get(region, float("nan"))
    w(f"| {region} | {cr:+.4f} | {fr:+.4f} | {hr:+.4f} | {er:+.4f} |")
w()

w(f"## Feature Importance ({overall_best}, top 20)")
w()
w("| Rank | Feature | Gain |")
w("|------|---------|------|")
for rank, (feat, gain) in enumerate(fi[:20], 1):
    w(f"| {rank} | {feat} | {gain:.0f} |")
w()

w("## Anomaly-Space KFold R²")
w()
for key in ["H1", "H2", "H3", "H4"]:
    c = configs[key]
    w(f"- {key}: R²_anom = {c['kf_r2_anom']:.4f} (absolute R² = {c['kf_r2']:.4f})")
w()

w("## Analysis")
w()

bh = max(["H1", "H2", "H3"], key=lambda k: configs[k]["loso_r2"])

w("### 1. Does regional anomaly beat neighbor station anomaly? (H vs F)")
w()
w(f"- Best H config: {bh} LOSO R² = {configs[bh]['loso_r2']:.4f}")
w(f"- Config F (K=5): LOSO R² = {ref['F_K5']['loso']:.4f}")
beat = "YES" if configs[bh]["loso_r2"] > ref["F_K5"]["loso"] else "NO"
w(f"- Regional anomaly beats neighbor? **{beat}** (Δ = {configs[bh]['loso_r2'] - ref['F_K5']['loso']:+.4f})")
w()

w("### 2. Does it approach the oracle ceiling? (H vs E)")
w()
w(f"- Best H: LOSO R² = {configs[bh]['loso_r2']:.4f}")
w(f"- Oracle E: LOSO R² = {ref['E']['loso']:.4f}")
w(f"- Gap to oracle: {ref['E']['loso'] - configs[bh]['loso_r2']:.4f}")
w()

w("### 3. National vs region-specific mean — which works better?")
w()
w(f"- H1 (national): LOSO R² = {configs['H1']['loso_r2']:.4f}")
w(f"- H2 (regional): LOSO R² = {configs['H2']['loso_r2']:.4f}")
w(f"- Winner: **{'National (H1)' if configs['H1']['loso_r2'] >= configs['H2']['loso_r2'] else 'Regional (H2)'}**")
w()

w("### 4. Day+hour vs day-only — which is better?")
w()
w(f"- H1 (day+hour): LOSO R² = {configs['H1']['loso_r2']:.4f}")
w(f"- H3 (day-only): LOSO R² = {configs['H3']['loso_r2']:.4f}")
w(f"- Winner: **{'Day+hour (H1)' if configs['H1']['loso_r2'] >= configs['H3']['loso_r2'] else 'Day-only (H3)'}**")
w()

w("### 5. Does terrain help? (H4 vs best H)")
w()
w(f"- {best_key}: LOSO R² = {configs[best_key]['loso_r2']:.4f}")
w(f"- H4 ({best_key}+terrain): LOSO R² = {configs['H4']['loso_r2']:.4f}")
w(f"- Terrain effect: {configs['H4']['loso_r2'] - configs[best_key]['loso_r2']:+.4f}")
w()

w("### 6. Which features dominate?")
w()
w(f"Top 5 by gain: {', '.join(f[0] for f in fi[:5])}")
has_geo = any(f[0] in {"latitude", "longitude"} for f in fi[:20])
w(f"Geography in top 20: {'YES ⚠' if has_geo else 'No — physics-based ✓'}")

report = "\n".join(L)
with open("analysis/thesis_experiments/experiment_03_regional_anomaly.md", "w", encoding="utf-8") as fh:
    fh.write(report)

total_time = time.time() - t0
print(f"\nReport saved: analysis/thesis_experiments/experiment_03_regional_anomaly.md")
print(f"LOSO per-station: analysis/thesis_experiments/loso_per_station_exp03.csv")
print(f"Feature importance: analysis/thesis_experiments/feature_importance_exp03.csv")
print(f"\nDONE — total time: {total_time:.0f}s")
