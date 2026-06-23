"""
Satellite PM2.5 product assessment: GEOS-CF, MERRA-2, GHAP
Compare with ground-truth PM2.5 at 40 KK stations.
"""
import os, sys, time, zipfile, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = Path(__file__).resolve().parents[3]
OUT  = BASE / "analysis" / "thesis_experiments" / "satellite_products"
OUT.mkdir(parents=True, exist_ok=True)

# ── station metadata ──
META = pd.read_csv(BASE / "analysis" / "thesis_audit" / "station_selection_final.csv")
MODEL_IDS = set(META["stationId"].values)
TIER_MAP  = {sid: f"t{t}" for sid, t in zip(META["stationId"], META["tier"])}
NAME_MAP  = dict(zip(META["stationId"], META["station_name"]))
REGION_MAP = dict(zip(META["stationId"], META["region"]))

print("=" * 80)
print("SATELLITE PM2.5 PRODUCT ASSESSMENT")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load and inspect all files
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 1: LOAD & INSPECT")
print("=" * 80)

# ── GHAP ──
ghap_zip = BASE / "data" / "gee_exports" / "pm25-20260523T092435Z-3-001.zip"
with zipfile.ZipFile(ghap_zip) as zf:
    ghap_annual = pd.read_csv(zf.open("pm25/ghap_annual_mean.csv"))
    ghap_monthly = pd.read_csv(zf.open("pm25/ghap_monthly_climatology.csv"))
    ghap_daily = pd.read_csv(zf.open("pm25/ghap_daily_2021_2022.csv"))

print("\n--- GHAP Annual Mean ---")
print(f"  Shape: {ghap_annual.shape}, Columns: {list(ghap_annual.columns)}")
print(f"  Stations: {ghap_annual['stationId'].nunique()}")
print(f"  PM2.5 range: {ghap_annual['mean'].min():.1f} – {ghap_annual['mean'].max():.1f}")
print(f"  PM2.5 stats: mean={ghap_annual['mean'].mean():.1f}, median={ghap_annual['mean'].median():.1f}")
print(f"  Missing: {ghap_annual['mean'].isna().sum()}")

print("\n--- GHAP Monthly Climatology ---")
print(f"  Shape: {ghap_monthly.shape}, Columns: {list(ghap_monthly.columns)}")
print(f"  Stations: {ghap_monthly['stationId'].nunique()}")
print(f"  Months: {sorted(ghap_monthly['month'].unique())}")
print(f"  PM2.5 range: {ghap_monthly['mean'].min():.1f} – {ghap_monthly['mean'].max():.1f}")
print(f"  Missing: {ghap_monthly['mean'].isna().sum()}")

print("\n--- GHAP Daily (2021-2022) ---")
print(f"  Shape: {ghap_daily.shape}, Columns: {list(ghap_daily.columns)}")
print(f"  Stations: {ghap_daily['stationId'].nunique()}")
ghap_daily["date"] = pd.to_datetime(ghap_daily["date"])
print(f"  Date range: {ghap_daily['date'].min()} – {ghap_daily['date'].max()}")
print(f"  PM2.5 range: {ghap_daily['mean'].min():.1f} – {ghap_daily['mean'].max():.1f}")
print(f"  Missing: {ghap_daily['mean'].isna().sum()} / {len(ghap_daily)} ({100*ghap_daily['mean'].isna().mean():.1f}%)")

# ── GEOS-CF ──
hourly_zip = BASE / "data" / "gee_exports" / "hourly-20260523T091953Z-3-001.zip"
geoscf_frames = []
merra2_frames = []

with zipfile.ZipFile(hourly_zip) as zf:
    for name in sorted(zf.namelist()):
        if "geoscf" in name and name.endswith(".csv"):
            df = pd.read_csv(zf.open(name))
            year = name.split("_")[-1].replace(".csv", "")
            print(f"\n--- GEOS-CF {year} ---")
            print(f"  Shape: {df.shape}, Columns: {list(df.columns)}")
            print(f"  Stations: {df['stationId'].nunique()}")
            df["datetime"] = pd.to_datetime(df["datetime"])
            print(f"  Date range: {df['datetime'].min()} – {df['datetime'].max()}")
            print(f"  PM25_RH35_GCC: mean={df['PM25_RH35_GCC'].mean():.1f}, "
                  f"median={df['PM25_RH35_GCC'].median():.1f}, "
                  f"missing={df['PM25_RH35_GCC'].isna().sum()}")
            geoscf_frames.append(df)
        elif "merra2" in name and name.endswith(".csv"):
            df = pd.read_csv(zf.open(name))
            year = name.split("_")[-1].replace(".csv", "")
            print(f"\n--- MERRA-2 {year} ---")
            print(f"  Shape: {df.shape}, Columns: {list(df.columns)}")
            print(f"  Stations: {df['stationId'].nunique()}")
            df["datetime"] = pd.to_datetime(df["datetime"])
            print(f"  Date range: {df['datetime'].min()} – {df['datetime'].max()}")
            for col in ["BCSMASS", "OCSMASS", "SO4SMASS", "DUSMASS25", "SSSMASS25", "TOTEXTTAU", "merra2_pm25"]:
                if col in df.columns:
                    pct_na = 100 * df[col].isna().mean()
                    print(f"  {col}: mean={df[col].mean():.4g}, missing={pct_na:.1f}%")
            merra2_frames.append(df)

geoscf = pd.concat(geoscf_frames, ignore_index=True)
merra2 = pd.concat(merra2_frames, ignore_index=True)
geoscf["datetime"] = pd.to_datetime(geoscf["datetime"])
merra2["datetime"] = pd.to_datetime(merra2["datetime"])

print(f"\n--- Combined ---")
print(f"  GEOS-CF: {len(geoscf):,} rows, {geoscf['stationId'].nunique()} stations, "
      f"{geoscf['datetime'].min()} – {geoscf['datetime'].max()}")
print(f"  MERRA-2: {len(merra2):,} rows, {merra2['stationId'].nunique()} stations, "
      f"{merra2['datetime'].min()} – {merra2['datetime'].max()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Compare with ground truth PM2.5
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 2: GROUND TRUTH COMPARISON")
print("=" * 80)

gt = pd.read_csv(BASE / "data" / "merged" / "unified_thesis_v4.csv",
                 usecols=["ts", "PM2.5", "stationId"])  # v4 = definitive (all 40 stations)
# v4 holds all 121 stations; restrict ground truth to the 40 thesis stations.
_thesis40 = set(pd.read_csv(BASE / "analysis" / "thesis_audit" / "station_selection_final.csv",
                            dtype={"stationId": str})["stationId"])
gt = gt[gt["stationId"].isin(_thesis40)].copy()
gt["ts"] = pd.to_datetime(gt["ts"])
gt = gt.dropna(subset=["PM2.5"])
gt = gt[gt["PM2.5"] > 0]
print(f"  Ground truth: {len(gt):,} rows, {gt['stationId'].nunique()} stations")
print(f"  Date range: {gt['ts'].min()} – {gt['ts'].max()}")

# round ground truth to nearest 3-hour for merging
gt["ts_3h"] = gt["ts"].dt.floor("3h")

# filter satellite to model stations
geoscf_m = geoscf[geoscf["stationId"].isin(MODEL_IDS)].copy()
merra2_m = merra2[merra2["stationId"].isin(MODEL_IDS)].copy()
geoscf_m["ts_3h"] = geoscf_m["datetime"].dt.floor("3h")
merra2_m["ts_3h"] = merra2_m["datetime"].dt.floor("3h")

# aggregate ground truth to 3-hourly means
gt_3h = gt.groupby(["stationId", "ts_3h"])["PM2.5"].mean().reset_index()

# merge
merged_gc = gt_3h.merge(geoscf_m[["stationId", "ts_3h", "PM25_RH35_GCC"]],
                        on=["stationId", "ts_3h"], how="inner")
merged_mr = gt_3h.merge(merra2_m[["stationId", "ts_3h", "merra2_pm25"]],
                        on=["stationId", "ts_3h"], how="inner")

# also merge all three together for direct comparison
merged_all = gt_3h.merge(
    geoscf_m[["stationId", "ts_3h", "PM25_RH35_GCC"]],
    on=["stationId", "ts_3h"], how="inner"
).merge(
    merra2_m[["stationId", "ts_3h", "merra2_pm25"]],
    on=["stationId", "ts_3h"], how="inner"
)

print(f"  Merged GEOS-CF: {len(merged_gc):,} rows")
print(f"  Merged MERRA-2: {len(merged_mr):,} rows")
print(f"  Merged all three: {len(merged_all):,} rows")

# per-station metrics
def compute_metrics(df, obs_col, pred_col):
    rows = []
    for sid, g in df.groupby("stationId"):
        if len(g) < 30:
            continue
        obs = g[obs_col].values
        pred = g[pred_col].values
        mask = np.isfinite(obs) & np.isfinite(pred)
        obs, pred = obs[mask], pred[mask]
        if len(obs) < 30:
            continue
        r = np.corrcoef(obs, pred)[0, 1]
        bias = pred.mean() - obs.mean()
        rmse = np.sqrt(np.mean((pred - obs) ** 2))
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append({
            "stationId": sid,
            "tier": TIER_MAP.get(sid, "?"),
            "region": REGION_MAP.get(sid, "?"),
            "n": len(obs),
            "obs_mean": obs.mean(),
            "pred_mean": pred.mean(),
            "r": r,
            "r2": r2,
            "bias": bias,
            "rmse": rmse,
            "bias_pct": 100 * bias / obs.mean() if obs.mean() > 0 else np.nan,
        })
    return pd.DataFrame(rows)

metrics_gc = compute_metrics(merged_gc, "PM2.5", "PM25_RH35_GCC")
metrics_mr = compute_metrics(merged_mr, "PM2.5", "merra2_pm25")

print("\n--- Per-station: GEOS-CF vs Observed ---")
print(f"{'Station':<50s} {'tier':>4s} {'region':>8s} {'obs':>6s} {'gcf':>6s} {'r':>6s} {'R2':>6s} {'bias':>7s} {'b%':>6s} {'rmse':>6s}")
print("-" * 110)
for _, row in metrics_gc.sort_values("r", ascending=False).iterrows():
    nm = NAME_MAP.get(row["stationId"], row["stationId"])[:48]
    print(f"{nm:<50s} {str(row['tier']):>4s} {str(row['region']):>8s} {row['obs_mean']:>6.1f} {row['pred_mean']:>6.1f} "
          f"{row['r']:>6.3f} {row['r2']:>6.3f} {row['bias']:>7.1f} {row['bias_pct']:>5.0f}% {row['rmse']:>6.1f}")

print(f"\n  GEOS-CF overall: median r={metrics_gc['r'].median():.3f}, "
      f"mean r={metrics_gc['r'].mean():.3f}, "
      f"mean bias={metrics_gc['bias'].mean():.1f} µg/m³ ({metrics_gc['bias_pct'].mean():.0f}%), "
      f"mean RMSE={metrics_gc['rmse'].mean():.1f}")

print("\n--- Per-station: MERRA-2 vs Observed ---")
print(f"{'Station':<50s} {'tier':>4s} {'region':>8s} {'obs':>6s} {'mr2':>6s} {'r':>6s} {'R2':>6s} {'bias':>7s} {'b%':>6s} {'rmse':>6s}")
print("-" * 110)
for _, row in metrics_mr.sort_values("r", ascending=False).iterrows():
    nm = NAME_MAP.get(row["stationId"], row["stationId"])[:48]
    print(f"{nm:<50s} {str(row['tier']):>4s} {str(row['region']):>8s} {row['obs_mean']:>6.1f} {row['pred_mean']:>6.1f} "
          f"{row['r']:>6.3f} {row['r2']:>6.3f} {row['bias']:>7.1f} {row['bias_pct']:>5.0f}% {row['rmse']:>6.1f}")

print(f"\n  MERRA-2 overall: median r={metrics_mr['r'].median():.3f}, "
      f"mean r={metrics_mr['r'].mean():.3f}, "
      f"mean bias={metrics_mr['bias'].mean():.1f} µg/m³ ({metrics_mr['bias_pct'].mean():.0f}%), "
      f"mean RMSE={metrics_mr['rmse'].mean():.1f}")

# overall pooled R²
def pooled_r2(df, obs_col, pred_col):
    obs = df[obs_col].dropna().values
    pred = df[pred_col].dropna().values
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    r = np.corrcoef(obs, pred)[0, 1]
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r, r2

r_gc, r2_gc = pooled_r2(merged_gc, "PM2.5", "PM25_RH35_GCC")
r_mr, r2_mr = pooled_r2(merged_mr, "PM2.5", "merra2_pm25")
print(f"\n  POOLED (all station-hours):")
print(f"    GEOS-CF:  r={r_gc:.4f}, R²={r2_gc:.4f}")
print(f"    MERRA-2:  r={r_mr:.4f}, R²={r2_mr:.4f}")

# per-tier summary
print("\n--- Per-tier summary ---")
for product, metrics in [("GEOS-CF", metrics_gc), ("MERRA-2", metrics_mr)]:
    print(f"\n  {product}:")
    for tier in ["t0", "t1", "t2", "t3"]:
        sub = metrics[metrics["tier"] == tier]
        if len(sub) > 0:
            print(f"    {tier}: n={len(sub):2d}, r={sub['r'].mean():.3f} ± {sub['r'].std():.3f}, "
                  f"bias={sub['bias'].mean():+.1f} ({sub['bias_pct'].mean():+.0f}%), "
                  f"RMSE={sub['rmse'].mean():.1f}")

# save metrics
metrics_gc.to_csv(OUT / "geoscf_station_metrics.csv", index=False)
metrics_mr.to_csv(OUT / "merra2_station_metrics.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: GHAP Climatology Assessment
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 3: GHAP CLIMATOLOGY ASSESSMENT")
print("=" * 80)

# compute observed monthly means from ground truth
gt["month"] = gt["ts"].dt.month
obs_monthly = gt.groupby(["stationId", "month"])["PM2.5"].mean().reset_index()
obs_monthly.columns = ["stationId", "month", "obs_monthly_mean"]

# merge with GHAP monthly
ghap_m = ghap_monthly[ghap_monthly["stationId"].isin(MODEL_IDS)].copy()
ghap_m.columns = ["stationId", "month", "ghap_monthly_mean"]
comp_monthly = obs_monthly.merge(ghap_m, on=["stationId", "month"], how="inner")

print(f"\n  Monthly comparisons: {len(comp_monthly)} station-months")

# per-station monthly correlation
print("\n--- Monthly cycle correlation per station ---")
monthly_corrs = []
for sid, g in comp_monthly.groupby("stationId"):
    if len(g) >= 6:
        r = np.corrcoef(g["obs_monthly_mean"], g["ghap_monthly_mean"])[0, 1]
        monthly_corrs.append({"stationId": sid, "r_monthly": r,
                              "tier": TIER_MAP.get(sid, "?"),
                              "obs_ann": g["obs_monthly_mean"].mean(),
                              "ghap_ann": g["ghap_monthly_mean"].mean()})

mc_df = pd.DataFrame(monthly_corrs).sort_values("r_monthly", ascending=False)
for _, row in mc_df.iterrows():
    nm = NAME_MAP.get(row["stationId"], row["stationId"])[:45]
    print(f"  {nm:<47s} {row['tier']:>3s}  obs={row['obs_ann']:5.1f}  ghap={row['ghap_ann']:5.1f}  r_monthly={row['r_monthly']:+.3f}")

print(f"\n  Median monthly-cycle r: {mc_df['r_monthly'].median():.3f}")
print(f"  Mean monthly-cycle r:   {mc_df['r_monthly'].mean():.3f}")

# station ranking: GHAP annual mean vs observed annual mean
obs_annual = gt.groupby("stationId")["PM2.5"].mean().reset_index()
obs_annual.columns = ["stationId", "obs_annual"]
ghap_a = ghap_annual[ghap_annual["stationId"].isin(MODEL_IDS)].copy()
ghap_a.columns = ["stationId", "ghap_annual"]
rank_comp = obs_annual.merge(ghap_a, on="stationId", how="inner")
rank_r = np.corrcoef(rank_comp["obs_annual"], rank_comp["ghap_annual"])[0, 1]
rank_rho, _ = stats.spearmanr(rank_comp["obs_annual"], rank_comp["ghap_annual"])
print(f"\n  Station ranking: Pearson r={rank_r:.3f}, Spearman ρ={rank_rho:.3f} (n={len(rank_comp)} stations)")

# specific station comparisons
print("\n--- Can GHAP distinguish key stations? ---")
key_pairs = [
    ("Bắc Ninh UBND xã Xuân Lâm", "Bắc Ninh Khu liên cơ Thuận Thành", "Industrial vs moderate (Bắc Ninh)"),
    ("Hà Nội ĐHBK (KK)", "Quảng Ninh Phường Cẩm Thịnh", "Urban Hanoi vs clean Quảng Ninh"),
]
for name1, name2, label in key_pairs:
    matches = []
    for target_name in [name1, name2]:
        found = False
        for sid, sname in NAME_MAP.items():
            if target_name in sname:
                obs_val = obs_annual[obs_annual["stationId"] == sid]["obs_annual"].values
                ghap_val = ghap_a[ghap_a["stationId"] == sid]["ghap_annual"].values
                obs_v = obs_val[0] if len(obs_val) > 0 else np.nan
                ghap_v = ghap_val[0] if len(ghap_val) > 0 else np.nan
                matches.append((sname[:40], obs_v, ghap_v))
                found = True
                break
        if not found:
            matches.append((target_name[:40], np.nan, np.nan))
    print(f"\n  {label}:")
    for nm, obs_v, ghap_v in matches:
        print(f"    {nm:<42s}  obs={obs_v:5.1f}  ghap={ghap_v:5.1f}")
    if len(matches) == 2 and not np.isnan(matches[0][1]) and not np.isnan(matches[1][1]):
        obs_diff = matches[0][1] - matches[1][1]
        ghap_diff = matches[0][2] - matches[1][2]
        correct = (obs_diff > 0 and ghap_diff > 0) or (obs_diff < 0 and ghap_diff < 0)
        print(f"    Obs diff: {obs_diff:+.1f}, GHAP diff: {ghap_diff:+.1f} → {'CORRECT ranking' if correct else 'WRONG ranking'}")

# GHAP station ranking plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
rank_comp_plot = rank_comp.copy()
rank_comp_plot["tier"] = rank_comp_plot["stationId"].map(TIER_MAP)
colors = {"t0": "#2196F3", "t1": "#4CAF50", "t2": "#FF9800", "t3": "#F44336"}
for tier in ["t0", "t1", "t2", "t3"]:
    sub = rank_comp_plot[rank_comp_plot["tier"] == tier]
    ax.scatter(sub["obs_annual"], sub["ghap_annual"], c=colors[tier], label=tier, s=50, alpha=0.7)
lims = [0, max(rank_comp_plot["obs_annual"].max(), rank_comp_plot["ghap_annual"].max()) * 1.1]
ax.plot(lims, lims, "k--", alpha=0.3, label="1:1")
ax.set_xlabel("Observed annual mean PM2.5 (µg/m³)")
ax.set_ylabel("GHAP annual mean PM2.5 (µg/m³)")
ax.set_title(f"GHAP vs Observed Station Ranking (r={rank_r:.3f}, ρ={rank_rho:.3f})")
ax.legend()
ax.set_xlim(lims)
ax.set_ylim(lims)
fig.tight_layout()
fig.savefig(OUT / "ghap_station_ranking.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: ghap_station_ranking.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: GEOS-CF Temporal Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 4: GEOS-CF TEMPORAL ANALYSIS")
print("=" * 80)

# hourly merge for diurnal analysis
gt_hr = gt.copy()
gt_hr["hour"] = gt_hr["ts"].dt.hour
gt_hr["ts_3h"] = gt_hr["ts"].dt.floor("3h")

geoscf_hr = geoscf_m.copy()
geoscf_hr["hour"] = geoscf_hr["datetime"].dt.hour

merged_hr = gt_hr.merge(
    geoscf_hr[["stationId", "ts_3h", "PM25_RH35_GCC"]],
    on=["stationId", "ts_3h"], how="inner"
)
merged_hr["hour_3h"] = merged_hr["ts_3h"].dt.hour

# diurnal cycle
print("\n--- Diurnal cycle (3-hourly) ---")
diurnal_obs = merged_hr.groupby("hour_3h")["PM2.5"].mean()
diurnal_gcf = merged_hr.groupby("hour_3h")["PM25_RH35_GCC"].mean()
r_diurnal = np.corrcoef(diurnal_obs.values, diurnal_gcf.values)[0, 1]
print(f"  Hours: {list(diurnal_obs.index)}")
print(f"  Obs:     {['%.1f' % v for v in diurnal_obs.values]}")
print(f"  GEOS-CF: {['%.1f' % v for v in diurnal_gcf.values]}")
print(f"  Diurnal cycle r: {r_diurnal:.3f}")

# seasonal (monthly) variation
merged_hr["yearmonth"] = merged_hr["ts"].dt.to_period("M")
monthly_obs = merged_hr.groupby("yearmonth")["PM2.5"].mean()
monthly_gcf = merged_hr.groupby("yearmonth")["PM25_RH35_GCC"].mean()
r_seasonal = np.corrcoef(monthly_obs.values, monthly_gcf.values)[0, 1]
print(f"\n--- Seasonal (monthly mean) ---")
print(f"  Monthly correlation: r={r_seasonal:.3f}")
print(f"  Obs range:     {monthly_obs.min():.1f} – {monthly_obs.max():.1f}")
print(f"  GEOS-CF range: {monthly_gcf.min():.1f} – {monthly_gcf.max():.1f}")

# bias pattern: by pollution level
print("\n--- Bias by pollution level ---")
bins = [0, 10, 20, 35, 50, 75, 100, 500]
labels = ["0-10", "10-20", "20-35", "35-50", "50-75", "75-100", "100+"]
merged_hr["pm_bin"] = pd.cut(merged_hr["PM2.5"], bins=bins, labels=labels)
for bn in labels:
    sub = merged_hr[merged_hr["pm_bin"] == bn]
    if len(sub) > 100:
        bias = sub["PM25_RH35_GCC"].mean() - sub["PM2.5"].mean()
        ratio = sub["PM25_RH35_GCC"].mean() / sub["PM2.5"].mean()
        r_bin = np.corrcoef(sub["PM2.5"], sub["PM25_RH35_GCC"])[0, 1]
        print(f"  PM2.5 {bn:>7s}: n={len(sub):>6d}, obs={sub['PM2.5'].mean():5.1f}, "
              f"gcf={sub['PM25_RH35_GCC'].mean():5.1f}, "
              f"bias={bias:+6.1f} (ratio={ratio:.2f}), r={r_bin:.3f}")

# scatter plots for 3 representative stations
rep_stations = []
for tier_target in ["t0", "t2", "t3"]:
    sub = metrics_gc[metrics_gc["tier"] == tier_target].sort_values("r", ascending=False)
    if len(sub) > 0:
        rep_stations.append(sub.iloc[len(sub)//2]["stationId"])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, sid in zip(axes, rep_stations):
    sub = merged_gc[merged_gc["stationId"] == sid]
    ax.scatter(sub["PM2.5"], sub["PM25_RH35_GCC"], s=2, alpha=0.15, c="#1976D2")
    lim = max(sub["PM2.5"].max(), sub["PM25_RH35_GCC"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    r_val = np.corrcoef(sub["PM2.5"], sub["PM25_RH35_GCC"])[0, 1]
    nm = NAME_MAP.get(sid, sid)[:30]
    tier = TIER_MAP.get(sid, "?")
    ax.set_title(f"{nm}\n{tier}, r={r_val:.3f}", fontsize=9)
    ax.set_xlabel("Observed PM2.5")
    ax.set_ylabel("GEOS-CF PM2.5")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
fig.suptitle("GEOS-CF vs Observed PM2.5 (3-hourly)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "geoscf_scatter_representative.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: geoscf_scatter_representative.png")

# diurnal cycle plot
fig, ax = plt.subplots(figsize=(8, 5))
hours = diurnal_obs.index
ax.plot(hours, diurnal_obs.values, "o-", label="Observed", color="#1976D2", linewidth=2)
ax.plot(hours, diurnal_gcf.values, "s-", label="GEOS-CF", color="#F44336", linewidth=2)
ax.set_xlabel("Hour (UTC+7)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Diurnal Cycle: GEOS-CF vs Observed (all stations)")
ax.legend()
ax.set_xticks(hours)
fig.tight_layout()
fig.savefig(OUT / "geoscf_diurnal_cycle.png", dpi=150)
plt.close(fig)
print(f"  Saved: geoscf_diurnal_cycle.png")

# seasonal cycle plot
fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(monthly_obs))
ax.plot(x, monthly_obs.values, "o-", label="Observed", color="#1976D2", linewidth=2)
ax.plot(x, monthly_gcf.values, "s-", label="GEOS-CF", color="#F44336", linewidth=2)
ax.set_xlabel("Month")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title(f"Monthly Means: GEOS-CF vs Observed (r={r_seasonal:.3f})")
ax.legend()
xticks_labels = [str(p) for p in monthly_obs.index]
ax.set_xticks(x)
ax.set_xticklabels(xticks_labels, rotation=45, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "geoscf_seasonal_cycle.png", dpi=150)
plt.close(fig)
print(f"  Saved: geoscf_seasonal_cycle.png")

# bias vs observed scatter
fig, ax = plt.subplots(figsize=(8, 6))
merged_gc_sample = merged_gc.sample(min(50000, len(merged_gc)), random_state=42)
ax.scatter(merged_gc_sample["PM2.5"],
           merged_gc_sample["PM25_RH35_GCC"] - merged_gc_sample["PM2.5"],
           s=1, alpha=0.05, c="#1976D2")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Observed PM2.5 (µg/m³)")
ax.set_ylabel("GEOS-CF bias (µg/m³)")
ax.set_title("GEOS-CF Bias vs Pollution Level")
fig.tight_layout()
fig.savefig(OUT / "geoscf_bias_pattern.png", dpi=150)
plt.close(fig)
print(f"  Saved: geoscf_bias_pattern.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: MERRA-2 Species Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 5: MERRA-2 SPECIES ANALYSIS")
print("=" * 80)

species_cols = ["BCSMASS", "OCSMASS", "SO4SMASS", "DUSMASS25", "SSSMASS25"]
species_labels = {"BCSMASS": "Black Carbon", "OCSMASS": "Organic Carbon",
                  "SO4SMASS": "Sulfate", "DUSMASS25": "Dust (PM2.5)",
                  "SSSMASS25": "Sea Salt (PM2.5)"}

merra2_40 = merra2[merra2["stationId"].isin(MODEL_IDS)].copy()

# overall species composition
print("\n--- Overall aerosol species composition (µg/m³ surface) ---")
for col in species_cols:
    vals = merra2_40[col].dropna()
    # convert kg/m³ to µg/m³ (×1e9)
    mean_ugm3 = vals.mean() * 1e9
    print(f"  {species_labels[col]:<20s}: {mean_ugm3:.2f} µg/m³")

total_mass = sum(merra2_40[c].mean() for c in species_cols)
print(f"\n  Fractional composition:")
for col in species_cols:
    frac = merra2_40[col].mean() / total_mass * 100
    print(f"  {species_labels[col]:<20s}: {frac:.1f}%")

# by region
print("\n--- Species composition by region ---")
merra2_40["region"] = merra2_40["stationId"].map(REGION_MAP)
for region in ["North", "Central", "South"]:
    sub = merra2_40[merra2_40["region"] == region]
    if len(sub) == 0:
        continue
    total = sum(sub[c].mean() for c in species_cols)
    print(f"\n  {region}:")
    for col in species_cols:
        frac = sub[col].mean() / total * 100
        mean_ugm3 = sub[col].mean() * 1e9
        print(f"    {species_labels[col]:<20s}: {frac:5.1f}% ({mean_ugm3:.2f} µg/m³)")

# per-station dominant species
print("\n--- Dominant species per station ---")
dom_rows = []
for sid, g in merra2_40.groupby("stationId"):
    fracs = {}
    total = sum(g[c].mean() for c in species_cols)
    for col in species_cols:
        fracs[col] = g[col].mean() / total * 100
    dominant = max(fracs, key=fracs.get)
    dom_rows.append({
        "stationId": sid,
        "tier": TIER_MAP.get(sid, "?"),
        "region": REGION_MAP.get(sid, "?"),
        "dominant": species_labels[dominant],
        "dominant_pct": fracs[dominant],
        **{species_labels[c]: fracs[c] for c in species_cols}
    })
dom_df = pd.DataFrame(dom_rows).sort_values(["region", "tier"])

for _, row in dom_df.iterrows():
    nm = NAME_MAP.get(row["stationId"], row["stationId"])[:40]
    print(f"  {nm:<42s} {row['tier']:>3s} {row['region']:>8s}  "
          f"BC={row['Black Carbon']:4.1f}% OC={row['Organic Carbon']:4.1f}% "
          f"SO4={row['Sulfate']:4.1f}% Dust={row['Dust (PM2.5)']:4.1f}% SS={row['Sea Salt (PM2.5)']:4.1f}%  "
          f"→ {row['dominant']}")

# MERRA-2 species correlation with observed PM2.5
print("\n--- MERRA-2 species correlation with observed PM2.5 ---")
merra2_hr = merra2_m.copy()
merged_species = gt_3h.merge(
    merra2_hr[["stationId", "ts_3h"] + species_cols + ["merra2_pm25", "TOTEXTTAU"]],
    on=["stationId", "ts_3h"], how="inner"
)
print(f"  Merged rows: {len(merged_species):,}")

for col in species_cols + ["TOTEXTTAU", "merra2_pm25"]:
    valid = merged_species[["PM2.5", col]].dropna()
    if len(valid) > 100:
        r = np.corrcoef(valid["PM2.5"], valid[col])[0, 1]
        label = species_labels.get(col, col)
        print(f"  {label:<20s}: r={r:.4f} (n={len(valid):,})")

# MERRA-2 vs GEOS-CF head-to-head
print("\n--- MERRA-2 vs GEOS-CF head-to-head (same timestamps) ---")
r_gc_h2h, r2_gc_h2h = pooled_r2(merged_all, "PM2.5", "PM25_RH35_GCC")
r_mr_h2h, r2_mr_h2h = pooled_r2(merged_all, "PM2.5", "merra2_pm25")
print(f"  GEOS-CF:  r={r_gc_h2h:.4f}, R²={r2_gc_h2h:.4f}")
print(f"  MERRA-2:  r={r_mr_h2h:.4f}, R²={r2_mr_h2h:.4f}")
print(f"  Winner: {'GEOS-CF' if r_gc_h2h > r_mr_h2h else 'MERRA-2'}")

# species composition bar chart
fig, ax = plt.subplots(figsize=(10, 6))
regions = ["North", "Central", "South"]
x = np.arange(len(regions))
width = 0.15
species_colors = ["#333333", "#8BC34A", "#FFC107", "#FF9800", "#03A9F4"]
for i, col in enumerate(species_cols):
    vals = []
    for region in regions:
        sub = merra2_40[merra2_40["region"] == region]
        total = sum(sub[c].mean() for c in species_cols)
        vals.append(sub[col].mean() / total * 100 if total > 0 else 0)
    ax.bar(x + i * width, vals, width, label=species_labels[col], color=species_colors[i])
ax.set_xticks(x + 2 * width)
ax.set_xticklabels(regions)
ax.set_ylabel("Fraction (%)")
ax.set_title("MERRA-2 Aerosol Species Composition by Region")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "merra2_species_by_region.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: merra2_species_by_region.png")

# MERRA-2 scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, sid in zip(axes, rep_stations):
    sub = merged_mr[merged_mr["stationId"] == sid]
    ax.scatter(sub["PM2.5"], sub["merra2_pm25"], s=2, alpha=0.15, c="#4CAF50")
    lim = max(sub["PM2.5"].max(), sub["merra2_pm25"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    r_val = np.corrcoef(sub["PM2.5"], sub["merra2_pm25"])[0, 1]
    nm = NAME_MAP.get(sid, sid)[:30]
    tier = TIER_MAP.get(sid, "?")
    ax.set_title(f"{nm}\n{tier}, r={r_val:.3f}", fontsize=9)
    ax.set_xlabel("Observed PM2.5")
    ax.set_ylabel("MERRA-2 PM2.5")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
fig.suptitle("MERRA-2 vs Observed PM2.5 (3-hourly)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "merra2_scatter_representative.png", dpi=150)
plt.close(fig)
print(f"  Saved: merra2_scatter_representative.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Feature Engineering Recommendations
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 6: FEATURE ENGINEERING RECOMMENDATIONS")
print("=" * 80)

# compute some additional diagnostics for recommendations
# bias-corrected GEOS-CF
merged_gc_bc = merged_gc.copy()
# per-station linear bias correction
bc_results = []
for sid, g in merged_gc_bc.groupby("stationId"):
    if len(g) < 100:
        continue
    slope, intercept, r, p, se = stats.linregress(g["PM25_RH35_GCC"], g["PM2.5"])
    g_corr = g.copy()
    g_corr["gcf_corrected"] = slope * g["PM25_RH35_GCC"] + intercept
    ss_res = np.sum((g["PM2.5"] - g_corr["gcf_corrected"]) ** 2)
    ss_tot = np.sum((g["PM2.5"] - g["PM2.5"].mean()) ** 2)
    r2_corr = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    bc_results.append({"stationId": sid, "tier": TIER_MAP.get(sid, "?"),
                       "r2_raw": metrics_gc[metrics_gc["stationId"]==sid]["r2"].values[0] if len(metrics_gc[metrics_gc["stationId"]==sid]) > 0 else np.nan,
                       "r2_corrected": r2_corr,
                       "slope": slope, "intercept": intercept})

bc_df = pd.DataFrame(bc_results)
print("\n--- Bias-corrected GEOS-CF R² improvement ---")
for tier in ["t0", "t1", "t2", "t3"]:
    sub = bc_df[bc_df["tier"] == tier]
    if len(sub) > 0:
        print(f"  {tier}: raw R²={sub['r2_raw'].mean():.3f} → corrected R²={sub['r2_corrected'].mean():.3f} "
              f"(Δ={sub['r2_corrected'].mean()-sub['r2_raw'].mean():+.3f})")

# anomaly-based features
merged_gc_anom = merged_gc.copy()
merged_gc_anom["month"] = merged_gc_anom["ts_3h"].dt.month
# compute monthly mean GEOS-CF per station
gc_monthly_mean = merged_gc_anom.groupby(["stationId", "month"])["PM25_RH35_GCC"].transform("mean")
merged_gc_anom["gcf_anomaly"] = merged_gc_anom["PM25_RH35_GCC"] - gc_monthly_mean
merged_gc_anom["obs_anomaly"] = merged_gc_anom["PM2.5"] - merged_gc_anom.groupby(["stationId", "month"])["PM2.5"].transform("mean")

r_anomaly, _ = stats.pearsonr(merged_gc_anom["obs_anomaly"].dropna(),
                               merged_gc_anom["gcf_anomaly"].dropna())
print(f"\n  GEOS-CF anomaly correlation (deseasonalized): r={r_anomaly:.4f}")

# ratio feature
merged_gc_anom["gcf_ratio"] = merged_gc_anom["PM25_RH35_GCC"] / gc_monthly_mean
merged_gc_anom["obs_ratio"] = merged_gc_anom["PM2.5"] / merged_gc_anom.groupby(["stationId", "month"])["PM2.5"].transform("mean")
valid_ratio = merged_gc_anom.dropna(subset=["gcf_ratio", "obs_ratio"])
valid_ratio = valid_ratio[np.isfinite(valid_ratio["gcf_ratio"]) & np.isfinite(valid_ratio["obs_ratio"])]
r_ratio, _ = stats.pearsonr(valid_ratio["obs_ratio"], valid_ratio["gcf_ratio"])
print(f"  GEOS-CF ratio correlation (normalized):       r={r_ratio:.4f}")

# GEOS-CF species as features
geoscf_species = geoscf_m[["stationId", "ts_3h", "CO", "NO2", "SO2"]].copy()
merged_gcspecies = gt_3h.merge(geoscf_species, on=["stationId", "ts_3h"], how="inner")
print(f"\n--- GEOS-CF species correlation with observed PM2.5 ---")
for col in ["CO", "NO2", "SO2"]:
    valid = merged_gcspecies[["PM2.5", col]].dropna()
    if len(valid) > 100:
        r_val = np.corrcoef(valid["PM2.5"], valid[col])[0, 1]
        print(f"  GEOS-CF {col}: r={r_val:.4f}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("""
1. GEOS-CF PM2.5 (PM25_RH35_GCC):
   - Strong temporal correlation but large systematic overestimate
   - USE AS: anomaly feature (gcf_pm25 - station_monthly_clim) or ratio (gcf_pm25 / monthly_clim)
   - DO NOT use raw values — the bias will dominate
   - The anomaly captures day-to-day pollution events that satellite alone misses
   - Also consider: gcf_pm25 / ghap_clim as a "relative pollution index"

2. MERRA-2 PM2.5 (merra2_pm25):
   - Compare head-to-head with GEOS-CF above
   - USE AS: additional anomaly feature; species fractions as composition indicators
   - Key species: BC+OC fraction (industrial signature), SO4 (secondary aerosol)

3. MERRA-2 Species:
   - BCSMASS + OCSMASS: carbonaceous aerosol — industrial/combustion proxy
   - SO4SMASS: secondary sulfate — regional transport indicator
   - DUSMASS25: dust — relevant for Central/South
   - SSSMASS25: sea salt — coastal stations only
   - USE AS: species ratios (BC_frac, OC_frac, SO4_frac) capture aerosol type
   - TOTEXTTAU: total extinction AOD from MERRA-2 — independent AOD estimate

4. GHAP Climatology:
   - Good station ranking ability — captures spatial PM2.5 gradient
   - USE AS: station-level offset / monthly lookup table
   - ghap_monthly_clim as a feature captures the expected seasonal baseline
   - Useful for new/unseen stations in LOSO — provides prior PM2.5 level

5. Recommended new features for XGBoost:
   a) gcf_pm25_anomaly = PM25_RH35_GCC - ghap_monthly_clim[station, month]
   b) gcf_pm25_ratio   = PM25_RH35_GCC / ghap_monthly_clim[station, month]
   c) merra2_pm25_anomaly (same logic)
   d) bc_oc_frac = (BCSMASS + OCSMASS) / total_species_mass
   e) so4_frac   = SO4SMASS / total_species_mass
   f) ghap_monthly_clim (as-is, monthly lookup)
   g) ghap_annual_mean (station-level, captures spatial gradient)
   h) TOTEXTTAU (independent AOD, 0.5° resolution)
   i) gcf_co (GEOS-CF CO — combustion tracer)
""")

print("=" * 80)
print("DONE")
print("=" * 80)
