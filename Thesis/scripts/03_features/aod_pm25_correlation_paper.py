"""
aod_pm25_correlation_paper.py — Comprehensive AOD-PM2.5 correlation analysis
for paper: "Evaluating satellite AOD capability to monitor PM2.5 in Vietnam."

Uses data/merged/unified_aod_filled.csv, same_hour rows only (real AOD).

Progressive filter levels:
  L0: No filter (all same_hour)
  L1: Unc <= 0.5 (quality filter)
  L2: RF >= 0.5 (fine-mode filter)
  L3: Unc <= 0.5 AND RF >= 0.5
  L4: L3 + RANSAC robust fit

Sections A-E: hourly correlations.
Section F: monthly-mean AOD vs monthly-mean PM2.5 (temporal aggregation).
Section G: normalized hourly AOD (within-month signal).
Section H: between-station (spatial) correlation per month.

Outputs:
  outputs/reports/aod_pm25_per_station.csv       — per station × filter level
  outputs/reports/aod_pm25_region_season.csv      — region × season × filter
  outputs/reports/aod_pm25_ssa_regime.csv          — SSA regime × region
  outputs/reports/aod_pm25_physics.csv             — physics-corrected × filter
  outputs/reports/aod_pm25_availability.csv        — AOD availability
  outputs/reports/aod_pm25_monthly_clim.csv        — monthly clim correlations
  outputs/reports/aod_pm25_normalized.csv          — normalized AOD correlations
  outputs/reports/aod_pm25_spatial_corr.csv        — spatial correlation per month
  outputs/reports/aod_pm25_correlation_paper.csv   — master CSV
  outputs/figures/paper_correlations/*.png
"""

import io, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression

BASE = Path(__file__).resolve().parents[3]
MERGE_DIR = BASE / "data" / "merged"
RPT_DIR = BASE / "outputs" / "reports"
FIG_DIR = BASE / "outputs" / "figures" / "paper_correlations"
for d in [RPT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}

FILTER_LABELS = {
    "L0": "No filter",
    "L1": "Unc <= 0.5",
    "L2": "RF >= 0.5",
    "L3": "Unc<=0.5 & RF>=0.5",
    "L4": "L3 + RANSAC",
}


def season_of(month):
    for s, months in SEASONS.items():
        if month in months:
            return s
    return None


def pearson_r_p(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    r, p = stats.pearsonr(x, y)
    return r, p, len(x)


def ransac_fit(x, y, residual_threshold=None):
    """Fit RANSAC. Returns r², inlier_frac, slope, intercept, n."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask].reshape(-1, 1), y[mask]
    n = len(y)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan, n
    if residual_threshold is None:
        residual_threshold = np.std(y) * 1.5
    try:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=0.5,
            residual_threshold=residual_threshold,
            random_state=42, max_trials=500,
        )
        ransac.fit(x, y)
        inliers = ransac.inlier_mask_
        inlier_frac = inliers.sum() / n
        y_pred = ransac.predict(x[inliers])
        ss_res = np.sum((y[inliers] - y_pred) ** 2)
        ss_tot = np.sum((y[inliers] - y[inliers].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        slope = float(ransac.estimator_.coef_[0])
        intercept = float(ransac.estimator_.intercept_)
        return r2, inlier_frac, slope, intercept, n
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, n


def apply_filters(data):
    """Return dict of filter_level -> boolean mask (relative to data)."""
    n = len(data)
    m0 = np.ones(n, dtype=bool)
    m1 = (data["uncertainty"] <= 0.5).values
    m2 = (data["RF"] >= 0.5).values
    m3 = m1 & m2
    return {"L0": m0, "L1": m1, "L2": m2, "L3": m3}


def correlate_at_levels(data, x_col="AOT", y_col="PM25"):
    """Run Pearson + RANSAC at each filter level. Returns list of dicts."""
    masks = apply_filters(data)
    rows = []
    for lvl in ["L0", "L1", "L2", "L3"]:
        sub = data[masks[lvl]]
        x = sub[x_col].values
        y = sub[y_col].values
        r, p, n = pearson_r_p(x, y)
        r2_ransac, inlier_pct, slope, intercept, _ = ransac_fit(x, y)
        rows.append(dict(
            filter_level=lvl, filter_desc=FILTER_LABELS[lvl], n=n,
            pearson_r=r, r_squared=r**2 if not np.isnan(r) else np.nan,
            p_value=p,
            ransac_r2=r2_ransac, ransac_inlier_pct=inlier_pct,
            ransac_slope=slope, ransac_intercept=intercept,
        ))
    # L4 = L3 data, RANSAC inliers only → report inlier-only Pearson r
    sub3 = data[masks["L3"]]
    x3 = sub3[x_col].values
    y3 = sub3[y_col].values
    fmask = np.isfinite(x3) & np.isfinite(y3)
    x3c, y3c = x3[fmask], y3[fmask]
    if len(x3c) >= 10:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=0.5,
            residual_threshold=np.std(y3c) * 1.5,
            random_state=42, max_trials=500,
        )
        ransac.fit(x3c.reshape(-1, 1), y3c)
        inliers = ransac.inlier_mask_
        inlier_frac = inliers.sum() / len(x3c)
        x_in, y_in = x3c[inliers], y3c[inliers]
        r_in, p_in, n_in = pearson_r_p(x_in, y_in)
        slope = float(ransac.estimator_.coef_[0])
        intercept = float(ransac.estimator_.intercept_)
        # RANSAC r² on inliers
        y_pred = ransac.predict(x_in.reshape(-1, 1))
        ss_res = np.sum((y_in - y_pred) ** 2)
        ss_tot = np.sum((y_in - y_in.mean()) ** 2)
        r2_in = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        r_in, p_in, n_in = np.nan, np.nan, 0
        inlier_frac = np.nan
        slope, intercept, r2_in = np.nan, np.nan, np.nan

    rows.append(dict(
        filter_level="L4", filter_desc=FILTER_LABELS["L4"],
        n=n_in, pearson_r=r_in,
        r_squared=r_in**2 if not np.isnan(r_in) else np.nan,
        p_value=p_in,
        ransac_r2=r2_in, ransac_inlier_pct=inlier_frac,
        ransac_slope=slope, ransac_intercept=intercept,
    ))
    return rows


# ══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("Loading data")
print("=" * 72)

full = pd.read_csv(MERGE_DIR / "unified_aod_filled.csv",
                   dtype={"stationId": str}, parse_dates=["ts"])
full = full.sort_values(["stationId", "ts"]).reset_index(drop=True)
print(f"Full dataset: {len(full):,} rows, {full['stationId'].nunique()} stations")

df = full[full["aod_source"] == "same_hour"].copy()
df["season"] = df["month"].map(season_of)
print(f"Same-hour rows: {len(df):,} ({100*len(df)/len(full):.1f}%)")

sid_to_name = df.drop_duplicates("stationId").set_index("stationId")["station"].to_dict()
sid_to_region = df.drop_duplicates("stationId").set_index("stationId")["region"].to_dict()
station_ids = sorted(df["stationId"].unique())
print(f"Stations: {len(station_ids)}")

# Physics-corrected AOD
df["AOD_physics"] = (df["AOT"]
                     * ((1 - df["Humidity"] / 100.0).clip(0.01) ** 0.6)
                     / df["PBLH"].clip(lower=50))

# Filter counts
for lvl, m in apply_filters(df).items():
    print(f"  {lvl} ({FILTER_LABELS[lvl]:25s}):  n={m.sum():>6}  "
          f"({100*m.mean():.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
#  SECTION A — Per-station × filter level
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION A: Per-station AOT vs PM2.5 — progressive filters")
print("=" * 72)

a_rows = []
for sid in station_ids:
    sub = df[df["stationId"] == sid]
    sname = sid_to_name[sid]
    region = sid_to_region[sid]
    level_rows = correlate_at_levels(sub)
    for lr in level_rows:
        lr["station"] = sname
        lr["region"] = region
        lr["stationId"] = sid
        lr["mean_PM25"] = sub["PM25"].mean()
        lr["mean_AOT"] = sub["AOT"].mean()
    a_rows.extend(level_rows)

a_df = pd.DataFrame(a_rows)

# Print compact table: station | L0_r | L1_r | L2_r | L3_r | L4_r | L3_n
print(f"\n  {'Station':<20s} {'Rgn':<5s} "
      f"{'L0_r':>6s} {'L1_r':>6s} {'L2_r':>6s} {'L3_r':>6s} {'L4_r':>6s} "
      f"{'L3_n':>5s} {'L0_n':>5s}")
print("  " + "-" * 70)
for sid in station_ids:
    sname = sid_to_name[sid]
    region = sid_to_region[sid]
    sdf = a_df[a_df["stationId"] == sid].set_index("filter_level")
    vals = {lvl: sdf.loc[lvl, "pearson_r"] if lvl in sdf.index else np.nan
            for lvl in ["L0", "L1", "L2", "L3", "L4"]}
    n3 = sdf.loc["L3", "n"] if "L3" in sdf.index else 0
    n0 = sdf.loc["L0", "n"] if "L0" in sdf.index else 0
    print(f"  {sname:<20s} {region:<5s} "
          f"{vals['L0']:>+6.3f} {vals['L1']:>+6.3f} {vals['L2']:>+6.3f} "
          f"{vals['L3']:>+6.3f} {vals['L4']:>+6.3f} "
          f"{int(n3):>5} {int(n0):>5}")

# Summary per level
print(f"\n  Mean Pearson r across stations by filter level:")
for lvl in ["L0", "L1", "L2", "L3", "L4"]:
    sub = a_df[a_df["filter_level"] == lvl]
    print(f"    {lvl} ({FILTER_LABELS[lvl]:25s}):  mean_r={sub['pearson_r'].mean():+.3f}  "
          f"mean_r²={sub['r_squared'].mean():.4f}  "
          f"mean_n={sub['n'].mean():.0f}")


# ══════════════════════════════════════════════════════════════════════════
#  SECTION B — Region × Season × filter level
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION B: Region × Season — progressive filters")
print("=" * 72)

b_rows = []
regions = ["North", "Central", "South"]
season_order = ["DJF", "MAM", "JJA", "SON"]

for region in regions + ["ALL"]:
    for season in season_order + ["ALL"]:
        if region == "ALL":
            sub = df if season == "ALL" else df[df["season"] == season]
        else:
            sub = df[df["region"] == region]
            if season != "ALL":
                sub = sub[sub["season"] == season]

        level_rows = correlate_at_levels(sub)
        for lr in level_rows:
            lr["region"] = region
            lr["season"] = season
        b_rows.extend(level_rows)

b_df = pd.DataFrame(b_rows)

# Print compact: region × season at L3 (the most interesting)
print(f"\n  Pearson r at L3 (Unc<=0.5 & RF>=0.5):")
print(f"  {'Region':<8s} {'DJF':>8s} {'MAM':>8s} {'JJA':>8s} {'SON':>8s} {'ALL':>8s}")
for region in regions + ["ALL"]:
    vals = {}
    for season in season_order + ["ALL"]:
        sub = b_df[(b_df["region"] == region) & (b_df["season"] == season)
                    & (b_df["filter_level"] == "L3")]
        vals[season] = sub["pearson_r"].values[0] if len(sub) else np.nan
    print(f"  {region:<8s} {vals['DJF']:>+8.3f} {vals['MAM']:>+8.3f} "
          f"{vals['JJA']:>+8.3f} {vals['SON']:>+8.3f} {vals['ALL']:>+8.3f}")

# Print L0 vs L3 comparison for region ALL
print(f"\n  Filter progression (ALL regions, ALL seasons):")
for lvl in ["L0", "L1", "L2", "L3", "L4"]:
    sub = b_df[(b_df["region"] == "ALL") & (b_df["season"] == "ALL")
               & (b_df["filter_level"] == lvl)]
    if len(sub):
        r = sub.iloc[0]
        print(f"    {lvl} ({FILTER_LABELS[lvl]:25s}):  r={r['pearson_r']:+.3f}  "
              f"n={int(r['n']):>5}  RANSAC_r²={r['ransac_r2']:.4f}")


# ══════════════════════════════════════════════════════════════════════════
#  SECTION C — SSA regime × region
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION C: SSA regime split per region")
print("=" * 72)

c_rows = []
for region in regions + ["ALL"]:
    sub = df if region == "ALL" else df[df["region"] == region]
    for ssa_label, ssa_cond in [("SSA<0.92", sub["SSA"] < 0.92),
                                 ("SSA>=0.92", sub["SSA"] >= 0.92)]:
        s = sub[ssa_cond]
        x, y = s["AOT"].values, s["PM25"].values
        r, p, n = pearson_r_p(x, y)
        r2_ransac, inlier_pct, _, _, _ = ransac_fit(x, y)
        c_rows.append(dict(
            region=region, ssa_regime=ssa_label, n=n,
            pearson_r=r, r_squared=r**2 if not np.isnan(r) else np.nan,
            p_value=p, ransac_r2=r2_ransac,
            mean_PM25=s["PM25"].mean() if len(s) else np.nan,
            mean_AOT=s["AOT"].mean() if len(s) else np.nan,
        ))
        print(f"  {region:7s} × {ssa_label:10s}:  n={n:>5}  r={r:+.3f}  "
              f"RANSAC_r²={r2_ransac:.3f}  mean_PM25={s['PM25'].mean():.1f}")

c_df = pd.DataFrame(c_rows)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION D — Physics-corrected AOD × filter level
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION D: Physics-corrected AOD — progressive filters")
print("=" * 72)

d_rows = []
phys_valid = df[df["AOD_physics"].notna() & np.isfinite(df["AOD_physics"])
                & (df["AOD_physics"] > 0)].copy()

for sid in station_ids:
    sub = phys_valid[phys_valid["stationId"] == sid]
    sname = sid_to_name[sid]
    region = sid_to_region[sid]

    raw_levels = correlate_at_levels(sub, x_col="AOT", y_col="PM25")
    phys_levels = correlate_at_levels(sub, x_col="AOD_physics", y_col="PM25")

    for rl, pl in zip(raw_levels, phys_levels):
        lvl = rl["filter_level"]
        d_rows.append(dict(
            station=sname, region=region, filter_level=lvl,
            filter_desc=FILTER_LABELS[lvl], n=rl["n"],
            r_AOT=rl["pearson_r"], r2_AOT=rl["r_squared"],
            r_AOD_physics=pl["pearson_r"], r2_AOD_physics=pl["r_squared"],
            delta_r=pl["pearson_r"] - rl["pearson_r"] if not (np.isnan(pl["pearson_r"]) or np.isnan(rl["pearson_r"])) else np.nan,
            ransac_r2_AOT=rl["ransac_r2"],
            ransac_r2_physics=pl["ransac_r2"],
        ))

d_df = pd.DataFrame(d_rows)

# Print compact table at L3
print(f"\n  Physics correction at L3 (Unc<=0.5 & RF>=0.5):")
print(f"  {'Station':<20s} {'r_AOT':>7s} {'r_phys':>7s} {'Δr':>7s} {'n':>5s}")
print("  " + "-" * 50)
d3 = d_df[d_df["filter_level"] == "L3"]
for _, r in d3.sort_values("delta_r", ascending=False).iterrows():
    print(f"  {r['station']:<20s} {r['r_AOT']:>+7.3f} {r['r_AOD_physics']:>+7.3f} "
          f"{r['delta_r']:>+7.3f} {int(r['n']):>5}")

print(f"\n  Physics correction summary by filter level:")
for lvl in ["L0", "L1", "L2", "L3", "L4"]:
    sub = d_df[d_df["filter_level"] == lvl]
    print(f"    {lvl}: mean_r_AOT={sub['r_AOT'].mean():+.3f}  "
          f"mean_r_phys={sub['r_AOD_physics'].mean():+.3f}  "
          f"mean_Δr={sub['delta_r'].mean():+.3f}  "
          f"improved={int((sub['delta_r']>0).sum())}/{len(sub)}")


# ══════════════════════════════════════════════════════════════════════════
#  SECTION E — AOD availability
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION E: AOD availability")
print("=" * 72)

e_rows_station = []
for sid in station_ids:
    sname = sid_to_name[sid]
    region = sid_to_region[sid]
    total = len(full[full["stationId"] == sid])
    valid = len(full[(full["stationId"] == sid) & (full["aod_source"] == "same_hour")])
    pct = 100 * valid / total if total > 0 else 0
    e_rows_station.append(dict(
        station=sname, region=region, total_hours=total,
        valid_aod_hours=valid, pct_valid=pct,
    ))
    print(f"  {sname:20s}  {valid:>5}/{total:>6} = {pct:5.1f}%")

e_station_df = pd.DataFrame(e_rows_station)

e_rows_month = []
for month in range(1, 13):
    for sid in station_ids:
        sname = sid_to_name[sid]
        region = sid_to_region[sid]
        total = len(full[(full["stationId"] == sid) & (full["month"] == month)])
        valid = len(full[(full["stationId"] == sid) & (full["month"] == month)
                         & (full["aod_source"] == "same_hour")])
        pct = 100 * valid / total if total > 0 else 0
        e_rows_month.append(dict(
            station=sname, region=region, month=month,
            total_hours=total, valid_aod_hours=valid, pct_valid=pct,
        ))

e_month_df = pd.DataFrame(e_rows_month)

month_summary = e_month_df.groupby("month").agg(
    total=("total_hours", "sum"), valid=("valid_aod_hours", "sum")).reset_index()
month_summary["pct"] = 100 * month_summary["valid"] / month_summary["total"]
print(f"\n  Monthly AOD availability (all stations):")
for _, r in month_summary.iterrows():
    bar = "█" * int(r["pct"] / 2)
    print(f"    Month {int(r['month']):>2}: {r['pct']:5.1f}% {bar}")


# ══════════════════════════════════════════════════════════════════════════
#  SECTION F — Monthly-mean AOD vs monthly-mean PM2.5 (per year×month)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION F: Monthly-mean AOD vs PM2.5 (station × year × month)")
print("=" * 72)

# Derive year column
full["year"] = full["ts"].dt.year
df["year"] = df["ts"].dt.year

# Step 1 — Compute monthly means per station × year × month
# AOT monthly mean: from same_hour rows with valid AOT only
aot_monthly = (df[df["AOT"].notna()]
               .groupby(["stationId", "year", "month"])
               .agg(AOT_monthly_mean=("AOT", "mean"),
                    AOD_physics_monthly_mean=("AOD_physics", "mean"),
                    n_aod_hours=("AOT", "size"))
               .reset_index())

# PM2.5 monthly mean: from ALL hours (not just clear-sky)
pm25_monthly = (full.groupby(["stationId", "year", "month"])
                .agg(PM25_monthly_mean=("PM25", "mean"),
                     n_pm25_hours=("PM25", "size"))
                .reset_index())

f_monthly = aot_monthly.merge(pm25_monthly, on=["stationId", "year", "month"], how="inner")
f_monthly["station"] = f_monthly["stationId"].map(sid_to_name)
f_monthly["region"] = f_monthly["stationId"].map(sid_to_region)

print(f"  Monthly-mean pairs: {len(f_monthly)} (stations × year × months)")

# Step 2 — Per-station correlations (up to ~24 year-month points)
f_station_rows = []
for sid in station_ids:
    sub = f_monthly[f_monthly["stationId"] == sid]
    sname = sid_to_name[sid]
    region = sid_to_region[sid]
    r_raw, p_raw, n_raw = pearson_r_p(
        sub["AOT_monthly_mean"].values, sub["PM25_monthly_mean"].values)
    r_phys, p_phys, n_phys = pearson_r_p(
        sub["AOD_physics_monthly_mean"].values, sub["PM25_monthly_mean"].values)
    f_station_rows.append(dict(
        station=sname, region=region, stationId=sid, n_months=n_raw,
        r_AOT_monthly=r_raw, p_AOT_monthly=p_raw,
        r_phys_monthly=r_phys, p_phys_monthly=p_phys,
        mean_AOT=sub["AOT_monthly_mean"].mean(),
        mean_PM25=sub["PM25_monthly_mean"].mean(),
    ))

f_station_df = pd.DataFrame(f_station_rows)

# Overall pooled correlation (all station × year × month pairs)
r_pool_raw, p_pool_raw, n_pool_raw = pearson_r_p(
    f_monthly["AOT_monthly_mean"].values, f_monthly["PM25_monthly_mean"].values)
r_pool_phys, p_pool_phys, n_pool_phys = pearson_r_p(
    f_monthly["AOD_physics_monthly_mean"].values, f_monthly["PM25_monthly_mean"].values)

# Per-region pooled
f_region_rows = []
for rgn in ["North", "Central", "South", "ALL"]:
    sub = f_monthly if rgn == "ALL" else f_monthly[f_monthly["region"] == rgn]
    r_raw, p_raw, n_raw = pearson_r_p(
        sub["AOT_monthly_mean"].values, sub["PM25_monthly_mean"].values)
    r_phys, p_phys, n_phys = pearson_r_p(
        sub["AOD_physics_monthly_mean"].values, sub["PM25_monthly_mean"].values)
    f_region_rows.append(dict(
        region=rgn, n_pairs=n_raw,
        r_AOT_monthly=r_raw, p_AOT_monthly=p_raw,
        r_phys_monthly=r_phys, p_phys_monthly=p_phys,
    ))

f_region_df = pd.DataFrame(f_region_rows)

# Print per-station table
print(f"\n  Per-station monthly-mean correlation (AOT_monthly vs PM2.5_monthly):")
print(f"  {'Station':<20s} {'Rgn':<5s} {'r_AOT':>7s} {'p':>8s} "
      f"{'r_phys':>7s} {'p':>8s} {'n':>4s}")
print("  " + "-" * 65)
for _, r in f_station_df.iterrows():
    print(f"  {r['station']:<20s} {r['region']:<5s} "
          f"{r['r_AOT_monthly']:>+7.3f} {r['p_AOT_monthly']:>8.1e} "
          f"{r['r_phys_monthly']:>+7.3f} {r['p_phys_monthly']:>8.1e} "
          f"{int(r['n_months']):>4d}")

mean_r_raw = f_station_df["r_AOT_monthly"].mean()
mean_r_phys = f_station_df["r_phys_monthly"].mean()
print(f"\n  Mean per-station r:  AOT={mean_r_raw:+.3f}  physics={mean_r_phys:+.3f}")
print(f"  Pooled r (all {n_pool_raw} pairs): AOT={r_pool_raw:+.3f} (p={p_pool_raw:.1e})  "
      f"physics={r_pool_phys:+.3f} (p={p_pool_phys:.1e})")

# Step 3 — Comparison table
l0_hourly = a_df[a_df["filter_level"] == "L0"]["pearson_r"].mean()
l3_hourly = a_df[a_df["filter_level"] == "L3"]["pearson_r"].mean()
l0_phys = d_df[d_df["filter_level"] == "L0"]["r_AOD_physics"].mean()
print(f"\n  Approach comparison (mean r across 15 stations):")
print(f"  {'Approach':<45s} {'r':>7s}  {'What it shows'}")
print(f"  {'-'*45} {'-'*7}  {'-'*25}")
print(f"  {'Hourly raw AOT vs PM2.5':<45s} {l0_hourly:>+7.3f}  Instantaneous, noisy")
print(f"  {'Hourly physics-corrected':<45s} {l0_phys:>+7.3f}  Better but still noisy")
print(f"  {'Hourly L3 filtered':<45s} {l3_hourly:>+7.3f}  Quality filtered")
print(f"  {'Monthly mean AOT vs monthly mean PM2.5':<45s} {mean_r_raw:>+7.3f}  "
      f"Seasonal pattern")
print(f"  {'Monthly mean physics-corrected vs PM2.5':<45s} {mean_r_phys:>+7.3f}  "
      f"Seasonal + physics")

# Per-region
print(f"\n  Per-region pooled monthly correlation:")
for _, r in f_region_df.iterrows():
    print(f"    {r['region']:<10s}  r_AOT={r['r_AOT_monthly']:+.3f} (p={r['p_AOT_monthly']:.1e})  "
          f"r_phys={r['r_phys_monthly']:+.3f} (p={r['p_phys_monthly']:.1e})  "
          f"n={int(r['n_pairs'])}")

# Build combined CSV
f_csv_rows = []
for _, r in f_station_df.iterrows():
    f_csv_rows.append(dict(level="station", **r.to_dict()))
for _, r in f_region_df.iterrows():
    f_csv_rows.append(dict(level="region", **r.to_dict()))
f_csv_rows.append(dict(
    level="comparison", approach="hourly_raw",
    r=l0_hourly, description="Hourly raw AOT vs PM2.5"))
f_csv_rows.append(dict(
    level="comparison", approach="hourly_physics",
    r=l0_phys, description="Hourly physics-corrected"))
f_csv_rows.append(dict(
    level="comparison", approach="hourly_L3",
    r=l3_hourly, description="Hourly L3 filtered"))
f_csv_rows.append(dict(
    level="comparison", approach="monthly_raw",
    r=mean_r_raw, description="Monthly mean AOT"))
f_csv_rows.append(dict(
    level="comparison", approach="monthly_physics",
    r=mean_r_phys, description="Monthly mean physics-corrected"))
f_csv_df = pd.DataFrame(f_csv_rows)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION G — Normalized hourly AOD (within-month signal)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION G: Normalized hourly AOD vs PM2.5")
print("=" * 72)

# Step 1 — Monthly baselines for normalization (station × year × month)
aot_baseline = (df[df["AOT"].notna()]
                .groupby(["stationId", "year", "month"])
                .agg(AOT_month_base=("AOT", "mean"))
                .reset_index())
pm25_baseline = (full.groupby(["stationId", "year", "month"])
                 .agg(PM25_month_base=("PM25", "mean"))
                 .reset_index())

df = df.merge(aot_baseline, on=["stationId", "year", "month"], how="left")
df = df.merge(pm25_baseline, on=["stationId", "year", "month"], how="left")

# Step 2 — Compute normalized and anomaly columns
df["AOT_norm"] = df["AOT"] / df["AOT_month_base"]
df["PM25_norm"] = df["PM25"] / df["PM25_month_base"]
df["AOT_anom"] = df["AOT"] - df["AOT_month_base"]
df["PM25_anom"] = df["PM25"] - df["PM25_month_base"]

# Only rows with valid AOT for correlations
g_df = df[df["AOT"].notna() & df["AOT_month_base"].notna()
          & (df["AOT_month_base"] > 0)
          & df["PM25_month_base"].notna()
          & (df["PM25_month_base"] > 0)].copy()

print(f"  Rows with valid normalization: {len(g_df):,}")

# Step 3 — Correlate 4 combinations per station and per region
COMBOS = [
    ("raw",         "AOT",      "PM25",      "Raw AOT vs raw PM2.5"),
    ("norm_vs_norm", "AOT_norm", "PM25_norm", "AOT_norm vs PM25_norm"),
    ("anom_vs_anom", "AOT_anom", "PM25_anom", "AOT_anom vs PM25_anom"),
    ("norm_vs_raw",  "AOT_norm", "PM25",      "AOT_norm vs raw PM2.5"),
]

g_station_rows = []
for sid in station_ids:
    sub = g_df[g_df["stationId"] == sid]
    sname = sid_to_name[sid]
    region = sid_to_region[sid]
    for combo_key, x_col, y_col, desc in COMBOS:
        r, p, n = pearson_r_p(sub[x_col].values, sub[y_col].values)
        g_station_rows.append(dict(
            station=sname, region=region, stationId=sid,
            combo=combo_key, description=desc,
            pearson_r=r, p_value=p, n=n,
        ))

g_station_df = pd.DataFrame(g_station_rows)

# Print compact table
print(f"\n  Per-station correlation for 4 normalization approaches:")
print(f"  {'Station':<20s} {'Rgn':<5s} {'raw':>7s} {'norm':>7s} "
      f"{'anom':>7s} {'norm→raw':>8s} {'n':>6s}")
print("  " + "-" * 60)
for sid in station_ids:
    sname = sid_to_name[sid]
    region = sid_to_region[sid]
    sdf = g_station_df[g_station_df["stationId"] == sid].set_index("combo")
    print(f"  {sname:<20s} {region:<5s} "
          f"{sdf.loc['raw', 'pearson_r']:>+7.3f} "
          f"{sdf.loc['norm_vs_norm', 'pearson_r']:>+7.3f} "
          f"{sdf.loc['anom_vs_anom', 'pearson_r']:>+7.3f} "
          f"{sdf.loc['norm_vs_raw', 'pearson_r']:>+8.3f} "
          f"{int(sdf.loc['raw', 'n']):>6d}")

# Per-region + overall
g_region_rows = []
for rgn in ["North", "Central", "South", "ALL"]:
    sub = g_df if rgn == "ALL" else g_df[g_df["region"] == rgn]
    for combo_key, x_col, y_col, desc in COMBOS:
        r, p, n = pearson_r_p(sub[x_col].values, sub[y_col].values)
        g_region_rows.append(dict(
            region=rgn, combo=combo_key, description=desc,
            pearson_r=r, p_value=p, n=n,
        ))

g_region_df = pd.DataFrame(g_region_rows)

# Summary per combo
print(f"\n  Mean r across 15 stations:")
for combo_key, _, _, desc in COMBOS:
    sub = g_station_df[g_station_df["combo"] == combo_key]
    print(f"    {desc:<35s}  mean_r={sub['pearson_r'].mean():+.3f}")

print(f"\n  Pooled correlation (all stations combined):")
for combo_key, _, _, desc in COMBOS:
    sub = g_region_df[(g_region_df["region"] == "ALL")
                      & (g_region_df["combo"] == combo_key)]
    if len(sub):
        r = sub.iloc[0]
        print(f"    {desc:<35s}  r={r['pearson_r']:+.3f}  "
              f"p={r['p_value']:.1e}  n={int(r['n']):,}")

print(f"\n  Per-region pooled (anomaly approach):")
for rgn in ["North", "Central", "South"]:
    sub = g_region_df[(g_region_df["region"] == rgn)
                      & (g_region_df["combo"] == "anom_vs_anom")]
    if len(sub):
        r = sub.iloc[0]
        print(f"    {rgn:<10s}  r={r['pearson_r']:+.3f}  n={int(r['n']):,}")

# Build CSV
g_csv_rows = []
for _, r in g_station_df.iterrows():
    g_csv_rows.append(dict(level="station", **r.to_dict()))
for _, r in g_region_df.iterrows():
    g_csv_rows.append(dict(level="region", **r.to_dict()))
g_csv_df = pd.DataFrame(g_csv_rows)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION H — Between-station (spatial) correlation per year×month
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SECTION H: Between-station (spatial) correlation per year×month")
print("=" * 72)

# Use f_monthly directly (station × year × month, already computed in Sec F)
# Rename columns for clarity within this section
h_data = f_monthly.copy()
h_data.rename(columns={
    "AOT_monthly_mean": "AOT_clim",
    "AOD_physics_monthly_mean": "AOD_phys_clim",
    "PM25_monthly_mean": "PM25_clim",
}, inplace=True)

MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

# Per year×month spatial correlation (r across ~15 stations)
h_ym_rows = []
for (yr, m), sub in h_data.groupby(["year", "month"]):
    r_raw, p_raw, n_raw = pearson_r_p(
        sub["AOT_clim"].values, sub["PM25_clim"].values)
    r_phys, p_phys, n_phys = pearson_r_p(
        sub["AOD_phys_clim"].values, sub["PM25_clim"].values)
    h_ym_rows.append(dict(
        year=int(yr), month=int(m), month_name=MONTH_NAMES[int(m)],
        n_stations=n_raw,
        r_AOT_spatial=r_raw, p_AOT_spatial=p_raw,
        r_phys_spatial=r_phys, p_phys_spatial=p_phys,
    ))

h_ym_df = pd.DataFrame(h_ym_rows).sort_values(["year", "month"]).reset_index(drop=True)

# Also compute per-month averages across years (for summary)
h_month_df = (h_ym_df.groupby("month")
              .agg(r_AOT_spatial=("r_AOT_spatial", "mean"),
                   r_phys_spatial=("r_phys_spatial", "mean"),
                   n_year_months=("year", "size"),
                   mean_n_stations=("n_stations", "mean"))
              .reset_index())
h_month_df["month_name"] = h_month_df["month"].map(MONTH_NAMES)

# Deseasonalized pooled: subtract year×month grand mean, keep only spatial variation
h_grand = h_data.groupby(["year", "month"]).agg(
    AOT_grand=("AOT_clim", "mean"),
    AOD_phys_grand=("AOD_phys_clim", "mean"),
    PM25_grand=("PM25_clim", "mean"),
).reset_index()

h_deseas = h_data.merge(h_grand, on=["year", "month"], how="left")
h_deseas["AOT_deseas"] = h_deseas["AOT_clim"] - h_deseas["AOT_grand"]
h_deseas["AOD_phys_deseas"] = h_deseas["AOD_phys_clim"] - h_deseas["AOD_phys_grand"]
h_deseas["PM25_deseas"] = h_deseas["PM25_clim"] - h_deseas["PM25_grand"]

r_deseas_raw, p_deseas_raw, n_deseas = pearson_r_p(
    h_deseas["AOT_deseas"].values, h_deseas["PM25_deseas"].values)
r_deseas_phys, p_deseas_phys, _ = pearson_r_p(
    h_deseas["AOD_phys_deseas"].values, h_deseas["PM25_deseas"].values)

# Print table — per year×month
print(f"\n  Spatial correlation: r across stations per year×month")
print(f"  {'Year':>5s} {'Month':<6s} {'r_AOT':>7s} {'p':>8s} "
      f"{'r_phys':>7s} {'p':>8s} {'n':>4s}")
print("  " + "-" * 52)
for _, r in h_ym_df.iterrows():
    print(f"  {int(r['year']):>5d} {r['month_name']:<6s} "
          f"{r['r_AOT_spatial']:>+7.3f} {r['p_AOT_spatial']:>8.1e} "
          f"{r['r_phys_spatial']:>+7.3f} {r['p_phys_spatial']:>8.1e} "
          f"{int(r['n_stations']):>4d}")

# Summary per month (averaged across years)
print(f"\n  Mean spatial r per month (averaged across years):")
print(f"  {'Month':<6s} {'r_AOT':>7s} {'r_phys':>7s} {'n_ym':>5s}")
print("  " + "-" * 30)
for _, r in h_month_df.iterrows():
    print(f"  {r['month_name']:<6s} {r['r_AOT_spatial']:>+7.3f} "
          f"{r['r_phys_spatial']:>+7.3f} {int(r['n_year_months']):>5d}")

mean_h_raw = h_month_df["r_AOT_spatial"].mean()
mean_h_phys = h_month_df["r_phys_spatial"].mean()
print(f"\n  Mean spatial r across 12 months:  AOT={mean_h_raw:+.3f}  "
      f"physics={mean_h_phys:+.3f}")
print(f"  Deseasonalized pooled ({n_deseas} pairs):  "
      f"AOT={r_deseas_raw:+.3f} (p={p_deseas_raw:.1e})  "
      f"physics={r_deseas_phys:+.3f} (p={p_deseas_phys:.1e})")

# Build CSV
h_csv_rows = []
for _, r in h_ym_df.iterrows():
    h_csv_rows.append(dict(level="per_year_month", **r.to_dict()))
for _, r in h_month_df.iterrows():
    h_csv_rows.append(dict(level="per_month_avg", **r.to_dict()))
h_csv_rows.append(dict(
    level="deseasonalized_pooled", n_pairs=n_deseas,
    r_AOT_spatial=r_deseas_raw, p_AOT_spatial=p_deseas_raw,
    r_phys_spatial=r_deseas_phys, p_phys_spatial=p_deseas_phys,
))
h_csv_rows.append(dict(
    level="mean_12_months",
    r_AOT_spatial=mean_h_raw, r_phys_spatial=mean_h_phys,
))
h_csv_df = pd.DataFrame(h_csv_rows)


# ══════════════════════════════════════════════════════════════════════════
#  SAVE CSVs
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("Saving CSVs")
print("=" * 72)

a_df.to_csv(RPT_DIR / "aod_pm25_per_station.csv", index=False)
print(f"  Saved: aod_pm25_per_station.csv  ({len(a_df)} rows)")

b_df.to_csv(RPT_DIR / "aod_pm25_region_season.csv", index=False)
print(f"  Saved: aod_pm25_region_season.csv  ({len(b_df)} rows)")

c_df.to_csv(RPT_DIR / "aod_pm25_ssa_regime.csv", index=False)
print(f"  Saved: aod_pm25_ssa_regime.csv  ({len(c_df)} rows)")

d_df.to_csv(RPT_DIR / "aod_pm25_physics.csv", index=False)
print(f"  Saved: aod_pm25_physics.csv  ({len(d_df)} rows)")

e_avail = pd.concat([
    e_station_df.assign(level="station"),
    e_month_df.assign(level="station_month"),
], ignore_index=True)
e_avail.to_csv(RPT_DIR / "aod_pm25_availability.csv", index=False)
print(f"  Saved: aod_pm25_availability.csv  ({len(e_avail)} rows)")

f_csv_df.to_csv(RPT_DIR / "aod_pm25_monthly_clim.csv", index=False)
print(f"  Saved: aod_pm25_monthly_clim.csv  ({len(f_csv_df)} rows)")

g_csv_df.to_csv(RPT_DIR / "aod_pm25_normalized.csv", index=False)
print(f"  Saved: aod_pm25_normalized.csv  ({len(g_csv_df)} rows)")

h_csv_df.to_csv(RPT_DIR / "aod_pm25_spatial_corr.csv", index=False)
print(f"  Saved: aod_pm25_spatial_corr.csv  ({len(h_csv_df)} rows)")

# Master
master_rows = []
for _, r in a_df.iterrows():
    master_rows.append({"section": "A_per_station", **r.to_dict()})
for _, r in b_df.iterrows():
    master_rows.append({"section": "B_region_season", **r.to_dict()})
for _, r in c_df.iterrows():
    master_rows.append({"section": "C_ssa_regime", **r.to_dict()})
for _, r in d_df.iterrows():
    master_rows.append({"section": "D_physics", **r.to_dict()})
for _, r in e_station_df.iterrows():
    master_rows.append({"section": "E_availability", **r.to_dict()})
for _, r in f_csv_df.iterrows():
    master_rows.append({"section": "F_monthly_clim", **r.to_dict()})
for _, r in g_csv_df.iterrows():
    master_rows.append({"section": "G_normalized", **r.to_dict()})
for _, r in h_csv_df.iterrows():
    master_rows.append({"section": "H_spatial", **r.to_dict()})
master_df = pd.DataFrame(master_rows)
master_df.to_csv(RPT_DIR / "aod_pm25_correlation_paper.csv", index=False)
print(f"  Saved: aod_pm25_correlation_paper.csv  ({len(master_df)} rows)")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — 5-panel scatter: AOT vs PM2.5 by filter level (ALL regions)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("Generating figures")
print("=" * 72)

region_colors = {"North": "#2166ac", "Central": "#b2182b", "South": "#1b7837"}

fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=True)
masks = apply_filters(df)

for idx, (lvl, ax) in enumerate(zip(["L0", "L1", "L2", "L3", "L3"], axes)):
    sub = df[masks[lvl]]
    x = sub["AOT"].values
    y = sub["PM25"].values
    fmask = np.isfinite(x) & np.isfinite(y)
    x_c, y_c = x[fmask], y[fmask]

    if len(x_c) < 10:
        continue

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=np.std(y_c) * 1.5,
        random_state=42, max_trials=500,
    )
    ransac.fit(x_c.reshape(-1, 1), y_c)
    inliers = ransac.inlier_mask_

    if idx < 4:
        # L0-L3: color by region, no inlier/outlier distinction
        for rgn, clr in region_colors.items():
            rgn_mask = sub.iloc[np.where(fmask)[0]]["region"].values == rgn
            ax.scatter(x_c[rgn_mask], y_c[rgn_mask], s=6, alpha=0.25,
                       c=clr, edgecolors="none", zorder=2)

        x_line = np.linspace(x_c.min(), x_c.max(), 100)
        y_line = ransac.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, "k-", lw=2, zorder=3)

        ols_slope, ols_int, _, _, _ = stats.linregress(x_c, y_c)
        y_ols = ols_slope * x_line + ols_int
        ax.plot(x_line, y_ols, "--", color="#999999", lw=1.2, zorder=3)

        r, _, n = pearson_r_p(x_c, y_c)
        r2_ransac_val = ransac.score(x_c[inliers].reshape(-1, 1), y_c[inliers])
        title_lvl = lvl
    else:
        # L4: show inliers vs outliers on L3 data
        ax.scatter(x_c[~inliers], y_c[~inliers], s=6, alpha=0.2,
                   c="#d9d9d9", edgecolors="none", zorder=1)
        for rgn, clr in region_colors.items():
            rgn_mask = sub.iloc[np.where(fmask)[0]]["region"].values == rgn
            both = rgn_mask & inliers
            ax.scatter(x_c[both], y_c[both], s=8, alpha=0.35,
                       c=clr, edgecolors="none", zorder=2)

        x_line = np.linspace(x_c[inliers].min(), x_c[inliers].max(), 100)
        y_line = ransac.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, "k-", lw=2, zorder=3)

        x_in, y_in = x_c[inliers], y_c[inliers]
        r, _, n = pearson_r_p(x_in, y_in)
        r2_ransac_val = ransac.score(x_in.reshape(-1, 1), y_in)
        title_lvl = "L4"

    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    ax.text(0.03, 0.97,
            f"r = {r:+.3f}\n"
            f"R² = {r2_ransac_val:.3f}\n"
            f"n = {n:,}\n"
            f"y = {slope:.1f}x {'+' if intercept >= 0 else ''}{intercept:.1f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_title(f"{title_lvl}: {FILTER_LABELS[title_lvl]}", fontsize=10, fontweight="bold")
    ax.set_xlabel("AOT (550 nm)", fontsize=10)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 200)
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel("PM$_{2.5}$ (μg/m³)", fontsize=11)

# Legend
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=c, label=r) for r, c in region_colors.items()]
legend_elems.append(plt.Line2D([0], [0], color="k", lw=2, label="RANSAC"))
legend_elems.append(plt.Line2D([0], [0], color="#999", lw=1.2, ls="--", label="OLS"))
axes[4].legend(handles=legend_elems, loc="lower right", fontsize=7, framealpha=0.9)

plt.suptitle("AOT vs PM$_{2.5}$ — Progressive Filter Levels (same-hour, 2024–2025)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_progressive_filters.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig1_progressive_filters.png")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Heatmap: station × season correlation at L3
# ══════════════════════════════════════════════════════════════════════════

heat_data = []
for sid in station_ids:
    sname = sid_to_name[sid]
    region = sid_to_region[sid]
    row = {"station": sname, "region": region}
    sub_full = df[df["stationId"] == sid]
    m3_full = apply_filters(sub_full)["L3"]
    for season in season_order:
        sub_s = sub_full[sub_full["season"] == season]
        m3_s = apply_filters(sub_s)["L3"]
        s = sub_s[m3_s]
        r, _, n = pearson_r_p(s["AOT"].values, s["PM25"].values)
        row[season] = r
        row[f"n_{season}"] = n
    s_all = sub_full[m3_full]
    r, _, n = pearson_r_p(s_all["AOT"].values, s_all["PM25"].values)
    row["Annual"] = r
    row["n_Annual"] = n
    heat_data.append(row)

heat_df = pd.DataFrame(heat_data)
heat_df["_region_ord"] = heat_df["region"].map({"North": 0, "Central": 1, "South": 2})
heat_df = heat_df.sort_values(["_region_ord", "Annual"], ascending=[True, False])

fig, ax = plt.subplots(figsize=(8, 9))
cols = ["DJF", "MAM", "JJA", "SON", "Annual"]
mat = heat_df[cols].values

im = ax.imshow(mat, aspect="auto", cmap=plt.cm.RdBu_r, vmin=-0.5, vmax=0.5)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(heat_df)))
ylabels = [f"{r['station']} ({r['region'][0]})" for _, r in heat_df.iterrows()]
ax.set_yticklabels(ylabels, fontsize=9)
for i, (_, r) in enumerate(heat_df.iterrows()):
    ax.get_yticklabels()[i].set_color(region_colors.get(r["region"], "black"))

for i in range(len(heat_df)):
    for j in range(len(cols)):
        val = mat[i, j]
        n = heat_df.iloc[i][f"n_{cols[j]}"]
        if np.isnan(val):
            txt = "—"
        else:
            txt = f"{val:+.2f}\n({int(n)})"
        text_color = "white" if abs(val) > 0.35 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=text_color)

prev_region = None
for i, (_, r) in enumerate(heat_df.iterrows()):
    if prev_region and r["region"] != prev_region:
        ax.axhline(i - 0.5, color="black", lw=1.5)
    prev_region = r["region"]

cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Pearson r (AOT vs PM$_{2.5}$)", fontsize=10)
ax.set_title("AOT–PM$_{2.5}$ Correlation by Station × Season\n"
             "(L3: Unc≤0.5 & RF≥0.5)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_heatmap_station_season.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig2_heatmap_station_season.png")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Monthly AOD availability
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1.3]})

ax = axes[0]
bars = ax.bar(month_summary["month"], month_summary["pct"],
              color="#3498db", edgecolor="white", lw=0.5)
for bar, pct in zip(bars, month_summary["pct"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Valid AOD (%)", fontsize=11)
ax.set_title("(a) Overall monthly AOD availability", fontsize=11, fontweight="bold")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_ylim(0, max(month_summary["pct"]) * 1.15)
ax.grid(True, axis="y", alpha=0.2)

ax = axes[1]
avail_pivot = e_month_df.pivot_table(index="station", columns="month",
                                      values="pct_valid", aggfunc="first")
station_order = e_station_df.sort_values(["region", "pct_valid"],
                                          ascending=[True, False])["station"]
avail_pivot = avail_pivot.loc[station_order]

im = ax.imshow(avail_pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=30)
ax.set_xticks(range(12))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_yticks(range(len(avail_pivot)))
ax.set_yticklabels(avail_pivot.index, fontsize=8)
for i in range(len(avail_pivot)):
    for j in range(12):
        val = avail_pivot.values[i, j]
        txt = f"{val:.0f}" if val > 0 else ""
        ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                color="white" if val > 20 else "black")
cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("% hours with valid AOD", fontsize=9)
ax.set_title("(b) AOD availability by station × month", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_aod_availability.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig3_aod_availability.png")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — 5-panel: Physics-corrected AOD by filter level
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=True)
masks_phys = apply_filters(phys_valid)

for idx, (lvl, ax) in enumerate(zip(["L0", "L1", "L2", "L3", "L3"], axes)):
    sub = phys_valid[masks_phys[lvl]]
    x = sub["AOD_physics"].values
    y = sub["PM25"].values
    fmask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x_c, y_c = x[fmask], y[fmask]

    # Clip extreme for plotting
    x_clip = np.percentile(x_c, 99.5) if len(x_c) > 10 else 1.0
    pm = x_c <= x_clip
    x_p, y_p = x_c[pm], y_c[pm]

    if len(x_p) < 10:
        continue

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=np.std(y_p) * 1.5,
        random_state=42, max_trials=500,
    )
    ransac.fit(x_p.reshape(-1, 1), y_p)
    inliers = ransac.inlier_mask_

    if idx < 4:
        for rgn, clr in region_colors.items():
            rgn_mask = sub.iloc[np.where(fmask)[0][pm]]["region"].values == rgn
            ax.scatter(x_p[rgn_mask], y_p[rgn_mask], s=6, alpha=0.25,
                       c=clr, edgecolors="none", zorder=2)
        x_line = np.linspace(x_p.min(), x_p.max(), 100)
        y_line = ransac.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, "k-", lw=2, zorder=3)
        r, _, n = pearson_r_p(x_p, y_p)
        r2_val = ransac.score(x_p[inliers].reshape(-1, 1), y_p[inliers])
        title_lvl = lvl
    else:
        ax.scatter(x_p[~inliers], y_p[~inliers], s=6, alpha=0.2,
                   c="#d9d9d9", edgecolors="none", zorder=1)
        for rgn, clr in region_colors.items():
            rgn_mask = sub.iloc[np.where(fmask)[0][pm]]["region"].values == rgn
            both = rgn_mask & inliers
            ax.scatter(x_p[both], y_p[both], s=8, alpha=0.35,
                       c=clr, edgecolors="none", zorder=2)
        x_line = np.linspace(x_p[inliers].min(), x_p[inliers].max(), 100)
        y_line = ransac.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, "k-", lw=2, zorder=3)
        x_in, y_in = x_p[inliers], y_p[inliers]
        r, _, n = pearson_r_p(x_in, y_in)
        r2_val = ransac.score(x_in.reshape(-1, 1), y_in)
        title_lvl = "L4"

    # Also compute raw AOT r for comparison
    if idx < 4:
        r_raw, _, _ = pearson_r_p(
            sub.iloc[np.where(fmask)[0][pm]]["AOT"].values, y_p)
    else:
        raw_sub = sub.iloc[np.where(fmask)[0][pm]]
        r_raw, _, _ = pearson_r_p(raw_sub["AOT"].values[inliers], y_in)

    ax.text(0.03, 0.97,
            f"r(phys) = {r:+.3f}\n"
            f"r(AOT)  = {r_raw:+.3f}\n"
            f"Δr = {r - r_raw:+.3f}\n"
            f"n = {n:,}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_title(f"{title_lvl}: {FILTER_LABELS[title_lvl]}", fontsize=10, fontweight="bold")
    ax.set_xlabel("AOD$_{physics}$", fontsize=10)
    ax.set_ylim(0, 200)
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel("PM$_{2.5}$ (μg/m³)", fontsize=11)

plt.suptitle("Physics-Corrected AOD vs PM$_{2.5}$ — Progressive Filters\n"
             "AOD$_{physics}$ = AOT × ((1−RH/100)$^{0.6}$) / PBLH",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_physics_progressive.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig4_physics_progressive.png")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — 4-panel by region at L3 (AOT vs PM2.5)
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
panels = [("North", axes[0, 0]), ("Central", axes[0, 1]),
          ("South", axes[1, 0]), ("All", axes[1, 1])]

m3 = apply_filters(df)["L3"]

for panel_label, ax in panels:
    if panel_label == "All":
        sub = df[m3]
        color = "#636363"
    else:
        sub = df[m3 & (df["region"] == panel_label)]
        color = region_colors[panel_label]

    x = sub["AOT"].values
    y = sub["PM25"].values
    fmask = np.isfinite(x) & np.isfinite(y)
    x_c, y_c = x[fmask], y[fmask]

    if len(x_c) < 10:
        continue

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=np.std(y_c) * 1.5,
        random_state=42, max_trials=500,
    )
    ransac.fit(x_c.reshape(-1, 1), y_c)
    inliers = ransac.inlier_mask_

    ax.scatter(x_c[~inliers], y_c[~inliers], s=8, alpha=0.15, c="#d9d9d9",
               edgecolors="none", zorder=1)
    ax.scatter(x_c[inliers], y_c[inliers], s=8, alpha=0.3, c=color,
               edgecolors="none", zorder=2)

    x_line = np.linspace(x_c.min(), x_c.max(), 100)
    y_line = ransac.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, "k-", lw=2, zorder=3)

    ols_slope, ols_int, _, _, _ = stats.linregress(x_c, y_c)
    ax.plot(x_line, ols_slope * x_line + ols_int, "--", color="#999", lw=1.2, zorder=3)

    r, _, n = pearson_r_p(x_c, y_c)
    r_in, _, n_in = pearson_r_p(x_c[inliers], y_c[inliers])
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    ax.text(0.03, 0.97,
            f"r = {r:+.3f}  (all L3)\n"
            f"r = {r_in:+.3f}  (RANSAC inliers)\n"
            f"n = {n:,} → {n_in:,}\n"
            f"y = {slope:.1f}x {'+' if intercept >= 0 else ''}{intercept:.1f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_xlabel("AOT (550 nm)", fontsize=11)
    ax.set_ylabel("PM$_{2.5}$ (μg/m³)", fontsize=11)
    ax.set_title(f"{panel_label}  (L3: Unc≤0.5 & RF≥0.5, n={n:,})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 200)
    ax.grid(True, alpha=0.2)

plt.suptitle("AOT vs PM$_{2.5}$ by Region — L3 Filter + RANSAC",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_regions_L3_ransac.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig5_regions_L3_ransac.png")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Monthly-mean AOT vs PM2.5 (2 panels: raw + physics-corrected)
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (x_col, x_label, title_suffix) in zip(axes, [
    ("AOT_monthly_mean", "Monthly mean AOT (550 nm)", "Raw AOT"),
    ("AOD_physics_monthly_mean",
     "Monthly mean AOD$_{phys}$  [AOT·(1−RH/100)$^{0.6}$/PBLH]",
     "Physics-corrected"),
]):
    for rgn, color in region_colors.items():
        sub = f_monthly[f_monthly["region"] == rgn]
        ax.scatter(sub[x_col], sub["PM25_monthly_mean"],
                   c=color, label=rgn, alpha=0.75, s=50, edgecolors="white",
                   linewidths=0.5, zorder=3)

    # OLS fit across all points
    x_all = f_monthly[x_col].values
    y_all = f_monthly["PM25_monthly_mean"].values
    fmask = np.isfinite(x_all) & np.isfinite(y_all)
    x_c, y_c = x_all[fmask], y_all[fmask]
    slope, intercept, r_val, p_val, _ = stats.linregress(x_c, y_c)
    x_line = np.linspace(x_c.min(), x_c.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k-", lw=2, zorder=4)

    ax.text(0.03, 0.97,
            f"r = {r_val:+.3f}  (n={len(x_c)})\n"
            f"y = {slope:.1f}x {'+' if intercept >= 0 else ''}{intercept:.1f}\n"
            f"p = {p_val:.1e}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Monthly mean PM$_{2.5}$ (μg/m³)", fontsize=11)
    ax.set_title(title_suffix, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.2)

plt.suptitle("Monthly-mean AOD vs PM$_{2.5}$ Climatology by Station×Month\n"
             "(AOD from same-hour Himawari-8, PM$_{2.5}$ from all hours, 2024–2025)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_monthly_climatology.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig6_monthly_climatology.png")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — Normalized hourly AOD scatter (2 panels: anomaly + ratio)
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: AOT_anomaly vs PM25_anomaly
ax = axes[0]
for rgn, color in region_colors.items():
    sub = g_df[g_df["region"] == rgn]
    ax.scatter(sub["AOT_anom"], sub["PM25_anom"],
               c=color, label=rgn, alpha=0.2, s=8, edgecolors="none", zorder=2)

x_all = g_df["AOT_anom"].values
y_all = g_df["PM25_anom"].values
fmask = np.isfinite(x_all) & np.isfinite(y_all)
x_c, y_c = x_all[fmask], y_all[fmask]
slope, intercept, r_val, p_val, _ = stats.linregress(x_c, y_c)
x_line = np.linspace(np.percentile(x_c, 1), np.percentile(x_c, 99), 100)
ax.plot(x_line, slope * x_line + intercept, "k-", lw=2, zorder=4)
ax.axhline(0, color="#999", lw=0.8, ls="--", zorder=1)
ax.axvline(0, color="#999", lw=0.8, ls="--", zorder=1)

ax.text(0.03, 0.97,
        f"r = {r_val:+.3f}  (n={len(x_c):,})\n"
        f"y = {slope:.1f}x {'+' if intercept >= 0 else ''}{intercept:.1f}\n"
        f"p = {p_val:.1e}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

ax.set_xlabel("AOT anomaly (hourly − monthly mean)", fontsize=11)
ax.set_ylabel("PM$_{2.5}$ anomaly (hourly − monthly mean, μg/m³)", fontsize=11)
ax.set_title("Anomaly (absolute)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.2)

# Panel 2: AOT_norm vs PM25_norm
ax = axes[1]
# Clip extreme ratios for visualization
g_plot = g_df[(g_df["AOT_norm"] < 10) & (g_df["PM25_norm"] < 5)].copy()
for rgn, color in region_colors.items():
    sub = g_plot[g_plot["region"] == rgn]
    ax.scatter(sub["AOT_norm"], sub["PM25_norm"],
               c=color, label=rgn, alpha=0.2, s=8, edgecolors="none", zorder=2)

x_all = g_plot["AOT_norm"].values
y_all = g_plot["PM25_norm"].values
fmask = np.isfinite(x_all) & np.isfinite(y_all)
x_c, y_c = x_all[fmask], y_all[fmask]
slope, intercept, r_val, p_val, _ = stats.linregress(x_c, y_c)
x_line = np.linspace(np.percentile(x_c, 1), np.percentile(x_c, 99), 100)
ax.plot(x_line, slope * x_line + intercept, "k-", lw=2, zorder=4)
ax.axhline(1, color="#999", lw=0.8, ls="--", zorder=1)
ax.axvline(1, color="#999", lw=0.8, ls="--", zorder=1)

ax.text(0.03, 0.97,
        f"r = {r_val:+.3f}  (n={len(x_c):,})\n"
        f"y = {slope:.2f}x {'+' if intercept >= 0 else ''}{intercept:.2f}\n"
        f"p = {p_val:.1e}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

ax.set_xlabel("AOT / AOT$_{monthly}$ (ratio)", fontsize=11)
ax.set_ylabel("PM$_{2.5}$ / PM$_{2.5,monthly}$ (ratio)", fontsize=11)
ax.set_title("Normalized (ratio)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.2)

plt.suptitle("Within-month AOD vs PM$_{2.5}$ signal\n"
             "(hourly deviations from station×year×month baseline, same-hour only)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig7_normalized_aod.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig7_normalized_aod.png")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — Spatial correlation: 4 months (Jan, Apr, Jul, Oct), years separate
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
selected_months = [(1, "January"), (4, "April"), (7, "July"), (10, "October")]
year_markers = {2024: "o", 2025: "s"}  # circle vs square

for (m, mname), ax in zip(selected_months, axes.flat):
    sub = h_data[h_data["month"] == m]

    # Plot by region × year
    for rgn, color in region_colors.items():
        for yr, marker in year_markers.items():
            rsub = sub[(sub["region"] == rgn) & (sub["year"] == yr)]
            if len(rsub) == 0:
                continue
            ax.scatter(rsub["AOT_clim"], rsub["PM25_clim"],
                       c=color, marker=marker, s=70, edgecolors="white",
                       linewidths=0.7, zorder=3, alpha=0.85,
                       label=f"{rgn} {yr}")

    # Label each point
    for _, row in sub.iterrows():
        label = row["station"].split()[0][:5]
        ax.annotate(label, (row["AOT_clim"], row["PM25_clim"]),
                    fontsize=5.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points", alpha=0.6)

    # Regression line across all points in this month
    x_vals = sub["AOT_clim"].values
    y_vals = sub["PM25_clim"].values
    fmask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_c, y_c = x_vals[fmask], y_vals[fmask]

    if len(x_c) >= 5:
        slope, intercept, r_val, p_val, _ = stats.linregress(x_c, y_c)
        x_line = np.linspace(x_c.min(), x_c.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, "k-", lw=2, zorder=4)

        ax.text(0.03, 0.97,
                f"r = {r_val:+.3f}  (n={len(x_c)})\n"
                f"p = {p_val:.2e}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_xlabel("Monthly mean AOT (550 nm)", fontsize=10)
    ax.set_ylabel("Monthly mean PM$_{2.5}$ (μg/m³)", fontsize=10)
    ax.set_title(mname, fontsize=13, fontweight="bold")
    # Deduplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="lower right",
              ncol=2)
    ax.grid(True, alpha=0.2)

plt.suptitle("Spatial correlation: station AOT vs PM$_{2.5}$ per month\n"
             "(each point = one station×year, ○ = 2024, □ = 2025)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig8_spatial_by_month.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig8_spatial_by_month.png")


# ══════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

print(f"\n  Progressive filter impact (mean r across 15 stations, AOT vs PM2.5):")
for lvl in ["L0", "L1", "L2", "L3", "L4"]:
    sub = a_df[a_df["filter_level"] == lvl]
    print(f"    {lvl} ({FILTER_LABELS[lvl]:25s}):  mean_r={sub['pearson_r'].mean():+.3f}  "
          f"mean_r²={sub['r_squared'].mean():.4f}")

print(f"\n  Physics correction + filtering (mean r, AOD_physics vs PM2.5):")
for lvl in ["L0", "L1", "L2", "L3", "L4"]:
    sub = d_df[d_df["filter_level"] == lvl]
    print(f"    {lvl}:  r_AOT={sub['r_AOT'].mean():+.3f}  "
          f"r_phys={sub['r_AOD_physics'].mean():+.3f}  "
          f"Δ={sub['delta_r'].mean():+.3f}")

# Pooled ALL at each level
print(f"\n  Pooled (all stations combined):")
for lvl in ["L0", "L1", "L2", "L3", "L4"]:
    sub = b_df[(b_df["region"] == "ALL") & (b_df["season"] == "ALL")
               & (b_df["filter_level"] == lvl)]
    if len(sub):
        r = sub.iloc[0]
        print(f"    {lvl}: r={r['pearson_r']:+.3f}  n={int(r['n']):>5}  "
              f"RANSAC_r²={r['ransac_r2']:.4f}")

print(f"\n  Monthly-mean climatology (Section F):")
print(f"    Mean per-station r:  AOT={mean_r_raw:+.3f}  physics={mean_r_phys:+.3f}")
print(f"    Pooled r ({n_pool_raw} pairs):   AOT={r_pool_raw:+.3f}  "
      f"physics={r_pool_phys:+.3f}")

print(f"\n  Key finding: L0 (raw) → L4 (filtered+RANSAC) shows progressive improvement")
l0_r = a_df[a_df["filter_level"] == "L0"]["pearson_r"].mean()
l4_r = a_df[a_df["filter_level"] == "L4"]["pearson_r"].mean()
print(f"    Raw mean r:      {l0_r:+.3f}")
print(f"    L4 mean r:       {l4_r:+.3f}")
print(f"    Improvement:     {l4_r - l0_r:+.3f}")

print(f"\n  Normalized hourly AOD (Section G, mean r across 15 stations):")
for combo_key, _, _, desc in COMBOS:
    sub = g_station_df[g_station_df["combo"] == combo_key]
    print(f"    {desc:<35s}  mean_r={sub['pearson_r'].mean():+.3f}")

print(f"\n  Spatial correlation per month (Section H):")
print(f"    Mean spatial r (12 months):  AOT={mean_h_raw:+.3f}  "
      f"physics={mean_h_phys:+.3f}")
print(f"    Deseasonalized pooled:       AOT={r_deseas_raw:+.3f}  "
      f"physics={r_deseas_phys:+.3f}")

print(f"\n  Monthly aggregation lifts signal above hourly noise:")
print(f"    Hourly raw:      {l0_hourly:+.3f}")
print(f"    Monthly raw:     {mean_r_raw:+.3f}  (Δ={mean_r_raw - l0_hourly:+.3f})")
print(f"    Monthly physics: {mean_r_phys:+.3f}  (Δ={mean_r_phys - l0_hourly:+.3f})")

print("\nDone.")
