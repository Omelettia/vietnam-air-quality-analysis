"""
Comprehensive data profile of the final unified thesis table.
Outputs to console + analysis/thesis_experiments/data_profile.md
"""
import os, sys, io, warnings
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
warnings.filterwarnings("ignore")

def _repo_root():
    p = os.path.abspath(os.path.dirname(__file__))
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "data", "merged")):
            return p
        p = os.path.dirname(p)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 3))

REPO = _repo_root()
os.chdir(REPO)

FEATURE_DIR = os.path.join(REPO, "Thesis", "scripts", "03_features")
if FEATURE_DIR not in sys.path:
    sys.path.insert(0, FEATURE_DIR)
from regional_feature_pipeline import (
    DAILY_SAT, MET, OBS_DERIVED, PHYSICS_FEATS, PRECIP,
    SAT_AOD, SAT_REGIME, STABILITY, TEMPORAL,
)

OUT_MD = "analysis/thesis_experiments/data_profile.md"

lines = []
def P(s=""):
    print(s)
    lines.append(s)

# =====================================================================
P("# Data Profile: unified_thesis.csv")
P()

data_path = "data/merged/unified_thesis.csv"
df = pd.read_csv(data_path, dtype={"stationId": str})
N_CSV_COLS = len(df.columns)  # before profiling adds helper columns
df["ts"] = pd.to_datetime(df["ts"])

meta = pd.read_csv("Thesis/results/01_stations/station_selection_final.csv",
                    dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))

# Thesis policy: no station-level exclusions; row-level QC only.
EXCLUDE = set()

# =====================================================================
# 1. BASIC SHAPE
# =====================================================================
P("## 1. Basic Shape")
P()
P(f"- **Total rows**: {len(df):,}")
P(f"- **Total columns**: {N_CSV_COLS}")
P(f"- **Date range**: {df['ts'].min()} to {df['ts'].max()}")
n_stn_all = df["stationId"].nunique()
P(f"- **Unique stations (all)**: {n_stn_all}")

df_clean = df[~df["stationId"].isin(EXCLUDE)].copy()
n_stn = df_clean["stationId"].nunique()
P(f"- **Unique stations in profiled analysis subset**: {n_stn}")

pm_nonnull = df_clean["PM2.5"].notna().sum()
P(f"- **PM2.5 non-null observations**: {pm_nonnull:,} ({100*pm_nonnull/len(df_clean):.1f}%)")
P()

# From here on, work with clean data
df = df_clean.reset_index(drop=True)

# =====================================================================
# 2. PM2.5 TARGET VARIABLE
# =====================================================================
P("## 2. PM2.5 Target Variable")
P()

pm = df["PM2.5"].dropna()
P("### Overall distribution")
P()
P(f"| Statistic | Value |")
P(f"|-----------|-------|")
P(f"| Count | {len(pm):,} |")
P(f"| Mean | {pm.mean():.2f} |")
P(f"| Median | {pm.median():.2f} |")
P(f"| Std | {pm.std():.2f} |")
P(f"| Min | {pm.min():.2f} |")
P(f"| Max | {pm.max():.2f} |")
for pct in [5, 25, 50, 75, 95, 99]:
    P(f"| P{pct} | {pm.quantile(pct/100):.2f} |")
P()

# Per-station stats
stn_stats = df.groupby("stationId").agg(
    mean_pm=("PM2.5", "mean"),
    std_pm=("PM2.5", "std"),
    count_pm=("PM2.5", "count"),
    count_total=("PM2.5", "size"),
).reset_index()
stn_stats["tier"] = stn_stats["mean_pm"].apply(
    lambda x: "t0" if x < 10 else ("t1" if x < 20 else ("t2" if x < 35 else "t3")))
stn_stats["name"] = stn_stats["stationId"].map(sid_name)
stn_stats = stn_stats.sort_values("mean_pm", ascending=False)

P("### Per-tier summary")
P()
tier_agg = stn_stats.groupby("tier").agg(
    n_stations=("stationId", "count"),
    mean_pm=("mean_pm", "mean"),
    total_hours=("count_pm", "sum"),
).reindex(["t0", "t1", "t2", "t3"])
P("| Tier | Threshold | N stations | Mean PM2.5 | Total hours |")
P("|------|-----------|------------|------------|-------------|")
thresholds = {"t0": "<10", "t1": "10-20", "t2": "20-35", "t3": ">=35"}
for t in ["t0", "t1", "t2", "t3"]:
    if t in tier_agg.index:
        r = tier_agg.loc[t]
        P(f"| {t} | {thresholds[t]} | {int(r['n_stations'])} | {r['mean_pm']:.1f} | {int(r['total_hours']):,} |")
P()

P("### Hours per station")
P()
P(f"- Min: {int(stn_stats['count_pm'].min()):,}")
P(f"- Max: {int(stn_stats['count_pm'].max()):,}")
P(f"- Median: {int(stn_stats['count_pm'].median()):,}")
P(f"- Mean: {int(stn_stats['count_pm'].mean()):,}")
P()

P("### Per-station table (sorted by mean PM2.5 descending)")
P()
P("| # | Station | Mean | Std | Hours | Tier |")
P("|---|---------|------|-----|-------|------|")
for i, (_, r) in enumerate(stn_stats.iterrows()):
    nm = (r["stationId"] if pd.isna(r["name"]) else r["name"])[:45]
    P(f"| {i+1} | {nm} | {r['mean_pm']:.1f} | {r['std_pm']:.1f} | {int(r['count_pm']):,} | {r['tier']} |")
P()

# =====================================================================
# 3. FEATURE COVERAGE
# =====================================================================
P("## 3. Feature Coverage (Missing Values)")
P()

cov = df.notna().mean() * 100
cov_df = pd.DataFrame({"column": cov.index, "pct_nonnull": cov.values})
cov_df = cov_df.sort_values("pct_nonnull", ascending=False)

full = cov_df[cov_df["pct_nonnull"] >= 99.9]
high = cov_df[(cov_df["pct_nonnull"] >= 50) & (cov_df["pct_nonnull"] < 99.9)]
med = cov_df[(cov_df["pct_nonnull"] >= 10) & (cov_df["pct_nonnull"] < 50)]
low = cov_df[cov_df["pct_nonnull"] < 10]

P(f"### Coverage groups")
P()
P(f"- **>=99.9% coverage**: {len(full)} columns")
P(f"- **50-99.9% coverage**: {len(high)} columns")
P(f"- **10-50% coverage**: {len(med)} columns")
P(f"- **<10% coverage**: {len(low)} columns")
P()

P("### Full coverage (>=99.9%)")
P()
P("| Column | Coverage % |")
P("|--------|-----------|")
for _, r in full.iterrows():
    P(f"| {r['column']} | {r['pct_nonnull']:.1f}% |")
P()

P("### High coverage (50-99.9%)")
P()
P("| Column | Coverage % |")
P("|--------|-----------|")
for _, r in high.iterrows():
    P(f"| {r['column']} | {r['pct_nonnull']:.1f}% |")
P()

P("### Medium coverage (10-50%)")
P()
P("| Column | Coverage % |")
P("|--------|-----------|")
for _, r in med.iterrows():
    P(f"| {r['column']} | {r['pct_nonnull']:.1f}% |")
P()

P("### Low coverage (<10%)")
P()
P("| Column | Coverage % |")
P("|--------|-----------|")
for _, r in low.iterrows():
    P(f"| {r['column']} | {r['pct_nonnull']:.1f}% |")
P()

P("### Key feature coverage")
P()
key_feats = ["PM2.5", "AOT_ffill_48h", "AOT_outer_mean", "hours_since_valid_AOT",
             "AOT__0_0", "AOT_mean", "PBLH", "Temperature_final", "Humidity_final",
             "Pressure_final", "wind_u", "wind_v", "VC", "WS_local",
             "precip_mm", "rain_sum_48h", "rain_days_7d",
             "RH_factor", "hour_sin", "month_sin"]
P("| Feature | Coverage % |")
P("|---------|-----------|")
for f in key_feats:
    if f in cov.index:
        P(f"| {f} | {cov[f]:.2f}% |")
    else:
        P(f"| {f} | NOT IN CSV |")
P()

# =====================================================================
# 4. FEATURE VALUE RANGES
# =====================================================================
P("## 4. Feature Value Ranges")
P()

# Final regional-model feature policy, imported from the shared pipeline:
# 59 observation/physics features + 8 fold-specific RFSI features = 67.
RFSI_FEATURES = [
    "PM25_nn_idw", "PM25_nn1", "PM25_nn2", "PM25_nn3",
    "PM25_upwind_idw", "PM25_downwind_idw",
    "PM25_wind_spread", "PM25_neighbor_spread",
]
MODEL_FEATURES = list(dict.fromkeys(
    SAT_AOD + DAILY_SAT + MET + PRECIP + TEMPORAL
    + STABILITY + SAT_REGIME + OBS_DERIVED + PHYSICS_FEATS
)) + RFSI_FEATURES

in_csv = [f for f in MODEL_FEATURES if f in df.columns]
not_in_csv = [f for f in MODEL_FEATURES if f not in df.columns]

P(f"Of the {len(MODEL_FEATURES)} final regional-model features:")
P(f"- **{len(in_csv)}** exist in the CSV (source columns of the enriched unified table)")
P(f"- **{len(not_in_csv)}** are computed at runtime (deterministic interactions in "
  f"regional_feature_pipeline.prepare_observation_features and fold-specific RFSI "
  f"in exp_red_river_delta.py)")
P()

if not_in_csv:
    P("### Runtime-computed features (not in CSV)")
    P()
    for f in not_in_csv:
        P(f"- `{f}`")
    P()

P("### Features present in CSV")
P()
P("| Feature | Mean | Std | Min | Max | %NonNull | Corr w/ PM2.5 |")
P("|---------|------|-----|-----|-----|----------|---------------|")

pm_vals = df["PM2.5"]
for f in in_csv:
    col = df[f]
    nn = col.notna().mean() * 100
    if col.notna().sum() > 10:
        mn = col.mean()
        sd = col.std()
        mi = col.min()
        ma = col.max()
        both_valid = col.notna() & pm_vals.notna()
        if both_valid.sum() > 100:
            corr = col[both_valid].corr(pm_vals[both_valid])
        else:
            corr = np.nan
        P(f"| {f} | {mn:.4g} | {sd:.4g} | {mi:.4g} | {ma:.4g} | {nn:.1f}% | {corr:+.3f} |")
    else:
        P(f"| {f} | N/A | N/A | N/A | N/A | {nn:.1f}% | N/A |")
P()

# =====================================================================
# 5. TEMPORAL PATTERNS
# =====================================================================
P("## 5. Temporal Patterns")
P()

df["hour"] = df["ts"].dt.hour
df["month"] = df["ts"].dt.month
df["year_month"] = df["ts"].dt.to_period("M")

P("### Diurnal cycle (average PM2.5 by hour)")
P()
hourly = df.groupby("hour")["PM2.5"].mean()
P("| Hour | Mean PM2.5 |")
P("|------|-----------|")
for h in range(24):
    if h in hourly.index:
        P(f"| {h:02d}:00 | {hourly[h]:.2f} |")
P()

P("### Seasonal cycle (average PM2.5 by month)")
P()
monthly = df.groupby("month")["PM2.5"].mean()
P("| Month | Mean PM2.5 |")
P("|-------|-----------|")
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for m in range(1, 13):
    if m in monthly.index:
        P(f"| {m:2d} ({month_names[m-1]}) | {monthly[m]:.2f} |")
P()

P("### Station reporting coverage by month")
P()
ym_cov = df.groupby("year_month")["stationId"].nunique()
P("| Year-Month | N Stations |")
P("|------------|-----------|")
for ym in sorted(ym_cov.index):
    P(f"| {ym} | {ym_cov[ym]} |")
P()

# =====================================================================
# 6. SPATIAL PATTERNS
# =====================================================================
P("## 6. Spatial Patterns")
P()

region_stats = df.groupby("region").agg(
    n_stations=("stationId", "nunique"),
    mean_pm=("PM2.5", "mean"),
    std_pm=("PM2.5", "std"),
    n_hours=("PM2.5", "count"),
).reset_index()

P("### Per-region summary")
P()
P("| Region | N Stations | Mean PM2.5 | Std | Total Hours |")
P("|--------|------------|-----------|-----|-------------|")
for _, r in region_stats.iterrows():
    P(f"| {r['region']} | {int(r['n_stations'])} | {r['mean_pm']:.1f} | {r['std_pm']:.1f} | {int(r['n_hours']):,} |")
P()

P("### Top 5 stations by mean PM2.5")
P()
top5 = stn_stats.head(5)
P("| Station | Mean PM2.5 | Tier |")
P("|---------|-----------|------|")
for _, r in top5.iterrows():
    nm = ("" if pd.isna(r["name"]) else r["name"])[:50]
    P(f"| {nm} | {r['mean_pm']:.1f} | {r['tier']} |")
P()

P("### Bottom 5 stations by mean PM2.5")
P()
bot5 = stn_stats.tail(5)
P("| Station | Mean PM2.5 | Tier |")
P("|---------|-----------|------|")
for _, r in bot5.iterrows():
    nm = ("" if pd.isna(r["name"]) else r["name"])[:50]
    P(f"| {nm} | {r['mean_pm']:.1f} | {r['tier']} |")
P()

# =====================================================================
# 7. CORRELATIONS WITH PM2.5
# =====================================================================
P("## 7. Correlations with PM2.5")
P()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "PM2.5"]

corrs = {}
for c in numeric_cols:
    both = df[["PM2.5", c]].dropna()
    if len(both) > 100:
        corrs[c] = both["PM2.5"].corr(both[c])

corr_series = pd.Series(corrs).sort_values(ascending=False)

P("### Top 20 positive correlations")
P()
P("| Feature | Pearson r |")
P("|---------|----------|")
for f, v in corr_series.head(20).items():
    P(f"| {f} | {v:+.4f} |")
P()

P("### Top 10 negative correlations")
P()
P("| Feature | Pearson r |")
P("|---------|----------|")
for f, v in corr_series.tail(10).items():
    P(f"| {f} | {v:+.4f} |")
P()

# =====================================================================
# 8. QUICK SUMMARY FOR DEFENSE
# =====================================================================
P("## 8. Quick Summary for Defense")
P()

date_range_days = (df["ts"].max() - df["ts"].min()).days
P(f"- Dataset spans **{date_range_days} days** ({df['ts'].min().date()} to {df['ts'].max().date()})")
P(f"- **{n_stn} stations** profiled (no station-level exclusions; QC is row-level)")
P(f"- **{pm_nonnull:,} hourly PM2.5 observations** ({100*pm_nonnull/len(df):.1f}% non-null)")
P(f"- **{N_CSV_COLS} columns** in canonical CSV; final regional model selects 59 observation/physics columns and adds 8 fold-specific RFSI features (67 total)")
P(f"- PM2.5 range: {pm.min():.1f} - {pm.max():.1f} ug/m3, mean {pm.mean():.1f}, median {pm.median():.1f}")
P(f"- Tier distribution: t0={len(stn_stats[stn_stats['tier']=='t0'])}, t1={len(stn_stats[stn_stats['tier']=='t1'])}, t2={len(stn_stats[stn_stats['tier']=='t2'])}, t3={len(stn_stats[stn_stats['tier']=='t3'])}")

# most/least covered key features
P(f"- Best-covered key features: Temperature ({cov.get('Temperature_final', 0):.1f}%), "
  f"PBLH ({cov.get('PBLH', 0):.1f}%), wind ({cov.get('wind_u', 0):.1f}%)")
raw_aod = cov.get("AOT__0_0", 0)
ffill_aod = cov.get("AOT_ffill_48h", 0)
P(f"- AOD coverage: raw center pixel {raw_aod:.1f}%, after 48h forward-fill {ffill_aod:.1f}%")
P(f"- Top correlated features: {', '.join(corr_series.head(5).index.tolist())}")
P()

# Save
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n\nSaved to {OUT_MD}")
