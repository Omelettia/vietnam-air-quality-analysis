"""
Phase 1: Deep quality audit of all Tier 1+2+3 stations for thesis dataset.
Checks: flat-line detection, zero-inflation, diurnal cycle, temporal gaps,
value distribution, seasonal coverage, sentinel contamination, and cross-source
file matching.
"""

import io, sys, os, warnings, glob, unicodedata, json
from datetime import timedelta
from collections import OrderedDict

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# Repo root (Thesis/scripts/02_processing -> dirname x4). Reads analysis/... paths.
def _repo_root():
    p = os.path.abspath(os.path.dirname(__file__))
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "data", "merged")):
            return p
        p = os.path.dirname(p)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE = _repo_root()
os.chdir(BASE)

# ── File index builders (from v5) ─────────────────────────────────────
def normalize(s):
    return unicodedata.normalize("NFC", s.replace(":", "").replace("", "").strip())

def build_index(pattern, prefix_strip="", suffix_strip=".csv", space_char=" "):
    idx = {}
    for f in glob.glob(pattern):
        base = os.path.basename(f).replace(suffix_strip, "")
        if prefix_strip:
            base = base.replace(prefix_strip, "", 1)
        if space_char != " ":
            base = base.replace(space_char, " ")
        idx[normalize(base)] = f
    return idx

env_idx = {k: v for k, v in build_index("data/stations/historical_full/*.csv").items()
           if not v.endswith(".log")}
aod_idx = build_index("data/station_aod/L2/*.csv")
gpm_idx = build_index("data/gpm/station_gis_extracted_v2/*.csv", space_char="_")
wx_idx  = build_index("data/stations/weather/*.csv", prefix_strip="weather_")

# ── Load station tiers ─────────────────────────────────────────────────
tiers = pd.read_csv("analysis/v2_full_audit/station_tiers.csv")
tiers_123 = tiers[tiers["tier"].isin([1, 2, 3])].copy()
print(f"Stations to audit: {len(tiers_123)} (Tier 1: {(tiers_123['tier']==1).sum()}, "
      f"Tier 2: {(tiers_123['tier']==2).sum()}, Tier 3: {(tiers_123['tier']==3).sum()})")

# ── QC constants ───────────────────────────────────────────────────────
SENTINEL_VALUES = {-9999, -999, 9999, -9999.0, -999.0, 9999.0}
QC_RANGES = {
    "PM2.5": (0, 500), "PM10": (0, 1000), "Temperature": (-10, 50),
    "Humidity": (0, 100), "Pressure": (900, 1100),
    "Wind Speed": (0, 50), "Wind Direction": (0, 360),
}

# ── Audit functions ────────────────────────────────────────────────────

def detect_flatlines(pm25_series, threshold_hours=24):
    """Find consecutive runs of identical PM2.5 values."""
    if pm25_series.dropna().empty:
        return 0, 0
    vals = pm25_series.dropna().values
    max_run = 1
    current_run = 1
    total_flatline_hours = 0
    for i in range(1, len(vals)):
        if vals[i] == vals[i-1] and vals[i] != 0:
            current_run += 1
        else:
            if current_run >= threshold_hours:
                total_flatline_hours += current_run
            current_run = 1
            max_run = max(max_run, current_run)
    if current_run >= threshold_hours:
        total_flatline_hours += current_run
    max_run = max(max_run, current_run)
    return max_run, total_flatline_hours


def check_diurnal_cycle(df):
    """Check if PM2.5 shows diurnal variation. Returns std of hourly means."""
    if "PM2.5" not in df.columns or df["PM2.5"].dropna().empty:
        return np.nan, False
    hourly_mean = df.groupby(df["ts"].dt.hour)["PM2.5"].mean()
    if len(hourly_mean) < 12:
        return np.nan, False
    std_hourly = hourly_mean.std()
    has_cycle = std_hourly > 2.0
    return std_hourly, has_cycle


def find_longest_gap(ts_series):
    """Find the longest gap in timestamps (hours)."""
    if len(ts_series) < 2:
        return 0
    sorted_ts = ts_series.sort_values()
    diffs = sorted_ts.diff().dt.total_seconds() / 3600
    return diffs.max() if len(diffs) > 0 else 0


def check_seasonal_coverage(df):
    """Check if station has data in both dry (Oct-Mar) and wet (Apr-Sep) seasons."""
    if df.empty:
        return False, False, False
    months = df["ts"].dt.month
    has_dry = months.isin([10, 11, 12, 1, 2, 3]).any()
    has_wet = months.isin([4, 5, 6, 7, 8, 9]).any()
    return has_dry, has_wet, has_dry and has_wet


def check_sentinels(df, numeric_cols):
    """Count sentinel values in numeric columns."""
    count = 0
    details = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        for sv in SENTINEL_VALUES:
            n = (df[col] == sv).sum()
            if n > 0:
                count += n
                details.append(f"{col}={sv}({n})")
    return count, "; ".join(details) if details else ""


def check_negative_pm(df):
    """Count negative PM2.5 and PM10 values."""
    neg_pm25 = (df["PM2.5"] < 0).sum() if "PM2.5" in df.columns else 0
    neg_pm10 = (df["PM10"] < 0).sum() if "PM10" in df.columns else 0
    return neg_pm25, neg_pm10


def check_stuck_temp(df, threshold_hours=24):
    """Check for temperature stuck at exactly 0.0 for extended periods."""
    if "Temperature" not in df.columns:
        return 0
    temp = df["Temperature"].values
    max_run = 0
    current_run = 0
    for v in temp:
        if v == 0.0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


# ── Main audit loop ───────────────────────────────────────────────────
results = []
print("\n" + "=" * 80)
print("PHASE 1: DEEP QUALITY AUDIT")
print("=" * 80)

for i, (_, row) in enumerate(tiers_123.iterrows()):
    name = row["station_name"]
    sid = str(row["station_id"])
    tier = row["tier"]
    region = row["region"]
    stype = row["station_type"]
    norm = normalize(name)

    if (i + 1) % 10 == 0 or i == 0:
        print(f"\n  Processing {i+1}/{len(tiers_123)}...")

    # Find Envisoft file
    env_path = env_idx.get(norm)
    if not env_path:
        results.append({
            "station_name": name, "station_id": sid, "tier": tier,
            "region": region, "station_type": stype,
            "quality_flag": "fail", "fail_reason": "no_envisoft_file",
        })
        continue

    # Load data
    try:
        df = pd.read_csv(env_path)
    except Exception as e:
        results.append({
            "station_name": name, "station_id": sid, "tier": tier,
            "region": region, "station_type": stype,
            "quality_flag": "fail", "fail_reason": f"read_error: {e}",
        })
        continue

    df["ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    n_rows = len(df)

    if n_rows == 0:
        results.append({
            "station_name": name, "station_id": sid, "tier": tier,
            "region": region, "station_type": stype,
            "quality_flag": "fail", "fail_reason": "empty_file",
        })
        continue

    # Basic info
    data_start = df["ts"].min()
    data_end = df["ts"].max()
    data_months = (data_end - data_start).days / 30.44

    # PM2.5 stats
    has_pm25 = "PM2.5" in df.columns
    if has_pm25:
        pm_valid = df["PM2.5"].notna().sum()
        pm_total = len(df)
        active_hours = (data_end - data_start).total_seconds() / 3600 + 1
        pm_coverage = pm_valid / active_hours if active_hours > 0 else 0
        pm_mean = df["PM2.5"].mean()
        pm_median = df["PM2.5"].median()
        pm_std = df["PM2.5"].std()
        zero_count = (df["PM2.5"] == 0.0).sum()
        zero_pct = zero_count / pm_valid if pm_valid > 0 else 0
    else:
        pm_valid = 0
        pm_coverage = 0
        pm_mean = pm_median = pm_std = np.nan
        zero_count = 0
        zero_pct = 0

    # 1A checks
    max_flatline, total_flatline = detect_flatlines(df.get("PM2.5", pd.Series(dtype=float)))
    diurnal_std, has_diurnal = check_diurnal_cycle(df)
    longest_gap_h = find_longest_gap(df["ts"])
    has_dry, has_wet, has_both_seasons = check_seasonal_coverage(df)

    # 1B checks
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sentinel_count, sentinel_details = check_sentinels(df, numeric_cols)
    neg_pm25, neg_pm10 = check_negative_pm(df)
    stuck_temp_hours = check_stuck_temp(df)

    # Meteorological coverage
    has_temp = "Temperature" in df.columns and df["Temperature"].notna().mean() > 0.5
    has_hum = "Humidity" in df.columns and df["Humidity"].notna().mean() > 0.5

    # File matching
    has_aod = norm in aod_idx
    has_gpm = norm in gpm_idx
    has_wx = norm in wx_idx

    # 1C: Selection criteria
    fail_reasons = []
    marginal_reasons = []

    if pm_coverage < 0.50:
        fail_reasons.append(f"pm25_coverage={pm_coverage:.1%}")
    if data_months < 12:
        fail_reasons.append(f"data_months={data_months:.1f}")
    if max_flatline > 72:
        fail_reasons.append(f"flatline_{max_flatline}h")
    elif max_flatline > 24:
        marginal_reasons.append(f"flatline_{max_flatline}h")
    if zero_pct > 0.30:
        fail_reasons.append(f"zero_inflation={zero_pct:.1%}")
    elif zero_pct > 0.20:
        marginal_reasons.append(f"zero_inflation={zero_pct:.1%}")
    if has_pm25 and not has_diurnal:
        marginal_reasons.append("no_diurnal_cycle")
    if has_pm25 and not np.isnan(pm_mean):
        if pm_mean < 5 or pm_mean > 150:
            fail_reasons.append(f"pm25_mean={pm_mean:.1f}")
    if not has_aod and not has_gpm and not has_wx:
        fail_reasons.append("no_matching_external_files")
    if not has_aod:
        marginal_reasons.append("no_aod_file")
    if not has_both_seasons and data_months >= 12:
        marginal_reasons.append("single_season")
    if sentinel_count > 100:
        marginal_reasons.append(f"sentinels={sentinel_count}")
    if neg_pm25 > 50:
        marginal_reasons.append(f"neg_pm25={neg_pm25}")
    if stuck_temp_hours > 24:
        marginal_reasons.append(f"stuck_temp_0={stuck_temp_hours}h")

    if fail_reasons:
        quality_flag = "fail"
    elif marginal_reasons:
        quality_flag = "marginal"
    else:
        quality_flag = "pass"

    results.append({
        "station_name": name,
        "station_id": sid,
        "tier": tier,
        "region": region,
        "station_type": stype,
        "n_rows": n_rows,
        "data_start": str(data_start),
        "data_end": str(data_end),
        "data_months": round(data_months, 1),
        "pm25_valid": pm_valid,
        "pm25_coverage": round(pm_coverage, 4),
        "pm25_mean": round(pm_mean, 2) if not np.isnan(pm_mean) else np.nan,
        "pm25_median": round(pm_median, 2) if not np.isnan(pm_median) else np.nan,
        "pm25_std": round(pm_std, 2) if not np.isnan(pm_std) else np.nan,
        "zero_pct": round(zero_pct, 4),
        "max_flatline_h": max_flatline,
        "total_flatline_h": total_flatline,
        "diurnal_std": round(diurnal_std, 2) if not np.isnan(diurnal_std) else np.nan,
        "has_diurnal_cycle": has_diurnal,
        "longest_gap_h": round(longest_gap_h, 1),
        "has_dry_season": has_dry,
        "has_wet_season": has_wet,
        "has_both_seasons": has_both_seasons,
        "sentinel_count": sentinel_count,
        "sentinel_details": sentinel_details,
        "neg_pm25": neg_pm25,
        "neg_pm10": neg_pm10,
        "stuck_temp_0_h": stuck_temp_hours,
        "has_envisoft_met": has_temp and has_hum,
        "has_aod_file": has_aod,
        "has_gpm_file": has_gpm,
        "has_wx_file": has_wx,
        "quality_flag": quality_flag,
        "fail_reason": "; ".join(fail_reasons) if fail_reasons else "",
        "marginal_reason": "; ".join(marginal_reasons) if marginal_reasons else "",
    })

# ── Save results ───────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
out_path = "analysis/thesis_audit/station_selection.csv"
df_results.to_csv(out_path, index=False, encoding="utf-8-sig")

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PHASE 1 RESULTS SUMMARY")
print("=" * 80)

total = len(df_results)
n_pass = (df_results["quality_flag"] == "pass").sum()
n_marginal = (df_results["quality_flag"] == "marginal").sum()
n_fail = (df_results["quality_flag"] == "fail").sum()

print(f"\nTotal stations audited: {total}")
print(f"  PASS:     {n_pass:3d} ({100*n_pass/total:.0f}%)")
print(f"  MARGINAL: {n_marginal:3d} ({100*n_marginal/total:.0f}%)")
print(f"  FAIL:     {n_fail:3d} ({100*n_fail/total:.0f}%)")

print(f"\nBy Tier:")
for t in [1, 2, 3]:
    sub = df_results[df_results["tier"] == t]
    p = (sub["quality_flag"] == "pass").sum()
    m = (sub["quality_flag"] == "marginal").sum()
    f = (sub["quality_flag"] == "fail").sum()
    print(f"  Tier {t}: {len(sub):3d} total — {p} pass, {m} marginal, {f} fail")

print(f"\nBy Type:")
for t in df_results["station_type"].unique():
    sub = df_results[df_results["station_type"] == t]
    p = (sub["quality_flag"] == "pass").sum()
    m = (sub["quality_flag"] == "marginal").sum()
    f = (sub["quality_flag"] == "fail").sum()
    print(f"  {t}: {len(sub):3d} total — {p} pass, {m} marginal, {f} fail")

print(f"\nBy Region:")
for r in sorted(df_results["region"].dropna().unique()):
    sub = df_results[df_results["region"] == r]
    p = (sub["quality_flag"] == "pass").sum()
    print(f"  {r}: {len(sub):3d} total — {p} pass")

# Detailed failure reasons
print(f"\n--- FAILED STATIONS ({n_fail}) ---")
for _, r in df_results[df_results["quality_flag"] == "fail"].iterrows():
    print(f"  {r['station_name'][:55]:55s} | Tier {r['tier']} | {r['fail_reason']}")

# Marginal stations
print(f"\n--- MARGINAL STATIONS ({n_marginal}) ---")
for _, r in df_results[df_results["quality_flag"] == "marginal"].iterrows():
    print(f"  {r['station_name'][:55]:55s} | Tier {r['tier']} | {r['marginal_reason']}")

# Known bad wind stations check
print(f"\n--- KNOWN BAD WIND STATIONS ---")
bad_wind = ["Gia Lai Chư Sê", "Bắc Ninh KCN Tiên Sơn", "Bắc Ninh KCN Yên Phong", "Bắc Ninh Quế Võ"]
for _, r in df_results.iterrows():
    for bw in bad_wind:
        if bw in r["station_name"]:
            print(f"  {r['station_name'][:55]:55s} | Flag: {r['quality_flag']} | Sentinels: {r.get('sentinel_details','')}")

# Summary stats for passing stations
passing = df_results[df_results["quality_flag"].isin(["pass", "marginal"])]
if len(passing) > 0:
    print(f"\n--- PASSING + MARGINAL STATIONS STATS ---")
    print(f"  Count: {len(passing)}")
    print(f"  Total rows: {passing['n_rows'].sum():,}")
    print(f"  PM2.5 mean range: {passing['pm25_mean'].min():.1f} – {passing['pm25_mean'].max():.1f} µg/m³")
    print(f"  Coverage range: {passing['pm25_coverage'].min():.1%} – {passing['pm25_coverage'].max():.1%}")
    print(f"  With AOD: {passing['has_aod_file'].sum()}")
    print(f"  With GPM: {passing['has_gpm_file'].sum()}")
    print(f"  With OpenMeteo: {passing['has_wx_file'].sum()}")
    print(f"  With Envisoft met: {passing['has_envisoft_met'].sum()}")
    print(f"  KK: {(passing['station_type']=='KK').sum()}, LCS: {(passing['station_type']=='LCS').sum()}")
    print(f"  Provinces: {passing['region'].nunique()}")
    both = passing[passing["has_both_seasons"]]
    print(f"  With both seasons: {len(both)}")

print(f"\nSaved: {out_path}")
