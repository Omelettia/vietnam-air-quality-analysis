"""
Generate readable text reports from result CSVs.
Reproduces the same formatted output the experiment scripts print to console.

Usage: python generate_reports.py
Outputs one .txt report per result set into Thesis/results/
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

RESULTS = "results"


def report_within_station():
    ws = pd.read_csv(f"{RESULTS}/03_model/within_station_predictability.csv")
    lines = []
    P = lines.append

    P("=" * 80)
    P("WITHIN-STATION PREDICTABILITY")
    P("Train on station's own data, predict using satellite + weather (no PM2.5)")
    P("=" * 80)

    P(f"\n  Stations: {len(ws)}")
    P(f"\n  {'Metric':<20s} {'Median':>8s} {'Mean':>8s} {'Min':>8s} {'Max':>8s}")
    P("  " + "-" * 52)
    for col, label in [("wk_exog", "KFold exog R2"),
                       ("wt_exog", "Temporal exog R2"),
                       ("wk_lag", "KFold + lag R2")]:
        s = ws[col]
        P(f"  {label:<20s} {s.median():>8.3f} {s.mean():>8.3f} "
          f"{s.min():>8.3f} {s.max():>8.3f}")

    P(f"\n  PER-TIER BREAKDOWN:")
    P(f"  {'Tier':<6s} {'N':>3s} {'KFold exog':>12s} {'Temporal':>12s} "
      f"{'KFold+lag':>12s} {'PM2.5 mean':>12s}")
    P("  " + "-" * 57)
    for t in ["t0", "t1", "t2", "t3"]:
        ts = ws[ws["tier"] == t]
        if len(ts) == 0:
            continue
        P(f"  {t:<6s} {len(ts):>3d} {ts['wk_exog'].median():>12.3f} "
          f"{ts['wt_exog'].median():>12.3f} {ts['wk_lag'].median():>12.3f} "
          f"{ts['pm'].median():>12.1f}")

    P(f"\n  STATION DETAIL (sorted by KFold exog R2):")
    P(f"  {'SID':<12s} {'Tier':>4s} {'PM2.5':>6s} {'KFold':>8s} "
      f"{'Temporal':>8s} {'+Lag':>8s} {'N hours':>8s}")
    P("  " + "-" * 54)
    for _, r in ws.sort_values("wk_exog", ascending=False).iterrows():
        sid = str(r["sid"])[-8:]
        P(f"  {sid:<12s} {r['tier']:>4s} {r['pm']:>6.1f} {r['wk_exog']:>8.3f} "
          f"{r['wt_exog']:>8.3f} {r['wk_lag']:>8.3f} {int(r['n']):>8d}")

    return "\n".join(lines)


def report_definitive_v3():
    df = pd.read_csv(f"{RESULTS}/03_model/definitive_v3.csv")
    imp = pd.read_csv(f"{RESULTS}/03_model/definitive_v3_importance.csv")
    lines = []
    P = lines.append

    ALL_CONFIGS = df["config"].unique().tolist()

    P("=" * 80)
    P("DEFINITIVE V3 — LOSO CROSS-VALIDATION RESULTS")
    P("XGBoost-DART, 66 features, per-tier tuning, 37 stations")
    P("=" * 80)

    # Config comparison
    P(f"\n{'='*80}")
    P("CONFIG COMPARISON SUMMARY  (mean across all stations)")
    P("=" * 80)

    SUMMARY_METRICS = [
        ("r2_hourly", "R2_h", "+.4f"),
        ("r2_daily", "R2_d", "+.4f"),
        ("rmse_hourly", "RMSE_h", ".2f"),
        ("rmse_daily", "RMSE_d", ".2f"),
        ("bias", "bias", "+.3f"),
        ("bias_pct", "bias%", "+.2f"),
        ("dir_acc_hourly", "dir_h%", ".1f"),
        ("dir_acc_daily", "dir_d%", ".1f"),
        ("p95_error", "P95err", ".2f"),
        ("bias_polluted", "bias>50", "+.2f"),
        ("bias_clean", "bias<15", "+.2f"),
        ("r2_djf", "R2_DJF", "+.4f"),
        ("r2_jja", "R2_JJA", "+.4f"),
    ]

    hdr = f"  {'Metric':<12s}"
    for c in ALL_CONFIGS:
        hdr += f" {c:>16s}"
    P(f"\n{hdr}")
    P("  " + "-" * (12 + 17 * len(ALL_CONFIGS)))

    for col, label, fmt in SUMMARY_METRICS:
        line = f"  {label:<12s}"
        for c in ALL_CONFIGS:
            v = df[df["config"] == c][col].dropna()
            if len(v) > 0:
                line += f" {format(v.mean(), fmt):>16s}"
            else:
                line += f" {'N/A':>16s}"
        P(line)

    # Per-tier breakdown
    P(f"\n{'='*80}")
    P("PER-TIER BREAKDOWN")
    P("=" * 80)

    KEY_METRICS = [
        ("r2_hourly", "R2_h", "+.4f"),
        ("r2_daily", "R2_d", "+.4f"),
        ("rmse_hourly", "RMSE_h", ".2f"),
        ("bias", "bias", "+.3f"),
        ("dir_acc_hourly", "dir_h%", ".1f"),
        ("p95_error", "P95err", ".2f"),
        ("bias_polluted", "bias>50", "+.2f"),
        ("bias_clean", "bias<15", "+.2f"),
    ]

    for tier in ["t0", "t1", "t2", "t3"]:
        t_df = df[df["tier"] == tier]
        n_t = len(t_df) // len(ALL_CONFIGS) if len(t_df) > 0 else 0
        if n_t == 0:
            continue

        P(f"\n  {tier} ({n_t} stations):")
        hdr = f"  {'Metric':<12s}"
        for c in ALL_CONFIGS:
            hdr += f" {c:>16s}"
        P(hdr)
        P("  " + "-" * (12 + 17 * len(ALL_CONFIGS)))

        for col, label, fmt in KEY_METRICS:
            line = f"  {label:<12s}"
            for c in ALL_CONFIGS:
                v = t_df[t_df["config"] == c][col].dropna()
                if len(v) > 0:
                    line += f" {format(v.mean(), fmt):>16s}"
                else:
                    line += f" {'N/A':>16s}"
            P(line)

    # Per-station detail for best config
    best_cfg = "dart_ensemble"
    P(f"\n{'='*80}")
    P(f"PER-STATION DETAIL  (config={best_cfg})")
    P("=" * 80)

    best = df[df["config"] == best_cfg].sort_values("r2_hourly", ascending=False)
    P(f"\n  {'Station':<40s} {'Tier':>4s} {'Region':>8s} {'PM2.5':>6s} "
      f"{'R2_h':>7s} {'RMSE':>7s} {'MAE':>7s} {'Bias':>7s} {'R2_d':>7s}")
    P("  " + "-" * 86)
    for _, r in best.iterrows():
        nm = str(r["station_name"])[:39]
        P(f"  {nm:<40s} {r['tier']:>4s} {str(r['region']):>8s} "
          f"{r['pm25_mean']:>6.1f} {r['r2_hourly']:>7.3f} {r['rmse_hourly']:>7.2f} "
          f"{r['mae_hourly']:>7.2f} {r['bias']:>7.2f} {r['r2_daily']:>7.3f}")

    # Feature importance — full ranking
    P(f"\n{'='*80}")
    P(f"FEATURE IMPORTANCE — FULL RANKING ({len(imp)} features)")
    P("=" * 80)

    gain_col = "gain" if "gain" in imp.columns else "importance"
    imp_sorted = imp.sort_values(gain_col, ascending=False)
    total_gain = imp[gain_col].sum()

    P(f"\n  {'Rank':>4s} {'Feature':<40s} {'Gain':>10s} {'Gain%':>8s} {'Cumul%':>8s}")
    P("  " + "-" * 72)
    cumul = 0.0
    for i, (_, r) in enumerate(imp_sorted.iterrows(), 1):
        pct = 100 * r[gain_col] / total_gain if total_gain > 0 else 0
        cumul += pct
        P(f"  {i:>4d} {str(r.iloc[0]):<40s} {r[gain_col]:>10.4f} {pct:>7.1f}% {cumul:>7.1f}%")

    # Grouped feature importance
    GROUPS = {
        "RFSI (nearby PM2.5)": [
            "PM25_nn_idw", "PM25_nn1", "PM25_nn2", "PM25_nn3",
            "PM25_nn1_lag1h", "PM25_nn1_lag3h", "PM25_nn1_lag6h", "dist_nn1",
        ],
        "Meteorological": [
            "Temperature_final", "Humidity_final", "Pressure_final",
            "PBLH", "PBLH_min_24h", "WS_local", "wind_u", "wind_v",
            "VC", "VC_min_24h", "rain_sum_48h", "rain_days_7d",
            "hrs_since_rain", "consecutive_dry_days", "RH_factor",
            "dT_6h", "dRH_6h", "temp_diurnal_anomaly",
            "stagnation_hours_12h",
        ],
        "Satellite AOD": [
            "AOT_outer_mean", "AOT_ffill_48h", "aod_outer_day_mean",
            "aod_outer_x_VC_inv", "aod_outer_gradient", "aod_outer_surface",
            "aod_outer_pm25", "hours_since_valid_AOT", "AE", "RF",
        ],
        "Satellite trace gases + emissions": [
            "so2_contrast", "so2_center", "so2_upwind",
            "so2_upwind_x_VC_inv", "so2_daily_anom", "so2_anom_x_vc_inv",
            "co_center", "co_upwind", "co_daily_anom", "co_anom_x_vc_inv",
            "hcho_center", "no2_daily_anom",
            "smart_v1_contrast", "smart_v1_max", "smart_v1_center",
            "smart_v1_upwind_x_VC_inv", "smart_v1_upwind",
        ],
        "Satellite LST anomaly": [
            "lst_day_anom", "lst_night_anom",
            "lst_anom_x_vc_inv", "lst_anom_upwind_x_VC_inv",
        ],
        "Temporal / cyclical": [
            "month_cos", "month_sin", "day_of_year_cos", "day_of_year_sin",
            "hour_sin", "hour_cos", "dow_is_weekend",
        ],
        "Land-use": [
            "building_area_1km",
        ],
    }

    P(f"\n{'='*80}")
    P("FEATURE GROUP SUMMARY")
    P("=" * 80)

    feat_gain = dict(zip(imp.iloc[:, 0].astype(str), imp[gain_col]))
    group_rows = []

    P(f"\n  {'Group':<35s} {'N feat':>6s} {'Sum Gain%':>10s} {'Top feature':<35s} {'Top%':>6s}")
    P("  " + "-" * 96)

    for gname, feats in GROUPS.items():
        matched = [(f, feat_gain.get(f, 0)) for f in feats if f in feat_gain]
        if not matched:
            continue
        gsum = sum(v for _, v in matched)
        gpct = 100 * gsum / total_gain if total_gain > 0 else 0
        top_f, top_v = max(matched, key=lambda x: x[1])
        top_pct = 100 * top_v / total_gain if total_gain > 0 else 0
        group_rows.append((gname, len(matched), gpct, top_f, top_pct))
        P(f"  {gname:<35s} {len(matched):>6d} {gpct:>9.1f}% {top_f:<35s} {top_pct:>5.1f}%")

    # Check for unassigned features
    all_grouped = set()
    for feats in GROUPS.values():
        all_grouped.update(feats)
    unassigned = [f for f in imp.iloc[:, 0].astype(str) if f not in all_grouped]
    if unassigned:
        usum = sum(feat_gain.get(f, 0) for f in unassigned)
        upct = 100 * usum / total_gain if total_gain > 0 else 0
        P(f"  {'(ungrouped)':35s} {len(unassigned):>6d} {upct:>9.1f}%")
        for f in unassigned:
            fpct = 100 * feat_gain.get(f, 0) / total_gain
            P(f"    - {f} ({fpct:.1f}%)")

    P(f"\n  Total: {len(imp)} features, {total_gain:.4f} total gain")

    # Per-group detail (top features within each group)
    P(f"\n{'='*80}")
    P("PER-GROUP DETAIL (features ranked within group)")
    P("=" * 80)

    for gname, feats in GROUPS.items():
        matched = [(f, feat_gain.get(f, 0)) for f in feats if f in feat_gain]
        if not matched:
            continue
        matched.sort(key=lambda x: -x[1])
        gsum = sum(v for _, v in matched)
        gpct = 100 * gsum / total_gain if total_gain > 0 else 0

        P(f"\n  {gname} — {len(matched)} features, {gpct:.1f}% total gain")
        P(f"  {'Feature':<40s} {'Gain%':>8s}")
        P("  " + "-" * 50)
        for f, v in matched:
            fpct = 100 * v / total_gain if total_gain > 0 else 0
            P(f"  {f:<40s} {fpct:>7.1f}%")

    return "\n".join(lines)


def report_ctm_baseline():
    lines = []
    P = lines.append

    P("=" * 80)
    P("CTM PRODUCT EVALUATION  (GEOS-CF, MERRA-2 vs ground truth)")
    P("=" * 80)

    for name, fname in [("GEOS-CF", "geoscf_station_metrics.csv"),
                        ("MERRA-2", "merra2_station_metrics.csv")]:
        path = f"{RESULTS}/02_ctm_baseline/{fname}"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        P(f"\n{'='*80}")
        P(f"{name} — {len(df)} stations")
        P("=" * 80)
        P(f"\n  Columns: {list(df.columns)}")

        r2_col = [c for c in df.columns if "r2" in c.lower() or "R2" in c]
        bias_col = [c for c in df.columns if "bias" in c.lower()]
        rmse_col = [c for c in df.columns if "rmse" in c.lower()]

        for col_list, label in [(r2_col, "R2"), (bias_col, "Bias"), (rmse_col, "RMSE")]:
            for col in col_list:
                v = df[col].dropna()
                if len(v) > 0:
                    P(f"\n  {col}:")
                    P(f"    median={v.median():.3f}, mean={v.mean():.3f}, "
                      f"min={v.min():.3f}, max={v.max():.3f}")

        if "tier" in df.columns:
            P(f"\n  Per-tier:")
            for t in sorted(df["tier"].unique()):
                ts = df[df["tier"] == t]
                for col in r2_col[:1]:
                    P(f"    {t}: n={len(ts)}, {col} median={ts[col].median():.3f}")

    return "\n".join(lines)


def report_tier_operational():
    df = pd.read_csv(f"{RESULTS}/04_validation/tier_operational_test.csv")
    lines = []
    P = lines.append

    P("=" * 80)
    P("TIER OPERATIONAL EXPERIMENT")
    P("Can we assign tiers without oracle labels?")
    P("=" * 80)

    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg]
        P(f"\n  {cfg} ({len(sub)} stations):")
        P(f"    Overall R2 median={sub['r2_hourly'].median():.3f}, "
          f"mean={sub['r2_hourly'].mean():.3f}")
        for t in ["t0", "t1", "t2", "t3"]:
            ts = sub[sub["tier"] == t]
            if len(ts):
                P(f"    {t}: n={len(ts)}, R2 median={ts['r2_hourly'].median():.3f}, "
                  f"mean={ts['r2_hourly'].mean():.3f}")

    return "\n".join(lines)


def report_external_validation():
    df = pd.read_csv(f"{RESULTS}/04_validation/external_validation.csv")
    lines = []
    P = lines.append

    P("=" * 80)
    P("EXTERNAL VALIDATION  (LCS + US Embassy)")
    P("=" * 80)

    P(f"\n  Total stations: {len(df)}")
    if "station_type" in df.columns:
        for st in df["station_type"].unique():
            sub = df[df["station_type"] == st]
            P(f"\n  {st} ({len(sub)} stations):")
            P(f"    Hourly R2: median={sub['r2_hourly'].median():.3f}, "
              f"mean={sub['r2_hourly'].mean():.3f}")
            if "r2_daily" in df.columns:
                P(f"    Daily R2:  median={sub['r2_daily'].median():.3f}, "
                  f"mean={sub['r2_daily'].mean():.3f}")
            P(f"    RMSE:      median={sub['rmse_hourly'].median():.2f}")

    if "region" in df.columns:
        P(f"\n  By region:")
        for rg in sorted(df["region"].dropna().unique()):
            sub = df[df["region"] == rg]
            P(f"    {rg:<10s}: n={len(sub):>3d}, "
              f"R2 median={sub['r2_hourly'].median():.3f}")

    P(f"\n  {'Station':<40s} {'Type':>4s} {'Region':>8s} {'Tier':>4s} "
      f"{'PM2.5':>6s} {'R2_h':>7s} {'RMSE':>7s}")
    P("  " + "-" * 76)
    for _, r in df.sort_values("r2_hourly", ascending=False).iterrows():
        nm = str(r["station_name"])[:39]
        P(f"  {nm:<40s} {str(r.get('station_type','')):>4s} "
          f"{str(r.get('region','')):>8s} {str(r.get('tier','')):>4s} "
          f"{r['pm25_mean']:>6.1f} {r['r2_hourly']:>7.3f} {r['rmse_hourly']:>7.2f}")

    return "\n".join(lines)


def report_red_river_delta():
    path = f"{RESULTS}/04_validation/delta_v1_test.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    lines = []
    P = lines.append

    P("=" * 80)
    P("RED RIVER DELTA — REGIONAL LOSO")
    P("12 KK stations, train within delta only, no tier grouping")
    P("=" * 80)

    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg]
        P(f"\n  {cfg} ({len(sub)} stations):")
        P(f"    R2 median={sub['r2_hourly'].median():.3f}, "
          f"mean={sub['r2_hourly'].mean():.3f}, "
          f"RMSE mean={sub['rmse'].mean():.2f}")

    best_cfg = "delta_rfsi"
    best = df[df["config"] == best_cfg].sort_values("r2_hourly", ascending=False)
    P(f"\n  PER-STATION ({best_cfg}):")
    P(f"  {'Station':<40s} {'Tier':>4s} {'PM2.5':>6s} {'R2':>7s} "
      f"{'RMSE':>7s} {'MAE':>7s} {'Bias':>7s}")
    P("  " + "-" * 71)
    for _, r in best.iterrows():
        nm = str(r["station_name"])[:39]
        P(f"  {nm:<40s} {r['tier']:>4s} {r['pm25_mean']:>6.1f} "
          f"{r['r2_hourly']:>7.3f} {r['rmse']:>7.2f} {r['mae']:>7.2f} "
          f"{r['bias']:>7.2f}")

    return "\n".join(lines)


def report_conformal():
    cov = pd.read_csv(f"{RESULTS}/05_conformal/conformal_coverage_v3.csv")
    lines = []
    P = lines.append

    P("=" * 80)
    P("CONFORMAL PREDICTION — COVERAGE & UNCERTAINTY")
    P("=" * 80)

    P(f"\n  {'Binning':<25s} {'Marginal':>8s} {'t0':>8s} {'t1':>8s} "
      f"{'t2':>8s} {'t3':>8s} {'TierGap':>8s} {'Nominal':>8s}")
    P("  " + "-" * 83)
    for _, r in cov.iterrows():
        P(f"  {r['binning']:<25s} {r['marginal_cov']:>8.3f} {r['cov_t0']:>8.3f} "
          f"{r['cov_t1']:>8.3f} {r['cov_t2']:>8.3f} {r['cov_t3']:>8.3f} "
          f"{r['tier_cov_range']:>8.3f} {r['nominal']:>8.1f}")

    rc_path = f"{RESULTS}/05_conformal/risk_coverage_v3.csv"
    if os.path.exists(rc_path):
        rc = pd.read_csv(rc_path)
        P(f"\n  Risk-coverage curve: {len(rc)} points")
        P(f"  Columns: {list(rc.columns)}")

    return "\n".join(lines)


# ── Generate all reports ──
reports = [
    ("03_model/report_within_station.txt", report_within_station),
    ("03_model/report_loso_results.txt", report_definitive_v3),
    ("02_ctm_baseline/report_ctm_evaluation.txt", report_ctm_baseline),
    ("04_validation/report_tier_operational.txt", report_tier_operational),
    ("04_validation/report_external_validation.txt", report_external_validation),
    ("04_validation/report_red_river_delta.txt", report_red_river_delta),
    ("05_conformal/report_conformal.txt", report_conformal),
]

for path, fn in reports:
    full_path = os.path.join(RESULTS, path)
    text = fn()
    if text is None:
        print(f"  SKIP (no data): {path}")
        continue
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"  Wrote: {path}")

print("\nDone — open any report_*.txt to review results.")
