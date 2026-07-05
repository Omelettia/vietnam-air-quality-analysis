"""Rerun the national station-held-out diagnostic used in Chapter 4.

The diagnostic asks a narrow question: with the same observable feature set,
how much does station-held-out skill change when the training pool is organized
as one global model, coarse geographic experts, or oracle pollution-tier
experts?

The oracle tier row is intentionally not deployable because it uses the held
station's true PM2.5 tier. It is kept only as a diagnostic ceiling for the
missing baseline/regime problem.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "Thesis" / "results" / "03_model"
RAW_OUT_DIR = ROOT / "analysis" / "thesis_experiments"
REPORT = OUT_DIR / "report_loso.txt"
SUMMARY_CSV = OUT_DIR / "national_loso_diagnostic_summary.csv"
STATION_CSV = OUT_DIR / "national_loso_diagnostic_station_metrics.csv"
OOF_CSV = RAW_OUT_DIR / "national_loso_diagnostic_oof.csv"

QC_DIR = ROOT / "Thesis" / "scripts" / "02_processing"
if str(QC_DIR) not in sys.path:
    sys.path.insert(0, str(QC_DIR))
from pm25_qc import pm25_quality_masks


CONFIGS = [
    {
        "config": "Global XGBoost",
        "interpretation": "One model trained on the other 39 stations",
        "deployable": "Yes",
        "pool": "global",
    },
    {
        "config": "Geographic region",
        "interpretation": "Expert trained from stations in the same North/Central/South region",
        "deployable": "Yes",
        "pool": "region",
    },
    {
        "config": "Oracle true-tier",
        "interpretation": "Expert trained from stations in the held station's true PM2.5 tier",
        "deployable": "No",
        "pool": "tier",
    },
]

EXCLUDE_FEATURES = {
    "PM2.5",
    "PM10",
    # These are local ground observations, not gridded predictors at a new site.
    "NO2",
    "O3",
    "SO2",
    "CO",
}
NON_FEATURE_COLUMNS = {
    "ts",
    "station",
    "stationId",
    "station_id",
    "station_name",
    "station_type",
    "region",
    "tier",
    "tier_qc",
    "pm25_mean_qc",
    "quality_flag",
    "fail_reason",
    "data_start",
    "data_end",
    "PBLH_source",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--save-oof", action="store_true")
    parser.add_argument("--quick-stations", type=int, default=0)
    return parser.parse_args()


def assign_tier(mean_pm: float) -> int:
    if mean_pm < 10:
        return 0
    if mean_pm < 20:
        return 1
    if mean_pm < 35:
        return 2
    return 3


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data_path = ROOT / "data" / "merged" / "unified_thesis.csv"
    meta_path = ROOT / "Thesis" / "results" / "01_stations" / "station_selection_final.csv"
    feature_path = ROOT / "Thesis" / "results" / "01_stations" / "feature_list.csv"

    df = pd.read_csv(data_path, dtype={"stationId": str}, low_memory=False)
    df["ts"] = pd.to_datetime(df["ts"])
    meta = pd.read_csv(meta_path, dtype={"stationId": str}, encoding="utf-8-sig")
    keep_sids = set(meta["stationId"])
    df = df[df["stationId"].isin(keep_sids)].copy()

    masks = pm25_quality_masks(df, station_col="stationId", ts_col="ts", value_col="PM2.5")
    df = df.loc[pd.to_numeric(df["PM2.5"], errors="coerce").notna() & ~masks.any(axis=1)].copy()
    df = df.merge(
        meta[["stationId", "station_name", "region", "tier", "lat", "lon"]],
        on="stationId",
        how="left",
        suffixes=("", "_meta"),
    )
    if "region_meta" in df.columns:
        df["region"] = df["region"].fillna(df["region_meta"])
        df.drop(columns=["region_meta"], inplace=True)

    # Recompute diagnostic tiers from the post-QC rows to avoid relying on a stale
    # metadata tier if the mask changes.
    qc_means = df.groupby("stationId")["PM2.5"].mean()
    tier_map = qc_means.map(assign_tier).to_dict()
    df["tier_qc"] = df["stationId"].map(tier_map).astype(int)
    meta["tier_qc"] = meta["stationId"].map(tier_map).astype("Int64")
    meta["pm25_mean_qc"] = meta["stationId"].map(qc_means)

    if feature_path.exists():
        feature_list = pd.read_csv(feature_path)
        candidate_cols = [c for c in feature_list["feature"].astype(str) if c in df.columns]
    else:
        candidate_cols = list(df.columns)

    feature_cols: list[str] = []
    for col in candidate_cols:
        if col in EXCLUDE_FEATURES or col in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    # Region is observable at a new location, so include it as one-hot context.
    region_dummies = pd.get_dummies(df["region"].fillna("Unknown"), prefix="region", dtype=float)
    df = pd.concat([df, region_dummies], axis=1)
    feature_cols.extend(list(region_dummies.columns))

    feature_cols = list(dict.fromkeys(feature_cols))
    return df, meta, feature_cols


def train_predict(
    x_all: np.ndarray,
    y_log: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    params: dict[str, object],
) -> np.ndarray:
    model = xgb.XGBRegressor(**params)
    model.fit(x_all[train_idx], y_log[train_idx])
    pred = np.expm1(model.predict(x_all[test_idx]))
    return np.clip(pred, 0, 500)


def station_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=float)
    p = frame["y_pred"].to_numpy(dtype=float)
    r2 = r2_score(y, p) if len(frame) >= 2 and np.nanstd(y) > 1e-9 else np.nan
    return {
        "n": int(len(frame)),
        "r2": float(r2),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "bias": float(np.mean(p - y)),
        "pm25_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
    }


def summarize(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_rows = []
    for (config, sid), sub in oof.groupby(["config", "stationId"], sort=False):
        met = station_metrics(sub)
        first = sub.iloc[0]
        station_rows.append(
            {
                "config": config,
                "stationId": sid,
                "station_name": first["station_name"],
                "region": first["region"],
                "tier_qc": int(first["tier_qc"]),
                **met,
            }
        )
    station_df = pd.DataFrame(station_rows)

    summary_rows = []
    for spec in CONFIGS:
        config = spec["config"]
        sub = oof[oof["config"] == config]
        st = station_df[station_df["config"] == config]
        pooled = r2_score(sub["y_true"], sub["y_pred"]) if len(sub) else np.nan
        summary_rows.append(
            {
                "config": config,
                "interpretation": spec["interpretation"],
                "deployable": spec["deployable"],
                "pooled_r2": float(pooled),
                "mean_station_r2": float(st["r2"].mean()),
                "median_station_r2": float(st["r2"].median()),
                "positive_station_pct": float((st["r2"] > 0).mean() * 100.0),
                "rmse": float(np.sqrt(mean_squared_error(sub["y_true"], sub["y_pred"]))),
                "mae": float(mean_absolute_error(sub["y_true"], sub["y_pred"])),
                "bias": float(np.mean(sub["y_pred"] - sub["y_true"])),
                "n_stations": int(st["stationId"].nunique()),
                "n_rows": int(len(sub)),
            }
        )
    return pd.DataFrame(summary_rows), station_df


def write_report(summary: pd.DataFrame, feature_cols: list[str], elapsed: float) -> None:
    lines = [
        "INTERNAL LOSO NATIONAL DIAGNOSTICS",
        "40 KK stations, train 39 -> predict 1",
        "",
        "This report is generated by rerunning the LOSO experiment from",
        "Thesis/scripts/04_experiments/exp_national_loso_diagnostic.py.",
        "",
        "Feature policy",
        "--------------",
        "Input features are observable numeric columns plus region one-hot context.",
        "PM2.5 is the target. PM10 and local Envisoft gas observations are excluded",
        "from the predictors to avoid using target-site ground observations as inputs.",
        f"Feature count: {len(feature_cols)}",
        "",
        "National diagnostic arc",
        "-----------------------",
        f"{'Config':24s} {'Pooled_R2':>9s} {'Mean_R2':>9s} {'Median_R2':>11s} {'Positive':>10s} {'RMSE':>8s} {'MAE':>8s} {'Bias':>8s}",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['config']:24s} "
            f"{row['pooled_r2']:9.3f} "
            f"{row['mean_station_r2']:9.3f} "
            f"{row['median_station_r2']:11.3f} "
            f"{row['positive_station_pct']:9.1f}% "
            f"{row['rmse']:8.2f} "
            f"{row['mae']:8.2f} "
            f"{row['bias']:8.2f}"
        )

    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            "The national rows are diagnostics, not the final deployable headline.",
            "Global XGBoost tests a single national model. Geographic region tests a",
            "coarse observable split. Oracle true-tier tests the upper-bound value of",
            "knowing the held station's true PM2.5 baseline tier and is not deployable.",
            "",
            "Final model direction",
            "---------------------",
            "The final thesis model is the Red River Delta regional XGBoost + RFSI/wind",
            "pipeline. See:",
            "",
            "Thesis/results/04_validation/report_red_river_delta_v5h.txt",
            "",
            f"Elapsed seconds: {elapsed:.1f}",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, meta, feature_cols = prepare_data()
    if args.quick_stations:
        quick_sids = sorted(meta["stationId"].dropna().astype(str).unique())[: args.quick_stations]
        df = df[df["stationId"].isin(quick_sids)].copy()
        meta = meta[meta["stationId"].isin(quick_sids)].copy()

    station_ids = sorted(df["stationId"].unique())
    params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=50,
        reg_alpha=0.1,
        reg_lambda=10.0,
        tree_method="hist",
        max_bin=256,
        device=args.device,
        random_state=42,
        n_jobs=args.n_jobs,
        verbosity=0,
    )

    print("=" * 80)
    print("NATIONAL LOSO DIAGNOSTIC")
    print("=" * 80)
    print(f"Rows after QC: {len(df):,}")
    print(f"Stations: {len(station_ids)}")
    print(f"Features: {len(feature_cols)}")
    print(f"XGBoost: {params}")

    x_all = df[feature_cols].astype(np.float32).to_numpy()
    y_true_all = df["PM2.5"].to_numpy(dtype=float)
    y_log = np.log1p(np.clip(y_true_all, 0, None))
    sid_values = df["stationId"].to_numpy()

    oof_rows = []
    for fold_i, held_sid in enumerate(station_ids, start=1):
        test_mask = sid_values == held_sid
        train_all_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        test = df.iloc[test_idx]
        train_all = df.loc[train_all_mask]
        first = test.iloc[0]
        print(
            f"[{fold_i:02d}/{len(station_ids)}] {str(first['station_name'])[:45]:45s} "
            f"region={first['region']} tier=t{int(first['tier_qc'])} n={len(test):,}"
        )

        for spec in CONFIGS:
            if spec["pool"] == "global":
                train_mask = train_all_mask
            elif spec["pool"] == "region":
                train_mask = train_all_mask & (df["region"].to_numpy() == first["region"])
                train = df.loc[train_mask]
                if train["stationId"].nunique() < 3:
                    train_mask = train_all_mask
            elif spec["pool"] == "tier":
                train_mask = train_all_mask & (df["tier_qc"].to_numpy() == int(first["tier_qc"]))
                train = df.loc[train_mask]
                if train["stationId"].nunique() < 3:
                    train_mask = train_all_mask
            else:
                raise ValueError(spec["pool"])
            train_idx = np.where(train_mask)[0]

            pred = train_predict(x_all, y_log, train_idx, test_idx, params)
            rows = pd.DataFrame(
                {
                    "config": spec["config"],
                    "stationId": held_sid,
                    "station_name": first["station_name"],
                    "region": first["region"],
                    "tier_qc": int(first["tier_qc"]),
                    "ts": test["ts"].to_numpy(),
                    "y_true": test["PM2.5"].to_numpy(dtype=float),
                    "y_pred": pred,
                }
            )
            met = station_metrics(rows)
            print(
                f"    {spec['config']:<18s} train_st={df.loc[train_mask, 'stationId'].nunique():2d} "
                f"R2={met['r2']:+.3f} RMSE={met['rmse']:.1f} bias={met['bias']:+.1f}"
            )
            oof_rows.append(rows)

    oof = pd.concat(oof_rows, ignore_index=True)
    summary, station_df = summarize(oof)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    station_df.to_csv(STATION_CSV, index=False, encoding="utf-8")
    if args.save_oof:
        oof.to_csv(OOF_CSV, index=False, encoding="utf-8")
    elapsed = time.time() - t0
    write_report(summary, feature_cols, elapsed)

    print("\nSUMMARY")
    print(summary[["config", "pooled_r2", "mean_station_r2", "median_station_r2", "positive_station_pct"]])
    print(f"\nwrote {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"wrote {STATION_CSV.relative_to(ROOT)}")
    if args.save_oof:
        print(f"wrote {OOF_CSV.relative_to(ROOT)}")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
