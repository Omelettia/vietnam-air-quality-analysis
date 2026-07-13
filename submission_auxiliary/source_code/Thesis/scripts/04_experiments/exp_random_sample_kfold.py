"""Global sample-wise KFold diagnostic for the 40-station national dataset.

All hourly rows from all stations are mixed and split randomly into five folds.
Consequently, every station has observations in both training and validation.
This measures prediction at known stations and is deliberately reported
separately from station-wise holdout evaluation (LOSO).

The headline metric is pooled out-of-fold R2 over all rows. Station-level R2
statistics are saved as supplementary diagnostics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "Thesis" / "results" / "03_model"

from exp_national_loso_diagnostic import prepare_data


XGB_RANDOM_SAMPLE_PARAMS = {
    "n_estimators": 800,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "tree_method": "hist",
    "max_bin": 256,
    "random_state": 42,
    "verbosity": 0,
}

# Controlled-comparison variant: exactly the national LOSO configuration
# (exp_national_loso_diagnostic.py) so that the only difference from the LOSO
# result is the fold scheme (random rows vs whole held-out stations).
XGB_LOSO_MATCHED_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 10.0,
    "tree_method": "hist",
    "max_bin": 256,
    "random_state": 42,
    "verbosity": 0,
}

CONFIG_SPECS = {
    "strong": {
        "params": XGB_RANDOM_SAMPLE_PARAMS,
        "log_target": False,
        "station_weights": True,
        "suffix": "",
        "model_line": "XGBoost, 800 trees, depth 8, learning rate 0.05",
        "target_line": "raw PM2.5",
        "balance_line": "inverse-frequency station weights",
    },
    "loso": {
        "params": XGB_LOSO_MATCHED_PARAMS,
        "log_target": True,
        "station_weights": False,
        "suffix": "_loso_config",
        "model_line": "XGBoost, 500 trees, depth 4 (identical to national LOSO)",
        "target_line": "log1p PM2.5 (metrics on expm1 scale)",
        "balance_line": "none (identical to national LOSO)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run global five-fold random-sample KFold on the national data."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="XGBoost device. 'auto' probes CUDA and otherwise uses CPU.",
    )
    parser.add_argument(
        "--config",
        choices=sorted(CONFIG_SPECS),
        default="strong",
        help=(
            "'strong' is the headline known-station diagnostic; 'loso' reruns the "
            "same random KFold with the exact national LOSO model/target so the "
            "fold scheme is the only difference."
        ),
    )
    parser.add_argument(
        "--features",
        choices=["full", "obs"],
        default="full",
        help=(
            "'obs' drops static/location features (lat/lon, DEM, region one-hots) "
            "for an observation-only control; outputs get an _obsonly suffix."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def resolve_device(requested: str, X: np.ndarray, y: np.ndarray) -> str:
    if requested != "auto":
        return requested
    probe = xgb.XGBRegressor(
        n_estimators=2,
        max_depth=2,
        tree_method="hist",
        device="cuda",
        n_jobs=1,
        verbosity=0,
    )
    try:
        probe.fit(X[:300], y[:300])
        return "cuda"
    except Exception:
        return "cpu"


def build_report(
    summary: dict[str, object],
    fold_metrics: pd.DataFrame,
    station_metrics: pd.DataFrame,
    elapsed_seconds: float,
    spec: dict[str, object],
) -> str:
    title = "GLOBAL RANDOM-SAMPLE KFOLD DIAGNOSTIC"
    if spec["suffix"]:
        title += " (LOSO-MATCHED CONFIG CONTROL)"
    lines = [
        "=" * 80,
        title,
        "=" * 80,
        "",
        "Protocol: mix all hourly rows from 40 stations, then shuffled 5-fold KFold.",
        "Every station therefore appears in both training and validation.",
        "This is a known-station/sample-wise diagnostic, not spatial transfer.",
    ]
    if spec["suffix"]:
        lines += [
            "",
            "Purpose: controlled comparison against the national LOSO result.",
            "Model, target transform and weighting are identical to",
            "exp_national_loso_diagnostic.py; only the fold scheme differs.",
        ]
    lines += [
        "",
        "Data and model:",
        f"  rows                 = {summary['n_rows']:,}",
        f"  stations             = {summary['n_stations']}",
        f"  observable features  = {summary['n_features']} (no RFSI)",
        f"  target               = {spec['target_line']}",
        f"  model                = {spec['model_line']}",
        f"  balancing            = {spec['balance_line']}",
        "",
        "Headline result:",
        f"  pooled OOF R2        = {summary['pooled_r2']:.4f}",
        f"  pooled RMSE          = {summary['pooled_rmse']:.2f}",
        f"  pooled MAE           = {summary['pooled_mae']:.2f}",
        f"  mean of 5 fold R2    = {fold_metrics['r2'].mean():.4f}",
        "",
        "Supplementary station-level aggregation:",
        f"  mean station R2      = {station_metrics['r2'].mean():.4f}",
        f"  median station R2    = {station_metrics['r2'].median():.4f}",
        f"  positive stations    = {(station_metrics['r2'] > 0).sum()}/{len(station_metrics)}",
        "",
        "Fold R2:",
    ]
    for row in fold_metrics.itertuples(index=False):
        lines.append(
            f"  fold {row.fold}: R2={row.r2:.4f}, RMSE={row.rmse:.2f}, MAE={row.mae:.2f}"
        )
    lines += ["", f"Elapsed seconds: {elapsed_seconds:.1f}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    spec = CONFIG_SPECS[args.config]
    started = time.time()

    print("Loading the national observable-feature matrix...", flush=True)
    df, meta, feature_cols = prepare_data(obs_only=args.features == "obs")
    X = df[feature_cols].astype(np.float32).to_numpy()
    y = df["PM2.5"].to_numpy(dtype=float)
    y_fit = np.log1p(np.clip(y, 0, None)) if spec["log_target"] else y
    station_ids = df["stationId"].astype(str).to_numpy()

    counts = pd.Series(station_ids).value_counts()
    if spec["station_weights"]:
        target_rows_per_station = len(df) / len(counts)
        sample_weights = np.asarray(
            [target_rows_per_station / counts[sid] for sid in station_ids],
            dtype=np.float32,
        )
    else:
        sample_weights = np.ones(len(df), dtype=np.float32)

    device = resolve_device(args.device, X, y)
    params = {
        **spec["params"],
        "device": device,
        "n_jobs": args.n_jobs,
    }
    print(
        f"Config={args.config} rows={len(df):,} stations={len(counts)} "
        f"features={len(feature_cols)} device={device}",
        flush=True,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(df), np.nan, dtype=float)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = xgb.XGBRegressor(**params)
        model.fit(X[train_idx], y_fit[train_idx], sample_weight=sample_weights[train_idx])
        pred = model.predict(X[test_idx])
        if spec["log_target"]:
            pred = np.clip(np.expm1(pred), 0, 500)
        oof[test_idx] = pred
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "r2": r2_score(y[test_idx], pred),
                "rmse": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
                "mae": mean_absolute_error(y[test_idx], pred),
            }
        )
        print(f"Fold {fold}/5 complete: R2={fold_rows[-1]['r2']:.4f}", flush=True)

    valid = np.isfinite(oof)
    pooled_r2 = r2_score(y[valid], oof[valid])
    pooled_rmse = float(np.sqrt(mean_squared_error(y[valid], oof[valid])))
    pooled_mae = mean_absolute_error(y[valid], oof[valid])

    meta_by_sid = meta.set_index("stationId")
    station_rows: list[dict[str, object]] = []
    for sid in sorted(counts.index):
        mask = station_ids == sid
        station_rows.append(
            {
                "station_id": sid,
                "station_name": meta_by_sid.at[sid, "station_name"],
                "region": meta_by_sid.at[sid, "region"],
                "tier": int(df.loc[mask, "tier_qc"].iloc[0]),
                "n_rows": int(mask.sum()),
                "pm25_mean": float(y[mask].mean()),
                "r2": r2_score(y[mask], oof[mask]),
                "rmse": float(np.sqrt(mean_squared_error(y[mask], oof[mask]))),
                "mae": mean_absolute_error(y[mask], oof[mask]),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)
    station_metrics = pd.DataFrame(station_rows)
    summary = {
        "model": f"XGBoost global random-sample KFold ({args.config} config)",
        "protocol": "global shuffled 5-fold KFold over hourly rows",
        "target": spec["target_line"],
        "n_features": len(feature_cols),
        "n_stations": len(counts),
        "n_rows": int(valid.sum()),
        "pooled_r2": pooled_r2,
        "pooled_rmse": pooled_rmse,
        "pooled_mae": pooled_mae,
        "mean_fold_r2": fold_metrics["r2"].mean(),
        "mean_station_r2": station_metrics["r2"].mean(),
        "median_station_r2": station_metrics["r2"].median(),
        "positive_station_pct": 100.0 * (station_metrics["r2"] > 0).mean(),
    }
    elapsed = time.time() - started
    report = build_report(summary, fold_metrics, station_metrics, elapsed, spec)

    suffix = spec["suffix"] + ("_obsonly" if args.features == "obs" else "")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(
        RESULT_DIR / f"random_sample_kfold_summary{suffix}.csv", index=False
    )
    fold_metrics.to_csv(RESULT_DIR / f"random_sample_kfold_folds{suffix}.csv", index=False)
    station_metrics.to_csv(
        RESULT_DIR / f"random_sample_kfold_station_metrics{suffix}.csv", index=False
    )
    (RESULT_DIR / f"report_random_sample_kfold{suffix}.txt").write_text(
        report, encoding="utf-8"
    )

    print("\n" + report, flush=True)
    print(f"Saved results under: {RESULT_DIR}")


if __name__ == "__main__":
    main()
