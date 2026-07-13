"""Validate the effect of the shared PM2.5 QC mask as evidence for the row-level QC decision."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pm25_qc import flatline_runs, pm25_quality_masks, stuck_low_runs


PREVIOUSLY_EXCLUDED = {
    "31616865099255512061948816121",
    "30991938797551443885460120607",
    "29098319146067624969113973428",
}


def configure_console() -> None:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def assign_tier(pm: float) -> str:
    if not np.isfinite(pm):
        return "missing"
    if pm < 10:
        return "t0"
    if pm < 20:
        return "t1"
    if pm < 35:
        return "t2"
    return "t3"


def old_mask(df: pd.DataFrame) -> pd.Series:
    zero_mask = df["PM2.5"] == 0
    flat_mask = df.groupby("stationId")["PM2.5"].transform(
        lambda x: x.diff().eq(0).rolling(3, min_periods=3).sum().ge(3)
    )
    return zero_mask | flat_mask.fillna(False)


def station_recommendation(row: pd.Series) -> str:
    if row["remaining_flatline_rows"] > 0:
        return "fix_mask_before_model"
    if row["remaining_stuck_low_rows"] > 0:
        return "fix_mask_before_model"
    if row["new_qc_pct"] >= 0.20:
        return "exclude_or_sensitivity_only"
    if row["new_qc_pct"] >= 0.08:
        return "include_with_sensor_warning"
    if row["stationId"] in PREVIOUSLY_EXCLUDED:
        return "readd_candidate_after_qc"
    return "include"


def main() -> None:
    configure_console()
    root = project_root()
    out_dir = root / "Thesis/results/06_data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = root / "data/merged/unified_thesis.csv"
    df = pd.read_csv(
        data_path,
        usecols=["stationId", "station", "region", "ts", "PM2.5"],
        dtype={"stationId": str},
    )
    df["ts"] = pd.to_datetime(df["ts"])

    old = old_mask(df)
    new_masks = pm25_quality_masks(df)
    new = new_masks.any(axis=1)

    cleaned = df.copy()
    cleaned.loc[new, "PM2.5"] = np.nan
    remaining_runs = flatline_runs(cleaned)
    remaining_low_runs = stuck_low_runs(cleaned)
    if remaining_runs.empty:
        remaining_by_station = pd.DataFrame(
            columns=["stationId", "remaining_flatline_runs", "remaining_flatline_rows", "remaining_max_run"]
        )
    else:
        remaining_by_station = (
            remaining_runs.groupby("stationId")
            .agg(
                remaining_flatline_runs=("hours", "size"),
                remaining_flatline_rows=("hours", "sum"),
                remaining_max_run=("hours", "max"),
            )
            .reset_index()
        )
    if remaining_low_runs.empty:
        remaining_low_by_station = pd.DataFrame(
            columns=["stationId", "remaining_stuck_low_runs", "remaining_stuck_low_rows", "remaining_stuck_low_max_run"]
        )
    else:
        remaining_low_by_station = (
            remaining_low_runs.groupby("stationId")
            .agg(
                remaining_stuck_low_runs=("hours", "size"),
                remaining_stuck_low_rows=("hours", "sum"),
                remaining_stuck_low_max_run=("hours", "max"),
            )
            .reset_index()
        )

    base = (
        df.assign(
            old_mask=old,
            new_mask=new,
            new_zero_or_negative=new_masks["zero_or_negative"],
            new_flatline=new_masks["flatline"],
            new_stuck_low=new_masks["stuck_low"],
            new_too_high=new_masks["too_high"],
        )
        .groupby("stationId")
        .agg(
            station=("station", "first"),
            region=("region", "first"),
            n_rows=("PM2.5", "size"),
            raw_valid=("PM2.5", "count"),
            raw_mean=("PM2.5", "mean"),
            old_mask_rows=("old_mask", "sum"),
            new_mask_rows=("new_mask", "sum"),
            new_zero_or_negative_rows=("new_zero_or_negative", "sum"),
            new_flatline_rows=("new_flatline", "sum"),
            new_stuck_low_rows=("new_stuck_low", "sum"),
            new_too_high_rows=("new_too_high", "sum"),
        )
        .reset_index()
    )
    clean_stats = (
        cleaned.groupby("stationId")
        .agg(clean_valid=("PM2.5", "count"), clean_mean=("PM2.5", "mean"))
        .reset_index()
    )
    out = base.merge(clean_stats, on="stationId", how="left")
    out = out.merge(remaining_by_station, on="stationId", how="left")
    out = out.merge(remaining_low_by_station, on="stationId", how="left")
    for col in [
        "remaining_flatline_runs",
        "remaining_flatline_rows",
        "remaining_max_run",
        "remaining_stuck_low_runs",
        "remaining_stuck_low_rows",
        "remaining_stuck_low_max_run",
    ]:
        out[col] = out[col].fillna(0).astype(int)

    out["old_qc_pct"] = out["old_mask_rows"] / out["n_rows"]
    out["new_qc_pct"] = out["new_mask_rows"] / out["n_rows"]
    out["additional_rows_masked"] = out["new_mask_rows"] - out["old_mask_rows"]
    out["clean_coverage"] = out["clean_valid"] / out["n_rows"]
    out["raw_tier"] = out["raw_mean"].map(assign_tier)
    out["clean_tier"] = out["clean_mean"].map(assign_tier)
    out["previously_excluded"] = out["stationId"].isin(PREVIOUSLY_EXCLUDED)
    out["recommendation"] = out.apply(station_recommendation, axis=1)
    out = out.sort_values(["new_mask_rows", "additional_rows_masked"], ascending=False)

    csv_path = out_dir / "pm25_qc_effect_validation.csv"
    report_path = out_dir / "report_pm25_qc_effect_validation.txt"
    out.to_csv(csv_path, index=False)

    totals = {
        "old_mask_rows": int(old.sum()),
        "new_mask_rows": int(new.sum()),
        "additional_rows_masked": int(new.sum() - old.sum()),
        "new_zero_or_negative_rows": int(new_masks["zero_or_negative"].sum()),
        "new_flatline_rows": int(new_masks["flatline"].sum()),
        "new_stuck_low_rows": int(new_masks["stuck_low"].sum()),
        "new_too_high_rows": int(new_masks["too_high"].sum()),
        "remaining_flatline_runs_after_new_mask": int(len(remaining_runs)),
        "remaining_stuck_low_runs_after_new_mask": int(len(remaining_low_runs)),
    }
    readd = out[out["previously_excluded"]].copy()
    top = out.head(20)[
        [
            "stationId",
            "station",
            "region",
            "n_rows",
            "raw_mean",
            "clean_mean",
            "old_mask_rows",
            "new_mask_rows",
            "new_stuck_low_rows",
            "additional_rows_masked",
            "clean_coverage",
            "raw_tier",
            "clean_tier",
            "recommendation",
        ]
    ]

    report = []
    report.append("=" * 80)
    report.append("PM2.5 QC EFFECT VALIDATION")
    report.append("=" * 80)
    report.append("")
    report.append("Totals:")
    for key, value in totals.items():
        report.append(f"  {key}: {value:,}")
    report.append("")
    report.append("Previously excluded stations after new mask:")
    report.append(
        readd[
            [
                "stationId",
                "station",
                "region",
                "n_rows",
                "raw_mean",
                "clean_mean",
                "new_mask_rows",
                "clean_coverage",
                "raw_tier",
                "clean_tier",
                "recommendation",
            ]
        ].to_string(index=False, max_colwidth=72)
    )
    report.append("")
    report.append("Top stations affected by new mask:")
    report.append(top.to_string(index=False, max_colwidth=72))
    report.append("")
    if remaining_runs.empty and remaining_low_runs.empty:
        report.append("PASS: no >=5-hour one-decimal flatline runs or >=48-hour stuck-low runs remain after the new mask.")
    else:
        report.append("FAIL: flatline or stuck-low runs remain after the new mask.")
    report.append("")
    report.append(f"Wrote: {csv_path.relative_to(root)}")
    text = "\n".join(report)
    report_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote: {report_path.relative_to(root)}")


if __name__ == "__main__":
    main()
