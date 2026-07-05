"""Report PM2.5 rows removed by the shared thesis QC mask."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd

from pm25_qc import flatline_runs, pm25_quality_masks, stuck_low_runs, summarize_pm25_quality


def _configure_console() -> None:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    _configure_console()
    root = project_root()
    out_dir = root / "Thesis/results/06_data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = root / "data/merged/unified_thesis.csv"
    df = pd.read_csv(
        data_path,
        usecols=["stationId", "station", "region", "ts", "PM2.5"],
        dtype={"stationId": str},
    )
    masks = pm25_quality_masks(df)
    summary = summarize_pm25_quality(df, masks)
    names = df[["stationId", "station", "region"]].drop_duplicates("stationId")
    summary = summary.merge(names, on="stationId", how="left")
    summary = summary.sort_values(["qc_stuck_low", "qc_flatline", "qc_any"], ascending=False)
    runs = flatline_runs(df).merge(names, on="stationId", how="left")
    runs = runs.sort_values(["hours", "station"], ascending=[False, True])
    low_runs = stuck_low_runs(df).merge(names, on="stationId", how="left")
    low_runs = low_runs.sort_values(["hours", "station"], ascending=[False, True])

    csv_path = out_dir / "pm25_qc_mask_summary.csv"
    runs_path = out_dir / "pm25_flatline_runs.csv"
    low_runs_path = out_dir / "pm25_stuck_low_runs.csv"
    report_path = out_dir / "report_pm25_qc_mask.txt"
    summary.to_csv(csv_path, index=False)
    runs.to_csv(runs_path, index=False)
    low_runs.to_csv(low_runs_path, index=False)

    totals = summary[["qc_any", "qc_zero_or_negative", "qc_too_high", "qc_flatline", "qc_stuck_low"]].sum()
    top = summary.head(20)[
        [
            "stationId",
            "station",
            "region",
            "n_rows",
            "raw_mean",
            "qc_any",
            "qc_zero_or_negative",
            "qc_flatline",
            "qc_stuck_low",
            "qc_pct",
        ]
    ]

    report = []
    report.append("=" * 80)
    report.append("PM2.5 QC MASK SUMMARY")
    report.append("=" * 80)
    report.append("")
    report.append("Mask definition:")
    report.append("  - PM2.5 <= 0 is removed.")
    report.append("  - PM2.5 > 500 is removed.")
    report.append("  - Flatline runs are removed when rounded PM2.5 is unchanged")
    report.append("    to 0.1 ug/m3 for >=5 consecutive hourly records.")
    report.append("  - Stuck-low runs are removed when 0 < PM2.5 <= 2 ug/m3")
    report.append("    for >=48 consecutive hourly records.")
    report.append("")
    report.append("Totals:")
    for key, value in totals.items():
        report.append(f"  {key}: {int(value):,}")
    report.append("")
    report.append("Top stations by removed rows:")
    report.append(top.to_string(index=False, max_colwidth=72))
    report.append("")
    if not runs.empty:
        report.append("Longest detected flatline runs:")
        report.append(
            runs.head(20)[
                ["stationId", "station", "region", "start", "end", "hours", "value_rounded"]
            ].to_string(index=False, max_colwidth=72)
        )
        report.append("")
    if not low_runs.empty:
        report.append("Longest detected stuck-low runs:")
        report.append(
            low_runs.head(20)[
                ["stationId", "station", "region", "start", "end", "hours", "threshold", "raw_min", "raw_max"]
            ].to_string(index=False, max_colwidth=72)
        )
        report.append("")
    report.append(f"Wrote: {csv_path.relative_to(root)}")
    report.append(f"Wrote: {runs_path.relative_to(root)}")
    report.append(f"Wrote: {low_runs_path.relative_to(root)}")
    text = "\n".join(report)
    report_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote: {report_path.relative_to(root)}")


if __name__ == "__main__":
    main()
