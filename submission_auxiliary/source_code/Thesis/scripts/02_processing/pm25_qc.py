"""Shared PM2.5 quality-control helpers for thesis experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pm25_quality_masks(
    df: pd.DataFrame,
    station_col: str = "stationId",
    ts_col: str = "ts",
    value_col: str = "PM2.5",
    flat_run_hours: int = 5,
    round_decimals: int = 1,
    stuck_low_threshold: float = 2.0,
    stuck_low_run_hours: int = 48,
    max_gap_hours: float = 1.5,
    zero_floor: float = 0.0,
    max_valid: float = 500.0,
) -> pd.DataFrame:
    """Return row-level QC masks for PM2.5.

    Flatlines are detected on values rounded to one decimal place and only across
    consecutive hourly records. Sustained "stuck-low" runs catch sensors that sit
    near the detection floor without repeating the exact same value. If a run is
    long enough, the whole run is masked, including the first values in the run.
    """
    required = {station_col, ts_col, value_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for PM2.5 QC: {sorted(missing)}")

    values = pd.to_numeric(df[value_col], errors="coerce")
    masks = pd.DataFrame(
        False,
        index=df.index,
        columns=["zero_or_negative", "too_high", "flatline", "stuck_low"],
    )
    masks["zero_or_negative"] = values <= zero_floor
    masks["too_high"] = values > max_valid

    sorted_df = df[[station_col, ts_col, value_col]].copy()
    sorted_df[ts_col] = pd.to_datetime(sorted_df[ts_col])
    sorted_df[value_col] = pd.to_numeric(sorted_df[value_col], errors="coerce")
    sorted_df = sorted_df.sort_values([station_col, ts_col])

    flat = pd.Series(False, index=df.index)
    stuck_low = pd.Series(False, index=df.index)
    for _, group in sorted_df.groupby(station_col, sort=False):
        v = group[value_col]
        rounded = v.round(round_decimals)
        dt_hours = group[ts_col].diff().dt.total_seconds().div(3600)
        valid = v.notna() & (v > zero_floor) & (v <= max_valid)
        consecutive = dt_hours.le(max_gap_hours)
        same_as_prev = valid & valid.shift(fill_value=False) & consecutive & rounded.eq(rounded.shift())
        run_id = (~same_as_prev).cumsum()
        run_len = run_id.groupby(run_id).transform("size")
        flat.loc[group.index] = valid & run_len.ge(flat_run_hours)

        low = valid & v.le(stuck_low_threshold)
        same_low = low & low.shift(fill_value=False) & consecutive
        low_run_id = (~same_low).cumsum()
        low_run_len = low_run_id.groupby(low_run_id).transform("size")
        stuck_low.loc[group.index] = low & low_run_len.ge(stuck_low_run_hours)

    masks["flatline"] = flat
    masks["stuck_low"] = stuck_low
    return masks


def summarize_pm25_quality(
    df: pd.DataFrame,
    masks: pd.DataFrame,
    station_col: str = "stationId",
    value_col: str = "PM2.5",
) -> pd.DataFrame:
    """Summarize PM2.5 QC masks by station."""
    summary = (
        pd.DataFrame(
            {
                station_col: df[station_col],
                "pm25_raw": pd.to_numeric(df[value_col], errors="coerce"),
                "qc_any": masks.any(axis=1),
                "qc_zero_or_negative": masks["zero_or_negative"],
                "qc_too_high": masks["too_high"],
                "qc_flatline": masks["flatline"],
                "qc_stuck_low": masks["stuck_low"],
            }
        )
        .groupby(station_col)
        .agg(
            n_rows=("pm25_raw", "size"),
            raw_valid=("pm25_raw", "count"),
            raw_mean=("pm25_raw", "mean"),
            qc_any=("qc_any", "sum"),
            qc_zero_or_negative=("qc_zero_or_negative", "sum"),
            qc_too_high=("qc_too_high", "sum"),
            qc_flatline=("qc_flatline", "sum"),
            qc_stuck_low=("qc_stuck_low", "sum"),
        )
        .reset_index()
    )
    summary["qc_pct"] = summary["qc_any"] / summary["n_rows"]
    return summary


def flatline_runs(
    df: pd.DataFrame,
    station_col: str = "stationId",
    ts_col: str = "ts",
    value_col: str = "PM2.5",
    flat_run_hours: int = 5,
    round_decimals: int = 1,
    max_gap_hours: float = 1.5,
    zero_floor: float = 0.0,
    max_valid: float = 500.0,
) -> pd.DataFrame:
    """Return detected nonzero PM2.5 flatline runs."""
    sorted_df = df[[station_col, ts_col, value_col]].copy()
    sorted_df[ts_col] = pd.to_datetime(sorted_df[ts_col])
    sorted_df[value_col] = pd.to_numeric(sorted_df[value_col], errors="coerce")
    sorted_df = sorted_df.sort_values([station_col, ts_col])

    rows: list[dict[str, object]] = []
    for station_id, group in sorted_df.groupby(station_col, sort=False):
        v = group[value_col]
        rounded = v.round(round_decimals)
        dt_hours = group[ts_col].diff().dt.total_seconds().div(3600)
        valid = v.notna() & (v > zero_floor) & (v <= max_valid)
        consecutive = dt_hours.le(max_gap_hours)
        same_as_prev = valid & valid.shift(fill_value=False) & consecutive & rounded.eq(rounded.shift())
        run_id = (~same_as_prev).cumsum()
        tmp = group.assign(_rounded=rounded, _valid=valid, _run_id=run_id)
        for _, run in tmp[tmp["_valid"]].groupby("_run_id", sort=False):
            if len(run) < flat_run_hours:
                continue
            rows.append(
                {
                    station_col: station_id,
                    "start": run[ts_col].iloc[0],
                    "end": run[ts_col].iloc[-1],
                    "hours": len(run),
                    "value_rounded": run["_rounded"].iloc[0],
                    "raw_min": run[value_col].min(),
                    "raw_max": run[value_col].max(),
                }
            )

    return pd.DataFrame(rows)


def stuck_low_runs(
    df: pd.DataFrame,
    station_col: str = "stationId",
    ts_col: str = "ts",
    value_col: str = "PM2.5",
    stuck_low_threshold: float = 2.0,
    stuck_low_run_hours: int = 48,
    max_gap_hours: float = 1.5,
    zero_floor: float = 0.0,
    max_valid: float = 500.0,
) -> pd.DataFrame:
    """Return detected sustained near-zero PM2.5 runs."""
    sorted_df = df[[station_col, ts_col, value_col]].copy()
    sorted_df[ts_col] = pd.to_datetime(sorted_df[ts_col])
    sorted_df[value_col] = pd.to_numeric(sorted_df[value_col], errors="coerce")
    sorted_df = sorted_df.sort_values([station_col, ts_col])

    rows: list[dict[str, object]] = []
    for station_id, group in sorted_df.groupby(station_col, sort=False):
        v = group[value_col]
        dt_hours = group[ts_col].diff().dt.total_seconds().div(3600)
        valid = v.notna() & (v > zero_floor) & (v <= max_valid)
        low = valid & v.le(stuck_low_threshold)
        consecutive = dt_hours.le(max_gap_hours)
        same_low = low & low.shift(fill_value=False) & consecutive
        run_id = (~same_low).cumsum()
        tmp = group.assign(_valid=valid, _low=low, _run_id=run_id)
        for _, run in tmp[tmp["_low"]].groupby("_run_id", sort=False):
            if len(run) < stuck_low_run_hours:
                continue
            rows.append(
                {
                    station_col: station_id,
                    "start": run[ts_col].iloc[0],
                    "end": run[ts_col].iloc[-1],
                    "hours": len(run),
                    "threshold": stuck_low_threshold,
                    "raw_min": run[value_col].min(),
                    "raw_max": run[value_col].max(),
                }
            )

    return pd.DataFrame(rows)
