"""Build the non-fold-dependent feature layer for the regional PM2.5 model.

The base ``unified_thesis.csv`` is assembled from hourly ground observations,
Himawari, weather, rainfall and terrain.  This module enriches that same table
with the daily GEE/TROPOMI, MODIS temporal and static urban/source-context
features used by the Red River Delta experiment.

RFSI is intentionally *not* built here.  It depends on which station is held
out and must be recomputed inside every LOSO fold to prevent target leakage.

The module is both importable by ``build_unified.py`` and runnable on an
existing unified table::

    python Thesis/scripts/03_features/regional_feature_pipeline.py
"""

from __future__ import annotations

import argparse
import glob
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROLL_DAYS = 30
SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

ANOM_COLS = [
    "so2_daily_anom",
    "co_daily_anom",
    "no2_daily_anom",
    "hcho_daily_anom",
]
TROPOMI_ROLL_COLS = [
    "hcho_30d_mean",
    "hcho_30d_p90",
    "hcho_30d_cv",
    "co_30d_mean",
    "co_30d_std",
    "co_30d_iqr",
]
MODIS_DAILY_COLS = ["modis_aod_7d", "modis_fine_aod_7d"]
MODIS_ROLL_COLS = [
    "aod_30d_mean",
    "aod_30d_std",
    "aod_30d_iqr",
    "aod_30d_p90",
    "aod_30d_cv",
]

NO2_STATIC_COLS = ["no2_center"] + [f"no2_clim_{d}" for d in SECTOR_NAMES]
EMISSION_STATIC_COLS = (
    ["ntl_center"]
    + [f"ntl_clim_{d}" for d in SECTOR_NAMES]
    + ["lst_anom_center"]
    + [f"lst_anom_clim_{d}" for d in SECTOR_NAMES]
)
OTHER_STATIC_COLS = ["fmf_center"]
BUILDING_COLS = ["building_area_1km"]

ENRICHED_REQUIRED_COLS = (
    ANOM_COLS
    + TROPOMI_ROLL_COLS
    + MODIS_DAILY_COLS
    + MODIS_ROLL_COLS
    + NO2_STATIC_COLS
    + EMISSION_STATIC_COLS
    + OTHER_STATIC_COLS
    + BUILDING_COLS
)

# Final regional model feature policy.  These are defined here so feature
# construction and experiment code share one explicit source of truth.
SAT_AOD = [
    "AOT_ffill_48h",
    "AOT_outer_mean",
    "AOT_inner_mean",
    "AOT_fine",
    "RF",
    "AE",
    "AOT_spatial_std",
    "AOT_rolling_mean_24h",
    "hours_since_valid_AOT",
]
DAILY_SAT = MODIS_DAILY_COLS + ANOM_COLS
MET = [
    "PBLH",
    "VC",
    "wind_u",
    "wind_v",
    "WS_local",
    "Temperature_final",
    "Humidity_final",
    "Pressure_final",
    "dT_6h",
    "dRH_6h",
]
PRECIP = ["rain_days_7d", "consecutive_dry_days", "hrs_since_rain"]
TEMPORAL = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
STABILITY = ["PBLH_min_24h", "stagnation_hours_12h"]
SAT_REGIME = MODIS_ROLL_COLS + TROPOMI_ROLL_COLS
OBS_DERIVED = ["RH_factor", "aod_outer_pm25"]
PHYSICS_FEATS = [
    "aod_surface",
    "aod_dry",
    "co_surface",
    "hcho_surface",
    "no2_surface",
    "so2_surface",
    "combustion_aod",
    "secondary_form",
    "modis_surface",
    "stagnant_aod",
    "stagnant_co",
    "aod_anomaly",
]
REGIME_FEATS = [
    "building_area_1km",
    "no2_center",
    "ntl_center",
    "source_intensity_center",
    "urban_score",
    "industrial_score",
    "peri_rural_score",
    "urban_aod",
    "urban_stagnation",
    "urban_pblh_inv",
    "industrial_vent",
]

# Monotonicity constraints of the final regional model, shared by the LOSO
# experiment and the grid-mapping scripts.  Keys not present in a feature list
# simply default to 0 (unconstrained).
MONO_DICT = {
    "VC": -1, "PBLH": -1, "WS_local": -1,
    "PM25_nn_idw": 1, "PM25_nn1": 1,
    "PM25_nn2": 1, "PM25_nn3": 1,
    "PM25_upwind_idw": 1, "PM25_downwind_idw": 1,
    "modis_aod_7d": 1, "AOT_ffill_48h": 1,
    "aod_30d_mean": 1, "aod_30d_p90": 1,
    "hcho_30d_mean": 1, "co_30d_mean": 1,
    "aod_surface": 1, "aod_dry": 1, "co_surface": 1, "hcho_surface": 1,
    "combustion_aod": 1, "modis_surface": 1,
    "stagnant_aod": 1, "stagnant_co": 1,
}

# Columns that the regional experiment must read from the wide unified table.
# Derived physics/regime variables are rebuilt below, so loading their stored
# copies (or unrelated national-model columns) would only waste memory.
REGIONAL_SOURCE_COLUMNS = sorted(
    set(
        ["stationId", "ts", "PM2.5"]
        + SAT_AOD
        + DAILY_SAT
        + MET
        + PRECIP
        + TEMPORAL
        + SAT_REGIME
        + NO2_STATIC_COLS
        + EMISSION_STATIC_COLS
        + OTHER_STATIC_COLS
        + BUILDING_COLS
    )
)


def _drop_existing(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [c for c in columns if c in frame.columns]
    return frame.drop(columns=present) if present else frame


def _load_gee_daily(root: Path, station_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = sorted(glob.glob(str(root / "data" / "gee_exports" / "last-*.zip")))
    if not candidates:
        raise FileNotFoundError("No data/gee_exports/last-*.zip file was found")

    chunks: list[pd.DataFrame] = []
    with zipfile.ZipFile(candidates[-1]) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".csv"):
                continue
            with archive.open(name) as stream:
                chunks.append(pd.read_csv(stream, dtype={"stationId": str}))
    if not chunks:
        raise ValueError(f"GEE archive contains no CSV files: {candidates[-1]}")

    sat_long = pd.concat(chunks, ignore_index=True)
    sat_wide = sat_long.pivot_table(
        index=["stationId", "date"],
        columns="variable",
        values="mean",
        aggfunc="first",
    ).reset_index()
    sat_wide.columns.name = None
    sat_wide["date"] = pd.to_datetime(sat_wide["date"])
    sat_wide = sat_wide[sat_wide["stationId"].isin(station_ids)].copy()
    sat_wide["month"] = sat_wide["date"].dt.month

    clim_cols = ["NO2", "SO2", "CO", "HCHO"]
    for col in clim_cols:
        if col not in sat_wide.columns:
            sat_wide[col] = np.nan
    climatology = sat_wide.groupby(["stationId", "month"])[clim_cols].transform("mean")
    sat_wide["so2_daily_anom"] = sat_wide["SO2"] - climatology["SO2"]
    sat_wide["co_daily_anom"] = sat_wide["CO"] - climatology["CO"]
    sat_wide["no2_daily_anom"] = sat_wide["NO2"] - climatology["NO2"]
    sat_wide["hcho_daily_anom"] = sat_wide["HCHO"] - climatology["HCHO"]

    rolling_frames: list[pd.DataFrame] = []
    for sid, group in sat_wide.sort_values(["stationId", "date"]).groupby("stationId"):
        group = group.set_index("date").sort_index()
        out = pd.DataFrame(index=group.index)
        out["stationId"] = sid
        hcho = group["HCHO"]
        hcho_window = hcho.rolling(f"{ROLL_DAYS}D", min_periods=5)
        out["hcho_30d_mean"] = hcho_window.mean()
        out["hcho_30d_p90"] = hcho_window.quantile(0.9)
        out["hcho_30d_cv"] = hcho_window.std() / (hcho_window.mean().abs() + 1e-12)
        co = group["CO"]
        co_window = co.rolling(f"{ROLL_DAYS}D", min_periods=5)
        out["co_30d_mean"] = co_window.mean()
        out["co_30d_std"] = co_window.std()
        out["co_30d_iqr"] = co_window.quantile(0.75) - co_window.quantile(0.25)
        rolling_frames.append(out.reset_index())

    tropomi_roll = pd.concat(rolling_frames, ignore_index=True)
    sat_wide["date_merge"] = sat_wide["date"].dt.date
    tropomi_roll["date_merge"] = tropomi_roll["date"].dt.date
    return sat_wide, tropomi_roll


def _load_modis(root: Path, station_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "data" / "stations" / "metadata" / "modis_temporal_features.csv"
    modis = pd.read_csv(path, dtype={"stationId": str})
    modis["date"] = pd.to_datetime(modis["date"])
    modis = modis[modis["stationId"].isin(station_ids)].sort_values(["stationId", "date"])

    rolling_frames: list[pd.DataFrame] = []
    for sid, group in modis.groupby("stationId"):
        group = group.set_index("date").sort_index()
        aod = group["modis_aod_7d"]
        window = aod.rolling(f"{ROLL_DAYS}D", min_periods=10)
        out = pd.DataFrame(index=group.index)
        out["stationId"] = sid
        out["aod_30d_mean"] = window.mean()
        out["aod_30d_std"] = window.std()
        out["aod_30d_iqr"] = window.quantile(0.75) - window.quantile(0.25)
        out["aod_30d_p90"] = window.quantile(0.9)
        out["aod_30d_cv"] = out["aod_30d_std"] / (out["aod_30d_mean"] + 1e-9)
        rolling_frames.append(out.reset_index())

    modis_roll = pd.concat(rolling_frames, ignore_index=True)
    modis["date_merge"] = modis["date"].dt.date
    modis_roll["date_merge"] = modis_roll["date"].dt.date
    return modis, modis_roll


def _merge_static_features(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    meta = root / "data" / "stations" / "metadata"
    sources = [
        (meta / "station_no2_features.csv", NO2_STATIC_COLS),
        (meta / "station_emission_features.csv", EMISSION_STATIC_COLS),
        (meta / "station_all_satellite_features.csv", OTHER_STATIC_COLS),
        (meta / "station_building_density.csv", BUILDING_COLS),
    ]
    for path, requested in sources:
        table = pd.read_csv(path, dtype={"stationId": str})
        columns = [c for c in requested if c in table.columns]
        frame = _drop_existing(frame, columns)
        frame = frame.merge(
            table[["stationId"] + columns].drop_duplicates("stationId"),
            on="stationId",
            how="left",
            validate="m:1",
        )
    frame["building_area_1km"] = frame["building_area_1km"].fillna(0)
    return frame


def enrich_unified_dataframe(
    frame: pd.DataFrame,
    root: str | Path,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Attach all non-fold-dependent GEE/MODIS/static regional features."""
    root = Path(root).resolve()
    frame["stationId"] = frame["stationId"].astype(str)
    had_date = "date" in frame.columns
    if "ts" not in frame.columns:
        raise KeyError("unified table must contain a 'ts' column")
    frame["ts"] = pd.to_datetime(frame["ts"])
    frame["date"] = frame["ts"].dt.date
    station_ids = set(frame["stationId"].dropna().unique())

    if verbose:
        print(f"Enriching unified table for {len(station_ids)} stations...")
    sat_wide, tropomi_roll = _load_gee_daily(root, station_ids)
    modis, modis_roll = _load_modis(root, station_ids)

    dynamic_cols = ANOM_COLS + TROPOMI_ROLL_COLS + MODIS_DAILY_COLS + MODIS_ROLL_COLS
    frame = _drop_existing(frame, dynamic_cols)
    frame = frame.merge(
        sat_wide[["stationId", "date_merge"] + ANOM_COLS],
        left_on=["stationId", "date"],
        right_on=["stationId", "date_merge"],
        how="left",
        validate="m:1",
    ).drop(columns="date_merge")
    frame = frame.merge(
        tropomi_roll[["stationId", "date_merge"] + TROPOMI_ROLL_COLS],
        left_on=["stationId", "date"],
        right_on=["stationId", "date_merge"],
        how="left",
        validate="m:1",
    ).drop(columns="date_merge")
    frame = frame.merge(
        modis[["stationId", "date_merge"] + MODIS_DAILY_COLS],
        left_on=["stationId", "date"],
        right_on=["stationId", "date_merge"],
        how="left",
        validate="m:1",
    ).drop(columns="date_merge")
    frame = frame.merge(
        modis_roll[["stationId", "date_merge"] + MODIS_ROLL_COLS],
        left_on=["stationId", "date"],
        right_on=["stationId", "date_merge"],
        how="left",
        validate="m:1",
    ).drop(columns="date_merge")
    frame = _merge_static_features(frame, root)
    if not had_date:
        frame = frame.drop(columns="date")

    missing = [c for c in ENRICHED_REQUIRED_COLS if c not in frame.columns]
    if missing:
        raise RuntimeError(f"regional enrichment failed; missing columns: {missing}")
    if verbose:
        print(
            f"Regional enrichment complete: {len(frame):,} rows, "
            f"{len(ENRICHED_REQUIRED_COLS)} managed columns"
        )
    return frame


def require_enriched_unified(frame: pd.DataFrame) -> None:
    missing = [c for c in ENRICHED_REQUIRED_COLS if c not in frame.columns]
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            "unified_thesis.csv is missing regional GEE/MODIS enrichment "
            f"({preview}). Rebuild it with build_unified.py or run "
            "regional_feature_pipeline.py first."
        )


def read_unified_stations(
    path: str | Path,
    station_ids: set[str],
    *,
    usecols: list[str] | None = None,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Read selected stations and optional columns from the canonical CSV."""
    wanted = {str(sid) for sid in station_ids}
    selected_columns = set(usecols or [])
    selected_columns.add("stationId")
    usecols_filter = (
        (lambda column: column in selected_columns) if usecols is not None else None
    )
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        dtype={"stationId": str},
        low_memory=False,
        usecols=usecols_filter,
        chunksize=chunksize,
    ):
        selected = chunk[chunk["stationId"].isin(wanted)]
        if len(selected):
            chunks.append(selected)
    if not chunks:
        raise ValueError(f"none of the requested station IDs were found in {path}")
    return pd.concat(chunks, ignore_index=True)


def prepare_observation_features(
    frame: pd.DataFrame,
    label: str = "rows",
    skip_station_rolling: bool = False,
) -> pd.DataFrame:
    """Apply the canonical regional cleaning and observation-only interactions.

    ``skip_station_rolling=True`` keeps precomputed ``PBLH_min_24h`` and
    ``stagnation_hours_12h`` columns instead of rebuilding them from hourly
    per-station series — used for grid-cell rows, which carry only the map
    timestamps and no ``stationId``.
    """
    rh_bad = frame["Humidity_final"] < 5
    frame.loc[rh_bad, "Humidity_final"] = np.nan
    pressure_bad = (frame["Pressure_final"] < 950) | (frame["Pressure_final"] > 1040)
    frame.loc[pressure_bad, "Pressure_final"] = np.nan
    temperature_bad = (frame["Temperature_final"] < 0) | (frame["Temperature_final"] > 50)
    frame.loc[temperature_bad, "Temperature_final"] = np.nan
    pblh_bad = (frame["PBLH"] < 0) | (frame["PBLH"] > 6000)
    frame.loc[pblh_bad, "PBLH"] = np.nan
    frame.loc[frame["AE"].abs() > 5, "AE"] = np.nan
    frame.loc[frame["AOT_ffill_48h"] > 5, "AOT_ffill_48h"] = np.nan
    frame.loc[frame["AOT_outer_mean"] > 5, "AOT_outer_mean"] = np.nan
    wind_bad = (frame["WS_local"] == 0) & (
        np.sqrt(frame["wind_u"] ** 2 + frame["wind_v"] ** 2) > 1.0
    )
    frame.loc[wind_bad, "WS_local"] = np.nan

    cleaned = rh_bad | pressure_bad | temperature_bad | pblh_bad | wind_bad
    print(
        f"  Feature cleaning ({label}): {int(cleaned.sum()):>7,} rows "
        f"({100 * cleaned.sum() / max(len(frame), 1):.1f}%)"
    )

    wind_speed = np.sqrt(frame["wind_u"].to_numpy() ** 2 + frame["wind_v"].to_numpy() ** 2)
    pblh_fill = frame["PBLH"].fillna(200).to_numpy()
    frame["RH_factor"] = 1.0 / (1.0 - (frame["Humidity_final"] / 100.0).clip(upper=0.95))
    frame["VC"] = pblh_fill * np.clip(wind_speed, 0.1, None)

    aot_outer = frame["AOT_outer_mean"].fillna(0).to_numpy()
    rh_fraction = (frame["Humidity_final"] / 100.0).clip(0, 0.95).to_numpy()
    frame["aod_outer_pm25"] = aot_outer / (pblh_fill + 100.0) / (1.0 / (1.0 - rh_fraction))
    if not skip_station_rolling:
        frame["PBLH_min_24h"] = frame.groupby("stationId")["PBLH"].transform(
            lambda x: x.rolling(24, min_periods=1).min()
        )
        stagnant = ((frame["PBLH"] < 500) & (frame["WS_local"].fillna(0) < 2)).astype(float)
        frame["stagnation_hours_12h"] = (
            stagnant.groupby(frame["stationId"])
            .rolling(12, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    pblh_km = frame["PBLH"].fillna(200).clip(lower=50) / 1000.0
    rh_safe = frame["RH_factor"].fillna(1.0).clip(lower=1.0)
    frame["aod_surface"] = frame["AOT_inner_mean"].fillna(0) / (pblh_km + 0.1)
    frame["aod_dry"] = frame["AOT_inner_mean"].fillna(0) / rh_safe
    frame["co_surface"] = frame["co_30d_mean"].fillna(0) / (pblh_km + 0.1)
    frame["hcho_surface"] = frame["hcho_30d_mean"].fillna(0) / (pblh_km + 0.1)
    frame["no2_surface"] = frame["no2_daily_anom"].fillna(0) / (pblh_km + 0.1)
    frame["so2_surface"] = frame["so2_daily_anom"].fillna(0) / (pblh_km + 0.1)
    frame["combustion_aod"] = frame["co_30d_mean"].fillna(0) * frame["AOT_fine"].fillna(0)
    frame["secondary_form"] = frame["hcho_30d_mean"].fillna(0) * (
        frame["Humidity_final"].fillna(50) / 100
    )
    frame["modis_surface"] = frame["modis_aod_7d"].fillna(0) / (pblh_km + 0.1)
    frame["stagnant_aod"] = frame["AOT_inner_mean"].fillna(0) * frame[
        "stagnation_hours_12h"
    ].fillna(0)
    frame["stagnant_co"] = frame["co_30d_mean"].fillna(0) * frame[
        "stagnation_hours_12h"
    ].fillna(0)
    frame["aod_anomaly"] = frame["AOT_inner_mean"].fillna(0) - frame[
        "aod_30d_mean"
    ].fillna(0)
    return frame


def _norm01_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = np.nanquantile(numeric, [0.05, 0.95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((numeric - lo) / (hi - lo)).clip(0, 1).fillna(0)


def attach_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the optional observable regime controls after regional filtering."""
    stations = sorted(frame["stationId"].astype(str).unique())
    station_first = frame.groupby("stationId", sort=False).first()

    def matrix(prefix: str) -> np.ndarray:
        return np.array(
            [
                [station_first.at[sid, f"{prefix}_{direction}"] for direction in SECTOR_NAMES]
                for sid in stations
            ],
            dtype=float,
        )

    no2_sector = matrix("no2_clim")
    ntl_sector = matrix("ntl_clim")
    lst_sector = matrix("lst_anom_clim")
    no2_center = station_first.loc[stations, "no2_center"].to_numpy(dtype=float)
    ntl_center = station_first.loc[stations, "ntl_center"].to_numpy(dtype=float)
    lst_center = station_first.loc[stations, "lst_anom_center"].to_numpy(dtype=float)
    fmf_center = station_first.loc[stations, "fmf_center"].to_numpy(dtype=float)

    def limits(sectors: np.ndarray, centers: np.ndarray) -> tuple[float, float]:
        combined = np.concatenate([sectors.ravel(), centers])
        return float(np.nanmin(combined)), float(np.nanmax(combined))

    def norm(value: float, lo: float, hi: float) -> float:
        if not np.isfinite(value) or hi - lo < 1e-12:
            return 0.0
        return float((value - lo) / (hi - lo))

    no2_lo, no2_hi = limits(no2_sector, no2_center)
    ntl_lo, ntl_hi = limits(ntl_sector, ntl_center)
    lst_lo, lst_hi = limits(lst_sector, lst_center)
    # Emission-source intensity proxy: NO2 column scaled by night lights,
    # thermal anomaly and fine-mode fraction at the station.
    source_map: dict[str, float] = {}
    for index, sid in enumerate(stations):
        fmf = fmf_center[index] if np.isfinite(fmf_center[index]) else 0.5
        source_map[sid] = (
            norm(no2_center[index], no2_lo, no2_hi)
            * (1.0 + norm(ntl_center[index], ntl_lo, ntl_hi))
            * (1.0 + norm(lst_center[index], lst_lo, lst_hi))
            * fmf
        )
    frame["source_intensity_center"] = frame["stationId"].astype(str).map(source_map)

    building = np.log1p(pd.to_numeric(frame["building_area_1km"], errors="coerce").fillna(0))
    building = (building - building.quantile(0.05)) / max(
        building.quantile(0.95) - building.quantile(0.05), 1e-6
    )
    building = building.clip(0, 1).fillna(0)
    no2 = _norm01_series(frame["no2_center"])
    ntl = _norm01_series(frame["ntl_center"])
    source_intensity = _norm01_series(frame["source_intensity_center"])
    frame["urban_score"] = (0.45 * building + 0.35 * ntl + 0.20 * no2).clip(0, 1)
    frame["industrial_score"] = (0.55 * source_intensity + 0.30 * no2 + 0.15 * building).clip(0, 1)
    frame["peri_rural_score"] = (1.0 - frame["urban_score"]).clip(0, 1)
    frame["urban_aod"] = frame["urban_score"] * frame["AOT_inner_mean"].fillna(0)
    frame["urban_stagnation"] = frame["urban_score"] * frame[
        "stagnation_hours_12h"
    ].fillna(0)
    pblh_km = frame["PBLH"].fillna(200).clip(lower=50) / 1000.0
    frame["urban_pblh_inv"] = frame["urban_score"] / (pblh_km + 0.1)
    frame["industrial_vent"] = frame["industrial_score"] / frame["VC"].fillna(
        frame["VC"].median()
    ).clip(lower=0.1)
    return frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich unified_thesis.csv with GEE/MODIS features")
    parser.add_argument("--input", default="data/merged/unified_thesis.csv")
    parser.add_argument("--output", default=None, help="Defaults to replacing --input safely")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[3]
    input_path = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_path = (
        (root / args.output).resolve()
        if args.output and not Path(args.output).is_absolute()
        else Path(args.output).resolve()
        if args.output
        else input_path
    )
    print(f"Loading {input_path}...")
    frame = pd.read_csv(input_path, dtype={"stationId": str}, low_memory=False)
    frame = enrich_unified_dataframe(frame, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".enriching.tmp")
    print(f"Writing {temporary}...")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output_path)
    print(f"Saved enriched unified table: {output_path}")


if __name__ == "__main__":
    main()
