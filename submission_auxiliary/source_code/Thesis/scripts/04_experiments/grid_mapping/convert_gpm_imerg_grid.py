# -*- coding: utf-8 -*-
"""Convert raw half-hourly IMERG GIS zips to grid-builder rain GeoTIFFs.

Outputs:
  D:/map_data/gpm/gpm_rain_dec.tif
  D:/map_data/gpm/gpm_rain_jul.tif

Each output contains daily bands named rain_YYYYMMDD and hourly bands named
rainh_YYYYMMDD_HH, matching defense grid builder expectations.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import re
import zipfile

import numpy as np
import rasterio
from rasterio.io import MemoryFile

MAP_DATA = os.environ.get("MAP_DATA", "D:/map_data")


RAW = MAP_DATA + "/gpm_raw"
OUT_DIR = MAP_DATA + "/gpm"
BOX = (105.3, 20.1, 107.2, 21.5)  # lon_min, lat_min, lon_max, lat_max
N_HOURS = 46 * 24  # full window: winter hrs_since_rain reaches ~450h, matching training scale

WINDOWS = {
    "dec": ("2025-12-09", 46),
    "jul": ("2025-07-30", 47),
}

FILL_THRESHOLD = 20000
_GRID: dict[str, object] = {}


def choose_precip_layer(names: list[str]) -> str:
    """Select total precipitation accumulation from either IMERG GIS format."""
    tif_names = [n for n in names if n.lower().endswith(".tif")]
    preferred = [
        lambda n: n.lower().endswith(".total.accum.tif"),
        lambda n: n.lower().endswith(".30min.tif"),
        lambda n: n.lower().endswith(".total.rate.tif"),
    ]
    for pred in preferred:
        hits = [n for n in tif_names if pred(n)]
        if hits:
            return sorted(hits, key=len)[0]
    raise ValueError("No precipitation TIFF layer found in zip")


def scale_from_tags(tags: dict[str, str]) -> float:
    """Return multiplier from stored DN to millimeters per 30-minute slot."""
    desc = tags.get("TIFFTAG_IMAGEDESCRIPTION", "")
    scale_factor = 1.0
    m = re.search(r"ScaleFactor=([0-9.]+)", desc)
    if m:
        scale_factor = float(m.group(1))

    scale = 1.0 / scale_factor
    if "mm/hr" in desc:
        scale *= 0.5
    return scale


def read_slot(day: dt.date, hour: int, minute: int) -> np.ndarray | None:
    folder = f"{RAW}/2025/{day.month:02d}/{day.day:02d}"
    pattern = f"{folder}/*3IMERG.{day:%Y%m%d}-S{hour:02d}{minute:02d}00-*.zip"
    matches = glob.glob(pattern)
    if not matches:
        return None

    try:
        with zipfile.ZipFile(matches[0]) as zf:
            layer = choose_precip_layer(zf.namelist())
            data = zf.read(layer)
        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                if "window" not in _GRID:
                    window = rasterio.windows.from_bounds(*BOX, transform=src.transform)
                    window = window.round_offsets().round_lengths()
                    _GRID["window"] = window
                    _GRID["transform"] = src.window_transform(window)
                    _GRID["nodata"] = src.nodata

                arr = src.read(1, window=_GRID["window"]).astype(np.float32)
                nodata = _GRID["nodata"]
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                arr[arr > FILL_THRESHOLD] = np.nan
                return arr * scale_from_tags(src.tags())
    except Exception as exc:
        print(f"  bad file {matches[0]}: {str(exc)[:120]}")
        return None


def hourly(day: dt.date, hour: int) -> np.ndarray | None:
    parts = [read_slot(day, hour, 0), read_slot(day, hour, 30)]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return np.nansum(np.stack(parts), axis=0)


def first_valid(arrays: list[np.ndarray | None]) -> np.ndarray:
    for arr in arrays:
        if arr is not None:
            return arr
    raise RuntimeError("No valid IMERG slot found; cannot infer output shape")


def convert_window(tag: str, end_s: str, n_days: int) -> None:
    end = dt.date.fromisoformat(end_s)
    days = [end - dt.timedelta(days=i) for i in range(n_days - 1, -1, -1)]
    bands: list[np.ndarray] = []
    names: list[str] = []
    hourly_cache: dict[dt.date, list[np.ndarray | None]] = {}

    print(f"== {tag}: {days[0]} .. {days[-1]}")
    for day in days:
        hourly_arrays = [hourly(day, h) for h in range(24)]
        hourly_cache[day] = hourly_arrays
        valid_hours = [h for h in hourly_arrays if h is not None]
        if valid_hours:
            daily_sum = np.nansum(np.stack(valid_hours), axis=0)
        else:
            daily_sum = np.full_like(first_valid(sum(hourly_cache.values(), [])), np.nan)
        bands.append(daily_sum)
        names.append(f"rain_{day:%Y%m%d}")

    end_hour = dt.datetime.combine(end + dt.timedelta(days=1), dt.time())
    template = bands[0]
    for h in range(N_HOURS, 0, -1):
        ts = end_hour - dt.timedelta(hours=h)
        arr = hourly_cache.get(ts.date(), [None] * 24)[ts.hour]
        if arr is None:
            arr = np.full_like(template, np.nan)
        bands.append(arr)
        names.append(f"rainh_{ts:%Y%m%d_%H}")

    stack = np.stack(bands)
    height, width = stack.shape[1:]
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(bands),
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": _GRID["transform"],
        "nodata": np.nan,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/gpm_rain_{tag}.tif"
    with rasterio.open(out, "w", **profile) as dst:
        for i, (band, name) in enumerate(zip(stack, names), 1):
            dst.write(band, i)
            dst.set_band_description(i, name)

    non_empty = int(sum(np.isfinite(band).any() for band in stack))
    print(f"  saved {out}: {len(bands)} bands ({non_empty} non-empty), {width}x{height}")


def main() -> None:
    for tag, (end_s, n_days) in WINDOWS.items():
        convert_window(tag, end_s, n_days)
    print("DONE")


if __name__ == "__main__":
    main()
