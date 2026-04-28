#!/usr/bin/env python
"""
Download ACAG V6.GL.02.04 monthly PM2.5 (2020-2023), extract nearest-pixel
values at 124 Vietnamese station locations, compute climatology, delete
the large NetCDF files.  Only the station CSV remains.

Usage:
    python scripts/download_acag_extract.py \
        --meta data/stations/metadata/station_building_density.csv
"""

import argparse, os, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUCKET = "v6.gl.02.04"
S3_BASE = f"https://s3.us-west-2.amazonaws.com/{BUCKET}"
YEARS = range(2020, 2024)

# S3 key templates — AS (Asia) region keeps files small
KEY_001 = "V6.GL.02.04/AS/Monthly/{year}/V6GL02.04.CNNPM25.AS.{ym}-{ym}.nc"
KEY_01 = "V6.GL.02.04-0p10/AS/Monthly/{year}/V6GL02.04.CNNPM25.0p10.AS.{ym}-{ym}.nc"

SANITY_KEYWORDS = ["ĐHBK", "Trà Vinh", "Thái Nguyên"]


# ── download helpers ────────────────────────────────────────────────

def _download_requests(url, dest):
    import requests

    try:
        r = requests.get(url, stream=True, timeout=300)
        if r.status_code != 200:
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        return True
    except Exception:
        return False


def _download_boto3(key, dest):
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
            region_name="us-west-2",
        )
        s3.download_file(BUCKET, key, dest)
        return True
    except Exception:
        return False


def download(key, dest):
    """Try direct HTTP GET, then unsigned boto3."""
    url = f"{S3_BASE}/{key}"
    if _download_requests(url, dest):
        return True
    return _download_boto3(key, dest)


# ── extraction ──────────────────────────────────────────────────────

def extract_at_stations(nc_path, lats, lons):
    """Open NetCDF, return nearest-pixel PM2.5 for each station."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    pm_var = None
    for v in ds.data_vars:
        if "pm" in v.lower():
            pm_var = v
            break
    if pm_var is None:
        pm_var = list(ds.data_vars)[0]

    da = ds[pm_var].squeeze()

    lat_dim = lon_dim = None
    for d in da.dims:
        dl = d.lower()
        if dl in ("lat", "latitude", "y"):
            lat_dim = d
        elif dl in ("lon", "longitude", "x"):
            lon_dim = d

    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Cannot identify lat/lon dims in {da.dims}")

    vals = np.full(len(lats), np.nan)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        v = da.sel({lat_dim: la, lon_dim: lo}, method="nearest").values
        vals[i] = float(np.squeeze(v))
    ds.close()
    return vals


# ── main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download ACAG PM2.5 and extract at station locations"
    )
    ap.add_argument("--meta", required=True, help="Station metadata CSV")
    args = ap.parse_args()

    meta_path = args.meta
    if not os.path.isabs(meta_path):
        meta_path = os.path.join(REPO_DIR, meta_path)
    meta = pd.read_csv(meta_path)
    lats = meta["latitude"].values
    lons = meta["longitude"].values
    n_stations = len(meta)
    print(f"Loaded {n_stations} stations")

    out_dir = os.path.join(REPO_DIR, "data", "acag")
    os.makedirs(out_dir, exist_ok=True)

    monthly = {}  # (year, month) → ndarray shape (n_stations,)
    chosen_res = None

    for label, key_tpl in [("0.01°", KEY_001), ("0.1°", KEY_01)]:
        print(f"\n{'='*50}")
        print(f"Trying {label} resolution (AS region) from AWS S3 …")
        print(f"{'='*50}")

        # Probe first file
        year0 = list(YEARS)[0]
        ym = f"{year0}01"
        key = key_tpl.format(year=year0, ym=ym)
        probe = os.path.join(out_dir, "_probe.nc")

        if not download(key, probe):
            print(f"  ✗ {label} not accessible, trying next resolution")
            if os.path.exists(probe):
                os.remove(probe)
            continue

        size_mb = os.path.getsize(probe) / (1 << 20)
        print(f"  File size: {size_mb:.1f} MB")

        try:
            vals = extract_at_stations(probe, lats, lons)
            monthly[(year0, 1)] = vals
            med = np.nanmedian(vals)
            print(f"  ✓ Probe OK  (median = {med:.1f} µg/m³)")
        except Exception as e:
            print(f"  ✗ Cannot read file: {e}")
            if os.path.exists(probe):
                os.remove(probe)
            continue
        finally:
            if os.path.exists(probe):
                os.remove(probe)

        chosen_res = label

        # Download remaining months
        for year in YEARS:
            for month in range(1, 13):
                if (year, month) in monthly:
                    continue
                ym = f"{year}{month:02d}"
                key = key_tpl.format(year=year, ym=ym)
                fpath = os.path.join(out_dir, f"_tmp_{ym}.nc")

                print(f"  {year}-{month:02d} … ", end="", flush=True)
                if not download(key, fpath):
                    print("FAILED")
                    continue
                try:
                    monthly[(year, month)] = extract_at_stations(fpath, lats, lons)
                    print("ok")
                except Exception as e:
                    print(f"read error: {e}")
                finally:
                    if os.path.exists(fpath):
                        os.remove(fpath)
        break

    # ── check we got data ───────────────────────────────────────────
    if not monthly:
        print("\nERROR: All AWS downloads failed.")
        print("Manual fallback — download 0.1° monthly AS files from:")
        print("  https://wustl.box.com/v/ACAG-V6GL0204-CNNPM25c0p10")
        print("Place them in data/acag/ and re-run with --local flag (not yet implemented).")
        sys.exit(1)

    n_ok = len(monthly)
    n_expected = len(list(YEARS)) * 12
    print(f"\nExtracted {n_ok}/{n_expected} monthly grids ({chosen_res})")
    if n_ok < n_expected:
        missing = [(y, m) for y in YEARS for m in range(1, 13) if (y, m) not in monthly]
        print(f"Missing months: {missing}")

    # ── compute climatology ─────────────────────────────────────────
    result = meta[["stationId", "stationName", "latitude", "longitude"]].copy()

    all_vals = np.column_stack(
        [monthly[k] for k in sorted(monthly.keys())]
    )  # (n_stations, n_months_ok)
    result["ACAG_annual_mean"] = np.nanmean(all_vals, axis=1)

    for m in range(1, 13):
        cols = [monthly[(y, m)] for y in YEARS if (y, m) in monthly]
        if cols:
            result[f"ACAG_monthly_clim_{m:02d}"] = np.nanmean(
                np.column_stack(cols), axis=1
            )
        else:
            result[f"ACAG_monthly_clim_{m:02d}"] = np.nan

    # ── save ────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "acag_station_climatology.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved → {out_path}")
    print(f"Shape:  {result.shape}")

    # ── summary stats ───────────────────────────────────────────────
    am = result["ACAG_annual_mean"]
    print(f"\nACAG annual mean across {n_stations} stations:")
    print(f"  min  = {am.min():.2f} µg/m³")
    print(f"  mean = {am.mean():.2f} µg/m³")
    print(f"  max  = {am.max():.2f} µg/m³")
    print(f"  std  = {am.std():.2f} µg/m³")

    # ── sanity check ────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Sanity check")
    print(f"{'='*50}")
    for kw in SANITY_KEYWORDS:
        mask = result["stationName"].str.contains(kw, case=False, na=False)
        rows = result[mask]
        if rows.empty:
            print(f"  {kw}: NOT FOUND in metadata")
            continue
        for _, row in rows.iterrows():
            name = row["stationName"]
            if len(name) > 65:
                name = name[:62] + "…"
            print(f"\n  {name}")
            print(f"    ACAG annual mean = {row['ACAG_annual_mean']:.2f} µg/m³")
            clim = [row[f"ACAG_monthly_clim_{m:02d}"] for m in range(1, 13)]
            print(f"    monthly clim range = {min(clim):.1f} – {max(clim):.1f} µg/m³")
            months_str = " ".join(f"{v:.1f}" for v in clim)
            print(f"    J F M A M J J A S O N D: {months_str}")

    print("\nDone.")


if __name__ == "__main__":
    main()
