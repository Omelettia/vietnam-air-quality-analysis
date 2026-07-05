#!/usr/bin/env python
"""
Extract ACAG V6.GL.02.04 monthly PM2.5 (2020-2023) at station locations,
compute per-station annual mean and monthly climatology.

Usage (local zip/nc files — recommended, S3 blocks downloads):
    python scripts/download_acag_extract.py \
        --meta data/stations/metadata/station_building_density.csv \
        --local .

Usage (auto-download from AWS S3 — may return 403):
    python scripts/download_acag_extract.py \
        --meta data/stations/metadata/station_building_density.csv

Output: data/acag/acag_station_climatology.csv
"""

import argparse, glob, os, re, sys, io, tempfile, zipfile
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUCKET = "v6.gl.02.04"
S3_BASE = f"https://s3.us-west-2.amazonaws.com/{BUCKET}"
YEARS = range(2020, 2024)

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
            "s3", config=Config(signature_version=UNSIGNED),
            region_name="us-west-2",
        )
        s3.download_file(BUCKET, key, dest)
        return True
    except Exception:
        return False


def download(key, dest):
    url = f"{S3_BASE}/{key}"
    if _download_requests(url, dest):
        return True
    return _download_boto3(key, dest)


# ── extraction ──────────────────────────────────────────────────────

def extract_at_stations(nc_path, lats, lons):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    da = ds["PM25"].squeeze()
    vals = np.full(len(lats), np.nan)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        vals[i] = float(da.sel(lat=la, lon=lo, method="nearest").values)
    ds.close()
    return vals


# ── local file loading ──────────────────────────────────────────────

def _parse_ym(fname):
    m = re.search(r"(\d{4})(\d{2})-\d{6}\.nc", fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def load_local(local_dir, lats, lons):
    monthly = {}

    nc_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.nc"), recursive=True))
    zip_files = sorted(glob.glob(os.path.join(local_dir, "*.zip")))
    print(f"  Found {len(nc_files)} loose .nc, {len(zip_files)} zip archives")

    for fpath in nc_files:
        fname = os.path.basename(fpath)
        year, month = _parse_ym(fname)
        if year is None or year not in YEARS:
            continue
        print(f"  {year}-{month:02d} ({fname}) … ", end="", flush=True)
        try:
            monthly[(year, month)] = extract_at_stations(fpath, lats, lons)
            print("ok")
        except Exception as e:
            print(f"error: {e}")

    for zpath in zip_files:
        zname = os.path.basename(zpath)
        print(f"  Archive: {zname}")
        with zipfile.ZipFile(zpath, "r") as zf:
            for entry in sorted(zf.namelist()):
                if not entry.endswith(".nc"):
                    continue
                fname = os.path.basename(entry)
                year, month = _parse_ym(fname)
                if year is None or year not in YEARS:
                    continue
                if (year, month) in monthly:
                    continue
                print(f"    {year}-{month:02d} ({fname}) … ", end="", flush=True)
                tmp = os.path.join(tempfile.gettempdir(), fname)
                try:
                    with zf.open(entry) as src, open(tmp, "wb") as dst:
                        dst.write(src.read())
                    monthly[(year, month)] = extract_at_stations(tmp, lats, lons)
                    print("ok")
                except Exception as e:
                    print(f"error: {e}")
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)

    return monthly


# ── main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract ACAG PM2.5 at station locations"
    )
    ap.add_argument("--meta", required=True, help="Station metadata CSV")
    ap.add_argument(
        "--local", default=None,
        help="Directory with .nc files or .zip archives (skip AWS)",
    )
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

    monthly = {}
    chosen_res = None

    # ── local mode ──────────────────────────────────────────────────
    if args.local:
        local_dir = args.local
        if not os.path.isabs(local_dir):
            local_dir = os.path.join(REPO_DIR, local_dir)
        print(f"\nLoading local files from: {local_dir}")
        monthly = load_local(local_dir, lats, lons)
        chosen_res = "local"

    # ── AWS download mode ───────────────────────────────────────────
    else:
        for label, key_tpl in [("0.01°", KEY_001), ("0.1°", KEY_01)]:
            print(f"\n{'='*50}")
            print(f"Trying {label} resolution (AS region) from AWS S3 …")
            print(f"{'='*50}")

            year0 = list(YEARS)[0]
            ym = f"{year0}01"
            key = key_tpl.format(year=year0, ym=ym)
            probe = os.path.join(out_dir, "_probe.nc")

            if not download(key, probe):
                print(f"  {label} not accessible, trying next")
                if os.path.exists(probe):
                    os.remove(probe)
                continue

            try:
                vals = extract_at_stations(probe, lats, lons)
                monthly[(year0, 1)] = vals
                print(f"  Probe OK (median = {np.nanmedian(vals):.1f} µg/m³)")
            except Exception as e:
                print(f"  Cannot read: {e}")
                if os.path.exists(probe):
                    os.remove(probe)
                continue
            finally:
                if os.path.exists(probe):
                    os.remove(probe)

            chosen_res = label
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
                        monthly[(year, month)] = extract_at_stations(
                            fpath, lats, lons
                        )
                        print("ok")
                    except Exception as e:
                        print(f"read error: {e}")
                    finally:
                        if os.path.exists(fpath):
                            os.remove(fpath)
            break

    # ── check ───────────────────────────────────────────────────────
    if not monthly:
        print("\nERROR: No data extracted.")
        print("Download 0.1° monthly AS .nc files (2020-2023) and use --local:")
        print("  python scripts/download_acag_extract.py \\")
        print("    --meta <csv> --local /path/to/zip/or/nc/")
        sys.exit(1)

    n_ok = len(monthly)
    n_expected = len(list(YEARS)) * 12
    print(f"\nExtracted {n_ok}/{n_expected} monthly grids ({chosen_res})")
    if n_ok < n_expected:
        missing = [
            (y, m) for y in YEARS for m in range(1, 13) if (y, m) not in monthly
        ]
        print(f"Missing: {missing}")

    # ── climatology ─────────────────────────────────────────────────
    result = meta[["stationId", "stationName", "latitude", "longitude"]].copy()

    all_vals = np.column_stack(
        [monthly[k] for k in sorted(monthly.keys())]
    )
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

    # ── summary ─────────────────────────────────────────────────────
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
            print(
                f"    monthly clim range = {min(clim):.1f} – {max(clim):.1f} µg/m³"
            )
            months_str = " ".join(f"{v:.1f}" for v in clim)
            print(f"    J F M A M J J A S O N D: {months_str}")

    print("\nDone.")


if __name__ == "__main__":
    main()
