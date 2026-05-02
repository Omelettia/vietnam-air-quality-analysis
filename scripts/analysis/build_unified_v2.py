"""
Build unified_thesis_v2.csv — add RF and SSA spatial features from new L2 extraction.

Takes unified_thesis_v1.csv (40 stations, 116 cols) and merges in:
  - RF:  RF_mean, RF_std, RF_center, RF_inner_mean, RF_outer_mean, RF_valid_count
  - SSA: SSA_mean, SSA_std, SSA_center, SSA_inner_mean, SSA_outer_mean, SSA_valid_count
  - Gradients: {RF,SSA}_grad_ns, _grad_ew, _grad_mag, _local_vs_regional
  - Derived: AOT_fine = AOT_center * RF_center

Replaces v1's scalar SSA/RF columns with new L2 spatial center values.
"""

import io, sys, os, warnings, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

V1_PATH = "data/merged/unified_thesis_v1.csv"
L2_DIR  = "data/himawari/L2"
OUT_PATH = "data/merged/unified_thesis_v2.csv"

PIXEL_POSITIONS = [
    "m2m2", "m2m1", "m2_0", "m2p1", "m2p2",
    "m1m2", "m1m1", "m1_0", "m1p1", "m1p2",
    "_0m2", "_0m1", "_0_0", "_0p1", "_0p2",
    "p1m2", "p1m1", "p1_0", "p1p1", "p1p2",
    "p2m2", "p2m1", "p2_0", "p2p1", "p2p2",
]

SUMMARY_COLS = ["valid_count", "mean", "std", "center", "inner_mean", "outer_mean"]


def pixel_cols(band):
    return [f"{band}_{p}" for p in PIXEL_POSITIONS]


def classify_pixels(band):
    north, south, west, east = [], [], [], []
    for p in PIXEL_POSITIONS:
        col = f"{band}_{p}"
        row_idx = p[:2]
        col_idx = p[2:]
        if row_idx in ("m1", "m2"):
            north.append(col)
        if row_idx in ("p1", "p2"):
            south.append(col)
        if col_idx in ("m1", "m2"):
            west.append(col)
        if col_idx in ("p1", "p2"):
            east.append(col)
    return north, south, west, east


def compute_gradients(df, band):
    north, south, west, east = classify_pixels(band)
    present_n = [c for c in north if c in df.columns]
    present_s = [c for c in south if c in df.columns]
    present_w = [c for c in west if c in df.columns]
    present_e = [c for c in east if c in df.columns]

    if present_n and present_s:
        df[f"{band}_grad_ns"] = (df[present_s].mean(axis=1) - df[present_n].mean(axis=1)) / 4
    else:
        df[f"{band}_grad_ns"] = np.nan

    if present_e and present_w:
        df[f"{band}_grad_ew"] = (df[present_e].mean(axis=1) - df[present_w].mean(axis=1)) / 4
    else:
        df[f"{band}_grad_ew"] = np.nan

    df[f"{band}_grad_mag"] = np.sqrt(df[f"{band}_grad_ns"]**2 + df[f"{band}_grad_ew"]**2)

    inner_col = f"{band}_inner_mean"
    outer_col = f"{band}_outer_mean"
    if inner_col in df.columns and outer_col in df.columns:
        df[f"{band}_local_vs_regional"] = df[inner_col] - df[outer_col]
    else:
        df[f"{band}_local_vs_regional"] = np.nan

    return df


def process_station(l2_path, station_id):
    """Load L2 CSV, compute gradients at 10-min, aggregate to hourly."""
    need_pixel_cols = pixel_cols("RF") + pixel_cols("SSA")
    need_summary = (
        [f"RF_{s}" for s in SUMMARY_COLS] +
        [f"SSA_{s}" for s in SUMMARY_COLS] +
        ["RF_inner_count", "RF_outer_count", "SSA_inner_count", "SSA_outer_count"]
    )
    use_cols = ["timestamp", "stationId"] + need_pixel_cols + need_summary

    df = pd.read_csv(l2_path, usecols=lambda c: c in use_cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    compute_gradients(df, "RF")
    compute_gradients(df, "SSA")

    df["ts_hour"] = df["timestamp"].dt.floor("h")

    keep_cols = ["ts_hour"]
    for band in ("RF", "SSA"):
        keep_cols += [f"{band}_{s}" for s in SUMMARY_COLS]
        keep_cols += [f"{band}_grad_ns", f"{band}_grad_ew", f"{band}_grad_mag", f"{band}_local_vs_regional"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    hourly = df[keep_cols].groupby("ts_hour").mean().reset_index()
    hourly["stationId"] = station_id
    hourly.rename(columns={"ts_hour": "ts"}, inplace=True)
    hourly["ts"] = hourly["ts"].astype(str)
    return hourly


# ======================================================================

t0 = time.time()
print("=" * 70)
print("BUILD UNIFIED_THESIS_V2")
print("=" * 70)

print(f"\nLoading v1: {V1_PATH}")
v1 = pd.read_csv(V1_PATH)
print(f"  v1 shape: {v1.shape}")
v1_stations = v1["stationId"].astype(str).unique()
print(f"  Stations: {len(v1_stations)}")

l2_files = os.listdir(L2_DIR)
l2_index = {}
for f in l2_files:
    if not f.endswith(".csv"):
        continue
    peek = pd.read_csv(os.path.join(L2_DIR, f), usecols=["stationId"], nrows=1)
    sid = str(peek["stationId"].iloc[0])
    l2_index[sid] = os.path.join(L2_DIR, f)

matched = [sid for sid in v1_stations if sid in l2_index]
print(f"  L2 files matched: {len(matched)}/{len(v1_stations)}")

print(f"\nProcessing L2 files ...")
chunks = []
for i, sid in enumerate(matched):
    sdf = process_station(l2_index[sid], sid)
    chunks.append(sdf)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(matched)} done ...")

l2_hourly = pd.concat(chunks, ignore_index=True)
print(f"  L2 hourly table: {l2_hourly.shape}")

new_cols = [c for c in l2_hourly.columns if c not in ("ts", "stationId")]
print(f"  New columns ({len(new_cols)}): {new_cols}")

v1["stationId"] = v1["stationId"].astype(str)
l2_hourly["stationId"] = l2_hourly["stationId"].astype(str)

drop_old = [c for c in new_cols if c in v1.columns]
if drop_old:
    print(f"\n  Replacing existing columns in v1: {drop_old}")
    v1.drop(columns=drop_old, inplace=True)

print(f"\nMerging on (stationId, ts) ...")
v2 = v1.merge(l2_hourly, on=["stationId", "ts"], how="left")
print(f"  v2 shape: {v2.shape}")
assert len(v2) == len(v1), f"Row count changed: {len(v1)} -> {len(v2)}"

print(f"\nComputing AOT_fine = AOT * RF_center ...")
if "AOT" in v2.columns and "RF_center" in v2.columns:
    v2["AOT_fine"] = v2["AOT"] * v2["RF_center"]
    nn = v2["AOT_fine"].notna().sum()
    print(f"  AOT_fine non-NaN: {nn:,}/{len(v2):,} ({100*nn/len(v2):.1f}%)")

print(f"\n--- Coverage summary ---")
for c in new_cols + ["AOT_fine"]:
    if c in v2.columns:
        nn = v2[c].notna().sum()
        print(f"  {c:30s}: {nn:>8,}/{len(v2):,} ({100*nn/len(v2):5.1f}%)")

print(f"\nSaving: {OUT_PATH}")
v2.to_csv(OUT_PATH, index=False)
fsize = os.path.getsize(OUT_PATH) / 1e6
print(f"  Size: {fsize:.1f} MB")
print(f"  Final columns: {v2.shape[1]}")

elapsed = time.time() - t0
print(f"\nDone — {elapsed:.0f}s total")
