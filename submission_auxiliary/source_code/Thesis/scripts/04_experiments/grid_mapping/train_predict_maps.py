# -*- coding: utf-8 -*-
"""Retrain the thesis delta_rfsi_wind model on 12 RRD KK stations
(same feature pipeline as exp_red_river_delta.py, LOSO skipped) and predict
PM2.5 on the 0.02-deg grid for the 4 map timestamps.

All non-fold-dependent features come from the enriched canonical table via
regional_feature_pipeline; this script only computes RFSI (anchor side and
grid-cell side) and runs the model.

Inputs : data/merged/unified_thesis.csv (KK rows, enriched),
         D:/map_data/grid/grid_unified_cols.csv
Outputs: D:/map_data/maps/grid_predictions.csv  (override with --out)
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4].as_posix()
MAP_DATA = os.environ.get("MAP_DATA", "D:/map_data")
sys.path.insert(0, os.path.join(ROOT, "Thesis", "scripts", "02_processing"))
sys.path.insert(0, os.path.join(ROOT, "Thesis", "scripts", "03_features"))
from pm25_qc import pm25_quality_masks
from regional_feature_pipeline import (
    DAILY_SAT,
    MET,
    MONO_DICT,
    OBS_DERIVED,
    PHYSICS_FEATS,
    PRECIP,
    REGIONAL_SOURCE_COLUMNS,
    SAT_AOD,
    SAT_REGIME,
    STABILITY,
    TEMPORAL,
    prepare_observation_features,
    read_unified_stations,
    require_enriched_unified,
)

parser = argparse.ArgumentParser()
parser.add_argument("--grid", default=MAP_DATA + "/grid/grid_unified_cols.csv")
parser.add_argument("--out", default=MAP_DATA + "/maps/grid_predictions.csv")
parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
args = parser.parse_args()

DELTA_SIDS = {
    "28560877461938780203765592307", "28916504310234840885489983032",
    "28916774462801800655608897080", "29196010501691076420299004774",
    "29196021237696127337075448678", "29203727697074312726675247132",
    "31388868531618872623864101418", "31388883344354363840031242796",
    "31390903576425084107499649578", "31390908889087377344742439468",
    "31390921469766835629621918251", "31390957404024291365397346858",
}
K_NN = 5
RFSI_FEATURES = ["PM25_nn_idw", "PM25_nn1", "PM25_nn2", "PM25_nn3"]
RFSI_WIND_FEATURES = RFSI_FEATURES + [
    "PM25_upwind_idw", "PM25_downwind_idw", "PM25_wind_spread", "PM25_neighbor_spread"]

XGB_BASE = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=50,
    reg_alpha=0.1, reg_lambda=10.0, tree_method="hist",
    random_state=42, n_jobs=-1,
)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

def bearing_degrees(lat1, lon1, lat2, lon2):
    y_b = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
    x_b = (np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) -
           np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) *
           np.cos(np.radians(lon2 - lon1)))
    return (np.degrees(np.arctan2(y_b, x_b)) + 360.0) % 360.0

# ============================================================== load KK rows
print("loading unified (12 delta stations, enriched columns only)...")
df = read_unified_stations(
    os.path.join(ROOT, "data/merged/unified_thesis.csv"),
    DELTA_SIDS, usecols=REGIONAL_SOURCE_COLUMNS,
)
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
print(f"  {len(df):,} rows, {df['stationId'].nunique()} stations")

meta = pd.read_csv(os.path.join(ROOT,
    "Thesis/results/01_stations/station_selection_final.csv"), dtype={"stationId": str})
sid_lat = dict(zip(meta["stationId"], meta["lat"]))
sid_lon = dict(zip(meta["stationId"], meta["lon"]))
station_ids = sorted(df["stationId"].unique())
n_stn = len(station_ids)

qc = pm25_quality_masks(df)
df.loc[qc.any(axis=1), "PM2.5"] = np.nan
print(f"  QC masked {int(qc.any(axis=1).sum())} rows")

y_all = df["PM2.5"].values
y_model = np.log1p(np.nan_to_num(y_all, nan=0.0))

# ============================================================== features
require_enriched_unified(df)
df = prepare_observation_features(df, label="KK model rows")

FEAT_OBS = [f for f in (SAT_AOD + DAILY_SAT + MET + PRECIP + TEMPORAL +
            STABILITY + SAT_REGIME + OBS_DERIVED + PHYSICS_FEATS) if f in df.columns]
FEAT_OBS_WIND_RFSI = FEAT_OBS + RFSI_WIND_FEATURES
print(f"FEAT_OBS = {len(FEAT_OBS)} features (+8 RFSI = {len(FEAT_OBS_WIND_RFSI)})")

obs_arr = df[FEAT_OBS].values.astype(np.float32)

# ============================================================== RFSI (training)
print("RFSI (train, all 12 anchors)...")
coords = {s: (sid_lat[s], sid_lon[s]) for s in station_ids}
sid_to_idx = {s: i for i, s in enumerate(station_ids)}
dist_full = np.zeros((n_stn, n_stn)); bearing_full = np.zeros((n_stn, n_stn))
for i in range(n_stn):
    for j in range(i + 1, n_stn):
        la1, lo1 = coords[station_ids[i]]; la2, lo2 = coords[station_ids[j]]
        d_ = haversine(la1, lo1, la2, lo2)
        dist_full[i, j] = d_; dist_full[j, i] = d_
        b_ij = bearing_degrees(la1, lo1, la2, lo2)
        bearing_full[i, j] = b_ij; bearing_full[j, i] = (b_ij + 180.0) % 360.0
neighbor_order = {i: sorted([(j, dist_full[i, j]) for j in range(n_stn) if j != i],
                            key=lambda x: x[1]) for i in range(n_stn)}
pm25_wide = df.pivot_table(index="ts", columns="stationId", values="PM2.5", aggfunc="first")
pm25_mat = pm25_wide.values
sid_cols = list(pm25_wide.columns)
sid_to_col = {s: i for i, s in enumerate(sid_cols)}
ts_to_row = pd.Series(range(len(pm25_wide)), index=pm25_wide.index)
df["ts_row"] = df["ts"].map(ts_to_row).astype(int).values
stationId_vals = df["stationId"].values

def compute_rfsi_full(K=5):
    n = len(df)
    pm_nn = np.full((n, K), np.nan); d_nn = np.full((n, K), np.nan)
    upwind_idw = np.full(n, np.nan); downwind_idw = np.full(n, np.nan)
    wind_spread = np.full(n, np.nan); neighbor_spread = np.full(n, np.nan)
    ts_row_vals = df["ts_row"].values
    wind_from = (np.degrees(np.arctan2(-df["wind_u"].values, -df["wind_v"].values)) + 360.0) % 360.0
    for sid in station_ids:
        si = sid_to_idx[sid]
        mask = stationId_vals == sid
        if not mask.any():
            continue
        ri = np.where(mask)[0]; tr = ts_row_vals[ri]
        cands = neighbor_order[si]
        ccols = np.array([sid_to_col[station_ids[j]] for j, _ in cands])
        cdists = np.array([d for _, d in cands])
        cbear = np.array([bearing_full[si, j] for j, _ in cands])
        nbr = pm25_mat[np.ix_(tr, ccols)]
        valid = ~np.isnan(nbr)
        cumv = np.cumsum(valid, axis=1)
        for k in range(K):
            reached = cumv >= (k + 1)
            has = reached.any(axis=1)
            if not has.any():
                break
            pos = np.argmax(reached, axis=1)
            ih = np.where(has)[0]
            pm_nn[ri[ih], k] = nbr[ih, pos[has]]
            d_nn[ri[ih], k] = cdists[pos[has]]
        row_from = wind_from[ri]
        diff_up = np.abs(((cbear[None, :] - row_from[:, None] + 180.0) % 360.0) - 180.0)
        diff_down = np.abs(((cbear[None, :] - ((row_from[:, None] + 180.0) % 360.0) + 180.0) % 360.0) - 180.0)
        align_up = np.clip(np.cos(np.radians(diff_up)), 0.0, 1.0)
        align_down = np.clip(np.cos(np.radians(diff_down)), 0.0, 1.0)
        dist_w = 1.0 / np.maximum(cdists, 0.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            wu = align_up * dist_w[None, :] * valid
            wd = align_down * dist_w[None, :] * valid
            upwind_idw[ri] = np.nansum(nbr * wu, axis=1) / np.nansum(wu, axis=1)
            downwind_idw[ri] = np.nansum(nbr * wd, axis=1) / np.nansum(wd, axis=1)
            wind_spread[ri] = upwind_idw[ri] - downwind_idw[ri]
            neighbor_spread[ri] = np.nanmax(nbr, axis=1) - np.nanmin(nbr, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d_nn
        pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)
    return np.column_stack([pm_idw, pm_nn[:, 0], pm_nn[:, 1], pm_nn[:, 2],
                            upwind_idw, downwind_idw, wind_spread, neighbor_spread])

rfsi_wind_arr = compute_rfsi_full(K=K_NN)

# ============================================================== train final
mono = str(tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_WIND_RFSI))
valid_train = ~np.isnan(y_all)
idx = np.where(valid_train)[0]
X_train = np.hstack([obs_arr, rfsi_wind_arr])[idx]
y_train = y_model[idx]

params = {**XGB_BASE, "monotone_constraints": mono}
if args.device == "cpu":
    m_final = xgb.XGBRegressor(**params)
    m_final.fit(X_train, y_train)
    print("trained on CPU")
else:
    try:
        m_final = xgb.XGBRegressor(**{**params, "device": "cuda"})
        m_final.fit(X_train, y_train)
        print("trained on CUDA")
    except Exception as e:
        if args.device == "cuda":
            raise
        print("cuda failed ->", str(e)[:120], "-> cpu")
        m_final = xgb.XGBRegressor(**params)
        m_final.fit(X_train, y_train)
print(f"final model trained: {len(idx):,} rows")

# sanity: training-fit R2 (not a validation metric, just a wiring check)
from sklearn.metrics import r2_score

pred_tr = m_final.predict(X_train[:100000])
print("  wiring check R2 (train subset, log):",
      round(r2_score(y_model[idx][:100000], pred_tr), 3))

# ============================================================== grid predict
print("\ngrid features...")
grid = pd.read_csv(args.grid, parse_dates=["ts"])
grid = prepare_observation_features(grid, label="grid cells", skip_station_rolling=True)

missing = [f for f in FEAT_OBS if f not in grid.columns]
if missing:
    print("MISSING grid cols:", missing); sys.exit(1)
Xg_obs = grid[FEAT_OBS].values.astype(np.float32)

# RFSI for grid cells (per timestamp, vectorized over cells)
print("grid RFSI...")
anchor_lat = np.array([coords[s][0] for s in station_ids])
anchor_lon = np.array([coords[s][1] for s in station_ids])

def grid_rfsi(lat, lon, ts, wind_u, wind_v, K=5):
    n = len(lat)
    dists = np.stack([haversine(lat, lon, anchor_lat[j], anchor_lon[j])
                      for j in range(n_stn)], axis=1)          # [N,12]
    bears = np.stack([bearing_degrees(lat, lon, anchor_lat[j], anchor_lon[j])
                      for j in range(n_stn)], axis=1)
    order = np.argsort(dists, axis=1)
    d_sorted = np.take_along_axis(dists, order, axis=1)
    b_sorted = np.take_along_axis(bears, order, axis=1)
    if ts not in pm25_wide.index:
        pm_t = np.full(n_stn, np.nan)
    else:
        pm_t = pm25_wide.loc[ts].reindex(station_ids).values
    pm_sorted = pm_t[order]                                     # [N,12]
    valid = np.isfinite(pm_sorted)
    cumv = np.cumsum(valid, axis=1)
    pm_nn = np.full((n, K), np.nan); d_nn = np.full((n, K), np.nan)
    for k in range(K):
        reached = cumv >= (k + 1)
        has = reached.any(axis=1)
        if not has.any():
            break
        pos = np.argmax(reached, axis=1)
        ih = np.where(has)[0]
        pm_nn[ih, k] = pm_sorted[ih, pos[has]]
        d_nn[ih, k] = d_sorted[ih, pos[has]]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d_nn
        pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)
    wind_from = (np.degrees(np.arctan2(-wind_u, -wind_v)) + 360.0) % 360.0
    diff_up = np.abs(((b_sorted - wind_from[:, None] + 180.0) % 360.0) - 180.0)
    diff_down = np.abs(((b_sorted - ((wind_from[:, None] + 180.0) % 360.0) + 180.0) % 360.0) - 180.0)
    align_up = np.clip(np.cos(np.radians(diff_up)), 0.0, 1.0)
    align_down = np.clip(np.cos(np.radians(diff_down)), 0.0, 1.0)
    dist_w = 1.0 / np.maximum(d_sorted, 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        wu = align_up * dist_w * valid
        wd = align_down * dist_w * valid
        upwind = np.nansum(pm_sorted * wu, axis=1) / np.nansum(wu, axis=1)
        downwind = np.nansum(pm_sorted * wd, axis=1) / np.nansum(wd, axis=1)
        wspread = upwind - downwind
        nspread = np.nanmax(np.where(valid, pm_sorted, np.nan), axis=1) - \
                  np.nanmin(np.where(valid, pm_sorted, np.nan), axis=1)
    return np.column_stack([pm_idw, pm_nn[:, 0], pm_nn[:, 1], pm_nn[:, 2],
                            upwind, downwind, wspread, nspread])

preds = []
for ts, g in grid.groupby("ts"):
    gi = g.index.values
    rfsi_g = grid_rfsi(g["lat"].values, g["lon"].values, ts,
                       g["wind_u"].values, g["wind_v"].values, K=K_NN)
    Xg = np.hstack([Xg_obs[gi], rfsi_g]).astype(np.float32)
    p = m_final.predict(Xg)
    pm = np.clip(np.expm1(p), 0, None)
    n_anchor = int(np.isfinite(pm25_wide.loc[ts].reindex(station_ids).values).sum()) \
        if ts in pm25_wide.index else 0
    preds.append(pd.DataFrame({
        "ts": ts, "lat": g["lat"].values, "lon": g["lon"].values,
        "pm25_pred": pm, "n_anchors": n_anchor,
        "rfsi_idw": rfsi_g[:, 0],
    }))
    print(f"  {ts}: anchors={n_anchor} pred mean={np.nanmean(pm):.1f} "
          f"min={np.nanmin(pm):.1f} max={np.nanmax(pm):.1f}")

out = pd.concat(preds, ignore_index=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
out.to_csv(args.out, index=False)
print(f"SAVED {args.out}", len(out))
