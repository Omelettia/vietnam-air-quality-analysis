"""Shared feature pipeline for the diverse-streams model — SINGLE SOURCE OF TRUTH.

Used by both exp_diverse_streams.py (trainer) and validate_diverse_knn_lcs.py (external
validator) so their feature sets can never drift again. (They drifted before: the
validator used a simplified ground-gas STREAMS while the trainer used satellite gas,
which made the diverse model look like it didn't transfer to LCS — an artifact.)

Key design choices:
- Uses the *_123 directional/static files, which cover all 123 stations (40 thesis +
  57 LCS), keyed by the long stationId. No fuzzy id-matching (the legacy 40-station
  directional files needed it; the _123 files don't).
- Pure function: build_diverse_features(df, meta, station_ids, ...) returns the enriched
  df, the STREAMS dict, and the two RFSI closures. No module-level side effects.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def _dir123(meta_dir, data_dir, stem):
    """Resolve a _123 directional file, falling back to the bare name / data_dir."""
    for cand in (os.path.join(meta_dir, f"{stem}_123.csv"),
                 os.path.join(meta_dir, f"{stem}.csv"),
                 os.path.join(data_dir, f"{stem}_123.csv"),
                 os.path.join(data_dir, f"{stem}.csv")):
        if os.path.exists(cand):
            return cand
    return os.path.join(meta_dir, f"{stem}_123.csv")


def build_diverse_features(df, meta, station_ids, *, data_dir, meta_dir, k_nn=5):
    """Enrich df with the full diverse-streams feature set.

    df must already have: stationId, ts, month, date, PM2.5 (QC-masked), and the base
    v4 columns (PBLH, VC, wind_u/v, WS_local, Temperature_final, ... AOT_*, etc.).
    meta must have stationId, station_name, lat, lon for every station in station_ids.
    Returns (df, STREAMS, compute_rfsi, compute_lagged_rfsi).
    """
    sid_lat = dict(zip(meta["stationId"], meta["lat"]))
    sid_lon = dict(zip(meta["stationId"], meta["lon"]))
    stationId_vals = df["stationId"].values
    n_stn = len(station_ids)
    K_NN = k_nn

    # === 2. SATELLITE FEATURES (TROPOMI + MODIS LST, daily anomalies) ===========
    import glob, zipfile
    zip_path = sorted(glob.glob(os.path.join(data_dir, "data", "gee_exports", "last-*.zip")))[-1]
    all_sat = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            if name.endswith(".csv"):
                with z.open(name) as f:
                    all_sat.append(pd.read_csv(f, dtype={"stationId": str}))
    sat_long = pd.concat(all_sat, ignore_index=True)
    sat_wide = sat_long.pivot_table(index=["stationId", "date"], columns="variable",
                                    values="mean", aggfunc="first").reset_index()
    sat_wide.columns.name = None
    sat_wide["date"] = pd.to_datetime(sat_wide["date"])
    sat_wide = sat_wide[sat_wide["stationId"].isin(set(station_ids))].copy()
    sat_wide["month"] = sat_wide["date"].dt.month
    CLIM_COLS = ["NO2", "SO2", "CO", "HCHO", "LST_terra_day", "LST_terra_night"]
    for c in CLIM_COLS:
        if c not in sat_wide.columns:
            sat_wide[c] = np.nan
    clim = sat_wide.groupby(["stationId", "month"])[CLIM_COLS].transform("mean")
    sat_wide["so2_daily_anom"] = sat_wide["SO2"] - clim["SO2"]
    sat_wide["co_daily_anom"] = sat_wide["CO"] - clim["CO"]
    sat_wide["no2_daily_anom"] = sat_wide["NO2"] - clim["NO2"]
    sat_wide["lst_day_anom"] = sat_wide["LST_terra_day"] - clim["LST_terra_day"]
    sat_wide["lst_night_anom"] = sat_wide["LST_terra_night"] - clim["LST_terra_night"]
    ANOM_RAW = ["so2_daily_anom", "co_daily_anom", "no2_daily_anom",
                "lst_day_anom", "lst_night_anom"]
    sat_wide["date_merge"] = sat_wide["date"].dt.date
    df = df.merge(sat_wide[["stationId", "date_merge"] + ANOM_RAW],
                  left_on=["stationId", "date"], right_on=["stationId", "date_merge"], how="left")
    df.drop(columns=["date_merge"], inplace=True)

    # === 3. BUILDING DENSITY (123-station file) =================================
    bld = pd.read_csv(os.path.join(meta_dir, "station_building_density.csv"), dtype={"stationId": str})
    BUILDING_COLS = ["building_count_1km", "building_area_1km", "building_count_3km", "building_area_3km"]
    bld_map = bld.set_index("stationId")[BUILDING_COLS]
    df = df.merge(bld_map, left_on="stationId", right_index=True, how="left")
    for col in BUILDING_COLS:
        df[col] = df[col].fillna(0)

    # === 4. MATCHED AOD FEATURES (Himawari, 123-station) ========================
    matched_static = pd.read_csv(os.path.join(meta_dir, "aod_source_matched_static.csv"), dtype={"stationId": str})
    matched_static_map = matched_static.set_index("stationId")
    matched_temporal = pd.read_csv(os.path.join(meta_dir, "aod_source_matched_temporal.csv"),
                                   dtype={"stationId": str}, parse_dates=["date"])
    matched_temporal["date_merge"] = matched_temporal["date"].dt.date
    prefix = "him"
    temporal_rename = {f"{prefix}_aod_7d": "src_aod_7d", f"{prefix}_fmf_7d": "src_fmf_7d",
                       f"{prefix}_fine_aod_7d": "src_fine_aod_7d"}
    keep_temporal = ["stationId", "date_merge"] + [c for c in temporal_rename if c in matched_temporal.columns]
    matched_temporal = matched_temporal[keep_temporal].rename(columns=temporal_rename)
    df = df.merge(matched_temporal, left_on=["stationId", "date"],
                  right_on=["stationId", "date_merge"], how="left")
    df.drop(columns=["date_merge"], inplace=True)
    static_rename = {}
    for suffix in (["aod_center", "aod_DJF", "aod_MAM", "aod_JJA", "aod_SON", "aod_contrast",
                    "aod_directionality", "aod_max_nearby", "fmf_center", "fine_aod_center", "ae_center"] +
                   [f"aod_clim_{d}" for d in SECTOR_NAMES] + [f"fine_aod_clim_{d}" for d in SECTOR_NAMES]):
        col = f"{prefix}_{suffix}"
        if col in matched_static_map.columns:
            static_rename[col] = f"src_{suffix}"
    if static_rename:
        df = df.merge(matched_static_map[list(static_rename)].rename(columns=static_rename),
                      left_on="stationId", right_index=True, how="left")
    SOURCE_AOD_FEATURES = (["src_aod_7d", "src_fmf_7d", "src_fine_aod_7d", "src_aod_center",
                            "src_fmf_center", "src_fine_aod_center", "src_ae_center", "src_aod_DJF",
                            "src_aod_MAM", "src_aod_JJA", "src_aod_SON", "src_aod_contrast",
                            "src_aod_directionality", "src_aod_max_nearby"] +
                           [f"src_aod_clim_{d}" for d in SECTOR_NAMES] +
                           [f"src_fine_aod_clim_{d}" for d in SECTOR_NAMES])
    SOURCE_AOD_FEATURES = [f for f in SOURCE_AOD_FEATURES if f in df.columns]

    # === 5. SATELLITE STATIC FEATURES (NO2, emission, NTL, LST; 123-station) ====
    no2_map = pd.read_csv(os.path.join(meta_dir, "station_no2_features.csv"), dtype={"stationId": str}).set_index("stationId")
    emit_map = pd.read_csv(os.path.join(meta_dir, "station_emission_features.csv"), dtype={"stationId": str}).set_index("stationId")
    new_sat_map = pd.read_csv(os.path.join(meta_dir, "station_all_satellite_features.csv"), dtype={"stationId": str}).set_index("stationId")
    NO2_STATIC_COLS = ["no2_center", "no2_contrast", "no2_directionality"]
    NO2_SECTOR_COLS = [f"no2_clim_{d}" for d in SECTOR_NAMES]
    df = df.merge(no2_map[NO2_STATIC_COLS + NO2_SECTOR_COLS], left_on="stationId", right_index=True, how="left")
    NTL_SECTOR_COLS = [f"ntl_clim_{d}" for d in SECTOR_NAMES]
    LST_SECTOR_COLS = [f"lst_anom_clim_{d}" for d in SECTOR_NAMES]
    merge_emit = [c for c in (["ntl_center"] + NTL_SECTOR_COLS + ["lst_anom_center"] + LST_SECTOR_COLS) if c in emit_map.columns]
    df = df.merge(emit_map[merge_emit], left_on="stationId", right_index=True, how="left")
    merge_new = [c for c in (["faod_center", "fmf_center", "ae_center", "so2_center"] +
                             [f"faod_clim_{d}" for d in SECTOR_NAMES] + [f"so2_clim_{d}" for d in SECTOR_NAMES])
                 if c in new_sat_map.columns]
    df = df.merge(new_sat_map[merge_new], left_on="stationId", right_index=True, how="left")

    # === 6. DIRECTIONAL CLIMATOLOGY (_123, long ids, NO id_map) + SMART_V1 ======
    def _load_dir_clim(csv_path, value_col="mean"):
        raw = pd.read_csv(csv_path, dtype={"stationId": str})
        clim_df = raw.groupby(["stationId", "direction"])[value_col].mean().reset_index()
        sectors, centers = {}, {}
        for sid in station_ids:
            sf = clim_df[clim_df["stationId"] == sid]
            sec = np.full(8, np.nan)
            for di, d in enumerate(SECTOR_NAMES):
                vals = sf[sf["direction"] == d][value_col]
                if len(vals) > 0:
                    sec[di] = float(vals.iloc[0])
            sectors[sid] = sec
            cvals = sf[sf["direction"] == "C"][value_col]
            centers[sid] = float(cvals.iloc[0]) if len(cvals) > 0 else np.nan
        return sectors, centers

    station_so2_sectors, station_so2_centers = _load_dir_clim(_dir123(meta_dir, data_dir, "tropomi_so2_directional"), "mean")
    station_co_sectors, station_co_centers = _load_dir_clim(_dir123(meta_dir, data_dir, "tropomi_co_directional"), "mean")
    station_hcho_sectors, station_hcho_centers = _load_dir_clim(_dir123(meta_dir, data_dir, "tropomi_hcho_directional"), "mean")
    station_lstd_sectors, station_lstd_centers = _load_dir_clim(_dir123(meta_dir, data_dir, "lst_anomaly_directional"), "lst_anomaly")

    wd_from = np.degrees(np.arctan2(-df["wind_u"].values, -df["wind_v"].values)) % 360
    sector_idx = ((wd_from + 22.5) / 45).astype(int) % 8
    ws = np.sqrt(df["wind_u"].values ** 2 + df["wind_v"].values ** 2)
    vc_inv = 1.0 / (df["PBLH"].clip(lower=50).values * ws.clip(min=0.1) + 1)

    all_no2_sec = np.array([no2_map.loc[s, [f"no2_clim_{d}" for d in SECTOR_NAMES]].values
                            if s in no2_map.index else np.full(8, np.nan) for s in station_ids])
    all_ntl_sec = np.array([emit_map.loc[s, [f"ntl_clim_{d}" for d in SECTOR_NAMES]].values
                            if s in emit_map.index else np.full(8, np.nan) for s in station_ids])
    all_lst_sec = np.array([emit_map.loc[s, [f"lst_anom_clim_{d}" for d in SECTOR_NAMES]].values
                            if s in emit_map.index else np.full(8, np.nan) for s in station_ids])
    no2_center_all = np.array([no2_map.loc[s, "no2_center"] if s in no2_map.index else np.nan for s in station_ids])
    ntl_center_all = np.array([emit_map.loc[s, "ntl_center"] if s in emit_map.index else np.nan for s in station_ids])
    lst_center_all = np.array([emit_map.loc[s, "lst_anom_center"] if s in emit_map.index else np.nan for s in station_ids])
    fmf_col = f"{prefix}_fmf_center"
    fmf_center_all = np.array([matched_static_map.loc[s, fmf_col]
                               if (s in matched_static_map.index and fmf_col in matched_static_map.columns)
                               else np.nan for s in station_ids])

    def _lohi(sec_arr, cen_arr):
        combined = np.concatenate([sec_arr.ravel(), cen_arr])
        return float(np.nanmin(combined)), float(np.nanmax(combined))

    no2_lo, no2_hi = _lohi(all_no2_sec, no2_center_all)
    ntl_lo, ntl_hi = _lohi(all_ntl_sec, ntl_center_all)
    lst_lo, lst_hi = _lohi(all_lst_sec, lst_center_all)

    def norm01(v, lo, hi):
        if hi - lo < 1e-12:
            return 0.0
        return float((v - lo) / (hi - lo)) if not np.isnan(v) else 0.0

    station_smart_v1_sec, station_smart_v1_cen = {}, {}
    for si, sid in enumerate(station_ids):
        fmf = fmf_center_all[si]
        if np.isnan(fmf):
            fmf = 0.5
        v1_sec = np.zeros(8)
        for di in range(8):
            no2_n = norm01(all_no2_sec[si, di], no2_lo, no2_hi)
            ntl_n = norm01(all_ntl_sec[si, di], ntl_lo, ntl_hi)
            lst_n = norm01(all_lst_sec[si, di], lst_lo, lst_hi)
            v1_sec[di] = no2_n * (1.0 + ntl_n) * (1.0 + lst_n) * fmf
        station_smart_v1_sec[sid] = v1_sec
        no2_cn = norm01(no2_center_all[si], no2_lo, no2_hi)
        ntl_cn = norm01(ntl_center_all[si], ntl_lo, ntl_hi)
        lst_cn = norm01(lst_center_all[si], lst_lo, lst_hi)
        station_smart_v1_cen[sid] = no2_cn * (1.0 + ntl_cn) * (1.0 + lst_cn) * fmf

    df["smart_v1_center"] = np.array([station_smart_v1_cen[s] for s in stationId_vals])
    smart_v1_upwind = np.zeros(len(df))
    for sid in station_ids:
        mask = stationId_vals == sid
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        smart_v1_upwind[idx] = station_smart_v1_sec[sid][sector_idx[idx]]
    df["smart_v1_upwind"] = smart_v1_upwind
    df["smart_v1_max"] = np.array([station_smart_v1_sec[s].max() for s in stationId_vals])
    s1_min = np.array([station_smart_v1_sec[s].min() for s in stationId_vals])
    df["smart_v1_contrast"] = df["smart_v1_max"].values / (s1_min + 0.001)
    df["smart_v1_upwind_x_VC_inv"] = smart_v1_upwind * vc_inv

    so2_upwind_vals = np.zeros(len(df)); co_upwind_vals = np.zeros(len(df)); lst_anom_upwind_vals = np.zeros(len(df))
    for sid in station_ids:
        mask = stationId_vals == sid
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        so2_upwind_vals[idx] = station_so2_sectors[sid][sector_idx[idx]]
        co_upwind_vals[idx] = station_co_sectors[sid][sector_idx[idx]]
        lst_anom_upwind_vals[idx] = station_lstd_sectors[sid][sector_idx[idx]]
    so2_upwind_vals = np.nan_to_num(so2_upwind_vals); co_upwind_vals = np.nan_to_num(co_upwind_vals)
    lst_anom_upwind_vals = np.nan_to_num(lst_anom_upwind_vals)
    df["so2_upwind"] = so2_upwind_vals
    df["co_upwind"] = co_upwind_vals
    df["so2_center"] = df["stationId"].map({sid: float(np.nan_to_num(station_so2_centers.get(sid, 0.0), nan=0.0)) for sid in station_ids}).fillna(0.0)
    df["co_center"] = df["stationId"].map({sid: float(np.nan_to_num(station_co_centers.get(sid, 0.0), nan=0.0)) for sid in station_ids}).fillna(0.0)
    so2_contrast_map = {}
    for sid in station_ids:
        sec = station_so2_sectors[sid]
        valid = sec[~np.isnan(sec)]
        so2_contrast_map[sid] = float(valid.max() / valid.mean()) if len(valid) > 0 and valid.mean() > 1e-12 else 1.0
    df["so2_contrast"] = df["stationId"].map(so2_contrast_map).fillna(1.0)
    df["hcho_center"] = df["stationId"].map({sid: float(np.nan_to_num(station_hcho_centers.get(sid, 0.0), nan=0.0)) for sid in station_ids}).fillna(0.0)
    df["so2_upwind_x_VC_inv"] = so2_upwind_vals * vc_inv
    df["lst_anom_upwind_x_VC_inv"] = lst_anom_upwind_vals * vc_inv
    df["so2_anom_x_vc_inv"] = df["so2_daily_anom"].values * vc_inv
    df["co_anom_x_vc_inv"] = df["co_daily_anom"].values * vc_inv
    df["lst_anom_x_vc_inv"] = df["lst_day_anom"].values * vc_inv
    ANOM_INTERACT = ["so2_anom_x_vc_inv", "co_anom_x_vc_inv", "lst_anom_x_vc_inv"]
    DAILY_ANOM_ALL = ANOM_RAW + ANOM_INTERACT

    # Fire upwind (_123)
    fire_raw = pd.read_csv(_dir123(meta_dir, data_dir, "fire_counts_directional"), dtype={"stationId": str}).rename(columns={"mean": "fire_val"})
    fire_raw = fire_raw.dropna(subset=["stationId"])
    station_fire_dir_season = {}
    for sid in station_ids:
        sf = fire_raw[fire_raw["stationId"] == sid]
        lookup = {}
        for di, d in enumerate(SECTOR_NAMES):
            for szn in ["DJF", "MAM", "JJA", "SON"]:
                vals = sf[(sf["direction"] == d) & (sf["season"] == szn)]["fire_val"]
                lookup[(di, szn)] = float(vals.mean()) if len(vals) > 0 else 0.0
        for szn in ["DJF", "MAM", "JJA", "SON"]:
            vals = sf[(sf["direction"] == "C") & (sf["season"] == szn)]["fire_val"]
            lookup[(-1, szn)] = float(vals.mean()) if len(vals) > 0 else 0.0
        station_fire_dir_season[sid] = lookup
    season_vals = np.array([SEASON_MAP[m] for m in df["month"].values])
    fire_upwind = np.zeros(len(df))
    for sid in station_ids:
        mask = stationId_vals == sid
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        lookup = station_fire_dir_season.get(sid, {})
        for i in idx:
            fire_upwind[i] = lookup.get((sector_idx[i], season_vals[i]), 0.0)
    df["fire_upwind"] = fire_upwind

    fill_cols = (NO2_STATIC_COLS + NO2_SECTOR_COLS +
                 ["ntl_center", "smart_v1_center", "smart_v1_upwind", "smart_v1_max",
                  "smart_v1_contrast", "smart_v1_upwind_x_VC_inv", "so2_upwind", "so2_center",
                  "so2_contrast", "co_upwind", "co_center", "hcho_center",
                  "so2_upwind_x_VC_inv", "lst_anom_upwind_x_VC_inv", "fire_upwind"])
    for c in set(fill_cols):
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # === 7. OUTER AOD PHYSICS ===================================================
    aot_outer = df["AOT_outer_mean"].fillna(0).values
    aot_center = df["AOT_ffill_48h"].fillna(0).values
    pblh = df["PBLH"].fillna(200).values
    rh_frac = (df["Humidity_final"] / 100.0).clip(0, 0.95).values
    f_rh = 1.0 / (1.0 - rh_frac)
    hours_since = df["hours_since_valid_AOT"].fillna(999).values
    df["aod_outer_surface"] = aot_outer / (pblh + 100.0)
    df["aod_outer_pm25"] = aot_outer / (pblh + 100.0) / f_rh
    df["aod_outer_x_VC_inv"] = aot_outer * vc_inv
    df["aod_outer_gradient"] = aot_outer - aot_center
    is_real_aod = (hours_since == 0) & (aot_center > 0)
    df["_outer_real"] = np.where(is_real_aod, aot_outer, np.nan)
    df["_is_real_aod"] = is_real_aod
    day_outer = df.groupby(["stationId", "date"]).agg(
        aod_outer_day_mean=("_outer_real", "mean"), _day_count=("_is_real_aod", "sum")).reset_index()
    day_outer.loc[day_outer["_day_count"] == 0, "aod_outer_day_mean"] = np.nan
    df = df.merge(day_outer[["stationId", "date", "aod_outer_day_mean"]], on=["stationId", "date"], how="left")
    df.drop(columns=["_outer_real", "_is_real_aod"], inplace=True)
    OUTER_ALL_EXTRA = ["aod_outer_surface", "aod_outer_pm25", "aod_outer_x_VC_inv",
                       "aod_outer_gradient", "aod_outer_day_mean"]

    # === 8. WEATHER PERSISTENCE + TEMPORAL EXTRAS ==============================
    df["dow_is_weekend"] = (df["ts"].dt.dayofweek >= 5).astype(float)
    df["PBLH_min_24h"] = df.groupby("stationId")["PBLH"].transform(lambda x: x.rolling(24, min_periods=1).min())
    df["VC_min_24h"] = df.groupby("stationId")["VC"].transform(lambda x: x.rolling(24, min_periods=1).min())
    stag_col = ((df["PBLH"] < 500) & (df["WS_local"].fillna(0) < 2)).astype(float)
    df["stagnation_hours_12h"] = stag_col.groupby(df["stationId"]).rolling(12, min_periods=1).sum().reset_index(level=0, drop=True)
    df["temp_diurnal_anomaly"] = df["Temperature_final"] - df.groupby(
        ["stationId", df["ts"].dt.month, df["ts"].dt.hour])["Temperature_final"].transform("mean")
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["ts"].dt.dayofyear / 365.25)
    WEATHER_PERSIST = ["PBLH_min_24h", "VC_min_24h", "stagnation_hours_12h", "temp_diurnal_anomaly"]

    # === 9. RFSI SETUP =========================================================
    coords = {s: (sid_lat[s], sid_lon[s]) for s in station_ids}
    sid_to_idx = {s: i for i, s in enumerate(station_ids)}
    dist_full = np.zeros((n_stn, n_stn))
    for i in range(n_stn):
        for j in range(i + 1, n_stn):
            d = haversine(*coords[station_ids[i]], *coords[station_ids[j]])
            dist_full[i, j] = d
            dist_full[j, i] = d
    neighbor_order = {i: sorted([(j, dist_full[i, j]) for j in range(n_stn) if j != i], key=lambda x: x[1]) for i in range(n_stn)}
    pm25_wide = df.pivot_table(index="ts", columns="stationId", values="PM2.5", aggfunc="first")
    pm25_mat = pm25_wide.values
    sid_cols = list(pm25_wide.columns)
    sid_to_col = {s: i for i, s in enumerate(sid_cols)}
    ts_to_row = pd.Series(range(len(pm25_wide)), index=pm25_wide.index)
    df["ts_row"] = df["ts"].map(ts_to_row).astype(int).values
    n_ts = pm25_mat.shape[0]
    LAG_HOURS = [1, 3, 6]

    def compute_rfsi(exclude_sid=None):
        n = len(df)
        pm_nn = np.full((n, K_NN), np.nan); d_nn = np.full((n, K_NN), np.nan)
        excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
        ts_row_vals = df["ts_row"].values
        for sid in station_ids:
            si = sid_to_idx[sid]
            mask = stationId_vals == sid
            if not mask.any():
                continue
            ri = np.where(mask)[0]; tr = ts_row_vals[ri]
            cands = [(j, d) for j, d in neighbor_order[si] if excl is None or j != excl]
            if not cands:
                continue
            ccols = np.array([sid_to_col[station_ids[j]] for j, _ in cands])
            cdists = np.array([d for _, d in cands])
            nbr = pm25_mat[np.ix_(tr, ccols)]
            valid = ~np.isnan(nbr); cumv = np.cumsum(valid, axis=1)
            for k in range(K_NN):
                reached = cumv >= (k + 1); has = reached.any(axis=1)
                if not has.any():
                    break
                pos = np.argmax(reached, axis=1); ih = np.where(has)[0]
                pm_nn[ri[ih], k] = nbr[ih, pos[has]]; d_nn[ri[ih], k] = cdists[pos[has]]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / d_nn
            pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)
        return {"PM25_nn_idw": pm_idw, "PM25_nn1": pm_nn[:, 0], "dist_nn1": d_nn[:, 0],
                "PM25_nn2": pm_nn[:, 1], "PM25_nn3": pm_nn[:, 2]}

    def compute_lagged_rfsi(exclude_sid=None):
        n = len(df)
        lags = {lh: np.full(n, np.nan) for lh in LAG_HOURS}
        excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
        ts_row_vals = df["ts_row"].values
        for sid in station_ids:
            si = sid_to_idx[sid]
            mask = stationId_vals == sid
            if not mask.any():
                continue
            ri = np.where(mask)[0]; tr = ts_row_vals[ri]
            cands = [(j, d) for j, d in neighbor_order[si] if excl is None or j != excl]
            if not cands:
                continue
            nn1_col = sid_to_col[station_ids[cands[0][0]]]
            for lag_h in LAG_HOURS:
                tr_lag = tr - lag_h; in_bounds = tr_lag >= 0
                tr_safe = np.clip(tr_lag, 0, n_ts - 1)
                vals = pm25_mat[tr_safe, nn1_col]; vals[~in_bounds] = np.nan
                lags[lag_h][ri] = vals
        return {f"PM25_nn1_lag{lh}h": lags[lh] for lh in LAG_HOURS}

    # === 10. FEATURE SET DEFINITIONS + STREAMS =================================
    MET_CORE = ["PBLH", "VC", "wind_u", "wind_v", "WS_local", "Temperature_final", "Humidity_final",
                "Pressure_final", "dT_6h", "dRH_6h", "rain_days_7d", "rain_sum_48h",
                "consecutive_dry_days", "hrs_since_rain", "RH_factor"]
    TEMPORAL = ["hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_cos", "day_of_year_sin", "dow_is_weekend"]
    AOD_CORE = ["AOT_ffill_48h", "AOT_outer_mean", "AE", "RF", "hours_since_valid_AOT"] + SOURCE_AOD_FEATURES
    AOD_EXTENDED = AOD_CORE + OUTER_ALL_EXTRA + ["RF_center", "RF_mean", "SSA_center", "SSA_mean",
                                                 "AOT_fine", "AOT_grad_mag", "AOT_local_vs_regional"]
    SMART_EMISSION = ["smart_v1_center", "smart_v1_upwind", "smart_v1_max", "smart_v1_contrast", "smart_v1_upwind_x_VC_inv"]
    GAS_STANDALONE = ["so2_upwind", "so2_center", "so2_contrast", "co_upwind", "co_center", "hcho_center",
                      "so2_upwind_x_VC_inv", "lst_anom_upwind_x_VC_inv"]
    EMISSION_STATIC = NO2_STATIC_COLS + NO2_SECTOR_COLS + ["ntl_center"] + NTL_SECTOR_COLS + ["lst_anom_center"] + LST_SECTOR_COLS
    BUILDING = ["building_area_1km", "building_count_3km"]
    TERRAIN = ["elevation_m", "slope_deg"]
    RFSI_ALL = ["PM25_nn_idw", "PM25_nn1", "dist_nn1", "PM25_nn2", "PM25_nn3",
                "PM25_nn1_lag1h", "PM25_nn1_lag3h", "PM25_nn1_lag6h"]
    STREAMS = {
        "dispersion": MET_CORE + WEATHER_PERSIST + TEMPORAL + TERRAIN + BUILDING + ["fire_upwind"],
        "satellite": AOD_EXTENDED + ["PBLH", "VC", "RH_factor", "hours_since_rain"] + TEMPORAL[:5],
        "emission": SMART_EMISSION + GAS_STANDALONE + DAILY_ANOM_ALL + EMISSION_STATIC + BUILDING + TERRAIN + ["PBLH", "VC", "fire_upwind"] + TEMPORAL,
        "spatial": RFSI_ALL + MET_CORE + TEMPORAL + TERRAIN,
        "full": MET_CORE + WEATHER_PERSIST + TEMPORAL + TERRAIN + BUILDING + AOD_EXTENDED + SMART_EMISSION + GAS_STANDALONE + DAILY_ANOM_ALL + EMISSION_STATIC + RFSI_ALL + ["fire_upwind"],
    }
    for name, feats in list(STREAMS.items()):
        feats_clean = sorted(set(f for f in feats if f in df.columns))
        STREAMS[name] = feats_clean
        STREAMS[f"raw_{name}"] = feats_clean

    return df, STREAMS, compute_rfsi, compute_lagged_rfsi
