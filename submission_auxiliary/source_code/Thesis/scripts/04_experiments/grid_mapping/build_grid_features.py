# -*- coding: utf-8 -*-
"""Build the FEAT_OBS input columns for a 0.02-deg grid over the Red River Delta
at 4 map timestamps (2025-12-09 & 2025-07-30, 08:00 & 20:00 LOCAL VN).

Replicates unified_thesis.csv column semantics (build_unified.py) so the
exp_red_river_delta.py feature pipeline can be applied to grid cells exactly
as it is applied to external validation stations.

Output: D:/map_data/grid/grid_unified_cols.csv  (one row per cell per timestamp)
"""
import glob
import os
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import convolve
from scipy.interpolate import RegularGridInterpolator
import netCDF4

from pathlib import Path

MAP_DATA = os.environ.get("MAP_DATA", "D:/map_data")

HIMA = MAP_DATA + "/himawari_raw"
OUT = MAP_DATA + "/grid/grid_unified_cols.csv"
ROOT = Path(__file__).resolve().parents[4].as_posix()

# ---- grid over DELTA_BOX ----
BOX = (20.3, 21.3, 105.5, 107.0)
RES = 0.02
cell_lats = np.round(np.arange(BOX[0] + RES/2, BOX[1], RES), 3)   # 50
cell_lons = np.round(np.arange(BOX[2] + RES/2, BOX[3], RES), 3)   # 75
LON_G, LAT_G = np.meshgrid(cell_lons, cell_lats)
N_CELL = LAT_G.size
print(f"grid: {len(cell_lats)} x {len(cell_lons)} = {N_CELL} cells")

WINDOWS = {
    "dec": {"utc_days": ["202512/07", "202512/08", "202512/09"],
            "targets_local": ["2025-12-09 08:00", "2025-12-09 20:00"]},
    "jul": {"utc_days": ["202507/28", "202507/29", "202507/30"],
            "targets_local": ["2025-07-30 08:00", "2025-07-30 20:00"]},
}

# =====================================================================
# 1) HIMAWARI hourly cube (local time) + window stats at target hours
# =====================================================================
K5 = np.ones((5, 5)); K3 = np.ones((3, 3))

def win_stats(A):
    """5x5 / 3x3 NaN-aware window stats on a 2D array."""
    valid = np.isfinite(A).astype(float)
    A0 = np.where(np.isfinite(A), A, 0.0)
    s1_5 = convolve(A0, K5, mode="constant", cval=0.0)
    c5 = convolve(valid, K5, mode="constant", cval=0.0)
    s2_5 = convolve(A0 * A0, K5, mode="constant", cval=0.0)
    s1_3 = convolve(A0, K3, mode="constant", cval=0.0)
    c3 = convolve(valid, K3, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean5 = np.where(c5 > 0, s1_5 / c5, np.nan)
        var5 = np.where(c5 > 1, (s2_5 - c5 * mean5**2) / (c5 - 1), np.nan)
        std5 = np.sqrt(np.clip(var5, 0, None))
        inner = np.where(c3 > 0, s1_3 / c3, np.nan)
        outer_c = c5 - c3
        outer = np.where(outer_c > 0, (s1_5 - s1_3) / outer_c, np.nan)
    return mean5, std5, inner, outer

def load_hima_window(utc_days):
    """Hourly (local) stacks of AOT/AE/RF + window stats sampled later."""
    files = []
    for d in utc_days:
        files += sorted(glob.glob(os.path.join(HIMA, d, "*", "*.tif")))
    if not files:
        raise RuntimeError("no himawari files for " + str(utc_days))
    # group by UTC hour from filename ..._YYYYMMDD_HHMM_...
    by_hour = {}
    for f in files:
        base = os.path.basename(f)
        parts = base.split("_")
        ymd, hm = parts[4], parts[5]
        key = ymd + hm[:2]
        by_hour.setdefault(key, []).append(f)

    hours = sorted(by_hour)
    ref = rasterio.open(by_hour[hours[0]][0])
    H, W = ref.height, ref.width
    transform = ref.transform
    ref.close()

    def band_idx(src):
        desc = list(src.descriptions)
        if "AOT" in desc:      # 7-band variant with names (AE,AOT,Hour,...)
            return desc.index("AOT") + 1, desc.index("AE") + 1, desc.index("RF") + 1, False
        # 6-band files from process_himawari.py: AOT,Unc,AE,QA,SSA,RF; 0 = fill
        return 1, 3, 6, True

    ts_local, cubes = [], {"AOT": [], "AE": [], "RF": []}
    for key in hours:
        acc = {b: [] for b in cubes}
        for f in by_hour[key]:
            with rasterio.open(f) as src:
                iAOT, iAE, iRF, zero_is_fill = band_idx(src)
                nod = src.nodata
                for b, idx in [("AOT", iAOT), ("AE", iAE), ("RF", iRF)]:
                    a = src.read(idx).astype(float)
                    if nod is not None:
                        a = np.where(a == nod, np.nan, a)
                    if zero_is_fill:
                        a = np.where(a == 0, np.nan, a)
                    acc[b].append(a)
        for b in cubes:
            stack = np.stack(acc[b])
            with np.errstate(invalid="ignore"):
                cubes[b].append(np.nanmean(stack, axis=0))
        ts = pd.Timestamp(key[:8] + " " + key[8:] + ":00") + pd.Timedelta(hours=7)
        ts_local.append(ts)
    for b in cubes:
        cubes[b] = np.stack(cubes[b])  # [T, H, W]
    return np.array(ts_local), cubes, transform, (H, W)

def hima_features_at(ts_local, cubes, transform, targets):
    """Return dict of per-cell himawari features for each target timestamp."""
    # cell -> pixel index
    rows, cols = rasterio.transform.rowcol(transform, LON_G.ravel(), LAT_G.ravel())
    rows = np.asarray(rows); cols = np.asarray(cols)

    T = len(ts_local)
    # precompute window stats per hour (only AOT needs windows)
    mean5 = np.full((T,) + cubes["AOT"].shape[1:], np.nan)
    std5 = np.full_like(mean5, np.nan)
    inner = np.full_like(mean5, np.nan)
    outer = np.full_like(mean5, np.nan)
    for t in range(T):
        mean5[t], std5[t], inner[t], outer[t] = win_stats(cubes["AOT"][t])

    aot_c = cubes["AOT"][:, rows, cols]      # [T, N]
    ae_c = cubes["AE"][:, rows, cols]
    rf_c = cubes["RF"][:, rows, cols]
    in_c = inner[:, rows, cols]
    out_c = outer[:, rows, cols]
    std_c = std5[:, rows, cols]

    res = {}
    ts_index = pd.DatetimeIndex(ts_local)
    for tgt in targets:
        tgt = pd.Timestamp(tgt)
        it = ts_index.get_indexer([tgt])[0]
        if it < 0:
            # target hour missing from archive (rare) -> nearest earlier hour
            it = int(np.searchsorted(ts_index.values, np.datetime64(tgt), "right") - 1)
        hist = slice(0, it + 1)
        aot_hist = aot_c[hist]               # [t, N]
        # rolling 24h mean (nan-skip) over last 24 hourly frames
        last24 = aot_hist[-24:]
        with np.errstate(invalid="ignore"):
            roll24 = np.nanmean(last24, axis=0)
        # ffill 48h + hours_since_valid
        n_hist = aot_hist.shape[0]
        idx_grid = np.arange(n_hist)[:, None] * np.isfinite(aot_hist)
        idx_grid[~np.isfinite(aot_hist)] = -1
        last_idx = idx_grid.max(axis=0)      # -1 if never valid
        hrs_since = np.where(last_idx >= 0, (n_hist - 1) - last_idx, np.nan)
        ffill48 = np.where(
            (last_idx >= 0) & (hrs_since <= 48),
            aot_hist[np.clip(last_idx, 0, None), np.arange(aot_hist.shape[1])],
            np.nan)
        res[str(tgt)] = {
            "AOT": aot_c[it], "AE": ae_c[it], "RF": rf_c[it],
            "AOT_inner_mean": in_c[it], "AOT_outer_mean": out_c[it],
            "AOT_spatial_std": std_c[it],
            "AOT_rolling_mean_24h": roll24,
            "AOT_ffill_48h": ffill48,
            "hours_since_valid_AOT": hrs_since,
        }
    return res

# =====================================================================
# 2) MODIS + TROPOMI daily stacks -> rolling stats per cell
# =====================================================================
def sample_multiband(path, scale=1.0):
    """Read a GEE multiband tif, return {band_desc: 2d}, plus cell sampler."""
    with rasterio.open(path) as src:
        arr = src.read().astype(float)
        nod = src.nodata
        if nod is not None:
            arr = np.where(arr == nod, np.nan, arr)
        arr *= scale
        rows, cols = rasterio.transform.rowcol(
            src.transform, LON_G.ravel(), LAT_G.ravel())
        rows = np.clip(np.asarray(rows), 0, src.height - 1)
        cols = np.clip(np.asarray(cols), 0, src.width - 1)
        sampled = arr[:, rows, cols]        # [bands, N]
        descs = list(src.descriptions)
    return descs, sampled

def daily_series(descs, sampled, prefix):
    """Extract {date: values[N]} for bands named prefix_YYYYMMDD."""
    out = {}
    for i, d in enumerate(descs):
        if d and d.startswith(prefix + "_"):
            out[pd.Timestamp(d.split("_")[-1])] = sampled[i]
    dates = sorted(out)
    return pd.DatetimeIndex(dates), np.stack([out[d] for d in dates])  # [D, N]

def roll_stats(dates, X, end_date, window_days, min_periods, stats):
    """Time-window rolling stats ending at end_date (inclusive) per cell."""
    m = (dates > end_date - pd.Timedelta(days=window_days)) & (dates <= end_date)
    sub = X[m]
    n = np.isfinite(sub).sum(axis=0)
    res = {}
    with np.errstate(invalid="ignore"):
        if "mean" in stats: res["mean"] = np.nanmean(sub, axis=0)
        if "std" in stats: res["std"] = np.nanstd(sub, axis=0, ddof=1)
        if "p90" in stats: res["p90"] = np.nanpercentile(sub, 90, axis=0)
        if "iqr" in stats:
            res["iqr"] = (np.nanpercentile(sub, 75, axis=0) -
                          np.nanpercentile(sub, 25, axis=0))
    for k in res:
        res[k] = np.where(n >= min_periods, res[k], np.nan)
    return res

# =====================================================================
# 3) MET (Open-Meteo regular 0.1-deg grid) + PBLH (ERA5 nc)
# =====================================================================
met = pd.read_csv(MAP_DATA + "/met/met_grid.csv", parse_dates=["ts_utc"])
met["ts_local"] = met["ts_utc"] + pd.Timedelta(hours=7)
met_lats = np.sort(met["lat"].unique())
met_lons = np.sort(met["lon"].unique())

def met_interp_series(ts_list, var):
    """Return [T, N] of var interpolated to cells for local timestamps."""
    out = np.full((len(ts_list), N_CELL), np.nan)
    piv = met.pivot_table(index="ts_local", columns=["lat", "lon"], values=var)
    for i, ts in enumerate(ts_list):
        if ts not in piv.index:
            continue
        vals = piv.loc[ts].values.reshape(len(met_lats), len(met_lons))
        f = RegularGridInterpolator((met_lats, met_lons), vals,
                                    bounds_error=False, fill_value=None)
        out[i] = f(np.column_stack([LAT_G.ravel(), LON_G.ravel()]))
    return out

def pblh_series(ts_list, month_tag):
    fn = ROOT + f"/data/era5/pblh_2025_{month_tag}.nc"
    ds = netCDF4.Dataset(fn)
    blh = ds.variables["blh"][:]                     # [T, lat, lon]
    lat = ds.variables["latitude"][:]
    lon = ds.variables["longitude"][:]
    tv = ds.variables["valid_time"]
    times = pd.to_datetime(netCDF4.num2date(tv[:], tv.units,
              only_use_cftime_datetimes=False)) + pd.Timedelta(hours=7)
    ds.close()
    lat_o = np.argsort(lat); lat_s = lat[lat_o]
    out = np.full((len(ts_list), N_CELL), np.nan)
    tidx = pd.DatetimeIndex(times)
    pts = np.column_stack([LAT_G.ravel(), LON_G.ravel()])
    for i, ts in enumerate(ts_list):
        j = tidx.get_indexer([ts])[0]
        if j < 0:
            continue
        f = RegularGridInterpolator((lat_s, lon), blh[j][lat_o, :],
                                    bounds_error=False, fill_value=None)
        out[i] = f(pts)
    return out

# =====================================================================
# 4) GPM rain (optional — NaN if file absent)
# =====================================================================
def rain_features(tag, targets_local):
    path = f"{MAP_DATA}/gpm/gpm_rain_{tag}.tif"
    n = N_CELL
    empty = {str(pd.Timestamp(t)): {
        "rain_days_7d": np.full(n, np.nan),
        "consecutive_dry_days": np.full(n, np.nan),
        "hrs_since_rain": np.full(n, np.nan)} for t in targets_local}
    if not os.path.exists(path):
        print(f"  GPM {tag}: file not found -> NaN rain features")
        return empty
    descs, sampled = sample_multiband(path)
    ddates, D = daily_series(descs, sampled, "rain")          # daily mm
    # hourly bands rainh_YYYYMMDD_HH (UTC)
    hh, HH = [], []
    for i, d in enumerate(descs):
        if d and d.startswith("rainh_"):
            ts = pd.Timestamp(d[6:14] + " " + d[15:17] + ":00") + pd.Timedelta(hours=7)
            hh.append(ts); HH.append(sampled[i])
    hidx = pd.DatetimeIndex(hh); HO = np.stack(HH)            # [Th, N] local
    out = {}
    for t in targets_local:
        tgt = pd.Timestamp(t)
        # rain_days_7d: trailing 7 calendar days incl target date
        dmask = (ddates > tgt.normalize() - pd.Timedelta(days=7)) & \
                (ddates <= tgt.normalize())
        rd = np.nansum(np.where(D[dmask] > 0.1, 1, 0), axis=0).astype(float)
        # consecutive_dry_days: count back from target date
        dm = ddates <= tgt.normalize()
        sub = np.where(D[dm] > 0.1, 1, 0)                     # [D, N] rain flags
        cdd = np.zeros(N_CELL)
        for r in range(sub.shape[0] - 1, -1, -1):
            active = (cdd == (sub.shape[0] - 1 - r))          # still counting
            cdd = np.where(active & (sub[r] == 0), cdd + 1, cdd)
        # hrs_since_rain from hourly (local), within available window
        hm = hidx <= tgt
        subh = HO[hm]
        rainy = subh > 0.1
        nh = rainy.shape[0]
        idx = np.arange(nh)[:, None] * rainy
        idx[~rainy] = -1
        last = idx.max(axis=0)
        hsr = np.where(last >= 0, (nh - 1) - last, np.nan)
        out[str(tgt)] = {"rain_days_7d": rd,
                         "consecutive_dry_days": cdd,
                         "hrs_since_rain": hsr}
    return out

# =====================================================================
# MAIN
# =====================================================================
rows_out = []
for tag, W in WINDOWS.items():
    print(f"\n=== window {tag} ===")
    targets = [pd.Timestamp(t) for t in W["targets_local"]]

    print("  himawari cube...")
    ts_local, cubes, transform, hw = load_hima_window(W["utc_days"])
    print(f"    {len(ts_local)} hourly frames {ts_local[0]} -> {ts_local[-1]}")
    hima = hima_features_at(ts_local, cubes, transform, targets)

    print("  modis / tropomi rolling...")
    descs_m, samp_m = sample_multiband(f"{MAP_DATA}/modis/modis_maiac_{tag}.tif",
                                       scale=0.001)
    md, MX = daily_series(descs_m, samp_m, "aod055")
    descs_t, samp_t = sample_multiband(f"{MAP_DATA}/tropomi/tropomi_gases_{tag}.tif")
    gas = {g: daily_series(descs_t, samp_t, g) for g in ["no2", "so2", "co", "hcho"]}

    print("  met + pblh series...")
    # hourly series long enough for dT6h / PBLH_min_24h / stagnation_12h
    ts_all = sorted(set().union(*[
        pd.date_range(t - pd.Timedelta(hours=30), t, freq="h") for t in targets]))
    T2M = met_interp_series(ts_all, "temperature_2m")
    RH2 = met_interp_series(ts_all, "relative_humidity_2m")
    SP = met_interp_series(ts_all, "surface_pressure")
    WS = met_interp_series(ts_all, "wind_speed_10m")
    WD = met_interp_series(ts_all, "wind_direction_10m")
    PB = pblh_series(ts_all, "12" if tag == "dec" else "07")
    ts_pos = {t: i for i, t in enumerate(ts_all)}

    rain = rain_features(tag, W["targets_local"])

    for tgt in targets:
        k = str(tgt)
        i_t = ts_pos[tgt]
        i_6 = ts_pos[tgt - pd.Timedelta(hours=6)]
        sl24 = slice(i_t - 23, i_t + 1)
        sl12 = slice(i_t - 11, i_t + 1)

        wd_rad = np.radians(WD[i_t])
        base = {
            "ts": k, "window": tag,
            "lat": LAT_G.ravel(), "lon": LON_G.ravel(),
            # met (unified-level)
            "Temperature_final": T2M[i_t], "Humidity_final": RH2[i_t],
            "Pressure_final": SP[i_t], "PBLH": PB[i_t],
            "WS_om": WS[i_t], "WD_om": WD[i_t],
            "wind_u": WS[i_t] * np.sin(wd_rad),
            "wind_v": WS[i_t] * np.cos(wd_rad),
            # WS_local is a station-sensor field and has no grid equivalent.
            # Keep it missing; gridded wind is represented by wind_u/wind_v.
            "WS_local": np.full(N_CELL, np.nan),
            "dT_6h": T2M[i_t] - T2M[i_6],
            "dRH_6h": RH2[i_t] - RH2[i_6],
            "PBLH_min_24h": np.nanmin(PB[sl24], axis=0),
            # stagnation: (PBLH<500) & (WS_local.fillna(0)<2) == (PBLH<500)
            "stagnation_hours_12h": np.nansum(PB[sl12] < 500, axis=0).astype(float),
            # temporal
            "hour_sin": np.full(N_CELL, np.sin(2*np.pi*tgt.hour/24)),
            "hour_cos": np.full(N_CELL, np.cos(2*np.pi*tgt.hour/24)),
            "month_sin": np.full(N_CELL, np.sin(2*np.pi*(tgt.month-1)/12)),
            "month_cos": np.full(N_CELL, np.cos(2*np.pi*(tgt.month-1)/12)),
        }
        base.update({kk: vv for kk, vv in hima[k].items()})
        base["AOT_fine"] = base["AOT"] * base["RF"]

        # --- modis temporal (per exp script semantics) ---
        # modis_aod_7d = 7d rolling mean of daily MAIAC; fine = x 5.0 (modal fmf)
        m7 = roll_stats(md, MX, tgt.normalize(), 7, 1, ["mean"])["mean"]
        base["modis_aod_7d"] = m7
        base["modis_fine_aod_7d"] = m7 * 5.0
        # aod_30d_* rolled on the modis_aod_7d series
        m7_series = np.stack([
            roll_stats(md, MX, d, 7, 1, ["mean"])["mean"] for d in md])
        r30 = roll_stats(md, m7_series, tgt.normalize(), 30, 10,
                         ["mean", "std", "p90", "iqr"])
        base["aod_30d_mean"] = r30["mean"]; base["aod_30d_std"] = r30["std"]
        base["aod_30d_p90"] = r30["p90"]; base["aod_30d_iqr"] = r30["iqr"]
        base["aod_30d_cv"] = r30["std"] / (r30["mean"] + 1e-9)

        # --- tropomi daily anomalies + 30d rolling ---
        for g, (gd, GX) in gas.items():
            di = gd.get_indexer([tgt.normalize()])
            today = GX[di[0]] if di[0] >= 0 else np.full(N_CELL, np.nan)
            mm = gd.month == tgt.month
            clim = np.nanmean(GX[mm], axis=0)
            base[f"{g}_daily_anom"] = today - clim
        hd, HX = gas["hcho"]
        rh30 = roll_stats(hd, HX, tgt.normalize(), 30, 5, ["mean", "p90", "std"])
        base["hcho_30d_mean"] = rh30["mean"]; base["hcho_30d_p90"] = rh30["p90"]
        base["hcho_30d_cv"] = rh30["std"] / (np.abs(rh30["mean"]) + 1e-12)
        cd, CX = gas["co"]
        rc30 = roll_stats(cd, CX, tgt.normalize(), 30, 5, ["mean", "std", "iqr"])
        base["co_30d_mean"] = rc30["mean"]; base["co_30d_std"] = rc30["std"]
        base["co_30d_iqr"] = rc30["iqr"]

        base.update(rain[k])
        rows_out.append(pd.DataFrame(base))
        print(f"    {k}: AOT valid {np.isfinite(base['AOT']).mean()*100:.0f}% | "
              f"ffill48 {np.isfinite(base['AOT_ffill_48h']).mean()*100:.0f}% | "
              f"modis7d {np.isfinite(m7).mean()*100:.0f}% | "
              f"PBLH {np.isfinite(base['PBLH']).mean()*100:.0f}%")

grid = pd.concat(rows_out, ignore_index=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
grid.to_csv(OUT, index=False)
print(f"\nSAVED {OUT}  rows={len(grid)} cols={len(grid.columns)}")
