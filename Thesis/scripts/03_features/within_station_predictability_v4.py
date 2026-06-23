"""
Within-station predictability using full v4 enriched features.
Two runs:
  A) 40 thesis stations (KK, minus 3 broken)
  B) All stations (thesis + LCS pass), minus 3 broken

For each station: 5-fold KFold with station's OWN data in training.
Features: full v4 enrichment (met, AOD, emissions, temporal, building, terrain)
          — NO RFSI (that's spatial, not temporal).
"""
import sys, io, os, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def _repo_root():
    """Walk up to repo root (dir containing data/merged) so this runs from anywhere."""
    p = os.path.abspath(os.path.dirname(__file__))
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "data", "merged")):
            return p
        p = os.path.dirname(p)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(_repo_root())
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

t0 = time.time()

BROKEN = {"31616865099255512061948816121", "30991938797551443885460120607",
          "29098319146067624969113973428"}

SECTOR_NAMES = ["N","NE","E","SE","S","SW","W","NW"]
SEASON_MAP = {1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",
              9:"SON",10:"SON",11:"SON",12:"DJF"}

def tier(pm):
    return "t0" if pm < 10 else "t1" if pm < 20 else "t2" if pm < 35 else "t3"

def hgb():
    return HistGradientBoostingRegressor(max_iter=400, max_depth=7,
                                         learning_rate=0.05, min_samples_leaf=40,
                                         random_state=0)

def kfold_r2(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    preds = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        m = hgb().fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return r2_score(y, preds)

def temporal_r2(X, y, frac=0.7):
    n = len(y); cut = int(n * frac)
    if n - cut < 100: return np.nan
    m = hgb().fit(X[:cut], y[:cut])
    return r2_score(y[cut:], m.predict(X[cut:]))

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading unified_thesis_v4 ...")
df = pd.read_csv("data/merged/unified_thesis_v4.csv", dtype={"stationId": str}, low_memory=False)
df["ts"] = pd.to_datetime(df["ts"])
df = df.dropna(subset=["PM2.5"]).sort_values(["stationId","ts"])
df["date"] = df["ts"].dt.date
df["month"] = df["ts"].dt.month
df["hour"] = df["ts"].dt.hour
print(f"  {len(df):,} rows, {df['stationId'].nunique()} stations")

# Thesis station IDs
thesis_meta = pd.read_csv("Thesis/results/01_stations/station_selection_final.csv",
                          dtype={"stationId": str}, encoding="utf-8-sig")
THESIS_SIDS = set(thesis_meta["stationId"])

# LCS pass IDs
lcs_sel = pd.read_csv("Thesis/results/01_stations/station_selection_lcs.csv",
                       dtype={"station_id": str})
PASS_IDS = set(lcs_sel[lcs_sel["lcs_flag"]=="pass"]["station_id"])

all_sids = sorted(df["stationId"].unique())
stationId_vals = df["stationId"].values

# ── Enrichment (matching v4 pipeline) ────────────────────────────────────────
META_DIR = "Thesis/results/01_stations"

# Temporal cyclical
h = df["ts"].dt.hour; doy = df["ts"].dt.dayofyear; mo = df["ts"].dt.month
df["hour_sin"] = np.sin(2*np.pi*h/24); df["hour_cos"] = np.cos(2*np.pi*h/24)
df["month_sin"] = np.sin(2*np.pi*mo/12); df["month_cos"] = np.cos(2*np.pi*mo/12)
df["day_of_year_cos"] = np.cos(2*np.pi*doy/365); df["day_of_year_sin"] = np.sin(2*np.pi*doy/365)
df["dow_is_weekend"] = (df["ts"].dt.dayofweek >= 5).astype(float)

# Derived met
for col in ["Temperature_final","Humidity_final","dT_6h","dRH_6h","RH_factor",
            "rain_days_7d","rain_sum_48h","consecutive_dry_days","hrs_since_rain",
            "wind_u","wind_v","WS_local","PBLH","VC","Pressure_final"]:
    if col not in df.columns:
        df[col] = np.nan

# Satellite daily anomalies
sat_path = "data/merged/satellite_daily_features.csv"
if os.path.exists(sat_path):
    sat_long = pd.read_csv(sat_path, dtype={"stationId": str})
    sat_wide = sat_long.pivot_table(index=["stationId","date"], columns="variable",
        values="mean", aggfunc="first").reset_index()
    sat_wide.columns.name = None
    sat_wide["date"] = pd.to_datetime(sat_wide["date"])
    sat_wide = sat_wide[sat_wide["stationId"].isin(set(all_sids))].copy()
    sat_wide["month"] = sat_wide["date"].dt.month
    CLIM_COLS = ["NO2","SO2","CO","HCHO","LST_terra_day","LST_terra_night"]
    for c in CLIM_COLS:
        if c not in sat_wide.columns: sat_wide[c] = np.nan
    clim = sat_wide.groupby(["stationId","month"])[CLIM_COLS].transform("mean")
    ANOM_RAW = []
    for gas, raw in [("so2","SO2"),("co","CO"),("no2","NO2")]:
        col = f"{gas}_daily_anom"
        sat_wide[col] = sat_wide[raw] - clim[raw]
        ANOM_RAW.append(col)
    sat_wide["lst_day_anom"] = sat_wide["LST_terra_day"] - clim["LST_terra_day"]
    sat_wide["lst_night_anom"] = sat_wide["LST_terra_night"] - clim["LST_terra_night"]
    ANOM_RAW += ["lst_day_anom", "lst_night_anom"]
    sat_wide["date_merge"] = sat_wide["date"].dt.date
    df = df.merge(sat_wide[["stationId","date_merge"]+ANOM_RAW],
                  left_on=["stationId","date"], right_on=["stationId","date_merge"], how="left")
    if "date_merge" in df.columns: df.drop(columns=["date_merge"], inplace=True)
else:
    ANOM_RAW = []

# Building density
bld_path = os.path.join(META_DIR, "station_building_density.csv")
if os.path.exists(bld_path):
    bld = pd.read_csv(bld_path, dtype={"stationId": str})
    BUILDING_COLS = ["building_count_1km","building_area_1km","building_count_3km","building_area_3km"]
    bld_cols = [c for c in BUILDING_COLS if c in bld.columns]
    df = df.merge(bld.set_index("stationId")[bld_cols], left_on="stationId", right_index=True, how="left")
    for c in bld_cols: df[c] = df[c].fillna(0)

# Matched AOD
matched_static_path = os.path.join(META_DIR, "aod_source_matched_static.csv")
matched_temporal_path = os.path.join(META_DIR, "aod_source_matched_temporal.csv")
SOURCE_AOD_FEATURES = []
if os.path.exists(matched_static_path) and os.path.exists(matched_temporal_path):
    matched_static = pd.read_csv(matched_static_path, dtype={"stationId": str}).set_index("stationId")
    matched_temporal = pd.read_csv(matched_temporal_path, dtype={"stationId": str}, parse_dates=["date"])
    matched_temporal["date_merge"] = matched_temporal["date"].dt.date
    prefix = "him"
    temporal_rename = {f"{prefix}_aod_7d": "src_aod_7d", f"{prefix}_fmf_7d": "src_fmf_7d",
                       f"{prefix}_fine_aod_7d": "src_fine_aod_7d"}
    keep_t = ["stationId","date_merge"] + [c for c in temporal_rename if c in matched_temporal.columns]
    matched_temporal = matched_temporal[keep_t].rename(columns=temporal_rename)
    df = df.merge(matched_temporal, left_on=["stationId","date"],
                  right_on=["stationId","date_merge"], how="left")
    if "date_merge" in df.columns: df.drop(columns=["date_merge"], inplace=True)
    static_rename = {}
    for suffix in (["aod_center","aod_DJF","aod_MAM","aod_JJA","aod_SON",
                    "aod_contrast","aod_directionality","aod_max_nearby",
                    "fmf_center","fine_aod_center","ae_center"] +
                   [f"aod_clim_{d}" for d in SECTOR_NAMES] +
                   [f"fine_aod_clim_{d}" for d in SECTOR_NAMES]):
        col = f"{prefix}_{suffix}"
        if col in matched_static.columns: static_rename[col] = f"src_{suffix}"
    if static_rename:
        df = df.merge(matched_static[list(static_rename)].rename(columns=static_rename),
                      left_on="stationId", right_index=True, how="left")
    SOURCE_AOD_FEATURES = [f"src_{s}" for s in
        ["aod_7d","fmf_7d","fine_aod_7d","aod_center","fmf_center","fine_aod_center",
         "ae_center","aod_DJF","aod_MAM","aod_JJA","aod_SON","aod_contrast",
         "aod_directionality","aod_max_nearby"] +
        [f"aod_clim_{d}" for d in SECTOR_NAMES] +
        [f"fine_aod_clim_{d}" for d in SECTOR_NAMES]]
    SOURCE_AOD_FEATURES = [f for f in SOURCE_AOD_FEATURES if f in df.columns]

# Static satellite features (NO2, NTL, LST)
no2_path = os.path.join(META_DIR, "station_no2_features.csv")
emit_path = os.path.join(META_DIR, "station_emission_features.csv")
NO2_STATIC_COLS = ["no2_center","no2_contrast","no2_directionality"]
NO2_SECTOR_COLS = [f"no2_clim_{d}" for d in SECTOR_NAMES]
NTL_SECTOR_COLS = [f"ntl_clim_{d}" for d in SECTOR_NAMES]
LST_SECTOR_COLS = [f"lst_anom_clim_{d}" for d in SECTOR_NAMES]
if os.path.exists(no2_path):
    no2_map = pd.read_csv(no2_path, dtype={"stationId": str}).set_index("stationId")
    df = df.merge(no2_map[[c for c in NO2_STATIC_COLS + NO2_SECTOR_COLS if c in no2_map.columns]],
                  left_on="stationId", right_index=True, how="left")
else:
    no2_map = pd.DataFrame()
if os.path.exists(emit_path):
    emit_map = pd.read_csv(emit_path, dtype={"stationId": str}).set_index("stationId")
    merge_emit = [c for c in ["ntl_center"] + NTL_SECTOR_COLS + ["lst_anom_center"] + LST_SECTOR_COLS
                  if c in emit_map.columns]
    df = df.merge(emit_map[merge_emit], left_on="stationId", right_index=True, how="left")
else:
    emit_map = pd.DataFrame()

# SMART_V1 + directional wind transport
def _load_dir_clim_123(csv_path, value_col="mean"):
    raw = pd.read_csv(csv_path, dtype={"stationId": str})
    sectors, centers = {}, {}
    for sid in all_sids:
        sf = raw[raw["stationId"] == sid]
        sec = np.full(8, np.nan)
        for di, d in enumerate(SECTOR_NAMES):
            vals = sf[sf["direction"] == d][value_col]
            if len(vals) > 0: sec[di] = float(vals.iloc[0])
        sectors[sid] = sec
        cvals = sf[sf["direction"] == "C"][value_col]
        centers[sid] = float(cvals.iloc[0]) if len(cvals) > 0 else np.nan
    return sectors, centers

wd_from = np.degrees(np.arctan2(-df["wind_u"].values, -df["wind_v"].values)) % 360
sector_idx = ((wd_from + 22.5) / 45).astype(int) % 8
ws = np.sqrt(df["wind_u"].values**2 + df["wind_v"].values**2)
vc_inv = 1.0 / (df["PBLH"].clip(lower=50).values * ws.clip(min=0.1) + 1)

dir_files = {
    "so2": (os.path.join(META_DIR, "tropomi_so2_directional_123.csv"), "mean"),
    "co": (os.path.join(META_DIR, "tropomi_co_directional_123.csv"), "mean"),
    "hcho": (os.path.join(META_DIR, "tropomi_hcho_directional_123.csv"), "mean"),
    "lstd": (os.path.join(META_DIR, "lst_anomaly_directional_123.csv"), "lst_anomaly"),
}
dir_data = {}
for name, (path, col) in dir_files.items():
    if os.path.exists(path):
        dir_data[name] = _load_dir_clim_123(path, col)

# SMART_V1
if len(no2_map) > 0 and len(emit_map) > 0:
    all_no2_sec = np.array([no2_map.loc[s, [f"no2_clim_{d}" for d in SECTOR_NAMES]].values
        if s in no2_map.index else np.full(8, np.nan) for s in all_sids])
    all_ntl_sec = np.array([emit_map.loc[s, [f"ntl_clim_{d}" for d in SECTOR_NAMES]].values
        if s in emit_map.index else np.full(8, np.nan) for s in all_sids])
    all_lst_sec = np.array([emit_map.loc[s, [f"lst_anom_clim_{d}" for d in SECTOR_NAMES]].values
        if s in emit_map.index else np.full(8, np.nan) for s in all_sids])
    no2_cen = np.array([no2_map.loc[s, "no2_center"] if s in no2_map.index else np.nan for s in all_sids])
    ntl_cen = np.array([emit_map.loc[s, "ntl_center"] if s in emit_map.index else np.nan for s in all_sids])
    lst_cen = np.array([emit_map.loc[s, "lst_anom_center"] if s in emit_map.index else np.nan for s in all_sids])
    prefix = "him"
    fmf_col = f"{prefix}_fmf_center"
    fmf_cen = np.array([matched_static.loc[s, fmf_col]
        if s in matched_static.index and fmf_col in matched_static.columns else np.nan for s in all_sids])

    def _lohi(sec, cen):
        c = np.concatenate([sec.ravel(), cen])
        return float(np.nanmin(c)), float(np.nanmax(c))
    no2_lo, no2_hi = _lohi(all_no2_sec, no2_cen)
    ntl_lo, ntl_hi = _lohi(all_ntl_sec, ntl_cen)
    lst_lo, lst_hi = _lohi(all_lst_sec, lst_cen)
    def norm01(v, lo, hi):
        if hi - lo < 1e-12: return 0.0
        return float((v - lo) / (hi - lo)) if not np.isnan(v) else 0.0
    smart_sec, smart_cen = {}, {}
    for si, sid in enumerate(all_sids):
        fmf = fmf_cen[si]; fmf = 0.5 if np.isnan(fmf) else fmf
        v1 = np.zeros(8)
        for di in range(8):
            v1[di] = norm01(all_no2_sec[si,di],no2_lo,no2_hi) * \
                     (1+norm01(all_ntl_sec[si,di],ntl_lo,ntl_hi)) * \
                     (1+norm01(all_lst_sec[si,di],lst_lo,lst_hi)) * fmf
        smart_sec[sid] = v1
        smart_cen[sid] = norm01(no2_cen[si],no2_lo,no2_hi) * \
                         (1+norm01(ntl_cen[si],ntl_lo,ntl_hi)) * \
                         (1+norm01(lst_cen[si],lst_lo,lst_hi)) * fmf

    df["smart_v1_center"] = [smart_cen.get(s, 0) for s in stationId_vals]
    sv1_up = np.zeros(len(df))
    for sid in all_sids:
        m = stationId_vals == sid
        if m.any(): sv1_up[np.where(m)[0]] = smart_sec.get(sid, np.zeros(8))[sector_idx[np.where(m)[0]]]
    df["smart_v1_upwind"] = sv1_up
    s1mx = np.array([smart_sec.get(s, np.zeros(8)).max() for s in stationId_vals])
    s1mn = np.array([smart_sec.get(s, np.zeros(8)).min() for s in stationId_vals])
    df["smart_v1_max"] = s1mx
    df["smart_v1_contrast"] = s1mx / (s1mn + 0.001)
    df["smart_v1_upwind_x_VC_inv"] = sv1_up * vc_inv

# Gas standalone directional
for name in ["so2", "co"]:
    if name in dir_data:
        sec_d, cen_d = dir_data[name]
        up = np.zeros(len(df))
        for sid in all_sids:
            m = stationId_vals == sid
            if m.any(): up[np.where(m)[0]] = np.nan_to_num(sec_d.get(sid, np.zeros(8))[sector_idx[np.where(m)[0]]])
        df[f"{name}_upwind"] = up
        df[f"{name}_center_dir"] = df["stationId"].map(
            {s: float(np.nan_to_num(cen_d.get(s, 0), nan=0)) for s in all_sids}).fillna(0)
        if name == "so2":
            df["so2_upwind_x_VC_inv"] = up * vc_inv
            so2_contr = {}
            for sid in all_sids:
                sec = sec_d.get(sid, np.full(8, np.nan)); v = sec[~np.isnan(sec)]
                so2_contr[sid] = float(v.max()/v.mean()) if len(v)>0 and v.mean()>1e-12 else 1.0
            df["so2_contrast"] = df["stationId"].map(so2_contr).fillna(1.0)

if "lstd" in dir_data:
    sec_d, cen_d = dir_data["lstd"]
    lst_up = np.zeros(len(df))
    for sid in all_sids:
        m = stationId_vals == sid
        if m.any(): lst_up[np.where(m)[0]] = np.nan_to_num(sec_d.get(sid, np.zeros(8))[sector_idx[np.where(m)[0]]])
    df["lst_anom_upwind_x_VC_inv"] = lst_up * vc_inv

if "hcho" in dir_data:
    _, hcho_centers = dir_data["hcho"]
    df["hcho_center"] = df["stationId"].map(
        {s: float(np.nan_to_num(hcho_centers.get(s, 0), nan=0)) for s in all_sids}).fillna(0)

# Anomaly interactions
df["so2_anom_x_vc_inv"] = df.get("so2_daily_anom", pd.Series(0, index=df.index)).values * vc_inv
df["co_anom_x_vc_inv"] = df.get("co_daily_anom", pd.Series(0, index=df.index)).values * vc_inv
df["lst_anom_x_vc_inv"] = df.get("lst_day_anom", pd.Series(0, index=df.index)).values * vc_inv
ANOM_INTERACT = ["so2_anom_x_vc_inv", "co_anom_x_vc_inv", "lst_anom_x_vc_inv"]
DAILY_ANOM_ALL = ANOM_RAW + ANOM_INTERACT

# Fire
fire_path = os.path.join(META_DIR, "fire_counts_directional_123.csv")
if os.path.exists(fire_path):
    fire_raw = pd.read_csv(fire_path, dtype={"stationId": str}).rename(columns={"mean": "fire_val"})
    fire_lookup = {}
    for sid in all_sids:
        sf = fire_raw[fire_raw["stationId"] == sid]; lk = {}
        for di, d in enumerate(SECTOR_NAMES):
            for szn in ["DJF","MAM","JJA","SON"]:
                v = sf[(sf["direction"]==d)&(sf["season"]==szn)]["fire_val"]
                lk[(di,szn)] = float(v.mean()) if len(v)>0 else 0.0
        fire_lookup[sid] = lk
    season_vals = np.array([SEASON_MAP[m] for m in df["month"].values])
    fire_up = np.zeros(len(df))
    for sid in all_sids:
        m = stationId_vals == sid
        if not m.any(): continue
        lk = fire_lookup.get(sid, {})
        for i in np.where(m)[0]: fire_up[i] = lk.get((sector_idx[i], season_vals[i]), 0)
    df["fire_upwind"] = fire_up

# Outer AOD physics
aot_out = df["AOT_outer_mean"].fillna(0).values
aot_ctr = df.get("AOT_ffill_48h", pd.Series(0, index=df.index)).fillna(0).values
pblh = df["PBLH"].fillna(200).values
rh_frac = (df["Humidity_final"]/100).clip(0, 0.95).values
f_rh = 1.0 / (1.0 - rh_frac)
hrs_since = df.get("hours_since_valid_AOT", pd.Series(999, index=df.index)).fillna(999).values
df["aod_outer_surface"] = aot_out / (pblh + 100)
df["aod_outer_pm25"] = aot_out / (pblh + 100) / f_rh
df["aod_outer_x_VC_inv"] = aot_out * vc_inv
df["aod_outer_gradient"] = aot_out - aot_ctr
is_real = (hrs_since == 0) & (aot_ctr > 0)
df["_or"] = np.where(is_real, aot_out, np.nan); df["_ir"] = is_real
do = df.groupby(["stationId","date"]).agg(aod_outer_day_mean=("_or","mean"),_dc=("_ir","sum")).reset_index()
do.loc[do["_dc"]==0, "aod_outer_day_mean"] = np.nan
df = df.merge(do[["stationId","date","aod_outer_day_mean"]], on=["stationId","date"], how="left")
df.drop(columns=["_or","_ir"], inplace=True)
OUTER_ALL_EXTRA = ["aod_outer_surface","aod_outer_pm25","aod_outer_x_VC_inv","aod_outer_gradient","aod_outer_day_mean"]

# Weather persistence
df['PBLH_min_24h'] = df.groupby('stationId')['PBLH'].transform(lambda x: x.rolling(24,min_periods=1).min())
df['VC_min_24h'] = df.groupby('stationId')['VC'].transform(lambda x: x.rolling(24,min_periods=1).min())
stag = ((df['PBLH']<500)&(df['WS_local'].fillna(0)<2)).astype(float)
df['stagnation_hours_12h'] = stag.groupby(df['stationId']).rolling(12,min_periods=1).sum().reset_index(level=0,drop=True)
df['temp_diurnal_anomaly'] = df['Temperature_final'] - df.groupby(
    ['stationId',df['ts'].dt.month,df['ts'].dt.hour])['Temperature_final'].transform('mean')
WEATHER_PERSIST = ['PBLH_min_24h','VC_min_24h','stagnation_hours_12h','temp_diurnal_anomaly']

# Fill NaN for static features
fill_cols = (NO2_STATIC_COLS + NO2_SECTOR_COLS +
    ["ntl_center","smart_v1_center","smart_v1_upwind","smart_v1_max","smart_v1_contrast",
     "smart_v1_upwind_x_VC_inv","so2_upwind","so2_center_dir","so2_contrast",
     "co_upwind","co_center_dir","hcho_center","so2_upwind_x_VC_inv",
     "lst_anom_upwind_x_VC_inv","fire_upwind"])
for c in set(fill_cols):
    if c in df.columns: df[c] = df[c].fillna(0)

print(f"  Enrichment done ({time.time()-t0:.0f}s)")

# ── Feature set (NO RFSI) ───────────────────────────────────────────────────
MET_CORE = ["PBLH","VC","wind_u","wind_v","WS_local","Temperature_final","Humidity_final",
            "Pressure_final","dT_6h","dRH_6h","rain_days_7d","rain_sum_48h",
            "consecutive_dry_days","hrs_since_rain","RH_factor"]
TEMPORAL = ["hour_sin","hour_cos","month_sin","month_cos","day_of_year_cos","day_of_year_sin","dow_is_weekend"]
AOD_CORE = ["AOT_ffill_48h","AOT_outer_mean","AE","RF","hours_since_valid_AOT"] + SOURCE_AOD_FEATURES
AOD_EXTENDED = AOD_CORE + OUTER_ALL_EXTRA + ["RF_center","RF_mean","SSA_center","SSA_mean",
               "AOT_fine","AOT_grad_mag","AOT_local_vs_regional"]
SMART_EMISSION = ["smart_v1_center","smart_v1_upwind","smart_v1_max","smart_v1_contrast","smart_v1_upwind_x_VC_inv"]
GAS_STANDALONE = ["so2_upwind","so2_center_dir","so2_contrast","co_upwind","co_center_dir","hcho_center",
                  "so2_upwind_x_VC_inv","lst_anom_upwind_x_VC_inv"]
EMISSION_STATIC = NO2_STATIC_COLS + NO2_SECTOR_COLS + ["ntl_center"] + NTL_SECTOR_COLS + ["lst_anom_center"] + LST_SECTOR_COLS
BUILDING = ["building_area_1km","building_count_3km"]
TERRAIN = ["elevation_m","slope_deg"]

FEAT_TEMPORAL = sorted(set(f for f in
    MET_CORE + WEATHER_PERSIST + TEMPORAL + TERRAIN + BUILDING +
    AOD_EXTENDED + SMART_EMISSION + GAS_STANDALONE + DAILY_ANOM_ALL + EMISSION_STATIC + ["fire_upwind"]
    if f in df.columns))

print(f"  Feature set: {len(FEAT_TEMPORAL)} temporal features (no RFSI)")

# ── Station cleaning ─────────────────────────────────────────────────────────
def clean_station(g):
    g = g.sort_values("ts").copy()
    pm = pd.to_numeric(g["PM2.5"], errors="coerce")
    valid = pm.between(0.1, 250.0)
    rounded = pm.round(1)
    run_id = rounded.ne(rounded.shift()).cumsum()
    run_len = run_id.map(run_id.value_counts())
    flatline = valid & run_len.ge(24)
    out = g.loc[valid & ~flatline].copy()
    days = out["ts"].dt.date.nunique() if len(out) else 0
    return out, len(out) >= 500 and days >= 20

# ── Run evaluation ───────────────────────────────────────────────────────────
def run_within_station(sids, label):
    print(f"\n{'='*80}")
    print(f"  {label}: {len(sids)} stations")
    print(f"{'='*80}")
    rows = []
    for sid in sorted(sids):
        g = df[df["stationId"] == sid]
        g, ok = clean_station(g)
        if not ok:
            print(f"  SKIP {sid[-8:]}: {len(g)} rows, too few")
            continue
        y = g["PM2.5"].values.astype(float)
        X = g[FEAT_TEMPORAL].values.astype(float)
        pm_mean = float(np.mean(y))
        t = tier(pm_mean)
        wk = kfold_r2(X, y)
        wt = temporal_r2(X, y)
        rows.append(dict(sid=sid, tier=t, pm=pm_mean, n=len(g),
                         wk_exog=wk, wt_exog=wt,
                         is_thesis=sid in THESIS_SIDS))
        print(f"  {t} pm={pm_mean:5.1f} n={len(g):6d}  "
              f"kfold={wk:+.3f}  temporal={wt:+.3f}  {'KK' if sid in THESIS_SIDS else 'LCS'} {sid[-8:]}")

    res = pd.DataFrame(rows)
    if len(res) == 0:
        print("  No valid stations!")
        return res

    TIERS = ["t0","t1","t2","t3"]
    print(f"\n  PER-TIER MEAN:")
    print(f"  {'tier':<5}{'n':>3} | {'kfold_exog':>12} {'temporal_exog':>14} | {'pm_mean':>8}")
    print(f"  {'-'*55}")
    for t in TIERS:
        s = res[res["tier"]==t]
        if not len(s): continue
        print(f"  {t:<5}{len(s):>3} | {s['wk_exog'].mean():>+12.3f} {s['wt_exog'].mean():>+14.3f} | {s['pm'].mean():>8.1f}")
    print(f"  {'-'*55}")
    print(f"  {'ALL':<5}{len(res):>3} | {res['wk_exog'].mean():>+12.3f} {res['wt_exog'].mean():>+14.3f} | {res['pm'].mean():>8.1f}")
    print(f"  median:     | {res['wk_exog'].median():>+12.3f} {res['wt_exog'].median():>+14.3f} |")
    pos = (res['wk_exog'] > 0).sum()
    print(f"  positive R²: {pos}/{len(res)}")

    # Stations with high R²
    high = res[res['wk_exog'] >= 0.8].sort_values('wk_exog', ascending=False)
    if len(high):
        print(f"\n  Stations with KFold R² >= 0.8: {len(high)}")
        for _, r in high.iterrows():
            print(f"    {r['tier']} pm={r['pm']:5.1f} kf={r['wk_exog']:+.3f} tp={r['wt_exog']:+.3f} "
                  f"{'KK' if r['is_thesis'] else 'LCS'} {r['sid'][-8:]}")

    return res


# Run A: all 40 thesis stations
thesis_sids = sorted(THESIS_SIDS & set(all_sids))
res_thesis = run_within_station(thesis_sids, "A) THESIS STATIONS (KK, all 40)")

# Run B: All stations (thesis + LCS pass). Keep all 40 — the stronger v4 mask
# cleans the 3 ex-broken sensors row-wise rather than dropping the stations.
all_valid = sorted((THESIS_SIDS | PASS_IDS) & set(all_sids))
res_all = run_within_station(all_valid, "B) ALL STATIONS (thesis + LCS pass, all 40 kept)")

# Save
out_path = "analysis/thesis_experiments/within_station_predictability_v4.csv"
res_all["group"] = res_all["sid"].apply(lambda s: "thesis" if s in THESIS_SIDS else "lcs")
res_all.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# Summary comparison
if len(res_thesis) and len(res_all):
    lcs_only = res_all[~res_all["sid"].isin(THESIS_SIDS)]
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Group':<20} {'n':>4} {'kfold_mean':>12} {'kfold_median':>14} {'temporal_mean':>14}")
    print(f"  {'-'*70}")
    for lbl, r in [("Thesis (KK)", res_thesis), ("LCS only", lcs_only), ("All combined", res_all)]:
        if len(r):
            print(f"  {lbl:<20} {len(r):>4} {r['wk_exog'].mean():>+12.3f} "
                  f"{r['wk_exog'].median():>+14.3f} {r['wt_exog'].mean():>+14.3f}")

print(f"\nDONE — {time.time()-t0:.0f}s")
