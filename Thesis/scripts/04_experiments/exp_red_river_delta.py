"""
Experiment: Red River Delta PM2.5 Prediction v5h.

v5h: Focus on 14 KK stations in the Red River Delta (satellite cluster C2).
LOSO within 14 delta stations; regional leave-one-out mean as base margin.
Thesis result: can we predict PM2.5 variation within a known polluted region?

Station quality filter from v5f retained (3 broken sensors removed).
Physics features from v5d retained.

Configs:
  - delta_bm: regional LOO mean BM + obs+physics features
  - delta_rfsi: regional LOO mean BM + obs+physics + RFSI
  - oracle_bm: reference ceiling (oracle BM + all features + RFSI)

Output: analysis/thesis_experiments/delta_v1_test.csv
"""

import argparse, io, sys, os, warnings, time, glob, zipfile, unicodedata
from unicodedata import normalize as _unorm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler  # kept for potential future use
from sklearn.cluster import KMeans  # used only for initial cluster identification

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = args.data_dir or REPO_DIR
OUT_DIR = os.path.join(REPO_DIR, "analysis", "thesis_experiments")
META_DIR = os.path.join(DATA_DIR, "data", "stations", "metadata")
os.makedirs(OUT_DIR, exist_ok=True)

QC_DIR = os.path.join(REPO_DIR, "Thesis", "scripts", "02_processing")
if QC_DIR not in sys.path:
    sys.path.insert(0, QC_DIR)
from pm25_qc import pm25_quality_masks

K_NN = 5
ROLL_DAYS = 30  # Window for temporal satellite statistics

SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}

XGB_BASE = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=50,
    reg_alpha=0.1, reg_lambda=10.0, tree_method="hist",
    device="cuda", random_state=42, n_jobs=-1,
)

CONFIGS = ["delta_bm", "delta_rfsi", "oracle_bm"]
POST_HOC = []
FOLD_ABBREV = {"delta_bm": "DLT", "delta_rfsi": "D+R", "oracle_bm": "ORC"}

TIER_NAMES = ["t0", "t1", "t2", "t3"]
RFSI_FEATURES = ["PM25_nn_idw", "PM25_nn1", "PM25_nn2", "PM25_nn3"]


# =============================================================================
#  HELPERS
# =============================================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def assign_tier(mean_pm):
    if mean_pm < 10:
        return "t0"
    elif mean_pm < 20:
        return "t1"
    elif mean_pm < 35:
        return "t2"
    return "t3"


# =============================================================================
#  LOAD DATA
# =============================================================================
print("=" * 80)
print("RED RIVER DELTA PM2.5 PREDICTION v5h")
print("=" * 80)

t0_start = time.time()

df = pd.read_csv(os.path.join(DATA_DIR, "data/merged/unified_thesis_v4.csv"),
                 dtype={"stationId": str})  # v4 = definitive (all 40 stations, stronger mask)
# v4 holds all 121 stations; restrict to the 40 thesis stations (v2 was naturally 40).
_thesis40 = set(pd.read_csv(os.path.join(DATA_DIR,
    "analysis/thesis_audit/station_selection_final.csv"),
    dtype={"stationId": str})["stationId"])
df = df[df["stationId"].isin(_thesis40)].copy()
df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df["ts"])
df["month"] = df["ts"].dt.month
df["date"] = df["ts"].dt.date
print(f"Loaded: {len(df):,} rows, {df['stationId'].nunique()} stations "
      f"({time.time()-t0_start:.1f}s)")

# --- Station quality filter (sensor reliability) ---
# Based on investigation of KK low-cost sensor failure modes:
#   REMOVE: broken/degraded PMS5003 sensors with zero-floors and under-reporting
#   FLAG: placement-biased or source-impacted (kept but noted)
STATIONS_REMOVE = {
    "31616865099255512061948816121",  # Da Nang Pham Hung: 12% zeros, 986 flat runs, mean=6.2 vs city ~20
    "30991938797551443885460120607",  # Soc Trang: 3.7% zeros, 206 flat runs, mean=6.7 vs GHAP ~15
    "29098319146067624969113973428",  # Tra Vinh Dong Hai: mean=5.7 vs GHAP ~14
}
STATIONS_FLAG = {
    "28602897318711027016899843809",  # Quang Ninh Nha may tuyen than: placement inside water-sprayed compound
    "31651502905690497791503780869",  # Thai Nguyen: source-impacted (TISCO steelworks)
}
n_before = len(df)
n_stn_before = df["stationId"].nunique()
df = df[~df["stationId"].isin(STATIONS_REMOVE)].reset_index(drop=True)
print(f"Station quality filter: removed {n_before - len(df):,} rows "
      f"from {len(STATIONS_REMOVE)} broken sensors → "
      f"{n_stn_before}→{df['stationId'].nunique()} stations")

# --- Delta focus: keep only geographic Red River Delta stations ---
DELTA_SIDS = {
    "28560877461938780203765592307",  # Hà Nội 556 Nguyễn Văn Cừ
    "28916504310234840885489983032",  # Bắc Ninh Thuận Thành
    "28916774462801800655608897080",  # Bắc Ninh Xuân Lâm
    "29196010501691076420299004774",  # Bắc Ninh Suối Hoa
    "29196021237696127337075448678",  # Bắc Ninh Cao Đức
    "29203727697074312726675247132",  # Thái Bình Thái Thọ
    "31388868531618872623864101418",  # Hải Dương
    "31388883344354363840031242796",  # Hà Nam
    "31390903576425084107499649578",  # Hà Nội ĐHBK
    "31390908889087377344742439468",  # Hà Nội Nhân Chính
    "31390921469766835629621918251",  # Hưng Yên
    "31390957404024291365397346858",  # Thái Bình TP
}
n_before_delta = len(df)
df = df[df["stationId"].isin(DELTA_SIDS)].reset_index(drop=True)
print(f"Delta focus: {n_before_delta:,} → {len(df):,} rows "
      f"({df['stationId'].nunique()} delta stations)")

meta = pd.read_csv(os.path.join(DATA_DIR,
                    "analysis/thesis_audit/station_selection_final.csv"),
                    dtype={"stationId": str})
sid_name = dict(zip(meta["stationId"], meta["station_name"]))
sid_region = dict(zip(meta["stationId"], meta["region"]))
sid_lat = dict(zip(meta["stationId"], meta["lat"]))
sid_lon = dict(zip(meta["stationId"], meta["lon"]))
station_ids = sorted(df["stationId"].unique())
n_stn = len(station_ids)

TARGET = "PM2.5"
y_all = df[TARGET].values
stationId_vals = df["stationId"].values

# --- PM2.5 quality filter ---
qc_masks = pm25_quality_masks(df)
n_filtered = int(qc_masks.any(axis=1).sum())
df.loc[qc_masks.any(axis=1), 'PM2.5'] = np.nan
print(
    "PM2.5 quality filter: "
    f"{n_filtered} rows ({100*n_filtered/len(df):.1f}%) "
    f"[zero/neg={int(qc_masks['zero_or_negative'].sum())}, "
    f"flat={int(qc_masks['flatline'].sum())}, "
    f"stuck_low={int(qc_masks['stuck_low'].sum())}, "
    f"high={int(qc_masks['too_high'].sum())}]"
)
y_all = df['PM2.5'].values
y_log = np.log1p(np.nan_to_num(y_all, nan=0.0))

station_pm_means = df.groupby("stationId")["PM2.5"].mean()
sid_tier = {s: assign_tier(station_pm_means[s]) for s in station_ids}
for t in TIER_NAMES:
    sids_t = [s for s in station_ids if sid_tier[s] == t]
    print(f"  {t}: {len(sids_t)} stations")

global_pm_mean = float(station_pm_means.mean())
bm_global = np.log1p(global_pm_mean)

# =============================================================================
#  LOAD GEE DAILY EXPORT + COMPUTE TEMPORAL STATISTICS
# =============================================================================
print(f"\n  Loading GEE daily export...")

zip_candidates = sorted(glob.glob(os.path.join(DATA_DIR, "data", "gee_exports", "last-*.zip")))
if not zip_candidates:
    print("ERROR: No GEE export zip found"); sys.exit(1)
zip_path = zip_candidates[-1]

all_sat = []
with zipfile.ZipFile(zip_path) as z:
    for name in sorted(z.namelist()):
        if not name.endswith(".csv"):
            continue
        with z.open(name) as f:
            chunk = pd.read_csv(f, dtype={"stationId": str})
        all_sat.append(chunk)

sat_long = pd.concat(all_sat, ignore_index=True)
sat_wide = sat_long.pivot_table(
    index=["stationId", "date"], columns="variable",
    values="mean", aggfunc="first"
).reset_index()
sat_wide.columns.name = None
sat_wide["date"] = pd.to_datetime(sat_wide["date"])

model_sids = set(station_ids)
sat_wide = sat_wide[sat_wide["stationId"].isin(model_sids)].copy()
sat_wide["month"] = sat_wide["date"].dt.month

CLIM_COLS = ["NO2", "SO2", "CO", "HCHO"]
for c in CLIM_COLS:
    if c not in sat_wide.columns:
        sat_wide[c] = np.nan

# Daily anomalies
clim = sat_wide.groupby(["stationId", "month"])[CLIM_COLS].transform("mean")
sat_wide["so2_daily_anom"] = sat_wide["SO2"] - clim["SO2"]
sat_wide["co_daily_anom"] = sat_wide["CO"] - clim["CO"]
sat_wide["no2_daily_anom"] = sat_wide["NO2"] - clim["NO2"]
sat_wide["hcho_daily_anom"] = sat_wide["HCHO"] - clim["HCHO"]

ANOM_COLS = ["so2_daily_anom", "co_daily_anom", "no2_daily_anom",
             "hcho_daily_anom"]

# --- Rolling temporal statistics of TROPOMI (30-day window) ---
print(f"  Computing {ROLL_DAYS}-day rolling TROPOMI statistics...")
sat_wide = sat_wide.sort_values(["stationId", "date"])

tropomi_roll_feats = []
for sid, grp in sat_wide.groupby("stationId"):
    grp = grp.set_index("date").sort_index()
    row_dates = grp.index

    feats_df = pd.DataFrame(index=row_dates)
    feats_df["stationId"] = sid

    # HCHO rolling stats
    hcho = grp["HCHO"]
    feats_df["hcho_30d_mean"] = hcho.rolling(f"{ROLL_DAYS}D", min_periods=5).mean()
    feats_df["hcho_30d_p90"] = hcho.rolling(f"{ROLL_DAYS}D", min_periods=5).quantile(0.9)
    hcho_std = hcho.rolling(f"{ROLL_DAYS}D", min_periods=5).std()
    hcho_mean = hcho.rolling(f"{ROLL_DAYS}D", min_periods=5).mean()
    feats_df["hcho_30d_cv"] = hcho_std / (hcho_mean.abs() + 1e-12)

    # CO rolling stats
    co = grp["CO"]
    feats_df["co_30d_mean"] = co.rolling(f"{ROLL_DAYS}D", min_periods=5).mean()
    feats_df["co_30d_std"] = co.rolling(f"{ROLL_DAYS}D", min_periods=5).std()
    co_q75 = co.rolling(f"{ROLL_DAYS}D", min_periods=5).quantile(0.75)
    co_q25 = co.rolling(f"{ROLL_DAYS}D", min_periods=5).quantile(0.25)
    feats_df["co_30d_iqr"] = co_q75 - co_q25

    feats_df = feats_df.reset_index().rename(columns={"date": "date_trop"})
    tropomi_roll_feats.append(feats_df)

tropomi_roll = pd.concat(tropomi_roll_feats, ignore_index=True)
tropomi_roll["date_merge"] = tropomi_roll["date_trop"].dt.date

TROPOMI_ROLL_COLS = ["hcho_30d_mean", "hcho_30d_p90", "hcho_30d_cv",
                     "co_30d_mean", "co_30d_std", "co_30d_iqr"]

print(f"  TROPOMI rolling features: {len(TROPOMI_ROLL_COLS)} features")

# Merge daily anomalies
sat_wide["date_merge"] = sat_wide["date"].dt.date
merge_cols = ["stationId", "date_merge"] + ANOM_COLS
df = df.merge(sat_wide[merge_cols], left_on=["stationId", "date"],
              right_on=["stationId", "date_merge"], how="left")
df.drop(columns=["date_merge"], inplace=True)

# Merge TROPOMI rolling features
df = df.merge(tropomi_roll[["stationId", "date_merge"] + TROPOMI_ROLL_COLS],
              left_on=["stationId", "date"],
              right_on=["stationId", "date_merge"], how="left")
df.drop(columns=["date_merge"], inplace=True)

n_total = len(df)
n_with_hcho = df["hcho_daily_anom"].notna().sum()
n_with_roll = df["hcho_30d_mean"].notna().sum()
print(f"  GEE merged: HCHO anom {n_with_hcho:,}/{n_total:,}, "
      f"rolling stats {n_with_roll:,}/{n_total:,}")

# =============================================================================
#  LOAD MODIS TEMPORAL FEATURES + ROLLING STATISTICS
# =============================================================================
print("  Loading MODIS temporal features...")
modis_path = os.path.join(META_DIR, "modis_temporal_features.csv")
modis_tf = pd.read_csv(modis_path, dtype={"stationId": str})
modis_tf["date"] = pd.to_datetime(modis_tf["date"])

# Compute rolling AOD statistics (30-day)
print(f"  Computing {ROLL_DAYS}-day rolling MODIS AOD statistics...")
modis_tf = modis_tf.sort_values(["stationId", "date"])

modis_roll_feats = []
for sid, grp in modis_tf.groupby("stationId"):
    grp = grp.set_index("date").sort_index()
    aod = grp["modis_aod_7d"]

    feats_df = pd.DataFrame(index=grp.index)
    feats_df["stationId"] = sid
    feats_df["aod_30d_mean"] = aod.rolling(f"{ROLL_DAYS}D", min_periods=10).mean()
    feats_df["aod_30d_std"] = aod.rolling(f"{ROLL_DAYS}D", min_periods=10).std()
    aod_q75 = aod.rolling(f"{ROLL_DAYS}D", min_periods=10).quantile(0.75)
    aod_q25 = aod.rolling(f"{ROLL_DAYS}D", min_periods=10).quantile(0.25)
    feats_df["aod_30d_iqr"] = aod_q75 - aod_q25
    feats_df["aod_30d_p90"] = aod.rolling(f"{ROLL_DAYS}D", min_periods=10).quantile(0.9)
    aod_std = feats_df["aod_30d_std"]
    aod_mean = feats_df["aod_30d_mean"]
    feats_df["aod_30d_cv"] = aod_std / (aod_mean + 1e-9)

    feats_df = feats_df.reset_index()
    modis_roll_feats.append(feats_df)

modis_roll = pd.concat(modis_roll_feats, ignore_index=True)
modis_roll["date_m"] = modis_roll["date"].dt.date

MODIS_ROLL_COLS = ["aod_30d_mean", "aod_30d_std", "aod_30d_iqr",
                   "aod_30d_p90", "aod_30d_cv"]

# Merge MODIS base features
modis_tf["date_m"] = modis_tf["date"].dt.date
df = df.merge(modis_tf[["stationId", "date_m", "modis_aod_7d", "modis_fine_aod_7d"]].rename(
    columns={"date_m": "date"}), on=["stationId", "date"], how="left")

# Merge MODIS rolling features
df = df.merge(modis_roll[["stationId", "date_m"] + MODIS_ROLL_COLS],
              left_on=["stationId", "date"],
              right_on=["stationId", "date_m"], how="left")
df.drop(columns=["date_m"], inplace=True)

n_modis = df["modis_aod_7d"].notna().sum()
n_modis_roll = df["aod_30d_mean"].notna().sum()
print(f"  MODIS merged: {n_modis:,}/{n_total:,} daily, "
      f"{n_modis_roll:,}/{n_total:,} rolling")

# --- Station-level aggregates of rolling satellite features ---
print("  Computing station-level satellite regime fingerprints...")
ALL_ROLL_COLS = MODIS_ROLL_COLS + TROPOMI_ROLL_COLS
STN_AGG_COLS = []
for col in ALL_ROLL_COLS:
    stn_col = col + "_stn"
    df[stn_col] = df.groupby("stationId")[col].transform("mean")
    STN_AGG_COLS.append(stn_col)
n_stn_agg = df[STN_AGG_COLS[0]].notna().sum()
print(f"  Station-level regime features: {len(STN_AGG_COLS)} "
      f"({n_stn_agg:,}/{n_total:,} non-NaN)")

# =============================================================================
#  LOAD ACAG / GHAP (as features only, not base margin)
# =============================================================================
print("  Loading ACAG/GHAP as features...")
acag_path = os.path.join(DATA_DIR, "data", "acag", "acag_station_climatology.csv")
acag_df = pd.read_csv(acag_path, dtype={"stationId": str})
acag_annual = acag_df.set_index("stationId")["ACAG_annual_mean"].to_dict()
df["acag_annual"] = df["stationId"].map(acag_annual).fillna(global_pm_mean)

ghap_zip_candidates = sorted(glob.glob(os.path.join(DATA_DIR, "pm25*.zip")))
ghap_annual = {}
ghap_monthly = {}
if ghap_zip_candidates:
    with zipfile.ZipFile(ghap_zip_candidates[-1]) as z:
        for name in z.namelist():
            if "ghap_monthly_climatology" in name:
                with z.open(name) as f:
                    ghap_mc = pd.read_csv(f, dtype={"stationId": str})
                for _, row in ghap_mc.iterrows():
                    key = (row["stationId"], int(row["month"]))
                    if not np.isnan(row["mean"]):
                        ghap_monthly[key] = row["mean"]
                ghap_annual = ghap_mc.groupby("stationId")["mean"].mean().to_dict()
                break
    print(f"  GHAP: {len(ghap_annual)} stations")

df["ghap_annual"] = df["stationId"].map(
    lambda s: ghap_annual.get(s, global_pm_mean))

# =============================================================================
#  BUILDING DENSITY
# =============================================================================
bld_path = os.path.join(META_DIR, "station_building_density.csv")
bld = pd.read_csv(bld_path, dtype={"stationId": str})
bld_map = bld.set_index("stationId")[["building_area_1km"]]
df = df.merge(bld_map, left_on="stationId", right_index=True, how="left")
df["building_area_1km"] = df["building_area_1km"].fillna(0)

# =============================================================================
#  RFSI
# =============================================================================
print("  RFSI setup...")

coords = {s: (sid_lat[s], sid_lon[s]) for s in station_ids}
sid_to_idx = {s: i for i, s in enumerate(station_ids)}
dist_full = np.zeros((n_stn, n_stn))
for i in range(n_stn):
    for j in range(i + 1, n_stn):
        d = haversine(*coords[station_ids[i]], *coords[station_ids[j]])
        dist_full[i, j] = d
        dist_full[j, i] = d

neighbor_order = {}
for i in range(n_stn):
    neighbor_order[i] = sorted(
        [(j, dist_full[i, j]) for j in range(n_stn) if j != i],
        key=lambda x: x[1])

pm25_wide = df.pivot_table(index="ts", columns="stationId",
                           values="PM2.5", aggfunc="first")
pm25_mat = pm25_wide.values
sid_cols = list(pm25_wide.columns)
sid_to_col = {s: i for i, s in enumerate(sid_cols)}
ts_to_row = pd.Series(range(len(pm25_wide)), index=pm25_wide.index)
df["ts_row"] = df["ts"].map(ts_to_row).astype(int).values


def compute_rfsi(exclude_sid=None, K=5):
    n = len(df)
    pm_nn = np.full((n, K), np.nan)
    d_nn = np.full((n, K), np.nan)
    excl = sid_to_idx.get(exclude_sid) if exclude_sid else None
    ts_row_vals = df["ts_row"].values
    for sid in station_ids:
        si = sid_to_idx[sid]
        mask = stationId_vals == sid
        if not mask.any():
            continue
        ri = np.where(mask)[0]
        tr = ts_row_vals[ri]
        cands = [(j, d) for j, d in neighbor_order[si]
                 if excl is None or j != excl]
        if not cands:
            continue
        ccols = np.array([sid_to_col[station_ids[j]] for j, _ in cands])
        cdists = np.array([d for _, d in cands])
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
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d_nn
        pm_idw = np.nansum(pm_nn * w, axis=1) / np.nansum(w, axis=1)
    return {"PM25_nn_idw": pm_idw, "PM25_nn1": pm_nn[:, 0],
            "PM25_nn2": pm_nn[:, 1], "PM25_nn3": pm_nn[:, 2]}


# =============================================================================
#  LOAD SATELLITE FEATURES + BUILD SMART_V1
# =============================================================================
print("  Loading satellite feature CSVs...")

no2_feat = pd.read_csv(os.path.join(META_DIR, "station_no2_features.csv"),
                       dtype={"stationId": str})
emit_feat = pd.read_csv(os.path.join(META_DIR, "station_emission_features.csv"),
                        dtype={"stationId": str})
new_sat_feat = pd.read_csv(os.path.join(META_DIR,
                            "station_all_satellite_features.csv"),
                           dtype={"stationId": str})

no2_map = no2_feat.set_index("stationId")
emit_map = emit_feat.set_index("stationId")
new_sat_map = new_sat_feat.set_index("stationId")

NO2_SECTOR_COLS = [f"no2_clim_{d}" for d in SECTOR_NAMES]
df = df.merge(no2_map[["no2_center"] + NO2_SECTOR_COLS],
              left_on="stationId", right_index=True, how="left")

merge_emit = (["ntl_center"] + [f"ntl_clim_{d}" for d in SECTOR_NAMES] +
              ["lst_anom_center"] + [f"lst_anom_clim_{d}" for d in SECTOR_NAMES])
merge_emit = [c for c in merge_emit if c in emit_map.columns]
df = df.merge(emit_map[merge_emit],
              left_on="stationId", right_index=True, how="left")

merge_new = ["fmf_center"]
merge_new = [c for c in merge_new if c in new_sat_map.columns]
df = df.merge(new_sat_map[merge_new],
              left_on="stationId", right_index=True, how="left")

# --- Directional climatology ---
print("  Loading directional climatology...")


def _norm_tok(s):
    return unicodedata.normalize("NFKD", str(s)).encode(
        "ascii", "ignore").decode().lower()


def _tokenize(s):
    return set(_norm_tok(s).replace("-", " ").replace(",", " ").split())


no2_clim_csv = os.path.join(META_DIR, "no2_directional_clim.csv")
no2_dir_raw = pd.read_csv(no2_clim_csv, dtype={"stationId": str})
no2_dir_names = no2_dir_raw.groupby("stationId")["name"].first()

id_map = {}
for short_id, no2_name in no2_dir_names.items():
    no2_tokens = _tokenize(no2_name)
    best_score, best_long_id = -1, None
    for _, row in meta.iterrows():
        meta_tokens = _tokenize(row["station_name"])
        score = len(no2_tokens & meta_tokens)
        if score > best_score:
            best_score = score
            best_long_id = row["stationId"]
    id_map[short_id] = best_long_id


def _load_dir_clim(csv_path, value_col="mean"):
    raw = pd.read_csv(csv_path, dtype={"stationId": str})
    raw["stationId"] = raw["stationId"].map(id_map)
    raw = raw.dropna(subset=["stationId"])
    clim_df = raw.groupby(["stationId", "direction"])[value_col].mean() \
                  .reset_index()
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


def _resolve_path(filename):
    p = os.path.join(META_DIR, filename)
    if not os.path.exists(p):
        p = os.path.join(DATA_DIR, filename)
    return p


station_so2_sectors, _ = _load_dir_clim(
    _resolve_path("tropomi_so2_directional.csv"), "mean")
station_co_sectors, _ = _load_dir_clim(
    _resolve_path("tropomi_co_directional.csv"), "mean")

# --- Build smart_v1 ---
print("  Building smart_v1 + sector features...")

station_no2_sectors, station_ntl_sectors, station_lst_sectors = {}, {}, {}
for sid in station_ids:
    if sid in no2_map.index:
        station_no2_sectors[sid] = np.array(
            [no2_map.loc[sid, f"no2_clim_{d}"] for d in SECTOR_NAMES], dtype=float)
    else:
        station_no2_sectors[sid] = np.full(8, np.nan)
    if sid in emit_map.index:
        station_ntl_sectors[sid] = np.array(
            [emit_map.loc[sid, f"ntl_clim_{d}"] for d in SECTOR_NAMES], dtype=float)
        station_lst_sectors[sid] = np.array(
            [emit_map.loc[sid, f"lst_anom_clim_{d}"] for d in SECTOR_NAMES], dtype=float)
    else:
        station_ntl_sectors[sid] = np.full(8, np.nan)
        station_lst_sectors[sid] = np.full(8, np.nan)

wd_from = np.degrees(np.arctan2(-df["wind_u"].values, -df["wind_v"].values)) % 360
sector_idx = ((wd_from + 22.5) / 45).astype(int) % 8
month_vals = df["month"].values
season_vals = np.array([SEASON_MAP[m] for m in month_vals])

all_no2_sec = np.array([station_no2_sectors[s] for s in station_ids])
all_ntl_sec = np.array([station_ntl_sectors[s] for s in station_ids])
all_lst_sec = np.array([station_lst_sectors[s] for s in station_ids])

no2_center_all = np.array([
    no2_map.loc[s, "no2_center"] if s in no2_map.index else np.nan
    for s in station_ids])
ntl_center_all = np.array([
    emit_map.loc[s, "ntl_center"] if s in emit_map.index else np.nan
    for s in station_ids])
lst_center_all = np.array([
    emit_map.loc[s, "lst_anom_center"] if s in emit_map.index else np.nan
    for s in station_ids])
fmf_center_all = np.array([
    new_sat_map.loc[s, "fmf_center"] if s in new_sat_map.index else np.nan
    for s in station_ids])


def _lohi(sec_arr, center_arr):
    combined = np.concatenate([sec_arr.ravel(), center_arr])
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

# --- SO2 / CO upwind ---
so2_upwind_vals = np.zeros(len(df))
co_upwind_vals = np.zeros(len(df))
for sid in station_ids:
    mask = stationId_vals == sid
    if not mask.any():
        continue
    idx = np.where(mask)[0]
    so2_upwind_vals[idx] = station_so2_sectors[sid][sector_idx[idx]]
    co_upwind_vals[idx] = station_co_sectors[sid][sector_idx[idx]]

so2_upwind_vals = np.nan_to_num(so2_upwind_vals, nan=0.0)
co_upwind_vals = np.nan_to_num(co_upwind_vals, nan=0.0)
df["so2_upwind"] = so2_upwind_vals
df["co_upwind"] = co_upwind_vals

# --- Fire upwind ---
fire_csv_path = os.path.join(META_DIR, "fire_counts_directional.csv")
if not os.path.exists(fire_csv_path):
    fire_csv_path = os.path.join(DATA_DIR, "fire_counts_directional.csv")

fire_raw = pd.read_csv(fire_csv_path, dtype={"stationId": str})
fire_raw = fire_raw.rename(columns={"mean": "fire_val"})
fire_raw["stationId"] = fire_raw["stationId"].map(id_map)
fire_raw = fire_raw.dropna(subset=["stationId"])

station_fire_dir_season = {}
for sid in station_ids:
    sf = fire_raw[fire_raw["stationId"] == sid]
    lookup = {}
    for di, d in enumerate(SECTOR_NAMES):
        for szn in ["DJF", "MAM", "JJA", "SON"]:
            vals = sf[(sf["direction"] == d) & (sf["season"] == szn)]["fire_val"]
            lookup[(di, szn)] = float(vals.mean()) if len(vals) > 0 else 0.0
    station_fire_dir_season[sid] = lookup

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

for c in ["no2_center", "ntl_center", "smart_v1_center", "smart_v1_upwind",
          "so2_upwind", "co_upwind", "fire_upwind"]:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# =============================================================================
#  FEATURE CLEANING
# =============================================================================
print(f"\n{'='*80}")
print("FEATURE CLEANING")
print(f"{'='*80}")

rh_bad = df['Humidity_final'] < 5
df.loc[rh_bad, 'Humidity_final'] = np.nan
prs_bad = (df['Pressure_final'] < 950) | (df['Pressure_final'] > 1040)
df.loc[prs_bad, 'Pressure_final'] = np.nan
temp_bad = (df['Temperature_final'] < 0) | (df['Temperature_final'] > 50)
df.loc[temp_bad, 'Temperature_final'] = np.nan
pblh_bad = (df['PBLH'] < 0) | (df['PBLH'] > 6000)
df.loc[pblh_bad, 'PBLH'] = np.nan
ae_bad = df['AE'].abs() > 5
df.loc[ae_bad, 'AE'] = np.nan
aot_bad = df['AOT_ffill_48h'] > 5
df.loc[aot_bad, 'AOT_ffill_48h'] = np.nan
aot2_bad = df['AOT_outer_mean'] > 5
df.loc[aot2_bad, 'AOT_outer_mean'] = np.nan
ws_suspect = (df['WS_local'] == 0) & (np.sqrt(df['wind_u']**2 + df['wind_v']**2) > 1.0)
df.loc[ws_suspect, 'WS_local'] = np.nan

any_cleaned = rh_bad | prs_bad | temp_bad | pblh_bad | ws_suspect
print(f"  Total cleaned: {any_cleaned.sum():>7,} rows "
      f"({100*any_cleaned.sum()/n_total:.1f}%)")

# Derived features
ws_arr = np.sqrt(df["wind_u"].values**2 + df["wind_v"].values**2)
pblh_f = df["PBLH"].fillna(200).values
vc_inv_clean = 1.0 / (np.clip(pblh_f, 50, None) * np.clip(ws_arr, 0.1, None) + 1)

df["RH_factor"] = 1.0 / (1.0 - (df["Humidity_final"] / 100.0).clip(upper=0.95))
df["VC"] = pblh_f * np.clip(ws_arr, 0.1, None)
df["so2_upwind_x_VC_inv"] = df["so2_upwind"].values * vc_inv_clean

aot_outer_c = df["AOT_outer_mean"].fillna(0).values
rh_frac_c = (df["Humidity_final"] / 100.0).clip(0, 0.95).values
f_rh_c = 1.0 / (1.0 - rh_frac_c)
df["aod_outer_pm25"] = aot_outer_c / (pblh_f + 100.0) / f_rh_c

df['PBLH_min_24h'] = df.groupby('stationId')['PBLH'].transform(
    lambda x: x.rolling(24, min_periods=1).min())
stag_col = ((df['PBLH'] < 500) & (df['WS_local'].fillna(0) < 2)).astype(float)
df['stagnation_hours_12h'] = stag_col.groupby(df['stationId']).rolling(
    12, min_periods=1).sum().reset_index(level=0, drop=True)

# Physics-informed interaction features (v5d)
pblh_km = df["PBLH"].fillna(200).clip(lower=50) / 1000.0
rh_safe = df["RH_factor"].fillna(1.0).clip(lower=1.0)

df["aod_surface"]    = df["AOT_inner_mean"].fillna(0) / (pblh_km + 0.1)
df["aod_dry"]        = df["AOT_inner_mean"].fillna(0) / rh_safe
df["co_surface"]     = df["co_30d_mean"].fillna(0) / (pblh_km + 0.1)
df["hcho_surface"]   = df["hcho_30d_mean"].fillna(0) / (pblh_km + 0.1)
df["no2_surface"]    = df["no2_daily_anom"].fillna(0) / (pblh_km + 0.1)
df["so2_surface"]    = df["so2_daily_anom"].fillna(0) / (pblh_km + 0.1)
df["combustion_aod"] = df["co_30d_mean"].fillna(0) * df["AOT_fine"].fillna(0)
df["secondary_form"] = df["hcho_30d_mean"].fillna(0) * (df["Humidity_final"].fillna(50) / 100)
df["modis_surface"]  = df["modis_aod_7d"].fillna(0) / (pblh_km + 0.1)
df["stagnant_aod"]   = df["AOT_inner_mean"].fillna(0) * df["stagnation_hours_12h"].fillna(0)
df["stagnant_co"]    = df["co_30d_mean"].fillna(0) * df["stagnation_hours_12h"].fillna(0)
df["aod_anomaly"]    = df["AOT_inner_mean"].fillna(0) - df["aod_30d_mean"].fillna(0)

print("  Derived features recomputed.")

# =============================================================================
#  FEATURE SETS
# =============================================================================

# A. Satellite AOD (9)
SAT_AOD = ["AOT_ffill_48h", "AOT_outer_mean", "AOT_inner_mean", "AOT_fine",
           "RF", "AE", "AOT_spatial_std", "AOT_rolling_mean_24h",
           "hours_since_valid_AOT"]

# B. Daily satellite (6)
DAILY_SAT = ["modis_aod_7d", "modis_fine_aod_7d",
             "no2_daily_anom", "co_daily_anom", "so2_daily_anom",
             "hcho_daily_anom"]

# C. Meteorology (10)
MET = ["PBLH", "VC", "wind_u", "wind_v", "WS_local",
       "Temperature_final", "Humidity_final", "Pressure_final",
       "dT_6h", "dRH_6h"]

# D. Precipitation (3)
PRECIP = ["rain_days_7d", "consecutive_dry_days", "hrs_since_rain"]

# E. Temporal (4)
TEMPORAL = ["hour_sin", "hour_cos", "month_sin", "month_cos"]

# F. Wind-emission interaction (5)
EMISSION = ["smart_v1_center", "smart_v1_upwind",
            "so2_upwind_x_VC_inv", "co_upwind", "fire_upwind"]

# G. Spatial context (7)
SPATIAL = ["building_area_1km", "elevation_m", "ghap_annual", "acag_annual",
           "latitude", "RH_factor", "aod_outer_pm25"]

# H. Atmospheric stability (2)
STABILITY = ["PBLH_min_24h", "stagnation_hours_12h"]

# I. Satellite temporal regime indicators — daily rolling (11)
SAT_REGIME = MODIS_ROLL_COLS + TROPOMI_ROLL_COLS

# J. NEW: Station-level regime fingerprints — stable aggregates (11)
SAT_REGIME_STN = STN_AGG_COLS

FEAT_BASE = [f for f in (SAT_AOD + DAILY_SAT + MET + PRECIP + TEMPORAL +
              EMISSION + SPATIAL + STABILITY + SAT_REGIME + SAT_REGIME_STN)
             if f in df.columns]

# K. Physics-informed interactions (v5d) — observation-level (12)
PHYSICS_FEATS = ["aod_surface", "aod_dry", "co_surface", "hcho_surface",
                 "no2_surface", "so2_surface", "combustion_aod", "secondary_form",
                 "modis_surface", "stagnant_aod", "stagnant_co", "aod_anomaly"]

# L. Observation-only features (vary per hour/day, no station identity)
OBS_DERIVED = ["RH_factor", "aod_outer_pm25"]
FEAT_OBS = [f for f in (SAT_AOD + DAILY_SAT + MET + PRECIP + TEMPORAL +
            STABILITY + SAT_REGIME + OBS_DERIVED + PHYSICS_FEATS) if f in df.columns]
FEAT_OBS_RFSI = FEAT_OBS + RFSI_FEATURES

FEAT_BASE_PHYS = [f for f in (FEAT_BASE + PHYSICS_FEATS) if f in df.columns]
FEAT_BASE_PHYS = list(dict.fromkeys(FEAT_BASE_PHYS))
FEAT_ALL = FEAT_BASE_PHYS + RFSI_FEATURES

n_base = len(FEAT_BASE_PHYS)
n_obs = len(FEAT_OBS)
n_phys = sum(1 for f in PHYSICS_FEATS if f in df.columns)
n_feat = len(FEAT_ALL)
print(f"\n  Features: all={n_base}f + RFSI={len(RFSI_FEATURES)}f = {n_feat}f")
print(f"  Obs+physics: {n_obs}f (incl {n_phys} physics features, no station-level)")

base_arr = df[FEAT_BASE_PHYS].values.astype(np.float32)
base_arr = np.nan_to_num(base_arr, nan=np.nan)

obs_arr = df[FEAT_OBS].values.astype(np.float32)
obs_arr = np.nan_to_num(obs_arr, nan=np.nan)

# Monotonicity constraints
MONO_DICT = {
    "smart_v1_upwind": 1, "so2_upwind_x_VC_inv": 1,
    "co_upwind": 1, "fire_upwind": 1,
    "VC": -1, "PBLH": -1, "WS_local": -1,
    "PM25_nn_idw": 1, "PM25_nn1": 1,
    "modis_aod_7d": 1, "AOT_ffill_48h": 1,
    "aod_30d_mean": 1, "aod_30d_p90": 1,
    "hcho_30d_mean": 1, "co_30d_mean": 1,
    "aod_30d_mean_stn": 1, "aod_30d_p90_stn": 1,
    "hcho_30d_mean_stn": 1, "co_30d_mean_stn": 1,
    "aod_surface": 1, "aod_dry": 1, "co_surface": 1, "hcho_surface": 1,
    "combustion_aod": 1, "modis_surface": 1,
    "stagnant_aod": 1, "stagnant_co": 1,
}

mono_str = tuple(MONO_DICT.get(f, 0) for f in FEAT_ALL)
mono_str_base = tuple(MONO_DICT.get(f, 0) for f in FEAT_BASE_PHYS)
mono_str_obs = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS)
mono_str_obs_rfsi = tuple(MONO_DICT.get(f, 0) for f in FEAT_OBS_RFSI)

CONFIG_SETUP = {
    "delta_bm":   ("obs", False, "regional"),
    "delta_rfsi": ("obs", True,  "regional"),
    "oracle_bm":  ("all", True,  "oracle"),
}

sid_to_int = {s: i for i, s in enumerate(station_ids)}
row_sid_idx = np.array([sid_to_int[s] for s in stationId_vals])

# =============================================================================
#  BASE MARGINS
# =============================================================================
print(f"\n{'='*80}")
print("BASE MARGIN SETUP")
print(f"{'='*80}")

station_monthly_pm = df.groupby(['stationId', 'month'])['PM2.5'].mean()

bm_oracle = np.full(len(df), bm_global)
for sid in station_ids:
    sid_mask = stationId_vals == sid
    months = df['month'].values[sid_mask]
    for m in np.unique(months):
        try:
            val = station_monthly_pm.loc[(sid, int(m))]
        except KeyError:
            val = np.nan
        if np.isnan(val):
            val = global_pm_mean
        m_mask = sid_mask & (df['month'].values == m)
        bm_oracle[m_mask] = np.log1p(val)

print(f"  Global BM:   {bm_global:.4f} (log1p of {global_pm_mean:.1f} µg/m³)")
print(f"  Oracle BM:   mean={bm_oracle.mean():.4f}, std={bm_oracle.std():.4f}")

# Regional LOO BM diagnostic
print(f"\n  Delta regional BM (LOO on {n_stn} stations):")
regional_mean = float(station_pm_means.mean())
print(f"    Regional mean: {regional_mean:.1f} µg/m³")
reg_pred_loo = np.zeros(n_stn)
for si, sid in enumerate(station_ids):
    peers = [s for s in station_ids if s != sid]
    reg_pred_loo[si] = float(np.mean([station_pm_means[s] for s in peers]))
stn_actual = np.array([float(station_pm_means[sid]) for sid in station_ids])
reg_loo_r2 = r2_score(stn_actual, reg_pred_loo)
reg_loo_rmse = np.sqrt(np.mean((stn_actual - reg_pred_loo)**2))
print(f"    LOO R²: {reg_loo_r2:.3f}, RMSE: {reg_loo_rmse:.1f} µg/m³")
print(f"    {'Station':<35s} {'tier':>4s} {'actual':>7s} {'pred':>7s} {'err':>7s}")
for si, sid in enumerate(station_ids):
    actual = stn_actual[si]
    pred = reg_pred_loo[si]
    err = pred - actual
    print(f"      {sid_name.get(sid, sid)[:35]:35s} "
          f"{sid_tier[sid]:>4s} "
          f"{actual:7.1f} {pred:7.1f} {err:+7.1f}")

bm_map = {
    "oracle_bm": bm_oracle,
}

# =============================================================================
#  LOSO (14 delta folds)
# =============================================================================
n_configs = len(CONFIGS)
print(f"\n{'='*80}")
print(f"LOSO: {n_stn} folds, {n_configs} configs")
print(f"{'='*80}\n")

pred_all = {c: np.full(len(df), np.nan) for c in CONFIGS}

for fold_i, held_sid in enumerate(station_ids):
    nm = sid_name.get(held_sid, held_sid)[:35]
    held_tier = sid_tier[held_sid]
    mask_test = stationId_vals == held_sid
    n_test = mask_test.sum()
    if n_test < 10:
        print(f"  [{fold_i+1:2d}/{n_stn}] {nm:35s} | SKIP (n={n_test})")
        continue

    t_fold = time.time()
    pm_val = float(station_pm_means[held_sid])

    # --- RFSI (exclude held-out) ---
    rfsi_fold = compute_rfsi(exclude_sid=held_sid, K=K_NN)
    rfsi_arr = np.column_stack([rfsi_fold[c] for c in RFSI_FEATURES])

    test_idx = np.where(mask_test)[0]
    train_mask = (stationId_vals != held_sid) & ~np.isnan(y_all)
    train_idx = np.where(train_mask)[0]

    # Regional LOO BM: mean PM2.5 of all other delta stations
    train_sids = [s for s in station_ids if s != held_sid]
    regional_loo_mean = float(np.mean([station_pm_means[s] for s in train_sids]))
    regional_bm_val = np.log1p(regional_loo_mean)

    # Train separate model per config
    for cname in CONFIGS:
        feat_set, use_rfsi, bm_type = CONFIG_SETUP[cname]
        if feat_set == "obs":
            X_base_cfg = obs_arr
            mc = mono_str_obs_rfsi if use_rfsi else mono_str_obs
        else:
            X_base_cfg = base_arr
            mc = mono_str if use_rfsi else mono_str_base

        if use_rfsi:
            X_cfg = np.hstack([X_base_cfg, rfsi_arr])
        else:
            X_cfg = X_base_cfg

        if bm_type == "regional":
            bm_train = np.full(len(train_idx), regional_bm_val)
            bm_test = np.full(n_test, regional_bm_val)
        elif bm_type == "oracle":
            bm_train = bm_map["oracle_bm"][train_idx]
            bm_test = bm_map["oracle_bm"][mask_test]
        else:
            bm_train = np.full(len(train_idx), bm_global)
            bm_test = np.full(n_test, bm_global)

        y_tr = y_log[train_idx] - bm_train
        params = {**XGB_BASE, "monotone_constraints": mc}
        m = xgb.XGBRegressor(**params)
        m.fit(X_cfg[train_idx], y_tr)
        pred_res = m.predict(X_cfg[test_idx])
        pred_all[cname][mask_test] = np.clip(
            np.expm1(pred_res + bm_test), 0, None)

    # --- Fold summary ---
    fold_time = time.time() - t_fold
    remaining = fold_time * (n_stn - fold_i - 1)

    valid_y = ~np.isnan(y_all[mask_test])
    fold_r2s = {}
    for cname in CONFIGS:
        p = pred_all[cname][mask_test]
        v = valid_y & ~np.isnan(p)
        if v.sum() > 0:
            fold_r2s[cname] = r2_score(y_all[mask_test][v], p[v])
        else:
            fold_r2s[cname] = np.nan

    best = max(CONFIGS, key=lambda c: fold_r2s.get(c, -999))

    print(f"  [{fold_i+1:2d}/{n_stn}] {nm:35s} {held_tier} "
          f"pm={pm_val:5.1f} rbm={regional_loo_mean:5.1f} | "
          f"DLT={fold_r2s['delta_bm']:+.3f} "
          f"D+R={fold_r2s['delta_rfsi']:+.3f} "
          f"ORC={fold_r2s['oracle_bm']:+.3f} "
          f"best={FOLD_ABBREV.get(best, best[:3].upper())} "
          f"({fold_time:.0f}s, ETA {remaining/60:.0f}m)")

loso_time = time.time() - t0_start
print(f"\n  LOSO done: {loso_time:.0f}s")

# =============================================================================
#  RESULTS
# =============================================================================
print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")

ALL_CONFIGS = CONFIGS

# --- Pooled R2 ---
print(f"\n  Pooled R2_hourly:")
print(f"  {'Config':<18s} {'R2_h':>8s} {'RMSE':>7s} {'MAE':>7s} {'Bias':>8s}")
print("  " + "-" * 55)
for cname in ALL_CONFIGS:
    p = pred_all[cname]
    valid = ~np.isnan(p) & ~np.isnan(y_all)
    r2 = r2_score(y_all[valid], p[valid])
    rmse = np.sqrt(mean_squared_error(y_all[valid], p[valid]))
    mae = mean_absolute_error(y_all[valid], p[valid])
    bias = float(p[valid].mean() - y_all[valid].mean())
    print(f"  {cname:<18s} {r2:+8.4f} {rmse:7.2f} {mae:7.2f} {bias:+8.3f}")

# --- Per-station mean R2 ---
print(f"\n  Per-station mean R2:")
print(f"  {'Config':<18s} {'mean_R2':>8s} {'median':>8s} {'%>0':>6s}")
print("  " + "-" * 45)
for cname in ALL_CONFIGS:
    stn_r2s = []
    for sid in station_ids:
        mask = stationId_vals == sid
        p = pred_all[cname][mask]
        valid = ~np.isnan(p) & ~np.isnan(y_all[mask])
        if valid.sum() >= 10:
            stn_r2s.append(r2_score(y_all[mask][valid], p[valid]))
    arr = np.array(stn_r2s)
    pct_pos = 100 * (arr > 0).mean()
    print(f"  {cname:<18s} {arr.mean():+8.4f} {np.median(arr):+8.4f} "
          f"{pct_pos:5.0f}%")

# --- Per-station by tier ---
print(f"\n  By tier (mean R2):")
print(f"  {'Config':<18s} {'t0':>8s} {'t1':>8s} {'t2':>8s} {'t3':>8s}")
print("  " + "-" * 52)
for cname in ALL_CONFIGS:
    tier_r2 = {}
    for t in TIER_NAMES:
        t_sids = [s for s in station_ids if sid_tier[s] == t]
        r2s = []
        for sid in t_sids:
            mask = stationId_vals == sid
            p = pred_all[cname][mask]
            valid = ~np.isnan(p) & ~np.isnan(y_all[mask])
            if valid.sum() >= 10:
                r2s.append(r2_score(y_all[mask][valid], p[valid]))
        tier_r2[t] = np.mean(r2s) if r2s else np.nan
    print(f"  {cname:<18s} {tier_r2['t0']:+8.4f} {tier_r2['t1']:+8.4f} "
          f"{tier_r2['t2']:+8.4f} {tier_r2['t3']:+8.4f}")

# =============================================================================
#  FEATURE IMPORTANCE (from last fold model)
# =============================================================================
last_train_mask = (stationId_vals != station_ids[-1]) & ~np.isnan(y_all)
last_train_idx = np.where(last_train_mask)[0]
rfsi_last = compute_rfsi(exclude_sid=station_ids[-1], K=K_NN)
rfsi_arr_last = np.column_stack([rfsi_last[c] for c in RFSI_FEATURES])

# Regional BM for importance models
last_sid = station_ids[-1]
train_sids_imp = [s for s in station_ids if s != last_sid]
regional_bm_imp_val = np.log1p(float(np.mean([station_pm_means[s] for s in train_sids_imp])))
bm_regional_imp = np.full(len(last_train_idx), regional_bm_imp_val)

for imp_label, imp_feats, imp_X, imp_mc, imp_bm in [
    ("delta_bm (obs+physics, regional BM)", FEAT_OBS, obs_arr,
     mono_str_obs, bm_regional_imp),
    ("delta_rfsi (obs+physics+RFSI, regional BM)", FEAT_OBS_RFSI,
     np.hstack([obs_arr, rfsi_arr_last]), mono_str_obs_rfsi, bm_regional_imp),
]:
    print(f"\n  Top 20 feature importance — {imp_label} (gain, last fold):")
    y_tr_imp = y_log[last_train_idx] - imp_bm
    m_imp = xgb.XGBRegressor(**{**XGB_BASE, "monotone_constraints": imp_mc})
    m_imp.fit(imp_X[last_train_idx], y_tr_imp)
    imp = m_imp.feature_importances_
    imp_idx = np.argsort(imp)[::-1]
    for rank, i in enumerate(imp_idx[:20]):
        print(f"    {rank+1:2d}. {imp_feats[i]:25s}  gain={imp[i]:.4f}")

# =============================================================================
#  OUTPUT CSV
# =============================================================================
rows_out = []
for cname in ALL_CONFIGS:
    for sid in station_ids:
        mask = stationId_vals == sid
        p = pred_all[cname][mask]
        valid = ~np.isnan(p) & ~np.isnan(y_all[mask])
        if valid.sum() < 10:
            continue
        r2_h = r2_score(y_all[mask][valid], p[valid])
        rmse = np.sqrt(mean_squared_error(y_all[mask][valid], p[valid]))
        mae_v = mean_absolute_error(y_all[mask][valid], p[valid])
        bias_v = float(p[valid].mean() - y_all[mask][valid].mean())

        rows_out.append({
            "config": cname,
            "station_id": sid,
            "station_name": sid_name.get(sid, sid),
            "region": "Red River Delta",
            "tier": sid_tier[sid],
            "pm25_mean": float(station_pm_means[sid]),
            "ghap_annual": ghap_annual.get(sid, np.nan),
            "n_hours": int(valid.sum()),
            "r2_hourly": round(r2_h, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae_v, 2),
            "bias": round(bias_v, 3),
        })

out_df = pd.DataFrame(rows_out)
out_path = os.path.join(OUT_DIR, "delta_v1_test.csv")
out_df.to_csv(out_path, index=False)
print(f"\n  Results saved: {out_path}")

# =============================================================================
#  EXTERNAL VALIDATION: US Embassy + LCS
# =============================================================================
print(f"\n{'='*80}")
print("EXTERNAL VALIDATION: US Embassy + Delta LCS")
print(f"{'='*80}")

# --- Load LCS metadata ---
lcs_meta = pd.read_csv(os.path.join(REPO_DIR,
    "analysis/thesis_audit/station_selection_lcs.csv"), dtype={"station_id": str})
lcs_passed = lcs_meta[lcs_meta["lcs_flag"] == "pass"].copy()

env_map_all = pd.read_csv(os.path.join(META_DIR, "envisoft_station_map.csv"),
                           dtype={"stationId": str})
env_coord = env_map_all.set_index("stationId")[["latitude", "longitude"]]

DELTA_BOX = (20.3, 21.3, 105.5, 107.0)
val_stations = []
for _, row in lcs_passed.iterrows():
    sid = row["station_id"]
    if sid not in env_coord.index:
        continue
    lat, lon = env_coord.loc[sid, "latitude"], env_coord.loc[sid, "longitude"]
    if DELTA_BOX[0] <= lat <= DELTA_BOX[1] and DELTA_BOX[2] <= lon <= DELTA_BOX[3]:
        val_stations.append({"sid": sid, "name": row["station_name"],
                             "lat": lat, "lon": lon, "type": "LCS"})

val_stations.append({"sid": "US_EMBASSY_HAN", "name": "US Embassy Hanoi",
                      "lat": 21.0219, "lon": 105.8188, "type": "Embassy"})

print(f"  Validation stations: {len(val_stations)} "
      f"({sum(1 for v in val_stations if v['type']=='LCS')} LCS + 1 Embassy)")

# --- Build KK feature lookup: (stationId, ts) → row index ---
kk_coords = {sid: (sid_lat[sid], sid_lon[sid]) for sid in station_ids}
df["_ts_str"] = df["ts"].dt.strftime("%Y-%m-%dT%H:00:00")

# --- Train final model on all 12 KK delta stations ---
rfsi_full = compute_rfsi(exclude_sid=None, K=K_NN)
rfsi_arr_full = np.column_stack([rfsi_full[c] for c in RFSI_FEATURES])
regional_all_mean = float(station_pm_means.mean())
bm_final = np.log1p(regional_all_mean)

valid_train = ~np.isnan(y_all)
train_idx_all = np.where(valid_train)[0]
X_train_final = np.hstack([obs_arr, rfsi_arr_full])[train_idx_all]
y_train_final = y_log[train_idx_all] - bm_final

m_final = xgb.XGBRegressor(**{**XGB_BASE, "monotone_constraints": mono_str_obs_rfsi})
m_final.fit(X_train_final, y_train_final)
print(f"  Final model trained on {len(train_idx_all):,} rows from {n_stn} KK stations")

# --- Process each validation station ---
val_results = []
file_map_lcs = {}
for f in glob.glob(os.path.join(DATA_DIR, "data/stations/historical_full/*LCS*.csv")):
    fn = _unorm("NFC", os.path.basename(f).replace(".csv", ""))
    file_map_lcs[fn] = f
file_map_lcs[_unorm("NFC", "US Embassy Hanoi")] = os.path.join(
    DATA_DIR, "data/stations/historical_full/US Embassy Hanoi.csv")

for vi, vst in enumerate(val_stations):
    v_sid, v_lat, v_lon = vst["sid"], vst["lat"], vst["lon"]
    v_name = vst["name"]
    v_name_norm = _unorm("NFC", v_name)

    nearest_kk = min(station_ids,
                     key=lambda s: haversine(v_lat, v_lon, kk_coords[s][0], kk_coords[s][1]))
    nearest_dist = haversine(v_lat, v_lon, kk_coords[nearest_kk][0], kk_coords[nearest_kk][1])

    csv_path = file_map_lcs.get(v_name_norm)
    if csv_path is None:
        continue

    try:
        vdf = pd.read_csv(csv_path)
    except Exception:
        continue

    vdf["ts"] = pd.to_datetime(vdf["Timestamp"], errors="coerce")
    vdf = vdf.dropna(subset=["ts", "PM2.5"]).copy()
    vdf["PM2.5"] = pd.to_numeric(vdf["PM2.5"], errors="coerce")
    vdf = vdf.dropna(subset=["PM2.5"])
    vdf = vdf[(vdf["PM2.5"] >= 0) & (vdf["PM2.5"] <= 500)]
    vdf["ts"] = vdf["ts"].dt.floor("h")
    vdf = vdf.drop_duplicates("ts", keep="first")
    if len(vdf) < 100:
        continue

    vdf["_ts_str"] = vdf["ts"].dt.strftime("%Y-%m-%dT%H:00:00")

    kk_df = df[df["stationId"] == nearest_kk].copy()
    kk_df = kk_df.set_index("_ts_str")

    matched_ts = vdf["_ts_str"].isin(kk_df.index)
    vdf_matched = vdf[matched_ts].copy()
    if len(vdf_matched) < 100:
        continue

    kk_rows = kk_df.loc[vdf_matched["_ts_str"]]
    X_val_obs = kk_rows[FEAT_OBS].values.astype(np.float32)

    # RFSI: compute from all 12 KK stations to validation station location
    kk_dists = [(sid, haversine(v_lat, v_lon, kk_coords[sid][0], kk_coords[sid][1]))
                for sid in station_ids]
    kk_dists.sort(key=lambda x: x[1])
    nearest_k = kk_dists[:K_NN]

    pm_nn_val = np.full((len(vdf_matched), K_NN), np.nan)
    d_nn_val = np.array([d for _, d in nearest_k])
    val_ts_idx = pd.to_datetime(vdf_matched["ts"].values)

    for k, (kk_sid, _) in enumerate(nearest_k):
        if kk_sid in pm25_wide.columns:
            kk_pm = pm25_wide[kk_sid]
            matched_pm = kk_pm.reindex(val_ts_idx)
            pm_nn_val[:, k] = matched_pm.values

    with np.errstate(divide="ignore", invalid="ignore"):
        w_val = 1.0 / d_nn_val
        valid_mask_rfsi = ~np.isnan(pm_nn_val)
        pm_idw_val = (np.nansum(pm_nn_val * w_val, axis=1) /
                      np.nansum(w_val * valid_mask_rfsi, axis=1))

    rfsi_val = np.column_stack([
        pm_idw_val, pm_nn_val[:, 0], pm_nn_val[:, 1], pm_nn_val[:, 2]])

    X_val = np.hstack([X_val_obs, rfsi_val])
    bm_val = np.full(len(X_val), bm_final)
    pred_res_val = m_final.predict(X_val)
    pred_pm = np.clip(np.expm1(pred_res_val + bm_val), 0, None)

    y_val = vdf_matched["PM2.5"].values
    valid_both = ~np.isnan(pred_pm) & ~np.isnan(y_val)
    if valid_both.sum() < 50:
        continue

    r2_val = r2_score(y_val[valid_both], pred_pm[valid_both])
    rmse_val = np.sqrt(mean_squared_error(y_val[valid_both], pred_pm[valid_both]))
    mae_val = mean_absolute_error(y_val[valid_both], pred_pm[valid_both])
    bias_val = float(pred_pm[valid_both].mean() - y_val[valid_both].mean())

    val_results.append({
        "station": v_name, "type": vst["type"], "sid": v_sid,
        "lat": v_lat, "lon": v_lon,
        "nearest_kk": sid_name.get(nearest_kk, nearest_kk)[:30],
        "kk_dist_km": round(nearest_dist, 1),
        "pm25_mean": round(float(y_val.mean()), 1),
        "n_hours": int(valid_both.sum()),
        "r2": round(r2_val, 4), "rmse": round(rmse_val, 2),
        "mae": round(mae_val, 2), "bias": round(bias_val, 2),
    })

    if (vi + 1) % 10 == 0 or vi == len(val_stations) - 1:
        print(f"    [{vi+1}/{len(val_stations)}] processed...")

# --- Print LCS results ---
print(f"\n  External validation results ({len(val_results)} stations):")
print(f"  {'Station':<45s} {'type':>6s} {'dist':>5s} {'pm25':>5s} {'n':>5s} "
      f"{'R2':>7s} {'RMSE':>6s} {'MAE':>5s} {'bias':>6s}")
print("  " + "-" * 100)

lcs_r2s = []
for vr in sorted(val_results, key=lambda x: x["r2"], reverse=True):
    print(f"  {vr['station'][:45]:<45s} {vr['type']:>6s} {vr['kk_dist_km']:5.1f} "
          f"{vr['pm25_mean']:5.1f} {vr['n_hours']:5d} "
          f"{vr['r2']:+7.3f} {vr['rmse']:6.1f} {vr['mae']:5.1f} {vr['bias']:+6.1f}")
    if vr["type"] == "LCS":
        lcs_r2s.append(vr["r2"])

if lcs_r2s:
    arr_lcs = np.array(lcs_r2s)
    print(f"\n  LCS summary: n={len(arr_lcs)}, "
          f"mean R²={arr_lcs.mean():+.3f}, median={np.median(arr_lcs):+.3f}, "
          f"pct>0={100*(arr_lcs>0).mean():.0f}%")
emb = [vr for vr in val_results if vr["type"] == "Embassy"]
if emb:
    print(f"  US Embassy: R²={emb[0]['r2']:+.3f}, RMSE={emb[0]['rmse']:.1f}, "
          f"n={emb[0]['n_hours']} hours")

# Save validation results
val_df = pd.DataFrame(val_results)
val_path = os.path.join(OUT_DIR, "delta_v1_lcs_validation.csv")
val_df.to_csv(val_path, index=False)
print(f"\n  Validation saved: {val_path}")
print(f"  Total time: {time.time()-t0_start:.0f}s")
