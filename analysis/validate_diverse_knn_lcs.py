"""LCS external validation of the corrected + diverse kNN pipeline.

Strategy:
1. Train 5 diverse XGB models on thesis data (all 40 stations, no LOSO)
2. Predict on LCS stations (impute missing features)
3. Use thesis OOF R² to train kNN selector
4. Apply corrected+diverse kNN to LCS predictions
5. Compare with existing correct-first LCS results (+0.164)
"""
from __future__ import annotations
import sys, time, warnings, os
from pathlib import Path

warnings.filterwarnings("ignore")

_logdir = Path(__file__).resolve().parent / "experimental_shape_magnitude" / "himawari_safe_selector"
_logdir.mkdir(parents=True, exist_ok=True)
LOGFILE = _logdir / "lcs_diverse_knn_validation.log"

class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.orig = sys.stdout
    def write(self, s):
        try: self.orig.write(s)
        except: pass
        self.f.write(s); self.f.flush()
    def flush(self):
        try: self.orig.flush()
        except: pass
        self.f.flush()

sys.stdout = Tee(LOGFILE)
sys.stderr = sys.stdout

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer

t0 = time.time()

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "analysis" / "experimental_shape_magnitude"
THESIS_DATA = ROOT / "data" / "merged" / "unified_thesis_v2.csv"
LCS_PRED = ROOT / "analysis" / "thesis_experiments" / "external_validation_himawari_full_predictions.csv"
LCS_SELECTION = ROOT / "Thesis" / "results" / "01_stations" / "station_selection_lcs.csv"
THESIS_META = ROOT / "analysis" / "thesis_audit" / "station_selection_final.csv"
TIER_SUMMARY = EXP / "true_tier_moe_xgb" / "himawari_tierexperts.csv"
BASE_OOF = EXP / "true_tier_moe_xgb" / "himawari_fullfeature_oof.csv"
DIVERSE_OOF = EXP / "diverse_streams" / "oof_predictions.csv"
FEATURES = EXP / "station_feature_table.csv"
OUT_DIR = EXP / "himawari_safe_selector"

QC_DIR = ROOT / "Thesis" / "scripts" / "02_processing"
if str(QC_DIR) not in sys.path:
    sys.path.insert(0, str(QC_DIR))
from pm25_qc import pm25_quality_masks

PM_MIN, PM_MAX = 0.1, 250.0
FLATLINE_HOURS = 24
MIN_ROWS, MIN_DAYS = 500, 20


def pm_class(x):
    if x < 10: return "low"
    if x < 20: return "moderate_low"
    if x < 35: return "moderate"
    return "high"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def eval_lcs(df, col, stn_masks, sids):
    stn_r2s, h2nh, nh2h, flips = [], 0, 0, 0
    y_all, p_all = [], []
    for sid in sids:
        m = stn_masks[sid]
        y = df.loc[m, "PM2.5"].values
        p = df.loc[m, col].values if isinstance(col, str) else col[m]
        v = ~(np.isnan(y) | np.isnan(p))
        y, p = y[v], p[v]
        if len(y) < 10: continue
        y_all.extend(y); p_all.extend(p)
        r2 = float(r2_score(y, p))
        stn_r2s.append(r2)
        true_cls = pm_class(np.mean(y))
        pred_cls = pm_class(np.mean(p))
        if true_cls != pred_cls:
            flips += 1
            sev = {"low": 0, "moderate_low": 1, "moderate": 2, "high": 3}
            if sev[pred_cls] < sev[true_cls] and true_cls in ("moderate", "high") and pred_cls in ("low", "moderate_low"):
                h2nh += 1
            elif sev[pred_cls] > sev[true_cls] and pred_cls in ("moderate", "high") and true_cls in ("low", "moderate_low"):
                nh2h += 1
    y_all, p_all = np.array(y_all), np.array(p_all)
    return {
        "mean_r2": float(np.mean(stn_r2s)),
        "pool_r2": float(r2_score(y_all, p_all)),
        "pos": sum(1 for r in stn_r2s if r > 0),
        "h2nh": h2nh, "nh2h": nh2h, "flips": flips,
        "n_stn": len(stn_r2s),
    }


# ════════════════════════════════════════════════════════════════
print("=" * 80)
print("LCS VALIDATION: corrected + diverse kNN pipeline")
print("=" * 80)

# ── 1. Load thesis data & train diverse models ────────────────
print("\n1. Loading thesis data for diverse model training...")
thesis_raw = pd.read_csv(THESIS_DATA, dtype={"stationId": str}, parse_dates=["ts"])
thesis_raw = thesis_raw.dropna(subset=["PM2.5"]).reset_index(drop=True)

qc_masks = pm25_quality_masks(thesis_raw)
thesis_raw.loc[qc_masks.any(axis=1), "PM2.5"] = np.nan
thesis_raw = thesis_raw.dropna(subset=["PM2.5"])

thesis_sids = sorted(thesis_raw["stationId"].unique())
n_thesis = len(thesis_sids)
print(f"  Thesis: {len(thesis_raw):,} rows, {n_thesis} stations")

global_pm_mean = float(thesis_raw["PM2.5"].mean())
bm_global = np.log1p(global_pm_mean)

MET_CORE = ["PBLH", "VC", "wind_u", "wind_v", "WS_local",
            "Temperature_final", "Humidity_final", "Pressure_final",
            "dT_6h", "dRH_6h", "rain_days_7d", "rain_sum_48h",
            "consecutive_dry_days", "hrs_since_rain", "RH_factor"]
TEMPORAL = ["hour_sin", "hour_cos", "month_sin", "month_cos",
            "day_of_year_cos", "day_of_year_sin"]
TERRAIN = ["elevation_m", "slope_deg"]
AOD_CORE = ["AOT_ffill_48h", "AOT_outer_mean", "AE", "RF",
            "hours_since_valid_AOT", "RF_center", "RF_mean",
            "SSA_center", "SSA_mean", "AOT_fine",
            "AOT_grad_mag", "AOT_local_vs_regional"]
AOD_EXTENDED = AOD_CORE + ["AOD_physics", "AOT_rolling_mean_6h",
                            "AOT_rolling_mean_24h", "AOT_lag_1h", "AOT_lag_3h"]

STREAMS = {
    "dispersion": MET_CORE + TEMPORAL + TERRAIN,
    "satellite": AOD_EXTENDED + ["PBLH", "VC", "RH_factor"] + TEMPORAL[:4],
    "emission": ["NO2", "SO2", "CO"] + TERRAIN + ["PBLH", "VC"] + TEMPORAL,
    "spatial": MET_CORE + TEMPORAL + TERRAIN,
    "full": MET_CORE + TEMPORAL + TERRAIN + AOD_EXTENDED + ["NO2", "SO2", "CO"],
}

for name in list(STREAMS.keys()):
    feats = [f for f in STREAMS[name] if f in thesis_raw.columns]
    STREAMS[name] = feats

XGB_PARAMS = dict(
    booster="gbtree",
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=40,
    reg_alpha=0.1, reg_lambda=8.0, tree_method="hist",
    device="cuda", random_state=42, n_jobs=-1,
)

y_thesis_log = np.log1p(thesis_raw["PM2.5"].values) - bm_global
y_thesis_raw = thesis_raw["PM2.5"].values

print("\n  Training diverse models on ALL thesis stations...")
stream_models_log = {}
stream_models_raw = {}
for name, feats in STREAMS.items():
    X = thesis_raw[feats].fillna(0).values
    model_log = xgb.XGBRegressor(**XGB_PARAMS)
    model_log.fit(X, y_thesis_log)
    stream_models_log[name] = model_log

    model_raw = xgb.XGBRegressor(**XGB_PARAMS)
    model_raw.fit(X, y_thesis_raw)
    stream_models_raw[name] = model_raw
    print(f"    {name:12s}: {len(feats)} features — trained (log + raw)")

print(f"  {len(STREAMS)} × 2 = {len(STREAMS)*2} diverse models trained ({time.time()-t0:.0f}s)")


# ── 2. Load LCS data ─────────────────────────────────────────
print("\n2. Loading LCS data...")
lcs_raw = pd.read_csv(LCS_PRED, dtype={"stationId": str}, parse_dates=["ts"])
lcs_sel = pd.read_csv(LCS_SELECTION, dtype={"station_id": str})
pass_sids = set(lcs_sel[lcs_sel["lcs_flag"] == "pass"]["station_id"])
lcs_raw = lcs_raw[lcs_raw["stationId"].isin(pass_sids)].copy()
print(f"  LCS raw: {len(lcs_raw):,} rows, {lcs_raw['stationId'].nunique()} stations")

lcs_raw["PM2.5"] = lcs_raw["PM2.5"].clip(lower=PM_MIN, upper=PM_MAX)
lcs_raw = lcs_raw[lcs_raw["PM2.5"].between(PM_MIN, PM_MAX)].copy()

for sid in lcs_raw["stationId"].unique():
    m = lcs_raw["stationId"] == sid
    vals = lcs_raw.loc[m, "PM2.5"].values
    if len(vals) < 2:
        continue
    diffs = np.abs(np.diff(vals))
    flat = np.zeros(len(vals), dtype=bool)
    run = 1
    for i in range(len(diffs)):
        if diffs[i] < 0.01:
            run += 1
            if run >= FLATLINE_HOURS:
                flat[max(0, i+2-run):i+2] = True
        else:
            run = 1
    lcs_raw.loc[m, "PM2.5"] = np.where(flat, np.nan, vals)

lcs_raw = lcs_raw.dropna(subset=["PM2.5"]).reset_index(drop=True)

valid_sids = []
for sid in sorted(lcs_raw["stationId"].unique()):
    g = lcs_raw[lcs_raw["stationId"] == sid]
    if len(g) >= MIN_ROWS and g["ts"].dt.date.nunique() >= MIN_DAYS:
        valid_sids.append(sid)

lcs = lcs_raw[lcs_raw["stationId"].isin(valid_sids)].copy()
n_lcs = len(valid_sids)
print(f"  After cleaning: {len(lcs):,} rows, {n_lcs} stations")

lcs_masks = {sid: (lcs["stationId"] == sid).values for sid in valid_sids}


# ── 3. Build LCS features for diverse model prediction ───────
print("\n3. Building LCS features for diverse predictions...")

lcs["hour"] = lcs["ts"].dt.hour
lcs["month"] = lcs["ts"].dt.month
lcs["hour_sin"] = np.sin(2 * np.pi * lcs["hour"] / 24)
lcs["hour_cos"] = np.cos(2 * np.pi * lcs["hour"] / 24)
lcs["month_sin"] = np.sin(2 * np.pi * lcs["month"] / 12)
lcs["month_cos"] = np.cos(2 * np.pi * lcs["month"] / 12)
doy = lcs["ts"].dt.dayofyear
lcs["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
lcs["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)

rh = lcs["Humidity_final"].fillna(60).values
lcs["RH_factor"] = np.where(rh > 80, 1 + (rh - 80) / 20, 1.0)

if "wind_u" not in lcs.columns:
    ws = lcs["WS_local"].fillna(1.0).values
    lcs["wind_u"] = 0.0
    lcs["wind_v"] = -ws

if "VC" not in lcs.columns:
    lcs["VC"] = lcs["PBLH"].fillna(500).values * lcs["WS_local"].fillna(1.0).values

for col in ["dT_6h", "dRH_6h", "elevation_m", "slope_deg",
            "NO2", "SO2", "CO", "AOT_ffill_48h", "AOT_outer_mean",
            "AE", "RF", "hours_since_valid_AOT", "RF_center", "RF_mean",
            "SSA_center", "SSA_mean", "AOT_fine", "AOT_grad_mag",
            "AOT_local_vs_regional", "AOD_physics",
            "AOT_rolling_mean_6h", "AOT_rolling_mean_24h",
            "AOT_lag_1h", "AOT_lag_3h"]:
    if col not in lcs.columns:
        lcs[col] = 0.0

print(f"  LCS features ready")


# ── 4. Generate diverse predictions for LCS ──────────────────
print("\n4. Generating diverse predictions for LCS...")
stream_cols = []
for name, feats in STREAMS.items():
    X_lcs = lcs[feats].fillna(0).values

    pred_log = stream_models_log[name].predict(X_lcs)
    pred_log = np.clip(np.expm1(pred_log + bm_global), 1.0, 250.0)
    col_log = f"div_{name}"
    lcs[col_log] = pred_log.astype("float32")
    stream_cols.append(col_log)

    pred_raw = stream_models_raw[name].predict(X_lcs)
    pred_raw = np.clip(pred_raw, 1.0, 250.0)
    col_raw = f"div_raw_{name}"
    lcs[col_raw] = pred_raw.astype("float32")
    stream_cols.append(col_raw)

    lcs_r2_log = eval_lcs(lcs, col_log, lcs_masks, valid_sids)
    lcs_r2_raw = eval_lcs(lcs, col_raw, lcs_masks, valid_sids)
    print(f"    {name:12s}: log={lcs_r2_log['mean_r2']:+.4f}  raw={lcs_r2_raw['mean_r2']:+.4f}")

print(f"  {len(stream_cols)} diverse predictions generated")

# Also evaluate the backbone prediction
r_backbone = eval_lcs(lcs, "pred_PM25", lcs_masks, valid_sids)
print(f"\n  Backbone (pred_PM25): mean_r2={r_backbone['mean_r2']:+.4f}  pool_r2={r_backbone['pool_r2']:+.4f}")


# ── 5. Spatial prior for LCS ─────────────────────────────────
print("\n5. Computing spatial prior...")
thesis_meta = pd.read_csv(THESIS_META, dtype={"stationId": str})
train_coords = {}
train_means = {}
thesis_pm = thesis_raw.groupby("stationId")["PM2.5"].mean()
for _, row in thesis_meta.iterrows():
    sid = row["stationId"]
    if sid in thesis_pm.index:
        lat_col = "lat" if "lat" in thesis_meta.columns else "latitude"
        lon_col = "lon" if "lon" in thesis_meta.columns else "longitude"
        train_coords[sid] = (float(row[lat_col]), float(row[lon_col]))
        train_means[sid] = float(thesis_pm[sid])

global_train_mean = float(np.mean(list(train_means.values())))

prior_map = {}
for sid in valid_sids:
    m = lcs_masks[sid]
    lat = float(lcs.loc[m, "latitude"].iloc[0])
    lon = float(lcs.loc[m, "longitude"].iloc[0])
    dists = {s: haversine(lat, lon, *c) for s, c in train_coords.items()}
    sorted_ids = sorted(dists, key=dists.get)[:10]
    d = np.array([dists[s] for s in sorted_ids])
    w = np.exp(-(d / 60.0)**2)
    means = np.array([train_means[s] for s in sorted_ids])
    prior_map[sid] = float(np.dot(w, means) / w.sum())


# ── 6. HGB correction (train on thesis OOF, apply to LCS) ───
print("\n6. Training HGB correction on thesis OOF predictions...")
oof = pd.read_csv(BASE_OOF, dtype={"station_id": str}, parse_dates=["ts"])
oof_moe = oof[oof["config"] == "true_tier_moe_expert"][["station_id", "ts", "y_true", "y_pred"]].copy()
oof_moe = oof_moe.rename(columns={"y_pred": "moe_expert"})

oof_moe["hour"] = oof_moe["ts"].dt.hour
oof_moe["month"] = oof_moe["ts"].dt.month
oof_moe["hour_sin"] = np.sin(2 * np.pi * oof_moe["hour"] / 24)
oof_moe["hour_cos"] = np.cos(2 * np.pi * oof_moe["hour"] / 24)
oof_moe["month_sin"] = np.sin(2 * np.pi * oof_moe["month"] / 12)
oof_moe["month_cos"] = np.cos(2 * np.pi * oof_moe["month"] / 12)
oof_moe["moe_station_mean"] = oof_moe.groupby("station_id")["moe_expert"].transform("mean")
oof_moe["moe_anom_station"] = oof_moe["moe_expert"] - oof_moe["moe_station_mean"]
oof_moe["moe_station_hour_mean"] = oof_moe.groupby(["station_id", "hour"])["moe_expert"].transform("mean")
oof_moe["moe_anom_hour"] = oof_moe["moe_expert"] - oof_moe["moe_station_hour_mean"]

HGB_FEATS = ["moe_expert", "moe_station_mean", "moe_anom_station", "moe_anom_hour",
             "hour_sin", "hour_cos", "month_sin", "month_cos"]

pre = ColumnTransformer(
    transformers=[("num", SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), HGB_FEATS)], remainder="drop")
hgb = HistGradientBoostingRegressor(
    max_iter=45, learning_rate=0.05, max_leaf_nodes=7,
    min_samples_leaf=180, l2_regularization=1.5, random_state=42)
hgb_pipe = SkPipeline([("pre", pre), ("reg", hgb)])

sample_n = min(700, oof_moe.groupby("station_id").size().min())
sampled = oof_moe.groupby("station_id", group_keys=False).sample(
    n=sample_n, replace=True, random_state=42)
hgb_pipe.fit(sampled[HGB_FEATS], np.log1p(sampled["y_true"].clip(lower=0)))
print(f"  HGB trained on {len(sampled)} samples from {oof_moe['station_id'].nunique()} stations")

lcs["pred_station_mean"] = lcs.groupby("stationId")["pred_PM25"].transform("mean")
lcs["pred_anom_station"] = lcs["pred_PM25"] - lcs["pred_station_mean"]
lcs["pred_station_hour_mean"] = lcs.groupby(["stationId", "hour"])["pred_PM25"].transform("mean")
lcs["pred_anom_hour"] = lcs["pred_PM25"] - lcs["pred_station_hour_mean"]

lcs_hgb_feats = lcs[["pred_PM25", "pred_station_mean", "pred_anom_station",
                       "pred_anom_hour", "hour_sin", "hour_cos", "month_sin", "month_cos"]].copy()
lcs_hgb_feats.columns = HGB_FEATS
hgb_pred = hgb_pipe.predict(lcs_hgb_feats)
lcs["corrected"] = np.clip(np.expm1(hgb_pred), 1.0, 250.0).astype("float32")

r_corr = eval_lcs(lcs, "corrected", lcs_masks, valid_sids)
print(f"  HGB corrected: mean_r2={r_corr['mean_r2']:+.4f}")


# ── 7. Load thesis OOF data for kNN training ─────────────────
print("\n7. Setting up kNN selector from thesis OOF data...")
features = pd.read_csv(FEATURES, dtype={"station_id": str})
feat_thesis = features.drop_duplicates("station_id").set_index("station_id").reindex(thesis_sids)
num_fc = [c for c in feat_thesis.columns if pd.api.types.is_numeric_dtype(feat_thesis[c])]
fm = feat_thesis[num_fc].fillna(feat_thesis[num_fc].median()).values
scaler = StandardScaler().fit(fm)
fs_thesis = scaler.transform(fm)

div_oof = pd.read_csv(DIVERSE_OOF, dtype={"stationId": str}, parse_dates=["ts"])
div_oof = div_oof.rename(columns={"stationId": "station_id", "PM2.5": "y_true_div"})

oof_base = oof[oof["config"] == "no_t4f"][["station_id", "ts", "y_true", "y_pred"]].rename(columns={"y_pred": "no_t4f"})
oof_moe_short = oof[oof["config"] == "true_tier_moe_expert"][["station_id", "ts", "y_pred"]].rename(columns={"y_pred": "moe_expert"})
oof_df = oof_base.merge(oof_moe_short, on=["station_id", "ts"], how="left")
oof_df = oof_df.merge(
    div_oof[["station_id", "ts"] + [c for c in div_oof.columns if c.startswith("pred_")]],
    on=["station_id", "ts"], how="left")

thesis_stn_r2 = {}
candidate_names = ["moe_expert"] + [c for c in div_oof.columns if c.startswith("pred_")]
for sid in thesis_sids:
    g = oof_df[oof_df["station_id"] == sid]
    y = g["y_true"].values
    thesis_stn_r2[sid] = {}
    for cname in candidate_names:
        if cname not in g.columns:
            continue
        p = g[cname].values
        v = ~(np.isnan(y) | np.isnan(p))
        if v.sum() >= 10:
            thesis_stn_r2[sid][cname] = r2_score(y[v], p[v])
print(f"  Thesis OOF R² computed for {len(thesis_stn_r2)} stations × {len(candidate_names)} candidates")


# ── 8. kNN for LCS stations using thesis neighbors ──────────
print("\n8. Running kNN selection for LCS stations...")

lcs_coords = {}
for sid in valid_sids:
    m = lcs_masks[sid]
    lcs_coords[sid] = (float(lcs.loc[m, "latitude"].iloc[0]),
                        float(lcs.loc[m, "longitude"].iloc[0]))

thesis_lats = feat_thesis[feat_thesis.columns[feat_thesis.columns.str.contains("lat", case=False)]].iloc[:, 0].values
thesis_lons = feat_thesis[feat_thesis.columns[feat_thesis.columns.str.contains("lon", case=False)]].iloc[:, 0].values

DIVERSE_STREAM_MAP = {
    "pred_dispersion": "div_dispersion",
    "pred_satellite": "div_satellite",
    "pred_emission": "div_emission",
    "pred_spatial": "div_spatial",
    "pred_full": "div_full",
    "pred_raw_dispersion": "div_raw_dispersion",
    "pred_raw_satellite": "div_raw_satellite",
    "pred_raw_emission": "div_raw_emission",
    "pred_raw_spatial": "div_raw_spatial",
    "pred_raw_full": "div_raw_full",
}

def knn_select_for_lcs(candidate_map, k=3):
    """For each LCS station, find k nearest thesis stations and pick best candidate."""
    selections = {}
    for lcs_sid in valid_sids:
        lat, lon = lcs_coords[lcs_sid]
        dists = []
        for i, t_sid in enumerate(thesis_sids):
            d = haversine(lat, lon, float(thesis_lats[i]), float(thesis_lons[i]))
            dists.append((t_sid, d))
        dists.sort(key=lambda x: x[1])
        neighbors = [s for s, d in dists[:k]]

        scores = {}
        for thesis_cname, lcs_cname in candidate_map.items():
            if lcs_cname not in lcs.columns:
                continue
            nbr_scores = [thesis_stn_r2[n].get(thesis_cname, -999)
                          for n in neighbors if thesis_cname in thesis_stn_r2.get(n, {})]
            if nbr_scores:
                scores[lcs_cname] = np.mean(nbr_scores)
        if scores:
            selections[lcs_sid] = max(scores, key=scores.get)
        else:
            selections[lcs_sid] = "pred_PM25"
    return selections


# Build candidate map: thesis OOF name -> LCS column name
candidate_map = {"moe_expert": "pred_PM25"}
candidate_map.update(DIVERSE_STREAM_MAP)

selections_raw = knn_select_for_lcs(candidate_map, k=3)

selected_raw = lcs["pred_PM25"].values.copy()
for sid, chosen in selections_raw.items():
    m = lcs_masks[sid]
    if chosen in lcs.columns:
        selected_raw[m] = lcs.loc[m, chosen].values

lcs["knn_diverse_raw"] = selected_raw

n_div = sum(1 for v in selections_raw.values() if v.startswith("div_"))
n_backbone = sum(1 for v in selections_raw.values() if v == "pred_PM25")
print(f"  Selections: {n_div} diverse, {n_backbone} backbone")
for sid in sorted(selections_raw.keys())[:10]:
    print(f"    {sid[:8]}... → {selections_raw[sid]}")


# ── 9. Apply prior shift to selections ───────────────────────
print("\n9. Building evaluation variants...")

def apply_shift(col, alpha):
    vals = lcs[col].values.copy() if isinstance(col, str) else col.copy()
    for sid in valid_sids:
        m = lcs_masks[sid]
        pred_mean = float(vals[m].mean())
        shift = alpha * (prior_map[sid] - pred_mean)
        if pm_class(pred_mean + shift) == pm_class(pred_mean):
            vals[m] = vals[m] + shift
    return vals

def guard(vals):
    out = vals.copy()
    for sid in valid_sids:
        m = lcs_masks[sid]
        pred_mean = float(vals[m].mean())
        p = prior_map[sid]
        if pm_class(pred_mean) != pm_class(p):
            out[m] = vals[m] + 0.60 * (p - pred_mean)
    return out


variants = {}

# Raw backbone variants (reference)
variants["01_backbone"] = lcs["pred_PM25"].values
variants["02_backbone_s45"] = apply_shift("pred_PM25", 0.45)
variants["03_backbone_guard"] = guard(lcs["pred_PM25"].values)

# kNN diverse (no shift)
variants["04_knn_diverse_raw"] = selected_raw
variants["05_knn_diverse_s45"] = apply_shift(selected_raw, 0.45)
variants["06_knn_diverse_guard"] = guard(selected_raw)

# HGB corrected + diverse
candidate_map_corr = {"moe_expert": "corrected"}
candidate_map_corr.update(DIVERSE_STREAM_MAP)
sel_corr = knn_select_for_lcs(candidate_map_corr, k=3)
sel_corr_vals = lcs["corrected"].values.copy()
for sid, chosen in sel_corr.items():
    m = lcs_masks[sid]
    if chosen in lcs.columns:
        sel_corr_vals[m] = lcs.loc[m, chosen].values
variants["07_knn_corrdiv_raw"] = sel_corr_vals
variants["08_knn_corrdiv_s45"] = apply_shift(sel_corr_vals, 0.45)
variants["09_knn_corrdiv_guard"] = guard(sel_corr_vals)

# Blends
for ba in [0.30, 0.50, 0.70]:
    bl = (1 - ba) * lcs["pred_PM25"].values + ba * sel_corr_vals
    variants[f"10_blend_corrdiv_b{int(ba*100)}"] = bl
    bl_s = apply_shift(bl, 0.45)
    variants[f"11_blend_corrdiv_b{int(ba*100)}_s45"] = bl_s


# ── 10. Evaluate all variants ────────────────────────────────
print("\n10. Evaluating variants...")
print(f"\n  {'Variant':<35s}  {'mean_r2':>8s}  {'pool_r2':>8s}  {'pos':>4s}  {'flips':>5s}  {'h2nh':>5s}")
print(f"  {'-'*75}")

for name, vals in variants.items():
    lcs["_eval"] = vals
    r = eval_lcs(lcs, "_eval", lcs_masks, valid_sids)
    print(f"  {name:<35s}  {r['mean_r2']:+.4f}    {r['pool_r2']:+.4f}   {r['pos']:3d}   {r['flips']:4d}   {r['h2nh']:4d}")
    lcs.drop(columns=["_eval"], inplace=True)

print(f"\n  Reference results:")
print(f"  {'correct_first raw_shift_a45':<35s}  +0.1635    +0.3479    35     26     13")
print(f"  {'correct_first raw_guarded':<35s}  +0.1389    +0.3779    33     19      2")
print(f"  {'no_ghap_guarded (ref)':<35s}  +0.1083    +0.2865    33     18      4")

# Per-diverse-stream LCS metrics
print(f"\n  Per-stream LCS metrics:")
for sc in stream_cols:
    r = eval_lcs(lcs, sc, lcs_masks, valid_sids)
    print(f"    {sc:<25s}  mean_r2={r['mean_r2']:+.4f}  pool_r2={r['pool_r2']:+.4f}  pos={r['pos']}")


# ── Selection details ────────────────────────────────────────
print(f"\n  kNN selection breakdown (corr+diverse):")
from collections import Counter
sel_counts = Counter(sel_corr.values())
for k, v in sorted(sel_counts.items(), key=lambda x: -x[1]):
    print(f"    {k:<25s}: {v} stations")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("DONE")
