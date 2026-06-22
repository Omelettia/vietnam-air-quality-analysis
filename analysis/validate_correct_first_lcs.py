"""LCS external validation of the correct-first Himawari pipeline.

Pipeline (matching OOF experiment that achieved +0.190 mean_stn_r2):
1. HGB correction on base prediction (base_only features, no station-level)
2. Spatial prior shift (substitute for tier-expert kNN on unseen stations)
3. Blend corrected + uncorrected streams

HGB is trained on 40 thesis stations' OOF predictions (realistic out-of-sample
quality matching what LCS stations receive). Applied to LCS predictions from
the Himawari-full external backbone.
"""
from __future__ import annotations
import sys, time, math, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

_logdir = Path(__file__).resolve().parent / "experimental_shape_magnitude" / "himawari_safe_selector"
_logdir.mkdir(parents=True, exist_ok=True)
LOGFILE = _logdir / "lcs_correct_first_validation.log"

class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.orig = sys.stdout
    def write(self, s):
        try: self.orig.write(s)
        except: pass
        self.f.write(s)
        self.f.flush()
    def flush(self):
        try: self.orig.flush()
        except: pass
        self.f.flush()

sys.stdout = Tee(LOGFILE)
sys.stderr = sys.stdout

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

t0 = time.time()

ROOT = Path(__file__).resolve().parents[1]
LCS_PRED = ROOT / "analysis" / "thesis_experiments" / "external_validation_himawari_full_predictions.csv"
TRAIN_PRED = ROOT / "analysis" / "thesis_experiments" / "external_validation_himawari_full_train_predictions.csv"
OOF_PRED = ROOT / "analysis" / "experimental_shape_magnitude" / "true_tier_moe_xgb" / "himawari_fullfeature_oof.csv"
LCS_SELECTION = ROOT / "Thesis" / "results" / "01_stations" / "station_selection_lcs.csv"
THESIS_META = ROOT / "analysis" / "thesis_audit" / "station_selection_final.csv"
THESIS_DATA = ROOT / "data" / "merged" / "unified_thesis_v2.csv"
OUT_DIR = ROOT / "analysis" / "experimental_shape_magnitude" / "himawari_safe_selector"

PM_MIN, PM_MAX = 0.1, 250.0
FLATLINE_HOURS = 24
MIN_ROWS, MIN_DAYS = 500, 20
K_PRIOR = 10
KM_SCALE = 60.0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def spatial_prior(lat, lon, train_coords, train_means, global_mean):
    if not np.isfinite(lat) or not np.isfinite(lon):
        return global_mean, np.nan
    dists = {}
    for sid, (tlat, tlon) in train_coords.items():
        dists[sid] = haversine(lat, lon, tlat, tlon)
    sorted_ids = sorted(dists, key=dists.get)[:K_PRIOR]
    d = np.array([dists[s] for s in sorted_ids])
    v = np.array([train_means[s] for s in sorted_ids])
    w = np.exp(-((d / KM_SCALE)**2))
    if w.sum() < 1e-12:
        return float(np.mean(v)), float(d[0])
    return float(np.dot(w / w.sum(), v)), float(d[0])


def pm_class(x):
    if x < 10: return "low"
    if x < 20: return "moderate_low"
    if x < 35: return "moderate"
    return "high"


def classify_pm25_3(x):
    if x < 20: return "low"
    if x < 35: return "moderate"
    return "high"


def safe_r2(y, p):
    if len(y) < 3 or np.std(y) < 1e-9: return np.nan
    return float(r2_score(y, p))


def clean_station(g):
    g = g.sort_values("ts").copy()
    pm = pd.to_numeric(g["PM2.5"], errors="coerce")
    valid = pm.between(PM_MIN, PM_MAX)
    rounded = pm.round(1)
    run_id = rounded.ne(rounded.shift()).cumsum()
    run_len = run_id.map(run_id.value_counts())
    flatline = valid & run_len.ge(FLATLINE_HOURS)
    out = g.loc[valid & ~flatline].copy()
    days = out["ts"].dt.date.nunique() if len(out) else 0
    return out, len(out) >= MIN_ROWS and days >= MIN_DAYS


def eval_variant(df, pred_col, variant):
    rows = []
    for sid, g in df.groupby("stationId", sort=False):
        y = g["PM2.5"].astype(float).values
        p = g[pred_col].astype(float).values
        am, pm_ = float(np.mean(y)), float(np.mean(p))
        daily = g.assign(a=y, p=p).groupby(g["ts"].dt.date)[["a", "p"]].mean().dropna()
        dr2 = safe_r2(daily["a"].values, daily["p"].values) if len(daily) >= 3 else np.nan
        ac, pc = classify_pm25_3(am), classify_pm25_3(pm_)
        rows.append({
            "variant": variant, "station_id": str(sid),
            "station_name": str(g.iloc[0].get("station_name", "")),
            "n_rows": len(g), "actual_mean": am, "pred_mean": pm_,
            "bias": pm_ - am, "abs_mean_error": abs(pm_ - am),
            "hourly_r2": safe_r2(y, p),
            "hourly_rmse": float(math.sqrt(mean_squared_error(y, p))),
            "daily_r2": dr2,
            "actual_class": ac, "pred_class": pc,
            "class_flip": ac != pc,
            "high_to_nonhigh": ac == "high" and pc != "high",
            "nonhigh_to_high": ac != "high" and pc == "high",
            "nearest_train_km": float(g.iloc[0].get("nearest_train_km", np.nan)),
        })
    met = pd.DataFrame(rows)
    y_all = df["PM2.5"].astype(float).values
    p_all = df[pred_col].astype(float).values
    return {
        "variant": variant,
        "n_stations": int(met["station_id"].nunique()),
        "n_rows": len(df),
        "pooled_r2": safe_r2(y_all, p_all),
        "pooled_rmse": float(math.sqrt(mean_squared_error(y_all, p_all))),
        "mean_stn_r2": float(met["hourly_r2"].mean()),
        "median_stn_r2": float(met["hourly_r2"].median()),
        "pos_r2": int((met["hourly_r2"] > 0).sum()),
        "mean_daily_r2": float(met["daily_r2"].mean()),
        "mean_abs_err": float(met["abs_mean_error"].mean()),
        "flips": int(met["class_flip"].sum()),
        "h2nh": int(met["high_to_nonhigh"].sum()),
        "nh2h": int(met["nonhigh_to_high"].sum()),
    }, met


# ── 1. Load data ─────────────────────────────────────────────────
print("=" * 72)
print("LCS EXTERNAL VALIDATION: correct-first Himawari pipeline")
print("=" * 72)

print("\n1. Loading data...")
lcs_df = pd.read_csv(LCS_PRED, dtype={"stationId": str}, parse_dates=["ts"])
train_pred = pd.read_csv(TRAIN_PRED, dtype={"stationId": str}, parse_dates=["ts"])
oof = pd.read_csv(OOF_PRED, dtype={"station_id": str}, parse_dates=["ts"])

lcs_meta = pd.read_csv(LCS_SELECTION, dtype={"station_id": str})
thesis_meta = pd.read_csv(THESIS_META, dtype={"stationId": str})
thesis_ids = set(thesis_meta["stationId"].astype(str))
pass_ids = set(lcs_meta[lcs_meta["lcs_flag"] == "pass"]["station_id"].astype(str))

lcs_df = lcs_df[lcs_df["stationId"].isin(pass_ids) & ~lcs_df["stationId"].isin(thesis_ids)].copy()
print(f"  LCS raw: {len(lcs_df)} rows, {lcs_df['stationId'].nunique()} stations")


# ── 2. Clean LCS ─────────────────────────────────────────────────
print("\n2. Cleaning LCS data...")
cleaned_parts = []
for _, g in lcs_df.groupby("stationId", sort=False):
    c, usable = clean_station(g)
    if usable:
        cleaned_parts.append(c)
clean_df = pd.concat(cleaned_parts, ignore_index=True) if cleaned_parts else pd.DataFrame()
lcs_sids = sorted(clean_df["stationId"].unique())
print(f"  Cleaned: {len(clean_df)} rows, {len(lcs_sids)} stations")


# ── 3. Spatial prior ─────────────────────────────────────────────
print("\n3. Computing spatial prior...")
thesis_pm = pd.read_csv(THESIS_DATA, usecols=["stationId", "PM2.5"], dtype={"stationId": str})
train_means = {str(k): float(v) for k, v in thesis_pm.groupby("stationId")["PM2.5"].mean().items()}
train_coords = {}
for _, row in thesis_meta.iterrows():
    sid = str(row["stationId"])
    if np.isfinite(row["lat"]) and np.isfinite(row["lon"]):
        train_coords[sid] = (float(row["lat"]), float(row["lon"]))
global_mean = float(np.mean(list(train_means.values())))

for sid in lcs_sids:
    g = clean_df[clean_df["stationId"] == sid]
    first = g.iloc[0]
    lat = float(first.get("latitude", np.nan))
    lon = float(first.get("longitude", np.nan))
    prior, nearest = spatial_prior(lat, lon, train_coords, train_means, global_mean)
    clean_df.loc[clean_df["stationId"] == sid, "spatial_prior_pm25"] = prior
    clean_df.loc[clean_df["stationId"] == sid, "nearest_train_km"] = nearest
print(f"  Prior computed for {len(lcs_sids)} stations")


# ── 4. Train HGB correction on 40 thesis stations OOF ────────────
print("\n4. Training HGB correction on thesis station OOF predictions...")

moe_oof = oof[oof["config"] == "true_tier_moe_expert"][
    ["station_id", "ts", "y_pred", "y_true"]
].rename(columns={"station_id": "stationId", "y_pred": "moe_expert"}).copy()

moe_oof["hour"] = moe_oof["ts"].dt.hour
moe_oof["month"] = moe_oof["ts"].dt.month
moe_oof["hour_sin"] = np.sin(2 * np.pi * moe_oof["hour"] / 24)
moe_oof["hour_cos"] = np.cos(2 * np.pi * moe_oof["hour"] / 24)
moe_oof["month_sin"] = np.sin(2 * np.pi * moe_oof["month"] / 12)
moe_oof["month_cos"] = np.cos(2 * np.pi * moe_oof["month"] / 12)
moe_oof["pred_station_mean"] = moe_oof.groupby("stationId")["moe_expert"].transform("mean")
moe_oof["pred_anom_station"] = moe_oof["moe_expert"] - moe_oof["pred_station_mean"]
moe_oof["pred_station_hour_mean"] = moe_oof.groupby(["stationId", "hour"])["moe_expert"].transform("mean")
moe_oof["pred_anom_hour"] = moe_oof["moe_expert"] - moe_oof["pred_station_hour_mean"]

BASE_CONTEXT = [
    "moe_expert", "pred_station_mean", "pred_anom_station", "pred_anom_hour",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

train_sids = sorted(moe_oof["stationId"].unique())
sample_n = min(700, moe_oof.groupby("stationId").size().min())
sampled = moe_oof.groupby("stationId", group_keys=False).sample(
    n=sample_n, replace=True, random_state=142)

pre = ColumnTransformer(
    transformers=[("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), BASE_CONTEXT)], remainder="drop")
reg = HistGradientBoostingRegressor(
    max_iter=45, learning_rate=0.05, max_leaf_nodes=7,
    min_samples_leaf=180, l2_regularization=1.5, random_state=42)
hgb_model = Pipeline([("pre", pre), ("reg", reg)])
hgb_model.fit(sampled[BASE_CONTEXT], np.log1p(sampled["y_true"].clip(lower=0)))
print(f"  HGB trained on {len(sampled)} samples from {len(train_sids)} stations")


# ── 5. Apply pipeline to LCS ─────────────────────────────────────
print("\n5. Applying pipeline to LCS stations...")

clean_df["pred_raw"] = pd.to_numeric(clean_df["pred_PM25"], errors="coerce")
clean_df["hour"] = clean_df["ts"].dt.hour
clean_df["month"] = clean_df["ts"].dt.month
clean_df["hour_sin"] = np.sin(2 * np.pi * clean_df["hour"] / 24)
clean_df["hour_cos"] = np.cos(2 * np.pi * clean_df["hour"] / 24)
clean_df["month_sin"] = np.sin(2 * np.pi * clean_df["month"] / 12)
clean_df["month_cos"] = np.cos(2 * np.pi * clean_df["month"] / 12)
clean_df["pred_station_mean"] = clean_df.groupby("stationId")["pred_raw"].transform("mean")
clean_df["pred_anom_station"] = clean_df["pred_raw"] - clean_df["pred_station_mean"]
clean_df["pred_station_hour_mean"] = clean_df.groupby(["stationId", "hour"])["pred_raw"].transform("mean")
clean_df["pred_anom_hour"] = clean_df["pred_raw"] - clean_df["pred_station_hour_mean"]

lcs_features = clean_df[["pred_raw", "pred_station_mean", "pred_anom_station",
                          "pred_anom_hour", "hour_sin", "hour_cos",
                          "month_sin", "month_cos"]].copy()
lcs_features.columns = BASE_CONTEXT

hgb_pred = np.expm1(hgb_model.predict(lcs_features))
clean_df["pred_corrected"] = np.clip(hgb_pred, 1.0, 250.0).astype("float32")
print(f"  HGB correction applied to {len(clean_df)} rows")

# Prior shift variants
for alpha in [0.30, 0.45, 0.55]:
    for base_col, prefix in [("pred_raw", "raw"), ("pred_corrected", "corr")]:
        col = f"pred_{prefix}_shift_a{int(alpha*100)}"
        vals = clean_df[base_col].values.copy()
        for sid in lcs_sids:
            mask = (clean_df["stationId"] == sid).values
            base_mean = float(vals[mask].mean())
            prior = float(clean_df.loc[mask, "spatial_prior_pm25"].iloc[0])
            shift = alpha * (prior - base_mean)
            vals[mask] = vals[mask] + shift
        clean_df[col] = np.clip(vals, 1.0, 250.0)

# Blend variants: corrected_shifted + raw_shifted at different alphas
for shift_a in [0.30, 0.45, 0.55]:
    raw_col = f"pred_raw_shift_a{int(shift_a*100)}"
    corr_col = f"pred_corr_shift_a{int(shift_a*100)}"
    for blend_a in [0.30, 0.50, 0.70]:
        col = f"pred_blend_s{int(shift_a*100)}_b{int(blend_a*100)}"
        clean_df[col] = np.clip(
            (1 - blend_a) * clean_df[raw_col] + blend_a * clean_df[corr_col],
            1.0, 250.0
        )

# Fixed guard (matching existing pipeline)
for base_col, guard_col in [
    ("pred_raw_shift_a45", "pred_raw_guarded"),
    ("pred_corr_shift_a45", "pred_corr_guarded"),
    ("pred_blend_s45_b50", "pred_blend_guarded"),
]:
    vals = clean_df[base_col].values.copy()
    for sid in lcs_sids:
        mask = (clean_df["stationId"] == sid).values
        pred_mean = float(vals[mask].mean())
        prior = float(clean_df.loc[mask, "spatial_prior_pm25"].iloc[0])
        if prior >= 35.0 and pred_mean >= 18.0 and pred_mean < 35.0:
            target = min(max(prior, 35.0), 55.0)
            vals[mask] = np.clip(vals[mask] + (target - pred_mean), 1.0, 250.0)
        elif prior < 20.0 and pred_mean >= 35.0:
            vals[mask] = np.clip(vals[mask] + (34.5 - pred_mean), 1.0, 250.0)
        elif prior < 18.0 and pred_mean >= 25.0:
            vals[mask] = np.clip(vals[mask] + (24.5 - pred_mean), 1.0, 250.0)
    clean_df[guard_col] = vals


# ── 6. Evaluate all variants ──────────────────────────────────────
print("\n6. Evaluating variants...")

variants = {
    "1_raw_xgb":                 "pred_raw",
    "2_raw_shift_a45":           "pred_raw_shift_a45",
    "3_raw_guarded":             "pred_raw_guarded",
    "4_corrected_raw":           "pred_corrected",
    "5_corr_shift_a45":          "pred_corr_shift_a45",
    "6_corr_guarded":            "pred_corr_guarded",
    "7_blend_s45_b50":           "pred_blend_s45_b50",
    "8_blend_guarded":           "pred_blend_guarded",
    "9_blend_s30_b50":           "pred_blend_s30_b50",
    "10_blend_s55_b50":          "pred_blend_s55_b50",
    "11_blend_s45_b30":          "pred_blend_s45_b30",
    "12_blend_s45_b70":          "pred_blend_s45_b70",
}

all_summaries = []
all_metrics = []
for name, col in variants.items():
    summary, met = eval_variant(clean_df, col, name)
    all_summaries.append(summary)
    all_metrics.append(met)
    print(f"  {name:<28s}  mean_r2={summary['mean_stn_r2']:+.4f}  pool_r2={summary['pooled_r2']:+.4f}  "
          f"flips={summary['flips']:2d}  h2nh={summary['h2nh']}  nh2h={summary['nh2h']}")


# ── 7. Summary ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

sum_df = pd.DataFrame(all_summaries)
met_df = pd.concat(all_metrics, ignore_index=True)

print(f"\n  {'Variant':<28s}  mean_r2  pool_r2  median  pos  flips  h2nh  nh2h  daily_r2")
print(f"  {'-'*95}")
for _, r in sum_df.iterrows():
    print(f"  {r['variant']:<28s}  {r['mean_stn_r2']:+.4f}  {r['pooled_r2']:+.4f}  "
          f"{r['median_stn_r2']:+.4f}  {r['pos_r2']:3.0f}  {r['flips']:5.0f}  "
          f"{r['h2nh']:4.0f}  {r['nh2h']:4.0f}  {r['mean_daily_r2']:+.4f}")

print(f"\n  Reference (existing no-GHAP validation):")
print(f"  {'guarded_fixed_no_ghap':<28s}  +0.1083  +0.2865  +0.2529   33     18     4     9  +0.0139")
print(f"  {'external_xgb_no_ghap':<28s}  +0.0047  +0.1271  +0.0830   29     31    19     3  -0.1382")

# Save results
sum_df.to_csv(OUT_DIR / "lcs_correct_first_summary.csv", index=False, encoding="utf-8-sig")
met_df.to_csv(OUT_DIR / "lcs_correct_first_station_metrics.csv", index=False, encoding="utf-8-sig")

# Save predictions
pred_cols = ["stationId", "station_name", "ts", "PM2.5",
             "pred_raw", "pred_corrected", "pred_raw_shift_a45",
             "pred_corr_shift_a45", "pred_blend_s45_b50", "pred_blend_guarded",
             "spatial_prior_pm25", "nearest_train_km", "latitude", "longitude"]
available_cols = [c for c in pred_cols if c in clean_df.columns]
clean_df[available_cols].to_csv(
    OUT_DIR / "lcs_correct_first_predictions.csv", index=False, encoding="utf-8-sig")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("DONE")
