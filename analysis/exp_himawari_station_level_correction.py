"""Test station-level features in the HGB correction.

Key insight: the Himawari base XGBoost only uses HOURLY features.
Station-level features (MODIS seasonal AOD, Himawari climatology, TROPOMI gases,
terrain, nightlights) are NEW information the base model never saw.

The dual model's correction adds 5 MODIS features. We test whether:
1. Those same 5 MODIS features help Himawari correction (replicate dual model)
2. Himawari station-level AOD can substitute for MODIS
3. Satellite gases (NO2, SO2, CO) add value
4. Terrain/nightlights add value
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

_logdir = Path(__file__).resolve().parent / "experimental_shape_magnitude" / "himawari_safe_selector"
_logdir.mkdir(parents=True, exist_ok=True)
LOGFILE = _logdir / "station_level_correction_results.log"

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

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

t0 = time.time()

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "analysis" / "experimental_shape_magnitude"
TIER_OOF = EXP / "true_tier_moe_xgb" / "himawari_tierexperts_oof.csv"
BASE_OOF = EXP / "true_tier_moe_xgb" / "himawari_fullfeature_oof.csv"
TIER_SUMMARY = EXP / "true_tier_moe_xgb" / "himawari_tierexperts.csv"
FEATURES = EXP / "station_feature_table.csv"

def pm_class(x):
    if x < 10: return "low"
    if x < 20: return "moderate_low"
    if x < 35: return "moderate"
    return "high"

def tier_for_pm(x):
    if x < 10: return "t0"
    if x < 20: return "t1"
    if x < 35: return "t2"
    return "t3"

def eval_stream(df, col):
    stn_r2s, h2nh, flips = [], 0, 0
    y_all, p_all = [], []
    for sid, g in df.groupby("station_id"):
        y, p = g["y_true"].values, g[col].values
        v = ~(np.isnan(y) | np.isnan(p))
        y, p = y[v], p[v]
        y_all.extend(y); p_all.extend(p)
        if len(y) < 3 or np.std(y) < 1e-9: continue
        stn_r2s.append(float(r2_score(y, p)))
        true_cls, pred_cls = pm_class(np.mean(y)), pm_class(np.mean(p))
        if true_cls != pred_cls: flips += 1
        sev = {"low": 0, "moderate_low": 1, "moderate": 2, "high": 3}
        if sev[pred_cls] < sev[true_cls] and true_cls in ("moderate", "high") and pred_cls in ("low", "moderate_low"):
            h2nh += 1
    y_all, p_all = np.array(y_all), np.array(p_all)
    return {
        "mean_r2": float(np.mean(stn_r2s)), "pool_r2": float(r2_score(y_all, p_all)),
        "pool_rmse": float(np.sqrt(np.mean((y_all - p_all)**2))),
        "pos": sum(1 for r in stn_r2s if r > 0), "h2nh": h2nh, "flips": flips,
    }

# ── Load + build kNN k=3 (reuse logic from v3) ────────────────────
print("Loading data + building kNN k=3 selector...")
base_oof = pd.read_csv(BASE_OOF, dtype={"station_id": str}, parse_dates=["ts"])
tier_oof = pd.read_csv(TIER_OOF, dtype={"station_id": str}, parse_dates=["ts"])
meta = pd.read_csv(TIER_SUMMARY, dtype={"station_id": str}, encoding="utf-8-sig")
meta = meta[meta["config"] == "tierexpert_t0"][
    ["station_id", "station_name", "tier", "pm25_mean", "phase1_tier", "ghap_tier"]
].copy()

df = base_oof[base_oof["config"] == "no_t4f"][
    ["station_id", "ts", "y_true", "y_pred"]
].rename(columns={"y_pred": "no_t4f"})
moe = base_oof[base_oof["config"] == "true_tier_moe_expert"][
    ["station_id", "ts", "y_pred"]
].rename(columns={"y_pred": "moe_expert"})
df = df.merge(moe, on=["station_id", "ts"], how="left")
for cfg in ["tierexpert_t0", "tierexpert_t1", "tierexpert_t2", "tierexpert_t3"]:
    sub = tier_oof[tier_oof["config"] == cfg][["station_id", "ts", "y_pred"]].rename(columns={"y_pred": cfg})
    df = df.merge(sub, on=["station_id", "ts"], how="left")
df = df.merge(meta, on="station_id", how="left")

sids = sorted(df["station_id"].unique())
n = len(sids)
emap = {"t0": "tierexpert_t0", "t1": "tierexpert_t1", "t2": "tierexpert_t2", "t3": "tierexpert_t3"}
stn = df.groupby("station_id").agg(
    moe_mean=("moe_expert", "mean"),
    t0_mean=("tierexpert_t0", "mean"), t1_mean=("tierexpert_t1", "mean"),
    t2_mean=("tierexpert_t2", "mean"), t3_mean=("tierexpert_t3", "mean"),
).to_dict("index")
ghap_tier = dict(zip(meta["station_id"], meta["ghap_tier"]))
phase1_tier = dict(zip(meta["station_id"], meta["phase1_tier"]))
stn_masks = {sid: (df["station_id"] == sid).values for sid in sids}

# LOO prior + candidates
features = pd.read_csv(FEATURES, dtype={"station_id": str})
feat = features.drop_duplicates("station_id").set_index("station_id").reindex(sids)
lat_col = "lat" if "lat" in feat.columns else "latitude"
lon_col = "lon" if "lon" in feat.columns else "longitude"
lats = pd.to_numeric(feat[lat_col], errors="coerce").fillna(feat[lat_col].median()).values
lons = pd.to_numeric(feat[lon_col], errors="coerce").fillna(feat[lon_col].median()).values
km_mat = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        R = 6371.0
        dlat, dlon = np.radians(lats[j]-lats[i]), np.radians(lons[j]-lons[i])
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lats[i]))*np.cos(np.radians(lats[j]))*np.sin(dlon/2)**2
        km_mat[i,j] = km_mat[j,i] = R*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
actual_means = np.array([df[df["station_id"]==sid]["y_true"].mean() for sid in sids])

def loo_prior(k=10, km_scale=60.0):
    sim = np.exp(-((km_mat/km_scale)**2)); np.fill_diagonal(sim, 0.0)
    prior = np.zeros(n)
    for i in range(n):
        w = sim[i].copy(); w[i] = 0.0
        if k < n:
            idx = np.argpartition(w, -k)[-k:]; m = np.zeros(n, dtype=bool); m[idx] = True; w = np.where(m, w, 0.0)
        t = w.sum(); prior[i] = np.dot(w/t, actual_means) if t > 1e-12 else np.mean(actual_means)
    return dict(zip(sids, prior))

prior_map = loo_prior(k=10, km_scale=60.0)
tier_sources = {
    "ghap": ghap_tier, "phase1": phase1_tier,
    "baseclass": {sid: tier_for_pm(stn[sid]["moe_mean"]) for sid in sids},
    "prior": {sid: tier_for_pm(prior_map[sid]) for sid in sids},
}

def build_cand(df, tier_map, alpha):
    vals = df["moe_expert"].values.copy()
    for sid in sids:
        t = tier_map.get(sid, "t2"); exp_col = emap[t]
        exp_key = {"t0":"t0_mean","t1":"t1_mean","t2":"t2_mean","t3":"t3_mean"}[t]
        base_mean, exp_mean = stn[sid]["moe_mean"], stn[sid][exp_key]
        if pm_class((1-alpha)*base_mean + alpha*exp_mean) == pm_class(base_mean):
            vals[stn_masks[sid]] = (1-alpha)*df.loc[stn_masks[sid],"moe_expert"].values + alpha*df.loc[stn_masks[sid],exp_col].values
    return vals

for src, tmap in tier_sources.items():
    for a in [0.25, 0.35, 0.50, 0.65, 0.80]:
        df[f"cand_{src}_a{int(a*100)}"] = build_cand(df, tmap, a)
for a in [0.25, 0.40, 0.50, 0.65]:
    col = f"cand_ghap_high_t3_a{int(a*100)}"
    vals = df["moe_expert"].values.copy()
    for sid in sids:
        if ghap_tier.get(sid) == "t3":
            bm = stn[sid]["moe_mean"]
            if pm_class((1-a)*bm + a*stn[sid]["t3_mean"]) == pm_class(bm):
                vals[stn_masks[sid]] = (1-a)*df.loc[stn_masks[sid],"moe_expert"].values + a*df.loc[stn_masks[sid],"tierexpert_t3"].values
    df[col] = vals
for a in [0.25, 0.40, 0.55, 0.70]:
    col = f"cand_prior_shift_a{int(a*100)}"
    vals = df["moe_expert"].values.copy()
    for sid in sids:
        bm = stn[sid]["moe_mean"]; shift = a*(prior_map[sid]-bm)
        if pm_class(bm+shift) == pm_class(bm):
            vals[stn_masks[sid]] = df.loc[stn_masks[sid],"moe_expert"].values + shift
    df[col] = vals
for ash in [0.30, 0.50]:
    for abl in [0.25, 0.50]:
        for src in ["ghap", "prior"]:
            col = f"cand_combo_{src}_s{int(ash*100)}_b{int(abl*100)}"
            vals = df["moe_expert"].values.copy()
            tmap = tier_sources[src]
            for sid in sids:
                bm = stn[sid]["moe_mean"]; t = tmap.get(sid,"t2")
                ek = {"t0":"t0_mean","t1":"t1_mean","t2":"t2_mean","t3":"t3_mean"}[t]
                ec = emap[t]; shift = ash*(prior_map[sid]-bm)
                bl = (1-abl)*(df.loc[stn_masks[sid],"moe_expert"].values+shift) + abl*df.loc[stn_masks[sid],ec].values
                if pm_class(float(np.mean(bl))) == pm_class(bm): vals[stn_masks[sid]] = bl
            df[col] = vals

cand_cols = [c for c in df.columns if c.startswith("cand_")]
all_choices = ["moe_expert"] + cand_cols

# kNN k=3
per_stn = {}
for sid in sids:
    g = df[df["station_id"]==sid]; y = g["y_true"].values; per_stn[sid] = {}
    for cand in all_choices:
        p = g[cand].values; v = ~(np.isnan(y)|np.isnan(p))
        if v.sum() >= 10: per_stn[sid][cand] = r2_score(y[v], p[v])

num_fc = [c for c in feat.columns if pd.api.types.is_numeric_dtype(feat[c])]
fm = feat[num_fc].fillna(feat[num_fc].median()).values
fs = StandardScaler().fit_transform(fm)
fdist = np.zeros((n,n))
for i in range(n):
    for j in range(n): fdist[i,j] = np.sqrt(np.sum((fs[i]-fs[j])**2))

knn_choices = {}
for i, sid in enumerate(sids):
    d = fdist[i].copy(); d[i] = np.inf; ni = np.argpartition(d, 3)[:3]
    cs = {}
    for cand in all_choices:
        sc = [per_stn[sids[j]][cand] for j in ni if cand in per_stn[sids[j]]]
        if sc: cs[cand] = np.mean(sc)
    knn_choices[sid] = max(cs, key=cs.get) if cs else "moe_expert"

vals = df["moe_expert"].values.copy()
for sid, cand in knn_choices.items():
    vals[stn_masks[sid]] = df.loc[stn_masks[sid], cand].values
df["knn_selected"] = vals
knn_r = eval_stream(df, "knn_selected")
print(f"  kNN k=3: mean_r2={knn_r['mean_r2']:+.4f}, {time.time()-t0:.1f}s")

# ── Join station-level features to each row ────────────────────────
print("Joining station-level features...")
stn_feat = features.drop_duplicates("station_id").set_index("station_id")

# Define feature groups
MODIS_5 = ["modis_maod_center", "modis_maod_DJF", "modis_maod_MAM", "modis_maod_JJA", "modis_maod_SON"]

HIMAWARI_STATION_AOD = [
    "aoddir_aod_center_clim", "aoddir_aod_overall_clim",
    "aoddir_aod_clim_N", "aoddir_aod_clim_S", "aoddir_aod_clim_E", "aoddir_aod_clim_W",
    "aoddir_aod_directionality", "aoddir_aod_sector_std",
    "satall_fmf_center", "satall_faod_center",
]

SATELLITE_GASES = [
    "no2_no2_center", "no2_no2_center_DJF", "no2_no2_center_JJA",
    "satall_so2_center", "satall_co_center", "satall_hcho_center",
]

TERRAIN_NTL = [
    "topo_elevation_m", "topo_tpi", "topo_local_relief_m",
    "diag_ntl_center", "emit_ntl_center",
]

all_stn_cols = list(set(MODIS_5 + HIMAWARI_STATION_AOD + SATELLITE_GASES + TERRAIN_NTL))
available = [c for c in all_stn_cols if c in stn_feat.columns]
print(f"  {len(available)}/{len(all_stn_cols)} station-level features available")

for col in available:
    col_map = stn_feat[col].to_dict()
    df[col] = df["station_id"].map(col_map).astype(float)

# ── Build correction features ──────────────────────────────────────
df["hour"] = df["ts"].dt.hour
df["month"] = df["ts"].dt.month
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
df["pred_station_mean"] = df.groupby("station_id")["knn_selected"].transform("mean")
df["pred_anom_station"] = df["knn_selected"] - df["pred_station_mean"]
df["pred_station_hour_mean"] = df.groupby(["station_id", "hour"])["knn_selected"].transform("mean")
df["pred_anom_hour"] = df["knn_selected"] - df["pred_station_hour_mean"]

BASE_CONTEXT = [
    "knn_selected", "pred_station_mean", "pred_anom_station", "pred_anom_hour",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

def clean(cols):
    return [c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 100]

feature_sets = {
    "A_base_only":         clean(BASE_CONTEXT),
    "B_modis_5":           clean(BASE_CONTEXT + MODIS_5),
    "C_himawari_stn":      clean(BASE_CONTEXT + HIMAWARI_STATION_AOD),
    "D_sat_gases":         clean(BASE_CONTEXT + SATELLITE_GASES),
    "E_terrain_ntl":       clean(BASE_CONTEXT + TERRAIN_NTL),
    "F_modis_gases":       clean(BASE_CONTEXT + MODIS_5 + SATELLITE_GASES),
    "G_him_stn_gases":     clean(BASE_CONTEXT + HIMAWARI_STATION_AOD + SATELLITE_GASES),
    "H_all_station":       clean(BASE_CONTEXT + MODIS_5 + HIMAWARI_STATION_AOD + SATELLITE_GASES + TERRAIN_NTL),
}

for name, cols in feature_sets.items():
    extra = len(cols) - len(BASE_CONTEXT)
    print(f"  {name}: {len(cols)} features (+{extra} station-level)")

# ── LOSO HGB for each feature set ─────────────────────────────────
def run_loso_hgb(df, cols, label):
    corrected = np.zeros(len(df), dtype="float32")
    for fold, sid in enumerate(sids):
        train_mask = df["station_id"] != sid
        test_mask = ~train_mask
        train_df = df.loc[train_mask]
        sample_n = min(700, train_df.groupby("station_id").size().min())
        sampled = train_df.groupby("station_id", group_keys=False).sample(
            n=sample_n, replace=True, random_state=142+fold)
        pre = ColumnTransformer(
            transformers=[("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), cols)], remainder="drop")
        reg = HistGradientBoostingRegressor(
            max_iter=45, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=180, l2_regularization=1.5, random_state=42)
        model = Pipeline([("pre", pre), ("reg", reg)])
        model.fit(sampled[cols], np.log1p(sampled["y_true"].clip(lower=0)))
        pred = model.predict(df.loc[test_mask, cols])
        corrected[test_mask.values] = np.clip(np.expm1(pred), 1.0, 250.0).astype("float32")
    return corrected

print("\n" + "=" * 80)
print("LOSO HGB CORRECTION: station-level features")
print("=" * 80)

results = {}
for name, cols in feature_sets.items():
    print(f"\n  Running {name}...", flush=True)
    corrected = run_loso_hgb(df, cols, name)
    col_raw = f"hgb_{name}"
    df[col_raw] = corrected
    r_raw = eval_stream(df, col_raw)

    # Class-guarded
    col_g = f"hgb_{name}_g"
    df[col_g] = corrected.copy()
    for sid in sids:
        if pm_class(df.loc[stn_masks[sid], "knn_selected"].mean()) != pm_class(df.loc[stn_masks[sid], col_raw].mean()):
            df.loc[stn_masks[sid], col_g] = df.loc[stn_masks[sid], "knn_selected"].values
    r_g = eval_stream(df, col_g)

    # Best alpha blend
    best_ba, best_br = 0.0, knn_r["mean_r2"]
    for alpha in [0.15, 0.25, 0.35, 0.45]:
        df["_t"] = (1-alpha)*df["knn_selected"] + alpha*corrected
        rb = eval_stream(df, "_t")
        if rb["mean_r2"] > best_br: best_ba, best_br = alpha, rb["mean_r2"]
        df.drop(columns=["_t"], inplace=True)

    extra = len(cols) - len(BASE_CONTEXT)
    results[name] = {"raw": r_raw, "guarded": r_g, "blend_a": best_ba, "blend_r2": best_br, "n_extra": extra}
    print(f"    raw:     mean_r2={r_raw['mean_r2']:+.4f}  pool_r2={r_raw['pool_r2']:+.4f}  h2nh={r_raw['h2nh']}  flips={r_raw['flips']}")
    print(f"    guarded: mean_r2={r_g['mean_r2']:+.4f}  pool_r2={r_g['pool_r2']:+.4f}  h2nh={r_g['h2nh']}  flips={r_g['flips']}")
    if best_ba > 0:
        print(f"    blend:   mean_r2={best_br:+.4f}  (a={best_ba:.2f})")
    else:
        print(f"    blend:   no blend beats kNN alone")

# ── FINAL SUMMARY ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: What do station-level features add?")
print("=" * 80)
print(f"  kNN k=3 selector (no correction): mean_r2={knn_r['mean_r2']:+.4f}")
print()
print(f"  {'Feature set':25s} {'#extra':>6s} | {'raw':>8s} {'guarded':>8s} {'blend':>8s} {'blend_a':>7s}")
print(f"  {'-'*75}")
for name, r in results.items():
    ba_str = f"a={r['blend_a']:.2f}" if r['blend_a'] > 0 else "  none"
    print(f"  {name:25s} {r['n_extra']:>5d}  | {r['raw']['mean_r2']:+.4f}   {r['guarded']['mean_r2']:+.4f}   {r['blend_r2']:+.4f}   {ba_str}")

print(f"\n  MoE expert baseline: +0.0886")
print(f"  kNN k=3: {knn_r['mean_r2']:+.4f}")
print(f"  Safe oracle: +0.2061")
print(f"  Dual model ref: +0.197")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("DONE")
