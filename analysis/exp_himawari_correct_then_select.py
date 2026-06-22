"""Correct-first, select-second: HGB correction on MoE BEFORE kNN selection.

Matches the dual model architecture: correction adjusts the base prediction level
BEFORE routing/selection picks the best candidate.

Previous experiment (select-first): HGB correction on kNN-selected output
produced +0.091 raw, +0.166 blend — no improvement over kNN alone (+0.163).
Hypothesis: correction needs a weaker input (MoE +0.089) to have room to help.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

_logdir = Path(__file__).resolve().parent / "experimental_shape_magnitude" / "himawari_safe_selector"
_logdir.mkdir(parents=True, exist_ok=True)
LOGFILE = _logdir / "correct_then_select_results.log"

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


# ── Load data ─────────────────────────────────────────────────────
print("Loading data...")
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
stn_masks = {sid: (df["station_id"] == sid).values for sid in sids}
print(f"  {len(df)} rows, {n} stations, {time.time()-t0:.1f}s")

# ── LOO spatial prior ─────────────────────────────────────────────
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
ghap_tier = dict(zip(meta["station_id"], meta["ghap_tier"]))
phase1_tier = dict(zip(meta["station_id"], meta["phase1_tier"]))

# ── kNN feature distances (precompute once) ───────────────────────
num_fc = [c for c in feat.columns if pd.api.types.is_numeric_dtype(feat[c])]
fm = feat[num_fc].fillna(feat[num_fc].median()).values
fs = StandardScaler().fit_transform(fm)
fdist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        fdist[i, j] = np.sqrt(np.sum((fs[i] - fs[j])**2))

# ── Join station-level features ───────────────────────────────────
print("Joining station-level features...")
stn_feat = features.drop_duplicates("station_id").set_index("station_id")

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
    df[col] = df["station_id"].map(stn_feat[col].to_dict()).astype(float)

# ── Temporal features for MoE correction ──────────────────────────
df["hour"] = df["ts"].dt.hour
df["month"] = df["ts"].dt.month
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["moe_station_mean"] = df.groupby("station_id")["moe_expert"].transform("mean")
df["moe_anom_station"] = df["moe_expert"] - df["moe_station_mean"]
df["moe_station_hour_mean"] = df.groupby(["station_id", "hour"])["moe_expert"].transform("mean")
df["moe_anom_hour"] = df["moe_expert"] - df["moe_station_hour_mean"]

BASE_CONTEXT = [
    "moe_expert", "moe_station_mean", "moe_anom_station", "moe_anom_hour",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

def clean(cols):
    return [c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 100]

feature_sets = {
    "A_base_only":     clean(BASE_CONTEXT),
    "B_modis_5":       clean(BASE_CONTEXT + MODIS_5),
    "C_himawari_stn":  clean(BASE_CONTEXT + HIMAWARI_STATION_AOD),
    "D_sat_gases":     clean(BASE_CONTEXT + SATELLITE_GASES),
    "G_him_stn_gases": clean(BASE_CONTEXT + HIMAWARI_STATION_AOD + SATELLITE_GASES),
}

for name, cols in feature_sets.items():
    extra = len(cols) - len(BASE_CONTEXT)
    print(f"  {name}: {len(cols)} features (+{extra} station-level)")


# ── LOSO HGB correction ──────────────────────────────────────────
def loso_hgb_correct(df, cols):
    corrected = np.zeros(len(df), dtype="float32")
    for fold, sid in enumerate(sids):
        train_mask = df["station_id"] != sid
        test_mask = ~train_mask
        train_df = df.loc[train_mask]
        sample_n = min(700, train_df.groupby("station_id").size().min())
        sampled = train_df.groupby("station_id", group_keys=False).sample(
            n=sample_n, replace=True, random_state=142 + fold)
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


# ── Build safe candidates from a given base column ────────────────
def build_safe_candidates(df, base_col):
    """Build safe candidates using base_col as the base prediction."""
    base_means = {sid: float(df.loc[stn_masks[sid], base_col].mean()) for sid in sids}
    stn_info = {}
    for sid in sids:
        stn_info[sid] = {
            "base_mean": base_means[sid],
            "t0_mean": float(df.loc[stn_masks[sid], "tierexpert_t0"].mean()),
            "t1_mean": float(df.loc[stn_masks[sid], "tierexpert_t1"].mean()),
            "t2_mean": float(df.loc[stn_masks[sid], "tierexpert_t2"].mean()),
            "t3_mean": float(df.loc[stn_masks[sid], "tierexpert_t3"].mean()),
        }

    tier_sources_local = {
        "ghap": ghap_tier,
        "phase1": phase1_tier,
        "baseclass": {sid: tier_for_pm(base_means[sid]) for sid in sids},
        "prior": {sid: tier_for_pm(prior_map[sid]) for sid in sids},
    }

    cand_dict = {}

    def _build_cand(tier_map, alpha):
        vals = df[base_col].values.copy()
        for sid in sids:
            t = tier_map.get(sid, "t2")
            exp_col = emap[t]
            exp_key = {"t0": "t0_mean", "t1": "t1_mean", "t2": "t2_mean", "t3": "t3_mean"}[t]
            bm = stn_info[sid]["base_mean"]
            em = stn_info[sid][exp_key]
            if pm_class((1 - alpha) * bm + alpha * em) == pm_class(bm):
                vals[stn_masks[sid]] = (1 - alpha) * df.loc[stn_masks[sid], base_col].values + alpha * df.loc[stn_masks[sid], exp_col].values
        return vals

    for src, tmap in tier_sources_local.items():
        for a in [0.25, 0.35, 0.50, 0.65, 0.80]:
            cand_dict[f"cand_{src}_a{int(a*100)}"] = _build_cand(tmap, a)

    for a in [0.25, 0.40, 0.50, 0.65]:
        vals = df[base_col].values.copy()
        for sid in sids:
            if ghap_tier.get(sid) == "t3":
                bm = stn_info[sid]["base_mean"]
                if pm_class((1 - a) * bm + a * stn_info[sid]["t3_mean"]) == pm_class(bm):
                    vals[stn_masks[sid]] = (1 - a) * df.loc[stn_masks[sid], base_col].values + a * df.loc[stn_masks[sid], "tierexpert_t3"].values
        cand_dict[f"cand_ghap_high_t3_a{int(a*100)}"] = vals

    for a in [0.25, 0.40, 0.55, 0.70]:
        vals = df[base_col].values.copy()
        for sid in sids:
            bm = stn_info[sid]["base_mean"]
            shift = a * (prior_map[sid] - bm)
            if pm_class(bm + shift) == pm_class(bm):
                vals[stn_masks[sid]] = df.loc[stn_masks[sid], base_col].values + shift
        cand_dict[f"cand_prior_shift_a{int(a*100)}"] = vals

    for ash in [0.30, 0.50]:
        for abl in [0.25, 0.50]:
            for src in ["ghap", "prior"]:
                vals = df[base_col].values.copy()
                tmap = tier_sources_local[src]
                for sid in sids:
                    bm = stn_info[sid]["base_mean"]
                    t = tmap.get(sid, "t2")
                    ec = emap[t]
                    shift = ash * (prior_map[sid] - bm)
                    bl = (1 - abl) * (df.loc[stn_masks[sid], base_col].values + shift) + abl * df.loc[stn_masks[sid], ec].values
                    if pm_class(float(np.mean(bl))) == pm_class(bm):
                        vals[stn_masks[sid]] = bl
                cand_dict[f"cand_combo_{src}_s{int(ash*100)}_b{int(abl*100)}"] = vals

    return cand_dict


# ── kNN k=3 selector on a set of candidates ───────────────────────
def run_knn_selector(df, base_col, cand_dict, k=3):
    all_choices = {base_col: df[base_col].values}
    all_choices.update(cand_dict)
    choice_names = list(all_choices.keys())

    per_stn = {}
    for sid in sids:
        m = stn_masks[sid]
        y = df.loc[m, "y_true"].values
        per_stn[sid] = {}
        for cname, vals in all_choices.items():
            p = vals[m]
            v = ~(np.isnan(y) | np.isnan(p))
            if v.sum() >= 10:
                per_stn[sid][cname] = r2_score(y[v], p[v])

    choices = {}
    for i, sid in enumerate(sids):
        d = fdist[i].copy()
        d[i] = np.inf
        ni = np.argpartition(d, k)[:k]
        cs = {}
        for cname in choice_names:
            sc = [per_stn[sids[j]][cname] for j in ni if cname in per_stn[sids[j]]]
            if sc:
                cs[cname] = np.mean(sc)
        choices[sid] = max(cs, key=cs.get) if cs else base_col

    selected = df[base_col].values.copy()
    for sid, cname in choices.items():
        selected[stn_masks[sid]] = all_choices[cname][stn_masks[sid]]

    return selected, choices


# ── Reference: uncorrected kNN (from v3) ──────────────────────────
print("\n" + "=" * 80)
print("REFERENCE: uncorrected kNN k=3 (select on raw MoE)")
print("=" * 80)

uncorr_cands = build_safe_candidates(df, "moe_expert")
uncorr_sel, uncorr_choices = run_knn_selector(df, "moe_expert", uncorr_cands, k=3)
df["knn_uncorrected"] = uncorr_sel
r_ref = eval_stream(df, "knn_uncorrected")
print(f"  kNN k=3 (uncorrected): mean_r2={r_ref['mean_r2']:+.4f}  pool_r2={r_ref['pool_r2']:+.4f}  "
      f"h2nh={r_ref['h2nh']}  flips={r_ref['flips']}")

r_moe = eval_stream(df, "moe_expert")
print(f"  MoE expert (base):     mean_r2={r_moe['mean_r2']:+.4f}  pool_r2={r_moe['pool_r2']:+.4f}  "
      f"h2nh={r_moe['h2nh']}  flips={r_moe['flips']}")


# ── Main experiment: correct-first, select-second ─────────────────
print("\n" + "=" * 80)
print("CORRECT-FIRST, SELECT-SECOND")
print("=" * 80)

results = {}

for name, cols in feature_sets.items():
    print(f"\n  [{name}] LOSO HGB correction on moe_expert...", flush=True)
    t1 = time.time()

    corrected = loso_hgb_correct(df, cols)
    corr_col = f"corrected_{name}"
    df[corr_col] = corrected

    r_corr_raw = eval_stream(df, corr_col)
    print(f"    HGB corrected (raw):  mean_r2={r_corr_raw['mean_r2']:+.4f}  pool_r2={r_corr_raw['pool_r2']:+.4f}  "
          f"h2nh={r_corr_raw['h2nh']}  flips={r_corr_raw['flips']}")

    # Build safe candidates from corrected base
    print(f"    Building safe candidates from corrected base...", flush=True)
    cand_dict = build_safe_candidates(df, corr_col)
    n_cands = len(cand_dict)

    # kNN k=3 on corrected candidates
    selected, choices = run_knn_selector(df, corr_col, cand_dict, k=3)
    sel_col = f"knn_corrfirst_{name}"
    df[sel_col] = selected
    r_sel = eval_stream(df, sel_col)
    print(f"    kNN on corrected:     mean_r2={r_sel['mean_r2']:+.4f}  pool_r2={r_sel['pool_r2']:+.4f}  "
          f"h2nh={r_sel['h2nh']}  flips={r_sel['flips']}  ({n_cands} candidates)")

    # Also test: alpha blend of corrected + uncorrected kNN
    best_ba, best_br = 0.0, r_sel["mean_r2"]
    for alpha in [0.15, 0.25, 0.35, 0.50]:
        df["_blend"] = (1 - alpha) * uncorr_sel + alpha * selected
        rb = eval_stream(df, "_blend")
        if rb["mean_r2"] > best_br:
            best_ba, best_br = alpha, rb["mean_r2"]
        df.drop(columns=["_blend"], inplace=True)

    # Also test: direct kNN on corrected (no tier blending, just corrected as base for kNN)
    # This tests if the correction alone + simple prior shifts is enough
    simple_cands = {}
    corr_means = {sid: float(df.loc[stn_masks[sid], corr_col].mean()) for sid in sids}
    for a in [0.25, 0.40, 0.55, 0.70]:
        vals = df[corr_col].values.copy()
        for sid in sids:
            bm = corr_means[sid]
            shift = a * (prior_map[sid] - bm)
            if pm_class(bm + shift) == pm_class(bm):
                vals[stn_masks[sid]] = df.loc[stn_masks[sid], corr_col].values + shift
        simple_cands[f"prior_shift_a{int(a*100)}"] = vals
    simple_sel, _ = run_knn_selector(df, corr_col, simple_cands, k=3)
    df["_simple"] = simple_sel
    r_simple = eval_stream(df, "_simple")
    print(f"    kNN simple (corr+prior): mean_r2={r_simple['mean_r2']:+.4f}  pool_r2={r_simple['pool_r2']:+.4f}")
    df.drop(columns=["_simple"], inplace=True)

    extra = len(cols) - len(BASE_CONTEXT)
    results[name] = {
        "n_extra": extra,
        "raw": r_corr_raw,
        "knn_corrfirst": r_sel,
        "blend_with_uncorr": {"alpha": best_ba, "mean_r2": best_br},
        "knn_simple": r_simple,
    }
    print(f"    blend w/ uncorr kNN:  a={best_ba:.2f} → mean_r2={best_br:+.4f}")
    print(f"    time: {time.time()-t1:.0f}s", flush=True)


# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: correct-first vs select-first")
print("=" * 80)

print(f"\n  {'Stream':<35s}  mean_r2  pool_r2  h2nh  flips")
print(f"  {'-'*70}")
print(f"  {'MoE expert (base)':<35s}  {r_moe['mean_r2']:+.4f}  {r_moe['pool_r2']:+.4f}   {r_moe['h2nh']:3d}   {r_moe['flips']:3d}")
print(f"  {'kNN k=3 uncorrected (ref)':<35s}  {r_ref['mean_r2']:+.4f}  {r_ref['pool_r2']:+.4f}   {r_ref['h2nh']:3d}   {r_ref['flips']:3d}")
print()

for name, res in results.items():
    tag = f"{name} (+{res['n_extra']} stn feat)"
    r_raw = res["raw"]
    r_sel = res["knn_corrfirst"]
    ba = res["blend_with_uncorr"]
    print(f"  {tag:<35s}")
    print(f"    HGB raw:                       {r_raw['mean_r2']:+.4f}  {r_raw['pool_r2']:+.4f}   {r_raw['h2nh']:3d}   {r_raw['flips']:3d}")
    print(f"    correct→kNN:                   {r_sel['mean_r2']:+.4f}  {r_sel['pool_r2']:+.4f}   {r_sel['h2nh']:3d}   {r_sel['flips']:3d}")
    print(f"    blend w/ uncorr kNN (a={ba['alpha']:.2f}):  {ba['mean_r2']:+.4f}")
    print()

print(f"  {'Dual-AOD model (target)':<35s}  +0.197")
print(f"  {'Safe oracle (ceiling)':<35s}  +0.206")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("DONE")
