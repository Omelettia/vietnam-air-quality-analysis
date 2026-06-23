"""Diagnostic: Can diverse stream predictions raise the kNN oracle ceiling?

Tests whether adding diverse-feature XGB predictions to the MoE safe
candidate pool increases the oracle or the kNN-achievable result.

Current pipeline: MoE expert → safe candidates (tier blends) → kNN k=3 → +0.163
Safe oracle: +0.206
Target: does expanded candidate set push oracle past 0.206?
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

def _repo_root():
    """Walk up to the repo root (dir containing data/merged) so this script runs
    correctly from anywhere, including Thesis/scripts/04_experiments/."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data" / "merged").is_dir():
            return parent
    return p.parents[1]

ROOT = _repo_root()
_logdir = ROOT / "analysis" / "experimental_shape_magnitude" / "himawari_safe_selector"
_logdir.mkdir(parents=True, exist_ok=True)
LOGFILE = _logdir / "diverse_knn_diagnostic.log"

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
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

t0 = time.time()

EXP = ROOT / "analysis" / "experimental_shape_magnitude"
# v4 definitive: a single MoE run emits no_t4f + true_tier_moe_expert + tierexpert_t0..t3
# into one OOF file, so base and tier OOF come from the same v4 output.
TIER_OOF = EXP / "true_tier_moe_xgb" / "himawari_v4_definitive_oof.csv"
BASE_OOF = EXP / "true_tier_moe_xgb" / "himawari_v4_definitive_oof.csv"
TIER_SUMMARY = EXP / "true_tier_moe_xgb" / "himawari_v4_definitive.csv"
FEATURES = EXP / "station_feature_table.csv"
DIVERSE_OOF = EXP / "diverse_streams" / "oof_predictions.csv"


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

def eval_stream(df, col, stn_masks, sids):
    stn_r2s, h2nh, flips = [], 0, 0
    y_all, p_all = [], []
    for sid in sids:
        m = stn_masks[sid]
        y = df.loc[m, "y_true"].values
        p = df.loc[m, col].values if isinstance(col, str) else col[m]
        v = ~(np.isnan(y) | np.isnan(p))
        y, p = y[v], p[v]
        y_all.extend(y); p_all.extend(p)
        if len(y) < 3 or np.std(y) < 1e-9: continue
        stn_r2s.append(float(r2_score(y, p)))
        true_cls = pm_class(np.mean(y))
        pred_cls = pm_class(np.mean(p))
        if true_cls != pred_cls: flips += 1
        sev = {"low": 0, "moderate_low": 1, "moderate": 2, "high": 3}
        if sev[pred_cls] < sev[true_cls] and true_cls in ("moderate", "high") and pred_cls in ("low", "moderate_low"):
            h2nh += 1
    return {
        "mean_r2": float(np.mean(stn_r2s)),
        "pool_r2": float(r2_score(y_all, p_all)),
        "pos": sum(1 for r in stn_r2s if r > 0),
        "h2nh": h2nh, "flips": flips,
    }


# ── Load MoE data ────────────────────────────────────────────────
print("1. Loading MoE predictions...")
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
print(f"  {len(df)} rows, {n} stations")


# ── Load diverse streams ─────────────────────────────────────────
print("\n2. Loading diverse stream predictions...")
div = pd.read_csv(DIVERSE_OOF, dtype={"stationId": str}, parse_dates=["ts"])
div = div.rename(columns={"stationId": "station_id", "PM2.5": "y_true_div"})
stream_cols = [c for c in div.columns if c.startswith("pred_")]
print(f"  {len(div)} rows, {div['station_id'].nunique()} stations")
print(f"  Streams: {stream_cols}")

df = df.merge(div[["station_id", "ts"] + stream_cols], on=["station_id", "ts"], how="left")
n_matched = df[stream_cols[0]].notna().sum()
print(f"  Matched: {n_matched}/{len(df)} rows ({100*n_matched/len(df):.1f}%)")


# ── LOO spatial prior ────────────────────────────────────────────
print("\n3. Computing spatial prior & kNN distances...")
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
            idx = np.argpartition(w, -k)[-k:]; m2 = np.zeros(n, dtype=bool); m2[idx] = True; w = np.where(m2, w, 0.0)
        t = w.sum(); prior[i] = np.dot(w/t, actual_means) if t > 1e-12 else np.mean(actual_means)
    return dict(zip(sids, prior))

prior_map = loo_prior(k=10, km_scale=60.0)
ghap_tier = dict(zip(meta["station_id"], meta["ghap_tier"]))
phase1_tier = dict(zip(meta["station_id"], meta["phase1_tier"]))

num_fc = [c for c in feat.columns if pd.api.types.is_numeric_dtype(feat[c])]
fm = feat[num_fc].fillna(feat[num_fc].median()).values
fs = StandardScaler().fit_transform(fm)
fdist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        fdist[i, j] = np.sqrt(np.sum((fs[i] - fs[j])**2))


# ── Build safe candidates from MoE (same as correct_then_select) ─
print("\n4. Building safe candidates...")

def build_safe_candidates(base_col):
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
    tier_sources = {
        "ghap": ghap_tier, "phase1": phase1_tier,
        "baseclass": {sid: tier_for_pm(base_means[sid]) for sid in sids},
        "prior": {sid: tier_for_pm(prior_map[sid]) for sid in sids},
    }
    cand_dict = {}
    for src, tmap in tier_sources.items():
        for a in [0.25, 0.35, 0.50, 0.65, 0.80]:
            vals = df[base_col].values.copy()
            for sid in sids:
                t = tmap.get(sid, "t2")
                exp_col = emap[t]
                exp_key = {"t0": "t0_mean", "t1": "t1_mean", "t2": "t2_mean", "t3": "t3_mean"}[t]
                bm = stn_info[sid]["base_mean"]
                em = stn_info[sid][exp_key]
                if pm_class((1 - a) * bm + a * em) == pm_class(bm):
                    vals[stn_masks[sid]] = (1 - a) * df.loc[stn_masks[sid], base_col].values + a * df.loc[stn_masks[sid], exp_col].values
            cand_dict[f"cand_{src}_a{int(a*100)}"] = vals

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
                tmap = tier_sources[src]
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


# ── kNN selector ─────────────────────────────────────────────────
def run_knn(all_choices, k=3):
    choice_names = list(all_choices.keys())
    per_stn = {}
    for sid in sids:
        m = stn_masks[sid]
        y = df.loc[m, "y_true"].values
        per_stn[sid] = {}
        for cname, vals in all_choices.items():
            p = vals[m] if isinstance(vals, np.ndarray) else df.loc[m, vals].values
            v = ~(np.isnan(y) | np.isnan(p))
            if v.sum() >= 10:
                per_stn[sid][cname] = r2_score(y[v], p[v])
    choices = {}
    for i, sid in enumerate(sids):
        d = fdist[i].copy(); d[i] = np.inf
        ni = np.argpartition(d, k)[:k]
        cs = {}
        for cname in choice_names:
            sc = [per_stn[sids[j]][cname] for j in ni if cname in per_stn[sids[j]]]
            if sc: cs[cname] = np.mean(sc)
        choices[sid] = max(cs, key=cs.get) if cs else choice_names[0]
    selected = np.zeros(len(df), dtype="float64")
    for sid, cname in choices.items():
        m = stn_masks[sid]
        vals = all_choices[cname]
        selected[m] = vals[m] if isinstance(vals, np.ndarray) else df.loc[m, cname].values
    return selected, choices, per_stn


def oracle_from_per_stn(per_stn):
    best_r2s = []
    for sid in sids:
        if per_stn[sid]:
            best_r2s.append(max(per_stn[sid].values()))
        else:
            best_r2s.append(0.0)
    return float(np.mean(best_r2s))


# ── Test 1: MoE-only candidates (reference) ─────────────────────
print("\n" + "="*80)
print("TEST 1: MoE-only safe candidates (reference)")
print("="*80)

moe_cands = build_safe_candidates("moe_expert")
moe_all = {"moe_expert": df["moe_expert"].values}
moe_all.update(moe_cands)

sel_moe, choices_moe, ps_moe = run_knn(moe_all, k=3)
oracle_moe = oracle_from_per_stn(ps_moe)
df["_sel_moe"] = sel_moe
r_moe = eval_stream(df, "_sel_moe", stn_masks, sids)

print(f"  MoE candidates: {len(moe_all)}")
print(f"  Oracle ceiling:   mean_r2={oracle_moe:+.4f}")
print(f"  kNN k=3:          mean_r2={r_moe['mean_r2']:+.4f}  pool={r_moe['pool_r2']:+.4f}  flips={r_moe['flips']}")


# ── Test 2: MoE + diverse streams (raw candidates, no gating) ───
print("\n" + "="*80)
print("TEST 2: MoE + diverse streams (ungated)")
print("="*80)

div_all = dict(moe_all)
for sc in stream_cols:
    vals = df[sc].values.copy()
    vals[np.isnan(vals)] = df["moe_expert"].values[np.isnan(vals)]
    div_all[sc] = vals

sel_div, choices_div, ps_div = run_knn(div_all, k=3)
oracle_div = oracle_from_per_stn(ps_div)
df["_sel_div"] = sel_div
r_div = eval_stream(df, "_sel_div", stn_masks, sids)

print(f"  Total candidates: {len(div_all)} (MoE: {len(moe_all)}, diverse: {len(stream_cols)})")
print(f"  Oracle ceiling:   mean_r2={oracle_div:+.4f}  (Δ vs MoE-only: {oracle_div - oracle_moe:+.4f})")
print(f"  kNN k=3:          mean_r2={r_div['mean_r2']:+.4f}  pool={r_div['pool_r2']:+.4f}  flips={r_div['flips']}")
print(f"  Δ kNN vs MoE-only: {r_div['mean_r2'] - r_moe['mean_r2']:+.4f}")

# Which diverse streams were chosen?
div_chosen = {k: v for k, v in choices_div.items() if v.startswith("pred_")}
print(f"  Stations choosing diverse stream: {len(div_chosen)}/{n}")
for sid, ch in sorted(div_chosen.items()):
    nm = meta.set_index("station_id").loc[sid, "station_name"] if sid in meta["station_id"].values else "?"
    print(f"    {sid[:8]}... → {ch}  (station R²={ps_div[sid][ch]:+.4f})")


# ── Test 3: MoE + diverse with class-preservation gate ──────────
print("\n" + "="*80)
print("TEST 3: MoE + diverse streams (class-gated)")
print("="*80)

gated_all = dict(moe_all)
gated_count = 0
for sc in stream_cols:
    vals = df["moe_expert"].values.copy()
    for sid in sids:
        m = stn_masks[sid]
        base_mean = float(df.loc[m, "moe_expert"].mean())
        stream_mean = float(df.loc[m, sc].mean())
        if np.isnan(stream_mean):
            continue
        if pm_class(stream_mean) == pm_class(base_mean):
            vals[m] = df.loc[m, sc].values
            gated_count += 1
        else:
            vals[m] = df.loc[m, "moe_expert"].values
    gated_all[f"gated_{sc}"] = vals

sel_gated, choices_gated, ps_gated = run_knn(gated_all, k=3)
oracle_gated = oracle_from_per_stn(ps_gated)
df["_sel_gated"] = sel_gated
r_gated = eval_stream(df, "_sel_gated", stn_masks, sids)

print(f"  Class-preserved entries: {gated_count}/{len(stream_cols) * n}")
print(f"  Total candidates: {len(gated_all)}")
print(f"  Oracle ceiling:   mean_r2={oracle_gated:+.4f}  (Δ vs MoE-only: {oracle_gated - oracle_moe:+.4f})")
print(f"  kNN k=3:          mean_r2={r_gated['mean_r2']:+.4f}  pool={r_gated['pool_r2']:+.4f}  flips={r_gated['flips']}")
print(f"  Δ kNN vs MoE-only: {r_gated['mean_r2'] - r_moe['mean_r2']:+.4f}")


# ── Test 4: Diverse-only kNN (no MoE candidates) ────────────────
print("\n" + "="*80)
print("TEST 4: Diverse streams only (no MoE candidates)")
print("="*80)

divonly = {}
for sc in stream_cols:
    vals = df[sc].values.copy()
    vals[np.isnan(vals)] = df["moe_expert"].values[np.isnan(vals)]
    divonly[sc] = vals

sel_do, choices_do, ps_do = run_knn(divonly, k=3)
oracle_do = oracle_from_per_stn(ps_do)
df["_sel_do"] = sel_do
r_do = eval_stream(df, "_sel_do", stn_masks, sids)

print(f"  Candidates: {len(divonly)} diverse streams")
print(f"  Oracle ceiling:   mean_r2={oracle_do:+.4f}")
print(f"  kNN k=3:          mean_r2={r_do['mean_r2']:+.4f}  pool={r_do['pool_r2']:+.4f}  flips={r_do['flips']}")


# ── Test 5: Per-station oracle analysis ──────────────────────────
print("\n" + "="*80)
print("TEST 5: Oracle analysis — where do diverse streams help?")
print("="*80)

print(f"\n  {'Station':<12s} {'Tier':<4s} {'MoE oracle':>11s} {'+Div oracle':>12s} {'Gain':>8s}  Best diverse")
print(f"  {'-'*70}")
for sid in sids:
    moe_best = max(ps_moe[sid].values()) if ps_moe[sid] else 0.0
    div_best = max(ps_div[sid].values()) if ps_div[sid] else 0.0
    gain = div_best - moe_best
    tier = meta.set_index("station_id").loc[sid, "tier"] if sid in meta["station_id"].values else "?"
    div_winner = max(ps_div[sid], key=ps_div[sid].get) if ps_div[sid] else "?"
    is_div = "  ← DIVERSE" if div_winner.startswith("pred_") else ""
    if gain > 0.01:
        print(f"  {sid[:12]:12s} {tier:<4s}  {moe_best:+.4f}      {div_best:+.4f}   {gain:+.4f}  {div_winner}{is_div}")

n_improved = sum(1 for sid in sids
                 if ps_div[sid] and ps_moe[sid]
                 and max(ps_div[sid].values()) > max(ps_moe[sid].values()) + 0.001)
print(f"\n  Stations where diverse stream is oracle-best: {n_improved}/{n}")


# ── Summary ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n  {'Candidate set':<35s}  {'Oracle':>8s}  {'kNN k=3':>8s}  {'Δ kNN':>8s}")
print(f"  {'-'*68}")
print(f"  {'MoE safe candidates only':<35s}  {oracle_moe:+.4f}    {r_moe['mean_r2']:+.4f}    {'ref':>8s}")
print(f"  {'MoE + diverse (ungated)':<35s}  {oracle_div:+.4f}    {r_div['mean_r2']:+.4f}    {r_div['mean_r2']-r_moe['mean_r2']:+.4f}")
print(f"  {'MoE + diverse (class-gated)':<35s}  {oracle_gated:+.4f}    {r_gated['mean_r2']:+.4f}    {r_gated['mean_r2']-r_moe['mean_r2']:+.4f}")
print(f"  {'Diverse only':<35s}  {oracle_do:+.4f}    {r_do['mean_r2']:+.4f}    {r_do['mean_r2']-r_moe['mean_r2']:+.4f}")
print(f"\n  Reference: current best blend = +0.190, dual-AOD target = +0.197")

# ── Test 6: Correct-first + diverse streams combined ─────────────
print("\n" + "="*80)
print("TEST 6: CORRECT-FIRST + DIVERSE STREAMS")
print("="*80)

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer

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

print("  Training LOSO HGB correction (A_base_only)...")
corrected = np.zeros(len(df), dtype="float32")
for fold, sid in enumerate(sids):
    train_mask = df["station_id"] != sid
    test_mask = ~train_mask
    train_df = df.loc[train_mask]
    sample_n = min(700, train_df.groupby("station_id").size().min())
    sampled = train_df.groupby("station_id", group_keys=False).sample(
        n=sample_n, replace=True, random_state=142 + fold)
    pre = ColumnTransformer(
        transformers=[("num", SkPipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), BASE_CONTEXT)], remainder="drop")
    reg = HistGradientBoostingRegressor(
        max_iter=45, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=180, l2_regularization=1.5, random_state=42)
    model = SkPipeline([("pre", pre), ("reg", reg)])
    model.fit(sampled[BASE_CONTEXT], np.log1p(sampled["y_true"].clip(lower=0)))
    pred = model.predict(df.loc[test_mask, BASE_CONTEXT])
    corrected[test_mask.values] = np.clip(np.expm1(pred), 1.0, 250.0).astype("float32")

df["corrected"] = corrected

# Build safe candidates from corrected base
corr_cands = build_safe_candidates("corrected")
print(f"  Corrected safe candidates: {len(corr_cands)}")

# 6a: Corrected + diverse (expanded pool)
combo_all = {"corrected": df["corrected"].values}
combo_all.update(corr_cands)
for sc in stream_cols:
    vals = df[sc].values.copy()
    vals[np.isnan(vals)] = df["moe_expert"].values[np.isnan(vals)]
    combo_all[f"div_{sc}"] = vals

sel_combo, ch_combo, ps_combo = run_knn(combo_all, k=3)
oracle_combo = oracle_from_per_stn(ps_combo)
df["_sel_combo"] = sel_combo
r_combo = eval_stream(df, "_sel_combo", stn_masks, sids)

print(f"\n  6a. Corrected + diverse candidates:")
print(f"      Total candidates: {len(combo_all)}")
print(f"      Oracle:  {oracle_combo:+.4f}")
print(f"      kNN k=3: {r_combo['mean_r2']:+.4f}  pool={r_combo['pool_r2']:+.4f}  flips={r_combo['flips']}")

# 6b: Blend corrected+diverse kNN with uncorrected kNN
uncorr_knn = sel_moe  # from Test 1
print(f"\n  6b. Blending corrected+diverse kNN with uncorrected kNN:")
for alpha in [0.15, 0.25, 0.35, 0.50, 0.65]:
    bl = (1 - alpha) * uncorr_knn + alpha * sel_combo
    df["_bl"] = bl
    r_bl = eval_stream(df, "_bl", stn_masks, sids)
    tag = "  ← BEST" if r_bl['mean_r2'] >= 0.195 else ""
    print(f"      a={alpha:.2f}: mean_r2={r_bl['mean_r2']:+.4f}  pool={r_bl['pool_r2']:+.4f}  flips={r_bl['flips']}{tag}")
    df.drop(columns=["_bl"], inplace=True)

# 6c: Blend corrected-only kNN (no diverse) with uncorrected kNN (reference)
corr_only = {"corrected": df["corrected"].values}
corr_only.update(corr_cands)
sel_corr, _, _ = run_knn(corr_only, k=3)
print(f"\n  6c. Reference: corrected-only kNN blended with uncorrected:")
for alpha in [0.35, 0.50, 0.65]:
    bl = (1 - alpha) * uncorr_knn + alpha * sel_corr
    df["_bl"] = bl
    r_bl = eval_stream(df, "_bl", stn_masks, sids)
    print(f"      a={alpha:.2f}: mean_r2={r_bl['mean_r2']:+.4f}  pool={r_bl['pool_r2']:+.4f}  flips={r_bl['flips']}")
    df.drop(columns=["_bl"], inplace=True)

# 6d: Three-way blend: uncorrected + corrected + diverse
div_only_sel = sel_do  # from Test 4
print(f"\n  6d. Three-way blend (uncorrected + corrected + diverse):")
for w_uncorr, w_corr, w_div in [(0.40, 0.40, 0.20), (0.35, 0.35, 0.30),
                                   (0.30, 0.30, 0.40), (0.45, 0.35, 0.20),
                                   (0.50, 0.30, 0.20), (0.40, 0.30, 0.30)]:
    bl = w_uncorr * uncorr_knn + w_corr * sel_corr + w_div * div_only_sel
    df["_bl3"] = bl
    r_bl3 = eval_stream(df, "_bl3", stn_masks, sids)
    print(f"      ({w_uncorr:.2f},{w_corr:.2f},{w_div:.2f}): mean_r2={r_bl3['mean_r2']:+.4f}  pool={r_bl3['pool_r2']:+.4f}")
    df.drop(columns=["_bl3"], inplace=True)

# 6e: Three-way with diverse-kNN (from Test 2: MoE+diverse pool)
sel_div_knn = sel_div  # from Test 2
print(f"\n  6e. Three-way: uncorrected MoE-kNN + corrected MoE-kNN + MoE+diverse kNN:")
for w_uncorr, w_corr, w_divknn in [(0.35, 0.35, 0.30), (0.30, 0.30, 0.40),
                                     (0.25, 0.25, 0.50), (0.40, 0.30, 0.30),
                                     (0.35, 0.30, 0.35)]:
    bl = w_uncorr * uncorr_knn + w_corr * sel_corr + w_divknn * sel_div_knn
    df["_bl3e"] = bl
    r_bl3e = eval_stream(df, "_bl3e", stn_masks, sids)
    print(f"      ({w_uncorr:.2f},{w_corr:.2f},{w_divknn:.2f}): mean_r2={r_bl3e['mean_r2']:+.4f}  pool={r_bl3e['pool_r2']:+.4f}")
    df.drop(columns=["_bl3e"], inplace=True)


# ── Final summary ────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\n  {'Pipeline':<45s}  {'mean_r2':>8s}  {'pool_r2':>8s}")
print(f"  {'-'*68}")
print(f"  {'MoE kNN k=3 (uncorrected)':<45s}  {r_moe['mean_r2']:+.4f}    {r_moe['pool_r2']:+.4f}")
print(f"  {'MoE+diverse kNN k=3 (uncorrected)':<45s}  {r_div['mean_r2']:+.4f}    {r_div['pool_r2']:+.4f}")
print(f"  {'Corrected+diverse kNN k=3':<45s}  {r_combo['mean_r2']:+.4f}    {r_combo['pool_r2']:+.4f}")
print(f"  {'Correct-first blend (a=0.50, prev best)':<45s}  +0.1902    ---")
print(f"  {'Dual-AOD model (target)':<45s}  +0.1970    ---")
print(f"  {'MoE+diverse oracle (ceiling)':<45s}  {oracle_div:+.4f}    ---")
print(f"  {'Corrected+diverse oracle (ceiling)':<45s}  {oracle_combo:+.4f}    ---")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("DONE")
