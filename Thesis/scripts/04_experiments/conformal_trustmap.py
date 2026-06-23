"""
Conformal trust-map analysis on definitive_v3 per-hour OOF residuals.

Consumes oof_residuals_v3.{parquet,csv} (config, station_id, tier, pm25_mean,
ts, y_true, y_pred, residual) from exp_definitive_v3.py and computes:

  1. Nested-LOSO split-conformal coverage (marginal + per-tier conditional)
  2. Mondrian (regime-binned) conformal coverage -- binning by a DEPLOYABLE
     signal (the model's own predicted station level, and sat_pred_pm) vs the
     true-tier oracle. Tests whether regime-conditional calibration closes the
     clean-tier coverage gap.
  3. Risk-coverage (selective prediction) curve: served pooled R2 / RMSE vs
     coverage, ranked by a deployable confidence score, vs random + oracle.

Post-processing only; nothing is retrained. This ANNOTATES / ABSTAINS -- it
never routes training (so it is NOT the failed gating approach).
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
# chdir to repo root (Thesis/scripts/04_experiments -> dirname x4). Reads analysis/... paths.
def _repo_root():
    p = os.path.abspath(os.path.dirname(__file__))
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "data", "merged")):
            return p
        p = os.path.dirname(p)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(_repo_root())

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

EXP = "analysis/thesis_experiments"
CONFIG = os.environ.get("CFG", "dart_ensemble")
ALPHAS = [0.10, 0.20]            # 90% and 80% nominal intervals
TIERS = ["t0", "t1", "t2", "t3"]
RNG_SEED = 42


def load_resid():
    pq = os.path.join(EXP, "oof_residuals_v3.parquet")
    cs = os.path.join(EXP, "oof_residuals_v3.csv")
    if os.path.exists(pq):
        d = pd.read_parquet(pq)
    elif os.path.exists(cs):
        d = pd.read_csv(cs, dtype={"station_id": str})
    else:
        sys.exit("oof_residuals_v3 not found -- run exp_definitive_v3.py first")
    d["station_id"] = d["station_id"].astype(str)
    if "residual" not in d.columns:
        d["residual"] = d["y_pred"] - d["y_true"]
    return d


def load_satpred():
    p = os.path.join(EXP, "satellite_grouping_classifier.csv")
    if not os.path.exists(p):
        return {}
    s = pd.read_csv(p, dtype={"stationId": str})
    if "sat_pred_pm" not in s.columns:
        return {}
    return dict(zip(s["stationId"].astype(str), s["sat_pred_pm"]))


def station_table(d):
    rows = []
    for sid, g in d.groupby("station_id"):
        yt, yp = g["y_true"].values, g["y_pred"].values
        rows.append(dict(
            station_id=sid, tier=g["tier"].iloc[0],
            pm25_mean=float(g["pm25_mean"].iloc[0]), n=len(g),
            pred_level=float(np.mean(yp)),          # deployable: model's own mean
            r2=r2_score(yt, yp) if len(g) > 10 else np.nan,
            rmse=float(np.sqrt(mean_squared_error(yt, yp))),
            sum_y=float(yt.sum()), sum_y2=float((yt ** 2).sum()),
            ss_res=float(((yp - yt) ** 2).sum()),
        ))
    return pd.DataFrame(rows).set_index("station_id")


# ---------------------------------------------------------------------------
def nested_marginal_coverage(d, stn, alpha):
    cov, width = {}, {}
    for sid in stn.index:
        cal = d[d["station_id"] != sid]["residual"].abs().values
        q = np.quantile(cal, 1 - alpha)
        test = d[d["station_id"] == sid]["residual"].abs().values
        cov[sid] = float(np.mean(test <= q)); width[sid] = 2 * q
    return cov, width


def nested_mondrian_coverage(d, stn, alpha, bin_series, n_bins=4, min_cal=200):
    vals = bin_series.reindex(stn.index)
    try:
        bins = pd.qcut(vals, n_bins, labels=False, duplicates="drop")
    except Exception:
        bins = pd.cut(vals, n_bins, labels=False)
    sid_bin = dict(zip(stn.index, bins))
    d = d.copy(); d["bin"] = d["station_id"].map(sid_bin)
    cov = {}
    for sid in stn.index:
        b = sid_bin.get(sid)
        cal = d[(d["bin"] == b) & (d["station_id"] != sid)]["residual"].abs().values
        if len(cal) < min_cal:
            cal = d[d["station_id"] != sid]["residual"].abs().values
        q = np.quantile(cal, 1 - alpha)
        test = d[d["station_id"] == sid]["residual"].abs().values
        cov[sid] = float(np.mean(test <= q))
    return cov


def summarize_cov(cov, stn, label):
    s = pd.Series(cov)
    out = {"binning": label, "marginal_cov": float(s.mean())}
    for t in TIERS:
        sids = stn.index[stn["tier"] == t]
        out[f"cov_{t}"] = float(s.reindex(sids).mean()) if len(sids) else np.nan
    tcovs = [out[f"cov_{t}"] for t in TIERS if not np.isnan(out[f"cov_{t}"])]
    out["tier_cov_range"] = float(max(tcovs) - min(tcovs)) if tcovs else np.nan
    return out


def _cum_pooled(n, sy, sy2, ssr):
    """Cumulative pooled R2 / RMSE over a served order (sufficient stats)."""
    cn = np.cumsum(n); csy = np.cumsum(sy); csy2 = np.cumsum(sy2); cssr = np.cumsum(ssr)
    ybar = csy / cn
    sstot = csy2 - cn * ybar * ybar
    r2 = 1.0 - cssr / np.where(sstot > 0, sstot, np.nan)
    rmse = np.sqrt(cssr / cn)
    return r2, rmse  # index j -> served = first (j+1)


def risk_coverage(stn, rank_series, label):
    order = rank_series.reindex(stn.index).sort_values(ascending=False).index.tolist()
    sub = stn.loc[order]
    r2, rmse = _cum_pooled(sub["n"].values.astype(float), sub["sum_y"].values,
                           sub["sum_y2"].values, sub["ss_res"].values)
    r2vals = sub["r2"].values
    N = len(order)
    return pd.DataFrame([dict(order=label, coverage=(j + 1) / N, n_served=j + 1,
                              pooled_r2=float(r2[j]),
                              mean_station_r2=float(np.nanmean(r2vals[:j + 1])),
                              rmse=float(rmse[j])) for j in range(N)])


def random_cov_ci(stn, n_iter=2000):
    rng = np.random.default_rng(RNG_SEED)
    N = len(stn)
    nA = stn["n"].values.astype(float); syA = stn["sum_y"].values
    sy2A = stn["sum_y2"].values; ssrA = stn["ss_res"].values
    mat = np.empty((n_iter, N))
    for it in range(n_iter):
        p = rng.permutation(N)
        r2, _ = _cum_pooled(nA[p], syA[p], sy2A[p], ssrA[p])
        mat[it] = r2
    return pd.DataFrame([dict(coverage=(j + 1) / N, n_served=j + 1,
                              rand_mean=float(np.nanmean(mat[:, j])),
                              rand_lo=float(np.nanquantile(mat[:, j], 0.025)),
                              rand_hi=float(np.nanquantile(mat[:, j], 0.975)))
                         for j in range(N)])


# ===========================================================================
def main():
    d_all = load_resid()
    print(f"Configs in file: {sorted(d_all['config'].unique())}")
    d = d_all[d_all["config"] == CONFIG].copy()
    if d.empty:
        sys.exit(f"config {CONFIG} not in residual file")
    stn = station_table(d)
    print(f"\nConfig = {CONFIG}: {len(stn)} stations, {len(d):,} OOF hours")
    print(stn.groupby("tier").agg(n_stn=("n", "size"), mean_pm=("pm25_mean", "mean"),
          mean_r2=("r2", "mean"), mean_rmse=("rmse", "mean")).round(3).to_string())

    sat_series = pd.Series(load_satpred()).reindex(stn.index)
    has_sat = sat_series.notna().sum() >= len(stn) * 0.8
    print(f"\nsat_pred_pm available for {sat_series.notna().sum()}/{len(stn)} stations")

    # ---- coverage ----
    print(f"\n{'='*82}\nCONFORMAL COVERAGE (nested LOSO; nominal vs achieved)\n{'='*82}")
    rows = []
    for alpha in ALPHAS:
        nom = 1 - alpha
        cov_m, _ = nested_marginal_coverage(d, stn, alpha)
        r = summarize_cov(cov_m, stn, f"marginal"); r["nominal"] = nom; rows.append(r)
        r = summarize_cov(nested_mondrian_coverage(d, stn, alpha, stn["pred_level"]),
                          stn, "mondrian_predlevel"); r["nominal"] = nom; rows.append(r)
        if has_sat:
            r = summarize_cov(nested_mondrian_coverage(d, stn, alpha, sat_series),
                              stn, "mondrian_satpred"); r["nominal"] = nom; rows.append(r)
        r = summarize_cov(nested_mondrian_coverage(d, stn, alpha, stn["pm25_mean"]),
                          stn, "mondrian_ORACLE"); r["nominal"] = nom; rows.append(r)
    cov_df = pd.DataFrame(rows)
    cols = ["nominal", "binning", "marginal_cov", "cov_t0", "cov_t1", "cov_t2",
            "cov_t3", "tier_cov_range"]
    print(cov_df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    cov_df.to_csv(os.path.join(EXP, "conformal_coverage_v3.csv"), index=False)

    # ---- risk-coverage ----
    print(f"\n{'='*82}\nRISK-COVERAGE (selective prediction)\n{'='*82}")
    frames = [risk_coverage(stn, stn["pred_level"], "by_pred_level")]
    if has_sat:
        frames.append(risk_coverage(stn, sat_series, "by_sat_pred"))
    frames.append(risk_coverage(stn, stn["r2"], "ORACLE_by_r2"))
    rc = pd.concat(frames, ignore_index=True)
    rand = random_cov_ci(stn)
    rc.to_csv(os.path.join(EXP, "risk_coverage_v3.csv"), index=False)
    rand.to_csv(os.path.join(EXP, "risk_coverage_random_v3.csv"), index=False)

    for cov_pt in [1.0, 0.7, 0.5, 0.3]:
        print(f"\n  coverage ~= {cov_pt:.0%}")
        for lab in rc["order"].unique():
            sub = rc[rc["order"] == lab]
            row = sub.iloc[(sub["coverage"] - cov_pt).abs().argmin()]
            print(f"    {lab:16s}  served pooled R2={row['pooled_r2']:+.3f}  "
                  f"mean-stn R2={row['mean_station_r2']:+.3f}  RMSE={row['rmse']:.2f}  "
                  f"(n={int(row['n_served'])})")
        rr = rand.iloc[(rand["coverage"] - cov_pt).abs().argmin()]
        print(f"    {'random(95%CI)':16s}  pooled R2={rr['rand_mean']:+.3f}  "
              f"[{rr['rand_lo']:+.3f}, {rr['rand_hi']:+.3f}]")

    # ---- rank-signal quality ----
    print(f"\n{'='*82}\nRANK-SIGNAL QUALITY (does the confidence score track realized skill?)\n{'='*82}")
    if has_sat:
        m = stn[["r2"]].join(sat_series.rename("sat")).dropna()
        rho, p = spearmanr(m["sat"], m["r2"])
        print(f"  Spearman(sat_pred_pm, realized per-station R2) = {rho:+.3f} (p={p:.4f})")
    rho2, p2 = spearmanr(stn["pred_level"], stn["r2"])
    print(f"  Spearman(pred_level,  realized per-station R2) = {rho2:+.3f} (p={p2:.4f})")

    print("\nSaved: conformal_coverage_v3.csv, risk_coverage_v3.csv, "
          "risk_coverage_random_v3.csv\nDONE")


if __name__ == "__main__":
    main()
