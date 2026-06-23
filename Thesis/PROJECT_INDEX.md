# Thesis Package

Vietnam PM2.5 prediction from satellite data — Master's thesis, SOICT/HUST.

> **Status (2026-06-24): consolidated to a single definitive configuration.**
> - **Dataset:** `unified_thesis_v4.csv` is **definitive** — 121 stations total, all **40 thesis stations kept** (the 3 formerly-"broken" low-cost sensors are cleaned row-wise by the stronger PM2.5 QC mask, not dropped). v2/v3 are legacy and archived.
> - **Model family:** **no DART** — gbtree / HistGradientBoosting only, for consistency across every reported result.
> - **Everything runs on v4.** All scripts here resolve the repo root automatically and read v4.
> - Results under `results/` that predate this consolidation are being **regenerated on v4**; until then they are marked non-definitive.

```
Thesis/
  scripts/            Code that produces the results (all v4, no DART, self-contained)
    01_collection/      Fetch raw data from TEDP, GEE, JAXA, Open-Meteo
    02_processing/      QC (stronger mask), merge → unified_thesis_v4.csv
    03_features/        Directional climatologies, AOD-PM2.5, within-station ceiling
    04_experiments/     The 4 definitive runs + CTM baseline, conformal, regional
    05_thesis/          Generate figures, markdown → LaTeX, compile PDF
  results/            What the model produced (ordered by the thesis argument)
  latex/              Thesis writing (English .md source + Vietnamese .tex + PDF)

data/                 Raw + merged data (outside Thesis/, too large to submit)
archive/              Superseded scripts/results (DART era, journey, legacy v2/v3)
```

---

## The definitive pipeline — 4 runs that matter

All runs use **v4** and **gbtree (no DART)**. The first three are LOSO / within-station
on the 40 thesis stations; the fourth is the deployable headline + external check.

| # | Run | Script | Output | Thesis |
|---|-----|--------|--------|--------|
| 1 | **No-tier LOSO baseline** (lower bound) | `04_experiments/exp_true_tier_moe_xgb.py --configs no_t4f` | `…/true_tier_moe_xgb/*_oof.csv` (config `no_t4f`) | Ch.5 baseline |
| 2 | **Tier / MoE soft-gate** + tier experts | `04_experiments/exp_true_tier_moe_xgb.py --configs true_tier_moe_expert` (and tier-expert prefix) | MoE + `tierexpert_t0..t3` OOF | Ch.5 main |
| 3 | **Within-station temporal ceiling** | `03_features/within_station_predictability_v4.py` | `within_station_predictability_v4.csv` | Ch.4/5 ceiling |
| 4 | **Diverse streams + kNN-3 (deployable)** | `exp_diverse_streams.py` → `exp_diverse_knn_diagnostic.py` → `validate_diverse_knn_lcs.py` | diverse OOF, kNN blend, LCS external | Ch.5 §5.8 |

**Routing prior:** run #2 consumes `soft_tier_prior_station_table.csv` (40-row station→tier
probabilities) only to route a held-out station to a tier expert. It is reused as built
(its lineage predates v4); footnote this in the text. It does **not** touch v4 features/training.

**kNN spatial prior:** run #4 computes its **own** prior from a standalone station table
(`station_id, lat, lon, actual_mean`) regenerated from v4 — no dependency on other scripts.

**Not used / archived:** `gated_fourier`, `aod_harmonic`, the soft-tier heavy inputs, all
DART experiments, and the Himawari journey variants. See `archive/`.

---

## 01_collection/ — Raw data acquisition

Ground stations & weather (TEDP API, Open-Meteo ERA5), GEE Code-Editor scripts
(MODIS MAIAC AOD, MODIS LST, TROPOMI NO2/SO2/CO/HCHO, GHAP PM2.5, building density),
and Himawari-8 AOD (JAXA P-Tree). Superseded fetchers live in `01_collection/deprecated/`.

**Outputs:** `data/stations/…`, `data/gee_exports/`, `data/station_aod_v3/`.

---

## 02_processing/ — Cleaning & unification (→ v4)

| Script | What it does |
|--------|-------------|
| `thesis_pipeline.py` | Master merge of all sources → `unified_thesis_v1.csv` |
| `build_static_features_123.py`, `convert_gee_directional.py`, `convert_embassy_data.py` | Static/directional/embassy feature assembly |
| `build_unified_v4.py` | **Definitive build** → `unified_thesis_v4.csv` (137 cols, all stations, stronger mask, relaxed coverage so all 40 thesis stations are retained) |
| `pm25_qc.py` | **Stronger row-level PM2.5 QC mask** (flatline ≥5h, stuck-low ≤2µg ≥48h, range) — cleans the 3 ex-broken sensors instead of dropping them |
| `thesis_phase1_audit.py`, `diagnose_pm25_qc.py`, `validate_pm25_qc_effect.py`, `data_profile.py`, `normalize_envisoft_data.py` | QC audit, mask validation, dataset profile |

**Output:** `data/merged/unified_thesis_v4.csv` — definitive dataset.

---

## 03_features/ — Feature engineering & analysis

| Script | What it does | Thesis |
|--------|-------------|--------|
| `build_modis_aod_features.py`, `build_no2_features.py` | 8-direction MODIS AOD / TROPOMI NO2 climatology per station | — |
| `aod_pm25_correlation_paper.py` | Hourly/monthly AOD–PM2.5 correlation | Ch.4 §4.1 |
| `within_station_predictability_v4.py` | **Temporal ceiling** — 5-fold KFold per station, full v4 features (no RFSI). Keeps all 40 | Ch.4/5 |

---

## 04_experiments/ — The definitive 8 scripts

| Script | Role | Thesis |
|--------|------|--------|
| `exp_true_tier_moe_xgb.py` | **MoE backbone LOSO** — no-tier baseline (`no_t4f`), MoE soft-gate (`true_tier_moe_expert`), tier experts. v4, gbtree | Ch.5 main |
| `exp_diverse_streams.py` | 5 feature-subset gbtree models → diverse OOF (v4, 40 stations) | Ch.5 §5.8 |
| `exp_diverse_knn_diagnostic.py` | **Deployable headline** — combine MoE/tier/diverse OOF + kNN-3 spatial-prior shift | Ch.5 §5.8 |
| `validate_diverse_knn_lcs.py` | External LCS validation of the diverse + kNN pipeline | Ch.5 §5.8 |
| `exp_satellite_products.py` | CTM baselines (GEOS-CF, MERRA-2, GHAP) vs ground truth — they fail | Ch.4 §4.2 |
| `exp_tier_operational.py` | Deployable tier assignment (no oracle) | Ch.5 §5.5 |
| `exp_red_river_delta.py` | Red River Delta regional LOSO | Ch.5 §5.4 |
| `conformal_trustmap.py` | Mondrian conformal prediction + selective coverage | Ch.5 §5.7 |

---

## 05_thesis/ — Document generation

`generate_figures.py`, `generate_pipeline_diagram.py`, `generate_reports.py`,
`generate_appendix_and_translate.py`, `md_to_latex.py`.

**Build PDF:** `cd latex && pdflatex DoAn.tex && biber DoAn && pdflatex DoAn.tex && pdflatex DoAn.tex`
(biber needs a sanitized PATH — see the LaTeX build note.)

---

## Results — `results/`

```
results/
  01_stations/        40 thesis stations, feature list, dataset profile
  02_ctm_baseline/    GEOS-CF / MERRA-2 / GHAP fail (Ch.4 §4.2)
  03_model/           LOSO main results (regenerating on v4)
  04_validation/      LCS + Embassy external check
  05_conformal/       Mondrian conformal + selective coverage
  06_data_quality/    QC audit + stronger-mask validation
  07_deployable_model/ Deployable map stack, LCS stress tests (DART verdicts archived)
  08_himawari_pipeline/ Himawari MoE + kNN diverse pipeline (Ch.5 §5.8)
```

---

## Key numbers (defense quick-reference)

| Metric | Value |
|--------|-------|
| Stations | 40 thesis (all kept; 3 ex-broken cleaned by stronger mask) |
| Dataset | `unified_thesis_v4.csv`, 2023-01 → 2026-04, 137 cols |
| Model family | gbtree / HGB — **no DART** |
| Within-station temporal ceiling | KFold R² ≈ 0.73 (40 thesis) / 0.80 (all) |
| No-tier LOSO baseline | mean station R² ≈ -0.05 (lower bound) |
| MoE soft-gate LOSO | pooled R² ≈ 0.43, mean station R² ≈ 0.08 |
| Diverse + kNN-3 (deployable) | OOF lift ≈ +0.23; external LCS ≈ +0.16–0.19 |
| Top feature group | RFSI (spatial) |
| CTM baselines | GEOS-CF bias +244%; MERRA-2 IOA 0.42 — both fail |

> Numbers in this table are the consolidation targets; the v4 rerun refreshes the exact
> figures and the chapter prose is updated against them.
