# Thesis Writing Plan — what we write, why, and in what order

Master's thesis, SOICT/HUST — *Vietnam PM2.5 prediction from satellite data*.
Definitive configuration: **`unified_thesis_v4`, stronger QC mask, all 40 stations kept, no DART**.

> **Writing order strategy.** Chapters 1–3 (problem + theory + proposed method/data
> processing) do **not** depend on model results and can be written **now**. Chapter 4–5
> depend on results; the *stable* parts (AOD–PM2.5 correlation, CTM-baseline failures,
> within-station ceiling) can also be drafted now, while the *headline LOSO / MoE / diverse-kNN
> numbers* are left as placeholders until the v4 rerun lands (they are expected to shift only
> slightly). The Ch.5 footnote already marks current figures as non-definitive.

Legend for result status: **[LOCKED]** stable / config-independent · **[v4-PENDING]** refresh after the run.

---

## The one-paragraph thesis

Predicting hourly PM2.5 over Vietnam from satellite + meteorology is **two different problems**
that the literature conflates. *Temporal interpolation* (predict a station you trained on) is
easy — R² ≈ 0.8. *Spatial extrapolation* (predict a station never seen, the actual deployment
case) is hard — LOSO R² ≈ 0.2, deployable median ≈ 0.04. The gap is driven by **station density**
(distance to nearest training anchor), and **pollution level** sets a monotonic skill gradient
(clean low-tier stations are nearly unpredictable; polluted t3 stations predict well). We build a
deployable, **GHAP-free, no-DART** pipeline (MoE soft-gate over pollution tiers + diverse feature
streams + a kNN spatial-prior shift) that recovers most of the achievable spatial skill and
validate it on an external low-cost-sensor (LCS) network and US-Embassy reference monitors.

The recurring honesty device: always report **pooled R²**, **mean-station R²**, and
**median-station R²** together, plus **% stations with R² > 0**, because they diverge sharply.

---

## Chapter 1 — Giới thiệu (Introduction) · WRITE NOW (no results)

**Goal:** frame the problem and the three-number story.

- Motivation: Vietnam air quality, sparse reference monitoring, need for spatial maps.
- The core distinction (preview of §2.1): temporal interpolation vs spatial extrapolation.
- **The three-number spine:** R² ≈ **0.80** temporal / ≈ **0.20** spatial LOSO / ≈ **0.04** deployable median. State it up front; every chapter returns to it.
- Contributions: (1) the dataset (40 stations, satellite+met+emissions, stronger QC); (2) the temporal-vs-spatial reframing; (3) the deployable GHAP-free no-DART pipeline; (4) honest multi-metric evaluation + external validation.
- Thesis structure roadmap.

---

## Chapter 2 — Cơ sở lý thuyết (Background) · WRITE NOW (no results)

**Goal:** the concepts and the evaluation philosophy.

- §2.1 **Temporal interpolation vs spatial extrapolation** — the central framing; the matching evaluation protocols (random CV vs LOSO). *This is the conceptual heart; write it carefully.*
- PM2.5 health/standards; AOD and the AOD–PM2.5 relationship (column vs surface, humidity, PBLH).
- Satellite products: MODIS MAIAC AOD, Himawari-8 AOD, TROPOMI trace gases, MODIS LST; CTM products (GEOS-CF, MERRA-2) and global reanalysis PM2.5 (GHAP) — and why we ultimately avoid GHAP for a deployable claim.
- Low-cost sensors (Plantower PMS class) — characteristics and biases (motivates the QC mask).
- Evaluation metrics: pooled vs mean vs median station R², %positive — *why all are needed.*

---

## Chapter 3 — Đề xuất (Proposed method & data processing) · WRITE NOW (no results)

**Goal:** the pipeline and dataset — fully locked, describe with confidence.

- **Data sources & collection** (`scripts/01_collection/`): TEDP ground stations, Open-Meteo ERA5 meteorology, GEE exports (MODIS AOD/LST, TROPOMI, building density), Himawari-8 AOD (JAXA), GPM precipitation.
- **Stations:** 40 thesis stations (`station_selection_final.csv`) + 57 LCS for external validation (`station_selection_lcs.csv`). All KK stations are low-cost sensors — state this honestly.
- **PM2.5 quality control — the stronger mask** (`02_processing/pm25_qc.py`): flatline ≥5h, stuck-low ≤2µg ≥48h, range checks. **[LOCKED]** Masks ~25k rows (vs 2.6k old). Decision: **keep all 40 stations**; the 3 ex-broken low-cost sensors (Da Nang Pham Hung, Soc Trang, Tra Vinh Dong Hai) are **cleaned row-wise, not dropped** — `include_with_sensor_warning`. Cite `results/06_data_quality/report_pm25_qc_effect_validation.txt`.
- **Unification → `unified_thesis_v4.csv`** (`02_processing/build_unified_v4.py`): 137 columns, 2023-01→2026-04, relaxed coverage so all 40 are retained.
- **Features:** meteorology core + persistence (PBLH/VC/stagnation), temporal cyclicals, AOD (Himawari + MODIS climatology, directional), TROPOMI emissions (SO2/CO/NO2 anomalies), LST anomalies, building density, terrain, and **RFSI** (spatial nearest-station PM2.5 — the dominant feature group).
- **Model architecture (no DART — gbtree/HGB throughout):**
  - No-tier LOSO backbone (operational lower bound).
  - **MoE soft-gate** over pollution tiers (t0–t3): a station-level tier-routing prior selects per-tier experts. *Resolves the tier-circularity: you don't know a new station's tier.*
  - **Diverse feature streams** (5 subset models) + **kNN-3 spatial-prior shift** anchoring station means — the deployable headline.
  - Mondrian **conformal** prediction for selective/trustworthy coverage.
- Reproducibility: every script self-contained (auto repo-root), reads v4, gbtree only.

---

## Chapter 4 — Phân tích lý thuyết (What the model learns) · PARTIAL NOW

**Goal:** mechanism before metrics — *why* it behaves as it does.

- §4.1 **AOD–PM2.5 correlation** (`03_features/aod_pm25_correlation_paper.py`) — **[LOCKED]** hourly/monthly correlation structure; column-vs-surface caveats.
- §4.1 **Within-station temporal ceiling** (`03_features/within_station_predictability_v4.py`) — **[v4-PENDING, but stable]** KFold R² ≈ 0.73 (40) / 0.80 (all); the upper bound when a station is known.
- §4.2 **CTM baselines fail** (`04_experiments/exp_satellite_products.py`) — **[LOCKED]** GEOS-CF bias ≈ +244%, diurnal r ≈ −0.14; MERRA-2 IOA ≈ 0.42; GHAP is climatology, not dynamics. Three distinct failure modes.
- **Spatial interpolation dominates / nearest-anchor constraint** — RFSI is the top feature group (~37% of gain); skill tracks distance-to-nearest-anchor. **[v4-PENDING]** for exact ablation numbers.

---

## Chapter 5 — Đánh giá thực nghiệm (Experiments) · NEEDS v4 RESULTS

**Goal:** the full quantitative evaluation. (Footnote already flags current numbers as non-definitive.)

- §5.1 Evaluation parameters — pooled vs mean/median station R², %positive (write now).
- §5.2 **Temporal interpolation (random 5-fold CV)** — **[v4-PENDING]** ≈ 0.80; the easy regime.
- §5.3 **Spatial extrapolation (LOSO)** — **[v4-PENDING]** no-tier baseline (lower bound) → the gap to temporal.
- §5.4 **Tier gradient** — **[v4-PENDING]** monotonic skill ∝ pollution level; isolate the tier boost (≈ +0.17 in-file, up to +0.26). Honesty note: *level vs region as the driver is a hypothesis the data can't fully separate.*
- §5.5 **Deployable vs oracle ceiling** — **[v4-PENDING]** MoE soft-gate (pooled ≈ 0.43, mean ≈ 0.08) and the diverse+kNN headline (OOF ≈ +0.23) vs oracle; cost of not knowing a station's mean.
- §5.6 **External validation** — **[v4-PENDING]** LCS network + US-Embassy; spatial-prior shift effect (≈ +0.16–0.19).
- §5.7 **Conformal trust map** (`conformal_trustmap.py`) — selective coverage / where predictions are trustworthy.
- §5.8 **Unifying principle** — temporal vs spatial under one explanation for the 0.1→0.8 R² span. Also: Red River Delta regional study (`exp_red_river_delta.py`); summary of failed experiment families.

---

## Chapter 6 — Kết luận (Conclusion) · AFTER Ch.5

- Findings restated against the three-number spine.
- **Deployability verdict:** GHAP-free, no-DART pipeline; what it can/can't do as a map layer.
- Limitations: low-cost sensor ground truth; station density; level/region confound.
- Future work: denser anchors, better clean-station modeling, operational tiering.

---

## Result-status checklist (drives Ch.4–5 once the run completes)

| Result | Script | Status |
|--------|--------|--------|
| Stronger-mask QC effect | `validate_pm25_qc_effect.py` | **LOCKED** (`results/06_data_quality/`) |
| AOD–PM2.5 correlation | `aod_pm25_correlation_paper.py` | **LOCKED** |
| CTM baseline failure | `exp_satellite_products.py` | **LOCKED** (rerun on v4 for exact figures) |
| Within-station ceiling | `within_station_predictability_v4.py` | v4-PENDING (≈0.73/0.80) |
| No-tier LOSO baseline | `exp_true_tier_moe_xgb.py --configs no_t4f` | v4-PENDING |
| MoE soft-gate + tier experts | `exp_true_tier_moe_xgb.py` | v4-PENDING (≈0.43 pooled) |
| Diverse streams + kNN-3 | `exp_diverse_streams.py` → `exp_diverse_knn_diagnostic.py` | v4-PENDING (≈+0.23 OOF) |
| External LCS validation | `validate_diverse_knn_lcs.py` | v4-PENDING (≈+0.16–0.19) |
| Conformal coverage | `conformal_trustmap.py` | v4-PENDING |

**Run order when ready:** `build_station_feature_table.py` → `exp_true_tier_moe_xgb.py`
(all 6 configs → `himawari_v4_definitive_oof`) → `exp_diverse_streams.py` →
`exp_diverse_knn_diagnostic.py` → `within_station_predictability_v4.py` → `validate_diverse_knn_lcs.py`.
