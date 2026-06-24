# RUN PIPELINE — the single clean v4 run order

One-pass runbook to produce every definitive result. **All scripts read
`unified_thesis_v4.csv`, use gbtree (no DART), and resolve the repo root
automatically** (run from anywhere). Heavy intermediate OOF files land in
`analysis/experimental_shape_magnitude/` (scratch, too large to submit, like `data/`);
final small metrics/figures are curated into `Thesis/results/`.

Conda env: `C:/Users/asiat/.conda/envs/airqua_env/python.exe`.

---

## Config ↔ concept mapping (so names stay clear without risky renames)

The MoE script's internal config strings are kept stable (the OOF readers filter on
them); read them through this mapping. **The whole design-evolution arc comes from ONE
MoE run** that emits all of these into a single OOF file.

| Concept (thesis §5.4 arc) | Internal config | Note |
|---|---|---|
| **Global baseline** (monolithic, lower bound) | `no_t4f` | train on all-but-held-out |
| **Geographic split** (group by region) | `region_split` | observable region → same-region expert; operationally valid |
| **Tier MoE soft-gate** (final routing) | `true_tier_moe_expert` | tier *estimated*, not assumed |
| **Tier experts** (per-tier, feed diverse-kNN) | `tierexpert_t0..t3` | |
| **Oracle ceiling** (assumes true tier known) | `oracle_t4f` | reference only, operationally INVALID |

Output prefix for that run: **`himawari_v4_definitive`** → files
`himawari_v4_definitive.csv` (summary) + `himawari_v4_definitive_oof.csv` (all configs).

---

## Run order

### Step 0 — standalone station feature table (kNN spatial prior input)
```
python Thesis/scripts/03_features/build_station_feature_table.py
```
Produces `station_feature_table.csv` (40 stations, cleaned v4 means + coords). Fast.

### Step 1 — the design-arc MoE run (ONE run, all configs) · ~3–4h
```
python Thesis/scripts/04_experiments/exp_true_tier_moe_xgb.py \
  --configs oracle_t4f,no_t4f,region_split,true_tier_moe_expert,tierexpert_t0,tierexpert_t1,tierexpert_t2,tierexpert_t3 \
  --aod-mode himawari --out-prefix himawari_v4_definitive --save-oof
```
Gives §5.3 (global), §5.4 (global→region→tier arc), §5.5 (oracle ceiling), and the
base/MoE/tier OOF that the diverse-kNN headline consumes.

### Step 2 — diverse feature streams · long
```
python Thesis/scripts/04_experiments/exp_diverse_streams.py
```
Produces `diverse_streams/oof_predictions.csv` (v4, 40 thesis stations).

### Step 3 — diverse + kNN-3 deployable headline
```
python Thesis/scripts/04_experiments/exp_diverse_knn_diagnostic.py
```
Reads `himawari_v4_definitive_oof.csv` + diverse OOF + feature table.

### Step 4 — within-station temporal ceiling (§4.1)
```
python Thesis/scripts/03_features/within_station_predictability_v4.py
```

### Step 5 — external LCS validation (§5.6)
```
python Thesis/scripts/04_experiments/validate_diverse_knn_lcs.py
```

### Supporting (any time, independent)
```
python Thesis/scripts/04_experiments/exp_satellite_products.py     # §4.2 CTM baselines fail
python Thesis/scripts/03_features/aod_pm25_correlation_paper.py     # §4.1 AOD–PM2.5
python Thesis/scripts/04_experiments/exp_red_river_delta.py         # §5.8 regional study
python Thesis/scripts/04_experiments/conformal_trustmap.py          # §5.7 conformal (after Step 3)
python Thesis/scripts/02_processing/validate_pm25_qc_effect.py      # §3.2 stronger-mask QC
python Thesis/scripts/02_processing/data_profile.py                 # refresh dataset summary (v4)
```

---

## After the run — clean-up checklist (the “shed the old skin” pass)

1. Refresh `results/01_stations/dataset_summary.md` (stale: still says 3 dropped / 46 LCS /
   839k rows / 116 feat) → v4 (40 kept, 57 LCS, 137 cols, stronger mask). `data_profile.py`.
2. Copy the small final metrics/summaries into `Thesis/results/` (not the giant OOF blobs).
3. Retire the stale `pipeline_summary.csv` "partial GHAP-free" line — the soft-tier prior is
   GHAP-free (`exp_soft_tier_moe.py:235`); regenerate/confirm so the doc matches.
4. Rewrite Ch.4–6 prose against the real v4 numbers; remove the Ch.5 "non-definitive" footnote.
5. Re-run `scripts/05_thesis/*` (reports, abstract, pipeline diagram) — they still emit the
   old DART/37-station framing.

## Honesty caveats to keep (per the writing plan)
- Deployability is **network-anchored** (kNN prior needs nearby known stations), not
  satellite-only and not a gap-free national grid.
- Abandoned branches (DART, GHAP-as-considered, nationwide grid) do **not** appear in the
  thesis prose in any form — clean absence, not "considered then dropped."
