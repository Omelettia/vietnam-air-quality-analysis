# Run Pipeline

Clean runbook for the final Red River Delta thesis package.

Conda env:

```powershell
C:\Users\asiat\.conda\envs\airqua_env\python.exe
```

## Main Run Order

### 1. Build / refresh unified data

```powershell
python Thesis/scripts/02_processing/build_unified.py
python Thesis/scripts/02_processing/validate_pm25_qc_effect.py
python Thesis/scripts/02_processing/data_profile.py
```

### 2. National diagnostics

These runs diagnose the sparse-station and regime problem. They are not the
final regional model.

```powershell
python Thesis/scripts/03_features/aod_pm25_correlation_paper.py
python Thesis/scripts/03_features/within_station_predictability.py
python Thesis/scripts/04_experiments/exp_satellite_products.py
python Thesis/scripts/04_experiments/exp_national_loso_diagnostic.py
```

`within_station_predictability.py` is a supporting temporal diagnostic. It
uses the shared `pm25_quality_masks` PM2.5 filter and tests whether the features
contain within-station time signal when the station context is already observed.

The controlled national LOSO diagnostic used in Chapter 4 is rerunnable from
merged data and shared PM2.5 QC:

```text
Thesis/results/03_model/report_loso.txt
```

It reports the 40-station global XGBoost baseline, geographic region split, and
oracle true-tier ceiling as diagnostics.
Older national selector and stream-routing branches are archived and are not
part of the final package.

### 3. Final Red River Delta model

This is the final model path for the thesis.

```powershell
python Thesis/scripts/04_experiments/exp_red_river_delta.py
```

Expected headline result:

- best deployable model: `delta_rfsi_wind`
- hourly pooled R² = 0.4233
- hourly mean station R² = 0.2709
- daily pooled R² = 0.4911
- daily mean station R² = 0.2835
- US Embassy daily R² = 0.836
- LCS hourly median R² = 0.307
- LCS daily median R² = 0.355
- LCS hourly positive-station rate = 73%
- LCS daily positive-station rate = 61%

See:

```text
Thesis/results/04_validation/report_red_river_delta_v5h.txt
```

The experiment scripts write raw CSVs and figures under
`analysis/thesis_experiments/` or their own output folders. Files under
`Thesis/results/` are the locked, standardized copies used to generate the
tables, figures, and PDF.

## Scope

Only the scripts listed above are part of the final thesis pipeline. Earlier
exploratory branches are kept outside `Thesis/` under `archive/` and are not
needed to reproduce the manuscript.

## Claim Discipline

- The national experiments are diagnostic.
- The final deployable claim is regional, not nationwide.
- RFSI means the model is network-anchored: it requires nearby monitor data at
  prediction time.
- Pooled national R² is not enough; use mean/median station R² and external
  validation.
