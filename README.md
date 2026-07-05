# Vietnam Air Quality Analysis

Multi-source analysis of fine-particulate (PM2.5) air pollution over Vietnam,
combining ground-sensor networks with satellite and meteorological data. The work
spans two phases across two semesters.

## Overview

The project cross-validates ground-sensor air quality against satellite imagery to
assess regional pollution patterns. Automated crawlers scrape, clean, and
synchronise data from **Envisoft** (government monitoring), **IQAir** (commercial),
and **Himawari-8** / MODIS satellites.

## Phase 1 — Data collection and AOD–AQI correlation

Building the ground-truth dataset and testing how well satellite aerosol optical
depth (AOD) tracks surface air quality.

- `crawler.py`, `normalize_envisoft_data.py` — collect and clean hourly air-quality
  and weather records from the Envisoft network.
- `Station_aod/`, `AOD_data/` — satellite AOD extracted at station locations.
- `correlation.ipynb`, `aod_aqi_plot.ipynb`, `plot.ipynb` — correlation analysis.
- 📄 **Report:** [Final_Report.pdf](./Final_Report.pdf)

## Phase 2 — Graduation thesis: hourly PM2.5 estimation

Estimating hourly surface PM2.5 at unmonitored locations from satellite AOD
(Himawari, MODIS), TROPOMI trace gases, ERA5 reanalysis meteorology, and
neighbouring-station measurements (RFSI), evaluated under strict spatial
cross-validation (leave-one-station-out).

The central finding: a single national model does not extrapolate reliably to new
locations — the binding constraint is how well the sparse monitoring network
represents the country's diverse pollution regimes, not the algorithm or the
satellite input. The deployable result is a **Red River Delta regional model**
(`delta_rfsi_wind`): leave-one-station-out R² of 0.42 pooled / 0.27 mean-station /
0.39 median, and R² = 0.67 at the independent US Embassy Hanoi reference station.

- 📄 **Thesis:** [Thesis/DoAn.pdf](./Thesis/DoAn.pdf)
- `Thesis/scripts/` — the final pipeline (collection → quality control → features → model → evaluation).
- `Thesis/results/`, `Thesis/figures/` — outputs, reports, and figures.

## Repository layout

```
crawler.py, *.ipynb, Final_Report.pdf   Phase 1: data collection + AOD–AQI correlation
Thesis/DoAn.pdf                          Phase 2: compiled graduation thesis
Thesis/scripts/                          Phase 2: reproducible modelling pipeline
Thesis/results/, Thesis/figures/         Phase 2: outputs, reports, figures
scripts/, data/                          shared collection / processing utilities
```

## Reproducing

Python dependencies are in `requirements.txt`. Heavy raw data (merged feature
tables, satellite rasters, per-hour prediction dumps) are not tracked; the scripts
document how each stage is produced from the public sources — Envisoft ground
stations, IQAir, Himawari/MODIS AOD, TROPOMI, ERA5 (via Open-Meteo), and GPM.
