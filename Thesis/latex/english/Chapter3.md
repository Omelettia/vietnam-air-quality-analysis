# CHAPTER 3: PROPOSED METHODOLOGY

Chapter 2 established the theoretical foundations for satellite-based PM2.5 estimation: the AOD-PM2.5 relationship, the challenges of tropical retrieval, and the critical distinction between temporal and spatial validation. This chapter presents the specific methodology developed in this thesis. Section 3.1 provides an overview of the solution pipeline. Section 3.2 describes the study area, monitoring network, and satellite data sources. Section 3.3 details the data quality control procedures. Section 3.4 presents the feature engineering pipeline. Section 3.5 describes the tier-stratified modeling approach and the circular dependency it introduces. Section 3.6 presents the three deployable approaches developed to address this dependency. Section 3.7 describes the regional model applied to the Red River Delta subnetwork.


## 3.1 Solution Overview

The proposed methodology follows a multi-stage pipeline. In the first stage, ground-level PM2.5 measurements are collected from Vietnam's regulatory monitoring network and merged with satellite-derived features extracted via Google Earth Engine, meteorological reanalysis data from ERA5, and auxiliary spatial datasets (nighttime lights, building density). In the second stage, data quality control procedures identify and remove malfunctioning sensors and anomalous observations. In the third stage, feature engineering produces the predictor variables, including spatial interpolation from neighboring stations (RFSI), temporal aggregations, and station-level regime fingerprints derived from satellite time series. In the fourth stage, XGBoost models with DART regularization are trained and evaluated under three validation frameworks: random five-fold cross-validation, leave-one-station-out (LOSO) cross-validation, and external validation against an independent low-cost sensor network.

A distinctive aspect of this methodology is the systematic investigation of pollution-regime stratification (T4F), in which stations are grouped by their mean PM2.5 level and separate models are trained for each group. This technique provides the largest single improvement in prediction accuracy but introduces a circular dependency — the grouping variable is the prediction target — that motivates the development of three independent proxy approaches for deployable prediction at unmonitored locations.


## 3.2 Study Area and Data Sources

### 3.2.1 Study Area

Vietnam extends over approximately 331,000 km² along the eastern coast of the Indochinese Peninsula, spanning latitudes from approximately 8.5°N to 23.4°N. The country's geography encompasses the Red River Delta in the north, the narrow Central Highlands, and the Mekong River Delta in the south. This geographic diversity produces distinct PM2.5 regimes: the northern Red River Delta experiences severe winter pollution episodes driven by temperature inversions and transboundary transport from southern China, with annual mean PM2.5 exceeding 35 µg/m³ at urban stations; central coastal cities maintain moderate levels of 15–25 µg/m³; and southern stations range from moderately polluted (Ho Chi Minh City, approximately 21 µg/m³) to very clean (Mekong Delta rural sites, 5–10 µg/m³).

### 3.2.2 Ground Monitoring Network

Three categories of ground stations are used in this thesis, each serving a distinct role.

The primary training dataset consists of 40 regulatory stations from the Vietnam Center for Environmental Monitoring (CEM) network, designated as KK (Không Khí, meaning "air") stations. These stations report hourly PM2.5 alongside meteorological variables (temperature, relative humidity, pressure, wind speed and direction) via the Envisoft data platform. After quality control (Section 3.3), 37 stations are retained for model training, spanning the period from January 2023 to April 2026 with a total of 684,837 valid hourly PM2.5 observations (the full 40-station, pre-removal pool contains 727,635 valid hourly observations). The retained stations are unevenly distributed: 21 in the northern region (concentrated in the Hanoi–Bắc Ninh–Hải Dương corridor), 9 in the central region, and 7 in the southern region (the two removed Mekong Delta stations are both Southern).

The second category consists of 83 low-cost sensor (LCS) stations, also operated by CEM, using Plantower PMS-class laser-scattering sensors. These stations came online primarily from October 2025 and are concentrated in the greater Hanoi metropolitan area, Hải Phòng, and several northern provinces. Because they cover only the dry season (October–April), they are excluded from model training to avoid seasonal confounding and are reserved exclusively for external validation. These 83 LCS stations function only as held-out validation targets; they are never used as RFSI spatial anchors in any reported model. The RFSI neighbor field is constructed exclusively from the regulatory (KK) stations.

The third station is the US Embassy in Hanoi, which operates a reference-grade Met One BAM-1022 beta-attenuation monitor — the same instrument class used in the US EPA regulatory network. This station provides 9,729 valid hourly observations and serves as the highest-quality external validation point. It is located at 21.022°N, 105.819°E in the Ba Đình district of central Hanoi, approximately 3.0 km from the nearest KK training station (ĐHBK).

### 3.2.3 Data Collection Procedure

Data from multiple heterogeneous sources are collected and integrated through three separate pipelines.

**Ground observation data.** PM2.5, PM10, trace gases (NO₂, SO₂, CO, O₃), and meteorological variables (temperature, humidity, pressure, wind speed and direction) are collected from the TEDP platform (tedp.vn) — the public data portal of the Envisoft system operated by the Vietnam Center for Environmental Monitoring. The collection procedure consists of two stages: (i) an indexing stage that queries the TEDP search API to obtain record IDs for each station over the desired time range; and (ii) a detail-fetching stage that uses a multi-threaded collector (4 workers, rate-limited to 3 requests/second/worker) to call the detail API for each record ID to obtain full hourly data. The collector supports resumable operation by diffing already-downloaded record IDs against the index, and writes data atomically per station. Approximately 840,000 hourly records are collected for 40 KK stations spanning January 2023 to April 2026. All timestamps are normalized to Vietnam time (UTC+7).

**Satellite remote sensing data.** Satellite data are extracted at each station's coordinates using the Google Earth Engine (GEE) Code Editor. Five custom JavaScript scripts are developed, each processing a specific image collection: (1) MAIAC AOD — extracting a 5×5 pixel grid around each station; (2) TROPOMI — extracting center pixel values for NO₂, SO₂, CO, HCHO; (3) MODIS LST — extracting day/night surface temperature from both Terra and Aqua over a 5×5 grid; (4) GEOS-CF and MERRA-2 — extracting hourly PM2.5 and AOD at each station's grid cell; (5) GHAP/ACAG — extracting monthly and annual PM2.5 climatology. Extraction results are exported as yearly CSV files to Google Drive, then downloaded and packaged as ZIP archives. The GEE extraction process takes approximately 2–4 hours per script due to server-side computation limits.

**Reanalysis meteorological data.** ERA5 data (temperature, humidity, pressure, wind, planetary boundary layer height) are collected via the Open-Meteo Archive API — a free service providing access to ECMWF's ERA5 reanalysis without requiring a CDS account registration. A Python script queries the API for each station individually, with caching and automatic retry to handle rate limits. UTC timestamps are converted to UTC+7 at the download step.

The entire collection pipeline is fully reproducible: collection, extraction, and dataset-building scripts are available in the code repository at `scripts/collection/` and `scripts/data/`.

### 3.2.4 Satellite Remote Sensing Data

Satellite data are extracted at each station's coordinates using Google Earth Engine (GEE). Five custom GEE scripts were developed for this purpose, each targeting a specific data source and extraction geometry.

MODIS MAIAC AOD (MCD19A2, Collection 6.1) from Lyapustin et al. (2018) provides aerosol optical depth at 1 km resolution from the Terra and Aqua satellites, with overpasses at approximately 10:30 and 13:30 local time respectively. For each station, AOD is extracted at the center pixel and as statistical summaries (mean, standard deviation, maximum) over a 5×5 pixel grid (25 km²), capturing both local and neighborhood aerosol loading. Due to Vietnam's persistent cloud cover, MAIAC AOD retrieval is available for only 14.9% of hourly observations in the dataset — a critical limitation addressed through XGBoost's native missing-value handling.

Himawari-8 AHI (Advanced Himawari Imager) Level 2 aerosol products provide AOD at approximately 5 km resolution from geostationary orbit, offering hourly temporal resolution during daylight hours — a substantial advantage over the twice-daily polar-orbiting MODIS overpasses. For each station, center-pixel AOT, 5×5 grid statistics (mean, standard deviation, inner/outer ring means), Ångström exponent, single scattering albedo, and retrieval uncertainty are extracted. The grid-average AOT coverage (approximately 51% of daytime hours) exceeds the MODIS center-pixel rate, as spatial averaging recovers information from partially cloudy scenes. Together, the Himawari AHI features constitute 9 of the 10 features in the satellite AOD group, contributing 8.5% of total model gain.

TROPOMI (TROPOspheric Monitoring Instrument) aboard Sentinel-5P provides daily tropospheric column densities of nitrogen dioxide (NO₂), sulfur dioxide (SO₂), carbon monoxide (CO), and formaldehyde (HCHO) at approximately 5.5 km resolution. These trace gases serve as proxies for emission source types: NO₂ indicates traffic and power generation, SO₂ indicates industrial combustion, CO indicates incomplete combustion (motorbikes, cooking), and HCHO indicates biogenic and secondary organic aerosol formation. For each station, the center-pixel value is extracted.

MODIS Land Surface Temperature (MOD11A1/MYD11A1) from Wan et al. (2021) provides day and night surface temperature at 1 km resolution. Temperature inversions — indicated by elevated nighttime LST relative to air temperature — are associated with PM2.5 accumulation episodes. LST is extracted as statistical summaries over a 5×5 pixel grid.

### 3.2.5 Chemical Transport Model Products

Two CTM-derived products are included as features and evaluated as independent PM2.5 estimates.

GEOS-CF (GEOS Composition Forecasting, version 1.0) provides hourly global PM2.5 estimates at 0.25° (approximately 25 km) resolution. Values are extracted at each station's grid cell for both PM2.5 and AOD. As documented in Chapter 4, GEOS-CF exhibits a systematically inverted diurnal cycle in Vietnam relative to ground observations — predicting peak PM2.5 during midday rather than during the early morning hours when surface measurements peak — rendering its absolute values unreliable as prior estimates for this region.

MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2) provides hourly aerosol reanalysis at 0.5° × 0.625° resolution. PM2.5 is derived from the speciated aerosol mass concentrations (sulfate, organic carbon, black carbon, dust, sea salt). MERRA-2 PM2.5 and AOD are extracted at each station's grid cell. Global validation indicates that MERRA-2's Index of Agreement for Southeast Asia is 0.39 — the worst among all global regions.

### 3.2.6 Meteorological Reanalysis Data

ERA5 reanalysis from Copernicus (2023) provides hourly meteorological fields at 0.25° resolution. The following variables are extracted at each station: 2-meter temperature, 2-meter dewpoint temperature (from which relative humidity is derived), 10-meter U and V wind components, surface pressure, total precipitation, and planetary boundary layer height (PBLH). ERA5 wind data is used in preference to in-situ Envisoft wind measurements, as a quality audit revealed that Envisoft wind direction has a circular correlation of only 0.21 with ERA5 — likely due to local obstructions at sensor mounting sites.

For in-situ temperature, relative humidity, and atmospheric pressure, Envisoft station measurements are used where available and supplemented with ERA5 values during gaps. OpenMeteo API data provides additional local meteorological parameters.

### 3.2.7 Auxiliary Spatial Data

Nighttime light intensity from the VIIRS DNB (Day/Night Band) annual composites serves as a proxy for urbanization and economic activity. Building footprint area within 1 km of each station, derived from Google's Open Buildings dataset (Google, 2023), provides a measure of built-environment density. GHAP (Global High-resolution Air Pollutants) monthly and annual PM2.5 climatology at 1 km resolution from Wei et al. (2023) provides a satellite-derived long-term baseline, used both as a feature and as a benchmark for evaluation.


## 3.3 Data Quality Control

Data quality control is applied at three levels: observation-level filtering, feature-level cleaning, and station-level removal.

### 3.3.1 Observation-Level Filtering

Three categories of anomalous PM2.5 observations are identified and removed. First, zero or negative readings, which in Plantower-class sensors typically indicate sensor floor artifacts rather than genuinely zero PM2.5, are excluded (2,448 observations). Second, flat-line sequences — defined as five or more consecutive hours with identical PM2.5 readings to the first decimal place — are removed, as these indicate a stuck or frozen sensor (16,209 observations). Third, stuck-low runs — defined as 48 or more consecutive hours with 0 < PM2.5 ≤ 2 µg/m³ — are removed, as sustained near-zero readings at this level indicate a sensor operating at its noise floor rather than measuring genuine ambient concentrations (14,683 observations). These three masks overlap where, for example, a stuck-low sequence also reads identically at 0.1 µg/m³ precision; the union removes 25,093 unique observations (3.4% of 727,635 valid PM2.5 records). While this fraction is modest, the affected observations are concentrated at specific stations and time periods, and their removal produces a measurable improvement of +0.011 in LOSO R².

### 3.3.2 Feature-Level Cleaning

Several satellite and meteorological features exhibit physically implausible values that indicate retrieval or sensor errors. The following cleaning rules are applied: relative humidity values below 5% are set to missing (indicating dry-bias sensor error); atmospheric pressure values outside the range 900–1,100 hPa are set to missing; MODIS Angström exponent values below 0 or above 3 are set to missing (indicating failed retrieval); and Envisoft wind speed values of exactly zero are set to missing (as true zero wind is physically rare and this value typically indicates sensor malfunction). These rules affect 86,912 observations (11.9% of the dataset) across the four feature groups and improve LOSO R² by +0.004.

### 3.3.3 Station-Level Removal

Three KK stations are removed from the training dataset based on statistical indicators of sensor malfunction, validated against physical plausibility assessment and satellite-derived PM2.5 estimates. The deletion criterion is sensor-intrinsic diagnostics (exact-zero fraction, flat-line runs, coefficient of variation); the GHAP and other satellite-derived PM2.5 values are used only to corroborate the resulting removals, not as the deletion criterion itself.

Đà Nẵng – Phạm Hùng (15.996°N, 108.207°E) exhibits a mean PM2.5 of 6.2 µg/m³ with a median of only 2.5 µg/m³, indicating an extreme right-skewed distribution dominated by near-zero readings. The station shows a 9.9% zero-or-negative fraction (1,602 of 16,221 observations) and 11.4% of all observations flagged by at least one QC mask — compared to under 1% for normally functioning stations. These are the canonical symptoms of a degraded Plantower PMS5003 sensor as documented by AirGradient (2024).

Sóc Trăng (9.614°N, 105.968°E) records a mean PM2.5 of 6.7 µg/m³, less than half the GHAP satellite estimate of approximately 15 µg/m³ for the same grid cell. The coefficient of variation of 1.51 is anomalously high, and 10.1% of observations are flagged by QC masks — predominantly flatline and stuck-low sequences — indicating intermittent sensor failure.

Trà Vinh – Đông Hải (9.576°N, 106.488°E) reports a mean PM2.5 of 5.7 µg/m³ against a GHAP estimate of approximately 14 µg/m³. While the temporal statistics appear cleaner than the other two removed stations (zero exact-zero readings, only 1.0% of observations flagged), the 2.5× discrepancy with every independent satellite product and field study in the Mekong Delta indicates a systematic low bias, possibly from a calibration offset.

Two additional stations are flagged for cautious interpretation but retained. Quảng Ninh – Nhà máy Tuyển Than is sited within a heavily water-sprayed coal processing compound and reads 6.6 µg/m³, likely reflecting the micro-environment rather than ambient air. Thái Nguyên reads 55.2 µg/m³ — consistent with IQAir's reported city average of 56.4 µg/m³ and the station's proximity to the TISCO steelworks — but is flagged as a source-impacted site rather than a representative urban monitor.

After station removal, 37 KK stations are retained for model training and LOSO evaluation.


## 3.4 Feature Engineering

The feature engineering pipeline produces 66 predictor variables from the data sources described in Section 3.2, organized into six groups.

### 3.4.1 Spatial Interpolation Features (RFSI)

Random Forest Spatial Interpolation (RFSI) features provide the model with information from neighboring ground stations. For each target observation at station $s$ and time $t$, the three nearest stations with valid PM2.5 readings at time $t$ are identified. Their PM2.5 values (PM25_nn1, PM25_nn2, PM25_nn3), distances (dist_nn1, dist_nn2, dist_nn3), and an inverse-distance-weighted average (PM25_nn_idw) are computed. Temporal lags of the nearest neighbor (PM25_nn1_lag1h, PM25_nn1_lag3h) are also included to capture persistence in neighboring conditions.

Critically, during LOSO cross-validation, the held-out station is excluded from the neighbor pool for all remaining stations, preventing spatial leakage. The neighbor pool is drawn entirely from the regulatory (KK) stations — 37 nationwide, of which 12 fall within the Red River Delta subnetwork. The 83 low-cost sensors are reserved as external validation targets and are not used as RFSI anchors in any reported model.

RFSI features collectively account for approximately 37% of total model gain in feature importance analysis, making them the most important feature group. This dominance reflects the strong spatial autocorrelation of PM2.5 at distances of 10–50 km, particularly in the dense northern station cluster.

### 3.4.2 Meteorological Features

ERA5 and station-derived meteorological features include temperature, relative humidity, atmospheric pressure, wind speed, U and V wind components, PBLH, and total precipitation. Derived features capture weather persistence: PBLH_min_24h (minimum PBLH in the preceding 24 hours, indicating the depth of the shallowest recent mixing layer), ventilation coefficient minimum (VC_min_24h = wind speed × PBLH), stagnation hours (count of hours in the preceding 24 where ventilation coefficient falls below a threshold), the 6-hour change in relative humidity (dRH_6h), and 48-hour cumulative rainfall (rain_sum_48h, indicating wet deposition washout).

### 3.4.3 Satellite Observation Features

Direct satellite observations include MODIS MAIAC AOD at center pixel and 5×5 grid statistics (mean, standard deviation), MODIS Angström exponent (indicating aerosol size distribution), TROPOMI trace gas columns (NO₂, SO₂, CO, HCHO) at center pixel, and MODIS land surface temperature statistics. A composite emission proxy (smart_v1_center, smart_v1_contrast) combines nighttime lights, building density, TROPOMI NO₂, and SO₂ into a single indicator of local emission intensity and its spatial gradient.

### 3.4.4 Chemical Transport Model Features

GEOS-CF hourly PM2.5 and AOD, MERRA-2 hourly PM2.5 and AOD, and GHAP monthly PM2.5 climatology are included as features. Despite the documented biases of these products in Vietnam (Chapter 4), XGBoost can potentially learn regional correction factors. In practice, feature importance analysis reveals that these features contribute minimally — and in some configurations, their removal improves performance — confirming the CTM evaluation findings.

### 3.4.5 Temporal Features

Hour of day and day of year are encoded as sine/cosine pairs to capture cyclic patterns: hour_sin, hour_cos, day_of_year_sin, day_of_year_cos, month_sin, month_cos. These features allow the model to learn diurnal and seasonal PM2.5 patterns without imposing a specific functional form.

### 3.4.6 Station Regime Fingerprints

A novel feature group introduced in the later stages of this thesis's experiments, station regime fingerprints capture the temporal statistics of satellite observations at each station over a 30-day rolling window: aod_30d_mean_stn (30-day mean AOD at the station), hcho_30d_mean_stn, co_30d_mean_stn, aod_30d_p90_stn, co_30d_std_stn, and similar statistics for other satellite variables. These features encode "what kind of place is this" from satellite time series alone, without requiring ground PM2.5 measurements. In experiment v5b, eight station regime fingerprints appeared among the top 20 features by model gain, with combined importance approximately 28% — nearly matching the RFSI contribution — indicating that satellite observations do carry information about station-level pollution regime, even if this information is insufficient to fully resolve the T4F circular dependency (Section 3.5).


## 3.5 Tier-Stratified Modeling (T4F)

### 3.5.1 Motivation and Definition

The most impactful technique discovered during the experimental development of this thesis is Tier-4-Fold (T4F) stratification. The 37 training stations are grouped into four tiers based on their mean PM2.5 concentration: t0 (< 10 µg/m³, 7 stations), t1 (10–20 µg/m³, 10 stations), t2 (20–35 µg/m³, 11 stations), and t3 (> 35 µg/m³, 9 stations). The three broken sensors removed in Section 3.3.3 were all clean-tier (t0) stations, which is why the t0 count drops from the pre-removal value of 10 to 7 while the other tiers are unchanged. The 10/20/35 µg/m³ tier boundaries were fixed a priori from the WHO/national air-quality categories and were not tuned on LOSO performance; tuning them on the held-out metric would have opened a leakage channel. The t3 boundary at 35 µg/m³ sits within a wide gap in the station-mean distribution and is robust to small perturbations, whereas a ±2 µg/m³ shift at the 10 and 20 µg/m³ boundaries would reassign roughly five to seven stations between adjacent tiers. Separate XGBoost models are trained for each tier, and during LOSO evaluation, each held-out station is predicted using only its own tier's model.

T4F improves LOSO R² by approximately +0.17 (within-file, measured against the same-file no-grouping baseline of 0.027) and up to +0.26 depending on the aggregation metric and reference configuration, with the most defensible estimate near +0.17. This makes it the single largest performance gain of any technique explored. The improvement arises because stations within the same tier share similar baseline PM2.5 levels, diurnal amplitudes, and seasonal patterns, allowing the tier-specific model to focus on modeling deviations from a shared regime rather than simultaneously spanning the full 5–55 µg/m³ range of Vietnamese stations.

### 3.5.2 The Circular Dependency

T4F requires knowing each station's mean PM2.5 to assign its tier — but mean PM2.5 is precisely the quantity the model aims to predict at unmonitored locations. This creates a circular dependency that is trivially satisfied during LOSO evaluation (where the held-out station's true tier is known from historical data) but fundamentally problematic for deployment at new locations.

This circular dependency is not unique to this thesis; it reflects a general challenge in any modeling approach that benefits from knowing the target variable's distribution a priori. What is distinctive about this thesis's contribution is the systematic, exhaustive testing of proxy methods to resolve the dependency, documented in Section 3.6 and evaluated in Chapter 5. The key observation that ultimately resolves it (Section 3.6.4) is that the tier label is nothing more than a coarse encoding of the station's baseline PM2.5 level, and that this level — unlike the tier label itself — can be estimated at an unmonitored location not from satellite observables but by spatial interpolation from the surrounding monitoring network.


## 3.6 Deployable Approaches

Four approaches are developed to approximate T4F-level performance without requiring ground-truth PM2.5 at the target location. The first three (Sections 3.6.1–3.6.3) estimate the station's tier from satellite, emission, and land-use observables and converge at a low deployable ceiling, establishing that no satellite-observable proxy recovers the tier gain. The fourth (Section 3.6.4) abandons tier estimation entirely and instead interpolates the station's baseline level from neighbouring monitors; this is the approach carried forward as the deployable model, and it recovers nearly the entire oracle gain wherever the network provides usable neighbours.

### 3.6.1 Soft-Gate Mixture of Experts

The soft-gate mixture of experts (MoE) trains four tier-specific XGBoost models (the "experts") alongside a logistic regression gating model that predicts tier membership probabilities from nine observable features: GHAP annual mean, smart_v1_center (emission composite), building_area_1km, TROPOMI column densities (SO₂, CO, HCHO, NO₂), ACAG AOD climatology, and latitude. For each prediction, the gating model produces a probability distribution over the four tiers, and the final prediction is the probability-weighted average of the four expert predictions.

The gate achieves 58% accuracy in leave-one-station-out evaluation (compared to 25% random baseline): 78% accuracy for t3 stations, 70% for t0, but only 45% for t2 and 40% for t1 — reflecting the difficulty of distinguishing intermediate pollution levels from satellite observables alone. The resulting deployable model achieves a per-station mean R² of +0.045 — modest but consistently the first positive per-station mean achieved by a fully deployable configuration.

### 3.6.2 Two-Phase Predicted Baseline

The two-phase approach separates the problem into (i) predicting the station's mean PM2.5 from satellite and land-use features, and (ii) using this predicted mean as a base margin for the hourly XGBoost model. Phase 1 uses a Ridge regression model trained with leave-one-station-out validation on 13 station-level features (GHAP annual, TROPOMI climatologies, nighttime lights, building density, latitude, longitude), achieving LOO R² = 0.591 and MAE = 7.08 µg/m³ for predicting station annual mean PM2.5.

The predicted station mean is then supplied to XGBoost as a base_margin — an initial prediction that the boosted trees learn to correct. This is conceptually analogous to providing the model with an informed prior about the station's pollution level. The deployable two-phase configuration (pred_bm) is a fully deployable design, but it does not beat the global baseline overall: its n-weighted pooled R² is approximately −0.04 and its per-station mean R² is approximately −0.063. The gains are confined to the most polluted stations, where the t3-tier mean R² reaches approximately 0.39; the per-station median R² is only approximately +0.04. The negative pooled and mean values indicate that the Phase 1 prediction errors are large enough to degrade performance at the majority of stations, particularly those with low PM2.5 where the predicted baseline systematically overshoots.

### 3.6.3 Satellite-Derived Regime Fingerprints

The third approach integrates the station regime fingerprint features described in Section 3.4.6 directly into a single global model (no tier stratification). By providing XGBoost with 30-day satellite statistics at each station, the model can implicitly learn to condition its predictions on the station's pollution regime without explicit tier assignment.

This approach achieves a per-station median R² of +0.043 — comparable to the soft-gate MoE — with eight regime fingerprint features appearing among the top 20 by model gain. The convergence of three independent approaches (soft gate: +0.045, predicted baseline: +0.036 median, regime fingerprints: +0.043 median) at approximately the same performance level provides strong evidence that this represents the deployable ceiling achievable from satellite observables alone, absent ground-truth calibration data.

### 3.6.4 Spatial-Prior Routing

The three approaches above share a common strategy — estimate the station's tier, then act on that estimate — and a common failure: the tier cannot be recovered from satellite observables with enough accuracy to be useful. The fourth approach abandons that strategy. It does not classify the tier at all. Instead it estimates the station's *baseline level* directly by spatial interpolation from the surrounding monitoring network, and uses that estimate to anchor and route a set of prediction streams. This is the deployable resolution of the circular dependency and the model carried forward in the remainder of the thesis.

The estimator is a distance-weighted **spatial prior**: for a target location, the prior is the Gaussian-distance-weighted mean of the observed mean PM2.5 of the *k* nearest training stations, $\text{prior} = \sum_i w_i \bar{y}_i / \sum_i w_i$ with $w_i = \exp(-d_i^2/s^2)$, where $d_i$ is the great-circle distance to training station $i$, $s$ a length scale of roughly 55–65 km, and $\bar{y}_i$ that station's own observed mean. The held-out station is always excluded from its own prior, so the estimate uses only the surrounding network and never the target's own data. A leave-one-station-out trust calibration accompanies the prior: it scores how reliable the spatial prior is around each location from the agreement of nearby station means and the effective number of usable neighbours, which later drives the reliability layer. This is the same spatial-interpolation principle that Chapter 4 identifies as the model's dominant predictor, but applied at the level of the station *baseline* rather than the hourly co-variation — and it is precisely the quantity the satellite-observable proxies could not supply. It is a substantially stronger station-mean estimator than the Phase-1 satellite regression of Section 3.6.2 (leave-one-out R² ≈ 0.59), because the observed concentrations of nearby monitors carry the regional pollution level more directly than any satellite or land-use feature.

The prior is consumed not as a hard base margin — the failure mode of Section 3.6.2, where an imperfect predicted mean injected directly causes systematic overshoot at clean sites — but conservatively, through two mechanisms. First, **stream construction**: four target-free streams are trained on all forty stations and preserve different aspects of the signal. A log-target XGBoost stream is conservative and stable; a raw-target stream with high-concentration sample weighting preserves high-event amplitude; a fixed 70/30 blend trades the two; and a gated stream raises the raw weight only where the local context is polluted. The log and raw streams are, in effect, the continuous analogue of a low-tier and a high-tier expert, but each is trained on the entire network rather than on a ten-station tier fragment, so neither is data-starved. Second, **routing**: the spatial prior selects and anchors among these streams by regime — where the prior indicates a polluted neighbourhood the gated stream is used; where it indicates a clean neighbourhood the log stream is shifted partway toward the prior; and in the moderate range the blend is shifted partway toward the prior (shrinkage coefficients near 0.40). The shift moves the predicted baseline toward the spatially interpolated level without discarding the stream's temporal shape, which is what avoids the overshoot that defeated the hard-base-margin two-phase design.

A single post-prediction **reliability guard** completes the design. It does not choose streams or alter the core prediction; it evaluates the finished prediction against the trustworthiness of the local neighbourhood and emits map-display flags, applying a numeric correction only in one clearly contradictory case. Where the model predicts non-high but reliable nearby evidence indicates a high-pollution neighbourhood, the location is *flagged* as a hidden-high warning rather than numerically lifted — keeping the guard from acting as a second router and avoiding low-to-high catastrophes. Where the model predicts high but a reliable local prior is much lower, that single clear false-high contradiction is suppressed toward the moderate boundary. This consolidates into one interpretable object what earlier exploration had spread across several overlapping post-processing layers. A compact MODIS seasonal-AOD correction is applied only after the baseline has been narrowed by the prior, adjusting magnitude once the regime is fixed.

Figure 3.1 summarizes the complete spatial-prior routing pipeline from input features through to the final hourly PM2.5 prediction.

![Figure 3.1: Spatial-prior routing pipeline — the six-stage deployable model architecture, from 66-variable input through XGBoost-DART ensemble, spatial-prior computation, three-regime routing, reliability guard with MODIS seasonal correction, to the final hourly PM2.5 estimate.](fig_3_pipeline.png)

Evaluated under leave-one-station-out on the forty thesis stations (with GHAP removed, so the map depends on no climatological PM2.5 product), this pipeline recovers nearly the entire oracle tier gain: a per-station mean R² of 0.197 and median of 0.117 against the oracle T4F ceiling of 0.203 and 0.147 — and with better class safety than the oracle, producing zero high-to-non-high and zero dangerous low-to-high station flips. The quantitative comparison against the oracle and the satellite-observable deployable band is presented in Section 5.5, and the external stress test on unseen low-cost sensors in Section 5.6. The essential point of method is stated here: the circular dependency is resolved not by guessing the tier from satellites but by interpolating the baseline from neighbours, which makes its skill conditional on the proximity of those neighbours — the same density constraint that governs the rest of the thesis.


## 3.7 Regional Delta Model

In addition to the nationwide model, a regional model is developed specifically for the Red River Delta — the densest station cluster, containing 12 KK stations within an approximately 150 km radius. The motivation is that stations in this region share meteorological conditions, emission source types (urban traffic, industrial parks, rice-straw burning), and transboundary pollution exposure, potentially enabling more accurate within-region prediction.

The delta model uses the same feature set and XGBoost-DART configuration as the nationwide model but is trained exclusively on the 12 delta stations. LOSO evaluation is performed within the 12-station set, and external validation is performed by training on all 12 delta stations and predicting at the 39 LCS stations and US Embassy — none of which were used in model development.

This regional approach produces LOSO results of mean R² = 0.302 and median R² = 0.433 (for the delta_rfsi configuration), modestly outperforming the deployable tier-blind nationwide configurations on the same 12 delta stations (0.302 vs 0.271 for the no_t4f variant and 0.290 for the no_group variant). This advantage holds only over deployable nationwide variants: a true-tier (oracle) nationwide model, which assumes each station's tier is known a priori, scores higher on the same 12 stations (mean R² ≈ 0.43), so the regional model's edge is over deployable nationwide configurations, not over the oracle ceiling. More importantly, the external LCS validation achieves a median R² of 0.529 with 85% of stations showing positive R², and R² = 0.684 at the US Embassy reference station. These results should be read as a nearest-station feature-transfer plus sensor-agreement test rather than as genuine spatial prediction at previously unseen sites: at each held-out low-cost sensor, the satellite, AOD, TROPOMI, and ERA5 meteorology features are taken from the nearest regulatory (KK) station's hourly record, and only the target PM2.5 comes from the held-out sensor itself (the RFSI spatial anchors remain correctly localized to the target coordinates). The headline external R² therefore partly reflects feature transfer and instrument agreement between the low-cost sensor and the nearby regulatory station, and the observed distance-decay partly reflects growing feature-substitution error as the nearest KK station becomes more distant. With this caveat, the external validation results remain competitive with the best published spatial cross-validation results internationally (Kawano et al. 2025 reported spatial-CV R² = 0.67 for India with approximately 1,000 training stations).


This chapter has described the complete methodology: data sources and quality control, the 66-feature engineering pipeline, the T4F stratification discovery, three deployable proxy approaches, and the regional delta model. Chapter 4 will analyze the results in depth, examining feature importance patterns, CTM product failures, and the role of station density. Chapter 5 will present the quantitative experimental evaluation across all validation frameworks.