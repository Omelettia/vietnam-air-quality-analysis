# Satellite-Augmented Machine Learning for Hourly PM2.5 Estimation across Vietnam under Rigorous Spatial Validation

---

> **Author:** Nguyễn Tài Khoa
>
> **Degree:** Bachelor of Engineering in Information Technology (Vietnamese-Japanese Program)
>
> **Institution / Department:** School of Information and Communication Technology, Hanoi University of Science and Technology (HUST)
>
> **Supervisor:** TS. Trần Nguyên Ngọc
>
> **Date:** July 2026

---

## Abstract

Air pollution is among the most severe environmental health threats facing Vietnam, yet only about forty regulatory stations across the country's 331,000 km² supplied continuous PM2.5 over the study period — thirty-seven after quality control — leaving the majority of the population without local air-quality information. Satellite-augmented machine learning has been proposed as a remedy, with studies in densely monitored regions routinely reporting coefficients of determination above 0.8. This thesis asks whether such performance is attainable for Vietnam's sparse, tropical network, and it confronts a systematic methodological concern: the discrepancy between the random cross-validation used in most of the literature and spatial cross-validation, which reflects the operational task of predicting at unmonitored locations.

A 66-feature XGBoost-DART model was trained over 37 regulatory stations using MODIS and Himawari aerosol optical depth, TROPOMI trace gases, ERA5 meteorology, chemical-transport-model (CTM) outputs, and land-use predictors, and evaluated under leave-one-station-out (LOSO) validation and external validation against an independent low-cost-sensor (LCS) network and a reference-grade US Embassy monitor. The central result is a three-number arc: an R² of approximately 0.80 under random cross-validation collapses to approximately 0.20 under honest LOSO, and to a deployable per-station median near 0.04 once oracle information is withdrawn. Tier stratification by pollution level (the T4F scheme) is the single largest performance lever, yielding the LOSO figure, but it carries a circular dependency — assigning a station's tier requires knowing its mean PM2.5, the very quantity to be predicted — and seven families of satellite-observable proxy methods failed to resolve it. The dependency is nonetheless resolved from the monitoring network rather than from observables: a spatial-prior routing model that anchors each station's baseline to the observed means of its neighbours recovers a deployable per-station mean R² of 0.197 against an oracle ceiling of 0.203, with no dangerous low-to-high misclassification, conditional on proximity to the network. The CTM and existing global products fail distinctly: GEOS-CF shows an inverted diurnal cycle and a bias exceeding 200%, MERRA-2 attains an Index of Agreement of only 0.39, and GHAP achieves roughly 50% tier concordance. External validation is healthy within dense coverage, with an LCS median R² of 0.53 and an Embassy R² of 0.68, degrading with distance from the nearest anchor. The binding constraint is therefore station density rather than algorithmic capacity, and the densification of the network with calibrated low-cost sensors is the most tractable path to improved national accuracy.

**Keywords:** PM2.5, remote sensing, aerosol optical depth, machine learning, spatial cross-validation, Vietnam, low-cost sensors

---

## Table of Contents

**Chapter 1: Introduction**
- 1.1 Problem Statement
- 1.2 Current Solutions and Limitations
- 1.3 Objectives and Solution Direction
- 1.4 Contributions
- 1.5 Thesis Structure

**Chapter 2: Theoretical Background**
- 2.1 Context of the Prediction Problem
- 2.2 Related Research Results
- 2.3 The AOD–PM2.5 Relationship
- 2.4 XGBoost and DART Regularization
- 2.5 Spatial Validation Methodology

**Chapter 3: Proposed Methodology**
- 3.1 Solution Overview
- 3.2 Study Area and Data Sources
- 3.3 Data Quality Control
- 3.4 Feature Engineering
- 3.5 Tier-Stratified Modeling (T4F)
- 3.6 Deployable Approaches
- 3.7 Regional Delta Model

**Chapter 4: Analysis and Discussion**
- 4.1 Feature Importance Analysis
- 4.2 Evaluation of Chemical Transport Model Products
- 4.3 Satellite-Only versus Ground-Station Ablation
- 4.4 Validation of Existing PM2.5 Maps
- 4.5 Station Density as the Binding Constraint

**Chapter 5: Experimental Evaluation**
- 5.1 Evaluation Metrics
- 5.2 Temporal Prediction (Random Cross-Validation)
- 5.3 Spatial Prediction (LOSO — Global Configuration)
- 5.4 Tier-Stratified Results
- 5.5 Deployable Model Comparison
- 5.6 External Validation (LCS Network)
- 5.7 Summary of Unsuccessful Experiments
- 5.8 A Unifying View: Interpolation versus Extrapolation

**Chapter 6: Conclusions**
- 6.1 Summary of Findings
- 6.2 Recommendations for Vietnam's Air Quality Monitoring
- 6.3 Limitations
- 6.4 Future Work
- Closing

**References**

---

## List of Figures

**Figure 3.1** Spatial-prior routing pipeline — the six-stage deployable model architecture, from 66-variable input through XGBoost-DART ensemble, spatial-prior computation, three-regime routing, reliability guard with MODIS seasonal correction, to the final hourly PM2.5 estimate.

**Figure 4.1** GEOS-CF diurnal cycle of PM2.5 against ground observations, showing the near-antiphase relationship in which the product reaches its minimum at the dawn hour when observed surface concentrations peak.

**Figure 4.2** GEOS-CF per-station bias pattern across the monitoring network, illustrating the systematic two-to-three-fold overestimate of absolute concentrations.

**Figure 4.3** GEOS-CF representative station scatter of modeled against observed PM2.5, characterizing the product's negative per-station coefficient of determination.

**Figure 4.4** MERRA-2 representative station scatter of modeled against observed PM2.5, characterizing its weak hourly temporal correlation despite approximate mass balance.

**Figure 4.5** MERRA-2 aerosol species composition by region, showing that the product reproduces the broad north–south aerosol-mass gradient but not the hour-to-hour variability.

**Figure 4.6** GHAP annual-climatology station ranking against observed station means, showing moderate spatial skill alongside the inflated floor that misorders the cleanest sites.

**Figure 5.1** Per-station leave-one-station-out R² across the 37 regulatory stations, mapped onto Vietnam and colour-coded by skill: a bright, contiguous Red River Delta in the north, two bright but isolated Ho Chi Minh City points, and a dim coastal and Mekong periphery — a geography of pollution level rather than of region.

---

## List of Tables

**Table 5.1** Config H full — per-fold random cross-validation R² (five folds and pooled out-of-fold).

**Table 5.2** Random cross-validation performance across the configuration sweep (features, KFold R², MAE, RMSE).

**Table 5.3** Evaluation regimes for Config H — random cross-validation, fake (leaky) daily LOSO, honest daily and hourly LOSO, and the leakage gap.

**Table 5.4** Global LOSO configurations — no-grouping, true-tier (T4F), and oracle base-margin pooled, mean, median, and fraction-positive metrics.

**Table 5.5** The six DART training variants of definitive_v3 under LOSO.

**Table 5.6** Per-station LOSO R² for the best non-oracle configuration (T4F), sorted by tier and within-tier R².

**Table 5.7** Tier-stratified LOSO results for the base DART variant (per-tier station counts, mean PM2.5, R², RMSE, MAE, bias).

**Table 5.8** Tier-stratified LOSO results for the best-overall ensemble variant.

**Table 5.9** Region-by-region comparison of t2 stations, demonstrating that pollution level rather than region governs predictability.

**Table 5.10** The +0.21 tier-stratification gain — true-tier versus no-grouping comparisons across variants and station sets.

**Table 5.11** Deployable versus oracle model comparison (mean, median, fraction-positive, t3 mean).

**Table 5.12** Information cost of the missing station mean — oracle-minus-deployable gaps on matched station sets.

**Table 5.13** Spatial-prior routing model versus oracle ceiling and satellite-observable deployable band under no-GHAP LOSO.

**Table 5.14** Red River Delta regional LOSO results (delta base-margin, RFSI, and oracle ceiling).

**Table 5.15** External LCS validation subsets — LCS-only, all sites including Embassy, and the US Embassy single-site result.

**Table 5.16** Ten best-predicted external low-cost sites (R², distance to nearest KK station, hours, RMSE).

**Table 5.17** Worst-predicted external low-cost sites (R², distance, hours, RMSE).

**Table 5.18** Distance–skill relationship summary statistics (Pearson and Spearman correlations, near- and far-bin medians, distance range).

**Table 5.19** Spatial-prior routing pipeline on forty-six unseen low-cost sensors (train-on-forty, predict-unseen).

**Table 5.20** Summary of the seven unsuccessful experiment families (approach, best per-station mean R², and reason for failure).

**Table 5.21** Per-station exogenous-feature model evaluated as four different tasks (gap-fill, forecast, spatial map), by pollution tier.

---

## Acknowledgements

The author wishes to thank his supervisor, TS. Trần Nguyên Ngọc, and the examining committee for their guidance and valuable feedback throughout this work. Sincere thanks are extended to the staff of the School of ICT and the Department of Computer Engineering at HUST for their support. The author is deeply grateful to the Vietnam Center for Environmental Monitoring and the operators of the low-cost-sensor network for providing the ground-truth data on which this study rests. Finally, the author thanks his family, friends, and colleagues whose encouragement and support made the completion of this thesis possible.
