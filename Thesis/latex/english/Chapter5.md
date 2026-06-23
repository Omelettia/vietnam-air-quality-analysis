# CHAPTER 5: EXPERIMENTAL EVALUATION

Chapter 4 explained what the model learns and why it behaves as it does: spatial interpolation from ground neighbors dominates its predictive power, satellites partially substitute for that signal, the chemical-transport-model products fail in three distinct registers, and station density — expressed as proximity to the nearest anchor — is the binding constraint. This chapter presents the full quantitative evaluation that those analyses were built to explain. Section 5.1 defines the metrics and explains why pooled and per-station summaries must be reported together. Section 5.2 reports the random cross-validation result, the inflated "published-paper" number that the rest of the chapter exists to deflate. Section 5.3 presents the honest leave-one-station-out (LOSO) evaluation of the global model and quantifies the leakage gap between the two regimes. Section 5.4 stratifies the LOSO result by pollution tier, establishing the monotone, model-invariant gradient by which predictability scales with pollution level — while treating the further claim that level rather than region governs that skill as a hypothesis the data cannot yet separate — and isolating the tier-stratification gain (approximately +0.17 within-file, up to +0.26 across framings). Section 5.5 compares every genuinely deployable configuration against the oracle ceiling and measures the information cost of not knowing a station's mean. Section 5.6 presents the external validation against the low-cost-sensor network and the US Embassy reference station. Section 5.7 summarizes the seven families of unsuccessful experiments that bound the solution space. Section 5.8 draws the three frameworks together under a single interpolation-versus-extrapolation principle that explains why the same model spans R² from below 0.1 to above 0.8. Throughout, the central three-number arc committed in Chapter 1 is made explicit: an R² of approximately 0.80 under random cross-validation, approximately 0.20 under honest LOSO, and a deployable per-station median near 0.04.


## 5.1 Evaluation Metrics

All models in this thesis are evaluated against hourly ground-truth PM2.5 using four metrics. The coefficient of determination computed at hourly resolution, denoted R²_hourly, measures the fraction of variance in observed hourly concentrations that the model explains relative to a constant-mean baseline; it is the primary headline metric and is bounded above by 1, equal to 0 when the model does no better than predicting the long-run mean, and unbounded below, becoming negative whenever the model is worse than that mean. The root-mean-square error (RMSE) and mean absolute error (MAE), both in µg/m³, quantify the typical magnitude of prediction error, with RMSE weighting large errors more heavily; and the bias, the mean signed residual in µg/m³, records systematic over- or under-prediction. These are reported alongside R² because two models with identical R² can differ substantially in absolute accuracy, and because at clean sites a near-zero or negative R² coexists with a small RMSE — the variance to be explained is itself small.

A methodological point central to this chapter is the distinction between pooled R² and per-station mean (or median) R², and the reason both must be reported. The pooled R² is computed by concatenating the hourly residuals of all stations into a single vector and evaluating the coefficient of determination once over the entire dataset; it is dominated by high-variance, high-concentration stations because those stations contribute both the most variance to explain and, typically, the most hours. The per-station R², by contrast, is computed separately for each held-out station and then summarized across stations by its mean or median; it weights every station equally regardless of its concentration or record length and therefore exposes how the model performs at the typical site rather than at the few sites that dominate the pooled variance. The two diverge sharply in this dataset: a small number of clean, low-variance stations with negative R² drag the per-station mean far below the per-station median, while the pooled figure, anchored by the polluted northern stations, sits higher than either. Reporting only one of the three summaries would systematically mislead — the pooled value overstates typical performance, the per-station mean understates it because of a handful of badly negative clean sites, and the per-station median is the most robust central tendency. Where a result hinges on the choice of summary, all three are given. A practical caveat applies throughout: the per-station LOSO result files persist only the per-station R²_hourly (with its accompanying RMSE, MAE, and bias) and do not retain the raw hourly residuals, so for those experiments a true cross-station pooled R² cannot be recomputed after the fact; where the term "pooled" is used for the LOSO configurations it denotes the n-hours-weighted mean of the per-station values, which is stated explicitly at each use and is not identical to a single residual-vector R².

A third, deliberately practical indicator accompanies the variance metrics: the percentage of held-out stations achieving a positive R². Because a negative R² means the model is worse than simply assuming the station's own long-run mean, the fraction of stations above zero answers a question that the average R² obscures — at how many locations would the model actually be useful as a deployed estimator? A configuration can post a respectable mean R² while failing outright at nearly half its stations, and the fraction-positive indicator is what surfaces that failure. It is reported for every LOSO and external-validation configuration in this chapter.


## 5.2 Temporal Prediction (Random Cross-Validation)

The first evaluation regime is random cross-validation, the standard against which the bulk of the published PM2.5 literature reports its accuracy. In this regime the 244,489 hourly observations of the main model are shuffled and partitioned without regard to station identity, so that hours from every station appear in both the training and the test folds. One naming clarification is warranted at the outset: the early project plan described this as ten-fold cross-validation, but the implementation that produced the result files uses a five-fold shuffled partition (a KFold split with five folds, shuffling enabled, and a fixed random seed). No ten-fold result file exists anywhere in the experimental record; the supported claim is therefore five-fold random cross-validation, and the figures below are reported on that basis. The distinction does not affect the headline number — the fold count changes the variance estimate, not the central tendency — but it is recorded here for accuracy.

The model evaluated in this regime is Config H — thirty-two features driving an XGBoost-DART regressor with eight hundred trees at depth eight — the configuration for which random cross-validation results were computed. The sixty-six-feature definitive model described in Chapter 3 was characterized under leave-one-station-out validation rather than re-run under random cross-validation; the leakage gap of Section 5.3 is therefore demonstrated on Config H, the one configuration for which both regimes were evaluated on an identical model, and the definitive model is the object of the spatial analysis from Section 5.3 onward. Under five-fold random cross-validation Config H achieves an out-of-fold R² of 0.8125, with a pooled MAE of 7.63 µg/m³ and RMSE of 11.12 µg/m³ over all 244,489 hours. This is the "published-paper number" — the figure directly comparable to the R² > 0.8 results that dominate the literature for China, the United States, and Europe — and it is the value committed in Chapter 1 as the random-cross-validation anchor of approximately 0.80. The result is stable across folds: the five out-of-fold R² values span only 0.8078 to 0.8172, a spread of well under one point, confirming that the 0.81 figure is not an artifact of a fortunate partition.

*Table 5.1: Config H per-fold out-of-fold R² under five-fold random cross-validation, with the pooled value.*

| Config H FULL — per-fold random cross-validation R² |
|---|
| fold 0 = 0.8141 |
| fold 1 = 0.8095 |
| fold 2 = 0.8139 |
| fold 3 = 0.8172 |
| fold 4 = 0.8078 |
| pooled out-of-fold = 0.8125 |

Across the full configuration sweep the random-cross-validation R² occupies a band consistent with the approximately 0.72–0.81 range anticipated in the outline, with the main-model figure sitting at the top of that band and the most feature-rich configuration just above it. A meteorology-only model (Track A, twelve features) and the unified meteorology-plus-raw-AOD model both reach 0.7386; the twenty-three-feature Config G reaches 0.7398; Config H at intermediate hyperparameters reaches 0.7217; the main Config H at full hyperparameters reaches 0.8125; and the forty-feature Config I reaches 0.8348. The lower bound of the band is thus a leaner feature set or shallower trees, and the upper bound is the richest configuration, with the designated main model deliberately chosen at 0.8125 as the most defensible compromise between fit and parsimony.

*Table 5.2: Random cross-validation R², MAE, and RMSE across the configuration sweep, from the meteorology-only baseline to the most feature-rich model.*

| Config | Features | Random KFold R² | MAE | RMSE |
|---|---|---|---|---|
| Track A (meteorology-only) | 12 | 0.7386 | 8.90 | 13.14 |
| Unified (meteorology + raw AOD) | 16 | 0.7386 | 8.90 | 13.14 |
| Config G FULL | 23 | 0.7398 | — | — |
| Config H (intermediate, n300/d7) | 32 | 0.7217 | 9.30 | 13.55 |
| **Config H FULL (main model)** | **32** | **0.8125** | **7.63** | **11.12** |
| Config I FULL | 40 | 0.8348 | 7.20 | 10.44 |

Independent robustness sweeps under alternate fold seeds corroborate this level: across five separate KFold experiment files the Config-H-class model returns R² values clustered between 0.7225 and 0.8117, with RMSE near 11–13 µg/m³ and MAE near 6.5–8.0 µg/m³, confirming that approximately 0.80–0.81 is the genuine random-cross-validation performance of this model rather than a single favorable run.

What this regime establishes, and what it does not, must be stated precisely. It confirms that the feature set and the learning algorithm are capable of representing the hourly dynamics of Vietnamese PM2.5: given hours from a station in training, the model reconstructs that station's remaining hours to within an R² of 0.81. But because every test hour belongs to a station the model has already seen, the result measures temporal interpolation under station-identity leakage, not spatial prediction at an unmonitored location. Two leakage channels make this concrete. The coarse channel is station identity itself: the model implicitly learns each station's baseline level from its training hours and need only predict deviations around a baseline it already knows. The fine channel is documented in the feature set — one of the thirty-two features, the per-station monthly AOD climatology, is computed per station and per month, so each station's own climatological signature is baked directly into its predictors; this single feature is the largest contributor to the model's gain in this regime. The 0.81 figure therefore confirms that the model works as a temporal interpolator but says essentially nothing about its capacity to generalize to a location with no ground history. Quantifying exactly how much of that 0.81 is leakage rather than spatial skill is the task of Section 5.3.


## 5.3 Spatial Prediction (LOSO — Global Configuration)

Leave-one-station-out cross-validation removes the station-identity leakage that inflates the random-cross-validation figure. Each of the thirty-seven retained stations is held out in turn, the model is retrained on the remaining stations with the held-out station excluded from every other station's neighbor pool, and the held-out station is then predicted as if it were an unmonitored location. This is the operationally relevant task — estimating PM2.5 where no ground measurements exist — and it is the primary evaluation metric of this thesis.

A clarification of sources is necessary, because the global LOSO configurations are distributed across several experiment files rather than concentrated in one. The model-variant file definitive_v3 contains six DART training variants — a base model, three nearest-neighbour variants, a tuned model, and an ensemble — each of which carries the true four-tier label and is therefore a true-tier model; the tuned variant in that file is byte-identical, across all thirty-seven stations, to the configuration labelled "T4F" in the grouping-experiment file, which establishes the linkage between the two. The genuinely tier-blind global model — a single pooled model with no tier grouping — is the "no-grouping" configuration in the grouping-experiment file, and the oracle upper bound that injects each held-out station's true mean PM2.5 as an additive offset is the "oracle base-margin" configuration in the two-phase file. These three are distinct objects, and distinguishing them is essential because the headline LOSO arc depends on which is which.

The contrast with random cross-validation is the central result of the chapter. The same Config H model that scored 0.8125 under five-fold random cross-validation falls to a per-station mean R² of 0.2093 under honest hourly LOSO — a drop of 0.603 attributable almost entirely to the removal of station-identity leakage. The per-station median under honest LOSO is 0.3305, the pooled MAE rises to 11.99 µg/m³, and the RMSE to 16.86 µg/m³. The 0.603 gap is the quantitative content of this thesis's central methodological claim: roughly three-quarters of the celebrated 0.81 is leakage, not generalization.

A point of precision on what the 0.603 gap differences. The honest hourly LOSO figure of 0.2093 is the result of the fifteen-station Config H run, in which the thirty-two-feature Config H model is retrained per held-out station; it is distinct from the thirty-seven-station definitive model reported from Section 5.3 onward, whose per-station mean is 0.1989. The leakage gap is therefore measured on the identical Config H model — its all-stations 244,489-hour KFold out-of-fold R² of 0.8125 against the same model's own honest leave-one-station-out R² of 0.2093 — not by differencing two models evaluated on different station sets. The two LOSO numbers, 0.2093 (Config H, fifteen stations) and 0.1989 (definitive model, thirty-seven stations), agree closely and both realize the approximately 0.20 LOSO anchor, but only the former shares its model with the random-cross-validation figure and so is the one used to quantify the leakage cost.

*Table 5.3: Config H R² and MAE across evaluation regimes, from leaky random and fake-daily LOSO to honest retrain-per-fold LOSO, with the leakage gap.*

| Evaluation regime (Config H, 32 features) | R² | MAE | Note |
|---|---|---|---|
| Random 5-fold cross-validation (published number) | 0.8125 | 7.63 | station identity leaks |
| Fake daily LOSO, model alone (leaky) | 0.8684 | 3.77 | KFold-based, identity leaks |
| Fake daily LOSO, model + AOD + kriging (leaky, best) | 0.8882 | 3.32 | highest leaky figure |
| True daily LOSO, model alone (honest) | 0.168 | — | retrained per held-out station |
| Honest hourly LOSO (honest, primary metric) | 0.2093 | 11.99 | Config H, 15-station honest LOSO |
| **Leakage gap (random CV − honest hourly LOSO)** | **−0.6032** | — | the cost of station identity |

The table also records a cautionary intermediate result. An earlier daily-resolution evaluation reported an apparently excellent LOSO R² of 0.8684 for the model alone, rising to 0.8882 when satellite AOD and ordinary-kriging post-processing were added. These figures are not honest LOSO. Although labelled "LOSO" in the producing script, the hourly predictions feeding them came from a five-fold KFold trained on all stations and were then aggregated to daily values and kriged across stations per date, so station identity leaked through the hourly stage exactly as in Section 5.2. The honest daily LOSO, obtained by retraining a fresh model for each held-out station, is only 0.168 for the model alone and 0.224 with kriging — meaning the 0.8684 figure was inflated by roughly +0.66. This example is retained in the chapter as a concrete illustration of how easily a leaky pipeline can manufacture a publication-grade number, and as the reason every headline figure in this thesis is reported under explicit retrain-per-fold LOSO.

Turning to the global LOSO configurations proper, the role of tier grouping is decisive. The genuinely tier-blind no-grouping global model achieves an n-hours-weighted pooled R²_hourly of only 0.0387, a per-station mean of 0.0273, a per-station median of −0.0012, and just 48.6% of stations positive — that is, a single pooled model that does not know each station's pollution regime is essentially useless for spatial prediction, failing at more than half of all stations. Injecting the true tier label (the T4F configuration, identical to the tuned variant in definitive_v3) lifts the pooled R² to 0.2004, the mean to 0.1989, the median to 0.1787, and the fraction positive to 70.3%. This is the best non-oracle configuration under the thesis's own definition and the realization of the approximately 0.20 LOSO anchor committed in Chapter 1. An independent cross-check in a separate experiment file reproduces the contrast: an oracle-tier configuration there reaches a pooled 0.2074 against −0.0438 for its no-grouping counterpart, confirming the effect is not specific to one file.

*Table 5.4: Global LOSO configurations across experiment files — no-grouping, true-tier (T4F), and oracle base-margin — with pooled, mean, median, fraction-positive, and t3-mean R².*

| File | Config | Role | n | Pooled (n-hr wt) | Mean | Median | % > 0 | t3 mean |
|---|---|---|---|---|---|---|---|---|
| satellite_grouping | no_group | No-grouping global | 37 | 0.0387 | 0.0273 | −0.0012 | 48.6% | 0.2683 |
| satellite_grouping | t4f (= dart_tuned) | T4F true-tier (best non-oracle) | 37 | 0.2004 | 0.1989 | 0.1787 | 70.3% | 0.5682 |
| twophase_bm | oracle_bm | Oracle base-margin (true mean offset) | 40 | 0.2487 | 0.2464 | 0.2669 | 82.5% | 0.5011 |
| twophase_bm | global_bm | Global single base-margin | 40 | −0.0602 | −0.0733 | 0.0261 | 55.0% | 0.2402 |
| tier_operational | oracle_t4f | Oracle true-tier (cross-check) | 40 | 0.2074 | 0.1980 | 0.1547 | 75.0% | 0.5666 |
| tier_operational | no_t4f | No-grouping (cross-check) | 40 | −0.0438 | −0.0615 | 0.0142 | 52.5% | 0.2552 |

The six DART variants in definitive_v3 differ only marginally among themselves, all clustering near the headline 0.20: the ensemble is highest at a pooled 0.2061 and a per-station mean of 0.2048, the tuned (T4F) variant follows at 0.2004 pooled and 0.1989 mean, and the untuned base model trails at 0.1912 pooled and 0.1902 mean. The narrowness of this spread — under 0.02 in per-station mean across six training strategies — indicates that the LOSO ceiling for the global true-tier model is set by the information available, not by the choice of regularization or ensembling. Restricting attention to the broken-sensor-removed thirty-seven-station set rather than the forty-station set raises the medians slightly for the oracle configurations, but the ordering is invariant.

*Table 5.5: The six DART training variants in definitive_v3, all clustering near the 0.20 LOSO ceiling, sorted by pooled R².*

| Config | Pooled (n-hr wt) | Mean | Median | % > 0 | t3 mean (n=9) |
|---|---|---|---|---|---|
| dart_ensemble | 0.2061 | 0.2048 | 0.2030 | 73.0% | 0.5639 |
| dart_tuned (= T4F) | 0.2004 | 0.1989 | 0.1787 | 70.3% | 0.5682 |
| dart_nn23_pruned | 0.1996 | 0.1987 | 0.1881 | 70.3% | 0.5608 |
| dart_nn23 | 0.1971 | 0.1969 | 0.2074 | 73.0% | 0.5622 |
| dart_nn23_dow | 0.1932 | 0.1930 | 0.1929 | 73.0% | 0.5606 |
| dart_base | 0.1912 | 0.1902 | 0.1819 | 70.3% | 0.5548 |

The per-station breakdown of the best non-oracle configuration (T4F, the tuned variant of definitive_v3) reveals that the headline mean of 0.20 masks an enormous range and a clean stratification by pollution level. Skill rises monotonically with concentration: the highest-pollution stations are predicted well, with the best station — Hưng Yên on Nguyễn Văn Linh, mean PM2.5 of 43.2 µg/m³ — reaching an R² of 0.7553, while the clean low-variance stations are predicted poorly or worse than their own mean, the worst being Trà Vinh Dân Thành at a mean of 9.1 µg/m³ and an R² of −0.8139. The full thirty-seven-station table below is sorted by tier and then by R² within tier, making the stratification visible directly.

*Table 5.6: Per-station honest LOSO R² for the best non-oracle configuration (T4F), all thirty-seven stations sorted by tier and then by R² within tier.*

| Tier | Station | PM2.5 mean | R²_hourly |
|---|---|---|---|
| t0 | Trà Vinh xã Dân Thành, TX Duyên Hải | 9.13 | −0.8139 |
| t0 | Quảng Ninh Gần KCN Cái Lân | 7.62 | −0.1553 |
| t0 | Quảng Ninh Nhuệ Hổ – Đông Triều | 9.24 | −0.1185 |
| t0 | Quảng Ninh Km11 – Minh Thành | 9.49 | −0.0537 |
| t0 | Quảng Ninh NM tuyển than Nam Cầu Trắng – Hạ Long | 6.67 | −0.0389 |
| t0 | Quảng Ninh TT văn hóa thể thao Cẩm Phả – Cẩm Trung | 6.96 | −0.0232 |
| t0 | Quảng Ninh Phường Cẩm Thịnh – Cẩm Phả | 6.85 | 0.1787 |
| t1 | Bình Định Hoa Lư – TP. Quy Nhơn | 18.48 | −0.5356 |
| t1 | Vũng Tàu Huyền Trân Công Chúa – Phường 8 | 14.67 | −0.3118 |
| t1 | Quảng Ninh UBND TP Uông Bí | 10.64 | −0.1273 |
| t1 | Thái Bình xã Thái Thọ, huyện Thái Thụy | 15.48 | −0.0567 |
| t1 | Ninh Thuận Công viên – TP Phan Rang | 15.49 | 0.0572 |
| t1 | Đà Nẵng 41 đường Lê Duẩn | 13.19 | 0.0617 |
| t1 | Gia Lai KCN Trà Đa – Tp Pleiku | 10.92 | 0.0696 |
| t1 | Lâm Đồng Vườn hoa – TP Đà Lạt | 18.08 | 0.1197 |
| t1 | Tây Ninh Thị xã Trảng Bàng | 10.92 | 0.1490 |
| t1 | Bình Định huyện Tuy Phước | 12.06 | 0.1883 |
| t2 | Bắc Ninh UBND xã Cao Đức – Gia Bình | 25.40 | −0.0552 |
| t2 | Quảng Ngãi UBND P. Nguyễn Nghiêm | 27.41 | 0.0228 |
| t2 | Quảng Nam KDC Hồ Xuân Hương | 20.07 | 0.2646 |
| t2 | Đà Nẵng ĐH Sư phạm Đà Nẵng | 23.60 | 0.2843 |
| t2 | Bắc Ninh Khu liên cơ Thuận Thành | 27.95 | 0.2866 |
| t2 | Bình Dương 593 Đại lộ Bình Dương | 24.22 | 0.2966 |
| t2 | Long An UBND Tp Tân An | 21.17 | 0.3692 |
| t2 | Phú Thọ đường Hùng Vương – Việt Trì | 27.10 | 0.4924 |
| t2 | Bắc Ninh TT Quan trắc – Bắc Ninh | 23.44 | 0.5071 |
| t2 | HCM Lê Hữu Kiều – Quận 2 | 22.60 | 0.5753 |
| t2 | HCM 20 Lý Chính Thắng | 21.34 | 0.6121 |
| t3 | Bắc Ninh UBND xã Xuân Lâm – Thuận Thành | 52.19 | 0.0696 |
| t3 | Thái Nguyên SVĐ Gang thép | 55.22 | 0.4361 |
| t3 | Thái Bình Cầu Thái Bình | 37.19 | 0.5197 |
| t3 | Hà Nội 556 Nguyễn Văn Cừ | 48.54 | 0.6153 |
| t3 | Hà Nội Công viên Nhân Chính | 36.98 | 0.6311 |
| t3 | Hà Nội ĐHBK Giải Phóng | 46.78 | 0.6722 |
| t3 | Hà Nam Công viên Nam Cao – Phủ Lý | 39.20 | 0.7036 |
| t3 | Hải Dương UBND TP. Hải Dương | 37.63 | 0.7110 |
| t3 | Hưng Yên 437 Nguyễn Văn Linh | 43.19 | 0.7553 |

The per-station LOSO skill is mapped geographically in Figure 5.1, with the thirty-seven stations plotted on a Vietnam basemap and colour-coded by R².

![Figure 5.1: Per-station leave-one-station-out R² across the 37 monitoring stations, colour-coded by skill.](figures/fig_5_3_station_r2_map.png)

*Figure 5.1: Per-station leave-one-station-out R² across the 37 monitoring stations, colour-coded by skill.*

The spatial pattern can be read directly from the per-station table, because skill tracks pollution level and pollution level has a strong geographic signature. The high-skill stations cluster tightly in the Red River Delta of the north — Hưng Yên, Hà Nam, Hải Dương, the three Hà Nội sites, Thái Bình, Thái Nguyên, and Bắc Ninh all exceed an R² of 0.4 — forming a contiguous high-performance region around Hanoi. A second pocket of high skill is isolated and southern: the two Ho Chi Minh City stations reach R² values of 0.61 and 0.58, standing apart from the surrounding low-skill southern sites. The remainder of the map would render predominantly in the low and negative range: the Quảng Ninh coastal-industrial cluster in the northeast, the central coastal stations from Đà Nẵng south to Bình Định, and the clean Mekong Delta sites all fall near or below zero. The visual impression would therefore be a bright, contiguous northern delta, two bright but isolated southern urban points, and a dim periphery — a geography that, as Section 5.4 argues, is most parsimoniously read as a geography of pollution level, with which region is partly confounded in this network.

This LOSO collapse situates the thesis squarely within the international literature on spatial validation. Kawano et al. (2025), with roughly a thousand training stations across India, saw their R² fall from 0.85 under random cross-validation to 0.67 under spatial cross-validation — a comparatively gentle drop of 0.18, sustained by network density. Meyer et al. (2018) reported a fall from 0.90 to 0.24, and Ploton et al. (2020) a fall from 0.53 to 0.14 — collapses of 0.66 and 0.39 respectively, on the same order as the 0.60 drop documented here. Vietnam's gap (0.81 to 0.21) is far larger than India's because Vietnam's network is far sparser; the thesis result is thus consistent with the literature critique and extends it to a tropical, sparse-network setting where the spatial penalty is at its most severe.


## 5.4 Tier-Stratified Results

The per-station table of Section 5.3 already hints at the chapter's most important interpretive finding: skill is organized by pollution level. This section makes that organization quantitative by breaking the true-tier LOSO result down by tier. A clarification of source mirrors the previous section: the model-variant file definitive_v3 does not contain a configuration literally named "oracle T4F," but each of its six DART variants carries the true four-tier label and so is effectively a true-tier (oracle-tier) model; the per-tier breakdown below is reported for the base variant, which most closely matches the committed tier numbers and the +0.21 gain, with the best-overall ensemble variant given alongside it. The literally-named oracle-tier-versus-no-grouping comparison lives in the paired operational file and is used as the cross-check for the gain calculation.

The per-tier pattern is unambiguous and monotone. The highest-pollution tier t3, comprising nine stations with a mean PM2.5 of 44.1 µg/m³, achieves a mean R² of 0.5548 (median 0.6084). The moderate tier t2 — eleven stations, mean 24.0 µg/m³ — falls to a mean R² of 0.3154 (median 0.2953). The low tier t1 — ten stations, mean 14.0 µg/m³ — drops to a mean R² of −0.0302 (median 0.0438), essentially the no-skill line. And the clean tier t0 — seven stations, mean 8.0 µg/m³ — is meaningfully negative on the mean at −0.1604 (median −0.0534), reflecting that at the cleanest sites the model cannot beat the station's own mean. The committed shorthand of t3 ≈ 0.56, t2 ≈ 0.31, t1 ≈ 0, t0 ≈ 0 is therefore confirmed on the mean for t3 and t2 and directionally for t1 and t0, with the small correction that the t0 mean is appreciably below zero rather than at zero — its median, −0.053, is the more forgiving but still sub-zero summary.

Because each tier holds only a handful of stations, these per-tier means carry substantial sampling uncertainty and are reported with it. On the base variant the standard errors of the per-tier mean R² are approximately 0.13 (t0, n = 7), 0.065 (t1, n = 10), 0.066 (t2, n = 11), and 0.078 (t3, n = 9), corresponding to 95% confidence half-widths of roughly 0.15–0.18 per tier (for t3, n = 9, SD ≈ 0.234, CI half-width ≈ 0.15). Two of the three adjacent-tier gaps are comparable in size to this sampling uncertainty: the t0→t1 gap (about +0.13) is smaller than the combined uncertainty of the two means, and even the t2→t3 gap (about +0.24) is only modestly larger than it; the t1→t2 gap (about +0.35) is the one transition that clears the noise floor decisively. The monotone ordering is consistent across all six variants, so the gradient as a whole is not an artifact of sampling, but the individual tier-to-tier increments — and the exact tier means — should be read as point estimates surrounded by intervals of order ±0.15 rather than as sharply separated levels.

*Table 5.7: Tier-stratified honest LOSO results for the base DART variant — per-tier station count, mean PM2.5, R², RMSE, MAE, and bias.*

| Tier | n stations | Mean PM2.5 | Mean R²_hourly | Median R²_hourly | Mean RMSE | Mean MAE | Mean bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | 7 | 7.99 | −0.1604 | −0.0534 | 8.92 | 5.52 | −1.75 |
| t1 | 10 | 13.99 | −0.0302 | 0.0438 | 13.45 | 8.11 | −3.35 |
| t2 | 11 | 24.03 | 0.3154 | 0.2953 | 14.83 | 9.85 | −2.74 |
| t3 | 9 | 44.10 | 0.5548 | 0.6084 | 23.31 | 15.20 | −4.17 |
| ALL | 37 | — | 0.1902 | — | — | — | — |

The best-overall ensemble variant produces the same tier structure, marginally higher at each level: t3 at 0.5639, t2 at 0.3324, t1 at −0.0195, and t0 at −0.1371. The invariance of the pattern across training strategies confirms that the tier gradient is a property of the data, not of the model.

*Table 5.8: Tier-stratified honest LOSO results for the best-overall ensemble variant, showing the same monotone structure marginally higher at each tier.*

| Tier | n stations | Mean PM2.5 | Mean R²_hourly | Median R²_hourly | Mean RMSE | Mean MAE | Mean bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | 7 | 7.99 | −0.1371 | −0.0685 | 8.84 | 5.40 | −1.83 |
| t1 | 10 | 13.99 | −0.0195 | 0.0429 | 13.38 | 8.09 | −3.26 |
| t2 | 11 | 24.03 | 0.3324 | 0.3106 | 14.62 | 9.74 | −2.59 |
| t3 | 9 | 44.10 | 0.5639 | 0.6120 | 23.13 | 15.13 | −3.51 |
| ALL | 37 | — | 0.2048 | — | — | — | — |

This tier gradient is the single most important quantitative finding of the chapter, and it is the principal evidence for the claim that predictability scales with pollution level. The gradient itself is robust: it is monotone (t3 > t2 > t1 > t0) and model-invariant, reproduced across all six training variants, so that skill rising with concentration is on firm footing. The accompanying claim that level, rather than region, is the operative variable is on weaker footing and is offered here as a hypothesis rather than an established result, because level and region are confounded at the top of the network. The early outline proposed to demonstrate the level-over-region claim by showing that the top tier t3 contains both northern and southern high performers. In this dataset it does not: all nine t3 stations are northern, in and around the Red River Delta, because no southern or central station reaches the t3 threshold of 35 µg/m³ — the south is simply not polluted enough to populate the top tier. The only place where the two regions coexist at comparable concentration is one tier down, at t2. There the two best-predicted stations are the southern Ho Chi Minh City sites, at R² values of 0.59 and 0.56 on the base variant, outscoring most of the northern t2 stations, and the southern t2 mean R² (0.459, n = 4) exceeds the northern t2 mean (0.278, n = 4). This difference is, however, not statistically significant: a Welch two-sample test gives p = 0.30, a Mann–Whitney U test p = 0.34, and the 95% confidence interval on the South-minus-North mean difference spans roughly [−0.13, +0.49] — an interval that comfortably includes zero. With only four stations per region at t2, the comparison simply lacks the power to separate a level effect from a regional effect, and no significant test is possible at this sample size. When region-level means are computed across all tiers they appear to favour the north (north 0.266, south 0.134, central 0.057), but that apparent regional gradient is largely an artifact of tier mix — the north holds all the t3 stations — and shrinks once tier is held fixed. The honest reading is therefore that level and region are confounded in this network: the level gradient is strong and model-invariant, but the available data cannot demonstrate that level rather than region is the governing variable, and the t2 evidence is suggestive rather than conclusive.

*Table 5.9: The eleven t2 stations by region, PM2.5, and R², showing the North–South comparison at matched pollution level.*

| Station | Region | PM2.5 | R²_hourly |
|---|---|---:|---:|
| HCM Lý Chính Thắng | South | 21.34 | 0.5947 |
| HCM Lê Hữu Kiều Q2 | South | 22.60 | 0.5610 |
| Bắc Ninh Suối Hoa | North | 23.44 | 0.5042 |
| Phú Thọ Việt Trì | North | 27.10 | 0.4578 |
| Long An Tân An | South | 21.17 | 0.3844 |
| Bình Dương Hiệp Thành | South | 24.22 | 0.2953 |
| Quảng Nam | Central | 20.07 | 0.2582 |
| Bắc Ninh Thuận Thành | North | 27.95 | 0.2551 |
| Đà Nẵng ĐHSP | Central | 23.60 | 0.2432 |
| Quảng Ngãi | Central | 27.41 | 0.0221 |
| Bắc Ninh Cao Đức Gia Bình | North | 25.40 | −0.1062 |

Finally, the tier-stratification technique is itself the single largest performance lever discovered in this thesis. The most defensible measure of the gain differences the true-tier model against the genuinely tier-blind no-grouping baseline within a single file, so that both share an identical station set: there the T4F per-station mean of 0.1989 exceeds the same-file no-grouping mean of 0.0273 by approximately +0.17, taking the model from essentially no useful spatial skill to the headline 0.20. Cross-file framings that pair the base variant (mean 0.1902) against a no-grouping baseline drawn from a different file (mean −0.0286) report a larger +0.219, and the literally-named oracle-tier configuration gives up to +0.260 on the full forty-station set; but these mix station sets and should be read as the upper end of the range rather than the headline figure. Across all framings the gain spans approximately +0.17 (within-file, against the same-file no-grouping baseline of 0.027) to +0.26, most defensibly about +0.17 — and on every framing it is larger than any feature group, any hyperparameter choice, or any ensembling strategy explored, confirming the Chapter 3 claim that T4F is the most impactful technique in the study. Its cost, the circular dependency of needing to know the target to assign the tier, is the subject of Section 5.5.

*Table 5.10: The tier-stratification gain across variants and station sets, each differencing a true-tier mean against a no-grouping baseline.*

| Comparison | T4F mean | No-grouping mean | Gain |
|---|---:|---:|---:|
| dart_base (37) vs no_t4f (37) | 0.1902 | −0.0286 | +0.219 |
| dart_ensemble (37) vs no_t4f (37) | 0.2048 | −0.0286 | +0.233 |
| oracle_t4f (37) vs no_t4f (37) | 0.2067 | −0.0286 | +0.235 |
| oracle_t4f (40) vs no_t4f (40) | 0.1980 | −0.0615 | +0.260 |


## 5.5 Deployable Model Comparison

The +0.21 gain of Section 5.4 is purchased with oracle information: assigning a station to its true tier requires knowing the station's mean PM2.5, which is the quantity the model is meant to predict. A genuinely deployable model — one usable at a location with no ground history — may not use the held-out station's true tier or true mean. This section compares every deployable configuration developed in the thesis against the oracle ceiling and measures the gap between them, which is the information cost of not knowing a station's pollution level.

Two premises in the early outline must be corrected before the comparison can be read honestly, both turning on what counts as deployable. First, the configuration the outline labels the best deployable method — a soft-gate model augmented with a real per-station base margin — is in fact an oracle: inspection of its producing code shows that it adds the held-out station's own true monthly PM2.5 mean as the base margin, which is the very leakage a deployable model must avoid. Its strong numbers (per-station mean 0.270, 92.5% positive) are therefore the oracle ceiling, not a deployable result, and they are reported below as such. Second, the outline calls the definitive_v3 base variant the "oracle base-margin" ceiling; in fact that configuration runs in an explicit no-oracle mode with a single global base margin and uses only the true held-out tier (T4F), making it an oracle-tier configuration, not an oracle per-station-mean configuration. Its numbers (mean 0.190, 70.3% positive) sit well below the true per-station-mean ceiling, exactly as one would expect of the milder tier-level oracle. With these corrections the picture is coherent.

The genuinely deployable configurations — those using only observable satellite, meteorological, and land-use features, with at most a global base margin — converge in a tight band just above zero. Their per-station medians range from about +0.01 to +0.04 and their per-station means hover near zero, between −0.06 and +0.04, with only roughly 47–57% of stations positive. The committed "+0.04 median" is the optimistic end of this band and is met by the better variants: the invariant balanced-loss model reaches a median of 0.0415 and the direct-LOSO satellite model with spatial neighbours reaches 0.0428, while the cluster-base-margin and no-grouping variants sit lower at +0.01 to +0.02. The soft-gate mixture of experts reaches a per-station mean of 0.0447 — the figure cited in Chapter 3 — but its median is in fact slightly negative at −0.0187, so its contribution to the "+0.04 convergence" is through its mean rather than its median. The honest statement is therefore that three independent deployable approaches converge in a near-zero positive band whose upper edge is approximately +0.04, not that they all land precisely on a single value.

The oracle ceiling, by contrast, is far higher. Injecting the held-out station's true per-station mean as an additive base margin lifts the per-station mean R² to between +0.25 and +0.31 — +0.306 on the broken-sensor-removed thirty-seven-station set, +0.254 to +0.271 on the forty-station sets — with 85% to 92% of stations positive. This confirms the committed oracle ceiling of approximately +0.27 per-station mean with roughly 92% positive. The mislabelled soft-gate-plus-real-mean configuration, now correctly classified as oracle, independently reproduces the ceiling at exactly +0.270 mean and 92.5% positive.

*Table 5.11: Deployable configurations versus oracle ceilings — mean, median, fraction-positive, and t3-mean R², with each configuration flagged as deployable or oracle.*

| Config | Source | Deployable | Mean R² | Median R² | % > 0 | t3 mean | n |
|---|---|---|---:|---:|---:|---:|---:|
| soft_gate_moe | soft_gate_moe | yes | +0.0447 | −0.0187 | 47.5% | +0.4129 | 40 |
| no_t4f | irm_invariant | yes | −0.0615 | +0.0142 | 52.5% | +0.2552 | 40 |
| balanced | irm_invariant | yes | −0.0508 | +0.0415 | 57.5% | +0.2445 | 40 |
| direct | satellite_v5b | yes | −0.0178 | +0.0428 | 55.0% | +0.3094 | 40 |
| clust_bm_rfsi | satellite_v5g | yes | +0.0292 | +0.0189 | 54.1% | +0.4249 | 37 |
| clust_bm | satellite_v5g | yes | −0.0067 | +0.0117 | 51.4% | +0.2413 | 37 |
| dart_base (T4F) | definitive_v3 | no (oracle tier) | +0.1902 | +0.1819 | 70.3% | +0.5548 | 37 |
| oracle_real_bm | irm_invariant | no (oracle mean) | +0.2429 | +0.1850 | 85.0% | +0.6031 | 40 |
| sgm_real_bm | irm_invariant | no (oracle mean) | +0.2701 | +0.2377 | 92.5% | +0.6038 | 40 |
| oracle_bm | satellite_v5g | no (oracle mean, ceiling) | +0.3063 | +0.2910 | 91.9% | +0.5694 | 37 |

The gap between the deployable band and the oracle ceiling is the information cost of not knowing the station mean, and it is large. Measured on matched folds within a single file — the only methodologically clean way to difference the two, since the configurations share a station set there — the oracle base margin exceeds the best deployable cluster-base-margin model by +0.277 in per-station mean R², +0.272 in median, and +37.8 percentage points in the fraction of positive stations. The same comparison in the invariant file, differencing the oracle real-mean configuration against the deployable no-grouping baseline, gives +0.332 in mean, +0.224 in median, and +40.0 percentage points positive. In round terms, knowing each station's true mean is worth roughly +0.28 to +0.33 in per-station mean R² and roughly forty additional percentage points of usable stations.

*Table 5.12: The information cost of not knowing the station mean — oracle-minus-deployable gaps in mean R², median R², and fraction positive, measured on matched station sets.*

| Gap (oracle − deployable) | Matched set | Mean-R² gap | Median-R² gap | % > 0 gap |
|---|---|---:|---:|---:|
| oracle_bm vs clust_bm_rfsi | satellite_v5g, 37 stns | +0.2771 | +0.2721 | +37.8 pts |
| sgm_real_bm vs no_t4f | irm_invariant, 40 stns | +0.3316 | +0.2235 | +40.0 pts |

The full three-number arc of the thesis is now in view. The model achieves R² ≈ 0.81 under random cross-validation, R² ≈ 0.20 under honest true-tier LOSO, and a deployable per-station median R² ≈ 0.04 once the oracle tier is withdrawn. The first step, from 0.81 to 0.20, is the cost of station-identity leakage (Section 5.3); the second, from 0.20 down to 0.04, is the cost of the circular dependency — the gap between knowing and not knowing each station's pollution regime. The deployable-to-oracle gap of roughly +0.27 per-station mean is the precise price of that missing information, and the convergence of three independent deployable approaches at the bottom of it is strong evidence that approximately +0.04 median is the genuine deployable ceiling from satellite observables alone, absent local calibration data. It is worth noting that even the deployable models perform far better at the polluted tier t3 (t3 means of +0.24 to +0.42) than overall, mirroring the tier gradient of Section 5.4; the near-zero overall medians are dragged down by the clean low-tier stations, not by uniform failure everywhere.

That +0.04 ceiling, however, is specifically the ceiling for recovering the station baseline *from satellite observables*. It is not the ceiling for a model that is allowed to interpolate the baseline from the surrounding monitoring network. The spatial-prior routing pipeline of Section 3.6.4 does exactly that, and it changes the picture materially. Rather than estimating the tier from observables — the strategy that produced the +0.04 band and that fails for the seven families of Section 5.7 — it estimates each held-out station's baseline level as a distance-weighted mean of the *observed* means of the surrounding training stations (the station itself always excluded), then shifts and routes target-free prediction streams toward that estimate. Evaluated under the same leave-one-station-out protocol on the forty stations, with GHAP removed so the result depends on no climatological PM2.5 product, this deployable pipeline reaches a per-station mean R² of 0.197 and a median of 0.117 — within 0.01 and 0.03 of the oracle T4F ceiling of 0.203 and 0.147 — with a pooled R² of 0.569 against the oracle's 0.572, thirty of forty stations positive, and, crucially, zero high-to-non-high and zero dangerous low-to-high class flips, marginally safer than the oracle itself. In short, a fully deployable model recovers nearly the entire oracle tier gain that the satellite-observable proxies could not.

*Table 5.13: The spatial-prior routing model against the oracle ceiling and the satellite-observable deployable band, all under no-GHAP leave-one-station-out. The deployable spatial-baseline estimator closes almost the whole oracle gap that observable tier-estimation could not.*

| Config | Deployable | Pooled R² | Mean stn R² | Median stn R² | High→non-high | Danger low→high |
|---|---|---:|---:|---:|---:|---:|
| oracle_t4f (true tier, ceiling) | no | 0.572 | 0.203 | 0.147 | 1 | 0 |
| per-tier MoE experts (literal idea) | yes | 0.434 | 0.083 | 0.081 | 6 | 1 |
| satellite-observable band (§3.6.1–3.6.3) | yes | — | ≈ +0.00 to +0.04 | ≈ +0.01 to +0.04 | — | — |
| **spatial-prior routing (§3.6.4, deployed)** | **yes** | **0.569** | **0.197** | **0.117** | **0** | **0** |

The mechanism of the difference is worth stating precisely, because it is easy to misread as a contradiction of the +0.04 ceiling and is not. The deployable configurations of the band above already use spatial-neighbour features (RFSI): the difference is not that the new model has access to nearby stations and the others do not. The difference is *what the neighbour information is used for*. The earlier configurations used nearby stations as hourly co-variation features, which fix the temporal *shape* of the series but leave the absolute *baseline level* unanchored — the canonical failure in which a held-out station's predicted series has the right pattern but the wrong mean. The spatial prior uses nearby stations to anchor that baseline level directly, as a deployable stand-in for the oracle base margin, supplying the one quantity the satellite observables could not. The earlier two-phase design (Section 3.6.2) attempted the same anchoring but estimated the mean from satellite features (leave-one-out R² ≈ 0.59) and injected it as a hard base margin, which overshot at clean sites; the spatial estimate is both more accurate and used more conservatively, which is why it succeeds where the two-phase design failed. The essential caveat follows from the mechanism: because the estimate is a spatial interpolation, its skill is conditional on the proximity of the interpolating monitors, and it degrades as the target moves away from the network — which is the density constraint of Chapter 4 restated, and which is tested directly on unseen stations in Section 5.6.


## 5.6 External Validation (LCS Network)

The validation frameworks of Sections 5.2 through 5.5 are all internal, using the thirty-seven regulatory stations both to train and, under LOSO, to test. The most stringent test is external: predicting at genuinely independent sites never involved in model development. This section reports two such tests of the regional Red River Delta model — leave-one-station-out within the twelve dense delta stations, and external validation against thirty-nine low-cost sensors and the US Embassy reference monitor.

The delta LOSO confirms two things established at the national scale: spatial interpolation helps, and an oracle baseline leaves clear headroom. Among the three configurations, the spatial-interpolation variant (delta_rfsi) clearly outperforms the plain base-margin model (delta_bm), lifting the mean R² from 0.2021 to 0.3022 and the median from 0.2944 to 0.4330, with both reaching 75.0% of stations positive. The oracle base-margin configuration sets the ceiling at a mean of 0.4109, a median of 0.4728, and 91.7% of stations positive — confirming that even within this dense, favourable subnetwork, knowing the true station mean still adds skill and the deployable models have not exhausted the achievable performance. The delta_rfsi mean and median (0.302 and 0.433) are the regional figures committed in Chapter 3, and they substantially exceed the equivalent nationwide configuration's mean of 0.255 on the same stations, vindicating the regional-model strategy for dense clusters.

*Table 5.14: Delta-model LOSO across the twelve dense Red River Delta stations — base-margin, spatial-interpolation, and oracle configurations.*

| Config | Mean R² | Median R² | % positive | Mean RMSE | Mean MAE |
|---|---:|---:|---:|---:|---:|
| delta_bm | 0.2021 | 0.2944 | 75.0% | 25.93 | 17.05 |
| delta_rfsi | 0.3022 | 0.4330 | 75.0% | 23.34 | 15.25 |
| oracle_bm (ceiling) | 0.4109 | 0.4728 | 91.7% | 21.30 | 13.92 |

The external low-cost-sensor validation is the most independent test available, but it must be characterized precisely, because it is not genuine spatial prediction at an unseen site. The held-out low-cost sensors supply only the target PM2.5; the satellite, AOD, TROPOMI, and ERA5 meteorology features fed to the model are taken from the nearest regulatory (KK) station's hourly record rather than measured at the sensor itself, and the RFSI spatial anchors are correctly localized to the sensor's coordinates. The test is therefore a nearest-station feature-transfer plus sensor-agreement check: it asks how well the model predicts an independent sensor's concentrations when its meteorological and satellite drivers are borrowed from the closest regulatory station, and how well the low-cost instrument agrees with that prediction. This framing matters for interpreting both the headline skill and its distance dependence. The validation file contains forty held-out sites — thirty-nine low-cost sensors and one US Embassy reference monitor — and the headline figures committed in Chapter 1 are the low-cost-sensor subset: across the thirty-nine sensors the median R² is 0.5293, the mean R² is 0.0310, and 84.6% of sites are positive. (Including the Embassy shifts these slightly to a median of 0.5416, a mean of 0.0473, and 85.0% positive; the committed numbers are the sensor-only values.) These figures are accordingly partly a feature-transfer result rather than a measure of standalone spatial skill. A second qualification applies to the LCS sample specifically: the low-cost records cover only the October–April window (each sensor contributes roughly 1,400 hours within the dry season), so the R² = 0.53 median is a dry-season figure and the model's year-round skill at these sites is untested. The wide gap between the strong median of 0.53 and the weak mean of 0.03 is the now-familiar signature of a few far, degraded sites: the mean is dragged down almost entirely by a single outlier 43 km from the nearest regulatory station with an R² of −14.75, while the median, robust to that outlier, faithfully reports that the typical low-cost site is well predicted.

The US Embassy in Hanoi, the highest-quality external point — a reference-grade BAM-1022 beta-attenuation monitor 3.0 km from the nearest training station — validates higher and is reported separately: R² = 0.6842, RMSE = 17.51 µg/m³, over 9,729 hours. This is the single most credible external point in the thesis, because the Embassy instrument is regulatory-grade rather than a laser-scattering low-cost sensor; subject to the same feature-transfer caveat as the LCS sites — its meteorological and satellite drivers are likewise borrowed from the nearest regulatory station — it shows that within dense network coverage the model's predictions agree closely with an independent reference monitor. Two cautions temper a direct comparison with the LCS median. First, the Embassy record (9,729 hours) spans multiple seasons, whereas the low-cost records are confined to the dry season (~1,400 hours each); the two are therefore not on the same seasonal footing, and the Embassy R² should not be read as a like-for-like upgrade of the dry-season LCS figure. Second, the longer record does not itself explain the higher R²: record length affects the precision of the estimate, not its expected value, and the Embassy's advantage is more plausibly its short anchor distance (3.0 km) and reference-grade instrumentation than its hour count. The Embassy result is thus best read as a high-quality, close-range, multi-season agreement check rather than as proof of standalone spatial skill.

*Table 5.15: External-validation summary — low-cost-sensor subset, all sites including the Embassy, and the US Embassy reference monitor alone.*

| Subset | n | Median R² | Mean R² | % positive |
|---|---:|---:|---:|---:|
| LCS sites only | 39 | 0.5293 | 0.0310 | 84.6% |
| All sites (incl. Embassy) | 40 | 0.5416 | 0.0473 | 85.0% |
| US Embassy Hanoi (single) | 1 | 0.6842 | — | RMSE 17.51, n = 9,729 h |

The site-level detail makes the density dependence concrete. The best low-cost sites are predicted nearly as well as the Embassy, with R² values from 0.65 to 0.76, and they are almost all close to a regulatory anchor — within a few kilometres in central Hanoi or in adjacent delta towns. The worst sites are predicted disastrously, with R² values from −0.16 to −14.75, and they are predominantly far from any anchor, in Hải Phòng and outlying Hưng Yên locations 15 to 43 km from the nearest regulatory station.

*Table 5.16: The ten best-predicted low-cost sites by R², with distance to the nearest regulatory station, record length, and RMSE.*

| Rank | R² | Dist. KK (km) | n hours | RMSE | Station |
|---|---:|---:|---:|---:|---|
| 1 | 0.7638 | 15.5 | 1082 | 20.26 | Ninh Bình trạm bơm Hoành Uyển, P. Hà Nam |
| 2 | 0.7468 | 3.7 | 1376 | 18.52 | Hà Nội UBND phường Định Công |
| 3 | 0.7463 | 2.7 | 1094 | 17.52 | Ninh Bình Đảng ủy – HĐND phường Hà Nam |
| 4 | 0.7363 | 1.1 | 1345 | 23.43 | Hà Nội Trường tiểu học Minh Khai, P. Thanh Nhàn |
| 5 | 0.6986 | 2.7 | 1482 | 16.72 | Hà Nội 83 Nguyễn Chí Thanh |
| 6 | 0.6865 | 3.0 | 1504 | 20.86 | Hà Nội Bộ TNMT, 10 Tôn Thất Thuyết |
| 7 | 0.6599 | 3.5 | 1507 | 23.08 | Hà Nội UBND phường Quan Hoa (cũ) |
| 8 | 0.6518 | 2.0 | 1346 | 19.94 | Hà Nội Trường tiểu học Thịnh Hào, P. Ô Chợ Dừa |
| 9 | 0.6488 | 17.1 | 1507 | 20.92 | Hà Nội UBND xã Chuyên Mỹ |
| 10 | 0.6467 | 9.0 | 1366 | 31.61 | Hà Nội Trường mầm non B, Ngọc Hồi |

*Table 5.17: The worst-predicted low-cost sites, all far from any regulatory anchor, with distance, record length, and RMSE.*

| R² | Dist. KK (km) | n hours | RMSE | Station |
|---:|---:|---:|---:|---|
| −14.7549 | 43.0 | 1443 | 33.89 | Hải Phòng cột điện P. Bạch Đằng |
| −0.5981 | 15.2 | 1428 | 53.56 | Hưng Yên thôn Lương xã Thượng Hồng |
| −0.3160 | 18.4 | 1462 | 34.92 | Hưng Yên xã Phạm Ngũ Lão |
| −0.2177 | 4.9 | 1465 | 26.50 | Hà Nội 18 Hoàng Quốc Việt |
| −0.1596 | 21.8 | 1508 | 20.05 | Hải Phòng UB phường Phạm Sư Mạnh |

The distance–skill relationship is statistically clear, and it is most robustly summarized by the rank correlation, which is insensitive to the extreme negative outlier. Across the thirty-nine low-cost sites the Spearman rank correlation between per-site R² and distance to the nearest regulatory station is ρ = −0.62 (p < 0.001). The Pearson correlation is −0.43 (p = 0.007) with all sites included, but it is sensitive to the single −14.75 outlier: excluding that one site weakens it to −0.34 (p = 0.034), so the linear association, while still significant, is partly carried by that extreme point, whereas the rank correlation is not. The median R² falls from 0.6396 for the sixteen sites within 10 km to 0.4487 for the twenty-three sites beyond 10 km. This is the distance-decay constraint of Section 4.5 expressed at site level: skill degrades monotonically with isolation from the nearest anchor. Under the feature-transfer framing above, part of this decay is mechanical — the further a low-cost site lies from its nearest regulatory station, the larger the error in substituting that station's meteorology and satellite features for the site's own conditions — so the distance-decay curve reflects growing feature-substitution error as well as genuine loss of spatial-interpolation skill.

*Table 5.18: Distance–skill statistics across the thirty-nine low-cost sites — Pearson and Spearman correlations and near-versus-far median R².*

| Metric | Value |
|---|---|
| Pearson r (distance, R²) | −0.4255 |
| Spearman ρ (distance, R²) | −0.6188 |
| Median R², distance ≤ 10 km (n = 16) | 0.6396 |
| Median R², distance > 10 km (n = 23) | 0.4487 |
| Distance range | 1.1 – 44.4 km |

Read against the international literature, these external results are competitive precisely where the network is dense, with the caveat that the Vietnam figures are a nearest-station feature-transfer test rather than the fully independent spatial cross-validation reported in those studies, so the comparison is indicative rather than strictly like-for-like. The low-cost-sensor median of 0.529 is moderately below Kawano et al.'s (2025) Indian spatial-cross-validation R² of 0.67 — unsurprising, given that their result rests on roughly a thousand training stations against Vietnam's thirty-seven — but the gap narrows at close range: the median R² of sites within 10 km (0.64) and the US Embassy reference site (0.684) are on par with or exceed the Indian benchmark, and the ten best low-cost sites span 0.647 to 0.764. The honest framing is therefore that the model's transfer-based predictions approach the best published spatial-cross-validation results within dense coverage and fall below them as the network thins — which is the density-constraint story of Chapter 4 restated as an external benchmark comparison. Two notes on comparability are warranted. First, the low-cost records are short (most around 1,000–1,500 hours), dry-season only, and the sensors are laser-scattering rather than reference-grade, so individual low-cost R² values carry more sampling noise than the regulatory and Embassy figures, which partly explains the wide spread including the negative tail. Second, because the satellite and meteorological features are transferred from the nearest regulatory station, some of the close-range parity reflects the small feature-substitution error at short distances rather than standalone interpolation skill.

A second, more stringent external test evaluates the spatial-prior routing pipeline of Section 3.6.4 — the deployable national model that resolves the circular dependency — directly on unseen stations. The pipeline is trained on the forty thesis stations and used to predict at forty-six independent low-cost sensors that played no part in model development; unlike the delta transfer test above, each held-out sensor carries its own satellite, AOD, and meteorological features computed at its own coordinates, and the spatial baseline prior is anchored to the forty training stations with the held-out sensor excluded. This is therefore a genuine train-on-forty, predict-unseen stress test of the full chain rather than a feature-transfer check. The pure pipeline reaches a pooled R² of 0.332, a per-station mean R² of 0.198 and median of 0.216, with thirty-five of forty-six sensors positive — a marked recovery over the raw high-amplitude stream alone (pooled R² 0.127) and a result that, notably, reproduces the internal per-station mean of about 0.20 on stations never seen in training. The decisive map-safety property is that it produces **zero dangerous low-to-high misclassifications**: the model does not map any clean sensor as polluted. Its residual errors are almost entirely moderate/high boundary confusions in the noisy low-cost data — ten true-high sensors predicted moderate and eight true-moderate predicted high — rather than gross regime errors. Applying the single reliability guard of Section 3.6.4 as a map-display layer leaves the numeric predictions essentially unchanged on the internal forty stations and, on the external sensors, flags six hidden-high warnings and one clear false-high; counting flagged hidden-high sites as not-missed reduces the effective high-to-non-high misses from ten to five while still introducing no low-to-high danger. The external numeric figures should be read as a stress test rather than a repeat of the internal score — the low-cost sensors are noisier, dry-season-only, and include local microenvironments absent from the forty-station training set — but the qualitative conclusion is robust: the deployable pipeline generalizes to genuinely unseen stations as an elevated-pollution screening map, with the conservative bias appropriate to a public-health product.

*Table 5.19: The spatial-prior routing pipeline on forty-six unseen low-cost sensors (train-on-forty, predict-unseen). Numeric metrics are a noisy-domain stress test; the operative result is zero dangerous low-to-high misclassification, with the reliability guard cutting effective high misses by half.*

| Variant | Pooled R² | Mean stn R² | Median stn R² | % positive | High→non-high | Danger low→high |
|---|---:|---:|---:|---:|---:|---:|
| raw high-amplitude stream only | 0.127 | 0.005 | 0.083 | — | 19 | 0 |
| spatial-prior routing (pure) | 0.332 | 0.198 | 0.216 | 35/46 | 10 | 0 |
| + reliability guard (numeric) | 0.335 | 0.205 | 0.264 | 35/46 | 10 | 0 |
| + reliability guard (effective map) | 0.335 | 0.205 | 0.264 | 35/46 | 5 | 0 |


## 5.7 Summary of Unsuccessful Experiments

The deployable ceiling of approximately +0.04 median was not accepted before it was tested. Seven distinct families of methods were developed specifically to break the circular dependency — to recover some of the +0.21 tier gain without using the held-out station's true tier or mean — and all seven failed. Documenting them is not an afterthought: the negative results bound the solution space and, taken together, constitute the empirical case that no satellite-observable proxy reliably resolves the dependency, which is the third contribution committed in Chapter 1. Each family is summarized below, with the best result it achieved and the reason it failed; the reference point throughout is the true-tier (hard-T4F) per-station mean of 0.198, which every deployable variant of these methods fails to reach.

The soft-T4F family replaced the hard tier boundaries with Gaussian-weighted soft boundaries, on the hypothesis that smooth tier membership would generalize better than discrete assignment. It did not: the best soft configuration reached a per-station mean of only 0.072, and performance declined monotonically as the boundary was softened (0.072, then 0.037, then 0.021, then 0.002), confirming that hard boundaries always win — the discrete tier carries information that smoothing destroys. The GHAP-hybrid family multiplied the GHAP satellite climatology against the spatial-interpolation features as an interaction term; it reached 0.166, the lowest of its four-configuration file and below the 0.196 base, because GHAP competes with rather than complements the RFSI signal under LOSO. The IRM invariant-feature family removed regime-dependent ("unstable") features in the hope of isolating a transferable invariant predictor; every removal variant went negative (the loosest removal worst at −0.169), demonstrating that the supposedly unstable features carry real, irreplaceable signal. The anomaly-prediction family reframed the target as a deviation from a rolling baseline rather than an absolute concentration; the best deviation configuration reached only 0.065, and the one-week-baseline variant was catastrophic at −0.78, because a noisy baseline makes the deviation harder to predict than the absolute level — the daily-anomaly target barely differed from its absolute counterpart (0.145 versus 0.142).

Three remaining families warrant a correction to the numbers proposed in the early outline, because the outline's figures for them are oracle ceilings or single-station values rather than the methods' own deployable scores. The tier-ensemble family trained all four tier models and averaged their predictions; the outline's "0.246" is not this method's score but the oracle ceiling bundled in the same experiment file, whereas every actual averaged-tier configuration is negative in mean (best, simple averaging, at −0.068), because the three wrong-tier models inject more noise than the one correct-tier model removes. The two-phase family predicted each station's tier from observables and then used the predicted tier's model; the outline's "0.249" is again the oracle base-margin ceiling in that file, not the deployable score — the actual predict-then-use configuration collapses to −0.063, because the predicted tier is correct too rarely (the gate reaches only about 40% tier accuracy) and misrouting destroys the gain. The KNN/emission-cluster family assigned tiers or clusters from observable features by nearest-neighbour and clustering schemes; the outline's "0.274–0.544" range corresponds to no configuration-level mean — those are scattered single-station R² values — and every cluster or KNN configuration mean is in fact negative (best, four-cluster emission grouping, at −0.030), with the broader emission-feature family of configurations all falling below the hard-T4F reference of 0.198. The defensible claim for this family is therefore the qualitative one stated in the outline: observable-based cluster and KNN routing is still worse than hard T4F.

*Table 5.20: The seven families of unsuccessful experiments, each with its best per-station-mean R² and the reason it failed to reach the hard-T4F reference of 0.198.*

| Experiment | Approach | Best R² (per-station mean) | Why it failed |
|---|---|---|---|
| Soft T4F | Gaussian-weighted tier boundaries | 0.072 | Hard boundaries always win; skill declines monotonically as the boundary softens |
| GHAP hybrid | GHAP × RFSI interaction | 0.166 (below base 0.196) | GHAP competes with RFSI under LOSO; ranks last of its configurations |
| Tier ensemble | Train all four tiers, average | −0.068 (oracle ceiling 0.198) | Wrong-tier models add more noise than the right one removes |
| Two-phase tier | Predict tier, then use tier model | −0.063 (oracle ceiling 0.249) | Predicted tier correct only ~40% of the time; misrouting destroys the gain |
| KNN / emission clusters | Observable-based tier/cluster assignment | −0.030 (all configs negative) | Cluster/KNN routing still worse than hard T4F (0.198) |
| IRM invariant features | Remove regime-dependent features | −0.051 (all configs negative) | "Unstable" features carry real signal; removal goes negative |
| Anomaly prediction | Predict deviation from baseline | 0.065 | Noisy baselines make deviations harder than absolutes; one-week baseline catastrophic (−0.78) |

The lesson of these seven failures is consistent and, in its way, more informative than a success would have been. The tier label encodes the station's mean pollution level, and that mean is exactly the quantity that cannot be observed at an unmonitored location without ground measurement. Every attempt to recover it from observables — by softening it, by clustering toward it, by predicting it, by removing dependence on it, or by sidestepping it through anomalies — either reproduces only a fraction of the signal or destroys it outright. The experiments therefore do not merely fail; they triangulate the boundary of what is achievable, and they establish that the deployable +0.04 median is a genuine information ceiling *for satellite-observable estimation of the baseline*, not a deficiency of effort. The circular dependency is real and binding for that class of method: no satellite-observable proxy tested in this thesis recovers the missing mean.

The boundary these seven failures triangulate is, however, specifically the boundary of *observable* estimation, and locating it precisely points to where the missing information actually lives. The mean pollution level cannot be read off satellite or land-use observables — but it can be interpolated from the surrounding monitoring network, because a station's baseline is strongly predicted by the observed baselines of its neighbours. This is the lever the seven families never pulled: each of them, including the two-phase predictor and the clustering schemes, sought the tier in observable feature space, whereas the resolution lies in physical space. The spatial-prior routing pipeline of Sections 3.6.4 and 5.5 does pull it, recovering a per-station mean of 0.197 against the oracle's 0.203 by anchoring the baseline to neighbouring monitors rather than guessing it from satellites. This does not weaken the seven negative results — it sharpens their interpretation. The circular dependency is unbreakable from observables alone, and breakable by spatial interpolation from the network; the cost of the resolution is that it inherits the network's reach, working where the target has usable neighbours and decaying where it does not. The deployable ceiling is thus not a single number but a function of station proximity, and the binding constraint is, once again, the density of the ground network rather than the algorithm.


## 5.8 A Unifying View: Interpolation versus Extrapolation

The preceding sections evaluated three validation frameworks that have so far been treated as separate. Read together, they resolve into a single organizing principle that explains why the headline numbers vary so widely — from above 0.8 to below 0.1 — for what is nominally one model on one dataset. Every prediction the model can be asked to make is either an *interpolation* between known observations or an *extrapolation* to a genuinely unobserved point, and the achievable skill is governed almost entirely by which of the two a given task requires. The high numbers in the literature, and in this thesis, are interpolation; the operational goal — a pollution map for unmonitored locations — is extrapolation; and conflating the two is precisely the error that the validation critique of Chapter 2 identified.

It is useful first to separate two tasks that the term "temporal prediction" silently merges, because they sit on opposite sides of the interpolation–extrapolation divide. The first is *temporal gap-filling*: imputing missing hours at a station that has a measurement history, the operational case of sensor downtime or maintenance gaps. The second is *temporal forecasting*: predicting hours that lie in the future, beyond the end of the training record. To measure each cleanly, the model was retrained per station using only exogenous features (meteorology, satellite observations, and temporal encodings — no PM history), once under a random five-fold split over the station's own hours (gap-filling) and once under a chronological split that trains on the first 70% of the record and predicts the last 30% (forecasting). The contrast is decisive and is reported in Table 5.19.

*Table 5.21: The same per-station exogenous-feature model evaluated as four different tasks, by pollution tier. Gap-filling and forecasting use the station's own data in training; the spatial map (LOSO) does not. Forecasting is decomposed into raw R² and shape skill (squared Pearson correlation of the predicted and observed future series), which separates pattern tracking from baseline drift.*

| Tier | Temporal gap-fill (within-station KFold R²) | Temporal forecast — raw R² (median) | Temporal forecast — shape r² (median) | Spatial map (LOSO R²) |
|---|---:|---:|---:|---:|
| t0 (clean) | +0.73 | −0.57 | +0.03 | −0.14 |
| t1 | +0.60 | +0.11 | +0.19 | −0.02 |
| t2 | +0.66 | −0.11 | +0.25 | +0.33 |
| t3 (polluted) | +0.76 | +0.01 | +0.35 | +0.56 |
| All | +0.68 | −0.07 | +0.20 | +0.20 |

Three readings follow directly from the table. First, *temporal gap-filling is genuine interpolation and it works everywhere*, reaching a within-station R² of 0.68 on average and 0.73 even at the cleanest tier — the very tier the spatial map fails on. Filling a missing hour is easy because the model interpolates between the same station's surrounding observed hours, and this is a real, deployable capability for the forty instrumented stations regardless of their pollution level. It is the same interpolation that, under the random cross-validation of Section 5.2, produced the headline R² of approximately 0.80; that figure is not wrong, it simply measures gap-filling rather than mapping.

Second, *temporal forecasting is extrapolation in time, and it degrades accordingly* — but not as catastrophically as the raw R² alone suggests, and the decomposition matters. The raw forecast R² is near zero or negative at every tier, yet the shape skill, the squared correlation between the predicted and observed future series, rises from 0.03 at clean sites to 0.35 at polluted ones, corresponding to a Pearson correlation near 0.5–0.6 at the moderate and polluted tiers. The model therefore *does* track the future pattern at stations with appreciable pollution variation; the negative raw R² is caused mostly by a *baseline drift* between the training period and the held-out future — a median level shift of roughly 8 to 13 µg/m³ at the polluted tiers — rather than by a failure to follow the temporal pattern. Correcting that offset lifts the forecast R² to between +0.11 and +0.32 at t1–t3. The honest conclusion is that future forecasting from exogenous features is modest and drift-limited, useful only with periodic recalibration, and absent at clean sites where even the pattern is uncorrelated; it is not, and is not intended to be, the objective of this thesis.

Third, and most importantly, *the spatial map is interpolation between stations in space*, and it succeeds and fails on exactly the same logic as the temporal tasks. Where the network is dense enough that a target location can be interpolated from near neighbours, the map performs well: the external validation of Section 5.6 reached a median R² of 0.53 across low-cost sites, 0.64 within 10 km of an anchor, and 0.68 at the US Embassy reference monitor. Where the target must be extrapolated beyond the reach of the network — far from any anchor, or in the sparsely instrumented clean and rural regimes — it collapses toward zero, as the distance-decay relationship of Section 4.5 and the per-tier LOSO breakdown of Section 5.4 both show. This is why the feature-importance analysis of Chapter 4 found spatial interpolation from ground neighbours (RFSI) to be the single largest contributor to model gain: the map is, mechanically, a spatial interpolator, and its skill is bounded by the density of the points it interpolates between.

The unifying picture is therefore a single axis. Tasks that interpolate between known observations — gap-filling a station's missing hours, or mapping a location surrounded by nearby stations — are solvable, with R² from roughly 0.5 to 0.8. Tasks that extrapolate to genuinely unobserved points — forecasting a station's future, or mapping a location far from any anchor or in an under-sampled clean regime — are not, with R² near zero. The same model spans this entire range, and the apparent contradiction between its "0.8 capability" and its "0.2 capability" dissolves once each number is labelled by the task it measures. The contribution of this thesis is not a single accuracy figure but this taxonomy: a precise statement of which air-quality estimation tasks are achievable from satellite data over a sparse tropical network, and which are bounded by the reach of the ground network rather than by the model. Gap-filling at instrumented sites and mapping within dense coverage are achievable today; forecasting and mapping into sparse, clean regions await a denser network.


Chapter 5 has presented the complete quantitative evaluation across the three validation frameworks. Random cross-validation confirmed that the model is a capable temporal interpolator at an R² of 0.81, while honest leave-one-station-out validation revealed that approximately three-quarters of that figure is station-identity leakage, leaving a true spatial-prediction R² near 0.20 once the model is given each station's true tier. Tier stratification was shown to be the single largest lever — a gain of approximately +0.17 within-file, rising to about +0.26 across framings — and the per-tier breakdown established a monotone, model-invariant gradient by which predictability scales with pollution level; whether level rather than region is the governing variable remains a hypothesis the network cannot resolve, since all top-tier stations are northern and the only North–South contrast, at t2, is not statistically significant. The deployable comparison demonstrated that withdrawing the oracle tier collapses performance to a per-station median near 0.04, with the +0.27 deployable-to-oracle gap quantifying the information cost of not knowing a station's mean, and the seven unsuccessful experiment families confirmed that no satellite-observable proxy recovers it. External validation against the low-cost-sensor network and the US Embassy reference monitor — a nearest-station feature-transfer plus sensor-agreement test rather than fully independent spatial prediction — showed the model achieving a dry-season median R² of 0.53 and a reference-grade R² of 0.68 within dense coverage, on par with the best international benchmarks at close range, while degrading sharply with distance from the nearest anchor. The unifying thread is that spatial prediction skill in Vietnam is bounded above by the reach of the ground network rather than by the algorithm. Chapter 6 draws these results together into conclusions and translates the binding density constraint into a concrete recommendation: that Vietnam pursue a hybrid monitoring strategy of sparse reference-grade stations densified by calibrated low-cost sensors, the intervention the distance-decay evidence identifies as the most direct and tractable lever on national PM2.5 prediction accuracy.