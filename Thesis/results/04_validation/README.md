# 04_validation

Validation results for the thesis.

## Canonical Result

Use this as the final validation report:

```text
report_red_river_delta_v5h.txt
```

Grouped feature-gain shares are stored separately in:

```text
red_river_delta_feature_gain_by_group.csv
```

Station-level external validation metrics used to draw the external-validation
distribution are stored in:

```text
red_river_delta_external_station_metrics.csv
```

It contains:

- Red River Delta internal LOSO for 12 KK stations;
- daily and hourly metrics;
- LCS external validation inside the regional bounding box;
- US Embassy Hanoi validation;
- the caveat that the final model requires concurrent nearby-anchor PM2.5;
- the final decision to pivot away from national diverse-stream/kNN routing.

## Archived Supporting Files

Older probes are not kept in this active folder. They were moved to `archive/`
so the validation folder only contains files used by the final thesis result.

Do not use old national external-validation files as the thesis headline.
