/*
 * Extract GHAP/ACAG PM2.5 climatology at station locations.
 * Three outputs: annual mean, monthly climatology, daily (2021-2022).
 *
 * Paste this entire script into code.earthengine.google.com and click Run.
 * Creates 3 export tasks — click the Tasks tab and run each one.
 *
 * After export completes, download CSVs and place at:
 *   data/gee_exports/pm25/ghap_annual_mean.csv
 *   data/gee_exports/pm25/ghap_monthly_climatology.csv
 *   data/gee_exports/pm25/ghap_daily_2021_2022.csv
 *
 * Columns (annual):  stationId, mean
 * Columns (monthly): stationId, month, mean
 * Columns (daily):   stationId, date, mean
 *
 * Source: Washington University ACAG V5GL04
 * Resolution: ~1 km
 *
 * NOTE: If the ACAG collection is not available in your GEE account, you may
 * need to import it as an asset from https://sites.wustl.edu/acag/datasets/
 * or use the community catalog: projects/sat-io/open-datasets/ACAG_PM25
 */

// ── Station coordinates ──
// Generate from metadata:
//   python -c "import csv; r=csv.DictReader(open('data/stations/metadata/envisoft_station_map.csv',encoding='utf-8-sig')); [print(f\"  ['{row['id']}', {row['latitude']}, {row['longitude']}],\") for row in r]"
var stationData = [
  // ['stationId', lat, lon],
];

if (stationData.length === 0) {
  print('ERROR: Paste station coordinates into stationData array first.');
}

// ── Build station point features ──
var stationPoints = [];
for (var si = 0; si < stationData.length; si++) {
  stationPoints.push(ee.Feature(
    ee.Geometry.Point([stationData[si][2], stationData[si][1]]), {
      stationId: String(stationData[si][0])
    }
  ));
}
var stationsFC = ee.FeatureCollection(stationPoints);

// ── GHAP/ACAG collection ──
// Try community catalog first; if not available, use your own asset
var GHAP_COLLECTION = 'projects/sat-io/open-datasets/ACAG_PM25/V5GL04';

// ══════════════════════════════════════════════════════════════════════════════
// 1. Annual mean (all available years averaged)
// ══════════════════════════════════════════════════════════════════════════════

var annualCol = ee.ImageCollection(GHAP_COLLECTION)
    .select('b1');  // PM2.5 band

var annualMean = annualCol.mean();

var annualSampled = annualMean.sampleRegions({
  collection: stationsFC,
  scale: 1000,
  geometries: false
});

var annualResults = annualSampled.map(function(f) {
  return ee.Feature(null, {
    stationId: f.get('stationId'),
    mean: f.get('b1')
  });
});

Export.table.toDrive({
  collection: annualResults,
  description: 'ghap_annual_mean',
  fileNamePrefix: 'ghap_annual_mean',
  fileFormat: 'CSV',
  selectors: ['stationId', 'mean']
});

// ══════════════════════════════════════════════════════════════════════════════
// 2. Monthly climatology (mean per calendar month)
// ══════════════════════════════════════════════════════════════════════════════

var allMonthly = ee.FeatureCollection([]);

for (var month = 1; month <= 12; month++) {
  var monthlyMean = annualCol
      .filter(ee.Filter.calendarRange(month, month, 'month'))
      .mean();

  var monthlySampled = monthlyMean.sampleRegions({
    collection: stationsFC,
    scale: 1000,
    geometries: false
  });

  var monthVal = month;
  var withMonth = monthlySampled.map(function(f) {
    return ee.Feature(null, {
      stationId: f.get('stationId'),
      month: monthVal,
      mean: f.get('b1')
    });
  });

  allMonthly = allMonthly.merge(withMonth);
}

Export.table.toDrive({
  collection: allMonthly,
  description: 'ghap_monthly_climatology',
  fileNamePrefix: 'ghap_monthly_climatology',
  fileFormat: 'CSV',
  selectors: ['stationId', 'month', 'mean']
});

// ══════════════════════════════════════════════════════════════════════════════
// 3. Daily PM2.5 for 2021-2022 (most recent available years)
// ══════════════════════════════════════════════════════════════════════════════

var dailyCol = ee.ImageCollection(GHAP_COLLECTION)
    .filterDate('2021-01-01', '2023-01-01')
    .select('b1');

var dailyResults = dailyCol.map(function(image) {
  var dateStr = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
  var sampled = image.sampleRegions({
    collection: stationsFC,
    scale: 1000,
    geometries: false
  });
  return sampled.map(function(f) {
    return ee.Feature(null, {
      stationId: f.get('stationId'),
      date: dateStr,
      mean: f.get('b1')
    });
  });
}).flatten();

Export.table.toDrive({
  collection: dailyResults,
  description: 'ghap_daily_2021_2022',
  fileNamePrefix: 'ghap_daily_2021_2022',
  fileFormat: 'CSV',
  selectors: ['stationId', 'date', 'mean']
});

print('3 export tasks created — click Tasks tab and run each one.');
