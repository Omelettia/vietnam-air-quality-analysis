/*
 * Extract daily TROPOMI trace gases and station-level MODIS LST means.
 * Output in long format: one row per (stationId, date, variable).
 *
 * Paste this entire script into code.earthengine.google.com and click Run.
 * Creates two export tasks per year (tropomi + lst) — run each from the Tasks tab.
 *
 * After export completes, download CSVs and place at:
 *   data/gee_exports/last/tropomi_daily_YYYY.csv
 *   data/gee_exports/last/modis_lst_daily_YYYY.csv
 *
 * Columns: stationId, date, variable, mean
 *
 * TROPOMI variables: NO2, SO2, CO, HCHO
 * LST variables: LST_terra_day, LST_terra_night, LST_aqua_day, LST_aqua_night
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

var START_YEAR = 2023;
var END_YEAR = 2026;

// ── Build station point features (center pixel only) ──
var stationPoints = [];
for (var si = 0; si < stationData.length; si++) {
  stationPoints.push(ee.Feature(
    ee.Geometry.Point([stationData[si][2], stationData[si][1]]), {
      stationId: String(stationData[si][0])
    }
  ));
}
var stationsFC = ee.FeatureCollection(stationPoints);
print('Stations:', stationsFC.size());

// ── TROPOMI collections ──
var tropomiProducts = [
  {variable: 'NO2',  collection: 'COPERNICUS/S5P/OFFL/L3_NO2',
   band: 'tropospheric_NO2_column_number_density'},
  {variable: 'SO2',  collection: 'COPERNICUS/S5P/OFFL/L3_SO2',
   band: 'SO2_column_number_density'},
  {variable: 'CO',   collection: 'COPERNICUS/S5P/OFFL/L3_CO',
   band: 'CO_column_number_density'},
  {variable: 'HCHO', collection: 'COPERNICUS/S5P/OFFL/L3_HCHO',
   band: 'tropospheric_HCHO_column_number_density'}
];

// ── MODIS LST collections ──
var lstProducts = [
  {variable: 'LST_terra_day',   collection: 'MODIS/061/MOD11A1', band: 'LST_Day_1km'},
  {variable: 'LST_terra_night', collection: 'MODIS/061/MOD11A1', band: 'LST_Night_1km'},
  {variable: 'LST_aqua_day',   collection: 'MODIS/061/MYD11A1', band: 'LST_Day_1km'},
  {variable: 'LST_aqua_night', collection: 'MODIS/061/MYD11A1', band: 'LST_Night_1km'}
];

// ── Helper: sample daily mean for a product over one year ──
function extractDailyMean(product, startDate, endDate, scale) {
  var col = ee.ImageCollection(product.collection)
      .filterDate(startDate, endDate)
      .select(product.band);

  var days = ee.List.sequence(0, ee.Date(endDate).difference(ee.Date(startDate), 'day').subtract(1));

  var results = days.map(function(dayOffset) {
    var date = ee.Date(startDate).advance(dayOffset, 'day');
    var dateStr = date.format('YYYY-MM-dd');
    var dayImage = col.filterDate(date, date.advance(1, 'day')).mean();

    var sampled = dayImage.sampleRegions({
      collection: stationsFC,
      scale: scale,
      geometries: false
    });

    var varName = product.variable;
    return sampled.map(function(f) {
      var val = f.get(product.band);
      if (product.band === 'LST_Day_1km' || product.band === 'LST_Night_1km') {
        val = ee.Number(val).multiply(0.02).subtract(273.15);
      }
      return ee.Feature(null, {
        stationId: f.get('stationId'),
        date: dateStr,
        variable: varName,
        mean: val
      });
    });
  });

  return ee.FeatureCollection(results).flatten();
}

// ── Export per year ──
for (var year = START_YEAR; year <= END_YEAR; year++) {
  var startDate = year + '-01-01';
  var endDate = (year === END_YEAR) ? year + '-04-16' : (year + 1) + '-01-01';

  // TROPOMI (scale ~5.5 km)
  var tropomiResults = ee.FeatureCollection([]);
  for (var ti = 0; ti < tropomiProducts.length; ti++) {
    tropomiResults = tropomiResults.merge(
      extractDailyMean(tropomiProducts[ti], startDate, endDate, 5500)
    );
  }

  Export.table.toDrive({
    collection: tropomiResults,
    description: 'tropomi_daily_' + year,
    fileNamePrefix: 'tropomi_daily_' + year,
    fileFormat: 'CSV',
    selectors: ['stationId', 'date', 'variable', 'mean']
  });

  // MODIS LST daily (scale 1 km, center pixel)
  var lstResults = ee.FeatureCollection([]);
  for (var li = 0; li < lstProducts.length; li++) {
    lstResults = lstResults.merge(
      extractDailyMean(lstProducts[li], startDate, endDate, 1000)
    );
  }

  Export.table.toDrive({
    collection: lstResults,
    description: 'modis_lst_daily_' + year,
    fileNamePrefix: 'modis_lst_daily_' + year,
    fileFormat: 'CSV',
    selectors: ['stationId', 'date', 'variable', 'mean']
  });

  print('Export tasks created for ' + year);
}
