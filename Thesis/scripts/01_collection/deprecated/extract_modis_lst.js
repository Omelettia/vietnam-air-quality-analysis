/*
 * Extract MODIS Land Surface Temperature over a 5x5 pixel grid per station.
 *
 * Paste this entire script into code.earthengine.google.com and click Run.
 * Creates one export task per year — click the Tasks tab and run each one.
 *
 * After export completes, download the CSVs from Drive and place them at:
 *   data/gee_exports/temp/modis_lst_5x5_YYYY.csv
 *
 * Columns: stationId, date, satellite, row, col, lst_day, lst_night
 *
 * GEE Collections: MODIS/061/MOD11A1 (Terra), MODIS/061/MYD11A1 (Aqua)
 * Resolution: 1 km
 * Grid: 5x5 pixels (row/col from -2 to +2) centered on each station
 * Units: Celsius (raw DN * 0.02 - 273.15)
 */

// ── Station coordinates ──
// Generate this list from your station metadata:
//   python -c "import csv; r=csv.DictReader(open('data/stations/metadata/envisoft_station_map.csv',encoding='utf-8-sig')); [print(f\"  ['{row['id']}', {row['latitude']}, {row['longitude']}],\") for row in r]"
var stationData = [
  // ['stationId', lat, lon],
];

if (stationData.length === 0) {
  print('ERROR: Paste station coordinates into stationData array first.');
}

var START_YEAR = 2023;
var END_YEAR = 2026;
var GRID_HALF = 2;
var PIXEL_DEG = 0.009;  // ~1 km

// ── Build 5x5 grid points per station ──
function buildGridPoints(stationData) {
  var features = [];
  for (var si = 0; si < stationData.length; si++) {
    var sid = String(stationData[si][0]);
    var lat = stationData[si][1];
    var lon = stationData[si][2];
    for (var r = -GRID_HALF; r <= GRID_HALF; r++) {
      for (var c = -GRID_HALF; c <= GRID_HALF; c++) {
        features.push(ee.Feature(
          ee.Geometry.Point([lon + c * PIXEL_DEG, lat + r * PIXEL_DEG]), {
            stationId: sid, row: r, col: c
          }
        ));
      }
    }
  }
  return ee.FeatureCollection(features);
}

var gridPoints = buildGridPoints(stationData);
print('Grid points:', gridPoints.size());

// ── Kelvin DN to Celsius ──
function dnToCelsius(image) {
  return image.multiply(0.02).subtract(273.15);
}

// ── Process one year at a time ──
for (var year = START_YEAR; year <= END_YEAR; year++) {
  var startDate = year + '-01-01';
  var endDate = (year === END_YEAR) ? year + '-04-16' : (year + 1) + '-01-01';

  var collections = [
    {name: 'Terra', id: 'MODIS/061/MOD11A1'},
    {name: 'Aqua',  id: 'MODIS/061/MYD11A1'}
  ];

  var yearResults = ee.FeatureCollection([]);

  for (var ci = 0; ci < collections.length; ci++) {
    var sat = collections[ci];

    var col = ee.ImageCollection(sat.id)
        .filterDate(startDate, endDate)
        .select(['LST_Day_1km', 'LST_Night_1km']);

    var days = col.aggregate_array('system:time_start')
        .map(function(t) { return ee.Date(t).format('YYYY-MM-dd'); })
        .distinct();

    var satName = sat.name;
    var satResults = days.map(function(dateStr) {
      dateStr = ee.String(dateStr);
      var date = ee.Date(dateStr);
      var dayImage = col.filterDate(date, date.advance(1, 'day')).mean();
      var lstDay = dnToCelsius(dayImage.select('LST_Day_1km'));
      var lstNight = dnToCelsius(dayImage.select('LST_Night_1km'));
      var combined = lstDay.rename('lst_day').addBands(lstNight.rename('lst_night'));

      var sampled = combined.sampleRegions({
        collection: gridPoints,
        scale: 1000,
        geometries: false
      });

      return sampled.map(function(f) {
        return ee.Feature(null, {
          stationId: f.get('stationId'),
          date: dateStr,
          satellite: satName,
          row: f.get('row'),
          col: f.get('col'),
          lst_day: f.get('lst_day'),
          lst_night: f.get('lst_night')
        });
      });
    });

    yearResults = yearResults.merge(ee.FeatureCollection(satResults).flatten());
  }

  Export.table.toDrive({
    collection: yearResults,
    description: 'modis_lst_5x5_' + year,
    fileNamePrefix: 'modis_lst_5x5_' + year,
    fileFormat: 'CSV',
    selectors: ['stationId', 'date', 'satellite', 'row', 'col', 'lst_day', 'lst_night']
  });

  print('Export task created for ' + year);
}
