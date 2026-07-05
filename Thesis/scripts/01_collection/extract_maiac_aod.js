/*
 * Extract MODIS MAIAC AOD over a 5x5 pixel grid per station.
 *
 * Paste this entire script into code.earthengine.google.com and click Run.
 * Creates one export task per year — click the Tasks tab and run each one.
 *
 * After export completes, download the CSVs from Drive and place them at:
 *   data/gee_exports/aod/maiac_aod_5x5_YYYY.csv
 *
 * Columns: stationId, date, orbit, row, col, aod47, aod55, ae
 *
 * GEE Collection: MODIS/061/MCD19A2
 * Resolution: 1 km
 * Grid: 5x5 pixels (row/col from -2 to +2) centered on each station
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
var PIXEL_DEG = 0.009;  // ~1 km at equator

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

// ── Process one year at a time ──
for (var year = START_YEAR; year <= END_YEAR; year++) {
  var startDate = year + '-01-01';
  var endDate = (year === END_YEAR) ? year + '-04-16' : (year + 1) + '-01-01';

  var maiac = ee.ImageCollection('MODIS/061/MCD19A2')
      .filterDate(startDate, endDate)
      .select(['Optical_Depth_047', 'Optical_Depth_055']);

  var days = maiac.aggregate_array('system:time_start')
      .map(function(t) { return ee.Date(t).format('YYYY-MM-dd'); })
      .distinct();

  var yearResults = days.map(function(dateStr) {
    dateStr = ee.String(dateStr);
    var date = ee.Date(dateStr);
    var dayImages = maiac.filterDate(date, date.advance(1, 'day'));

    // Terra (~10:30 local) and Aqua (~13:30 local) as separate orbits
    var terra = dayImages.filter(ee.Filter.lt('system:time_start',
        date.advance(12, 'hour').millis())).mean();
    var aqua = dayImages.filter(ee.Filter.gte('system:time_start',
        date.advance(12, 'hour').millis())).mean();

    function sampleOrbit(image, orbitName) {
      var sampled = image.sampleRegions({
        collection: gridPoints,
        scale: 1000,
        geometries: false
      });
      return sampled.map(function(f) {
        var aod47 = ee.Number(f.get('Optical_Depth_047')).multiply(0.001);
        var aod55 = ee.Number(f.get('Optical_Depth_055')).multiply(0.001);
        var ae = aod47.log().subtract(aod55.log())
            .divide(ee.Number(0.47).log().subtract(ee.Number(0.55).log()));
        return ee.Feature(null, {
          stationId: f.get('stationId'),
          date: dateStr,
          orbit: orbitName,
          row: f.get('row'),
          col: f.get('col'),
          aod47: aod47,
          aod55: aod55,
          ae: ae
        });
      });
    }

    var terraResults = sampleOrbit(terra, 'Terra');
    var aquaResults = sampleOrbit(aqua, 'Aqua');
    return terraResults.merge(aquaResults);
  });

  var flat = ee.FeatureCollection(yearResults).flatten();

  Export.table.toDrive({
    collection: flat,
    description: 'maiac_aod_5x5_' + year,
    fileNamePrefix: 'maiac_aod_5x5_' + year,
    fileFormat: 'CSV',
    selectors: ['stationId', 'date', 'orbit', 'row', 'col', 'aod47', 'aod55', 'ae']
  });

  print('Export task created for ' + year);
}
