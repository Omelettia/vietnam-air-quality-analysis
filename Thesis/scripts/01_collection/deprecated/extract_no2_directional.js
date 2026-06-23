/*
 * Extract directional NO2 climatology from Sentinel-5P.
 *
 * Paste this entire script into code.earthengine.google.com and click Run.
 * Results export to Google Drive as "no2_directional_clim".
 *
 * After export completes, download the CSV from Drive and place it at:
 *   data/stations/metadata/no2_directional_clim.csv
 *
 * Columns: stationId, direction, distance_km, month, no2_mean
 */

// ── Station coordinates (40 thesis stations) ──
// Paste your station list here. Format: [stationId, lat, lon]
// This list is loaded from station_selection_final.csv.
// Replace STATIONS_PLACEHOLDER with actual data, or load from a Fusion Table.

// Example for the first few stations — replace with full list:
var stationData = [
  // ['stationId', lat, lon],
  // Add all 40 stations from station_selection_final.csv
  // You can generate this list with:
  //   python -c "import csv; r=csv.DictReader(open('analysis/thesis_audit/station_selection_final.csv',encoding='utf-8-sig')); [print(f\"  ['{row['stationId']}', {row['lat']}, {row['lon']}],\") for row in r]"
];

if (stationData.length === 0) {
  print('ERROR: Paste station coordinates into stationData array first.');
  print('Run the Python one-liner in the comment above to generate the list.');
  // Return early — nothing to do
}

var DIRECTIONS = {
  'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
  'S': 180, 'SW': 225, 'W': 270, 'NW': 315
};
var DISTANCES_KM = [5, 10, 20];
var START_DATE = '2023-01-01';
var END_DATE = '2024-12-31';

// ── Offset a point by bearing and distance ──
function offsetPoint(lat, lon, bearingDeg, distKm) {
  var R = 6371.0;
  var d = distKm / R;
  var brng = bearingDeg * Math.PI / 180;
  var lat1 = lat * Math.PI / 180;
  var lon1 = lon * Math.PI / 180;
  var lat2 = Math.asin(Math.sin(lat1) * Math.cos(d) +
                       Math.cos(lat1) * Math.sin(d) * Math.cos(brng));
  var lon2 = lon1 + Math.atan2(Math.sin(brng) * Math.sin(d) * Math.cos(lat1),
                                Math.cos(d) - Math.sin(lat1) * Math.sin(lat2));
  return [lon2 * 180 / Math.PI, lat2 * 180 / Math.PI]; // [lon, lat] for GEE
}

// ── Build sampling points ──
var pointFeatures = [];

for (var si = 0; si < stationData.length; si++) {
  var stn = stationData[si];
  var sid = String(stn[0]);
  var lat = stn[1];
  var lon = stn[2];

  // Center point
  pointFeatures.push(ee.Feature(ee.Geometry.Point([lon, lat]), {
    stationId: sid, direction: 'C', distance_km: 0
  }));

  // Directional points
  var dirNames = Object.keys(DIRECTIONS);
  for (var di = 0; di < dirNames.length; di++) {
    var dirName = dirNames[di];
    var bearing = DIRECTIONS[dirName];
    for (var dsi = 0; dsi < DISTANCES_KM.length; dsi++) {
      var dist = DISTANCES_KM[dsi];
      var offsetCoords = offsetPoint(lat, lon, bearing, dist);
      pointFeatures.push(ee.Feature(
        ee.Geometry.Point(offsetCoords), {
          stationId: sid,
          direction: dirName,
          distance_km: dist
        }
      ));
    }
  }
}

var pointsFC = ee.FeatureCollection(pointFeatures);
print('Total sampling points:', pointFeatures.length);

// ── NO2 collection with quality filter ──
var no2Collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.gt('QUALITY_FLAG', 0.75))
    .select('tropospheric_NO2_column_number_density');

// ── Sample each month ──
var allResults = ee.FeatureCollection([]);

for (var month = 1; month <= 12; month++) {
  var monthlyMean = no2Collection
      .filter(ee.Filter.calendarRange(month, month, 'month'))
      .mean();

  var sampled = monthlyMean.sampleRegions({
    collection: pointsFC,
    scale: 1113.2,
    geometries: false
  });

  // Add month property
  var monthVal = month;
  var withMonth = sampled.map(function(f) {
    return f.set('month', monthVal);
  });

  allResults = allResults.merge(withMonth);
}

// ── Rename band column for clarity ──
allResults = allResults.map(function(f) {
  return ee.Feature(null, {
    stationId: f.get('stationId'),
    direction: f.get('direction'),
    distance_km: f.get('distance_km'),
    month: f.get('month'),
    no2_mean: f.get('tropospheric_NO2_column_number_density')
  });
});

print('Total result rows:', allResults.size());

// ── Export to Drive ──
Export.table.toDrive({
  collection: allResults,
  description: 'no2_directional_clim',
  fileNamePrefix: 'no2_directional_clim',
  fileFormat: 'CSV',
  selectors: ['stationId', 'direction', 'distance_km', 'month', 'no2_mean']
});

print('Export task created — click Tasks tab and click Run.');
