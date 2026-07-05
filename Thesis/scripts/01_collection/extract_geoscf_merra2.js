/*
 * Extract hourly GEOS-CF PM2.5/trace gases and MERRA-2 aerosol species.
 *
 * Paste this entire script into code.earthengine.google.com and click Run.
 * Creates two export tasks per year (geoscf + merra2) — run each from Tasks tab.
 *
 * After export completes, download CSVs and place at:
 *   data/gee_exports/hourly/geoscf_pm25_YYYY.csv
 *   data/gee_exports/hourly/merra2_aerosol_YYYY.csv
 *
 * GEOS-CF columns: stationId, datetime, PM25_RH35_GCC, CO, NO2, SO2
 * MERRA-2 columns: stationId, datetime, BCSMASS, OCSMASS, SO4SMASS,
 *                  DUSMASS25, SSSMASS25, TOTEXTTAU, merra2_pm25
 *
 * GEE Collections:
 *   NASA/GEOS-CF/v1/rpl/tavg1_2d_chm_Nx   (0.25 deg, 3-hourly)
 *   NASA/GSFC/MERRA/aer/2                  (0.5x0.625 deg, hourly)
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

// ══════════════════════════════════════════════════════════════════════════════
// GEOS-CF: PM2.5, CO, NO2, SO2 (3-hourly)
// ══════════════════════════════════════════════════════════════════════════════

var geoscfBands = ['PM25_RH35_GCC', 'CO', 'NO2', 'SO2'];

for (var year = START_YEAR; year <= END_YEAR; year++) {
  var startDate = year + '-01-01';
  var endDate = (year === END_YEAR) ? year + '-04-16' : (year + 1) + '-01-01';

  var geoscf = ee.ImageCollection('NASA/GEOS-CF/v1/rpl/tavg1_2d_chm_Nx')
      .filterDate(startDate, endDate)
      .select(geoscfBands);

  var geoscfSampled = geoscf.map(function(image) {
    var dt = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd HH:mm');
    var sampled = image.sampleRegions({
      collection: stationsFC,
      scale: 27750,  // ~0.25 deg
      geometries: false
    });
    return sampled.map(function(f) {
      return ee.Feature(null, {
        stationId: f.get('stationId'),
        datetime: dt,
        PM25_RH35_GCC: f.get('PM25_RH35_GCC'),
        CO: f.get('CO'),
        NO2: f.get('NO2'),
        SO2: f.get('SO2')
      });
    });
  }).flatten();

  Export.table.toDrive({
    collection: geoscfSampled,
    description: 'geoscf_pm25_' + year,
    fileNamePrefix: 'geoscf_pm25_' + year,
    fileFormat: 'CSV',
    selectors: ['stationId', 'datetime', 'PM25_RH35_GCC', 'CO', 'NO2', 'SO2']
  });

  print('GEOS-CF export task created for ' + year);
}

// ══════════════════════════════════════════════════════════════════════════════
// MERRA-2: aerosol species + derived PM2.5 (hourly)
// ══════════════════════════════════════════════════════════════════════════════

var merraBands = ['BCSMASS', 'OCSMASS', 'SO4SMASS', 'DUSMASS25', 'SSSMASS25', 'TOTEXTTAU'];

for (var year = START_YEAR; year <= END_YEAR; year++) {
  var startDate = year + '-01-01';
  var endDate = (year === END_YEAR) ? year + '-04-16' : (year + 1) + '-01-01';

  var merra2 = ee.ImageCollection('NASA/GSFC/MERRA/aer/2')
      .filterDate(startDate, endDate)
      .select(merraBands);

  var merraSampled = merra2.map(function(image) {
    var dt = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd HH:mm');
    var sampled = image.sampleRegions({
      collection: stationsFC,
      scale: 55500,  // ~0.5 deg
      geometries: false
    });
    return sampled.map(function(f) {
      // PM2.5 = (BC + 1.4*OC + 1.375*SO4 + dust25 + ss25) * 1e9 [kg/m3 -> ug/m3]
      var bc   = ee.Number(f.get('BCSMASS'));
      var oc   = ee.Number(f.get('OCSMASS'));
      var so4  = ee.Number(f.get('SO4SMASS'));
      var dust = ee.Number(f.get('DUSMASS25'));
      var ss   = ee.Number(f.get('SSSMASS25'));
      var pm25 = bc.add(oc.multiply(1.4)).add(so4.multiply(1.375))
                   .add(dust).add(ss).multiply(1e9);
      return ee.Feature(null, {
        stationId: f.get('stationId'),
        datetime: dt,
        BCSMASS: f.get('BCSMASS'),
        OCSMASS: f.get('OCSMASS'),
        SO4SMASS: f.get('SO4SMASS'),
        DUSMASS25: f.get('DUSMASS25'),
        SSSMASS25: f.get('SSSMASS25'),
        TOTEXTTAU: f.get('TOTEXTTAU'),
        merra2_pm25: pm25
      });
    });
  }).flatten();

  Export.table.toDrive({
    collection: merraSampled,
    description: 'merra2_aerosol_' + year,
    fileNamePrefix: 'merra2_aerosol_' + year,
    fileFormat: 'CSV',
    selectors: ['stationId', 'datetime', 'BCSMASS', 'OCSMASS', 'SO4SMASS',
                'DUSMASS25', 'SSSMASS25', 'TOTEXTTAU', 'merra2_pm25']
  });

  print('MERRA-2 export task created for ' + year);
}
