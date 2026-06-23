/*
 * Export a one-day TROPOMI gas slice for Vietnam.
 *
 * Paste into https://code.earthengine.google.com and run from the Code Editor.
 * This is intentionally small: use it to create a single map-day raster or a
 * sampled point grid, not a multi-year archive.
 *
 * Outputs to Google Drive:
 *   tropomi_vietnam_slice_YYYYMMDD.tif  (NO2, SO2, CO, HCHO bands)
 *   tropomi_vietnam_points_YYYYMMDD.csv (optional point-grid table)
 */

var DATE = '2025-01-15';
var SCALE_METERS = 5500;  // close to effective S5P/TROPOMI footprint scale
var EXPORT_POINTS_CSV = false;

var start = ee.Date(DATE);
var end = start.advance(1, 'day');
var dateTag = DATE.replace(/-/g, '');

var vietnam = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Viet Nam'))
  .geometry();

Map.centerObject(vietnam, 6);
Map.addLayer(vietnam, {color: 'white'}, 'Vietnam boundary', false);

var products = [
  {
    name: 'NO2',
    collection: 'COPERNICUS/S5P/OFFL/L3_NO2',
    band: 'tropospheric_NO2_column_number_density',
    qaMin: 0.75
  },
  {
    name: 'SO2',
    collection: 'COPERNICUS/S5P/OFFL/L3_SO2',
    band: 'SO2_column_number_density',
    qaMin: 0.50
  },
  {
    name: 'CO',
    collection: 'COPERNICUS/S5P/OFFL/L3_CO',
    band: 'CO_column_number_density',
    qaMin: 0.50
  },
  {
    name: 'HCHO',
    collection: 'COPERNICUS/S5P/OFFL/L3_HCHO',
    band: 'tropospheric_HCHO_column_number_density',
    qaMin: 0.50
  }
];

function dailyMean(product) {
  var collection = ee.ImageCollection(product.collection)
    .filterDate(start, end)
    .filterBounds(vietnam);

  var masked = collection.map(function(image) {
    var value = image.select(product.band);
    var qa = image.select('qa_value');
    return value.updateMask(qa.gte(product.qaMin));
  });

  return masked.mean().rename(product.name);
}

var images = products.map(function(product) {
  return dailyMean(product);
});

var gas = ee.Image.cat(images).clip(vietnam).toFloat();

print('Date:', DATE);
print('Daily gas image:', gas);

Map.addLayer(gas.select('NO2'), {
  min: 0,
  max: 0.0002,
  palette: ['0b1d4d', '2455a4', '36c2ff', 'f9e45b', 'f46d43', '7a0403']
}, 'NO2');

Map.addLayer(gas.select('SO2'), {
  min: -0.00005,
  max: 0.0003,
  palette: ['1b1b1b', '2c7bb6', 'abd9e9', 'ffffbf', 'fdae61', 'd7191c']
}, 'SO2', false);

Export.image.toDrive({
  image: gas,
  description: 'tropomi_vietnam_slice_' + dateTag,
  fileNamePrefix: 'tropomi_vietnam_slice_' + dateTag,
  region: vietnam,
  scale: SCALE_METERS,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

if (EXPORT_POINTS_CSV) {
  var pointGrid = gas.addBands(ee.Image.pixelLonLat()).sample({
    region: vietnam,
    scale: SCALE_METERS,
    geometries: false,
    tileScale: 4
  });

  Export.table.toDrive({
    collection: pointGrid,
    description: 'tropomi_vietnam_points_' + dateTag,
    fileNamePrefix: 'tropomi_vietnam_points_' + dateTag,
    fileFormat: 'CSV',
    selectors: ['longitude', 'latitude', 'NO2', 'SO2', 'CO', 'HCHO']
  });
}

print('Export task(s) created. Open the Tasks tab and click Run.');
