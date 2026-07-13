// ============================================================================
// GEE export cho 4 ban do PM2.5 DBSH — dan vao code.earthengine.google.com
// Bam Run -> tab Tasks ben phai hien 6 task -> bam RUN tung task.
// File GeoTIFF xuat ra Drive folder "map_data_export".
// Tai ve bo vao: D:\map_data\modis, D:\map_data\tropomi, D:\map_data\gpm
// (Ban v2: bo het getInfo() nen Run hien task ngay, khong bi treo)
// ============================================================================

var region = ee.Geometry.Rectangle([105.3, 20.1, 107.2, 21.5]);

var targets = [
  {tag: 'dec', end: '2025-12-09'},
  {tag: 'jul', end: '2025-07-30'},
];

var N_DAYS = 36;   // cua so cho rolling 30 ngay
var N_RAIN = 46;   // cua so mua (consecutive_dry_days mua dong)
var N_HOURS = 96;  // mua theo gio cho hrs_since_rain

// ---- helpers: tinh chuoi ngay/gio HOAN TOAN phia client (khong getInfo) ----
function pad2(n) { return (n < 10 ? '0' : '') + n; }
function fmtDay(d) {
  return '' + d.getUTCFullYear() + pad2(d.getUTCMonth() + 1) + pad2(d.getUTCDate());
}
function fmtHour(d) {
  return fmtDay(d) + '_' + pad2(d.getUTCHours());
}
function isoDay(d) {
  return d.getUTCFullYear() + '-' + pad2(d.getUTCMonth() + 1) + '-' + pad2(d.getUTCDate());
}
function parseISO(s) {
  var p = s.split('-');
  return new Date(Date.UTC(+p[0], +p[1] - 1, +p[2]));
}
var DAY_MS = 86400000, HOUR_MS = 3600000;

// stack anh daily thanh 1 anh nhieu band, band ten theo ngay
function dailyStack(col, bandName, endIso, nDays, reducer, outPrefix) {
  var endD = parseISO(endIso);           // ngay dich (bao gom)
  var imgs = [];
  for (var i = nDays - 1; i >= 0; i--) {
    var day = new Date(endD.getTime() - i * DAY_MS);
    var d0 = ee.Date(isoDay(day));
    var img = col.filterDate(d0, d0.advance(1, 'day'))
      .select(bandName).reduce(reducer)
      .rename(outPrefix + '_' + fmtDay(day));
    imgs.push(img);
  }
  return ee.Image.cat(imgs);
}

targets.forEach(function (t) {
  // ---------- 1) MODIS MAIAC AOD (MCD19A2, 1 km) ----------
  var maiac = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES').filterBounds(region);
  var aod55 = dailyStack(maiac, 'Optical_Depth_055', t.end, N_DAYS, ee.Reducer.mean(), 'aod055');
  var aod47 = dailyStack(maiac, 'Optical_Depth_047', t.end, N_DAYS, ee.Reducer.mean(), 'aod047');
  Export.image.toDrive({
    image: aod55.addBands(aod47).toFloat(),
    description: 'modis_maiac_' + t.tag,
    folder: 'map_data_export',
    region: region, scale: 1000, crs: 'EPSG:4326', maxPixels: 1e9,
  });

  // ---------- 2) TROPOMI 4 khi ----------
  var no2 = dailyStack(ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2').filterBounds(region),
    'tropospheric_NO2_column_number_density', t.end, N_DAYS, ee.Reducer.mean(), 'no2');
  var so2 = dailyStack(ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_SO2').filterBounds(region),
    'SO2_column_number_density', t.end, N_DAYS, ee.Reducer.mean(), 'so2');
  var co = dailyStack(ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CO').filterBounds(region),
    'CO_column_number_density', t.end, N_DAYS, ee.Reducer.mean(), 'co');
  var hcho = dailyStack(ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_HCHO').filterBounds(region),
    'tropospheric_HCHO_column_number_density', t.end, N_DAYS, ee.Reducer.mean(), 'hcho');
  Export.image.toDrive({
    image: no2.addBands(so2).addBands(co).addBands(hcho).toFloat(),
    description: 'tropomi_gases_' + t.tag,
    folder: 'map_data_export',
    region: region, scale: 2226, crs: 'EPSG:4326', maxPixels: 1e9,
  });

  // ---------- 3) GPM IMERG: mua ngay + mua gio ----------
  var imerg = ee.ImageCollection('NASA/GPM_L3/IMERG_V07').filterBounds(region)
    .map(function (im) { return im.select('precipitation').multiply(0.5); }); // mm/30ph
  var rainDaily = dailyStack(imerg, 'precipitation', t.end, N_RAIN, ee.Reducer.sum(), 'rain');

  var endD = parseISO(t.end);
  var endMs = endD.getTime() + DAY_MS;   // het ngay dich UTC
  var hImgs = [];
  for (var h = N_HOURS; h >= 1; h--) {
    var hd = new Date(endMs - h * HOUR_MS);
    var h0 = ee.Date(hd.toISOString());
    hImgs.push(imerg.filterDate(h0, h0.advance(1, 'hour'))
      .select('precipitation').reduce(ee.Reducer.sum())
      .rename('rainh_' + fmtHour(hd)));
  }
  Export.image.toDrive({
    image: rainDaily.addBands(ee.Image.cat(hImgs)).toFloat(),
    description: 'gpm_rain_' + t.tag,
    folder: 'map_data_export',
    region: region, scale: 11132, crs: 'EPSG:4326', maxPixels: 1e9,
  });
});

print('OK: mo tab Tasks (cot phai) va bam RUN cho 6 task.');
Map.centerObject(region, 8);
Map.addLayer(region, {color: 'red'}, 'DBSH region');
