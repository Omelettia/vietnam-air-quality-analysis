/*
 * Extract directional climatology for 123 stations.
 * Paste into code.earthengine.google.com, click Run,
 * then start each export task in the Tasks tab (right panel).
 *
 * Products: TROPOMI NO2/SO2/CO/HCHO (monthly), MODIS LST (seasonal),
 *           VIIRS NTL (annual), MODIS Fire (seasonal)
 *
 * Sampling: center + 8 directions × 3 distances (5/10/20 km) = 25 pts/station
 * Output: CSV files on Google Drive in folder "gee_directional_123"
 */

// ═══════════════════════════════════════════════════════════════
//  STATIONS (123)
// ═══════════════════════════════════════════════════════════════
var STATIONS = [
  {id:"28505268571336961948594948504",lat:21.3385,lon:105.367},
  {id:"28505272740301122608933325208",lat:16.074,lon:108.217},
  {id:"28915732959631398237539556920",lat:21.0356,lon:105.833},
  {id:"31388836637198369136462150188",lat:13.7851,lon:109.22},
  {id:"31388839920718814259329251882",lat:10.9923,lon:106.658},
  {id:"31388851800421997746903202346",lat:16.0622,lon:108.159},
  {id:"31388883344354363840031242796",lat:20.536,lon:105.916},
  {id:"31390903576425084107499649578",lat:21.0052,lon:105.842},
  {id:"31390908889087377344742439468",lat:21.0031,lon:105.795},
  {id:"31390912357075263208060500522",lat:10.7823,lon:106.683},
  {id:"31390916083317566102523755051",lat:10.7832,lon:106.753},
  {id:"31390921469766835629621918251",lat:20.6618,lon:106.059},
  {id:"31390928700890512538630765099",lat:11.9527,lon:108.43},
  {id:"31390932574706768021562473002",lat:10.5391,lon:106.405},
  {id:"31390939012620449761949792810",lat:11.5736,lon:108.992},
  {id:"31390944251495766704153049642",lat:15.5621,lon:108.487},
  {id:"31390951371938979254287350314",lat:15.121,lon:108.804},
  {id:"31390957404024291365397346858",lat:20.4578,lon:106.344},
  {id:"31390965631272148257054013994",lat:10.4955,lon:107.274},
  {id:"31388868531618872623864101418",lat:20.9381,lon:106.336},
  {id:"31836631011116458146308623213",lat:13.9425,lon:107.997},
  {id:"31570631866920915749872483233",lat:13.7619,lon:109.22},
  {id:"32226856627901174291787516193",lat:21.3076,lon:106.191},
  {id:"32226870949329422321657029896",lat:21.2684,lon:106.205},
  {id:"32364176301658391599618161719",lat:20.4398,lon:106.188},
  {id:"32364183997304919819454378150",lat:20.4378,lon:106.176},
  {id:"32364190084730464143606411447",lat:20.3996,lon:106.182},
  {id:"32364194875996560076825514951",lat:20.42,lon:106.162},
  {id:"32364200193546216536440697093",lat:20.4357,lon:106.156},
  {id:"28916504310234840885489983032",lat:21.0272,lon:106.034},
  {id:"29544533341424688633522251105",lat:14.0177,lon:108.035},
  {id:"30487895627496546123240481728",lat:13.6987,lon:108.076},
  {id:"29828513366380889034790000096",lat:20.9805,lon:105.974},
  {id:"28599103930259709512833128161",lat:21.0065,lon:107.344},
  {id:"29996239538320865599553202177",lat:11.0303,lon:106.356},
  {id:"29203727697074312726675247132",lat:20.4616,lon:106.517},
  {id:"30921862678974937849512304668",lat:9.92389,lon:106.34},
  {id:"29098314940209976158571475636",lat:9.59914,lon:106.52},
  {id:"29098319146067624969113973428",lat:9.57555,lon:106.488},
  {id:"28560877461938780203765592307",lat:21.0491,lon:105.883},
  {id:"31651502905690497791503780869",lat:21.5391,lon:105.872},
  {id:"31616865099255512061948816121",lat:15.9961,lon:108.207},
  {id:"30991938797551443885460120607",lat:9.61358,lon:105.968},
  {id:"30668578626453200136007481288",lat:10.2513,lon:105.947},
  {id:"28602837126985114455002404577",lat:21.0358,lon:106.764},
  {id:"29996225242094208443243751425",lat:11.3277,lon:106.104},
  {id:"29936546593921360616662752773",lat:20.6617,lon:106.059},
  {id:"29541549672804486395063249961",lat:13.9539,lon:108.656},
  {id:"29518885338952740244749551027",lat:21.07,lon:106.556},
  {id:"29518872592252585279589451187",lat:21.0062,lon:106.859},
  {id:"29518862280522648049760863667",lat:21.4528,lon:107.76},
  {id:"29518852005686198968139908531",lat:20.9725,lon:107.044},
  {id:"29518839738601389925132404147",lat:21.0093,lon:107.274},
  {id:"29196021237696127337075448678",lat:21.0832,lon:106.281},
  {id:"29196010501691076420299004774",lat:21.1873,lon:106.074},
  {id:"28916774462801800655608897080",lat:21.0243,lon:106.017},
  {id:"30018605256279546654144260098",lat:22.2955,lon:104.134},
  {id:"28916766954976962647717994040",lat:21.1518,lon:106.152},
  {id:"28916576381663936943098322488",lat:21.1968,lon:105.993},
  {id:"28916745851901742294496599608",lat:21.1194,lon:105.99},
  {id:"28602897318711027016899843809",lat:20.9454,lon:107.131},
  {id:"28602553176253587650986727137",lat:21.0585,lon:106.598},
  {id:"28601787986862666164115166945",lat:21.0652,lon:107.326},
  {id:"28601309072493024807602706145",lat:20.9797,lon:107.087},
  {id:"32496602817040565458030689725",lat:21.0608,lon:105.749},
  {id:"32496693563219291690449469264",lat:21.0272,lon:106.509},
  {id:"32497734637638860594066758096",lat:21.5761,lon:106.503},
  {id:"32497836911670612380284465171",lat:21.6959,lon:105.864},
  {id:"32497696449863912939996438674",lat:21.0608,lon:105.749},
  {id:"32496627929703877431864819302",lat:20.9968,lon:105.596},
  {id:"32496584622031367061634346762",lat:20.9769,lon:105.994},
  {id:"32496597338357575566293859742",lat:21.0487,lon:105.882},
  {id:"32496608539373531178394552735",lat:21.0224,lon:105.81},
  {id:"32496614298366895058622457929",lat:21.0683,lon:105.708},
  {id:"32496619672150063767061412126",lat:20.9436,lon:105.754},
  {id:"32496631939608265457394360507",lat:21.0045,lon:105.853},
  {id:"32496634188432693119199539147",lat:20.8327,lon:105.671},
  {id:"32497722117136617634216524795",lat:21.1126,lon:106.333},
  {id:"32497686304154672399743049842",lat:20.8632,lon:105.917},
  {id:"32496644961331232165577682920",lat:21.1044,lon:105.544},
  {id:"32497740397859994680382176275",lat:20.8084,lon:106.531},
  {id:"32497743881131035780167864566",lat:20.9915,lon:106.706},
  {id:"32497745839288232378111853921",lat:20.9618,lon:106.749},
  {id:"32496640019282169742177968358",lat:21.0346,lon:105.8},
  {id:"32496599090403802522844462004",lat:20.9615,lon:106.021},
  {id:"32496635793299427531930529746",lat:21.0188,lon:105.829},
  {id:"32496648232690046548195134965",lat:21.0833,lon:105.669},
  {id:"32496649796320141677664228873",lat:20.6957,lon:105.899},
  {id:"32496654659036040237005110994",lat:20.9242,lon:105.834},
  {id:"32496656699585544717774158962",lat:20.9777,lon:105.822},
  {id:"32496659849795661849369859648",lat:21.1959,lon:105.428},
  {id:"32497674830731204969056502556",lat:20.8707,lon:105.806},
  {id:"32497682732178779599258375030",lat:20.9662,lon:105.838},
  {id:"32497684349251147003185136481",lat:21.0233,lon:105.906},
  {id:"32497689422265798936416998264",lat:21.028,lon:105.783},
  {id:"32497693512462826621189636013",lat:21.0407,lon:105.816},
  {id:"32497698154816065885508975523",lat:21.0608,lon:105.749},
  {id:"32497715505350541081963258104",lat:20.8345,lon:106.221},
  {id:"32496615589638980218291071052",lat:20.9201,lon:106.138},
  {id:"32496646519546184688745583087",lat:21.0464,lon:105.802},
  {id:"32496653072616049897983672014",lat:21.0787,lon:105.945},
  {id:"32496658274959142730665276769",lat:21.0174,lon:105.697},
  {id:"32497687835845808597395559285",lat:21.1781,lon:105.729},
  {id:"32497691894129504813496914814",lat:21.1775,lon:105.839},
  {id:"32497713513553527537986241519",lat:21.0337,lon:106.597},
  {id:"32496578935958505757412009876",lat:20.892,lon:106.137},
  {id:"32496590242207758224773467210",lat:21.0104,lon:106.528},
  {id:"32496618056220078954907439411",lat:20.9178,lon:106.121},
  {id:"32496621120052989448308754893",lat:21.5761,lon:106.503},
  {id:"32496634971884394546566271369",lat:20.9378,lon:106.764},
  {id:"32496670530460480527200245359",lat:20.9562,lon:106.745},
  {id:"32496674426906599552653261962",lat:20.9399,lon:106.75},
  {id:"32496700750804852782040219324",lat:21.0062,lon:106.533},
  {id:"32497682375609562779368280919",lat:20.9867,lon:106.739},
  {id:"32497816774243732691318562431",lat:21.5581,lon:106.481},
  {id:"32497825983352144945122444532",lat:20.6748,lon:105.899},
  {id:"32497829304160538258698536470",lat:20.5831,lon:105.928},
  {id:"32497832479169427549528529636",lat:20.5535,lon:105.934},
  {id:"32497834813217474579076019998",lat:20.6415,lon:105.902},
  {id:"32497839438874550478493036568",lat:21.5683,lon:105.847},
  {id:"32497845274199713703151904970",lat:21.554,lon:105.842},
  {id:"US_EMBASSY_HAN",lat:21.0219,lon:105.8188},
  {id:"US_CONSULATE_HCM",lat:10.7769,lon:106.7009}
];

// ═══════════════════════════════════════════════════════════════
//  CONFIG
// ═══════════════════════════════════════════════════════════════
var DRIVE_FOLDER = 'gee_directional_123';
var START = '2023-01-01';
var END   = '2024-12-31';

var DIRS = {N:0, NE:45, E:90, SE:135, S:180, SW:225, W:270, NW:315};
var DISTS = [5, 10, 20]; // km

// ═══════════════════════════════════════════════════════════════
//  BUILD SAMPLING POINTS
// ═══════════════════════════════════════════════════════════════
function offsetPoint(lat, lon, bearingDeg, distKm) {
  var R = 6371.0;
  var d = distKm / R;
  var brng = bearingDeg * Math.PI / 180;
  var lat1 = lat * Math.PI / 180;
  var lon1 = lon * Math.PI / 180;
  var lat2 = Math.asin(Math.sin(lat1)*Math.cos(d) +
                       Math.cos(lat1)*Math.sin(d)*Math.cos(brng));
  var lon2 = lon1 + Math.atan2(Math.sin(brng)*Math.sin(d)*Math.cos(lat1),
                                Math.cos(d) - Math.sin(lat1)*Math.sin(lat2));
  return [lon2 * 180/Math.PI, lat2 * 180/Math.PI]; // [lon, lat] for GEE
}

var pointsList = [];
STATIONS.forEach(function(stn) {
  // Center point
  pointsList.push(ee.Feature(ee.Geometry.Point([stn.lon, stn.lat]), {
    stationId: stn.id, direction: 'C', distance_km: 0
  }));
  // 8 directions × 3 distances
  Object.keys(DIRS).forEach(function(dirName) {
    DISTS.forEach(function(dist) {
      var ll = offsetPoint(stn.lat, stn.lon, DIRS[dirName], dist);
      pointsList.push(ee.Feature(ee.Geometry.Point(ll), {
        stationId: stn.id, direction: dirName, distance_km: dist
      }));
    });
  });
});

var points = ee.FeatureCollection(pointsList);
print('Sampling points:', points.size()); // should be 3075

// ═══════════════════════════════════════════════════════════════
//  HELPER: stack monthly means into multi-band image
// ═══════════════════════════════════════════════════════════════
function monthlyStack(collection, band, prefix) {
  var images = [];
  for (var m = 1; m <= 12; m++) {
    var label = prefix + '_m' + (m < 10 ? '0' : '') + m;
    images.push(
      collection.filter(ee.Filter.calendarRange(m, m, 'month'))
        .select(band).mean().rename(label)
    );
  }
  return ee.Image.cat(images);
}

// ═══════════════════════════════════════════════════════════════
//  HELPER: stack seasonal means
// ═══════════════════════════════════════════════════════════════
function seasonalStack(collection, band, prefix) {
  var seasonDef = [
    {name: 'DJF', months: [12, 1, 2]},
    {name: 'MAM', months: [3, 4, 5]},
    {name: 'JJA', months: [6, 7, 8]},
    {name: 'SON', months: [9, 10, 11]}
  ];
  var images = [];
  seasonDef.forEach(function(s) {
    var filtered = collection.filter(ee.Filter.or(
      ee.Filter.calendarRange(s.months[0], s.months[0], 'month'),
      ee.Filter.calendarRange(s.months[1], s.months[1], 'month'),
      ee.Filter.calendarRange(s.months[2], s.months[2], 'month')
    ));
    images.push(filtered.select(band).mean().rename(prefix + '_' + s.name));
  });
  return ee.Image.cat(images);
}

// ═══════════════════════════════════════════════════════════════
//  1. TROPOMI NO2
// ═══════════════════════════════════════════════════════════════
var no2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
  .filterDate(START, END);
var no2Stack = monthlyStack(no2, 'tropospheric_NO2_column_number_density', 'no2');

var no2Sampled = no2Stack.sampleRegions({
  collection: points, scale: 1113.2, geometries: false
});
Export.table.toDrive({
  collection: no2Sampled,
  description: 'no2_directional_123',
  folder: DRIVE_FOLDER,
  fileFormat: 'CSV'
});

// ═══════════════════════════════════════════════════════════════
//  2. TROPOMI SO2
// ═══════════════════════════════════════════════════════════════
var so2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_SO2')
  .filterDate(START, END);
var so2Stack = monthlyStack(so2, 'SO2_column_number_density', 'so2');

var so2Sampled = so2Stack.sampleRegions({
  collection: points, scale: 1113.2, geometries: false
});
Export.table.toDrive({
  collection: so2Sampled,
  description: 'so2_directional_123',
  folder: DRIVE_FOLDER,
  fileFormat: 'CSV'
});

// ═══════════════════════════════════════════════════════════════
//  3. TROPOMI CO
// ═══════════════════════════════════════════════════════════════
var co = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CO')
  .filterDate(START, END);
var coStack = monthlyStack(co, 'CO_column_number_density', 'co');

var coSampled = coStack.sampleRegions({
  collection: points, scale: 1113.2, geometries: false
});
Export.table.toDrive({
  collection: coSampled,
  description: 'co_directional_123',
  folder: DRIVE_FOLDER,
  fileFormat: 'CSV'
});

// ═══════════════════════════════════════════════════════════════
//  4. TROPOMI HCHO
// ═══════════════════════════════════════════════════════════════
var hcho = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_HCHO')
  .filterDate(START, END);
var hchoStack = monthlyStack(hcho, 'tropospheric_HCHO_column_number_density', 'hcho');

var hchoSampled = hchoStack.sampleRegions({
  collection: points, scale: 1113.2, geometries: false
});
Export.table.toDrive({
  collection: hchoSampled,
  description: 'hcho_directional_123',
  folder: DRIVE_FOLDER,
  fileFormat: 'CSV'
});

// ═══════════════════════════════════════════════════════════════
//  5. MODIS LST (day + night, seasonal)
// ═══════════════════════════════════════════════════════════════
var lstCol = ee.ImageCollection('MODIS/061/MOD11A1')
  .filterDate(START, END);

// Scale factor: raw × 0.02 → Kelvin, then − 273.15 → Celsius
var lstDay  = lstCol.select('LST_Day_1km').map(function(img) {
  return img.multiply(0.02).subtract(273.15).copyProperties(img, ['system:time_start']);
});
var lstNight = lstCol.select('LST_Night_1km').map(function(img) {
  return img.multiply(0.02).subtract(273.15).copyProperties(img, ['system:time_start']);
});

var lstDayStack  = seasonalStack(lstDay,  'LST_Day_1km',   'lstDay');
var lstNightStack = seasonalStack(lstNight, 'LST_Night_1km', 'lstNight');
var lstStack = lstDayStack.addBands(lstNightStack);

var lstSampled = lstStack.sampleRegions({
  collection: points, scale: 1000, geometries: false
});
Export.table.toDrive({
  collection: lstSampled,
  description: 'lst_directional_123',
  folder: DRIVE_FOLDER,
  fileFormat: 'CSV'
});

// ═══════════════════════════════════════════════════════════════
//  6. VIIRS NIGHTLIGHTS (annual mean)
// ═══════════════════════════════════════════════════════════════
var ntl = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .filterDate('2023-01-01', '2023-12-31')
  .select('avg_rad')
  .mean()
  .rename('ntl_mean');

var ntlSampled = ntl.sampleRegions({
  collection: points, scale: 463.83, geometries: false
});
Export.table.toDrive({
  collection: ntlSampled,
  description: 'ntl_directional_123',
  folder: DRIVE_FOLDER,
  fileFormat: 'CSV'
});

// ═══════════════════════════════════════════════════════════════
//  7. MODIS FIRE (seasonal burned area count)
// ═══════════════════════════════════════════════════════════════
var fires = ee.ImageCollection('MODIS/061/MCD64A1')
  .filterDate(START, END)
  .select('BurnDate');
// Convert burn date to binary (burned=1, not=0), then seasonal sum
var fireBinary = fires.map(function(img) {
  return img.gt(0).selfMask().unmask(0)
    .copyProperties(img, ['system:time_start']);
});
var fireStack = seasonalStack(fireBinary, 'BurnDate', 'fire');

var fireSampled = fireStack.sampleRegions({
  collection: points, scale: 500, geometries: false
});
Export.table.toDrive({
  collection: fireSampled,
  description: 'fire_directional_123',
  folder: DRIVE_FOLDER,
  fileFormat: 'CSV'
});

// ═══════════════════════════════════════════════════════════════
print('Script loaded — go to Tasks tab and click Run on each export.');
print('All CSVs will appear in Google Drive folder: ' + DRIVE_FOLDER);
