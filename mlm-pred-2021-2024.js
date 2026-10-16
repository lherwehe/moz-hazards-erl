/**
 * GEE script: lag-split predictors for SHORTER-LONG-ENDLINE timing alternate (robustness).
 *
 * Exports one CSV per province with two rows per admin post (short, long).
 * This alternate keeps SHORT unchanged and sets LONG endline/hazard windows to:
 *   2021-11-01 .. 2024-04-01 (exclusive). Core panel uses long 2018-11-01 .. 2024-04-01 (exclusive).
 * Pop growth is fixed for both periods using WorldPop GP CAGR:
 *   baseline 2016-11-01..2019-03-14 and endline 2019-11-01..2021-01-01 (exclusive).
 * Includes hazard-window controls (SPEI, CHIRPS, VIIRS nighttime lights) by period.
 * Core SPEI drought: spei_cumulative_deficit / spei_endline_window = zonal mean of per-pixel
 * count of months with SPEI-6 < -1.0 in each endline window (CSV column names unchanged).
 *
 * Output naming:
 *   MOZ_MLM_predictors_lag_split_panel_2021_2024_<Province>.csv
 */

// ============== CONFIG ==============
var PROVINCES = ['Cabo Delgado', 'Nampula', 'Sofala', 'Gaza'];
var PROVINCE_FIELD = 'ADM1_PT';
var ADMIN3_NAME_FIELD = 'ADM3_PT';

var ADMIN3_ASSET = moz_adpo;
var GCS_BUCKET = 'moz-lulc';
var EXPORT_PREFIX = 'MOZ_MLM_predictors_lag_split_panel_2021_2024';

var BASELINE_START = '2016-11-01';
var BASELINE_END = '2019-03-14'; // exclusive (through 2019-03-13)
var POP_GROWTH_ENDLINE_START = '2019-11-01';
var POP_GROWTH_ENDLINE_END = '2021-01-01'; // exclusive (through 2020-12-31)

// Speed controls
var FAST_MODE = false;
var PROCESS_PROVINCE = null; // e.g. 'Gaza' or null for all
var RUN_PERIOD_KEYS = ['short', 'long'];
var INCLUDE_CHIRPS_SPLIT = true;

var REDUCE_SCALE = FAST_MODE ? 1000 : 500;
var TILE_SCALE = FAST_MODE ? 12 : 8;
var MAX_PIXELS_PER_REGION = 5e7;

var SPEI_CATALOG = 'CSIC/SPEI/2_10';
var SPEI_2024_GCS = 'gs://moz-lulc/spei6_v2.11_2024';
var SPEI_BAND = 'SPEI_06_month';
var SPEI_DATA_EXCLUSIVE_END = '2025-01-01'; // through 2024-12

var CHIRPS_R95P_REF_START = '2000-01-01';
var CHIRPS_R95P_REF_END = '2019-01-01'; // exclusive
var CHIRPS_WET_DAY_MM = 1;
var VIIRS_COLLECTION = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG';
var VIIRS_BAND = 'avg_rad';
// Fixed VIIRS window for both short/long rows.
var VIIRS_FIXED_START = '2016-11-01';
var VIIRS_FIXED_END_EXCLUSIVE = '2019-04-01'; // through 2019-03-31

var PERIODS = [
  {
    period_key: 'short',
    period_label: '2020-2021',
    hazard_start: '2018-11-01',
    hazard_end_exclusive: '2021-04-01',
    endline_window_start: '2018-11-01',
    endline_window_end_exclusive: '2021-04-01'
  },
  {
    period_key: 'long',
    period_label: '2023-2024',
    // Shorter long-endline window (timing sensitivity vs core 2018-11-01 .. 2024):
    hazard_start: '2021-11-01',
    hazard_end_exclusive: '2024-04-01',
    endline_window_start: '2021-11-01',
    endline_window_end_exclusive: '2024-04-01'
  }
];

var selectedProvinces = PROCESS_PROVINCE ? [PROCESS_PROVINCE] : PROVINCES;
var selectedPeriods = PERIODS.filter(function(p) {
  return RUN_PERIOD_KEYS.indexOf(p.period_key) >= 0;
});

// ============== DATA ==============
var adminAll = ee.FeatureCollection(ADMIN3_ASSET);
var adminPosts = adminAll
  .filter(ee.Filter.inList(PROVINCE_FIELD, selectedProvinces))
  .filter(ee.Filter.notNull([ADMIN3_NAME_FIELD]));
var studyBounds = adminPosts.geometry();

var haImg = ee.Image.pixelArea().divide(10000).rename('ha');

var popBaseline = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filterBounds(studyBounds)
  .filterDate('2016-01-01', '2021-01-01')
  .select('population')
  .filterDate(BASELINE_START, BASELINE_END)
  .mean()
  .rename('pop');

var popGpColl = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filterBounds(studyBounds)
  .filterDate('2016-01-01', '2021-01-01')
  .select('population');

var speiCatalog = ee.ImageCollection(SPEI_CATALOG)
  .filterDate('2019-03-01', '2024-01-01')
  .select(SPEI_BAND);
var spei2024 = ee.ImageCollection(ee.List.sequence(1, 12).map(function(m) {
  var mStr = ee.Number(m).format('%02d');
  var uri = ee.String(SPEI_2024_GCS).cat('/SPEI6_2024').cat(mStr).cat('01.tif');
  var img = ee.Image.loadGeoTIFF(uri).rename(SPEI_BAND);
  return img.set('system:time_start', ee.Date.fromYMD(2024, ee.Number(m), 1).millis());
}));
var speiFull = speiCatalog.merge(spei2024).sort('system:time_start');

var chirpsP95ref = null;
if (INCLUDE_CHIRPS_SPLIT) {
  var chirpsRefDaily = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterBounds(studyBounds)
    .filterDate(CHIRPS_R95P_REF_START, CHIRPS_R95P_REF_END)
    .select('precipitation');
  var chirpsRefWet = chirpsRefDaily.map(function(img) {
    var p = img.select('precipitation');
    return p.updateMask(p.gte(CHIRPS_WET_DAY_MM));
  });
  chirpsP95ref = chirpsRefWet.reduce(ee.Reducer.percentile([95]))
    .select('precipitation_p95')
    .rename('precipitation');
}

function capSpeiEnd(requestedExclusiveEnd) {
  var a = new Date(requestedExclusiveEnd + 'T00:00:00Z');
  var b = new Date(SPEI_DATA_EXCLUSIVE_END + 'T00:00:00Z');
  return a <= b ? requestedExclusiveEnd : SPEI_DATA_EXCLUSIVE_END;
}

function isNonEmptyWindow(startDateStr, endExclusiveStr) {
  return ee.Date(endExclusiveStr).millis().gt(ee.Date(startDateStr).millis());
}

function popMeanOrNearestPrior(coll, startDate, endExclusive) {
  var start = ee.Date(startDate);
  var endEx = ee.Date(endExclusive);
  var w = coll.filterDate(start, endEx);
  var prior = coll
    .filterDate('2000-01-01', endEx)
    .sort('system:time_start', false);
  return ee.Image(ee.Algorithms.If(
    w.size().gt(0),
    w.mean(),
    ee.Algorithms.If(
      prior.size().gt(0),
      ee.Image(prior.first()),
      ee.Image.constant(0).rename('population').toFloat()
    )
  )).rename('pop');
}

function midpointDate(startDate, endExclusive) {
  var start = ee.Date(startDate);
  var endEx = ee.Date(endExclusive);
  var halfDays = endEx.difference(start, 'day').divide(2.0);
  return start.advance(halfDays, 'day');
}

var popGrowthYears = midpointDate(POP_GROWTH_ENDLINE_START, POP_GROWTH_ENDLINE_END)
  .difference(midpointDate(BASELINE_START, BASELINE_END), 'year');

/** SPEI-6 drought threshold for counting dry months (SPEIbase / CSIC convention). */
var SPEI_DROUGHT_MONTH_THRESHOLD = -1.0;

/**
 * Core drought layer (still exported as band spei_cumulative_deficit for CSV compatibility):
 * per-pixel sum over the endline window of 1(SPEI-6 < threshold) each month, then zonal mean in reduceRegions.
 * Higher values = more frequent dry months at pixels in the admin post (larger = worse drought exposure).
 */
function speiCumulativeDeficit(startDate, endExclusive) {
  var cappedEnd = capSpeiEnd(endExclusive);
  return ee.Image(ee.Algorithms.If(
    isNonEmptyWindow(startDate, cappedEnd),
    speiFull.filterDate(startDate, cappedEnd).map(function(img) {
      return img.select(SPEI_BAND).lt(SPEI_DROUGHT_MONTH_THRESHOLD).rename(SPEI_BAND);
    }).sum(),
    ee.Image.constant(0).rename(SPEI_BAND).toFloat()
  ));
}

function speiMinValue(startDate, endExclusive) {
  var cappedEnd = capSpeiEnd(endExclusive);
  return ee.Image(ee.Algorithms.If(
    isNonEmptyWindow(startDate, cappedEnd),
    speiFull.filterDate(startDate, cappedEnd).select(SPEI_BAND).min(),
    ee.Image.constant(0).rename(SPEI_BAND).toFloat()
  ));
}

function chirpsR95pTotal(startDate, endExclusive) {
  if (!INCLUDE_CHIRPS_SPLIT) {
    return ee.Image.constant(0).rename('precipitation').toFloat();
  }
  return ee.Image(ee.Algorithms.If(
    isNonEmptyWindow(startDate, endExclusive),
    ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
      .filterBounds(studyBounds)
      .filterDate(startDate, endExclusive)
      .select('precipitation')
      .map(function(img) {
        var p = img.select('precipitation');
        return p.updateMask(p.gt(chirpsP95ref)).rename('precipitation');
      }).sum().add(1).log(),
    ee.Image.constant(0).rename('precipitation').toFloat()
  ));
}

function viirsMeanRadiance(startDate, endExclusive) {
  var coll = ee.ImageCollection(VIIRS_COLLECTION)
    .filterBounds(studyBounds)
    .filterDate(VIIRS_FIXED_START, VIIRS_FIXED_END_EXCLUSIVE)
    .select([VIIRS_BAND, 'cf_cvg']);

  var valid = coll.map(function(img) {
    var rad = img.select(VIIRS_BAND);
    var cvg = img.select('cf_cvg');
    // Keep only pixels with at least one cloud-free observation.
    return rad.updateMask(cvg.gt(0)).rename('viirs_endline_window');
  });

  return ee.Image(ee.Algorithms.If(
    isNonEmptyWindow(VIIRS_FIXED_START, VIIRS_FIXED_END_EXCLUSIVE),
    valid.mean().add(1).log().rename('viirs_endline_window'),
    // Empty window -> no observation (masked), not a forced zero.
    ee.Image.constant(0).rename('viirs_endline_window').selfMask().toFloat()
  ));
}

function periodStackedImage(periodCfg) {
  var gBaseline = popMeanOrNearestPrior(popGpColl, BASELINE_START, BASELINE_END);
  var gEndline = popMeanOrNearestPrior(popGpColl, POP_GROWTH_ENDLINE_START, POP_GROWTH_ENDLINE_END);

  var speiDeficit = speiCumulativeDeficit(periodCfg.endline_window_start, periodCfg.endline_window_end_exclusive)
    .rename('spei_cumulative_deficit');
  var speiMin = speiMinValue(periodCfg.endline_window_start, periodCfg.endline_window_end_exclusive)
    .rename('spei_min');
  var speiEndlineWindow = speiDeficit.rename('spei_endline_window');

  var chirpsEndlineWindow = chirpsR95pTotal(periodCfg.endline_window_start, periodCfg.endline_window_end_exclusive)
    .rename('chirps_endline_window');
  var viirsEndlineWindow = viirsMeanRadiance(periodCfg.endline_window_start, periodCfg.endline_window_end_exclusive)
    .rename('viirs_endline_window');

  return popBaseline.rename('pop_baseline')
    .addBands(gBaseline.rename('g_baseline'))
    .addBands(gEndline.rename('g_endline'))
    .addBands(haImg)
    .addBands(speiDeficit)
    .addBands(speiMin)
    .addBands(speiEndlineWindow)
    .addBands(chirpsEndlineWindow)
    .addBands(viirsEndlineWindow);
}

function periodPredictorFc(periodCfg, adminFc, stacked) {
  var reducer = ee.Reducer.sum().forEach([
    'pop_baseline', 'g_baseline', 'g_endline', 'ha'
  ])
    .combine(ee.Reducer.mean().forEach([
      'spei_endline_window', 'spei_cumulative_deficit', 'spei_min', 'chirps_endline_window', 'viirs_endline_window'
    ]), '', false);

  var reduced = stacked.reduceRegions({
    collection: adminFc,
    reducer: reducer,
    scale: REDUCE_SCALE,
    maxPixelsPerRegion: MAX_PIXELS_PER_REGION,
    tileScale: TILE_SCALE
  });

  var adminList = adminFc.toList(adminFc.size());
  var reducedList = reduced.toList(reduced.size());
  return ee.FeatureCollection(ee.List.sequence(0, adminFc.size().subtract(1)).map(function(j) {
    var a = ee.Feature(adminList.get(j));
    var r = ee.Feature(reducedList.get(j));

    var popSumGpBaseline = ee.Number(r.get('pop_baseline')).max(0);
    var popSumBaselineWindow = ee.Number(r.get('g_baseline')).max(0);
    var popSumEndlineWindow = ee.Number(r.get('g_endline')).max(0);
    var areaHa = ee.Number(r.get('ha')).max(0.01);
    var popGrowth = ee.Number(ee.Algorithms.If(
      popSumBaselineWindow.gt(0).and(popSumEndlineWindow.gt(0)),
      popSumEndlineWindow.divide(popSumBaselineWindow).pow(ee.Number(1).divide(popGrowthYears)).subtract(1),
      null
    ));

    return ee.Feature(null, {
      province: a.get(PROVINCE_FIELD),
      admin_post: a.get(ADMIN3_NAME_FIELD),
      period_key: periodCfg.period_key,
      period_label: periodCfg.period_label,
      hazard_start: periodCfg.hazard_start,
      hazard_end_exclusive: periodCfg.hazard_end_exclusive,
      endline_window_start: periodCfg.endline_window_start,
      endline_window_end_exclusive: periodCfg.endline_window_end_exclusive,
      pop_sum_baseline_window: popSumBaselineWindow,
      pop_sum_endline_window: popSumEndlineWindow,
      pop_growth: popGrowth,
      area_ha: areaHa,
      pop_density_period: popSumGpBaseline.divide(areaHa),
      spei_endline_window: r.get('spei_endline_window'),
      spei_cumulative_deficit: r.get('spei_cumulative_deficit'),
      spei_min: r.get('spei_min'),
      chirps_endline_window: r.get('chirps_endline_window'),
      viirs_endline_window: r.get('viirs_endline_window')
    });
  }));
}

var selectors = [
  'province', 'admin_post', 'period_key', 'period_label',
  'hazard_start', 'hazard_end_exclusive', 'endline_window_start', 'endline_window_end_exclusive',
  'pop_sum_baseline_window', 'pop_sum_endline_window', 'pop_growth', 'area_ha', 'pop_density_period',
  'spei_endline_window', 'spei_cumulative_deficit', 'spei_min', 'chirps_endline_window', 'viirs_endline_window'
];

var stackedByPeriod = {};
for (var p = 0; p < selectedPeriods.length; p++) {
  var cfgP = selectedPeriods[p];
  stackedByPeriod[cfgP.period_key] = periodStackedImage(cfgP);
}

for (var i = 0; i < selectedProvinces.length; i++) {
  var province = selectedProvinces[i];
  var adminOne = adminAll
    .filter(ee.Filter.eq(PROVINCE_FIELD, province))
    .filter(ee.Filter.notNull([ADMIN3_NAME_FIELD]));

  var out = ee.FeatureCollection([]);
  for (var k = 0; k < selectedPeriods.length; k++) {
    var cfg = selectedPeriods[k];
    var fc = periodPredictorFc(cfg, adminOne, stackedByPeriod[cfg.period_key]);
    out = out.merge(fc);
  }

  var safeName = province.replace(/\s/g, '_');
  Export.table.toCloudStorage({
    collection: out,
    description: EXPORT_PREFIX + '_' + safeName,
    bucket: GCS_BUCKET,
    fileNamePrefix: EXPORT_PREFIX + '_' + safeName,
    fileFormat: 'CSV',
    selectors: selectors
  });
}

print('ALT long-endline predictor export');
print('FAST_MODE:', FAST_MODE, '| REDUCE_SCALE:', REDUCE_SCALE, '| TILE_SCALE:', TILE_SCALE);
print('Selected periods:', selectedPeriods.map(function(x) { return x.period_key; }));
print('Selected provinces:', selectedProvinces);
print('Long alternate window: 2018-11-01..2024-04-01(excl) for hazard/endline (pop growth remains fixed GP CAGR).');
print('pop_growth: fixed WorldPop GP CAGR baseline', BASELINE_START, 'to', BASELINE_END, '(excl) and endline', POP_GROWTH_ENDLINE_START, 'to', POP_GROWTH_ENDLINE_END, '(excl), shared across short/long.');
print('Output prefix:', EXPORT_PREFIX, '-> gs://' + GCS_BUCKET + '/' + EXPORT_PREFIX + '_<Province>.csv');
print('Done. Run export tasks in GEE Tasks tab.');
