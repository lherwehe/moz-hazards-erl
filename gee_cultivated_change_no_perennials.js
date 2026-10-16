/**** CONFIG ****/
// Behavior-preserving streamlined script.
// Primary outputs:
// 1) Admin-post CSV with probability-weighted change metrics (+ optional threshold classes)
// 2) Province multiband GeoTIFF (base/end/delta probability + optional threshold classes)
// 3) Optional validation KML (stratified by threshold classes)
//
// Typical run knobs:
// - PROCESS_PROVINCE: one province per run
// - ENDLINE_YEARS: [2020,2021,2022] or [2023,2024,2025]
// - USE_*_MASK toggles
// - USE_PROB_THRESHOLD / INCLUDE_OBS_METRICS / INCLUDE_VALIDATION_EXPORTS
var SEASON_START_MMDD = [11, 1];
var SEASON_END_MMDD   = [3, 31];
var END_2019_MMDD     = [3, 13];

var ADMIN3_ASSET = moz_adpo;
// IBTrACS table asset: add Import in GEE Code Editor → Assets → CYCLONE_TRACKS_ASSET
var PROVINCE_FIELD = 'ADM1_PT';
var ADMIN3_NAME_FIELD = 'ADM3_PT';

var GCS_BUCKET = 'moz-lulc';
var FILE_PREFIX = 'MOZ_cultivated_change';

var PROCESS_PROVINCE = 'Cabo Delgado';
var ENDLINE_YEARS = [2020, 2021, 2022];

var STATS_SCALE_METERS = 10;
var TILE_SCALE = 4;

var FAST_CYCLONE_MODE = true;
var CYCLONE_STATS_SCALE_METERS = 30;
var CYCLONE_TILE_SCALE = 8;

var USE_PROB_THRESHOLD = false;
var PROB_THRESHOLD = 0.15;

var INCLUDE_OBS_METRICS = false;
var INCLUDE_VALIDATION_EXPORTS = false;

var USE_FLOODPLAIN_MASK = false;

var USE_CYCLONE_MASK =  false;
var CYCLONE_BUFFER_KM = 50;
var CYCLONE_WIND_KNOTS = 34;
var CYCLONE_BUFFER_MAX_ERROR_METERS = 2000;
var CYCLONE_STORM_ID_COLUMNS = ['SID', 'STORM_ID', 'USA_ATCF_ID', 'NAME'];
var CYCLONE_TIME_COLUMNS = ['ISO_TIME', 'iso_time'];
var CYCLONE_LAT_FALLBACK_COLUMNS = ['LAT', 'lat', 'USA_LAT', 'usa_lat'];
var CYCLONE_LON_FALLBACK_COLUMNS = ['LON', 'lon', 'USA_LON', 'usa_lon'];

var USE_CONFLICT_MASK = false;
var CONFLICT_EVENTS_ASSET = CONFLICT_EVENTS;
var CONFLICT_COL_ADMIN1 = 'admin1';
var CONFLICT_COL_ADMIN3 = 'admin3';
var CONFLICT_COL_YEAR = 'year';
var CONFLICT_COL_EVENT_DATE = 'event_date';
var CONFLICT_COL_LAT = 'latitude';
var CONFLICT_COL_LON = 'longitude';
var CONFLICT_COL_FATALITIES = 'fatalities';

/**** ADMIN GEOMETRIES - SINGLE PROVINCE ****/
var adminAll   = ee.FeatureCollection(ADMIN3_ASSET);
var provinces  = adminAll.filter(ee.Filter.eq(PROVINCE_FIELD, PROCESS_PROVINCE));
var adminPosts = provinces.filter(ee.Filter.notNull([ADMIN3_NAME_FIELD]));
var studyGeom = provinces.geometry();

/**** DATASETS - MINIMAL FOR SPEED ****/
var lastEndlineYear = ENDLINE_YEARS[ENDLINE_YEARS.length - 1];
var MASK_EXPOSURE_START_SHORT = '2018-11-01';
var MASK_EXPOSURE_END_SHORT = '2021-03-31';
var MASK_EXPOSURE_START_LONG = '2021-11-01';
var MASK_EXPOSURE_END_LONG = '2024-03-31';
var MASK_EXPOSURE_START = lastEndlineYear <= 2022 ? MASK_EXPOSURE_START_SHORT : MASK_EXPOSURE_START_LONG;
var MASK_EXPOSURE_END = lastEndlineYear <= 2022 ? MASK_EXPOSURE_END_SHORT : MASK_EXPOSURE_END_LONG;
var MASK_EXPOSURE_START_DATE = ee.Date(MASK_EXPOSURE_START);
var MASK_EXPOSURE_END_EXCLUSIVE_DATE = ee.Date(MASK_EXPOSURE_END).advance(1, 'day');

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(studyGeom)
  .filterDate('2016-01-01', (lastEndlineYear + 1) + '-12-31')
  .select(['crops']);

var wcModels = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100')
  .filter(ee.Filter.eq('product', 'temporarycrops'));
var worldcereal2021 = ee.Image(ee.Algorithms.If(
  wcModels.size().gt(0),
  wcModels.mosaic().select('classification').eq(100),
  ee.Image.constant(0)
)).rename('wcereal').toByte();

/// Part 1: Buffers
var floodplainMask = null;
if (USE_FLOODPLAIN_MASK) {
  var GFPLAIN_ASSET = gfplain;
  var gfplainImg = ee.Image(GFPLAIN_ASSET).select(0).rename('gfplain');
  floodplainMask = gfplainImg.unmask(999).eq(0);
} else {
  floodplainMask = ee.Image.constant(1).rename('mask');
}

var cycloneMask = null;
if (USE_CYCLONE_MASK && CYCLONE_TRACKS_ASSET) {
  var cycloneSearchGeom = studyGeom.buffer(CYCLONE_BUFFER_KM * 1000);
  var ibtracsAll = ee.FeatureCollection(CYCLONE_TRACKS_ASSET);
  function cyclonePropToNumberOrNull(f, p) {
    var v = f.get(p);
    var flat = ee.List([v])
      .flatten()
      .flatten()
      .flatten()
      .flatten()
      .flatten()
      .flatten();
    var first = ee.Algorithms.If(flat.size().gt(0), flat.get(0), null);
    var asStr = ee.String(ee.Algorithms.If(ee.Algorithms.IsEqual(first, null), '', first));
    return ee.Algorithms.If(
      asStr.length().eq(0),
      null,
      ee.Number(first)
    );
  }
  function cyclonePropToStringOrEmpty(f, p) {
    var v = f.get(p);
    var flat = ee.List([v])
      .flatten()
      .flatten()
      .flatten()
      .flatten()
      .flatten()
      .flatten();
    var first = ee.Algorithms.If(flat.size().gt(0), flat.get(0), null);
    return ee.String(ee.Algorithms.If(
      ee.Algorithms.IsEqual(first, null),
      '',
      first
    ));
  }
  function firstCycloneCoord(f, columns) {
    var expr = null;
    for (var i = 0; i < columns.length; i++) {
      var val = cyclonePropToNumberOrNull(f, columns[i]);
      expr = ee.Algorithms.If(
        ee.Algorithms.IsEqual(expr, null),
        val,
        expr
      );
    }
    return expr;
  }
  function firstCycloneTimeString(f) {
    var expr = ee.String('');
    for (var i = 0; i < CYCLONE_TIME_COLUMNS.length; i++) {
      var val = cyclonePropToStringOrEmpty(f, CYCLONE_TIME_COLUMNS[i]);
      expr = ee.String(ee.Algorithms.If(val.length().gt(0), val, expr));
    }
    return expr;
  }
  function parseCycloneTimeToMillis(rawTime) {
    rawTime = ee.String(rawTime);
    var token0 = ee.String(ee.List(rawTime.split(' ')).get(0));
    var datePart = ee.String(ee.Algorithms.If(
      token0.index('T').gte(0),
      ee.String(ee.List(token0.split('T')).get(0)),
      token0
    ));
    return ee.Number(ee.Algorithms.If(
      datePart.index('/').gte(0),
      ee.Date.parse('M/d/YYYY', datePart).millis(),
      ee.Date.parse('YYYY-MM-dd', datePart).millis()
    ));
  }
  var ibtracsWithGeom = ibtracsAll.map(function(f) {
    f = ee.Feature(f);
    var g = f.geometry();
    var hasPointGeom = ee.Algorithms.If(
      ee.Algorithms.IsEqual(g, null),
      false,
      ee.Geometry(g).type().compareTo(ee.String('Point')).eq(0)
    );
    var lonVal = firstCycloneCoord(f, CYCLONE_LON_FALLBACK_COLUMNS);
    var latVal = firstCycloneCoord(f, CYCLONE_LAT_FALLBACK_COLUMNS);
    var hasLatLon = ee.Algorithms.If(
      ee.Algorithms.IsEqual(lonVal, null),
      false,
      ee.Algorithms.If(ee.Algorithms.IsEqual(latVal, null), false, true)
    );
    var pointGeom = ee.Algorithms.If(
      hasPointGeom,
      g,
      ee.Algorithms.If(
        hasLatLon,
        ee.Geometry.Point(ee.Number(lonVal), ee.Number(latVal)),
        null
      )
    );
    var rawTime = firstCycloneTimeString(f);
    var tMs = ee.Algorithms.If(
      rawTime.length().gt(0),
      parseCycloneTimeToMillis(rawTime),
      null
    );
    return f
      .setGeometry(pointGeom)
      .set('_lat', latVal)
      .set('_lon', lonVal)
      .set('_event_ms', tMs)
      .set('_has_point_geom', ee.Algorithms.If(hasPointGeom, 1, ee.Algorithms.If(hasLatLon, 1, 0)));
  }).filter(ee.Filter.eq('_has_point_geom', 1));

  var ibtracs = ibtracsWithGeom.filterBounds(cycloneSearchGeom)
    .filter(ee.Filter.notNull(['_event_ms']))
    .filter(ee.Filter.gte('_event_ms', MASK_EXPOSURE_START_DATE.millis()))
    .filter(ee.Filter.lt('_event_ms', MASK_EXPOSURE_END_EXCLUSIVE_DATE.millis()))
    .map(function(f) {
      var c = ee.List(ee.Feature(f).geometry().coordinates());
      return ee.Feature(f)
        .set('_lon', ee.Number(c.get(0)))
        .set('_lat', ee.Number(c.get(1)));
    });
  var WIND_COLUMNS = ['WMO_WIND', 'USA_WIND', 'TOKYO_WIND', 'CMA_WIND', 'HKO_WIND', 'KMA_WIND',
    'NEWDELHI_WIND', 'REUNION_WIND', 'BOM_WIND', 'NADI_WIND', 'WELLINGTON_WIND',
    'DS824_WIND', 'TD9636_WIND', 'TD9635_WIND', 'NEUMANN_WIND', 'MLC_WIND'];
  function firstWindValue(f) {
    var expr = null;
    for (var i = WIND_COLUMNS.length - 1; i >= 0; i--) {
      var col = WIND_COLUMNS[i];
      var val = ee.Algorithms.If(
        ee.Algorithms.IsEqual(f.get(col), null),
        null,
        cyclonePropToNumberOrNull(f, col)
      );
      if (expr === null) {
        expr = val;
      } else {
        expr = ee.Algorithms.If(
          ee.Algorithms.IsEqual(val, null),
          expr,
          ee.Algorithms.If(ee.Number(val).gte(0), val, expr)
        );
      }
    }
    return expr;
  }
  function firstStormId(f) {
    var expr = ee.String('');
    for (var i = 0; i < CYCLONE_STORM_ID_COLUMNS.length; i++) {
      var col = CYCLONE_STORM_ID_COLUMNS[i];
      var val = cyclonePropToStringOrEmpty(f, col);
      expr = ee.String(ee.Algorithms.If(
        expr.length().gt(0),
        expr,
        ee.Algorithms.If(val.length().gt(0), val, expr)
      ));
    }
    var fallback = ee.String('pt_').cat(ee.Number(f.get('_event_ms')).format('%.0f'));
    return ee.String(ee.Algorithms.If(expr.length().gt(0), expr, fallback));
  }
  var withWind = ibtracs.map(function(f) {
    return f
      .set('wind', firstWindValue(f))
      .set('_storm_key', firstStormId(f));
  });
  var cycloneStrength = withWind.filter(ee.Filter.gte('wind', CYCLONE_WIND_KNOTS));
  var stormKeys = ee.List(cycloneStrength.aggregate_array('_storm_key')).distinct();
  var cycloneBuffersAll = ee.FeatureCollection(stormKeys.map(function(key) {
    key = ee.String(key);
    var stormPts = cycloneStrength
      .filter(ee.Filter.eq('_storm_key', key))
      .sort('_event_ms');
    var nPts = stormPts.size();
    var stormList = stormPts.toList(nPts);
    var coords = stormList.map(function(f) {
      f = ee.Feature(f);
      var c = ee.List(f.geometry().coordinates());
      return ee.List([ee.Number(c.get(0)), ee.Number(c.get(1))]);
    });
    var trackGeom = ee.Algorithms.If(
      nPts.gte(2),
      ee.Geometry.LineString(coords, null, true),
      ee.Geometry.Point(ee.List(coords.get(0)))
    );
    return ee.Feature(ee.Geometry(trackGeom))
      .set('_storm_key', key)
      .set('_n_pts', nPts)
      .buffer(CYCLONE_BUFFER_KM * 1000, CYCLONE_BUFFER_MAX_ERROR_METERS);
  }));
  var cycloneBuffers = cycloneBuffersAll.filterBounds(studyGeom);
  // Clip dissolved buffer geometry (more reliable than paint on large FCs).
  var cycloneFootprint = cycloneBuffers.geometry().dissolve(CYCLONE_BUFFER_MAX_ERROR_METERS);
  cycloneMask = ee.Image(ee.Algorithms.If(
    cycloneBuffers.size().gt(0),
    ee.Image(1).toByte().clip(cycloneFootprint).clip(studyGeom).rename('cyclone'),
    ee.Image(0).toByte().rename('cyclone').clip(studyGeom).updateMask(ee.Image(0))
  ));
} else if (USE_CYCLONE_MASK) {
  cycloneMask = ee.Image.constant(1).rename('mask');
} else {
  cycloneMask = ee.Image.constant(1).rename('mask');
}

var conflictMask = ee.Image.constant(1).rename('mask');
if (USE_CONFLICT_MASK) {
  var eventsRaw = ee.FeatureCollection(CONFLICT_EVENTS_ASSET).filterBounds(studyGeom);
  eventsRaw = eventsRaw.filter(ee.Filter.eq(CONFLICT_COL_ADMIN1, PROCESS_PROVINCE));
  var eventsPts = eventsRaw.map(function (f) {
    f = ee.Feature(f);
    var geom = f.geometry();
    var isPoint = geom.type().compareTo(ee.String('Point')).eq(0);

    function propToNumber(p) {
      var v = f.get(p);
      var flat = ee.List([v])
        .flatten()
        .flatten()
        .flatten()
        .flatten()
        .flatten()
        .flatten();
      return ee.Number(ee.Array(flat).get([0]));
    }

    function propToString(p) {
      var v = f.get(p);
      var flat = ee.List([v])
        .flatten()
        .flatten()
        .flatten()
        .flatten()
        .flatten()
        .flatten();
      var first = ee.Algorithms.If(flat.size().gt(0), flat.get(0), '');
      return ee.String(first);
    }

    var pointGeom = ee.Algorithms.If(
      isPoint,
      ee.Geometry.Point(geom.coordinates()),
      ee.Geometry.Point(propToNumber(CONFLICT_COL_LON), propToNumber(CONFLICT_COL_LAT))
    );

    var yr = propToNumber(CONFLICT_COL_YEAR);
    var fat = propToNumber(CONFLICT_COL_FATALITIES);
    var adm3 = propToString(CONFLICT_COL_ADMIN3);
    var eventDateStr = propToString(CONFLICT_COL_EVENT_DATE);
    var eventMs = ee.Algorithms.If(
      eventDateStr.length().gt(0),
      ee.Algorithms.If(
        eventDateStr.index('/').gte(0),
        ee.Date.parse('M/d/YYYY', eventDateStr).millis(),
        ee.Date.parse('YYYY-MM-dd', eventDateStr).millis()
      ),
      null
    );

    return f
      .setGeometry(pointGeom)
      .set(CONFLICT_COL_YEAR, yr)
      .set(CONFLICT_COL_FATALITIES, fat)
      .set(CONFLICT_COL_ADMIN3, adm3)
      .set('_event_ms', eventMs);
  });
  var eventsFiltered = eventsPts
    .filter(ee.Filter.notNull(['_event_ms']))
    .filter(ee.Filter.gte('_event_ms', MASK_EXPOSURE_START_DATE.millis()))
    .filter(ee.Filter.lt('_event_ms', MASK_EXPOSURE_END_EXCLUSIVE_DATE.millis()));

  var eventsWithAdm = eventsFiltered.filter(ee.Filter.notNull([CONFLICT_COL_ADMIN3]));
  var distinctAdm = eventsWithAdm.distinct(CONFLICT_COL_ADMIN3);
  var admNameList = distinctAdm.aggregate_array(CONFLICT_COL_ADMIN3);
  var matchedPosts = adminPosts.filter(ee.Filter.inList(ADMIN3_NAME_FIELD, admNameList));
  conflictMask = ee.Image(ee.Algorithms.If(
    matchedPosts.size().gt(0),
    ee.Image(0).byte()
      .paint(matchedPosts, 1)
      .selfMask()
      .clip(studyGeom),
    ee.Image.constant(1).clip(studyGeom)
  )).rename('conflict');
}

var analysisMask = ee.Image.constant(1);
if (USE_FLOODPLAIN_MASK) {
  analysisMask = floodplainMask;
}
if (USE_CYCLONE_MASK && CYCLONE_TRACKS_ASSET) {
  // Cyclone-only runs: use cyclone mask directly (not constant(1).updateMask(...)).
  analysisMask = USE_FLOODPLAIN_MASK
    ? analysisMask.updateMask(cycloneMask)
    : cycloneMask;
}
if (USE_CONFLICT_MASK) {
  analysisMask = analysisMask.updateMask(conflictMask);
}
var USE_ANY_SPATIAL_MASK = USE_FLOODPLAIN_MASK || (USE_CYCLONE_MASK && CYCLONE_TRACKS_ASSET) || USE_CONFLICT_MASK;
var EFFECTIVE_STATS_SCALE_METERS = (USE_CYCLONE_MASK && FAST_CYCLONE_MODE) ? CYCLONE_STATS_SCALE_METERS : STATS_SCALE_METERS;
var EFFECTIVE_TILE_SCALE = (USE_CYCLONE_MASK && FAST_CYCLONE_MODE) ? CYCLONE_TILE_SCALE : TILE_SCALE;

var worldcereal2021_masked = USE_ANY_SPATIAL_MASK
  ? worldcereal2021.updateMask(analysisMask)
  : worldcereal2021;

  /// Part 2: Seasonal Stats
/**** HELPERS ****/
var ha = ee.Image.pixelArea().divide(10000); // hectares
var ENDLINE_SUFFIX = '_' + ENDLINE_YEARS[0] + '-' + ENDLINE_YEARS[ENDLINE_YEARS.length - 1];
function dwSeasonalStats(seasonYear) {
  seasonYear = ee.Number(seasonYear);

  var start = ee.Date.fromYMD(seasonYear.subtract(1), SEASON_START_MMDD[0], SEASON_START_MMDD[1]);
  var endDefault = ee.Date.fromYMD(seasonYear, SEASON_END_MMDD[0], SEASON_END_MMDD[1]);
  var end = ee.Date(ee.Algorithms.If(
    seasonYear.eq(2019),
    ee.Date.fromYMD(2019, END_2019_MMDD[0], END_2019_MMDD[1]),
    endDefault
  ));

  var collSeason = dw.filterDate(start, end);
  var hasSeason = collSeason.size().gt(0);

  var p = ee.Image(ee.Algorithms.If(
    hasSeason, collSeason.select('crops').mean(), ee.Image.constant(0)
  )).rename('p').toFloat();

  var obs = collSeason.select('crops').count().rename('obs');
  var valid = obs.gte(1).rename('valid').toByte();

  return p.addBands([valid, obs]).set('seasonYear', seasonYear);
}

var s2017 = dwSeasonalStats(2017);
var s2018 = dwSeasonalStats(2018);
var s2019 = dwSeasonalStats(2019);

var endlineSeasons = ENDLINE_YEARS.map(function(year) { return dwSeasonalStats(year); });
var sEndline0 = endlineSeasons[0];
var sEndline1 = endlineSeasons[1];
var sEndline2 = endlineSeasons[2];

function windowStats(pImgsArray) {
  var stack = ee.ImageCollection(pImgsArray).toBands();
  var n = stack.mask().reduce(ee.Reducer.sum()).rename('n');
  var sum = stack.reduce(ee.Reducer.sum()).rename('sum');
  var sumsq = stack.multiply(stack).reduce(ee.Reducer.sum()).rename('sumsq');

  var mean = sum.divide(n.max(1)).rename('mu');
  var varNumer = sumsq.subtract(mean.multiply(mean).multiply(n));
  var denom = n.subtract(1).max(1);
  var sampleVar = varNumer.divide(denom).updateMask(n.gte(2));
  var varMean = sampleVar.divide(n.max(1)).rename('var_mean');
  
  return {
    mu: mean.updateMask(n.gte(1)),
    varMean: varMean,
    n: n
  };
}

function pMasked(seasonImg) {
  var p = seasonImg.select('p').updateMask(seasonImg.select('valid'));
  return USE_ANY_SPATIAL_MASK ? p.updateMask(analysisMask) : p;
}

var wsBase  = windowStats([pMasked(s2017), pMasked(s2018), pMasked(s2019)]);
var wsShort = windowStats([pMasked(sEndline0), pMasked(sEndline1), pMasked(sEndline2)]);

var baseProb       = wsBase.mu.rename('base_prob');
var endProb  = wsShort.mu.rename('end_prob');
if (USE_ANY_SPATIAL_MASK) {
  baseProb = baseProb.updateMask(analysisMask);
  endProb = endProb.updateMask(analysisMask);
}

var deltaProb = endProb.subtract(baseProb).rename('delta_prob');
var probGain = deltaProb.updateMask(deltaProb.gt(0)).rename('prob_gain');
var probLoss = deltaProb.updateMask(deltaProb.lt(0)).multiply(-1).rename('prob_loss');
var baseVarMean = wsBase.varMean.rename('base_var_mean');
var endVarMean  = wsShort.varMean.rename('end_var_mean');
if (USE_ANY_SPATIAL_MASK) {
  baseVarMean = baseVarMean.updateMask(analysisMask);
  endVarMean = endVarMean.updateMask(analysisMask);
}

/**** PROBABILITY THRESHOLD-BASED CLASSIFICATION ****/
var v2017 = s2017.select('valid'), v2018 = s2018.select('valid'), v2019 = s2019.select('valid');
var vEndline0 = sEndline0.select('valid'), vEndline1 = sEndline1.select('valid'), vEndline2 = sEndline2.select('valid');

var validBase  = v2017.add(v2018).add(v2019);
var validShort = vEndline0.add(vEndline1).add(vEndline2);

var validBothShort = validBase.gte(2).and(validShort.gte(2));
if (USE_ANY_SPATIAL_MASK) {
  validBothShort = validBothShort.and(analysisMask);
}

var baselineCult, endlineCult, gain, loss, st_c, st_u;
if (USE_PROB_THRESHOLD) {
  // Create binary cultivated masks from probability threshold
  // Pixels with crop probability >= PROB_THRESHOLD are classified as cultivated
  baselineCult = baseProb.gte(PROB_THRESHOLD).updateMask(validBase.gte(2)).toByte().rename('base_cult');
  endlineCult = endProb.gte(PROB_THRESHOLD).updateMask(validShort.gte(2)).toByte().rename('end_cult');

  // Apply floodplain mask if enabled
  if (USE_ANY_SPATIAL_MASK) {
    baselineCult = baselineCult.updateMask(analysisMask);
    endlineCult = endlineCult.updateMask(analysisMask);
  }

  // Calculate gain, loss, and stable categories
  // Gain: not cultivated in baseline, cultivated in endline
  gain = baselineCult.eq(0).and(endlineCult.eq(1)).updateMask(validBothShort).toByte().rename('gain');
  // Loss: cultivated in baseline, not cultivated in endline
  loss = baselineCult.eq(1).and(endlineCult.eq(0)).updateMask(validBothShort).toByte().rename('loss');
  // Stable cultivated: cultivated in both
  st_c = baselineCult.eq(1).and(endlineCult.eq(1)).updateMask(validBothShort).toByte().rename('stable_cult');
  // Stable uncultivated: not cultivated in both
  st_u = baselineCult.eq(0).and(endlineCult.eq(0)).updateMask(validBothShort).toByte().rename('stable_uncult');
}

/**** ADMIN-POST REDUCTION ****/

// Convert admin posts to list for iteration
var adminList = adminPosts.toList(adminPosts.size());

// Function to process each admin post individually
function processAdminPost(f) {
  f = ee.Feature(f);
  var geom = f.geometry();
  var adminName = f.get(ADMIN3_NAME_FIELD);
  var provinceName = f.get(PROVINCE_FIELD);
  
  // Helper function to safely get values from reduceRegion results
  function safeReduceRegion(img, bandName) {
    var result = img.reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: geom,
      scale: EFFECTIVE_STATS_SCALE_METERS,
      tileScale: EFFECTIVE_TILE_SCALE,
      maxPixels: 1e9
    });
    
    var value = ee.Algorithms.If(
      result.contains(bandName),
      result.get(bandName),
      0
    );
    
    return ee.Number(value);
  }
  var ha_gain = ee.Number(0);
  var ha_loss = ee.Number(0);
  var ha_stable_cult = ee.Number(0);
  var ha_stable_uncult = ee.Number(0);
  if (USE_PROB_THRESHOLD) {
    ha_gain = safeReduceRegion(gain.multiply(ha).rename('gain'), 'gain');
    ha_loss = safeReduceRegion(loss.multiply(ha).rename('loss'), 'loss');
    ha_stable_cult = safeReduceRegion(st_c.multiply(ha).rename('stable_cult'), 'stable_cult');
    ha_stable_uncult = safeReduceRegion(st_u.multiply(ha).rename('stable_uncult'), 'stable_uncult');
  }
  
  var ha_prob_baseline = safeReduceRegion(baseProb.multiply(ha), 'base_prob');
  var ha_prob_endline = safeReduceRegion(endProb.multiply(ha), 'end_prob');
  var ha_prob_change = safeReduceRegion(deltaProb.multiply(ha), 'delta_prob');
  var ha_prob_gain = safeReduceRegion(probGain.multiply(ha), 'prob_gain');
  var ha_prob_loss = safeReduceRegion(probLoss.multiply(ha), 'prob_loss');
  // Variance of expected area: Σ ha^2 * Var(μ̂)
  var ha_prob_var_baseline = safeReduceRegion(baseVarMean.multiply(ha.pow(2)), 'base_var_mean');
  var ha_prob_var_endline  = safeReduceRegion(endVarMean.multiply(ha.pow(2)),  'end_var_mean');
  var ha_wcereal21 = safeReduceRegion(worldcereal2021_masked.multiply(ha), 'wcereal');
  
  var mean_obs_base = ee.Number(0), mean_obs_end = ee.Number(0), frac_pix_obs2_base = ee.Number(0), frac_pix_obs2_end = ee.Number(0);
  var mean_obs_2017 = ee.Number(0), mean_obs_2018 = ee.Number(0), mean_obs_2019 = ee.Number(0);
  var mean_obs_endline0 = ee.Number(0), mean_obs_endline1 = ee.Number(0), mean_obs_endline2 = ee.Number(0);
  
  if (INCLUDE_OBS_METRICS) {
    function meanObsPerSeason(seasonImg) {
      var obs = seasonImg.select('obs').updateMask(seasonImg.select('valid'));
      var result = obs.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: geom,
        scale: EFFECTIVE_STATS_SCALE_METERS,
        tileScale: EFFECTIVE_TILE_SCALE,
        maxPixels: 1e9
      });
      return ee.Number(result.get('obs', 0));
    }
    
    mean_obs_2017 = meanObsPerSeason(s2017);
    mean_obs_2018 = meanObsPerSeason(s2018);
    mean_obs_2019 = meanObsPerSeason(s2019);
    mean_obs_endline0 = meanObsPerSeason(sEndline0);
    mean_obs_endline1 = meanObsPerSeason(sEndline1);
    mean_obs_endline2 = meanObsPerSeason(sEndline2);
    
    mean_obs_base = mean_obs_2017.add(mean_obs_2018).add(mean_obs_2019).divide(3);
    mean_obs_end  = mean_obs_endline0.add(mean_obs_endline1).add(mean_obs_endline2).divide(3);
    
    function fracObs2PerSeason(seasonImg) {
      var obs = seasonImg.select('obs').updateMask(seasonImg.select('valid'));
      var obs2 = obs.gte(2).toByte();
      var result = obs2.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: geom,
        scale: EFFECTIVE_STATS_SCALE_METERS,
        tileScale: EFFECTIVE_TILE_SCALE,
        maxPixels: 1e9
      });
      return ee.Number(result.get('obs', 0));
    }
    
    var frac2017 = fracObs2PerSeason(s2017);
    var frac2018 = fracObs2PerSeason(s2018);
    var frac2019 = fracObs2PerSeason(s2019);
    var fracEndline0 = fracObs2PerSeason(sEndline0);
    var fracEndline1 = fracObs2PerSeason(sEndline1);
    var fracEndline2 = fracObs2PerSeason(sEndline2);
    
    frac_pix_obs2_base = frac2017.add(frac2018).add(frac2019).divide(3);
    frac_pix_obs2_end  = fracEndline0.add(fracEndline1).add(fracEndline2).divide(3);
  }
  
  var ha_area_img = USE_ANY_SPATIAL_MASK
    ? ha.updateMask(analysisMask)
    : ha;
  var ha_admin_result = ha_area_img.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: geom,
    scale: EFFECTIVE_STATS_SCALE_METERS,
    tileScale: EFFECTIVE_TILE_SCALE,
    maxPixels: 1e9
  });
  
  var ha_admin = ee.Number(ee.List(ha_admin_result.values()).get(0));

  var nBase = wsBase.n;
  var nEnd = wsShort.n;
  var maskObsBase = nBase.gte(2);
  var maskObsEnd = nEnd.gte(2);
  var maskObsBoth = maskObsBase.and(maskObsEnd);

  if (USE_ANY_SPATIAL_MASK) {
    maskObsBase = maskObsBase.and(analysisMask);
    maskObsEnd = maskObsEnd.and(analysisMask);
    maskObsBoth = maskObsBoth.and(analysisMask);
  }
  
  var ha_observed_baseline_img = ha.updateMask(maskObsBase).rename('obs_base');
  var ha_observed_baseline = safeReduceRegion(ha_observed_baseline_img, 'obs_base');
  
  var ha_observed_endline_img = ha.updateMask(maskObsEnd).rename('obs_end');
  var ha_observed_endline = safeReduceRegion(ha_observed_endline_img, 'obs_end');
  
  var ha_observed_both_img = ha.updateMask(maskObsBoth).rename('obs_both');
  var ha_observed_both = safeReduceRegion(ha_observed_both_img, 'obs_both');

  var share_prob_baseline = ee.Number(ee.Algorithms.If(
    ha_observed_baseline.gt(0), 
    ha_prob_baseline.divide(ha_observed_baseline), 
    0
  ));
  
  var share_prob_endline = ee.Number(ee.Algorithms.If(
    ha_observed_endline.gt(0), 
    ha_prob_endline.divide(ha_observed_endline), 
    0
  ));
  
  var share_prob_change = ee.Number(ee.Algorithms.If(
    ha_observed_both.gt(0), 
    ha_prob_change.divide(ha_observed_both), 
    0
  ));
  var ha_prob_baseline_se = ha_prob_var_baseline.sqrt();
  var ha_prob_endline_se  = ha_prob_var_endline.sqrt();
  
  var se_prob_change_ha = ha_prob_baseline_se.pow(2).add(ha_prob_endline_se.pow(2)).sqrt();

  var se_prob_change_share = ee.Number(ee.Algorithms.If(
    ha_observed_both.gt(0),
    se_prob_change_ha.divide(ha_observed_both),
    ee.Number(0)
  ));
  
  var ci95_lo_chg_share = share_prob_change.subtract(ee.Number(1.96).multiply(se_prob_change_share));
  var ci95_hi_chg_share = share_prob_change.add(ee.Number(1.96).multiply(se_prob_change_share));

  var properties = {
    province: provinceName,
    admin_post: adminName,
    // Probability-based metrics
    ha_gain_prob: ha_prob_gain,
    ha_loss_prob: ha_prob_loss,
    ha_change_prob: ha_prob_change,
    ha_baseline_prob: ha_prob_baseline,
    ha_endline_prob: ha_prob_endline,
    share_baseline_prob: share_prob_baseline,
    share_endline_prob: share_prob_endline,
    share_change_prob: share_prob_change,
    ha_var_baseline_prob: ha_prob_var_baseline,
    ha_var_endline_prob: ha_prob_var_endline,
    ha_se_baseline_prob: ha_prob_baseline_se,
    ha_se_endline_prob: ha_prob_endline_se,
    ha_se_change_prob: se_prob_change_ha,
    share_se_change_prob: se_prob_change_share,
    share_ci95_lo_change: ci95_lo_chg_share,
    share_ci95_hi_change: ci95_hi_chg_share,
    ha_observed_baseline: ha_observed_baseline,
    ha_observed_endline: ha_observed_endline,
    ha_observed_both: ha_observed_both,
    ha_wcereal21: ha_wcereal21,
    ha_admin: ha_admin
  };
  if (USE_PROB_THRESHOLD) {
    properties.ha_gain = ha_gain;
    properties.ha_loss = ha_loss;
    properties.ha_stable_cult = ha_stable_cult;
    properties.ha_stable_uncult = ha_stable_uncult;
  }
  
  // Add observation metrics if enabled
  if (INCLUDE_OBS_METRICS) {
    properties.mean_obs_base = mean_obs_base;
    properties.mean_obs_end = mean_obs_end;
    properties.frac_pix_obs2_baseline = frac_pix_obs2_base;
    properties.frac_pix_obs2_endline = frac_pix_obs2_end;
    // Per-season observation metrics use the same toggle.
    properties.mean_obs_2017 = mean_obs_2017;
    properties.mean_obs_2018 = mean_obs_2018;
    properties.mean_obs_2019 = mean_obs_2019;
    properties.mean_obs_endline1 = mean_obs_endline0;
    properties.mean_obs_endline2 = mean_obs_endline1;
    properties.mean_obs_endline3 = mean_obs_endline2;
  }
  
  return ee.Feature(null, properties);
}

// Process all admin posts
var unitsOut = ee.FeatureCollection(adminList.map(processAdminPost));

var baseSelectors = [
  'province',
  'admin_post',
  // Probability-based metrics
  'ha_gain_prob',
  'ha_loss_prob',
  'ha_change_prob',
  'ha_baseline_prob',
  'ha_endline_prob',
  'share_baseline_prob',
  'share_endline_prob',
  'share_change_prob',
  'ha_var_baseline_prob',
  'ha_var_endline_prob',
  'ha_se_baseline_prob',
  'ha_se_endline_prob',
  'ha_se_change_prob',
  'share_se_change_prob',
  'share_ci95_lo_change',
  'share_ci95_hi_change',
  'ha_observed_baseline',
  'ha_observed_endline',
  'ha_observed_both'
];
if (USE_PROB_THRESHOLD) {
  baseSelectors.splice(2, 0,
    'ha_gain',
    'ha_loss',
    'ha_stable_cult',
    'ha_stable_uncult'
  );
}

var perSeasonObsSelectors = [
  'mean_obs_2017',
  'mean_obs_2018',
  'mean_obs_2019',
  'mean_obs_endline1',
  'mean_obs_endline2',
  'mean_obs_endline3'
];

var obsSelectors = [
  'mean_obs_base', 
  'mean_obs_end', 
  'frac_pix_obs2_baseline', 
  'frac_pix_obs2_endline'
];

var exportSelectors = baseSelectors.slice();
if (INCLUDE_OBS_METRICS) {
  exportSelectors = exportSelectors.concat(perSeasonObsSelectors);
  exportSelectors = exportSelectors.concat(obsSelectors);
}
exportSelectors = exportSelectors.concat(['ha_wcereal21', 'ha_admin']);

var maskParts = [];
if (USE_FLOODPLAIN_MASK) {
  maskParts.push('floodplain');
}
if (USE_CYCLONE_MASK && CYCLONE_TRACKS_ASSET) {
  maskParts.push('cyclone');
}
if (USE_CONFLICT_MASK) {
  maskParts.push('conflict_adm3');
}
var spatialMaskSuffix = maskParts.length > 0 ? '_' + maskParts.join('_') : '';

function sumColumn(fc, col) {
  return ee.Number(fc.reduceColumns({
    reducer: ee.Reducer.sum(),
    selectors: [col]
  }).get('sum', 0));
}

var haGainTotal = ee.Number(0);
var haStableCultTotal = ee.Number(0);
var haStableUncultTotal = ee.Number(0);
var haLossTotal = ee.Number(0);
var haCultivatedTotal = ee.Number(0);
var haClassifiedTotal = ee.Number(0);
var pctCultivated = ee.Number(0);
if (USE_PROB_THRESHOLD) {
  haGainTotal = sumColumn(unitsOut, 'ha_gain');
  haStableCultTotal = sumColumn(unitsOut, 'ha_stable_cult');
  haStableUncultTotal = sumColumn(unitsOut, 'ha_stable_uncult');
  haLossTotal = sumColumn(unitsOut, 'ha_loss');

  haCultivatedTotal = haGainTotal.add(haStableCultTotal);
  haClassifiedTotal = haGainTotal.add(haStableCultTotal).add(haStableUncultTotal).add(haLossTotal);
  pctCultivated = ee.Number(ee.Algorithms.If(
    haClassifiedTotal.gt(0),
    haCultivatedTotal.divide(haClassifiedTotal).multiply(100),
    0
  ));
}

var haAdminTotal = sumColumn(unitsOut, 'ha_admin');
var pctClassified = ee.Number(ee.Algorithms.If(
  haAdminTotal.gt(0),
  haClassifiedTotal.divide(haAdminTotal).multiply(100),
  0
));

var haProbBaselineTotal = sumColumn(unitsOut, 'ha_baseline_prob');
var haProbEndlineTotal = sumColumn(unitsOut, 'ha_endline_prob');
var haProbChangeTotal = sumColumn(unitsOut, 'ha_change_prob');

var summaryFeature = ee.Feature(null)
  .set(ADMIN3_NAME_FIELD, PROCESS_PROVINCE + '_TOTAL')
  .set(PROVINCE_FIELD, PROCESS_PROVINCE)
  .set('ha_admin', haAdminTotal)
  .set('ha_baseline_prob', haProbBaselineTotal)
  .set('ha_endline_prob', haProbEndlineTotal)
  .set('ha_change_prob', haProbChangeTotal)
  .set('endline_period', ENDLINE_YEARS[0] + '-' + ENDLINE_YEARS[ENDLINE_YEARS.length - 1])
  .set('note', 'Province-level summary for comparison with government statistics');
if (USE_PROB_THRESHOLD) {
  summaryFeature = summaryFeature
    .set('ha_gain', haGainTotal)
    .set('ha_loss', haLossTotal)
    .set('ha_stable_cult', haStableCultTotal)
    .set('ha_stable_uncult', haStableUncultTotal)
    .set('ha_cultivated_total', haCultivatedTotal)
    .set('ha_classified_total', haClassifiedTotal)
    .set('pct_cultivated_of_classified', pctCultivated)
    .set('pct_classified_of_admin', pctClassified);
}

var unitsOutWithSummary = unitsOut.merge(ee.FeatureCollection([summaryFeature]));

Export.table.toCloudStorage({
  collection: unitsOutWithSummary,
  description: 'MOZ_cultivated_change' + spatialMaskSuffix + '_' + PROCESS_PROVINCE.replace(/\s+/g,'_') + ENDLINE_SUFFIX,
  bucket: GCS_BUCKET,
  fileNamePrefix: FILE_PREFIX + spatialMaskSuffix + '_' + PROCESS_PROVINCE.replace(/\s+/g,'_') + ENDLINE_SUFFIX,
  fileFormat: 'CSV',
  selectors: exportSelectors
});

/**** MULTIBAND GEOTIFF EXPORT ****/
var safeProv = PROCESS_PROVINCE.replace(/\s+/g,'_');
var multibandList = [
  baseProb.rename('base_prob'),
  endProb.rename('end_prob'),
  deltaProb.rename('delta_prob'),
  worldcereal2021_masked.rename('wcereal21')
];
if (USE_PROB_THRESHOLD) {
  multibandList = multibandList.concat([
    st_c.rename('stable_cult'),
    st_u.rename('stable_uncult'),
    gain.rename('gain'),
    loss.rename('loss')
  ]);
}
var multibandImg = ee.Image.cat(multibandList).toFloat().clip(studyGeom);

Export.image.toCloudStorage({
  image: multibandImg,
  description: 'MOZ_multiband' + spatialMaskSuffix + '_' + safeProv + ENDLINE_SUFFIX,
  bucket: GCS_BUCKET,
  fileNamePrefix: FILE_PREFIX + '_multiband' + spatialMaskSuffix + '_' + safeProv + ENDLINE_SUFFIX,
  region: studyGeom,
  scale: 100,
  maxPixels: 1e13
});

if (INCLUDE_VALIDATION_EXPORTS && USE_PROB_THRESHOLD) {
  var NUM_POINTS_PER_CLASS = 75;
  var SAMPLE_POINTS_SCALE = 10;

  function samplePointsFromClass(classImage, className, numPoints, seedOffset) {
    var classMask = classImage.mask();
    var baseCultForExport = baselineCult.updateMask(classMask).rename('base_cult');
    var endCultForExport = endlineCult.updateMask(classMask).rename('end_cult');
    
    var samplingImage = ee.Image.cat([
      baseProb.rename('base_prob'),
      endProb.rename('end_prob'),
      deltaProb.rename('delta_prob'),
      baseCultForExport,
      endCultForExport
    ]);

    return samplingImage.updateMask(classMask).sample({
      region: studyGeom,
      scale: SAMPLE_POINTS_SCALE,
      numPixels: numPoints,
      seed: seedOffset,
      geometries: true,
      tileScale: 4
    }).map(function(feature) {
      return feature.set('predicted_class', className);
    });
  }

  var gainPoints = samplePointsFromClass(gain, 'gain', NUM_POINTS_PER_CLASS, 10001);
  var lossPoints = samplePointsFromClass(loss, 'loss', NUM_POINTS_PER_CLASS, 20002);
  var stableCultPoints = samplePointsFromClass(st_c, 'stable_cult', NUM_POINTS_PER_CLASS, 30003);
  var stableUncultPoints = samplePointsFromClass(st_u, 'stable_uncult', NUM_POINTS_PER_CLASS, 40004);
  
  var allSamplePoints = ee.FeatureCollection([])
    .merge(gainPoints)
    .merge(lossPoints)
    .merge(stableCultPoints)
    .merge(stableUncultPoints);
  
  var pointCount = allSamplePoints.size();
  var fallbackPoint = ee.FeatureCollection([
    ee.Feature(studyGeom.centroid(10), {
      predicted_class: 'fallback_dummy',
      base_prob: -999,
      end_prob: -999,
      delta_prob: -999,
      note: 'No validation points found - this should not happen'
    })
  ]);
  
  var allSamplePointsWithFallback = ee.FeatureCollection(
    ee.Algorithms.If(
      pointCount.gt(0),
      allSamplePoints,
      fallbackPoint
    )
  );

  Export.table.toCloudStorage({
    collection: allSamplePointsWithFallback,
    description: 'MOZ_validation_points' + spatialMaskSuffix + '_' + safeProv + ENDLINE_SUFFIX,
    bucket: GCS_BUCKET,
    fileNamePrefix: FILE_PREFIX + '_validation_points' + spatialMaskSuffix + '_' + safeProv + ENDLINE_SUFFIX,
    fileFormat: 'KML'
  });
}

