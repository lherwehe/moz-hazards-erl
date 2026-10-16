#!/usr/bin/env python3
"""
ERL Cabo Delgado + Nampula spatial analysis.

Standalone script — paste into Colab or run locally:

  python erl-nested-spatial.py
  python erl-nested-spatial.py --out-dir ./moz-analysis/ERL_setAB

Default output is the Desktop folder ERL_setAB_cabo_nampula, not Google Drive.

Set ENDLINE below before running.
  "long"   predictors 2018-2024, cultivated change 2023-2025
  "short"  predictors 2018-2021, cultivated change 2020-2022
The short run writes to ERL_setAB_cabo_nampula_short so it does not overwrite
the long-endline folder. One row per admin post. ACLED is standardized
log1p(events per month) with Riots removed.

Reported models. M1 through M5a each add the previous terms. M5b starts from M4:
  M1: floodplain + cyclone + drought + extreme rainfall + population density
  M2: M1 + Conflict
  M3: M2 + Cyclone x conflict
  M4: M3 + Cabo Delgado dummy (Nampula reference, not standardized)
  M5a: M4 + IOM all reasons pooled
  M5b: M4 + IOM conflict/insecurity/other/no reason and natural disaster
      (the pooled term is not in M5b; a row tagged with both enters both split terms)

Supplemental individual-predictor models, each with population density:
  IE1 Conflict, IE2 floodplain, IE3 cyclone, IE4 drought, IE5 extreme rainfall.

Lagrange multiplier tests are written for M1-M5b. The SAR versus SEM choice
uses the M5b tests. A single robust rejection chooses SAR or SEM. If neither
robust test rejects, queen contiguity is tried. If queen also names no form,
SAR and SEM are both fit and the lower AIC is the reported form (a gap under
2 is a tie, reported as SEM). Residual LISA maps are written for the reported
form and for SAR. SDM is fit only when both robust LM tests reject.

Paper figures: the M1-M5b coefficient and residual table, the M5b Lagrange
results, the preferred SAR or SEM for M5b, SAR and SEM coefficient tables for
M1-M5b, the M5b predictive-power plot, and SAR and SEM forest plots for M5b.

Supplemental alternates keep M5b and replace one exposure at a time: minimum
SPEI for drought, Holland post-landfall for cyclone, and distance-decay ACLED
for conflict. Those rows are added to the coefficient, diagnostic, and
predictive-power CSVs. One forest figure and one predictive-power figure
show all three alternates.
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union


def _resolve_base_dir() -> Path:
    """Colab-safe base directory (does not require ``__file__``)."""
    if "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ:
        return Path("/content")
    env = os.environ.get("MOZ_LOCAL_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve().parent
    try:
        return Path(__file__).resolve().parent  # type: ignore[name-defined]
    except NameError:
        return Path.cwd().resolve()


_BASE_DIR = _resolve_base_dir()
USE_GCS = True
if os.environ.get("MOZ_USE_GCS", "").strip().lower() in {"0", "false", "no"}:
    USE_GCS = False
GCS_BUCKET = "moz-lulc"
LOCAL_DATA_DIR = os.environ.get("MOZ_LOCAL_DATA_DIR", str(_BASE_DIR / "_gcs_cache"))

PROVINCES = ["Cabo Delgado", "Nampula", "Sofala", "Gaza"]
SUBSET_PROVINCES = ["Cabo Delgado", "Nampula"]
RUN_ONLY_CABO_DELGADO_NAMPULA = True
if os.environ.get("MOZ_ONLY_CABO_NAMPULA", "").strip().lower() in {"0", "false", "no", "off"}:
    RUN_ONLY_CABO_DELGADO_NAMPULA = False

ACLED_FILE = "ACLEDData_2026-04-11.csv"
CYCLONE_FILE = "ibtracs-mozambique-2017-2025-landfall-only.csv"
IOM_DTM_FILE = "moz-iom-dtm-from-api-admin-0-to-2.csv"
ADMIN_SHP = "shp/moz_dist/moz_admbnda_adm3_ine_20190607.shp"
FLOODPLAIN_TIF = "gfplain_clip.tif"
OUTCOME_COL = "share_prob_change"
SCALE_PREDICTORS = True
# M5a and M5b need the IOM DTM IDP terms (dasymetric district→post).
INCLUDE_IOM_IDP_ROBUSTNESS = True
# Printed at every run so Colab paste/kernel-cache mistakes are obvious.
PIPELINE_BUILD = "2026-10-16-endline-toggle"
REPORTED_MODELS = ("M1", "M2", "M3", "M4", "M5a", "M5b")
IE_MODELS = ("IE1", "IE2", "IE3", "IE4", "IE5")
DECISION_MODEL = "M5b"
FOREST_MODEL = "M5b"
# Riots are excluded from every ACLED term.
ACLED_DROP_EVENT_TYPES = ("Riots",)
# WorldPop GP is an annual grid (2000–2020). These agricultural windows each
# contain one annual layer: 2017-01-01 and 2020-01-01.
WORLDPOP_CHANGE_BASE_LABEL = "1 Nov 2016–31 Mar 2017"
WORLDPOP_CHANGE_END_LABEL = "1 Nov 2019–31 Mar 2020"
WORLDPOP_CHANGE_BASE_YEAR = 2017
WORLDPOP_CHANGE_END_YEAR = 2020
WORLDPOP_1KM_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/"
    "{year}/MOZ/moz_ppp_{year}_1km_Aggregated_UNadj.tif"
)
# "long" or "short". Short uses the 2018 predictor panel's short rows.
ENDLINE = "long"
# Ancillary for dasymetric IDP allocation (must exist on the predictor panel).
IOM_DASYMETRIC_WEIGHT_COL = "pop_density_period"
PRIMARY_W = "knn5"
SPATIAL_PRIMARY_WEIGHT = "knn5"
SPATIAL_INCLUDE_QUEEN = False
CYCLONE_RADIUS_KM = 50
CYCLONE_WIND_THRESHOLD_KTS = 34
CYCLONE_WIND_THRESHOLD_MS = 17.0
KTS_TO_MS = 0.514444
HOLLAND_B_DEFAULT = 1.3
HOLLAND_RMW_DEFAULT_KM = 30.0
CONFLICT_DECAY_KM = 50.0
# M5b with one exposure replaced. Colors match the MLM alternate scenarios.
ALT_MODELS = (
    (
        "M5b_alt_drought",
        "M5b with minimum SPEI in place of cumulative drought",
        "spei_endline_window_z",
        "spei_min_drought_z",
        "#0072B2",
        "Drought: minimum SPEI",
    ),
    (
        "M5b_alt_cyclone",
        "M5b with Holland post-landfall cyclone in place of the 50 km buffer",
        "cyclone_endline_window_z",
        "cyclone_holland_z",
        "#E69F00",
        "Cyclone: Holland",
    ),
    (
        "M5b_alt_conflict",
        "M5b with distance-decay conflict in place of in-post event rates",
        "conflict_endline_window_z",
        "conflict_decay_z",
        "#009E73",
        "Conflict: distance decay",
    ),
)
RUN_SEED = 42
IOM_DTM_SCOPE_PROVINCES = ["Cabo Delgado", "Nampula"]

# The 2018 panel has both period_key rows. Short rows are 2018-11-01..2021-03-31.
PREDICTOR_PANEL_PREFIX = "MOZ_MLM_predictors_lag_split_panel_2018_2024"

TIMELINES = {
    "long": {
        "label": "predictors 2018-2024; cultivated 2023-2025",
        "cultivated_suffix": "2023-2025",
        "hazard_start": "2018-11-01",
        "hazard_end": "2024-03-31",
        "endline_window_start": "2018-11-01",
        "endline_window_end": "2024-03-31",
        "outcome_phrase": "2023-2025 minus 2016-2019",
        "conflict_phrase": "2018-2024",
    },
    "short": {
        "label": "predictors 2018-2021; cultivated 2020-2022",
        "cultivated_suffix": "2020-2022",
        "hazard_start": "2018-11-01",
        "hazard_end": "2021-03-31",
        "endline_window_start": "2018-11-01",
        "endline_window_end": "2021-03-31",
        "outcome_phrase": "2020-2022 minus 2016-2019",
        "conflict_phrase": "2018-2021",
    },
}
if ENDLINE not in TIMELINES:
    raise ValueError(f"ENDLINE must be 'long' or 'short', got {ENDLINE!r}.")
RUN_PERIODS = [ENDLINE]
_ACTIVE_ENDLINE = TIMELINES[ENDLINE]
CULTIVATED_DELTA_PERIOD = _ACTIVE_ENDLINE["cultivated_suffix"]
OUTCOME_MAP_LABEL = (
    f"Cultivated land cover change ({_ACTIVE_ENDLINE['outcome_phrase']}, "
    "aggregated to post and divided by post area)"
)
CONFLICT_MAP_LABEL = f"Conflict events in the post, {_ACTIVE_ENDLINE['conflict_phrase']}"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and USE_GCS:
    try:
        from google.colab import auth  # type: ignore

        auth.authenticate_user()
    except Exception:
        print("Warning: Colab auth not available.")

_SPATIAL_UTM_EPSG = "EPSG:32736"
_ADMIN_GDF_CACHE: Optional[gpd.GeoDataFrame] = None
_IBT_DF_CACHE: Optional[pd.DataFrame] = None
_ACLED_RAW_CACHE: Optional[pd.DataFrame] = None
_ACLED_MAPPED_CACHE: Dict[int, pd.DataFrame] = {}
_PREDICTORS_CACHE: Dict[str, pd.DataFrame] = {}
_FLOODPLAIN_CACHE: Dict[int, pd.DataFrame] = {}
_POST_POLYGON_CACHE: Dict[int, gpd.GeoDataFrame] = {}
_CULTIVATED_PANEL_CACHE: Dict[str, pd.DataFrame] = {}

POP_COL = "pop_density_period_z"
ACLED_COL = "conflict_endline_window_z"
IOM_IDP_COL = "idp_conflict_pressure_z"
IOM_DISASTER_COL = "idp_disaster_pressure_z"
IOM_ALL_COL = "idp_all_reasons_pressure_z"
CABO_COL = "cabo_delgado"

TERM_LABELS = {
    "conflict_endline_window_z": "Conflict",
    "pop_density_period_z": "Population density",
    "spei_endline_window_z": "Drought",
    "chirps_endline_window_z": "Extreme rainfall",
    "floodplain_mean_z": "Floodplain",
    "cyclone_endline_window_z": "Cyclone",
    "cyclone_x_acled": "Cyclone x conflict",
    CABO_COL: "Cabo Delgado dummy",
    IOM_IDP_COL: "Conflict IDPs",
    IOM_DISASTER_COL: "Cyclone IDPs",
    IOM_ALL_COL: "All-reason IDPs",
    "CONSTANT": "Intercept",
    "rho": "Spatial lag (W)",
    "W_outcome": "Spatial lag (W)",
    "lambda": "Spatial error (lambda)",
}


@dataclass(frozen=True)
class NestedSpec:
    model_id: str
    label: str
    predictors: Tuple[str, ...]
    is_interaction: bool = False
    is_iom_robustness: bool = False
    set_name: str = ""
    skip_zscore: Tuple[str, ...] = ()


def desktop_output_dir() -> Optional[Path]:
    """Desktop folder for ERL outputs. None when this machine has no Desktop."""
    home = Path.home()
    parents = [
        home / "OneDrive - UCB-O365" / "Desktop",
        home / "Desktop",
        home / "OneDrive" / "Desktop",
    ]
    for parent in parents:
        if parent.is_dir():
            name = "ERL_setAB_cabo_nampula" if ENDLINE == "long" else "ERL_setAB_cabo_nampula_short"
            return parent / name
    return None


# Figures and sidecar files from earlier pipeline versions. A new run writes
# into the same folder, so these have to be deleted or they stay on the Desktop.
RETIRED_OUTPUT_NAMES = (
    "fig_erl_map_acled.png",
    "fig_erl_map_iom_all_reasons.png",
    "fig_erl_map_iom_conflict_disaster.png",
    "fig_erl_map_iom_idp.png",
    "fig_erl_map_outcome.png",
    "fig_erl_acled_zero_share.png",
    "fig_erl_B2_vs_B2iom.png",
    "fig_erl_lisa_B2_iom_ols_resid.png",
    "fig_erl_lisa_B2_iom_disaster_ols_resid.png",
    "fig_erl_resid_B2_iom_ols_resid.png",
    "fig_erl_resid_B2_iom_disaster_ols_resid.png",
    "fig_erl_lisa_A_acled_count_ols_resid.png",
    "fig_erl_resid_A_acled_count_ols_resid.png",
    "fig_erl_map_iom_three_panel_context.png",
    "table_erl_acled_coding_fit.png",
    "table_erl_vif.png",
    "ERL_erl_lisa_A_acled_count_ols_resid_counts.txt",
    "ERL_erl_lisa_B2_iom_ols_resid_counts.txt",
    "ERL_erl_lisa_B2_iom_disaster_ols_resid_counts.txt",
    "fig_erl_B3_x_predictive_contribution.png",
    "fig_erl_B4_x_predictive_contribution.png",
    "fig_erl_lisa_B3_x_ols_resid.png",
    "fig_erl_resid_B3_x_ols_resid.png",
    "fig_erl_lisa_B4_x_ols_resid.png",
    "fig_erl_resid_B4_x_ols_resid.png",
    "ERL_erl_lisa_B3_x_ols_resid_counts.txt",
    "ERL_erl_lisa_B4_x_ols_resid_counts.txt",
    "table_erl_setA_diagnostics.png",
    "table_erl_setA_coefs.png",
    "fig_erl_setA_moran.png",
    "fig_erl_setA_coefs.png",
    "table_erl_setB_diagnostics.png",
    "table_erl_setB_spatial_coefs_B1_B2.png",
    "table_erl_setB_spatial_coefs_B3_B4.png",
    "fig_erl_B2_lm_decision.png",
    "ERL_B2_lm_decision_note.txt",
    "fig_erl_B2_spatial_compare_acled.png",
    "fig_erl_B2_preferred_spatial_coefs.png",
    "fig_erl_B2_B3_B4_spatial_coefs.png",
    "fig_erl_B2_forms_key_coefs.png",
    "fig_erl_B2_predictive_contribution.png",
    "fig_erl_B2_x_predictive_contribution.png",
    "fig_erl_B3_predictive_contribution.png",
    "fig_erl_B4_predictive_contribution.png",
    "fig_erl_lisa_B2_ols_resid.png",
    "fig_erl_resid_B2_ols_resid.png",
    "fig_erl_lisa_B2_x_ols_resid.png",
    "fig_erl_resid_B2_x_ols_resid.png",
    "fig_erl_lisa_B3_ols_resid.png",
    "fig_erl_resid_B3_ols_resid.png",
    "fig_erl_lisa_B4_ols_resid.png",
    "fig_erl_resid_B4_ols_resid.png",
    "fig_erl_lisa_B2_spatial_resid.png",
    "fig_erl_resid_B2_spatial_resid.png",
    "table_erl_acled_no_riots.png",
    "ERL_acled_no_riots_coefficients.csv",
    "ERL_acled_no_riots_note.md",
    "ERL_setAB_ols_coefficients_short.csv",
    "ERL_setAB_ols_diagnostics_short.csv",
    "ERL_setB_spatial_coefficients_short.csv",
    "ERL_short_window_note.md",
    "ERL_B2_weights_lm.csv",
    "ERL_B2_SAR_SEM_SDM_compare_coefficients.csv",
    "fig_erl_M5_forest.png",
    "fig_erl_M5_SAR_forest.png",
    "ERL_M6_SAR_SEM_compare_coefficients.csv",
    "ERL_M6_weights_lm.csv",
    "fig_erl_M6_lm.png",
    "fig_erl_M6_forest.png",
    "fig_erl_M6_SAR_forest.png",
    "fig_erl_M6_predictive_contribution.png",
    "ERL_M1_M6_summary.csv",
    "table_erl_M1_M6.png",
    "table_erl_M1_M6_lm.png",
    "fig_erl_lisa_M6_ols_resid.png",
    "fig_erl_lisa_M6_spatial_resid.png",
    "fig_erl_lisa_M6_SAR_resid.png",
)


def remove_retired_outputs(folder: Path) -> None:
    folder = Path(folder)
    if not folder.is_dir():
        return
    for name in RETIRED_OUTPUT_NAMES:
        path = folder / name
        if path.is_file():
            path.unlink()
    for path in list(folder.glob("*.txt")) + list(folder.glob("*.md")):
        if path.is_file():
            path.unlink()
            print(f"  removed retired output: {name}")


def get_output_dir(out_dir: Optional[str] = None) -> Path:
    if out_dir:
        p = Path(out_dir)
    elif os.environ.get("MOZ_OUTPUT_DIR", "").strip():
        p = Path(os.environ["MOZ_OUTPUT_DIR"].strip())
    else:
        desk = desktop_output_dir()
        if desk is not None:
            p = desk
        elif IN_COLAB:
            folder = "ERL_setAB_cabo_nampula" if ENDLINE == "long" else "ERL_setAB_cabo_nampula_short"
            p = Path("/content/moz-analysis") / folder
        else:
            folder = "ERL_setAB_cabo_nampula" if ENDLINE == "long" else "ERL_setAB_cabo_nampula_short"
            p = _BASE_DIR / "moz-analysis" / folder
    p.mkdir(parents=True, exist_ok=True)
    return p


def publish_outputs_to_desktop(out: Path) -> None:
    """Copy the finished output folder onto the Desktop.

    If this machine has no Desktop (Colab), zip the folder and start a browser
    download of that archive.
    """
    import shutil

    src = Path(out).resolve()
    if not src.is_dir():
        print(f"Warning: output folder missing, nothing to copy: {src}")
        return
    remove_retired_outputs(src)
    dest = desktop_output_dir()
    if dest is not None:
        dest = dest.resolve()
        if src == dest:
            print(f"Outputs are already on the Desktop: {dest}")
            return
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, dirs_exist_ok=True)
        remove_retired_outputs(dest)
        n = sum(1 for p in src.rglob("*") if p.is_file())
        print(f"Copied {n} output files to the Desktop: {dest}")
        return
    if IN_COLAB:
        archive = shutil.make_archive(
            str(src.parent / src.name),
            "zip",
            root_dir=str(src),
        )
        try:
            from google.colab import files  # type: ignore

            print(f"Downloading all outputs: {archive}")
            files.download(archive)
            print("Save that zip to the Desktop. The figures and tables are at the top of the archive.")
        except Exception as exc:
            print(f"Warning: browser download failed ({exc}). Archive is at: {archive}")
        return
    print(f"Desktop folder not found. Outputs remain at: {src}")


def _term_label(term: str) -> str:
    return TERM_LABELS.get(term, term)


def _p_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _round_sig(x: float, sig: int = 3) -> float:
    """Round a float to ``sig`` significant digits (advisor reporting convention)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x  # type: ignore[return-value]
    if not np.isfinite(v) or v == 0.0:
        return v
    return float(round(v, sig - int(np.floor(np.log10(abs(v)))) - 1))


def round_df_sig(df: pd.DataFrame, sig: int = 3) -> pd.DataFrame:
    """Return a copy with float columns rounded to ``sig`` significant digits.

    Integer / bool / string columns are left unchanged.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]) or pd.api.types.is_integer_dtype(out[c]):
            continue
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v, s=sig: _round_sig(v, s) if pd.notna(v) else v)
    return out


def to_csv_sig(df: pd.DataFrame, path: Path, *, sig: int = 3, **kwargs: Any) -> None:
    """Write CSV with float values at ``sig`` significant digits."""
    round_df_sig(df, sig=sig).to_csv(path, index=kwargs.pop("index", False), **kwargs)


def _safe_tuple_stat(val: Any) -> Tuple[float, float]:
    if val is None:
        return (np.nan, np.nan)
    try:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return float(val[0]), float(val[1])
    except Exception:
        pass
    return (np.nan, np.nan)


def _zscore(v: np.ndarray) -> np.ndarray:
    sd = float(np.std(v))
    if not np.isfinite(sd) or sd <= 0:
        return v - float(np.mean(v))
    return (v - float(np.mean(v))) / sd


def _in_notebook() -> bool:
    return "google.colab" in sys.modules or "ipykernel" in sys.modules


# =============================================================================
# Data pipeline (extracted from mlm-gwr-sdm-sar-sem.py)
# =============================================================================

_IOM_DTM_CACHE: Optional[pd.DataFrame] = None


def _load_iom_dtm() -> pd.DataFrame:
    global _IOM_DTM_CACHE
    if _IOM_DTM_CACHE is None:
        path = f"gs://{GCS_BUCKET}/{IOM_DTM_FILE}" if USE_GCS else _local_data_path(IOM_DTM_FILE)
        _IOM_DTM_CACHE = read_csv(path)
    return _IOM_DTM_CACHE

def build_conflict_endline_metrics(
    admin: gpd.GeoDataFrame,
    *,
    drop_event_types: Sequence[str] = ACLED_DROP_EVENT_TYPES,
) -> pd.DataFrame:
    """ACLED counts, rates, and log1p(rate) per post for the active endline window.

    Riots are dropped before the counts. Pass ``drop_event_types=()`` to keep
    every event type.
    """
    ac = _acled_mapped_to_admin(admin)
    dropped = {str(t).strip().casefold() for t in drop_event_types if str(t).strip()}
    if dropped:
        if "event_type" not in ac.columns:
            raise ValueError("ACLED file has no event_type column; cannot drop Riots.")
        keep = ~ac["event_type"].astype(str).str.strip().str.casefold().isin(dropped)
        ac = ac.loc[keep].copy()
    out: List[pd.DataFrame] = []
    for pk in RUN_PERIODS:
        cfg = TIMELINES[pk]
        ws = pd.Timestamp(cfg["endline_window_start"])
        we = pd.Timestamp(cfg["endline_window_end"])
        window_days = max(1, int((we - ws).days) + 1)
        window_months = float(window_days / 30.4375)
        sub = ac[(ac["event_date"] >= ws) & (ac["event_date"] <= we)].copy()
        counts = (
            sub.groupby(["province", "admin_post"], as_index=False)
            .size()
            .rename(columns={"size": "conflict_endline_count"})
        )
        z = admin[["province", "admin_post"]].drop_duplicates().merge(
            counts, on=["province", "admin_post"], how="left"
        )
        z["conflict_endline_count"] = pd.to_numeric(z["conflict_endline_count"], errors="coerce").fillna(0.0)
        z["conflict_endline_rate"] = z["conflict_endline_count"] / window_months
        z["conflict_endline_window"] = np.log1p(z["conflict_endline_rate"])
        z["period_key"] = pk
        out.append(z)
    return pd.concat(out, ignore_index=True)
def _cultivated_cache_key() -> str:
    return "|".join(f"{pk}:{TIMELINES[pk].get('cultivated_suffix', '')}" for pk in RUN_PERIODS)


def load_cultivated_panel() -> pd.DataFrame:
    key = _cultivated_cache_key()
    if key not in _CULTIVATED_PANEL_CACHE:
        _CULTIVATED_PANEL_CACHE[key] = pd.concat(
            [load_cultivated(pk) for pk in RUN_PERIODS],
            ignore_index=True,
        )
    return _CULTIVATED_PANEL_CACHE[key]


def _load_acled_raw() -> pd.DataFrame:
    global _ACLED_RAW_CACHE
    if _ACLED_RAW_CACHE is None:
        path = f"gs://{GCS_BUCKET}/{ACLED_FILE}" if USE_GCS else _local_data_path(ACLED_FILE)
        ac = read_csv(path)
        ac["event_date"] = pd.to_datetime(ac["event_date"], errors="coerce")
        ac["admin1"] = normalize_key(ac["admin1"])
        ac["admin3"] = normalize_key(ac["admin3"])
        _ACLED_RAW_CACHE = ac
    return _ACLED_RAW_CACHE


def _acled_mapped_to_admin(admin: gpd.GeoDataFrame) -> pd.DataFrame:
    key = id(admin)
    if key not in _ACLED_MAPPED_CACHE:
        _ACLED_MAPPED_CACHE[key] = map_acled_to_ine(_load_acled_raw(), admin)
    return _ACLED_MAPPED_CACHE[key]


def _post_polygons_by_admin_id(admin_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    key = id(admin_gdf)
    if key in _POST_POLYGON_CACHE:
        return _POST_POLYGON_CACHE[key]
    g = admin_gdf.copy()
    g["admin_post_id"] = g["province"] + "||" + g["admin_post"]
    dissolved = g.dissolve(by="admin_post_id", as_index=False)
    out = dissolved[["admin_post_id", "geometry"]].set_geometry("geometry")
    _POST_POLYGON_CACHE[key] = out
    return out


def _norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return unicodedata.normalize("NFC", str(s).strip())


def normalize_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.normalize("NFC")


def _iom_reason_has_tag(series: pd.Series, tag: str) -> pd.Series:
    """Match an exact tag in a semicolon-delimited IOM displacement reason."""
    target = str(tag).strip().casefold()
    return series.astype(str).apply(
        lambda value: any(
            part.strip().casefold() == target for part in value.split(";")
        )
    )


def _iom_reason_bucket(series: pd.Series, bucket: str) -> pd.Series:
    """Split IOM displacement reasons.

    Conflict bucket: Conflict, Insecurity, Other reason, and No reason.
    Disaster bucket: any row tagged Natural disaster, including rows that are
    also tagged conflict, insecurity, other, or no reason. Those double-tagged
    rows stay in both buckets.
    All bucket: the union of the two, so each row is counted once.
    """
    reasons = series.astype(str)
    conflict = _iom_reason_has_tag(reasons, "Conflict")
    insecurity = _iom_reason_has_tag(reasons, "Insecurity")
    other = _iom_reason_has_tag(reasons, "Other reason")
    no_reason = _iom_reason_has_tag(reasons, "No reason for displacement reported")
    disaster = _iom_reason_has_tag(reasons, "Natural disaster")
    conflict_bucket = conflict | insecurity | other | no_reason
    if bucket == "conflict":
        return conflict_bucket
    if bucket == "disaster":
        return disaster
    if bucket == "all":
        return conflict_bucket | disaster
    raise ValueError(f"Unknown IOM reason bucket {bucket!r}")


def _idp_scope_provinces() -> set:
    """Normalized province names used for IOM DTM displacement inputs."""
    return set(normalize_key(pd.Series(IOM_DTM_SCOPE_PROVINCES, dtype=object)).dropna().tolist())


def _local_data_path(*parts: str) -> str:
    """Resolve a path under LOCAL_DATA_DIR (used when USE_GCS is False)."""
    return os.path.join(LOCAL_DATA_DIR, *parts)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _active_provinces() -> List[str]:
    if not RUN_ONLY_CABO_DELGADO_NAMPULA:
        return list(PROVINCES)
    allowed = set(PROVINCES)
    chosen = [p for p in SUBSET_PROVINCES if p in allowed]
    if not chosen:
        raise ValueError(
            "RUN_ONLY_CABO_DELGADO_NAMPULA is enabled but SUBSET_PROVINCES has no values in PROVINCES."
        )
    return chosen


def load_admin() -> gpd.GeoDataFrame:
    """Load INE admin-post polygons for active provinces (cached per run)."""
    global _ADMIN_GDF_CACHE
    if _ADMIN_GDF_CACHE is not None:
        return _ADMIN_GDF_CACHE
    shp = (
        f"gs://{GCS_BUCKET}/{ADMIN_SHP}"
        if USE_GCS
        else _local_data_path(*ADMIN_SHP.replace("\\", "/").split("/"))
    )
    gdf = gpd.read_file(shp).to_crs("EPSG:4326")
    gdf["province"] = normalize_key(gdf["ADM1_PT"])
    gdf["district"] = normalize_key(gdf["ADM2_PT"])
    gdf["district_pcode"] = gdf["ADM2_PCODE"].astype(str).str.strip().str.upper()
    gdf["admin_post"] = normalize_key(gdf["ADM3_PT"])
    active = _active_provinces()
    _ADMIN_GDF_CACHE = gdf[gdf["province"].isin(active)][
        ["province", "district", "district_pcode", "admin_post", "geometry"]
    ]
    return _ADMIN_GDF_CACHE


def build_floodplain_agg(admin_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Zonal mean of GFPLAIN floodplain indicator (1=in floodplain) per admin post."""
    fp_key = id(admin_gdf)
    if fp_key in _FLOODPLAIN_CACHE:
        return _FLOODPLAIN_CACHE[fp_key]
    tif_path = None
    if USE_GCS:
        try:
            from google.cloud import storage

            local_fp = _local_data_path(FLOODPLAIN_TIF)
            os.makedirs(os.path.dirname(local_fp), exist_ok=True)
            if not (os.path.isfile(local_fp) and os.path.getsize(local_fp) > 0):
                storage.Client().bucket(GCS_BUCKET).blob(FLOODPLAIN_TIF).download_to_filename(local_fp)
            if not (os.path.isfile(local_fp) and os.path.getsize(local_fp) > 0):
                raise FileNotFoundError(
                    f"Downloaded floodplain raster is missing or empty: {local_fp}"
                )
            tif_path = local_fp
        except Exception as e:
            raise RuntimeError(
                f"Could not cache gs://{GCS_BUCKET}/{FLOODPLAIN_TIF} at "
                f"{_local_data_path(FLOODPLAIN_TIF)}. Floodplain exposure cannot "
                "be estimated; refusing to substitute zeros."
            ) from e
    else:
        local_fp = _local_data_path(FLOODPLAIN_TIF)
        tif_path = local_fp if os.path.isfile(local_fp) else None

    if not tif_path:
        raise FileNotFoundError(
            f"Floodplain raster not found at {local_fp}. Set MOZ_LOCAL_DATA_DIR "
            f"to the directory containing {FLOODPLAIN_TIF}, or enable GCS input."
        )

    try:
        import rasterio
        from rasterstats import zonal_stats

        with rasterio.open(tif_path) as src:
            arr = src.read(1)
            affine = src.transform
            rcrs = src.crs
        floodplain_01 = np.where(arr == 0, 1.0, 0.0).astype(np.float32)
        admin_in_raster_crs = admin_gdf.to_crs(rcrs)
        stats_out = zonal_stats(
            admin_in_raster_crs,
            floodplain_01,
            affine=affine,
            stats=["mean"],
            all_touched=True,
            nodata=-999,
        )
        out = admin_gdf[["province", "admin_post"]].copy()
        out["floodplain_mean"] = [float(s["mean"]) if s.get("mean") is not None else 0.0 for s in stats_out]
        _FLOODPLAIN_CACHE[fp_key] = out
        return out
    except Exception as e:
        raise RuntimeError(
            f"Floodplain zonal statistics failed for {tif_path}; refusing to "
            "substitute a constant-zero exposure."
        ) from e


def load_cultivated(period_key: str) -> pd.DataFrame:
    """Load province cultivated-change CSVs for one period (short or long suffix)."""
    suffix = TIMELINES[period_key]["cultivated_suffix"]
    rows = []
    for prov in _active_provinces():
        name = f"MOZ_cultivated_change_{prov.replace(' ', '_')}_{suffix}.csv"
        p = f"gs://{GCS_BUCKET}/{name}" if USE_GCS else _local_data_path("cultivated_csvs", name)
        rows.append(read_csv(p))
    out = pd.concat(rows, ignore_index=True)
    if "share_prob_change" not in out.columns and "share_change_prob" in out.columns:
        out["share_prob_change"] = out["share_change_prob"]
    out["province"] = normalize_key(out["province"])
    out["admin_post"] = normalize_key(out["admin_post"])
    out["period_key"] = period_key
    return out


def load_predictors_panel_with_prefix(prefix: str = PREDICTOR_PANEL_PREFIX) -> pd.DataFrame:
    """Load lag-split predictor panel CSVs (SPEI, VIIRS, CHIRPS, pop density, etc.)."""
    if prefix in _PREDICTORS_CACHE:
        return _PREDICTORS_CACHE[prefix]
    parts = []
    for prov in _active_provinces():
        lag_name = f"{prefix}_{prov.replace(' ', '_')}.csv"
        base_name = f"MOZ_MLM_predictors_panel_{prov.replace(' ', '_')}.csv"
        lag_path = f"gs://{GCS_BUCKET}/{lag_name}" if USE_GCS else _local_data_path(lag_name)
        base_path = f"gs://{GCS_BUCKET}/{base_name}" if USE_GCS else _local_data_path(base_name)
        try:
            parts.append(read_csv(lag_path))
        except Exception:
            parts.append(read_csv(base_path))
    df = pd.concat(parts, ignore_index=True)
    df["province"] = normalize_key(df["province"])
    df["admin_post"] = normalize_key(df["admin_post"])
    df["period_key"] = normalize_key(df["period_key"]).str.lower()
    _PREDICTORS_CACHE[prefix] = df
    return df


def map_acled_to_ine(acled: pd.DataFrame, admin: gpd.GeoDataFrame) -> pd.DataFrame:
    lookup: Dict[Tuple[str, str], str] = {}
    for _, r in admin.drop_duplicates(["province", "admin_post"]).iterrows():
        p = _norm(r["province"]).casefold()
        a = _norm(r["admin_post"]).casefold()
        lookup[(p, a)] = r["admin_post"]
        stripped = re.sub(r"^(cidade|ilha|vila)\s+(de|da|do|dos)\s+", "", r["admin_post"], flags=re.I).strip().casefold()
        if stripped and stripped != a:
            lookup[(p, stripped)] = r["admin_post"]
    manual = {
        ("cabo delgado", "pemba"): "Cidade De Pemba",
        ("nampula", "nampula"): "Cidade De Nampula",
        ("sofala", "beira"): "Cidade Da Beira",
        ("gaza", "xai-xai"): "Cidade De Xai-Xai",
        ("gaza", "xai xai"): "Cidade De Xai-Xai",
    }
    lookup.update({(k[0], k[1]): v for k, v in manual.items()})
    ac = acled.copy()
    ac["admin_post"] = [
        lookup.get((_norm(p).casefold(), _norm(a).casefold()), _norm(a))
        for p, a in zip(ac["admin1"], ac["admin3"])
    ]
    ac["province"] = normalize_key(ac["admin1"])
    return ac


# =========================
# §5 HAZARD EXPOSURE BUILDERS
# Build period-specific conflict, cyclone, IDP columns merged into the panel.
# =========================


def _load_ibt() -> pd.DataFrame:
    global _IBT_DF_CACHE
    if _IBT_DF_CACHE is None:
        p = f"gs://{GCS_BUCKET}/{CYCLONE_FILE}" if USE_GCS else _local_data_path(CYCLONE_FILE)
        ibt = read_csv(p)
        ibt["date"] = pd.to_datetime(ibt["ISO_TIME"], format="%m/%d/%Y %H:%M", errors="coerce")
        ibt["DIST2LAND"] = pd.to_numeric(ibt.get("DIST2LAND"), errors="coerce")
        for c in ["REUNION_WIND", "WMO_WIND", "USA_WIND"]:
            if c in ibt.columns:
                ibt[c] = pd.to_numeric(ibt[c], errors="coerce")
        nan = pd.Series(np.nan, index=ibt.index)
        ibt["wind_kts"] = (
            (ibt["REUNION_WIND"] if "REUNION_WIND" in ibt.columns else nan).where(lambda s: s > 0)
            .fillna((ibt["WMO_WIND"] if "WMO_WIND" in ibt.columns else nan).where(lambda s: s > 0))
            .fillna((ibt["USA_WIND"] if "USA_WIND" in ibt.columns else nan).where(lambda s: s > 0))
        )
        ibt = ibt.dropna(subset=["date", "LAT", "LON", "wind_kts"])
        # Keep full storm tracks with valid winds. Post-landfall filtering is applied downstream.
        _IBT_DF_CACHE = ibt[(ibt["wind_kts"] >= CYCLONE_WIND_THRESHOLD_KTS)].copy()
    return _IBT_DF_CACHE.copy()


def _cyclone_fraction(admin_utm: gpd.GeoDataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    if len(sub) == 0:
        return admin_utm[["province", "admin_post"]].assign(cyclone=0.0)
    pts = gpd.GeoDataFrame(sub, geometry=gpd.points_from_xy(sub["LON"], sub["LAT"]), crs="EPSG:4326").to_crs("EPSG:32736")
    union = unary_union(pts.geometry.buffer(CYCLONE_RADIUS_KM * 1000).values)
    vals = []
    for _, r in admin_utm.iterrows():
        poly = r.geometry
        if poly is None or poly.is_empty or union is None:
            frac = 0.0
        else:
            if not poly.is_valid:
                poly = poly.buffer(0)
            area = poly.area
            frac = 0.0 if area <= 0 else min(1.0, poly.intersection(union).area / area)
        vals.append({"province": r["province"], "admin_post": r["admin_post"], "cyclone": float(frac)})
    return pd.DataFrame(vals)

def build_cyclone_buffer_fraction(admin: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Core cyclone specification: simple 50 km buffer fraction.

    For each period window, exposure is the fraction of each admin-post polygon area
    intersecting the union of 50 km buffers around qualifying IBTrACS points
    (wind_kts >= 34 in _load_ibt input data).
    """
    ibt = _load_ibt().copy()
    admin_utm = admin.to_crs("EPSG:32736")
    out = []
    timelines = TIMELINES
    for pk in RUN_PERIODS:
        cfg = timelines[pk]
        hs = pd.Timestamp(cfg["hazard_start"])
        he = pd.Timestamp(cfg["hazard_end"])
        ws = pd.Timestamp(cfg["endline_window_start"])
        we = pd.Timestamp(cfg["endline_window_end"])
        base = ibt[(ibt["date"] >= hs) & (ibt["date"] <= he)].copy()
        sub_window = base[(base["date"] >= ws) & (base["date"] <= we)]
        z = _cyclone_fraction(admin_utm, sub_window).rename(columns={"cyclone": "cyclone_endline_window"})
        z = z.fillna(0.0)
        z["period_key"] = pk
        out.append(z)
    return pd.concat(out, ignore_index=True)


def _holland_local_wind_ms(vmax_ms: np.ndarray, r_km: np.ndarray, rmw_km: np.ndarray, b: float) -> np.ndarray:
    """Holland radial profile: V(r) = Vmax * (rmw/r)^B * exp(1 - (rmw/r)^B)."""
    r_eff = np.maximum(np.asarray(r_km, dtype=float), 1e-3)
    rmw_eff = np.maximum(np.asarray(rmw_km, dtype=float), 1e-3)
    ratio_pow = np.power(rmw_eff / r_eff, float(b))
    return np.asarray(vmax_ms, dtype=float) * ratio_pow * np.exp(1.0 - ratio_pow)


def _cyclone_binary_holland_postlandfall(admin_utm: gpd.GeoDataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Binary cyclone exposure from a Holland-style radial wind profile, post-landfall only."""
    from shapely.geometry import LineString, Point

    base = admin_utm[["province", "admin_post", "geometry"]].copy()
    if len(sub) == 0:
        return base[["province", "admin_post"]].assign(cyclone_holland=0.0)

    sid_col = "SID" if "SID" in sub.columns else ("STORM_ID" if "STORM_ID" in sub.columns else None)
    if sid_col is None:
        sub = sub.copy()
        name_series = sub["NAME"].astype(str) if "NAME" in sub.columns else pd.Series("storm", index=sub.index)
        sub["_sid_fallback"] = name_series + "|" + sub["date"].dt.year.astype(str)
        sid_col = "_sid_fallback"

    rows: List[pd.DataFrame] = []
    for _, group in sub.groupby(sid_col, sort=False):
        group = group.sort_values("date").copy()
        d2l = pd.to_numeric(group.get("DIST2LAND"), errors="coerce")
        land_mask = np.isfinite(d2l) & (d2l <= 0)
        if not land_mask.any():
            continue
        first_land_idx = np.flatnonzero(land_mask.to_numpy(dtype=bool))[0]
        group_post = group.iloc[first_land_idx:].copy()
        d2l_post = pd.to_numeric(group_post.get("DIST2LAND"), errors="coerce")
        group_post = group_post[np.isfinite(d2l_post) & (d2l_post <= 0)].copy()
        if not group_post.empty:
            rows.append(group_post)

    if not rows:
        return base[["province", "admin_post"]].assign(cyclone_holland=0.0)
    post_pts = pd.concat(rows, ignore_index=True)
    pts = gpd.GeoDataFrame(
        post_pts, geometry=gpd.points_from_xy(post_pts["LON"], post_pts["LAT"]), crs="EPSG:4326"
    ).to_crs("EPSG:32736")
    pts["wind_ms"] = pd.to_numeric(pts.get("wind_kts"), errors="coerce") * KTS_TO_MS
    rmw = pd.Series(np.nan, index=pts.index, dtype=float)
    for col in ("USA_RMW", "WMO_RMW", "REUNION_RMW", "RMW", "TOKYO_RMW"):
        if col in pts.columns:
            rmw = rmw.fillna(pd.to_numeric(pts[col], errors="coerce"))
    pts["rmw_km"] = rmw.fillna(HOLLAND_RMW_DEFAULT_KM).clip(lower=1.0)
    pts = pts[np.isfinite(pts["wind_ms"]) & (pts["wind_ms"] > 0)].copy()
    if len(pts) == 0:
        return base[["province", "admin_post"]].assign(cyclone_holland=0.0)

    seg_rows: List[Dict[str, Any]] = []
    for _, group in pts.groupby(sid_col, sort=False):
        group = group.sort_values("date").copy()
        if len(group) == 1:
            rr = group.iloc[0]
            seg_rows.append({"geometry": rr.geometry, "wind_ms": float(rr["wind_ms"]), "rmw_km": float(rr["rmw_km"])})
            continue
        rec = list(group.itertuples(index=False))
        for k in range(len(rec) - 1):
            a = rec[k]
            b2 = rec[k + 1]
            p0 = a.geometry
            p1 = b2.geometry
            if p0 is None or p1 is None:
                continue
            line = LineString([p0, p1])
            if line.length <= 0:
                seg_rows.append({"geometry": Point(p0.x, p0.y), "wind_ms": float(a.wind_ms), "rmw_km": float(a.rmw_km)})
                continue
            mid = line.interpolate(0.5, normalized=True)
            seg_rows.append({"geometry": LineString([p0, mid]), "wind_ms": float(a.wind_ms), "rmw_km": float(a.rmw_km)})
            seg_rows.append({"geometry": LineString([mid, p1]), "wind_ms": float(b2.wind_ms), "rmw_km": float(b2.rmw_km)})

    if not seg_rows:
        return base[["province", "admin_post"]].assign(cyclone_holland=0.0)
    segs = gpd.GeoDataFrame(seg_rows, geometry="geometry", crs="EPSG:32736")
    vals: List[Dict[str, Any]] = []
    thr = float(CYCLONE_WIND_THRESHOLD_MS)
    b_holland = float(max(HOLLAND_B_DEFAULT, 0.2))
    for _, row in base.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty:
            vals.append({"province": row["province"], "admin_post": row["admin_post"], "cyclone_holland": 0.0})
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        d_km = segs.geometry.distance(poly.centroid).to_numpy(dtype=float) / 1000.0
        v_local = _holland_local_wind_ms(
            segs["wind_ms"].to_numpy(dtype=float),
            d_km,
            segs["rmw_km"].to_numpy(dtype=float),
            b_holland,
        )
        max_wind = float(np.nanmax(v_local)) if len(v_local) else 0.0
        vals.append(
            {
                "province": row["province"],
                "admin_post": row["admin_post"],
                "cyclone_holland": 1.0 if max_wind >= thr else 0.0,
            }
        )
    return pd.DataFrame(vals)


def build_cyclone_holland(admin: gpd.GeoDataFrame) -> pd.DataFrame:
    """Alternate cyclone spec: binary Holland post-landfall wind threshold."""
    ibt = _load_ibt()
    admin_utm = admin.to_crs("EPSG:32736")
    out = []
    for pk in RUN_PERIODS:
        cfg = TIMELINES[pk]
        hs = pd.Timestamp(cfg["hazard_start"])
        he = pd.Timestamp(cfg["hazard_end"])
        ws = pd.Timestamp(cfg["endline_window_start"])
        we = pd.Timestamp(cfg["endline_window_end"])
        base = ibt[(ibt["date"] >= hs) & (ibt["date"] <= he)].copy()
        sub_window = base[(base["date"] >= ws) & (base["date"] <= we)]
        z = _cyclone_binary_holland_postlandfall(admin_utm, sub_window)
        z = z.fillna(0.0)
        z["period_key"] = pk
        z = z.drop_duplicates(["province", "admin_post", "period_key"])
        out.append(z)
    return pd.concat(out, ignore_index=True)


def build_conflict_distance_decay(
    admin: gpd.GeoDataFrame,
    *,
    drop_event_types: Sequence[str] = ACLED_DROP_EVENT_TYPES,
) -> pd.DataFrame:
    """Alternate conflict spec: asinh of monthly distance-decay ACLED intensity."""
    ac = _acled_mapped_to_admin(admin)
    dropped = {str(t).strip().casefold() for t in drop_event_types if str(t).strip()}
    if dropped and "event_type" in ac.columns:
        keep = ~ac["event_type"].astype(str).str.strip().str.casefold().isin(dropped)
        ac = ac.loc[keep].copy()
    lon_col = next((c for c in ("longitude", "LONGITUDE", "lon", "LON") if c in ac.columns), None)
    lat_col = next((c for c in ("latitude", "LATITUDE", "lat", "LAT") if c in ac.columns), None)
    if lon_col is None or lat_col is None:
        raise ValueError("ACLED file is missing longitude/latitude columns for distance-decay conflict.")
    admin_utm = admin.to_crs("EPSG:32736").copy()
    centroids = admin_utm[["province", "admin_post"]].copy()
    centroids["geometry"] = admin_utm.geometry.centroid
    out = []
    for pk in RUN_PERIODS:
        cfg = TIMELINES[pk]
        ws = pd.Timestamp(cfg["endline_window_start"])
        we = pd.Timestamp(cfg["endline_window_end"])
        window_days = max(1, int((we - ws).days) + 1)
        window_months = float(window_days / 30.4375)
        sub = ac[(ac["event_date"] >= ws) & (ac["event_date"] <= we)].copy()
        if len(sub) == 0:
            z = centroids[["province", "admin_post"]].copy()
            z["conflict_decay"] = 0.0
        else:
            pts = gpd.GeoDataFrame(
                sub,
                geometry=gpd.points_from_xy(sub[lon_col], sub[lat_col]),
                crs="EPSG:4326",
            ).to_crs("EPSG:32736")
            vals = []
            for _, row in centroids.iterrows():
                d_km = pts.geometry.distance(row.geometry).to_numpy(dtype=float) / 1000.0
                wsum = float(np.exp(-d_km / CONFLICT_DECAY_KM).sum())
                vals.append(
                    {
                        "province": row["province"],
                        "admin_post": row["admin_post"],
                        "conflict_decay": np.arcsinh(wsum / window_months),
                    }
                )
            z = pd.DataFrame(vals)
        z["period_key"] = pk
        z = z.drop_duplicates(["province", "admin_post", "period_key"])
        out.append(z)
    return pd.concat(out, ignore_index=True)

def build_panel_core_moz(
    admin: Optional[gpd.GeoDataFrame] = None,
    *,
    conflict_builder: Optional[Callable[[gpd.GeoDataFrame], pd.DataFrame]] = None,
    cyclone_builder: Optional[Callable[[gpd.GeoDataFrame], pd.DataFrame]] = None,
    spei_source_col: str = "spei_cumulative_deficit",
    predictor_prefix: str = PREDICTOR_PANEL_PREFIX,
) -> pd.DataFrame:
    """Common panel constructor for core + alternates (csv ``pop_density_period``, shared listwise filter)."""
    admin = admin if admin is not None else load_admin()
    # --- Merge outcome + predictors + hazard exposures ---
    cultivated = load_cultivated_panel()
    pred = load_predictors_panel_with_prefix(predictor_prefix)
    conf_fn = conflict_builder or build_conflict_endline_metrics
    cyc_fn = cyclone_builder or build_cyclone_buffer_fraction
    conf = conf_fn(admin)
    cyc = cyc_fn(admin)
    flood = build_floodplain_agg(admin)

    keep = ["province", "admin_post", "period_key", OUTCOME_COL]
    if OUTCOME_COL not in cultivated.columns:
        raise ValueError(f"Outcome column {OUTCOME_COL!r} not found.")
    panel = cultivated[keep].copy().rename(columns={OUTCOME_COL: "outcome"})

    panel = panel.merge(pred, on=["province", "admin_post", "period_key"], how="left")
    panel = panel.merge(conf, on=["province", "admin_post", "period_key"], how="left")
    panel = panel.merge(cyc, on=["province", "admin_post", "period_key"], how="left")
    panel = panel.merge(flood, on=["province", "admin_post"], how="left")

    # --- IDs, migration proxy, SPEI orientation (higher = worse drought) ---
    panel["period_long"] = (panel["period_key"] == "long").astype(int)
    panel["admin_post_id"] = panel["province"] + "||" + panel["admin_post"]
    panel["province_id"] = panel["province"]

    panel["pop_density_period"] = pd.to_numeric(panel.get("pop_density_period"), errors="coerce")
    panel["conflict_endline_window"] = pd.to_numeric(panel.get("conflict_endline_window"), errors="coerce").fillna(0)
    panel["cyclone_endline_window"] = pd.to_numeric(panel.get("cyclone_endline_window"), errors="coerce").fillna(0)
    # VIIRS is upstream-fixed to VIIRS_FIXED_START..VIIRS_FIXED_END for all rows.
    # Keep raw VIIRS mean radiance scale.
    panel["viirs_endline_window"] = pd.to_numeric(panel.get("viirs_endline_window"), errors="coerce")
    panel["floodplain_mean"] = pd.to_numeric(panel.get("floodplain_mean"), errors="coerce").fillna(0)

    spei_source = None
    if spei_source_col in panel.columns:
        spei_source = spei_source_col
    elif "spei_cumulative_deficit" in panel.columns:
        spei_source = "spei_cumulative_deficit"
    elif "spei_endline_window" in panel.columns:
        spei_source = "spei_endline_window"
        print(f"Warning: spei column {spei_source_col!r} not found; using 'spei_endline_window'.")
    if spei_source is None:
        spei_like = [c for c in panel.columns if "spei" in c.lower()]
        raise ValueError(
            f"Expected SPEI column {spei_source_col!r} (or fallback 'spei_endline_window'). "
            f"Available SPEI-like columns: {spei_like}"
        )
    raw_spei = pd.to_numeric(panel[spei_source], errors="coerce")
    # Harmonize sign convention across SPEI variants: higher value = worse drought exposure.
    # - spei_cumulative_deficit is dry-month count (SPEI-6 <= -1 per pixel, summed in window; zonal mean); higher = drier.
    # - spei_min is the SPEI index minimum (more negative = drier), so negate it.
    # - legacy spei_endline_window from older exports is SPEI-like index; negate for consistency.
    if spei_source == "spei_min":
        panel["spei_endline_window"] = -raw_spei
    elif spei_source == "spei_cumulative_deficit":
        panel["spei_endline_window"] = raw_spei
    else:
        panel["spei_endline_window"] = -raw_spei

    if "chirps_endline_window" in panel.columns:
        panel["chirps_endline_window"] = pd.to_numeric(panel["chirps_endline_window"], errors="coerce")
    elif "chirps_r95p_mm" in panel.columns:
        chirps_raw = pd.to_numeric(panel["chirps_r95p_mm"], errors="coerce")
        panel["chirps_endline_window"] = np.log1p(chirps_raw.clip(lower=0))
    else:
        raise ValueError(
            "Expected CHIRPS column 'chirps_endline_window' or 'chirps_r95p_mm'. "
            f"Available columns: {sorted(panel.columns)}"
        )

    # --- Listwise complete case, then z-score predictors for MixedLM ---
    req = [
        "outcome",
        "pop_density_period",
        "conflict_endline_window",
        "conflict_endline_count",
        "conflict_endline_rate",
        "cyclone_endline_window",
        "spei_endline_window",
        "chirps_endline_window",
        "floodplain_mean",
    ]
    panel = panel.dropna(subset=req).copy()

    pred_cols = [
        "pop_density_period",
        "conflict_endline_window",
        "conflict_endline_count",
        "conflict_endline_rate",
        "cyclone_endline_window",
        "spei_endline_window",
        "chirps_endline_window",
        "floodplain_mean",
    ]
    for c in pred_cols:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)
        if SCALE_PREDICTORS:
            s = panel[c].std()
            panel[c + "_z"] = (panel[c] - panel[c].mean()) / s if (pd.notna(s) and s > 0) else 0.0
        else:
            panel[c + "_z"] = panel[c]

    return panel

def _build_w_variants(poly_gdf: Any) -> Dict[str, Any]:
    """Build spatial weights with kNN as primary and optional extras."""
    variants: Dict[str, Any] = {}
    try:
        from libpysal.weights import KNN, Queen
    except Exception:
        return variants

    g = poly_gdf.copy()
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g = g.to_crs("EPSG:32736")
    cent = g.geometry.centroid
    xy = np.column_stack([cent.x.to_numpy(dtype=float), cent.y.to_numpy(dtype=float)])

    for k in (5,):
        try:
            w = KNN.from_array(xy, k=k)
            w.transform = "r"
            variants[f"knn{k}"] = w
        except Exception:
            continue

    if SPATIAL_INCLUDE_QUEEN:
        try:
            if hasattr(Queen, "from_dataframe"):
                wq = Queen.from_dataframe(g)
            else:
                wq = Queen.from_geodataframe(g)
            if len(getattr(wq, "islands", [])) == 0:
                wq.transform = "r"
                variants["queen"] = wq
        except Exception:
            pass

    return variants


def _queen_weights(poly_gdf: Any) -> Tuple[Optional[Any], str]:
    """Shared-border weights. Posts with no neighbor are tied to the nearest centroid."""
    try:
        from libpysal.weights import KNN, Queen, W
    except Exception as exc:
        return None, f"Queen weights unavailable ({exc})."

    g = poly_gdf.copy().reset_index(drop=True)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g = g.to_crs("EPSG:32736")
    cent = g.geometry.centroid
    xy = np.column_stack([cent.x.to_numpy(dtype=float), cent.y.to_numpy(dtype=float)])
    try:
        if hasattr(Queen, "from_dataframe"):
            try:
                wq = Queen.from_dataframe(g, silence_warnings=True)
            except TypeError:
                wq = Queen.from_dataframe(g)
        else:
            wq = Queen.from_geodataframe(g)
    except Exception as exc:
        return None, f"Queen weights failed ({exc})."

    neighbors: Dict[Any, List[Any]] = {i: list(v) for i, v in wq.neighbors.items()}
    islands = [i for i in list(getattr(wq, "islands", []) or [])]
    for i, nbrs in list(neighbors.items()):
        if len(nbrs) == 0 and i not in islands:
            islands.append(i)
    note = "Queen contiguity uses shared borders and is row-standardized."
    if islands:
        knn1 = KNN.from_array(xy, k=1)
        for i in islands:
            j = knn1.neighbors[i][0]
            neighbors[i] = [j]
            neighbors.setdefault(j, [])
            if i not in neighbors[j]:
                neighbors[j] = list(neighbors[j]) + [i]
        note += (
            f" {len(islands)} post(s) shared no border and were linked to the nearest"
            " centroid so the matrix has no islands."
        )
    neighbors = {i: [j for j in nbrs if j != i] for i, nbrs in neighbors.items() if nbrs}
    try:
        built = W(neighbors, silence_warnings=True)
    except TypeError:
        built = W(neighbors)
    built.transform = "r"
    return built, note


def _admin_post_area_km2(admin: gpd.GeoDataFrame) -> pd.DataFrame:
    """Post polygon area (km²) in UTM; used as the geometric factor in dasymetric weights."""
    g = admin[["province", "admin_post", "geometry"]].copy()
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g_utm = g.to_crs(_SPATIAL_UTM_EPSG)
    g_utm["area_km2"] = g_utm.geometry.area / 1.0e6
    return (
        g_utm.groupby(["province", "admin_post"], as_index=False)["area_km2"]
        .sum()
        .replace([np.inf, -np.inf], np.nan)
    )


def _dasymetric_allocate_district_to_posts(
    posts: pd.DataFrame,
    district_totals: pd.DataFrame,
    *,
    value_col: str = "idp_conflict_pressure",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Allocate district totals to member posts proportional to ``weight_col``.

    Tobler-style dasymetric (ancillary-weighted) assignment: within each district,
    shares sum to 1 so post values sum to the district total. If all weights in a
    district are zero/missing, fall back to equal shares among posts; unmatched
    posts (no district total) are coded 0.
    """
    z = posts.merge(
        district_totals[["province", "district_pcode", value_col]],
        on=["province", "district_pcode"],
        how="left",
    )
    w = pd.to_numeric(z.get(weight_col), errors="coerce").fillna(0.0).clip(lower=0.0)
    z["_w"] = w
    z["_wsum"] = z.groupby(["province", "district_pcode"], dropna=False)["_w"].transform("sum")
    z["_n_posts"] = z.groupby(["province", "district_pcode"], dropna=False)["_w"].transform("size")
    district_val = pd.to_numeric(z[value_col], errors="coerce")
    share = np.where(
        z["_wsum"].to_numpy(dtype=float) > 0,
        z["_w"].to_numpy(dtype=float) / z["_wsum"].to_numpy(dtype=float),
        np.where(z["_n_posts"].to_numpy(dtype=float) > 0, 1.0 / z["_n_posts"].to_numpy(dtype=float), 0.0),
    )
    allocated = np.where(district_val.notna().to_numpy(), district_val.to_numpy(dtype=float) * share, 0.0)
    z[value_col] = allocated
    return z.drop(columns=["_w", "_wsum", "_n_posts"], errors="ignore")


def build_idp_conflict_pressure_lag_split(
    admin: gpd.GeoDataFrame,
    *,
    ancillary: Optional[pd.DataFrame] = None,
    weight_col: str = IOM_DASYMETRIC_WEIGHT_COL,
    verbose: bool = True,
    reason_bucket: str = "conflict",
    value_name: str = "idp_conflict_pressure",
) -> pd.DataFrame:
    """Destination-side conflict displacement pressure from district-level IOM DTM.

    District mean-monthly present IDP counts are allocated to admin posts with a
    Tobler-style **dasymetric** scheme: weight ∝ ``max(pop_density, 0) × area_km²``
    (proxy population), using the predictor-panel population density when
    ``ancillary`` is provided. Unmatched districts / posts are coded zero.

    ``verbose=False`` suppresses the district→post match report.
    """
    dtm = _load_iom_dtm()
    dtm["reportingDate"] = pd.to_datetime(dtm.get("reportingDate"), errors="coerce")
    dtm["displacementReason"] = normalize_key(dtm.get("displacementReason", pd.Series("", index=dtm.index)))
    dtm["operationStatus"] = normalize_key(dtm.get("operationStatus", pd.Series("", index=dtm.index)))
    dtm["province"] = normalize_key(dtm.get("admin1Name", pd.Series("", index=dtm.index)))
    dtm["district"] = normalize_key(dtm.get("admin2Name", pd.Series("", index=dtm.index)))
    dtm["district_pcode"] = dtm.get("admin2Pcode", pd.Series("", index=dtm.index)).astype(str).str.strip().str.upper()
    dtm["adminLevel"] = pd.to_numeric(dtm.get("adminLevel"), errors="coerce")
    dtm["numPresentIdpInd"] = pd.to_numeric(dtm.get("numPresentIdpInd"), errors="coerce").fillna(0.0)
    allowed_idp_provinces = _idp_scope_provinces()
    reason_ok = _iom_reason_bucket(dtm["displacementReason"], reason_bucket)

    valid_provinces = set(admin["province"].dropna().unique().tolist())
    dtm = dtm[
        dtm["reportingDate"].notna()
        & reason_ok
        & (dtm["operationStatus"].isin(["Active", ""]))
        & (dtm["province"].isin(allowed_idp_provinces))
        & (dtm["province"].isin(valid_provinces))
        & (dtm["adminLevel"] == 2)
        & dtm["district"].notna()
        & (~dtm["district"].isin(["", "Not available"]))
    ].copy()

    base_posts = admin[["province", "district", "district_pcode", "admin_post"]].drop_duplicates().copy()
    areas = _admin_post_area_km2(admin)
    base_posts = base_posts.merge(areas, on=["province", "admin_post"], how="left")
    base_posts["area_km2"] = pd.to_numeric(base_posts["area_km2"], errors="coerce").fillna(0.0).clip(lower=0.0)

    if len(dtm) == 0:
        return pd.concat(
            [
                base_posts[["province", "admin_post"]]
                .assign(period_key=pk, **{value_name: 0.0})
                for pk in RUN_PERIODS
            ],
            ignore_index=True,
        )

    dtm["month_key"] = dtm["reportingDate"].dt.to_period("M")
    anc = None
    if ancillary is not None and len(ancillary) > 0 and weight_col in ancillary.columns:
        anc = ancillary[["province", "admin_post", "period_key", weight_col]].copy()
        anc["province"] = normalize_key(anc["province"])
        anc["admin_post"] = normalize_key(anc["admin_post"])
        anc["period_key"] = normalize_key(anc["period_key"]).str.lower()
        anc[weight_col] = pd.to_numeric(anc[weight_col], errors="coerce")

    out = []
    for pk in RUN_PERIODS:
        cfg = TIMELINES[pk]
        ws = pd.Timestamp(cfg["endline_window_start"])
        we = pd.Timestamp(cfg["endline_window_end"])
        sub = dtm[(dtm["reportingDate"] >= ws) & (dtm["reportingDate"] <= we)].copy()
        posts = base_posts.copy()
        posts["period_key"] = pk

        # Ancillary intensity (pop density) × area → proxy population weight.
        posts["_anc"] = np.nan
        if anc is not None:
            a_pk = anc[anc["period_key"].eq(pk)][["province", "admin_post", weight_col]]
            posts = posts.merge(a_pk, on=["province", "admin_post"], how="left")
            posts["_anc"] = pd.to_numeric(posts[weight_col], errors="coerce")
            posts = posts.drop(columns=[weight_col], errors="ignore")
        posts["_anc"] = posts["_anc"].fillna(0.0).clip(lower=0.0)
        posts["weight"] = posts["_anc"] * posts["area_km2"]
        # If pop×area is all-zero in a district, fall back to area-only weights.
        wsum_d = posts.groupby(["province", "district_pcode"], dropna=False)["weight"].transform("sum")
        area_fallback = wsum_d <= 0
        posts.loc[area_fallback, "weight"] = posts.loc[area_fallback, "area_km2"]

        if len(sub) == 0:
            posts["idp_conflict_pressure"] = 0.0
            out.append(posts)
            continue

        # Monthly district totals, then average across months in window.
        d2_month = (
            sub.groupby(["province", "district_pcode", "district", "month_key"], as_index=False)["numPresentIdpInd"]
            .sum()
            .rename(columns={"numPresentIdpInd": "idp_conflict_pressure_month"})
        )
        d2 = (
            d2_month.groupby(["province", "district_pcode", "district"], as_index=False)["idp_conflict_pressure_month"]
            .mean()
            .rename(columns={"idp_conflict_pressure_month": "idp_conflict_pressure"})
        )
        z = _dasymetric_allocate_district_to_posts(posts, d2, value_col="idp_conflict_pressure", weight_col="weight")

        d2_pcodes = set(d2["district_pcode"].dropna().astype(str).str.strip().str.upper())
        miss = ~z["district_pcode"].astype(str).str.strip().str.upper().isin(d2_pcodes)
        n_fallback = int(miss.sum())
        n_total = int(len(z))
        n_anc_ok = int((z["_anc"] > 0).sum()) if "_anc" in z.columns else 0
        if verbose and n_total > 0:
            pct_fallback = 100.0 * n_fallback / n_total
            print(
                f"IDP dasymetric district→post ({reason_bucket}, {pk}): "
                f"weight={weight_col}×area; posts with ancillary>0: {n_anc_ok}/{n_total}; "
                f"unmatched districts: {n_fallback}/{n_total} ({pct_fallback:.1f}%) → 0."
            )
        elif verbose:
            print(f"IDP dasymetric district→post ({pk}): 0/0 rows (no admin-post rows).")
        if verbose and n_fallback > 0:
            fb_by_prov = (
                z.assign(_fallback=miss)
                .groupby("province", as_index=False)["_fallback"]
                .sum()
                .rename(columns={"_fallback": "n_fallback"})
            )
            fb_by_prov = fb_by_prov[fb_by_prov["n_fallback"] > 0].copy()
            if len(fb_by_prov) > 0:
                print("  By province (unmatched row count):")
                for _, rr in fb_by_prov.sort_values("n_fallback", ascending=False).iterrows():
                    print(f"    - {rr['province']}: {int(rr['n_fallback'])}")
        z["idp_conflict_pressure"] = pd.to_numeric(z["idp_conflict_pressure"], errors="coerce").fillna(0.0)
        z["period_key"] = pk
        out.append(z)
    out_df = pd.concat(out, ignore_index=True)
    return out_df[["province", "admin_post", "period_key", "idp_conflict_pressure"]].rename(
        columns={"idp_conflict_pressure": value_name}
    )

# =============================================================================
# Reported models M1-M5b and supplemental IE1-IE5
# ACLED is standardized log1p(events per month), with Riots removed.
# =============================================================================
_SKIP_INTERACTION = ("cyclone_x_acled",)
_SKIP_INTERACTION_CABO = ("cyclone_x_acled", CABO_COL)


def build_ie_specs() -> List[NestedSpec]:
    """Supplemental one-predictor models. Each includes population density."""
    return [
        NestedSpec("IE1", "IE1: Conflict + population density", (ACLED_COL, POP_COL), set_name="IE"),
        NestedSpec("IE2", "IE2: Floodplain + population density", ("floodplain_mean_z", POP_COL), set_name="IE"),
        NestedSpec("IE3", "IE3: Cyclone + population density", ("cyclone_endline_window_z", POP_COL), set_name="IE"),
        NestedSpec("IE4", "IE4: Drought + population density", ("spei_endline_window_z", POP_COL), set_name="IE"),
        NestedSpec("IE5", "IE5: Extreme rainfall + population density", ("chirps_endline_window_z", POP_COL), set_name="IE"),
    ]


def build_main_specs(include_iom: bool) -> List[NestedSpec]:
    """M1–M5a are nested. M5b is M4 plus the two IOM reason splits, without the pooled term."""
    climate = (
        "floodplain_mean_z",
        "cyclone_endline_window_z",
        "spei_endline_window_z",
        "chirps_endline_window_z",
        POP_COL,
    )
    m1 = NestedSpec(
        "M1",
        "M1: floodplain + cyclone + drought + extreme rainfall + population density",
        climate,
        set_name="M",
    )
    m2_preds = (ACLED_COL, *climate)
    m2 = NestedSpec("M2", "M2: M1 + Conflict", m2_preds, set_name="M")
    m3_preds = (*m2_preds, "cyclone_x_acled")
    m3 = NestedSpec(
        "M3",
        "M3: M2 + Cyclone x conflict",
        m3_preds,
        set_name="M",
        is_interaction=True,
        skip_zscore=_SKIP_INTERACTION,
    )
    m4_preds = (*m3_preds, CABO_COL)
    m4 = NestedSpec(
        "M4",
        "M4: M3 + Cabo Delgado dummy (Nampula reference)",
        m4_preds,
        set_name="M",
        is_interaction=True,
        skip_zscore=_SKIP_INTERACTION_CABO,
    )
    specs = [m1, m2, m3, m4]
    if include_iom:
        m5_preds = (*m4_preds, IOM_ALL_COL)
        m5 = NestedSpec(
            "M5a",
            "M5a: M4 + IOM all reasons pooled",
            m5_preds,
            set_name="M",
            is_interaction=True,
            is_iom_robustness=True,
            skip_zscore=_SKIP_INTERACTION_CABO,
        )
        m6_preds = (*m4_preds, IOM_IDP_COL, IOM_DISASTER_COL)
        specs.append(m5)
        specs.append(
            NestedSpec(
                "M5b",
                "M5b: M4 + IOM conflict/insecurity/other and natural disaster",
                m6_preds,
                set_name="M",
                is_interaction=True,
                is_iom_robustness=True,
                skip_zscore=_SKIP_INTERACTION_CABO,
            )
        )
    return specs


# =============================================================================
# Post-level frame
# =============================================================================
def _zscore_within_period(frame: pd.DataFrame) -> None:
    """Replace the panel file's z-scores with z-scores from this cross-section."""
    pairs = (
        ("pop_density_period", "pop_density_period_z"),
        ("conflict_endline_window", "conflict_endline_window_z"),
        ("conflict_endline_count", "conflict_endline_count_z"),
        ("conflict_endline_rate", "conflict_endline_rate_z"),
        ("cyclone_endline_window", "cyclone_endline_window_z"),
        ("spei_endline_window", "spei_endline_window_z"),
        ("chirps_endline_window", "chirps_endline_window_z"),
        ("floodplain_mean", "floodplain_mean_z"),
        ("spei_min_drought", "spei_min_drought_z"),
        ("cyclone_holland", "cyclone_holland_z"),
        ("conflict_decay", "conflict_decay_z"),
        ("idp_conflict_pressure", IOM_IDP_COL),
        ("idp_disaster_pressure", IOM_DISASTER_COL),
        ("idp_all_reasons_pressure", IOM_ALL_COL),
        ("pop_growth", "pop_growth_z"),
    )
    for raw, zcol in pairs:
        if raw not in frame.columns:
            continue
        v = pd.to_numeric(frame[raw], errors="coerce")
        sd = float(v.std())
        if not np.isfinite(sd) or sd <= 0:
            frame[zcol] = 0.0
        else:
            frame[zcol] = (v - float(v.mean())) / sd


def build_post_level_frame(
    *,
    include_iom_idp: Optional[bool] = None,
    period_key: str = ENDLINE,
) -> Tuple[pd.DataFrame, Any, gpd.GeoDataFrame]:
    """One cross-section: posts in ``period_key`` only (no short/long average)."""
    if period_key not in TIMELINES:
        raise ValueError(f"Unknown period_key {period_key!r}; expected one of {list(TIMELINES)}")
    include_iom = INCLUDE_IOM_IDP_ROBUSTNESS if include_iom_idp is None else bool(include_iom_idp)
    print("Loading admin boundaries (Cabo Delgado + Nampula)...")
    admin = load_admin()
    print(f"  admin features: {len(admin)}")
    print("Building core panel (ACLED endline count/rate/log1p)...")
    panel = build_panel_core_moz(
        admin,
        conflict_builder=build_conflict_endline_metrics,
        cyclone_builder=None,
        spei_source_col="spei_cumulative_deficit",
    )
    if include_iom:
        print("Attaching IOM DTM IDP pressure (pooled all-reasons, overlapping conflict and disaster)...")
        anc_cols = ["province", "admin_post", "period_key"]
        wcol = IOM_DASYMETRIC_WEIGHT_COL
        if wcol in panel.columns:
            anc_cols = [*anc_cols, wcol]
        else:
            print(f"  Warning: {wcol!r} missing on panel; dasymetric falls back to area-only weights.")
        anc = panel[anc_cols].copy()
        for bucket, raw_name in (
            ("conflict", "idp_conflict_pressure"),
            ("disaster", "idp_disaster_pressure"),
            ("all", "idp_all_reasons_pressure"),
        ):
            idp = build_idp_conflict_pressure_lag_split(
                admin,
                ancillary=anc,
                weight_col=wcol,
                verbose=True,
                reason_bucket=bucket,
                value_name=raw_name,
            )
            panel = panel.merge(idp, on=["province", "admin_post", "period_key"], how="left")
            panel[raw_name] = np.log1p(
                pd.to_numeric(panel[raw_name], errors="coerce").fillna(0.0).clip(lower=0.0)
            )
    if "pop_growth" in panel.columns:
        panel["pop_growth"] = pd.to_numeric(panel["pop_growth"], errors="coerce")
    if "spei_min" in panel.columns:
        panel["spei_min_drought"] = -pd.to_numeric(panel["spei_min"], errors="coerce")
    try:
        print("Building alternate cyclone (Holland post-landfall) and conflict (distance decay)...")
        holland = build_cyclone_holland(admin)
        decay = build_conflict_distance_decay(admin)
        panel = panel.merge(holland, on=["province", "admin_post", "period_key"], how="left")
        panel = panel.merge(decay, on=["province", "admin_post", "period_key"], how="left")
        panel["cyclone_holland"] = pd.to_numeric(panel["cyclone_holland"], errors="coerce").fillna(0.0)
        panel["conflict_decay"] = pd.to_numeric(panel["conflict_decay"], errors="coerce").fillna(0.0)
    except Exception as exc:
        print(f"  Warning: alternate exposures were not built: {exc}")

    needed = [
        "admin_post_id",
        "province",
        "outcome",
        "conflict_endline_count",
        "conflict_endline_rate",
        "conflict_endline_window",
        ACLED_COL,
        POP_COL,
        "floodplain_mean_z",
        "cyclone_endline_window_z",
        "spei_endline_window_z",
        "chirps_endline_window_z",
    ]
    optional_map_cols = [
        c
        for c in (
            "pop_growth",
            "pop_density_period",
            "spei_min",
            "spei_min_drought",
            "spei_min_drought_z",
            "cyclone_holland",
            "cyclone_holland_z",
            "conflict_decay",
            "conflict_decay_z",
        )
        if c in panel.columns
    ]
    if include_iom:
        optional_map_cols.extend(
            [
                c
                for c in (
                    "idp_conflict_pressure",
                    "idp_disaster_pressure",
                    "idp_all_reasons_pressure",
                )
                if c in panel.columns
            ]
        )
    missing = [c for c in needed if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")

    sub = panel.loc[panel["period_key"].astype(str).str.lower().eq(period_key)].copy()
    if sub.empty:
        raise ValueError(f"No complete posts for period_key={period_key!r}.")
    if sub["admin_post_id"].duplicated().any():
        raise ValueError(
            f"Duplicate admin posts in period {period_key!r}; refusing to average windows."
        )
    _zscore_within_period(sub)
    for col in ("spei_min_drought_z", "cyclone_holland_z", "conflict_decay_z"):
        if col in sub.columns and col not in optional_map_cols:
            optional_map_cols.append(col)
    if {ACLED_COL, "cyclone_endline_window_z"}.issubset(sub.columns):
        sub["cyclone_x_acled"] = pd.to_numeric(sub["cyclone_endline_window_z"], errors="coerce") * pd.to_numeric(
            sub[ACLED_COL], errors="coerce"
        )
    keep_cols = [
        "admin_post_id",
        "province",
        *needed[2:],
        "cyclone_x_acled",
        IOM_IDP_COL,
        IOM_DISASTER_COL,
        IOM_ALL_COL,
        *optional_map_cols,
    ]
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in sub.columns]
    post = sub[keep_cols].copy()
    post["period_key"] = period_key
    prov = post["province"].astype(str).str.strip().str.casefold()
    post[CABO_COL] = prov.eq("cabo delgado").astype(float)
    n_cabo = int(post[CABO_COL].sum())
    if n_cabo == 0 or n_cabo == len(post):
        raise ValueError(
            f"Cabo Delgado dummy is {n_cabo} of {len(post)} posts. "
            f"Province values: {sorted(post['province'].astype(str).unique().tolist())}"
        )
    print(f"  Cabo Delgado dummy: {n_cabo} posts; Nampula reference: {len(post) - n_cabo}")

    poly = _post_polygons_by_admin_id(admin)
    g = poly[["admin_post_id", "geometry"]].drop_duplicates("admin_post_id")
    post = post.merge(g[["admin_post_id"]], on="admin_post_id", how="inner").sort_values("admin_post_id")
    g = g[g["admin_post_id"].isin(post["admin_post_id"])].sort_values("admin_post_id").reset_index(drop=True)
    post = post.reset_index(drop=True)

    w_variants = _build_w_variants(g)
    if PRIMARY_W not in w_variants:
        raise ValueError(f"Primary weights {PRIMARY_W!r} missing; have {list(w_variants)}")
    cfg = TIMELINES[period_key]
    print(
        f"  post-level N = {len(post)}; {cfg['label']}; "
        f"cultivated {cfg['cultivated_suffix']}; exposures {cfg['endline_window_start']}.."
        f"{cfg['endline_window_end']}; W = {PRIMARY_W}; IOM in M5a/M5b = {include_iom}"
    )
    return post, w_variants[PRIMARY_W], g


def _design_matrix(
    post: pd.DataFrame,
    predictors: Sequence[str],
    *,
    skip_zscore: Sequence[str] = (),
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    skip = set(skip_zscore)
    cols = ["outcome", *predictors]
    d = post[["admin_post_id", *cols]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = _zscore(d["outcome"].to_numpy(dtype=float))
    x = np.zeros((len(d), len(predictors)), dtype=float)
    names: List[str] = []
    for j, col in enumerate(predictors):
        v = d[col].to_numpy(dtype=float)
        x[:, j] = v if col in skip else _zscore(v)
        names.append(col)
    return y, x, names, d


def _ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    X = np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y - fitted) ** 2)) / ss_tot


def _ols_abs_t(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Absolute t statistics for each column of ``x`` in an OLS with intercept."""
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = int(len(y))
    design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = max(n - design.shape[1], 1)
    sigma2 = float(np.sum(resid ** 2) / dof)
    try:
        xtx_inv = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return np.full(x.shape[1], np.nan)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_abs = np.abs(beta / se)
    return np.asarray(t_abs[1:], dtype=float)


def predictive_contribution_table(y: np.ndarray, x: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Drop-one predictive power, plus an LMG decomposition.

    O'Loughlin et al. (2014, PNAS), following Ward, Greenhill, and Bakke (2010),
    refit the model once with each covariate removed and record how much
    in-sample predictive power falls. Their outcome was a conflict count, so
    they measured that fall with AUC. Cultivated-share change is continuous, so
    the same drop-one comparison uses the fall in R². Absolute t is stored so
    the figure can plot significance against that predictive contribution.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    p = int(x.shape[1])
    full_r2 = _ols_r2(y, x)
    abs_t = _ols_abs_t(y, x)
    # Same Lindeman–Merenda–Gold average as enumerating all p! entry orders,
    # computed from the 2^p subset R² values. Listing orders is p! regressions
    # (9 predictors is about 3 million), which stalls the run.
    r2_of: Dict[int, float] = {0: 0.0}

    def _subset_r2(mask: int) -> float:
        cached = r2_of.get(mask)
        if cached is not None:
            return cached
        cols = [j for j in range(p) if mask & (1 << j)]
        val = _ols_r2(y, x[:, cols]) if cols else 0.0
        r2_of[mask] = float(val)
        return r2_of[mask]

    fact = [1]
    for i in range(1, p + 1):
        fact.append(fact[-1] * i)
    lmg = np.zeros(p, dtype=float)
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for subset in range(1 << (p - 1)):
            mask = 0
            for bit, idx in enumerate(others):
                if subset & (1 << bit):
                    mask |= 1 << idx
            k = bin(mask).count("1")
            weight = fact[k] * fact[p - k - 1] / fact[p]
            lmg[j] += weight * (_subset_r2(mask | (1 << j)) - _subset_r2(mask))
    rows = []
    for j, name in enumerate(names):
        x_drop = np.delete(x, j, axis=1) if p > 1 else np.zeros((len(y), 0))
        drop_r2 = _ols_r2(y, x_drop)
        rows.append(
            {
                "term": name,
                "term_label": _term_label(str(name)),
                "abs_t": float(abs_t[j]) if j < len(abs_t) else float("nan"),
                "lmg_r2": float(lmg[j]),
                "lmg_share_of_r2": float(lmg[j] / full_r2) if np.isfinite(full_r2) and full_r2 > 0 else np.nan,
                "drop_one_delta_r2": float(full_r2 - drop_r2) if np.isfinite(full_r2) and np.isfinite(drop_r2) else np.nan,
                "full_model_r2": float(full_r2),
            }
        )
    return pd.DataFrame(rows).sort_values("drop_one_delta_r2", ascending=False)


def _predictive_plot_label(term: str, fallback: str) -> str:
    """Short names for the predictive-power figure so nearby points can be labeled."""
    return _term_label(term) if str(term) in TERM_LABELS else fallback


def _place_scatter_labels(fig, ax, xs: np.ndarray, ys: np.ndarray, labels: Sequence[str]) -> None:
    """Put each label just beside its point, with no connecting line."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer)
    occupied: List[Any] = []
    legend = ax.get_legend()
    if legend is not None:
        occupied.append(legend.get_window_extent(renderer).expanded(1.04, 1.15))
    for x_i, y_i in zip(xs, ys):
        if not (np.isfinite(x_i) and np.isfinite(y_i)):
            continue
        px, py = ax.transData.transform((float(x_i), float(y_i)))
        occupied.append(Bbox.from_extents(px - 8, py - 8, px + 8, py + 8))

    finite = [
        (float(x), float(y), str(lab))
        for x, y, lab in zip(xs, ys, labels)
        if np.isfinite(x) and np.isfinite(y)
    ]
    x0, x1 = ax.get_xlim()
    xspan = max(x1 - x0, 1e-9)

    def _inside(bb) -> bool:
        return axbb.x0 + 1 <= bb.x0 and bb.x1 <= axbb.x1 - 1 and axbb.y0 + 1 <= bb.y0 and bb.y1 <= axbb.y1 - 1

    def _overlaps(bb) -> bool:
        return any(bb.overlaps(prev) for prev in occupied)

    # Nearby points try different sides first, and only small offsets.
    angles = (70, -75, 20, 160, -20, 110, -140, 0, 180, 90, -90)
    distances = (8, 13, 18, 24, 32)
    order = sorted(range(len(finite)), key=lambda i: finite[i][0])
    for rank, i in enumerate(order):
        x, y, lab = finite[i]
        prefer_left = x > x0 + 0.72 * xspan
        shift = 15 * (rank % 4)
        cands = []
        for dist in distances:
            for ang0 in angles:
                ang = np.deg2rad(ang0 + shift + (180 if prefer_left else 0))
                dx = dist * np.cos(ang)
                dy = dist * np.sin(ang)
                if prefer_left and dx > 0:
                    dx = -abs(dx)
                cands.append((float(dx), float(dy)))
        chosen = None
        for dx, dy in cands:
            ha = "left" if dx >= 0 else "right"
            dx = max(dx, 6) if ha == "left" else min(dx, -6)
            trial = ax.annotate(
                lab,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=8,
                color="0.15",
                zorder=4,
                annotation_clip=False,
            )
            fig.canvas.draw()
            bb = trial.get_window_extent(renderer).expanded(1.04, 1.25)
            if _inside(bb) and not _overlaps(bb):
                occupied.append(bb)
                chosen = trial
                break
            trial.remove()
        if chosen is None:
            ha = "right" if prefer_left else "left"
            dx = -8 if prefer_left else 8
            ann = ax.annotate(
                lab,
                xy=(x, y),
                xytext=(dx, 8),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=8,
                color="0.15",
                zorder=4,
                annotation_clip=False,
            )
            fig.canvas.draw()
            occupied.append(ann.get_window_extent(renderer).expanded(1.04, 1.25))


def _predictive_xy_labels(tab: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs = pd.to_numeric(tab["abs_t"], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(tab["drop_one_delta_r2"], errors="coerce").to_numpy(dtype=float)
    if "term" in tab.columns:
        labels = [
            _predictive_plot_label(term, fallback)
            for term, fallback in zip(tab["term"].astype(str), tab["term_label"].astype(str))
        ]
    else:
        labels = tab["term_label"].astype(str).tolist()
    return xs, ys, labels


def _draw_predictive_scatter(ax, tab: pd.DataFrame, *, title: str, color: str):
    """|t| versus drop-one ΔR² on an existing axes."""
    xs, ys, labels = _predictive_xy_labels(tab)
    ax.axhline(0.0, color="0.45", ls="--", lw=0.9, zorder=1)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if int(ok.sum()) >= 1 and float(np.sum(xs[ok] ** 2)) > 0:
        slope = float(np.sum(xs[ok] * ys[ok]) / np.sum(xs[ok] ** 2))
        x_hi = float(np.nanmax(xs[ok]))
        xline = np.linspace(0.0, x_hi * 1.05, 40)
        ax.plot(xline, slope * xline, color="0.25", lw=1.0, label="Best fit through origin", zorder=2)
        y_hi = float(np.nanmax(ys[ok]))
        y_lo = min(0.0, float(np.nanmin(ys[ok])))
        ax.set_xlim(0, x_hi * 1.12)
        ax.set_ylim(y_lo - max(0.01, 0.12 * y_hi), y_hi * 1.22)
    ax.scatter(xs, ys, s=36, color=color, zorder=3)
    ax.set_xlabel("Statistical significance (|t|)")
    ax.set_ylabel("Change in predictive power (drop-one ΔR²)")
    ax.set_xlim(left=0)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    return xs, ys, labels


def _plot_predictive_power(tab: pd.DataFrame, path: Path, model_id: str) -> None:
    """|t| versus drop-one ΔR², the continuous-outcome form of O'Loughlin Fig. 3."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    xs, ys, labels = _draw_predictive_scatter(
        ax, tab, title=f"{model_id}: significance vs predictive power", color="#2c5f8a"
    )
    fig.tight_layout()
    _place_scatter_labels(fig, ax, xs, ys, labels)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_alternate_predictive(tabs: Sequence[Tuple[str, str, pd.DataFrame]], path: Path) -> None:
    """Three alternate-specification predictive-power panels in one figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tabs), figsize=(5.6 * len(tabs), 5.8), sharey=False)
    if len(tabs) == 1:
        axes = [axes]
    drawn = []
    for ax, (title, color, tab) in zip(axes, tabs):
        xs, ys, labels = _draw_predictive_scatter(ax, tab, title=title, color=color)
        drawn.append((ax, xs, ys, labels))
    fig.suptitle("M5b alternate specifications: significance vs predictive power", fontsize=12)
    fig.tight_layout()
    for ax, xs, ys, labels in drawn:
        _place_scatter_labels(fig, ax, xs, ys, labels)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_predictive_contribution(out: Path, design_cache: Dict[str, Tuple]) -> None:
    """Drop-one predictive-power figure for M5b, plus the three alternate specifications."""
    parts: List[pd.DataFrame] = []
    alt_panels: List[Tuple[str, str, pd.DataFrame]] = []
    model_ids = (DECISION_MODEL, *[item[0] for item in ALT_MODELS])
    for mid in model_ids:
        packed = design_cache.get(mid)
        if not packed:
            continue
        y, x, names = packed[0], packed[1], packed[2]
        tab = predictive_contribution_table(y, x, names)
        tab.insert(0, "model_id", mid)
        parts.append(tab)
        if mid == DECISION_MODEL:
            _plot_predictive_power(tab, out / f"fig_erl_{mid}_predictive_contribution.png", mid)
            print(f"  figure: fig_erl_{mid}_predictive_contribution.png")
        else:
            legend = next(item[5] for item in ALT_MODELS if item[0] == mid)
            color = next(item[4] for item in ALT_MODELS if item[0] == mid)
            alt_panels.append((legend, color, tab))
    if alt_panels:
        _plot_alternate_predictive(alt_panels, out / "fig_erl_M5b_alt_predictive_contribution.png")
        print("  figure: fig_erl_M5b_alt_predictive_contribution.png")
    if not parts:
        return
    out_tab = pd.concat(parts, ignore_index=True)
    to_csv_sig(out_tab, out / "ERL_variable_predictive_contribution.csv")


# =============================================================================
# OLS + spatial diagnostics / models
# =============================================================================
def fit_ols_with_diagnostics(
    y: np.ndarray,
    x: np.ndarray,
    names: List[str],
    w: Any,
    *,
    model_id: str,
    model_label: str,
    set_name: str = "",
    weights_name: str = PRIMARY_W,
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    from esda.moran import Moran
    from scipy import stats as scipy_stats
    from spreg import OLS
    from spreg.diagnostics_sp import LMtests

    ols = OLS(y=y, x=x, w=w, name_y="outcome", name_x=names, spat_diag=False, nonspat_diag=False)
    betas = np.asarray(ols.betas).ravel()
    se = np.asarray(getattr(ols, "std_err", np.full_like(betas, np.nan))).ravel()
    zstats = getattr(ols, "t_stat", None) or getattr(ols, "z_stat", None)
    coef_rows: List[Dict[str, Any]] = []
    for i, term in enumerate(["CONSTANT", *names]):
        if zstats is not None and i < len(zstats):
            z_i, p_i = float(zstats[i][0]), float(zstats[i][1])
        else:
            z_i = betas[i] / se[i] if (i < len(se) and se[i] != 0) else np.nan
            p_i = float(2 * (1 - scipy_stats.norm.cdf(abs(z_i)))) if np.isfinite(z_i) else np.nan
        coef_rows.append(
            {
                "stage": "OLS",
                "set_name": set_name,
                "model_id": model_id,
                "model_label": model_label,
                "term": term,
                "term_label": _term_label(term),
                "coef": float(betas[i]),
                "se": float(se[i]) if i < len(se) else np.nan,
                "z_or_t": z_i,
                "p_value": p_i,
                "sig": _p_stars(p_i),
                "n_posts": int(len(y)),
                "weights": weights_name,
            }
        )
    resid = np.asarray(getattr(ols, "u", np.full_like(y, np.nan))).ravel()
    np.random.seed(RUN_SEED)
    mi = Moran(resid, w, permutations=999)
    lm = LMtests(ols, w, tests=["lme", "lml", "rlme", "rlml", "sarma"])
    diag = {
        "stage": "OLS",
        "set_name": set_name,
        "model_id": model_id,
        "model_label": model_label,
        "n_posts": int(len(y)),
        "weights": weights_name,
        "r2": float(getattr(ols, "r2", np.nan)),
        "adj_r2": float(getattr(ols, "ar2", getattr(ols, "adj_r2", np.nan))),
        "log_likelihood": float(getattr(ols, "logll", np.nan)),
        "aic": float(getattr(ols, "aic", np.nan)),
        "resid_moran_I": float(mi.I),
        "resid_moran_p_sim": float(getattr(mi, "p_sim", np.nan)),
        "lm_lag_stat": _safe_tuple_stat(getattr(lm, "lml", None))[0],
        "lm_lag_p": _safe_tuple_stat(getattr(lm, "lml", None))[1],
        "lm_error_stat": _safe_tuple_stat(getattr(lm, "lme", None))[0],
        "lm_error_p": _safe_tuple_stat(getattr(lm, "lme", None))[1],
        "robust_lm_lag_stat": _safe_tuple_stat(getattr(lm, "rlml", None))[0],
        "robust_lm_lag_p": _safe_tuple_stat(getattr(lm, "rlml", None))[1],
        "robust_lm_error_stat": _safe_tuple_stat(getattr(lm, "rlme", None))[0],
        "robust_lm_error_p": _safe_tuple_stat(getattr(lm, "rlme", None))[1],
        "lm_sarma_stat": _safe_tuple_stat(getattr(lm, "sarma", None))[0],
        "lm_sarma_p": _safe_tuple_stat(getattr(lm, "sarma", None))[1],
    }
    return pd.DataFrame(coef_rows), diag, resid


# An AIC gap smaller than this is a tie (spatial-epidemiology.org step-by-step).
AIC_TIE_GAP = 2.0


def both_lag_and_error_indicated(diag: Dict[str, Any], alpha: float = 0.05) -> bool:
    """True only when both robust LM tests reject (higher-order / omitted covariate)."""
    rlml_p = float(diag.get("robust_lm_lag_p", np.nan))
    rlme_p = float(diag.get("robust_lm_error_p", np.nan))
    rlag_sig = np.isfinite(rlml_p) and rlml_p < alpha
    rerr_sig = np.isfinite(rlme_p) and rlme_p < alpha
    return rlag_sig and rerr_sig


def anselin_florax_robust_choice(full_diag: Dict[str, Any], alpha: float = 0.05) -> Tuple[str, str]:
    """Anselin–Florax rule on the robust LM p-values only.

    Classic LM tests only establish that some spatial term is needed. They do
    not separate lag from error. Returns SAR, SEM, both, neither, or OLS.
    """
    lml_p = float(full_diag.get("lm_lag_p", np.nan))
    lme_p = float(full_diag.get("lm_error_p", np.nan))
    rlml_p = float(full_diag.get("robust_lm_lag_p", np.nan))
    rlme_p = float(full_diag.get("robust_lm_error_p", np.nan))
    lag_sig = np.isfinite(lml_p) and lml_p < alpha
    err_sig = np.isfinite(lme_p) and lme_p < alpha
    rlag_sig = np.isfinite(rlml_p) and rlml_p < alpha
    rerr_sig = np.isfinite(rlme_p) and rlme_p < alpha
    bits = [
        f"Classical LM-lag p={lml_p:.4g}; LM-error p={lme_p:.4g}.",
        f"Robust LM-lag p={rlml_p:.4g}; Robust LM-error p={rlme_p:.4g}.",
    ]
    if not lag_sig and not err_sig:
        return "OLS", " ".join(
            bits + ["Neither classical LM is significant, so no spatial term is indicated. OLS retained."]
        )
    if rlag_sig and not rerr_sig:
        return "SAR", " ".join(bits + ["Robust LM-lag only → SAR."])
    if rerr_sig and not rlag_sig:
        return "SEM", " ".join(bits + ["Robust LM-error only → SEM."])
    if rlag_sig and rerr_sig:
        return "both", " ".join(
            bits
            + [
                "Both robust LM tests reject. A single SAR or SEM is not identified;"
                " this points to a higher-order process or an omitted spatial covariate."
            ]
        )
    return "neither", " ".join(
        bits
        + [
            "Neither robust LM rejects. Classic tests can still reject because each has"
            " power against the other form. This branch is not forced into SAR or SEM:"
            " the smaller classic p-value is not used, and AIC is not used to break the tie."
        ]
    )


def resolve_lag_vs_error(
    robust_choice: str,
    robust_note: str,
    sar_summary: Dict[str, Any],
    sem_summary: Dict[str, Any],
) -> Tuple[str, str]:
    """Confirm the robust-LM verdict with AIC. Lower AIC is better.

    A gap under 2 is a tie. Pseudo-R² is logged and not used to choose, because
    the lag and error pseudo-R² are not comparable. When the robust tests are
    ambiguous and AIC is a tie, SEM is reported: its coefficients keep their
    ordinary marginal-effect meaning.
    """
    aic_sar = float(sar_summary.get("aic", np.nan))
    aic_sem = float(sem_summary.get("aic", np.nan))
    pr2_sar = float(sar_summary.get("pr2", np.nan))
    pr2_sem = float(sem_summary.get("pr2", np.nan))
    gap = abs(aic_sar - aic_sem) if np.isfinite(aic_sar) and np.isfinite(aic_sem) else float("nan")
    tie = (not np.isfinite(gap)) or gap < AIC_TIE_GAP
    aic_choice = "SAR" if np.isfinite(aic_sar) and np.isfinite(aic_sem) and aic_sar < aic_sem else "SEM"
    aic_bits = (
        f"AIC SAR={aic_sar:.2f}, SEM={aic_sem:.2f}, |ΔAIC|={gap:.2f}. "
        f"Spatial pseudo-R2 SAR={pr2_sar:.4f}, SEM={pr2_sem:.4f} "
        "(secondary check only; the two pseudo-R2 are not comparable)."
    )
    moran_bits = (
        f"Post-fit residual Moran p: SAR={float(sar_summary.get('resid_moran_p_sim', np.nan)):.4g}, "
        f"SEM={float(sem_summary.get('resid_moran_p_sim', np.nan)):.4g}."
    )
    if robust_choice == "OLS":
        return "OLS", robust_note
    if robust_choice in {"neither", "both", "unidentified"}:
        return "unidentified", (
            f"{robust_note} AIC is recorded when the fits exist and is not used to force SAR or SEM"
            " on this branch."
        )
    if robust_choice in {"SAR", "SEM"}:
        if tie:
            verdict = (
                f"{robust_note} {aic_bits} |ΔAIC| under {AIC_TIE_GAP:.0f} is a tie, so the robust LM verdict stands. {moran_bits}"
            )
        elif aic_choice == robust_choice:
            verdict = f"{robust_note} {aic_bits} AIC agrees with the robust LM. {moran_bits}"
        else:
            verdict = (
                f"{robust_note} {aic_bits} AIC prefers {aic_choice}. The robust LM verdict is kept; "
                f"the disagreement is a flag to investigate, not a reason to switch. {moran_bits}"
            )
        chosen_sum = sar_summary if robust_choice == "SAR" else sem_summary
        resid_p = float(chosen_sum.get("resid_moran_p_sim", np.nan))
        if np.isfinite(resid_p) and resid_p < 0.05:
            verdict += (
                " Residual Moran's I of the selected model is still significant, so one lag or error"
                " term did not absorb the spatial structure."
            )
        return robust_choice, verdict
    return "unidentified", (
        f"{robust_note} {aic_bits} The robust LM tests do not select one form, so SAR and SEM are not"
        f" forced. {moran_bits}"
    )


def force_form_by_aic(
    note: str,
    sar_summary: Dict[str, Any],
    sem_summary: Dict[str, Any],
) -> Tuple[str, str]:
    """Pick SAR or SEM by AIC after robust LM and queen both fail to name a form.

    A gap under 2 is a tie and SEM is reported, because its coefficients keep
    their ordinary marginal-effect meaning.
    """
    aic_sar = float(sar_summary.get("aic", np.nan))
    aic_sem = float(sem_summary.get("aic", np.nan))
    pr2_sar = float(sar_summary.get("pr2", np.nan))
    pr2_sem = float(sem_summary.get("pr2", np.nan))
    gap = abs(aic_sar - aic_sem) if np.isfinite(aic_sar) and np.isfinite(aic_sem) else float("nan")
    tie = (not np.isfinite(gap)) or gap < AIC_TIE_GAP
    aic_choice = "SAR" if np.isfinite(aic_sar) and np.isfinite(aic_sem) and aic_sar < aic_sem else "SEM"
    choice = "SEM" if tie else aic_choice
    moran_sar = float(sar_summary.get("resid_moran_p_sim", np.nan))
    moran_sem = float(sem_summary.get("resid_moran_p_sim", np.nan))
    if tie:
        how = (
            f"|ΔAIC|={gap:.2f} is under {AIC_TIE_GAP:.0f}, a tie, so SEM is reported because its"
            " coefficients keep their ordinary marginal-effect meaning."
        )
    else:
        how = f"|ΔAIC|={gap:.2f} is at least {AIC_TIE_GAP:.0f}, so {choice} is the reported form."
    verdict = (
        f"{note} AIC tie-break on knn5: SAR={aic_sar:.2f}, SEM={aic_sem:.2f}. {how} "
        f"Spatial pseudo-R2 SAR={pr2_sar:.4f}, SEM={pr2_sem:.4f} "
        "(recorded only; the two pseudo-R2 are not comparable). "
        f"Post-fit residual Moran p: SAR={moran_sar:.4g}, SEM={moran_sem:.4g}."
    )
    chosen = sem_summary if choice == "SEM" else sar_summary
    resid_p = float(chosen.get("resid_moran_p_sim", np.nan))
    if np.isfinite(resid_p) and resid_p < 0.05:
        verdict += (
            " Residual Moran's I of the selected model is still significant, so this AIC choice"
            " did not absorb the spatial structure."
        )
    return choice, verdict


def apply_queen_check(
    knn_choice: str,
    knn_note: str,
    queen_choice: str,
    queen_note: str,
) -> Tuple[str, str, str]:
    """Decide whether knn5 stands or queen contiguity replaces it.

    Returns (robust choice passed to the AIC check, weights name, note).
    A knn5 'neither' result is passed to the AIC tie-break unless queen names SAR or SEM.
    """
    queen_bit = f" Queen check: {queen_note}"
    if knn_choice in {"SAR", "SEM", "OLS"}:
        return knn_choice, PRIMARY_W, knn_note + queen_bit
    if knn_choice == "both":
        return "both", PRIMARY_W, knn_note + " The form is left unidentified." + queen_bit
    if queen_choice in {"SAR", "SEM"}:
        return queen_choice, "queen", (
            knn_note
            + " Under knn5 that neither branch is not forced into SAR or SEM."
            + f" Queen contiguity identifies {queen_choice}."
            + queen_bit
            + " Spatial refits use the queen weights."
        )
    return "unidentified", PRIMARY_W, (
        knn_note
        + " Queen contiguity does not identify a single form either."
        + queen_bit
        + " Both models are fit on knn5 and the lower AIC is the reported form."
    )


def fit_spatial_model(
    y: np.ndarray,
    x: np.ndarray,
    names: List[str],
    w: Any,
    *,
    form: str,
    model_id: str,
    model_label: str,
    set_name: str = "",
    weights_name: str = PRIMARY_W,
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    from esda.moran import Moran
    from spreg import ML_Error, ML_Lag

    if form == "SAR":
        mdl = ML_Lag(y=y, x=x, w=w, name_y="outcome", name_x=names)
        spatial_term = "rho"
    elif form == "SEM":
        mdl = ML_Error(y=y, x=x, w=w, name_y="outcome", name_x=names)
        spatial_term = "lambda"
    elif form == "SDM":
        if "slx_lags" not in inspect.signature(ML_Lag).parameters:
            raise RuntimeError("spreg ML_Lag lacks slx_lags (SDM unavailable).")
        mdl = ML_Lag(y=y, x=x, w=w, name_y="outcome", name_x=names, slx_lags=1)
        spatial_term = "rho"
    elif form == "SDEM":
        if "slx_lags" not in inspect.signature(ML_Error).parameters:
            raise RuntimeError("spreg ML_Error lacks slx_lags (SDEM unavailable).")
        mdl = ML_Error(y=y, x=x, w=w, name_y="outcome", name_x=names, slx_lags=1)
        spatial_term = "lambda"
    else:
        raise ValueError(form)

    betas = np.asarray(mdl.betas).ravel()
    vm = np.asarray(mdl.vm)
    se = np.sqrt(np.where(np.diag(vm) >= 0, np.diag(vm), np.nan))
    zstats = getattr(mdl, "z_stat", [(np.nan, np.nan)] * len(betas))
    term_names = list(getattr(mdl, "name_x", []) or [])
    if len(term_names) != len(betas):
        term_names = ["CONSTANT", *names]
        while len(term_names) < len(betas) - 1:
            term_names.append(f"x_{len(term_names)}")
        term_names = term_names[: len(betas) - 1] + [spatial_term]

    coef_rows: List[Dict[str, Any]] = []
    for i, term in enumerate(term_names):
        p_i = float(zstats[i][1]) if i < len(zstats) else np.nan
        coef_rows.append(
            {
                "stage": form,
                "set_name": set_name,
                "model_id": model_id,
                "model_label": model_label,
                "term": term,
                "term_label": _term_label(term),
                "coef": float(betas[i]),
                "se": float(se[i]) if i < len(se) else np.nan,
                "z_or_t": float(zstats[i][0]) if i < len(zstats) else np.nan,
                "p_value": p_i,
                "sig": _p_stars(p_i),
                "n_posts": int(len(y)),
                "weights": weights_name,
            }
        )

    resid = np.asarray(getattr(mdl, "u", np.full_like(y, np.nan))).ravel()
    resid_I = resid_p = np.nan
    try:
        np.random.seed(RUN_SEED)
        mi = Moran(resid, w, permutations=999)
        resid_I, resid_p = float(mi.I), float(getattr(mi, "p_sim", np.nan))
    except Exception:
        pass

    summary = {
        "stage": form,
        "set_name": set_name,
        "model_id": model_id,
        "model_label": model_label,
        "n_posts": int(len(y)),
        "weights": weights_name,
        "log_likelihood": float(getattr(mdl, "logll", np.nan)),
        "aic": float(getattr(mdl, "aic", np.nan)),
        "pr2": float(getattr(mdl, "pr2", np.nan)),
        "bic": float(getattr(mdl, "schwarz", np.nan)),
        "resid_moran_I": resid_I,
        "resid_moran_p_sim": resid_p,
    }
    return pd.DataFrame(coef_rows), summary, resid



# =============================================================================
# Reporting helpers
# =============================================================================
def save_table_png(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            # Usual 3 digits for presentation tables (advisor feedback).
            show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
        show[c] = show[c].astype(str).replace({"nan": "", "None": ""})
    # Estimate width from longest cell per column so labels don't collide.
    col_lens = []
    for c in show.columns:
        longest = max([len(str(c))] + [len(str(v)) for v in show[c].tolist()])
        col_lens.append(max(4, min(longest, 28)))
    fig_w = max(8.0, 0.18 * sum(col_lens) + 1.5)
    fig_h = max(2.8, 0.38 * (len(show) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=12)
    tbl = ax.table(
        cellText=show.values.tolist(),
        colLabels=[str(c) for c in show.columns],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.05, 1.45)
    try:
        tbl.auto_set_column_width(list(range(len(show.columns))))
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _forest_plot(
    rows: pd.DataFrame,
    path: Path,
    *,
    title: str,
    label_col: str = "label",
    coef_col: str = "coef",
    se_col: str = "se",
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = rows.copy().reset_index(drop=True)
    if len(d) == 0:
        return
    est = pd.to_numeric(d[coef_col], errors="coerce").to_numpy(dtype=float)
    se = pd.to_numeric(d[se_col], errors="coerce").to_numpy(dtype=float) if se_col in d.columns else np.full(len(d), np.nan)
    labels = d[label_col].astype(str).tolist()
    y = np.arange(len(d))
    h = figsize or (8.0, max(3.0, 0.38 * len(d) + 1.2))
    fig, ax = plt.subplots(figsize=h)
    ax.axvline(0.0, color="0.45", ls="--", lw=0.9)
    lo = est - 1.96 * se
    hi = est + 1.96 * se
    ax.hlines(y, lo, hi, color="#2c5f8a", lw=1.6)
    ax.plot(est, y, "o", color="#2c5f8a", ms=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(title)
    try:
        from matplotlib.ticker import FormatStrFormatter

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _forest_plot_compared(
    groups: Sequence[Tuple[str, str, pd.DataFrame]],
    path: Path,
    *,
    title: str,
) -> None:
    """One forest plot with several models, each in its own color."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order: List[str] = []
    labels: Dict[str, str] = {}
    for _name, _color, frame in groups:
        for _, row in frame.iterrows():
            term = str(row["term"])
            if term not in labels:
                order.append(term)
                labels[term] = str(row.get("term_label") or term)
    if not order:
        return
    y = np.arange(len(order))
    n = len(groups)
    offsets = (np.arange(n) - (n - 1) / 2.0) * 0.22
    fig, ax = plt.subplots(figsize=(9.2, max(4.6, 0.42 * len(order) + 1.4)))
    ax.axvline(0.0, color="0.45", ls="--", lw=0.9, zorder=1)
    for offset, (name, color, frame) in zip(offsets, groups):
        by_term = {str(row["term"]): row for _, row in frame.iterrows()}
        xs, xerr_lo, xerr_hi, ys = [], [], [], []
        for i, term in enumerate(order):
            row = by_term.get(term)
            if row is None:
                continue
            est = float(row["coef"])
            se = float(row["se"]) if pd.notna(row["se"]) else np.nan
            if not np.isfinite(est):
                continue
            ys.append(i + offset)
            xs.append(est)
            half = 1.96 * se if np.isfinite(se) else np.nan
            xerr_lo.append(est - half if np.isfinite(half) else est)
            xerr_hi.append(est + half if np.isfinite(half) else est)
        if not xs:
            continue
        ax.hlines(ys, xerr_lo, xerr_hi, color=color, lw=1.5, zorder=2)
        ax.plot(xs, ys, "o", color=color, ms=6, label=name, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[term] for term in order])
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False)
    try:
        from matplotlib.ticker import FormatStrFormatter

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _choropleth_limits(values: pd.Series, *, diverging: bool) -> Tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0) if diverging else (0.0, 1.0)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if diverging:
        lim = max(abs(vmin), abs(vmax), 1e-9)
        return -lim, lim
    if vmin >= 0:
        vmin = 0.0
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return vmin, vmax


def _label_provinces(ax, g: gpd.GeoDataFrame) -> None:
    """Write Cabo Delgado and Nampula at a point inside each province."""
    if g is None or len(g) == 0 or "geometry" not in g.columns:
        return
    work = g.copy()
    if "province" not in work.columns:
        if "admin_post_id" not in work.columns:
            return
        work["province"] = work["admin_post_id"].astype(str).str.split("||", regex=False).str[0]
    work = work.dropna(subset=["province", "geometry"])
    if len(work) == 0:
        return
    try:
        dissolved = work.dissolve(by="province", as_index=False)
    except Exception:
        return
    for _, row in dissolved.iterrows():
        geom = row.geometry
        name = str(row["province"]).strip()
        if not name or geom is None or geom.is_empty:
            continue
        pt = geom.representative_point()
        ax.annotate(
            name,
            xy=(float(pt.x), float(pt.y)),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=6,
            annotation_clip=True,
        )


def _draw_province_border(ax, g: gpd.GeoDataFrame) -> None:
    """Draw province outlines. The Cabo Delgado–Nampula line matches the coast."""
    if g is None or len(g) == 0 or "geometry" not in getattr(g, "columns", []):
        return
    work = g.copy()
    if "province" not in work.columns:
        if "admin_post_id" not in work.columns:
            return
        work["province"] = work["admin_post_id"].astype(str).str.split("||", regex=False).str[0]
    work = work.dropna(subset=["province", "geometry"])
    if len(work) == 0:
        return
    try:
        dissolved = work.dissolve(by="province", as_index=False)
    except Exception:
        return
    crs = getattr(work, "crs", None)
    line = {"color": "black", "linewidth": 1.15}
    try:
        outer = unary_union(list(dissolved.geometry))
        gpd.GeoSeries([outer], crs=crs).boundary.plot(ax=ax, zorder=5, **line)
    except Exception:
        return
    by_name = {
        str(row["province"]).strip().casefold(): row.geometry
        for _, row in dissolved.iterrows()
    }
    cabo = by_name.get("cabo delgado")
    namp = by_name.get("nampula")
    if cabo is None or namp is None or cabo.is_empty or namp.is_empty:
        return
    minx, miny, maxx, maxy = cabo.bounds
    tol = max(maxx - minx, maxy - miny, 1e-9) * 0.004
    try:
        shared = cabo.boundary.intersection(namp.buffer(tol))
    except Exception:
        return
    if shared is None or shared.is_empty:
        return
    try:
        gpd.GeoSeries([shared], crs=crs).plot(ax=ax, zorder=6, **line)
    except Exception:
        return


def _choropleth_on_ax(
    ax,
    g: gpd.GeoDataFrame,
    values: pd.Series,
    *,
    title: str,
    cmap: str,
    diverging: bool,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    legend: bool = True,
    cbar_label: Optional[str] = None,
) -> Tuple[float, float]:
    """Unlabeled choropleth: no post names, no percent labels, no point overlay."""
    plot_g = g.copy()
    plot_g["_v"] = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if vmin is None or vmax is None:
        auto_min, auto_max = _choropleth_limits(plot_g["_v"], diverging=diverging)
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax
    plot_kw = {}
    if cbar_label:
        plot_kw["legend_kwds"] = {"label": cbar_label}
    plot_g.plot(
        column="_v",
        cmap=cmap,
        linewidth=0.2,
        edgecolor="0.35",
        legend=legend,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "0.85"},
        **plot_kw,
    )
    ax.set_axis_off()
    if cbar_label:
        extras = [a for a in ax.figure.axes if a is not ax]
        if extras:
            extras[-1].set_ylabel(cbar_label)
            extras[-1].yaxis.label.set_visible(True)
    ax.set_title(title, fontsize=11)
    _label_provinces(ax, g)
    _draw_province_border(ax, g)
    return float(vmin), float(vmax)


def _choropleth(
    g: gpd.GeoDataFrame,
    values: pd.Series,
    path: Path,
    *,
    title: str,
    cmap: str = "RdBu_r",
    diverging: bool = True,
    cbar_label: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    _choropleth_on_ax(ax, g, values, title=title, cmap=cmap, diverging=diverging, cbar_label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _choropleth_pair(
    g: gpd.GeoDataFrame,
    left: pd.Series,
    right: pd.Series,
    path: Path,
    *,
    left_title: str,
    right_title: str,
    left_cmap: str,
    right_cmap: str,
    left_diverging: bool,
    right_diverging: bool,
    suptitle: str,
    left_cbar_label: Optional[str] = None,
    right_cbar_label: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4))
    _choropleth_on_ax(
        axes[0], g, left, title=left_title, cmap=left_cmap, diverging=left_diverging, cbar_label=left_cbar_label
    )
    _choropleth_on_ax(
        axes[1], g, right, title=right_title, cmap=right_cmap, diverging=right_diverging, cbar_label=right_cbar_label
    )
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _choropleth_shared(
    g: gpd.GeoDataFrame,
    panels: Sequence[Tuple[pd.Series, str]],
    path: Path,
    *,
    cmap: str,
    diverging: bool,
    suptitle: str,
    cbar_label: str,
) -> None:
    """Several choropleths with one color scale and one colorbar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    stacked = np.concatenate(
        [pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) for values, _ in panels]
    )
    vmin, vmax = _choropleth_limits(pd.Series(stacked), diverging=diverging)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 6.4))
    if n == 1:
        axes = [axes]
    for ax, (values, title) in zip(axes, panels):
        _choropleth_on_ax(
            ax,
            g,
            values,
            title=title,
            cmap=cmap,
            diverging=diverging,
            vmin=vmin,
            vmax=vmax,
            legend=False,
        )
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _worldpop_tif(year: int) -> Path:
    """Cache the 1 km UN-adjusted WorldPop GP grid for ``year``."""
    import urllib.request

    dest = Path(LOCAL_DATA_DIR) / "worldpop" / f"moz_ppp_{year}_1km_Aggregated_UNadj.tif"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = WORLDPOP_1KM_URL.format(year=year)
    print(f"  Downloading WorldPop {year}: {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _zonal_population_sum(gdf: gpd.GeoDataFrame, tif_path: Path) -> np.ndarray:
    """Sum gridded population inside each polygon, in ``gdf`` row order."""
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(tif_path) as src:
        polys = gdf.to_crs(src.crs)
        # Read the full 1 km national grid. Windowed reads crash in this rasterio build.
        arr = src.read(1).astype("float64")
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0] = 0.0
    shapes = ((geom, i + 1) for i, geom in enumerate(polys.geometry))
    ids = rasterize(
        shapes,
        out_shape=arr.shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    sums = np.bincount(ids.ravel(), weights=arr.ravel(), minlength=int(ids.max()) + 1)
    return sums[1:]


def worldpop_proportional_change(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Proportional WorldPop change between the two seasonal windows.

    WorldPop GP publishes one grid per year. The baseline window
    (1 Nov 2016–31 Mar 2017) contains the 2017 layer, and the later window
    (1 Nov 2019–31 Mar 2020) contains the 2020 layer. Change is
    (pop_2020 - pop_2017) / pop_2017 from zonal sums of the 1 km UN-adjusted grids.
    """
    base = _zonal_population_sum(gdf, _worldpop_tif(WORLDPOP_CHANGE_BASE_YEAR))
    end = _zonal_population_sum(gdf, _worldpop_tif(WORLDPOP_CHANGE_END_YEAR))
    change = np.full(len(gdf), np.nan, dtype=float)
    ok = base > 0
    change[ok] = (end[ok] - base[ok]) / base[ok]
    return pd.Series(change, index=gdf.index)


def _cultivated_delta_path(province: str, period: str = CULTIVATED_DELTA_PERIOD) -> Optional[Path]:
    """Local cache of the province multiband cultivated raster, downloading from GCS if needed."""
    name = f"MOZ_cultivated_change_multiband_{province.replace(' ', '_')}_{period}.tif"
    dest = Path(LOCAL_DATA_DIR) / "rasters" / name
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import google.auth
        from google.cloud import storage

        creds, project = google.auth.default()
        client = storage.Client(credentials=creds, project=project)
        blob = client.bucket(GCS_BUCKET).blob(name)
        if not blob.exists(client):
            print(f"  Warning: gs://{GCS_BUCKET}/{name} not found.")
            return None
        print(f"  Downloading gs://{GCS_BUCKET}/{name}")
        blob.download_to_filename(str(dest))
        return dest
    except Exception as exc:
        print(f"  Warning: could not fetch {name}: {exc}")
        return dest if dest.is_file() else None


def _read_delta_for_display(path: Path, *, max_pixels: int = 1_200_000) -> Tuple[np.ndarray, Tuple[float, float, float, float], Any]:
    """Read delta_prob, downsampled, in the raster CRS. Extent is (left, right, bottom, top)."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        band = 3 if src.count >= 3 else 1
        scale = max(1, int(np.ceil(np.sqrt((src.height * src.width) / max_pixels))))
        out_h = max(1, src.height // scale)
        out_w = max(1, src.width // scale)
        arr = src.read(band, out_shape=(out_h, out_w), resampling=Resampling.average).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[np.isclose(arr, nodata)] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        # Crop-probability change is in [-1, 1]. Export nodata is often -999.
        arr[np.abs(arr) > 1.5] = np.nan
        left, bottom, right, top = src.bounds
        extent = (float(left), float(right), float(bottom), float(top))
        crs = src.crs
    return arr, extent, crs


def _delta_to_wgs84(
    arr: np.ndarray,
    extent: Tuple[float, float, float, float],
    crs: Any,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Reproject a display array to EPSG:4326. Extent is (left, right, bottom, top)."""
    if crs is None or str(crs) in {"EPSG:4326", "OGC:CRS84"}:
        try:
            if crs is not None and crs.to_epsg() == 4326:
                return arr, extent
        except Exception:
            return arr, extent
        if crs is None:
            return arr, extent
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    left, right, bottom, top = extent
    src_transform = rasterio.transform.from_bounds(left, bottom, right, top, arr.shape[1], arr.shape[0])
    dst_transform, width, height = calculate_default_transform(
        crs, "EPSG:4326", arr.shape[1], arr.shape[0], left, bottom, right, top
    )
    dst = np.full((height, width), np.nan, dtype="float32")
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    dleft, dbottom, dright, dtop = rasterio.transform.array_bounds(height, width, dst_transform)
    return dst, (float(dleft), float(dright), float(dbottom), float(dtop))


_RIVERS_BLOB = "shp/stanford-rivers/stanford-rivers"
_RIVERS_TO_SHOW = (
    "R. Zambeze",
    "Zambeze",
    "Zambezi",
    "Rio Zambeze",
    "R. Limpopo",
    "Limpopo",
    "R. Rovuma",
    "Rovuma",
    "Rio Rovuma",
    "R. Lurio",
    "R. Messalo",
    "R. Pungue",
    "R. Buzi",
    "R. Save",
)
_CYCLONE_TRACK_COLORS = (
    "#1f77b4",
    "#9467bd",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _ensure_rivers_shp() -> Optional[Path]:
    """Local Stanford rivers shapefile, downloaded from GCS when it is missing."""
    dest = Path(LOCAL_DATA_DIR) / "shp" / "stanford-rivers" / "stanford-rivers.shp"
    if dest.is_file() and dest.with_suffix(".dbf").is_file() and dest.with_suffix(".shx").is_file():
        return dest
    if not USE_GCS:
        return dest if dest.is_file() else None
    try:
        import google.auth
        from google.cloud import storage

        creds, project = google.auth.default()
        client = storage.Client(credentials=creds, project=project)
        bucket = client.bucket(GCS_BUCKET)
        dest.parent.mkdir(parents=True, exist_ok=True)
        got_shp = False
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            blob = bucket.blob(_RIVERS_BLOB + ext)
            if not blob.exists(client):
                continue
            blob.download_to_filename(str(dest.with_suffix(ext)))
            got_shp = got_shp or ext == ".shp"
        if not got_shp:
            print(f"  Warning: gs://{GCS_BUCKET}/{_RIVERS_BLOB}.shp not found.")
            return None
        return dest
    except Exception as exc:
        print(f"  Warning: could not fetch rivers: {exc}")
        return dest if dest.is_file() else None


def _rivers_for_map(mask: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
    """Major rivers clipped to the mapped provinces, matching the context map."""
    shp = _ensure_rivers_shp()
    if shp is None:
        return None
    rivers = gpd.read_file(shp)
    if rivers.crs is None:
        rivers = rivers.set_crs("EPSG:4326")
    else:
        rivers = rivers.to_crs("EPSG:4326")
    name_col = next(
        (col for col in rivers.columns if str(col).upper() in {"NAME", "RIVER", "RIVER_NAME", "NAM"}),
        None,
    )
    if name_col is not None:
        keep = {name.strip().upper() for name in _RIVERS_TO_SHOW}
        rivers = rivers.loc[rivers[name_col].astype(str).str.strip().str.upper().isin(keep)].copy()
    if len(rivers) == 0:
        return None
    try:
        clip_geom = unary_union(list(mask.geometry))
        if clip_geom is None or clip_geom.is_empty:
            return None
        rivers = gpd.clip(rivers, clip_geom.buffer(0.08))
    except Exception as exc:
        print(f"  Warning: could not clip rivers: {exc}")
        return None
    return rivers if len(rivers) else None


def _cyclone_tracks_for_map(mask: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
    """IBTrACS tracks that cross the mapped provinces. Labels are the storm name only."""
    from shapely.geometry import LineString

    local = Path(LOCAL_DATA_DIR) / CYCLONE_FILE
    path = str(local) if local.is_file() else (
        f"gs://{GCS_BUCKET}/{CYCLONE_FILE}" if USE_GCS else _local_data_path(CYCLONE_FILE)
    )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  Warning: could not read cyclone tracks: {exc}")
        return None
    df["date"] = pd.to_datetime(df.get("ISO_TIME"), format="%m/%d/%Y %H:%M", errors="coerce")
    df["LAT"] = pd.to_numeric(df.get("LAT"), errors="coerce")
    df["LON"] = pd.to_numeric(df.get("LON"), errors="coerce")
    df["NAME"] = df.get("NAME", pd.Series("", index=df.index)).astype(str).str.strip()
    df = df.dropna(subset=["date", "LAT", "LON"])
    df = df.loc[df["NAME"].ne("") & df["NAME"].str.upper().ne("UNNAMED")].copy()
    if len(df) == 0:
        return None
    try:
        land = unary_union(list(mask.geometry))
    except Exception:
        return None
    if land is None or land.is_empty:
        return None
    rows = []
    for name, sub in df.groupby(df["NAME"].str.casefold(), sort=True):
        sub = sub.sort_values("date")
        coords = []
        for lon, lat in zip(sub["LON"].astype(float), sub["LAT"].astype(float)):
            xy = (float(lon), float(lat))
            if not coords or xy != coords[-1]:
                coords.append(xy)
        if len(coords) < 2:
            continue
        line = LineString(coords)
        if line.is_empty or not line.intersects(land):
            continue
        label = str(sub["NAME"].iloc[0]).strip().title()
        rows.append({"NAME": label, "geometry": line})
    if not rows:
        return None
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _overlay_rivers_and_cyclones(ax, provinces: gpd.GeoDataFrame) -> None:
    """Rivers and named cyclone tracks on the conflict panel."""
    try:
        rivers = _rivers_for_map(provinces)
    except Exception as exc:
        print(f"  Warning: rivers were not drawn: {exc}")
        rivers = None
    if rivers is not None and len(rivers):
        rivers.plot(ax=ax, color="#1E5AA0", linewidth=1.35, alpha=0.95, zorder=3)
    try:
        tracks = _cyclone_tracks_for_map(provinces)
    except Exception as exc:
        print(f"  Warning: cyclone tracks were not drawn: {exc}")
        tracks = None
    if tracks is None or len(tracks) == 0:
        return
    placed: List[Tuple[float, float]] = []
    for i, row in enumerate(tracks.itertuples(index=False)):
        color = _CYCLONE_TRACK_COLORS[i % len(_CYCLONE_TRACK_COLORS)]
        gpd.GeoSeries([row.geometry], crs="EPSG:4326").plot(
            ax=ax, color=color, linewidth=1.8, alpha=0.95, zorder=4
        )
        geom = row.geometry
        coords = list(geom.coords) if geom.geom_type == "LineString" else [
            xy for part in geom.geoms for xy in part.coords
        ]
        if not coords:
            continue
        tip = max(coords, key=lambda xy: xy[0])
        y = float(tip[1])
        x = float(tip[0])
        for px, py in placed:
            if abs(x - px) < 0.45 and abs(y - py) < 0.32:
                y = py + 0.34
        placed.append((x, y))
        ax.annotate(
            str(row.NAME),
            xy=(x, float(tip[1])),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color=color,
            fontweight="bold",
            zorder=8,
            annotation_clip=True,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.6,
                "alpha": 0.92,
            },
        )


def _lock_matched_maps(left, right, bounds: Tuple[float, float, float, float]) -> None:
    """Give both panels the same extent and the same on-page size."""
    xmin, xmax, ymin, ymax = bounds
    for ax in (left, right):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_axis_off()
    pos_left = left.get_position()
    pos_right = right.get_position()
    right.set_position([pos_right.x0, pos_left.y0, pos_left.width, pos_left.height])


def _place_colorbar(cax, ax) -> None:
    """Keep the colorbar beside its map after the map box is resized."""
    pos = ax.get_position()
    cpos = cax.get_position()
    height = pos.height * 0.72
    cax.set_position([cpos.x0, pos.y0 + (pos.height - height) / 2.0, cpos.width, height])


def _add_shared_scalebar(ax) -> None:
    """One scale bar. Both panels use this extent, so it applies to both."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lat = (ymin + ymax) / 2.0
    km_per_deg = 111.32 * float(np.cos(np.deg2rad(lat)))
    if km_per_deg <= 0:
        return
    width_km = (xmax - xmin) * km_per_deg
    length_km = 100 if width_km >= 350 else 50
    length_deg = length_km / km_per_deg
    x0 = xmin + 0.06 * (xmax - xmin)
    y0 = ymin + 0.065 * (ymax - ymin)
    tick = 0.011 * (ymax - ymin)
    ax.plot([x0, x0 + length_deg], [y0, y0], color="white", lw=4.0, solid_capstyle="butt", zorder=9)
    ax.plot([x0, x0 + length_deg], [y0, y0], color="black", lw=1.7, solid_capstyle="butt", zorder=10)
    for x in (x0, x0 + length_deg):
        ax.plot([x, x], [y0 - tick, y0 + tick], color="black", lw=1.2, zorder=10)
    ax.text(
        x0 + length_deg / 2.0,
        y0 + tick * 1.35,
        f"{int(length_km)} km",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        zorder=10,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )


def _plot_pixel_outcome_acled(
    g: gpd.GeoDataFrame,
    acled: pd.Series,
    path: Path,
) -> None:
    """Left: per-pixel cultivated probability change. Right: ACLED choropleth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    pieces = []
    for prov in _active_provinces():
        tif = _cultivated_delta_path(prov)
        if tif is None:
            continue
        arr, extent, crs = _read_delta_for_display(tif)
        arr, extent = _delta_to_wgs84(arr, extent, crs)
        pieces.append((prov, arr, extent, "EPSG:4326"))
    if not pieces:
        raise FileNotFoundError("No cultivated delta rasters available for the pixel map.")

    outline = g.to_crs(pieces[0][3]) if g.crs != pieces[0][3] else g
    minx, miny, maxx, maxy = outline.total_bounds
    span_x = max(float(maxx - minx), 1e-6)
    span_y = max(float(maxy - miny), 1e-6)
    # Extra room on the east holds cyclone names over the ocean on the conflict map.
    bounds = (
        float(minx - 0.03 * span_x),
        float(maxx + 0.22 * span_x),
        float(miny - 0.03 * span_y),
        float(maxy + 0.03 * span_y),
    )

    fig = plt.figure(figsize=(13.4, 7.3))
    map_w, map_h = 0.36, 0.78
    bottom, left0 = 0.06, 0.025
    cbar_w, cbar_pad, gap = 0.014, 0.008, 0.09
    ax = fig.add_axes([left0, bottom, map_w, map_h])
    cax0 = fig.add_axes([left0 + map_w + cbar_pad, bottom + 0.12, cbar_w, map_h - 0.24])
    left1 = left0 + map_w + cbar_pad + cbar_w + gap
    ax_acled = fig.add_axes([left1, bottom, map_w, map_h])
    cax1 = fig.add_axes([left1 + map_w + cbar_pad, bottom + 0.12, cbar_w, map_h - 0.24])

    finite = np.concatenate([p[1][np.isfinite(p[1])] for p in pieces if np.isfinite(p[1]).any()])
    if finite.size:
        lim = float(np.nanpercentile(np.abs(finite), 98))
        lim = max(lim, 1e-6)
    else:
        lim = 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    for _prov, arr, extent, crs in pieces:
        left, right, bottom_e, top = extent
        ax.imshow(
            arr,
            extent=(left, right, bottom_e, top),
            origin="upper",
            cmap="RdBu",
            norm=norm,
            interpolation="nearest",
            zorder=1,
        )
    outline.boundary.plot(ax=ax, color="0.25", linewidth=0.25, zorder=2)
    _draw_province_border(ax, outline)
    _label_provinces(ax, outline)
    ax.set_title("Cultivated probability change (per pixel)", fontsize=11)

    vmin, vmax = _choropleth_on_ax(
        ax_acled,
        outline,
        acled,
        title="Conflict events",
        cmap="YlOrRd",
        diverging=False,
        legend=False,
    )
    _overlay_rivers_and_cyclones(ax_acled, outline)
    _label_provinces(ax_acled, outline)

    _lock_matched_maps(ax, ax_acled, bounds)
    _place_colorbar(cax0, ax)
    _place_colorbar(cax1, ax_acled)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax0)
    cbar.set_label(f"Crop probability {_ACTIVE_ENDLINE['outcome_phrase']}")
    sm_acled = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(vmin=vmin, vmax=vmax))
    sm_acled.set_array([])
    cbar_acled = fig.colorbar(sm_acled, cax=cax1)
    cbar_acled.set_label(CONFLICT_MAP_LABEL)
    _add_shared_scalebar(ax)
    fig.suptitle(
        f"Cultivated land cover change ({_ACTIVE_ENDLINE['outcome_phrase']}) and conflict ({_ACTIVE_ENDLINE['conflict_phrase']})",
        fontsize=12,
        y=0.98,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _lisa_cluster_labels(values: np.ndarray, w: Any, *, alpha: float = 0.05) -> Tuple[List[str], Any]:
    """Local Moran cluster labels: HH / LL / HL / LH / Not significant."""
    from esda.moran import Moran_Local

    np.random.seed(RUN_SEED)
    lisa = Moran_Local(np.asarray(values, dtype=float), w, permutations=999)
    labels: List[str] = []
    for q, p in zip(np.asarray(lisa.q).ravel(), np.asarray(lisa.p_sim).ravel()):
        if not np.isfinite(p) or float(p) >= alpha:
            labels.append("Not significant")
        elif int(q) == 1:
            labels.append("HH")
        elif int(q) == 2:
            labels.append("LH")
        elif int(q) == 3:
            labels.append("LL")
        elif int(q) == 4:
            labels.append("HL")
        else:
            labels.append("Not significant")
    return labels, lisa


def _plot_lisa_map(
    g: gpd.GeoDataFrame,
    admin_post_ids: Sequence[str],
    labels: Sequence[str],
    path: Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = list(admin_post_ids)
    plot_g = g.set_index("admin_post_id").loc[order].reset_index()
    plot_g["lisa_cluster"] = list(labels)
    color_map = {
        "HH": "#d7191c",
        "LL": "#2c7bb6",
        "HL": "#fdae61",
        "LH": "#abd9e9",
        "Not significant": "#d9d9d9",
    }
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for lab, col in color_map.items():
        sub = plot_g[plot_g["lisa_cluster"].eq(lab)]
        if len(sub):
            sub.plot(ax=ax, color=col, edgecolor="0.35", linewidth=0.25)
    _draw_province_border(ax, plot_g)
    _label_provinces(ax, plot_g)
    ax.set_axis_off()
    ax.set_title(title)
    handles = [Patch(facecolor=c, edgecolor="0.4", label=k) for k, c in color_map.items()]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8, title="Local Moran cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _weights_for_ids(g: gpd.GeoDataFrame, admin_post_ids: Sequence[str], full_w: Any) -> Any:
    """Return spatial weights aligned to ``admin_post_ids`` order."""
    ids = list(admin_post_ids)
    if getattr(full_w, "n", None) == len(ids):
        # Likely already aligned with the analysis frame order.
        return full_w
    sub = g.set_index("admin_post_id").loc[ids].reset_index()
    return _build_w_variants(sub)[PRIMARY_W]


def export_presentation_figures(
    out: Path,
    *,
    post: pd.DataFrame,
    g: gpd.GeoDataFrame,
    ols_coefs: pd.DataFrame,
    ols_diag: pd.DataFrame,
    set_a_diag: pd.DataFrame,
    set_b_diag: pd.DataFrame,
    b2_diag: Dict[str, Any],
    choice: str,
    explanation: str,
    both: bool,
    b2_compare: pd.DataFrame,
    spat_coefs: pd.DataFrame,
    include_iom: bool,
    resid_maps: Optional[Dict[str, Dict[str, Any]]] = None,
    w: Any = None,
    queen_diag: Optional[Dict[str, Any]] = None,
    weights_name: str = PRIMARY_W,
) -> List[str]:
    """Paper figures: M1–M5b table, M5b Lagrange tests, M5b spatial model, and IE1–IE5."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    produced: List[str] = []

    def _note(name: str) -> None:
        produced.append(name)
        print(f"  figure: {name}")

    # Align geometry to post frame
    gg = g.merge(post, on="admin_post_id", how="inner")
    if "geometry" not in gg.columns and g.geometry.name:
        gg = g.set_index("admin_post_id").loc[post["admin_post_id"]].reset_index()
        gg = gg.merge(post, on="admin_post_id", how="left")

    # --- Maps (study design / pattern slides) ---
    try:
        acled_v = gg["conflict_endline_count"] if "conflict_endline_count" in gg.columns else gg.get(ACLED_COL)
        _choropleth_pair(
            gg,
            gg["outcome"],
            acled_v,
            out / "fig_erl_map_outcome_acled_side_by_side.png",
            left_title=OUTCOME_MAP_LABEL,
            right_title="Conflict events",
            left_cmap="RdBu",
            right_cmap="YlOrRd",
            left_diverging=True,
            right_diverging=False,
            suptitle=(
                f"Cultivated land cover change ({_ACTIVE_ENDLINE['outcome_phrase']}) "
                f"and conflict ({_ACTIVE_ENDLINE['conflict_phrase']})"
            ),
            left_cbar_label=(
                f"Crop probability {_ACTIVE_ENDLINE['outcome_phrase']},\n"
                "aggregated to the post and divided by post area"
            ),
            right_cbar_label=CONFLICT_MAP_LABEL,
        )
        _note("fig_erl_map_outcome_acled_side_by_side.png")
    except Exception as exc:
        print(f"  Warning: ACLED map failed: {exc}")

    try:
        acled_v = gg["conflict_endline_count"] if "conflict_endline_count" in gg.columns else gg[ACLED_COL]
        _plot_pixel_outcome_acled(
            gg,
            acled_v,
            out / "fig_erl_map_outcome_pixel_acled_side_by_side.png",
        )
        _note("fig_erl_map_outcome_pixel_acled_side_by_side.png")
    except Exception as exc:
        print(f"  Warning: per-pixel cultivated map failed: {exc}")

    try:
        pop_change = worldpop_proportional_change(gg)
        _choropleth(
            gg,
            pop_change,
            out / "fig_erl_map_pop_growth.png",
            title=(
                f"WorldPop population change\n{WORLDPOP_CHANGE_BASE_LABEL} to {WORLDPOP_CHANGE_END_LABEL}"
            ),
            cmap="RdBu",
            diverging=True,
            cbar_label="Proportional population change, (2020 − 2017) / 2017, summed within the post",
        )
        _note("fig_erl_map_pop_growth.png")
    except Exception as exc:
        print(f"  Warning: population-change map failed: {exc}")

    if "pop_density_period" in gg.columns:
        try:
            _choropleth(
                gg,
                gg["pop_density_period"],
                out / "fig_erl_map_pop_density.png",
                title="WorldPop population density (model control, people/ha)",
                cmap="YlOrRd",
                diverging=False,
                cbar_label="People per hectare, summed within the post and divided by post area",
            )
            _note("fig_erl_map_pop_density.png")
        except Exception as exc:
            print(f"  Warning: population-density map failed: {exc}")

    if include_iom and {"idp_conflict_pressure", "idp_disaster_pressure", "idp_all_reasons_pressure"}.issubset(gg.columns):
        try:
            _choropleth_shared(
                gg,
                [
                    (gg["idp_conflict_pressure"], "Conflict IDPs"),
                    (gg["idp_disaster_pressure"], "Cyclone IDPs"),
                    (gg["idp_all_reasons_pressure"], "Conflict & Cyclone IDPs"),
                ],
                out / "fig_erl_map_iom_three_panel.png",
                cmap="Blues",
                diverging=False,
                suptitle="IDPs arrivals by post and reason",
                cbar_label="log(1 + IDP arrivals), allocated from district to post",
            )
            _note("fig_erl_map_iom_three_panel.png")
        except Exception as exc:
            print(f"  Warning: IOM map failed: {exc}")

    # --- LISA / residual Moran cluster maps ---
    try:
        # Outcome LISA (raw pattern, not residual)
        ids = gg["admin_post_id"].astype(str).tolist()
        w_out = _weights_for_ids(g, ids, w) if w is not None else _build_w_variants(gg)[PRIMARY_W]
        y_out = pd.to_numeric(gg["outcome"], errors="coerce").to_numpy(dtype=float)
        lab_out, _ = _lisa_cluster_labels(y_out, w_out)
        _plot_lisa_map(
            g,
            ids,
            lab_out,
            out / "fig_erl_lisa_outcome.png",
            title=f"LISA clusters: {OUTCOME_MAP_LABEL}",
        )
        _note("fig_erl_lisa_outcome.png")
    except Exception as exc:
        print(f"  Warning: outcome LISA map failed: {exc}")

    if resid_maps:
        for key, payload in resid_maps.items():
            try:
                r = np.asarray(payload["resid"], dtype=float)
                ids = [str(x) for x in payload["admin_post_ids"]]
                title = str(payload.get("title") or key)
                fname = str(payload.get("filename") or f"fig_erl_lisa_{key}.png")
                map_w = payload.get("w", w)
                ww = _weights_for_ids(g, ids, map_w) if map_w is not None else _build_w_variants(
                    g.set_index("admin_post_id").loc[ids].reset_index()
                )[PRIMARY_W]
                labs, _lisa = _lisa_cluster_labels(r, ww)
                map_g = gg if "province" in getattr(gg, "columns", []) else g
                _plot_lisa_map(map_g, ids, labs, out / fname, title=title)
                _note(fname)
                # also save residual choropleth
                resid_fn = fname.replace("lisa_", "resid_")
                if resid_fn == fname:
                    resid_fn = f"fig_erl_resid_{key}.png"
                g_sub = map_g.set_index("admin_post_id").loc[ids].reset_index()
                _choropleth(
                    g_sub,
                    pd.Series(r),
                    out / resid_fn,
                    title=title.replace("LISA clusters:", "Residuals:"),
                    cmap="RdBu_r",
                    diverging=True,
                    cbar_label="Standardized crop-probability change minus the fitted value",
                )
                _note(resid_fn)
            except Exception as exc:
                print(f"  Warning: LISA map for {key} failed: {exc}")

    # --- Paper table: nested OLS M1-M5b, with residual Moran ---
    try:
        main_coefs = ols_coefs[ols_coefs["model_id"].astype(str).isin(REPORTED_MODELS)].copy()
        main_diag = ols_diag[ols_diag["model_id"].astype(str).isin(REPORTED_MODELS)].copy()
        if "weights" in main_diag.columns:
            knn_diag = main_diag[main_diag["weights"].astype(str).eq(PRIMARY_W)]
            if len(knn_diag):
                main_diag = knn_diag
        term_order = [
            "floodplain_mean_z",
            "cyclone_endline_window_z",
            "spei_endline_window_z",
            "chirps_endline_window_z",
            POP_COL,
            ACLED_COL,
            "cyclone_x_acled",
            CABO_COL,
            IOM_ALL_COL,
            IOM_IDP_COL,
            IOM_DISASTER_COL,
        ]
        present = set(main_coefs["term"].astype(str))
        rows = []
        for term in term_order:
            if term not in present:
                continue
            row = {"Term": _term_label(term)}
            for mid in REPORTED_MODELS:
                hit = main_coefs[main_coefs["model_id"].astype(str).eq(mid) & main_coefs["term"].astype(str).eq(term)]
                if len(hit) == 0:
                    row[mid] = ""
                    continue
                rr = hit.iloc[0]
                stars = str(rr.get("sig") or "").replace("nan", "")
                row[mid] = f"{float(rr['coef']):.3f}{stars} ({float(rr['se']):.3f})"
            rows.append(row)
        diag_ix = main_diag.set_index(main_diag["model_id"].astype(str))

        def _footer(label: str, fmt) -> None:
            row = {"Term": label}
            for mid in REPORTED_MODELS:
                row[mid] = fmt(diag_ix.loc[mid]) if mid in diag_ix.index else ""
            rows.append(row)

        _footer("R²", lambda r: f"{float(r['r2']):.3f}")
        _footer("Adj. R²", lambda r: f"{float(r['adj_r2']):.3f}")
        _footer("Residual Moran's I", lambda r: f"{float(r['resid_moran_I']):.3f}")
        _footer("Residual Moran p", lambda r: f"{float(r['resid_moran_p_sim']):.3f}")
        summary = pd.DataFrame(rows).rename(
            columns={mid: f"{mid} coefficient (SE)" for mid in REPORTED_MODELS}
        )
        summary.to_csv(out / "ERL_M1_M5b_summary.csv", index=False)
        _note("ERL_M1_M5b_summary.csv")
        save_table_png(
            summary,
            out / "table_erl_M1_M5b.png",
            "M1-M5b OLS coefficient (SE)",
        )
        _note("table_erl_M1_M5b.png")
    except Exception as exc:
        print(f"  Warning: M1-M5b table failed: {exc}")

    # --- Lagrange multiplier results for every reported model ---
    try:
        main_diag = ols_diag[ols_diag["model_id"].astype(str).isin(REPORTED_MODELS)].copy()
        if "weights" in main_diag.columns:
            knn_diag = main_diag[main_diag["weights"].astype(str).eq(PRIMARY_W)]
            if len(knn_diag):
                main_diag = knn_diag
        lm_rows = []
        for _, rr in main_diag.iterrows():
            lm_rows.append(
                {
                    "Model": str(rr["model_id"]),
                    "Moran I": f"{float(rr['resid_moran_I']):.3f}",
                    "Moran p": f"{float(rr['resid_moran_p_sim']):.3g}",
                    "LM-lag p": f"{float(rr['lm_lag_p']):.3g}",
                    "LM-error p": f"{float(rr['lm_error_p']):.3g}",
                    "Robust lag p": f"{float(rr['robust_lm_lag_p']):.3g}",
                    "Robust error p": f"{float(rr['robust_lm_error_p']):.3g}",
                }
            )
        if lm_rows:
            save_table_png(
                pd.DataFrame(lm_rows),
                out / "table_erl_M1_M5b_lm.png",
                "Lagrange multiplier tests, M1–M5b OLS residuals (knn5)",
            )
            _note("table_erl_M1_M5b_lm.png")
    except Exception as exc:
        print(f"  Warning: M1-M5b LM table failed: {exc}")

    # --- M5b Lagrange figure used to choose SAR or SEM ---
    try:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        names = ["LM-lag", "LM-error", "Robust LM-lag", "Robust LM-error"]
        keys = ["lm_lag_p", "lm_error_p", "robust_lm_lag_p", "robust_lm_error_p"]

        def _lm_height(diag: Dict[str, Any]) -> List[float]:
            out_h = []
            for key in keys:
                p = float(diag.get(key, np.nan))
                out_h.append(-np.log10(max(p, 1e-300)) if np.isfinite(p) else 0.0)
            return out_h

        y_pos = np.arange(len(names))
        if queen_diag:
            ax.barh(y_pos + 0.18, _lm_height(b2_diag), height=0.34, color="#1b9e77", label="knn5")
            ax.barh(y_pos - 0.18, _lm_height(queen_diag), height=0.34, color="#7570b3", label="queen")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
        else:
            heights = _lm_height(b2_diag)
            colors = ["#1b9e77" if h > -np.log10(0.05) else "#7570b3" for h in heights]
            ax.barh(names, heights, color=colors)
        ax.axvline(-np.log10(0.05), color="0.4", ls="--", lw=1, label="p=0.05")
        ax.set_xlabel("−log10(p)")
        ax.set_title(f"M5b Lagrange multiplier tests → {choice} ({weights_name})")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_erl_M5b_lm.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        _note("fig_erl_M5b_lm.png")
    except Exception as exc:
        print(f"  Warning: M5b LM figure failed: {exc}")

    # --- Coefficient tables for SAR and SEM, plus the M5b forest plots ---
    try:
        def _stage_rows(model_id: str, form: str) -> pd.DataFrame:
            sub = spat_coefs[
                spat_coefs["model_id"].astype(str).eq(model_id) & spat_coefs["stage"].astype(str).eq(form)
            ].copy()
            if len(sub) == 0:
                return sub
            return sub[~sub["term"].astype(str).str.match(r"(?i)^constant$")].copy()

        def _coef_table(rows: pd.DataFrame) -> pd.DataFrame:
            labeled = rows.copy()
            labeled["label"] = labeled.apply(lambda r: str(r.get("term_label") or r["term"]), axis=1)
            return pd.DataFrame(
                {
                    "Term": labeled["label"].astype(str),
                    "Coef": pd.to_numeric(labeled["coef"], errors="coerce"),
                    "SE": pd.to_numeric(labeled["se"], errors="coerce"),
                    "p": pd.to_numeric(labeled["p_value"], errors="coerce"),
                    "Sig": labeled.get("sig", pd.Series("", index=labeled.index)).fillna("").astype(str).replace("nan", ""),
                }
            )

        def _save_coef_table(model_id: str, form: str, filename: str) -> None:
            rows = _stage_rows(model_id, form)
            if len(rows) == 0:
                return
            save_table_png(_coef_table(rows), out / filename, f"{model_id} {form} coefficients")
            _note(filename)

        def _save_forest(model_id: str, form: str, filename: str) -> None:
            rows = _stage_rows(model_id, form)
            if len(rows) == 0:
                return
            rows = rows.copy()
            rows["label"] = rows.apply(lambda r: str(r.get("term_label") or r["term"]), axis=1)
            _forest_plot(
                rows,
                out / filename,
                title=f"{model_id} {form} coefficients",
                figsize=(8.4, max(4.2, 0.38 * len(rows) + 1.2)),
            )
            _note(filename)

        for mid in REPORTED_MODELS:
            if choice in {"SAR", "SEM"}:
                _save_coef_table(mid, choice, f"table_erl_{mid}_spatial.png")
            _save_coef_table(mid, "SAR", f"table_erl_{mid}_SAR.png")
        if choice in {"SAR", "SEM"}:
            _save_forest(FOREST_MODEL, choice, f"fig_erl_{FOREST_MODEL}_forest.png")
        _save_forest(FOREST_MODEL, "SAR", f"fig_erl_{FOREST_MODEL}_SAR_forest.png")
    except Exception as exc:
        print(f"  Warning: spatial coefficient tables failed: {exc}")

    # --- Supplemental individual-predictor models ---
    try:
        ie = ols_coefs[ols_coefs["model_id"].astype(str).isin(IE_MODELS)].copy()
        if len(ie):
            ie_tab = ie[~ie["term"].astype(str).str.match(r"(?i)^constant$")][
                [c for c in ["model_id", "term_label", "coef", "se", "z_or_t", "p_value", "sig"] if c in ie.columns]
            ]
            save_table_png(
                ie_tab,
                out / "table_erl_IE_coefs.png",
                "Supplemental IE1–IE5: one predictor plus population density",
            )
            _note("table_erl_IE_coefs.png")
            rows = []
            for mid in IE_MODELS:
                sub = ie[ie["model_id"].astype(str).eq(mid) & ~ie["term"].isin(["CONSTANT", POP_COL])]
                if len(sub) == 0:
                    continue
                rr = sub.iloc[0]
                rows.append({"label": f"{mid}: {rr.get('term_label', rr['term'])}", "coef": rr["coef"], "se": rr["se"]})
            if rows:
                _forest_plot(
                    pd.DataFrame(rows),
                    out / "fig_erl_IE_coefs.png",
                    title="Supplemental: individual predictors (each model also includes population density)",
                )
                _note("fig_erl_IE_coefs.png")
    except Exception as exc:
        print(f"  Warning: IE supplemental figures failed: {exc}")

    return produced


def write_choice_markdown(
    path: Path,
    choice: str,
    explanation: str,
    b2_diag: Dict[str, Any],
    *,
    both: bool,
    weights_name: str = PRIMARY_W,
    queen_diag: Optional[Dict[str, Any]] = None,
) -> None:
    lines = [
        "# ERL spatial model choice (M5b Lagrange multiplier tests)",
        "",
        f"**Chosen form:** `{choice}`",
        f"**Weights used for a committed SAR or SEM:** `{weights_name}`",
        "",
        explanation,
        "",
        "## M5b OLS diagnostics (knn5)",
        f"- Residual Moran's I = {b2_diag.get('resid_moran_I'):.4g} (p_sim = {b2_diag.get('resid_moran_p_sim'):.4g})",
        f"- LM-lag p = {b2_diag.get('lm_lag_p'):.4g}; LM-error p = {b2_diag.get('lm_error_p'):.4g}",
        f"- Robust LM-lag p = {b2_diag.get('robust_lm_lag_p'):.4g}; Robust LM-error p = {b2_diag.get('robust_lm_error_p'):.4g}",
        f"- Both lag and error indicated: **{both}**",
        "",
    ]
    if queen_diag:
        lines.extend(
            [
                "## M5b OLS diagnostics (queen)",
                f"- Residual Moran's I = {queen_diag.get('resid_moran_I'):.4g} (p_sim = {queen_diag.get('resid_moran_p_sim'):.4g})",
                f"- LM-lag p = {queen_diag.get('lm_lag_p'):.4g}; LM-error p = {queen_diag.get('lm_error_p'):.4g}",
                f"- Robust LM-lag p = {queen_diag.get('robust_lm_lag_p'):.4g}; Robust LM-error p = {queen_diag.get('robust_lm_error_p'):.4g}",
                "",
            ]
        )
    lines.extend(
        [
            "SDM is fit on M5b only when both robust LM tests reject. SDEM is attempted when SEM is the committed form.",
            "",
            "Decision order follows the Anselin–Florax step-by-step on that page:",
            "1. Classic LM tests only say whether a spatial term is needed.",
            "2. Robust LM-lag only → SAR. Robust LM-error only → SEM.",
            "3. Both robust tests reject → do not force SAR vs SEM (higher-order / omitted covariate).",
            "4. Neither robust test rejects → do not use the smaller classic p-value.",
            "5. Re-estimate the M5b LM tests with queen contiguity. If one robust test rejects, use those weights and that form.",
            "6. If queen also names no form, fit SAR and SEM on knn5 and report the lower AIC. A gap under 2 is a tie, reported as SEM.",
            "7. When a robust test has already selected SAR or SEM, AIC confirms it and does not override it. A gap under 2 is a tie.",
            "8. Pseudo-R2 is recorded and not used to choose.",
            "Guide: https://www.spatial-epidemiology.org/disease-clustering-spatial-statistical-modeling/spatial-lag-and-error-regression/choosing-between-spatial-lag-and-error-models/",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_advisor_summary(
    path: Path,
    *,
    choice: str,
    explanation: str,
    set_a_diag: pd.DataFrame,
    set_b_diag: pd.DataFrame,
    include_iom: bool,
) -> None:
    lines = [
        "# ERL M1–M5b summary",
        "",
        "Cabo Delgado and Nampula admin posts.",
        (
            f"Cultivated-share change is the {_ACTIVE_ENDLINE['cultivated_suffix']} product. "
            f"Predictors cover {_ACTIVE_ENDLINE['endline_window_start']} to {_ACTIVE_ENDLINE['endline_window_end']}."
        ),
        "Population density is in every model. The Cabo Delgado dummy (Nampula reference, not standardized)",
        "enters at M4 and stays in M5a and M5b. ACLED is standardized log1p(events per month) with Riots removed.",
        "",
        f"## M5b spatial form: **{choice}**",
        "",
        "The Lagrange multiplier tests are estimated for M1–M5b. SAR versus SEM is decided from M5b.",
        "knn5 is the starting weights. If its robust LM tests are in the neither branch,",
        "queen contiguity is checked. If queen also names no form, SAR and SEM are both fit",
        "and the lower AIC is the reported form.",
        "",
        explanation,
        "",
        "M1 = floodplain + cyclone + drought + extreme rainfall (CHIRPS) + population density.",
        "M2 = M1 + Conflict. M3 = M2 + Cyclone x conflict. M4 = M3 + Cabo Delgado dummy.",
        "M5a = M4 + IOM all reasons pooled.",
        "M5b = M4 + IOM conflict/insecurity/other/no reason and IOM natural disaster.",
        "The pooled IOM term is not in M5b. A row tagged with both reasons is in both split terms.",
        "",
        "## Supplemental IE models: residual Moran / LM",
        "",
        "| Model | Moran I | Moran p | LM-lag p | LM-error p |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in set_a_diag.iterrows():
        lines.append(
            f"| {r['model_id']} | {r['resid_moran_I']:.3g} | {r['resid_moran_p_sim']:.3g} | "
            f"{r['lm_lag_p']:.3g} | {r['lm_error_p']:.3g} |"
        )
    lines.extend(["", "## M1–M5b OLS diagnostics", "", "| Model | Moran I | LM-lag p | LM-error p |", "|---|---:|---:|---:|"])
    for _, r in set_b_diag.iterrows():
        lines.append(
            f"| {r['model_id']} | {r['resid_moran_I']:.3g} | {r['lm_lag_p']:.3g} | {r['lm_error_p']:.3g} |"
        )
    if include_iom:
        lines.extend(
            [
                "",
                "IDP counts are allocated district→post by population density × area.",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _post_dropping_acled_event_types(
    post: pd.DataFrame,
    admin: gpd.GeoDataFrame,
    *,
    period_key: str,
    drop_event_types: Sequence[str],
) -> Tuple[pd.DataFrame, int]:
    """Copy ``post`` with ACLED rebuilt after dropping event types.

    The replacement is log1p(events per month), z-scored on this cross-section,
    and the cyclone × ACLED product is rebuilt from those z-scores. The returned
    count is how many dropped-type events had been assigned to these posts.
    """
    full = build_conflict_endline_metrics(admin)
    reduced = build_conflict_endline_metrics(admin, drop_event_types=drop_event_types)

    def _tag(metrics: pd.DataFrame) -> pd.DataFrame:
        m = metrics.loc[metrics["period_key"].astype(str).eq(period_key)].copy()
        m["admin_post_id"] = m["province"].astype(str) + "||" + m["admin_post"].astype(str)
        return m

    full_p = _tag(full)
    reduced_p = _tag(reduced)
    both = full_p[["admin_post_id", "conflict_endline_count"]].merge(
        reduced_p[["admin_post_id", "conflict_endline_count"]],
        on="admin_post_id",
        how="outer",
        suffixes=("_all", "_reduced"),
    )
    in_sample = both["admin_post_id"].isin(set(post["admin_post_id"].astype(str)))
    n_dropped = int(
        (
            pd.to_numeric(both.loc[in_sample, "conflict_endline_count_all"], errors="coerce").fillna(0.0)
            - pd.to_numeric(both.loc[in_sample, "conflict_endline_count_reduced"], errors="coerce").fillna(0.0)
        )
        .clip(lower=0.0)
        .sum()
    )
    keep = reduced_p[
        ["admin_post_id", "conflict_endline_count", "conflict_endline_rate", "conflict_endline_window"]
    ]
    n_matched = int(post["admin_post_id"].astype(str).isin(set(keep["admin_post_id"].astype(str))).sum())
    if n_matched < len(post):
        raise ValueError(
            f"Riot-dropped ACLED matched {n_matched} of {len(post)} posts. "
            "Refusing to refit with unmatched posts set to zero."
        )
    out = post.drop(
        columns=[c for c in keep.columns if c != "admin_post_id" and c in post.columns]
    ).merge(keep, on="admin_post_id", how="left")
    for col in ("conflict_endline_count", "conflict_endline_rate", "conflict_endline_window"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    raw = out["conflict_endline_window"]
    sd = float(raw.std())
    if not np.isfinite(sd) or sd <= 0:
        out[ACLED_COL] = 0.0
    else:
        out[ACLED_COL] = (raw - float(raw.mean())) / sd
    if "cyclone_endline_window_z" in out.columns:
        out["cyclone_x_acled"] = pd.to_numeric(out["cyclone_endline_window_z"], errors="coerce") * pd.to_numeric(
            out[ACLED_COL], errors="coerce"
        )
    return out, n_dropped


def write_acled_no_riots_robustness(
    out: Path,
    *,
    post: pd.DataFrame,
    specs: Sequence[NestedSpec],
    ols_w: Any,
    spatial_w: Any,
    choice: str,
    weights_name: str,
    ols_coefs: pd.DataFrame,
    spat_coefs: pd.DataFrame,
) -> None:
    """Riots are already excluded from every ACLED model."""
    return
    targets = [s for s in specs if s.model_id in {"B2", "B2_x"}]
    if len(targets) < 2:
        print("  Warning: B2 and B2_x missing; ACLED riot check skipped.")
        return
    print("\n=== ACLED robustness: drop Riots, refit B2 and B2_x ===")
    admin = load_admin()
    period_key = str(post["period_key"].iloc[0]) if "period_key" in post.columns and len(post) else "long"
    alt_post, n_dropped = _post_dropping_acled_event_types(
        post, admin, period_key=period_key, drop_event_types=("Riots",)
    )
    print(f"  Riot events removed from sample posts in the {period_key} window: {n_dropped}")

    def _slice_main(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
        if frame is None or len(frame) == 0 or "model_id" not in frame.columns:
            return pd.DataFrame()
        sub = frame[frame["model_id"].astype(str).isin(["B2", "B2_x"])].copy()
        if "stage" in sub.columns:
            sub = sub[sub["stage"].astype(str).eq(stage)]
        return sub

    parts: List[pd.DataFrame] = []

    def _tag(frame: pd.DataFrame, coding: str) -> pd.DataFrame:
        tagged = frame.copy()
        tagged.insert(0, "acled_coding", coding)
        if "period" not in tagged.columns:
            tagged.insert(0, "period", period_key)
        else:
            tagged["period"] = period_key
        return tagged

    main_ols = _slice_main(ols_coefs, "OLS")
    if len(main_ols):
        parts.append(_tag(main_ols, "all_event_types"))
    if choice in {"SAR", "SEM"}:
        main_spat = _slice_main(spat_coefs, choice)
        if len(main_spat):
            parts.append(_tag(main_spat, "all_event_types"))

    for spec in targets:
        y, x, names, _d = _design_matrix(alt_post, spec.predictors, skip_zscore=spec.skip_zscore)
        try:
            cdf, diag, _resid = fit_ols_with_diagnostics(
                y,
                x,
                names,
                ols_w,
                model_id=spec.model_id,
                model_label=spec.label + " (Riots dropped)",
                set_name="B",
                weights_name=PRIMARY_W,
            )
            parts.append(_tag(cdf, "no_riots"))
            print(
                f"  {spec.model_id} OLS without Riots: R2={diag.get('r2', float('nan')):.3f}, "
                f"residual Moran I={diag.get('resid_moran_I', float('nan')):.3f}"
            )
        except Exception as exc:
            print(f"  Warning: OLS without Riots failed for {spec.model_id}: {exc}")
        if choice in {"SAR", "SEM"} and spatial_w is not None:
            try:
                cdf, summ, _resid = fit_spatial_model(
                    y,
                    x,
                    names,
                    spatial_w,
                    form=choice,
                    model_id=spec.model_id,
                    model_label=spec.label + " (Riots dropped)",
                    set_name="B",
                    weights_name=weights_name,
                )
                parts.append(_tag(cdf, "no_riots"))
                print(f"  {spec.model_id} {choice} without Riots: AIC={summ.get('aic', float('nan')):.2f}")
            except Exception as exc:
                print(f"  Warning: {choice} without Riots failed for {spec.model_id}: {exc}")

    if not parts:
        print("  Warning: ACLED riot check produced no coefficients.")
        return
    tab = pd.concat(parts, ignore_index=True)
    to_csv_sig(tab, out / "ERL_acled_no_riots_coefficients.csv")
    key_terms = {ACLED_COL, "cyclone_x_acled", "rho", "lambda"}
    show = tab[tab["term"].astype(str).isin(key_terms)].copy()
    if len(show):
        order = {"all_event_types": 0, "no_riots": 1, "OLS": 0, "SAR": 1, "SEM": 1, "B2": 0, "B2_x": 1}
        show["_c"] = show["acled_coding"].map(lambda v: order.get(str(v), 9))
        show["_s"] = show["stage"].map(lambda v: order.get(str(v), 9))
        show["_m"] = show["model_id"].map(lambda v: order.get(str(v), 9))
        show = show.sort_values(["_c", "_s", "_m", "term"]).drop(columns=["_c", "_s", "_m"])
        pretty = pd.DataFrame(
            {
                "Coding": show["acled_coding"].astype(str),
                "Stage": show["stage"].astype(str),
                "Model": show["model_id"].astype(str),
                "Term": show.get("term_label", show["term"]).astype(str),
                "Coef": pd.to_numeric(show["coef"], errors="coerce"),
                "SE": pd.to_numeric(show["se"], errors="coerce"),
                "p": pd.to_numeric(show["p_value"], errors="coerce"),
            }
        )
        save_table_png(
            pretty,
            out / "table_erl_acled_no_riots.png",
            f"B2 and B2_x with Riots dropped ({choice}, {weights_name})",
        )
    spatial_sentence = (
        f"The spatial refit uses the committed form, {choice} on {weights_name}."
        if choice in {"SAR", "SEM"}
        else "No spatial form was committed, so this file contains the OLS refit only."
    )
    note = [
        "# ACLED robustness: Riots dropped",
        "",
        "Reported models count every event type in the ACLED file: Battles,",
        "Explosions/Remote violence, Violence against civilians, and Riots.",
        "This check removes Riots and rebuilds log1p(events per month), the",
        "within-sample z-score, and the cyclone × ACLED product.",
        "",
        f"Riot events removed from sample posts in the {period_key} window: {n_dropped}.",
        spatial_sentence,
        "OLS coefficients for B2 and B2_x are included beside that refit.",
        "The SAR versus SEM decision is left on the full ACLED measure.",
        "B3 and B4 are unchanged; they answer the IOM question on the reported ACLED coding.",
        "",
    ]
    (out / "ERL_acled_no_riots_note.md").write_text("\n".join(note), encoding="utf-8")
    print("  wrote ERL_acled_no_riots_coefficients.csv")


# =============================================================================
# Main orchestrator
# =============================================================================
def _alternate_frames(post: pd.DataFrame) -> List[Tuple[str, str, pd.DataFrame]]:
    """M5b copies with one exposure column replaced by its alternate."""
    frames: List[Tuple[str, str, pd.DataFrame]] = []
    for model_id, label, dest, source, _color, _legend in ALT_MODELS:
        if source not in post.columns:
            print(f"  Warning: {model_id} skipped; {source} was not built.")
            continue
        frame = post.copy()
        frame[dest] = pd.to_numeric(frame[source], errors="coerce")
        if {ACLED_COL, "cyclone_endline_window_z"}.issubset(frame.columns):
            frame["cyclone_x_acled"] = pd.to_numeric(frame["cyclone_endline_window_z"], errors="coerce") * pd.to_numeric(
                frame[ACLED_COL], errors="coerce"
            )
        frames.append((model_id, label, frame))
    return frames


def run_erl_nested(
    *,
    out_dir: Optional[str] = None,
    include_sdm_robustness: bool = True,
    include_iom_idp: Optional[bool] = None,
) -> Path:
    include_iom = INCLUDE_IOM_IDP_ROBUSTNESS if include_iom_idp is None else bool(include_iom_idp)
    out = get_output_dir(out_dir)
    remove_retired_outputs(out)
    print("=" * 72)
    print(f"ERL pipeline build: {PIPELINE_BUILD}")
    cfg = TIMELINES[ENDLINE]
    print(f"  endline: {ENDLINE}")
    print(f"  predictors: {cfg['endline_window_start']} .. {cfg['endline_window_end']}")
    print(f"  cultivated share: {cfg['cultivated_suffix']}")
    print(f"  ACLED event types dropped: {', '.join(ACLED_DROP_EVENT_TYPES)}")
    print(f"  predictor prefix: {PREDICTOR_PANEL_PREFIX}")
    print(f"  IOM terms in M5a and M5b: {'ON' if include_iom else 'OFF'}")
    print(f"  output dir: {out.resolve()}")
    print("=" * 72)
    if not include_iom:
        print(
            "WARNING: IOM is OFF, so M5a and M5b are not fit. Call "
            "run_erl_nested(include_iom_idp=True). If this is Colab, use Runtime → Restart session first."
        )

    post, w, g = build_post_level_frame(include_iom_idp=include_iom, period_key=ENDLINE)

    ie_specs = build_ie_specs()
    main_specs = build_main_specs(include_iom)
    all_specs = ie_specs + main_specs
    print("Reported models:", ", ".join(s.model_id for s in main_specs))
    if DECISION_MODEL not in {s.model_id for s in main_specs}:
        raise RuntimeError(f"{DECISION_MODEL} was not built. IOM must be on so M5a and M5b are fit.")

    ols_coef_parts: List[pd.DataFrame] = []
    ols_diag_rows: List[Dict[str, Any]] = []
    design_cache: Dict[str, Tuple[np.ndarray, np.ndarray, List[str], NestedSpec]] = {}
    resid_maps: Dict[str, Dict[str, Any]] = {}

    print("\n=== IE1–IE5 and M1–M5b: OLS + Moran/LM ===")
    for spec in all_specs:
        print(f"  {spec.model_id}: {spec.label}")
        y, x, names, d_design = _design_matrix(post, spec.predictors, skip_zscore=spec.skip_zscore)
        design_cache[spec.model_id] = (y, x, names, spec)
        cdf, diag, resid = fit_ols_with_diagnostics(
            y, x, names, w, model_id=spec.model_id, model_label=spec.label, set_name=spec.set_name
        )
        ols_coef_parts.append(cdf)
        ols_diag_rows.append(diag)
        if spec.model_id == DECISION_MODEL:
            resid_maps[f"{spec.model_id}_ols"] = {
                "resid": resid,
                "admin_post_ids": d_design["admin_post_id"].astype(str).tolist(),
                "title": f"LISA clusters: {spec.model_id} OLS residuals",
                "filename": f"fig_erl_lisa_{spec.model_id}_ols_resid.png",
            }

    m5b_spec = next(s for s in main_specs if s.model_id == DECISION_MODEL)
    print("\n=== M5b alternate specifications: OLS + Moran/LM ===")
    for model_id, label, alt_post in _alternate_frames(post):
        print(f"  {model_id}: {label}")
        y, x, names, _d_design = _design_matrix(
            alt_post, m5b_spec.predictors, skip_zscore=m5b_spec.skip_zscore
        )
        alt_spec = NestedSpec(
            model_id,
            label,
            m5b_spec.predictors,
            is_interaction=True,
            is_iom_robustness=True,
            set_name="ALT",
            skip_zscore=m5b_spec.skip_zscore,
        )
        design_cache[model_id] = (y, x, names, alt_spec)
        cdf, diag, _resid = fit_ols_with_diagnostics(
            y, x, names, w, model_id=model_id, model_label=label, set_name="ALT"
        )
        ols_coef_parts.append(cdf)
        ols_diag_rows.append(diag)

    ols_coefs = pd.concat(ols_coef_parts, ignore_index=True)
    ols_diag = pd.DataFrame(ols_diag_rows)
    try:
        write_predictive_contribution(out, design_cache)
    except Exception as exc:
        print(f"Warning: predictive-contribution table failed: {exc}")
    ie_diag = ols_diag[ols_diag["set_name"].eq("IE")].copy()
    main_diag = ols_diag[ols_diag["set_name"].eq("M")].copy()

    m6_diag = ols_diag[ols_diag["model_id"].eq(DECISION_MODEL)].iloc[0].to_dict()
    knn_choice, knn_note = anselin_florax_robust_choice(m6_diag)
    both = both_lag_and_error_indicated(m6_diag)

    m6_compare_parts: List[pd.DataFrame] = []
    y_m6, x_m6, names_m6, m6_spec = design_cache[DECISION_MODEL]
    m6_compare_parts.append(ols_coefs[ols_coefs["model_id"].eq(DECISION_MODEL)].assign(stage="OLS"))

    print(f"\n=== {DECISION_MODEL} queen-contiguity check ===")
    queen_w, queen_build_note = _queen_weights(g)
    queen_diag: Dict[str, Any] = {}
    queen_choice, queen_note = "unavailable", queen_build_note
    if queen_w is not None:
        try:
            _cdf_q, queen_diag, _resid_q = fit_ols_with_diagnostics(
                y_m6,
                x_m6,
                names_m6,
                queen_w,
                model_id=DECISION_MODEL,
                model_label=m6_spec.label,
                set_name="M",
                weights_name="queen",
            )
            queen_choice, queen_lm_note = anselin_florax_robust_choice(queen_diag)
            queen_note = f"{queen_build_note} {queen_lm_note}"
            print(f"  queen robust LM → {queen_choice}")
        except Exception as exc:
            queen_choice, queen_note = "unavailable", f"{queen_build_note} Queen LM failed ({exc})."
            print(f"  Warning: {queen_note}")
    else:
        print(f"  Warning: {queen_build_note}")

    robust_choice, weights_name, robust_note = apply_queen_check(knn_choice, knn_note, queen_choice, queen_note)
    spatial_w = queen_w if weights_name == "queen" and queen_w is not None else w
    if weights_name == "queen":
        both = both_lag_and_error_indicated(queen_diag)
    spat_coef_parts: List[pd.DataFrame] = []
    spat_sum_rows: List[Dict[str, Any]] = []
    m6_by_form: Dict[str, Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]] = {}
    fit_w = spatial_w if robust_choice in {"SAR", "SEM"} else w
    fit_w_name = weights_name if robust_choice in {"SAR", "SEM"} else PRIMARY_W
    print(f"\n=== {DECISION_MODEL} SAR and SEM on {fit_w_name} ===")
    for form in ("SAR", "SEM"):
        try:
            cdf, summ, resid = fit_spatial_model(
                y_m6,
                x_m6,
                names_m6,
                fit_w,
                form=form,
                model_id=DECISION_MODEL,
                model_label=m6_spec.label,
                set_name="M",
                weights_name=fit_w_name,
            )
            m6_by_form[form] = (cdf, summ, resid)
            m6_compare_parts.append(cdf)
            spat_coef_parts.append(cdf)
            spat_sum_rows.append(summ)
            print(f"  {DECISION_MODEL} {form}: AIC={summ['aic']:.2f}, pseudo-R2={summ['pr2']:.4f}")
        except Exception as exc:
            print(f"  Warning: {DECISION_MODEL} {form} fit failed: {exc}")
    if robust_choice in {"SAR", "SEM"} and "SAR" in m6_by_form and "SEM" in m6_by_form:
        choice, explanation = resolve_lag_vs_error(
            robust_choice, robust_note, m6_by_form["SAR"][1], m6_by_form["SEM"][1]
        )
    elif robust_choice == "unidentified" and "SAR" in m6_by_form and "SEM" in m6_by_form:
        choice, explanation = force_form_by_aic(
            robust_note, m6_by_form["SAR"][1], m6_by_form["SEM"][1]
        )
    elif robust_choice in {"SAR", "SEM", "OLS"}:
        choice = robust_choice
        explanation = robust_note + " AIC confirmation could not be computed."
    else:
        choice = "unidentified"
        explanation = robust_note + " No AIC tie-break was applied."
    print(f"\n=== {DECISION_MODEL} knn5 → {knn_choice}; queen → {queen_choice}; reported form → {choice} ({weights_name}) ===")
    print(f"  {explanation}")

    if choice in {"SAR", "SEM"} and choice in m6_by_form:
        _cdf, _summ, resid = m6_by_form[choice]
        _y, _x, _n, d_design = _design_matrix(post, m6_spec.predictors, skip_zscore=m6_spec.skip_zscore)
        resid_maps[f"{DECISION_MODEL}_spatial"] = {
            "resid": resid,
            "admin_post_ids": d_design["admin_post_id"].astype(str).tolist(),
            "title": f"LISA clusters: {DECISION_MODEL} {choice} residuals ({weights_name})",
            "filename": f"fig_erl_lisa_{DECISION_MODEL}_spatial_resid.png",
            "w": spatial_w,
        }

    if "SAR" in m6_by_form:
        _cdf_sar, _sum_sar, resid_sar = m6_by_form["SAR"]
        _y, _x, _n, d_design = _design_matrix(post, m6_spec.predictors, skip_zscore=m6_spec.skip_zscore)
        resid_maps[f"{DECISION_MODEL}_sar"] = {
            "resid": resid_sar,
            "admin_post_ids": d_design["admin_post_id"].astype(str).tolist(),
            "title": f"LISA clusters: {DECISION_MODEL} SAR residuals ({fit_w_name})",
            "filename": f"fig_erl_lisa_{DECISION_MODEL}_SAR_resid.png",
            "w": fit_w,
        }

    if "SAR" in m6_by_form and "SEM" in m6_by_form:
        for form in ("SAR", "SEM"):
            for mid in REPORTED_MODELS:
                if mid == DECISION_MODEL or mid not in design_cache:
                    continue
                y_i, x_i, names_i, spec_i = design_cache[mid]
                try:
                    cdf, summ, _resid = fit_spatial_model(
                        y_i,
                        x_i,
                        names_i,
                        fit_w,
                        form=form,
                        model_id=mid,
                        model_label=spec_i.label,
                        set_name="M",
                        weights_name=fit_w_name,
                    )
                    spat_coef_parts.append(cdf)
                    spat_sum_rows.append(summ)
                    print(f"  {mid} {form} coefficient table OK")
                except Exception as exc:
                    print(f"  Warning: {mid} {form} coefficient table failed: {exc}")

    forms_to_try = []
    if include_sdm_robustness and both:
        forms_to_try.append("SDM")
    if choice == "SEM":
        forms_to_try.append("SDEM")
    for form in forms_to_try:
        try:
            cdf, summ, _resid = fit_spatial_model(
                y_m6,
                x_m6,
                names_m6,
                spatial_w,
                form=form,
                model_id=DECISION_MODEL,
                model_label=m6_spec.label,
                set_name="M",
                weights_name=weights_name,
            )
            m6_compare_parts.append(cdf)
            spat_coef_parts.append(cdf)
            spat_sum_rows.append(summ)
            print(f"  {DECISION_MODEL} {form} robustness OK")
        except Exception as exc:
            print(f"  Warning: {DECISION_MODEL} {form} failed: {exc}")

    alt_forest_groups: List[Tuple[str, str, pd.DataFrame]] = []
    print("\n=== M5b alternate specifications: SAR and SEM ===")
    for model_id, label, _dest, _source, color, legend in ALT_MODELS:
        packed = design_cache.get(model_id)
        if not packed:
            continue
        y_i, x_i, names_i, spec_i = packed
        by_form: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]] = {}
        for form in ("SAR", "SEM"):
            try:
                cdf, summ, _resid = fit_spatial_model(
                    y_i,
                    x_i,
                    names_i,
                    fit_w,
                    form=form,
                    model_id=model_id,
                    model_label=spec_i.label,
                    set_name="ALT",
                    weights_name=fit_w_name,
                )
                cdf = cdf.copy()
                cdf["reported_spatial_form"] = False
                by_form[form] = (cdf, summ)
                summ["reported_spatial_form"] = False
                spat_coef_parts.append(cdf)
                spat_sum_rows.append(summ)
                print(f"  {model_id} {form}: AIC={summ['aic']:.2f}")
            except Exception as exc:
                print(f"  Warning: {model_id} {form} failed: {exc}")
        alt_diag = ols_diag[ols_diag["model_id"].astype(str).eq(model_id)]
        if len(alt_diag) == 0 or "SAR" not in by_form or "SEM" not in by_form:
            continue
        robust_choice_i, robust_note_i = anselin_florax_robust_choice(alt_diag.iloc[0].to_dict())
        if robust_choice_i in {"SAR", "SEM"}:
            form_i, note_i = resolve_lag_vs_error(
                robust_choice_i, robust_note_i, by_form["SAR"][1], by_form["SEM"][1]
            )
        elif robust_choice_i == "OLS":
            form_i, note_i = "OLS", robust_note_i
        else:
            form_i, note_i = force_form_by_aic(robust_note_i, by_form["SAR"][1], by_form["SEM"][1])
        print(f"  {model_id} → {form_i}. {note_i}")
        if form_i in by_form:
            by_form[form_i][1]["reported_spatial_form"] = True
            by_form[form_i][0]["reported_spatial_form"] = True
            rows = by_form[form_i][0]
        elif form_i == "OLS":
            rows = ols_coefs[ols_coefs["model_id"].astype(str).eq(model_id)].copy()
        else:
            continue
        ols_diag.loc[ols_diag["model_id"].astype(str).eq(model_id), "reported_spatial_form"] = form_i
        rows = rows[~rows["term"].astype(str).str.match(r"(?i)^constant$")].copy()
        if len(rows):
            alt_forest_groups.append((f"{legend} ({form_i})", color, rows))

    if queen_diag:
        ols_diag = pd.concat([ols_diag, pd.DataFrame([queen_diag])], ignore_index=True)
    m6_compare = pd.concat(m6_compare_parts, ignore_index=True) if m6_compare_parts else pd.DataFrame()
    spat_coefs = pd.concat(spat_coef_parts, ignore_index=True) if spat_coef_parts else pd.DataFrame()
    spat_diag = pd.DataFrame(spat_sum_rows) if spat_sum_rows else pd.DataFrame()
    period_label = _ACTIVE_ENDLINE["label"]
    for frame in (ols_coefs, ols_diag, m6_compare, spat_coefs, spat_diag):
        if len(frame) and "period" not in frame.columns:
            frame.insert(0, "period", period_label)

    to_csv_sig(ols_coefs, out / "ERL_setAB_ols_coefficients.csv")
    to_csv_sig(ols_diag, out / "ERL_setAB_ols_diagnostics.csv")
    to_csv_sig(post, out / "ERL_post_level_analysis_frame.csv")
    if len(spat_coefs):
        to_csv_sig(spat_coefs, out / "ERL_setB_spatial_coefficients.csv")
    if len(spat_diag):
        to_csv_sig(spat_diag, out / "ERL_spatial_diagnostics.csv")

    if alt_forest_groups:
        try:
            _forest_plot_compared(
                alt_forest_groups,
                out / "fig_erl_M5b_alt_forest.png",
                title="M5b alternate specifications: reported spatial model",
            )
            print("  figure: fig_erl_M5b_alt_forest.png")
        except Exception as exc:
            print(f"  Warning: alternate forest plot failed: {exc}")

    print("\n=== Paper and supplemental figures ===")
    try:
        figs = export_presentation_figures(
            out,
            post=post,
            g=g,
            ols_coefs=ols_coefs,
            ols_diag=ols_diag,
            set_a_diag=ie_diag,
            set_b_diag=main_diag,
            b2_diag=m6_diag,
            choice=choice,
            explanation=explanation,
            both=both,
            b2_compare=m6_compare,
            spat_coefs=spat_coefs,
            include_iom=include_iom,
            resid_maps=resid_maps,
            w=w,
            queen_diag=queen_diag or None,
            weights_name=weights_name,
        )
        print(f"  wrote {len(figs)} figures")
    except Exception as exc:
        print(f"Warning: figure export failed: {exc}")

    print(f"\nWrote ERL outputs to: {out.resolve()}")
    model_ids = sorted(ols_coefs["model_id"].astype(str).unique().tolist()) if len(ols_coefs) else []
    print(f"  models in ERL_setAB_ols_coefficients.csv: {', '.join(model_ids)}")
    if include_iom and DECISION_MODEL not in model_ids:
        print(f"WARNING: IOM was ON but {DECISION_MODEL} is missing from the coefficient CSV.")
    elif include_iom:
        print(f"  OK: {DECISION_MODEL} is in the coefficient CSV.")

    try:
        publish_outputs_to_desktop(out)
    except Exception as exc:
        print(f"Warning: Desktop copy failed ({exc}). Outputs remain at: {out.resolve()}")

    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ERL M1–M5b spatial analysis (Cabo Delgado and Nampula)")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--no-sdm", action="store_true", help="Skip SDM/SDEM robustness on M5b")
    parser.add_argument(
        "--iom-idp",
        action="store_true",
        help="Include M5a and M5b IOM terms (default on)",
    )
    parser.add_argument("--no-iom-idp", action="store_true", help="Disable M5a and M5b IOM terms")
    # parse_known_args: ignore Jupyter/Colab kernel flags like ``-f kernel-....json``.
    if argv is None and _in_notebook():
        args, _unknown = parser.parse_known_args([])
    else:
        args, _unknown = parser.parse_known_args(list(argv) if argv is not None else None)
    include_iom = True if args.iom_idp else (False if args.no_iom_idp else None)
    warnings.filterwarnings("ignore", category=UserWarning)
    run_erl_nested(out_dir=args.out_dir or None, include_sdm_robustness=not args.no_sdm, include_iom_idp=include_iom)
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    # Colab/Jupyter cells set __name__ == "__main__" and inject ``-f kernel.json`` into argv.
    # Force IOM on in notebooks so an old in-memory False default cannot silently skip M5a and M5b.
    if _in_notebook():
        print(f"Notebook entry → run_erl_nested(include_iom_idp=True) [{PIPELINE_BUILD}]")
        run_erl_nested(include_iom_idp=True)
    else:
        raise SystemExit(main())
