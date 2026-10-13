"""
M3 Phase 2 mineral-ID constants (Global 85-band path).
Frozen after Gate4/G5 — do not retune mid multi-scene adapt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JOURNAL_ROOT = Path(r"D:\Journal\M3 Journal")
RELAB_DIR = Path(r"D:\spectral_libraries\RELAB")
USGS_ROOT = Path(r"D:\spectral_libraries\USGS\usgs_splib07\ASCIIdata")
USGS_M3T_DIR = USGS_ROOT / "ASCIIdata_splib07b_cvM3-target"
USGS_M3T_WAVES = USGS_M3T_DIR / "s07_M3t_M3_Wavelengths_TARGET_2011t3_micron_256c.txt"
USGS_M3T_MINERALS = USGS_M3T_DIR / "ChapterM_Minerals"

SHARED_DIR = JOURNAL_ROOT / "models" / "shared"
PHASE2_OUT = JOURNAL_ROOT / "Phase2_outputs"
GATES_DIR = PHASE2_OUT / "gates"

N_BANDS_GLOBAL = 85
LATENT_DIM = 64
N_ENDMEMBERS = 12

# Canonical Global-85 wavelengths (µm) — same as step0_load
M3_WAVELENGTHS_85_UM = np.array([
    0.46099, 0.50092, 0.54084, 0.58076, 0.62069, 0.66061, 0.70054, 0.73048,
    0.75044, 0.77040, 0.79037, 0.81033, 0.83029, 0.85025, 0.87021, 0.89017,
    0.91014, 0.93010, 0.95006, 0.97002, 0.98998, 1.00995, 1.02991, 1.04987,
    1.06983, 1.08979, 1.10976, 1.12972, 1.14968, 1.16964, 1.18960, 1.20957,
    1.22953, 1.24949, 1.26945, 1.28941, 1.30938, 1.32934, 1.34930, 1.36926,
    1.38922, 1.40919, 1.42915, 1.44911, 1.46907, 1.48903, 1.50899, 1.52896,
    1.54892, 1.57886, 1.61879, 1.65871, 1.69863, 1.73856, 1.77848, 1.81840,
    1.85833, 1.89825, 1.93818, 1.97810, 2.01802, 2.05795, 2.09787, 2.13780,
    2.17772, 2.21764, 2.25757, 2.29749, 2.33742, 2.37734, 2.41726, 2.45719,
    2.49711, 2.53703, 2.57696, 2.61688, 2.65681, 2.69673, 2.73665, 2.77658,
    2.81650, 2.85643, 2.89635, 2.93627, 2.97620,
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Scene strategy (Global 85 only; Bullialdus 256 deferred)
# ---------------------------------------------------------------------------
PILOT_SCENE = "Copernicus Crater"
PRIOR_SCENES = [
    "Mare Tranquillitatis",
    "Lunar Highlands South",
    "Copernicus Crater",
]
ADAPT_SCENES = [
    "Mare Imbrium",
    "Mare Serenitatis",
    "Jackson Crater",
]
DEFERRED_SCENES = [
    "Bullialdus Crater",       # 256-band target mode
    "Aristarchus Plateau",
    "South Pole Region",
    "Highlands near Tycho",
]

# Method tag — bump when continuum / SPM / proof gates change (re-run pilot through G4)
METHOD_VERSION = "depth_v2g_clusters"

# Match-filter CR (library + scene must share this)
CR_MATCH_FILTER_SIZE = 11   # ~ bands; M3 85-band spacing ~20–40 nm

# ---------------------------------------------------------------------------
# Lunar SPM suite (PRESENT drivers)
# ---------------------------------------------------------------------------
# 3-point BD on reflectance. LCP 2µm shorter (~1.90); HCP longer (~2.25).
SPM_TRIPLES = {
    "HCPINDEX2": (2.250, 2.050, 2.450),    # longer-wavelength 2µm pyroxene
    "LCPINDEX2": (1.900, 1.700, 2.100),    # shorter-wavelength 2µm pyroxene
    "BD1250": (1.250, 1.100, 1.400),       # plagioclase crystal-field
    "BD1000": (1.050, 0.890, 1.300),       # mafic 1µm (support / mute)
    "BD1500": (1.500, 1.430, 1.815),       # water ice (polar-gated)
    "BD1900": (1.930, 1.850, 2.070),       # diagnostic only
    "BD2210": (2.210, 2.165, 2.250),       # diagnostic only
}

# Multi-point olivine (simplified OLINDEX3 shoulders)
OLINDEX3_POINTS = (0.860, 0.980, 1.210, 1.330, 1.470, 1.750)

SPM_NAMES = [
    "OLINDEX3", "LCPINDEX2", "HCPINDEX2", "BD1250",
    "BD1000", "R750", "BD1500", "BD1900",
]
N_SPM = len(SPM_NAMES)

# Absolute detection floors (Gate2/G4 freeze candidates)
SPM_DETECT_THRESH = {
    "OLINDEX3": 0.010,
    "LCPINDEX2": 0.006,
    "HCPINDEX2": 0.006,
    "BD1250": 0.005,
    "BD1000": 0.008,
    "R750": 0.05,          # reflectance peak proxy — contrast-based
    "BD1500": 0.015,
    "BD1900": 0.008,
}

# Hotspot selectivity: raise thr to top-(TARGET) when absolute thr is too broad;
# disable only if still above MAX after the raise (true saturation).
SPM_TARGET_HOT_FRAC = 0.10
SPM_MAX_HOT_FRAC = 0.15
SPM_MIN_COMPONENT = 50
PROOF_DEPTH_BD = 0.008
PROOF_DEPTH_BD_MIN = 0.006
PROOF_HOT_MINUS_BLAND = 0.005
FAMILY_SAM_HIT_MAX = 0.28   # lunar mixtures; lab SAM often ~0.20-0.27
SAM_STRONG_MAX = 0.15
RARE_MIN_HOT = 80
MINOR_ABUND_THRESH = 0.28
# Unmixer secondary mixture report
UNMIX_EPOCHS = 40
UNMIX_MAX_PER_FAMILY = 3
FAMILY_ENDMEMBER_ORDER = ["Olivine", "LCP", "HCP", "Plagioclase", "Ilmenite"]
UNMIX_CORE_FAMILIES = ("Olivine", "LCP", "HCP", "Plagioclase")
# Collapse QA: mean max-abundance above this → mixture report marked FAILED (not citable)
UNMIX_COLLAPSE_MAX_ABUND = 0.85
# Flat QA: mean max-abundance below this → over-mixed / uninformative (not citable)
UNMIX_FLAT_MIN_MEAN_MAX = 0.20
# Flat QA: strongest core-family mean must reach this (else FAILED_FLAT)
UNMIX_FLAT_MIN_FAMILY_MAX = 0.32
# Freeze endmember matrix E during training (library anchors); only train abundance net
UNMIX_FREEZE_ENDMEMBERS = True
# Softmax temperature >1 softens one-hot collapse
UNMIX_SOFTMAX_TEMP = 1.5
# Per-pixel peakiness band (mean over batch of max_j a_j)
UNMIX_MEAN_MAX_LO = 0.32
UNMIX_MEAN_MAX_HI = 0.72
UNMIX_MEAN_MAX_WEIGHT = 0.05
# Batch-mean family abundance: no single core family should dominate the train set
UNMIX_FAM_MEAN_HI = 0.55
UNMIX_FAM_MEAN_LO = 0.12
UNMIX_FAM_DIV_WEIGHT = 0.08
# Mild entropy band around log(K_eff) — neither max-entropy flat nor zero-entropy one-hot
UNMIX_ENTROPY_TARGET = 1.20   # ~ between 2 and 4 effective endmembers
UNMIX_ENTROPY_BAND_WEIGHT = 0.02
# Balanced hotspot samples per enabled PRESENT SPM for unmix training
UNMIX_PER_SPM_SAMPLES = 800
UNMIX_BLAND_SAMPLES = 400
# Legacy knobs kept at 0 (v2c sparsity caused plag collapse; v2b entropy caused flat)
UNMIX_ENTROPY_WEIGHT = 0.0
UNMIX_SPARSITY_WEIGHT = 0.0

# Pyroxene soft support
PYX_D2_MIN = 0.006
# Stricter absolute 2µm floor when 1µm is non-positive (weak path)
PYX_D2_MIN_ABS_PATH = 0.008
# Only use these SPM names for cluster labeling (enabled PRESENT drivers)
CLUSTER_NAME_SPM = ("OLINDEX3", "LCPINDEX2", "HCPINDEX2", "BD1250")
# Multi-label clusters when 2nd contrast score within this fraction of 1st
CLUSTER_NAME_SECOND_FRAC = 0.75
# Clustering (informational — never overrides PRESENT claims)
CLUSTER_MAX_FIT = 15000
CLUSTER_ENCODE_MAX = 400_000
CLUSTER_MIN_N = 2
CLUSTER_MAX_N = 8
CLUSTER_MAX_NOISE = 0.85
CLUSTER_MAX_DOMINANCE = 0.995  # allow mildly imbalanced maps; reject only total collapse
CLUSTER_MCS_CANDIDATES = (25, 30, 40, 50, 60)
CLUSTER_METHODS = ("eom", "leaf")
CLUSTER_PER_SPM_FIT = 3500  # stratified fit samples per naming SPM

# Rare ilmenite: multi-evidence; PRESENT_RARE only on mare (highlands over-call fix)
RARE_ILM_MIN_HOT = 120
RARE_ILM_DEPTH_MIN = 0.015
RARE_ILM_HMB_MIN = 0.010
# Non-mare: never escalate above CANDIDATE (no PRESENT_RARE off-mare)
RARE_ILM_PRESENT_RARE_MARE_ONLY = True
RARE_ILM_MIN_HOT_NONMARE = 400
RARE_ILM_DEPTH_MIN_NONMARE = 0.030
RARE_ILM_HMB_MIN_NONMARE = 0.020
# Spatial clustering: need enough hits in at least one coarse cell
RARE_ILM_GRID_Y = 32
RARE_ILM_GRID_X = 16
RARE_ILM_MIN_CELL_HITS = 8
RARE_ILM_MIN_CELL_HITS_NONMARE = 20
# Olivine mixture flag: d2/d1 above this → olivine hotspot looks pyx-mixed
OL_MIX_D2_OVER_D1 = 0.30
# STRONG verdict requires this many strong (non-weak) PRESENT groups
STRONG_MIN_PRESENT_STRONG = 2
# Unmix soft prior toward PRESENT hotspot fractions
UNMIX_PRESENT_PRIOR_WEIGHT = 0.04

ICE_GROUP = "Water Ice"
POLAR_SCENE_NAMES = {"South Pole Region"}
# High-Ti mare terrains (ilmenite rare path uses mare thresholds)
MARE_SCENE_NAMES = {
    "Mare Tranquillitatis",
    "Mare Imbrium",
    "Mare Serenitatis",
}

# Diagnostic continuum depths for proof (on match-CR or reflectance)
PROOF_DEPTH_DEFS = {
    "d1.05": (1.050, 0.890, 1.300),
    "d1.25": (1.250, 1.100, 1.400),
    "d1.00": (1.000, 0.750, 1.300),
    "d1.50": (1.500, 1.430, 1.815),
    "d1.90": (1.900, 1.700, 2.100),   # LCP 2µm
    "d2.25": (2.250, 2.050, 2.450),   # HCP 2µm
}

# Groups claimable as PRESENT (depth keys aligned with 1µm / 2µm diagnostics)
PRESENT_GROUPS = {
    "Olivine": {"spm": "OLINDEX3", "depth_key": "d1.05", "family": "Olivine"},
    "Low-Ca Pyroxene": {"spm": "LCPINDEX2", "depth_key": "d1.90", "family": "LCP"},
    "High-Ca Pyroxene": {"spm": "HCPINDEX2", "depth_key": "d2.25", "family": "HCP"},
    "Plagioclase": {"spm": "BD1250", "depth_key": "d1.25", "family": "Plagioclase"},
    "Ilmenite": {"spm": "R750", "depth_key": "d1.00", "family": "Ilmenite"},
}

# ---------------------------------------------------------------------------
# Library QC / filters
# ---------------------------------------------------------------------------
USGS_FAMILY_KEYWORDS = {
    "Olivine": ["olivine", "forsterite", "fayalite"],
    "LCP": ["orthopyroxene", "enstatite", "hypersthene", "broncite", "pigeonite"],
    "HCP": ["clinopyroxene", "diopside", "augite", "hedenbergite"],
    "Plagioclase": ["plagioclase", "anorthite", "bytownite", "labradorite", "anorthosite", "andesine"],
    "Ilmenite": ["ilmenite"],
    "Spinel": ["spinel"],
}

USGS_BLOCKLIST = [
    "fiberglass", "roofing", "asphalt", "concrete", "paint", "plastic",
    "wood", "leaf", "grass", "soil", "manure", "ice_cream", "snow",
    "water_ice", "frost", "cloud", "aluminum", "steel", "rubber",
]

USGS_MAX_PER_FAMILY = 15
LIB_MIN_TOTAL = 40
LIB_MIN_PER_CORE_FAMILY = 3
CORE_FAMILIES = ["Olivine", "LCP", "HCP", "Plagioclase"]

# SRF FWHM (µm) for RELAB → M3 Global
SRF_FWHM_VNIR = 0.020
SRF_FWHM_SWIR = 0.030
SRF_SWIR_ONSET = 1.0

# Training
TARGET_TRAINING_SAMPLES = 12000
LIB_SAMPLE_N = 2000
SYNTH_MIXES = 4000
ENCODER_EPOCHS_FULL = 80
ENCODER_EPOCHS_ADAPT = 25
ENCODER_PATIENCE_FULL = 12
ENCODER_PATIENCE_ADAPT = 6

# USGS delete sentinel
USGS_DELETE = -1.23e34
