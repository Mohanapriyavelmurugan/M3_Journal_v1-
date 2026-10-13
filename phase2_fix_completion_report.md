# Phase 2 Fix Completion Report

Date: 2026-10-04  
Guide: `CURSOR_PHASE2_FIX_GUIDE.md` (§5 ordered plan)

## Summary

Structural Phase 2 fixes from the guide are implemented: shared `phase2_core`, lunar taxonomy without synthetics, **depth-space SAM** (hull continuum), removal of terrain / relative-SPM / reassignment cascades, deletion of fabricated literature targets and patch scripts, plus group-wise library calibration. Phase 1 was not modified.

---

## Done (by guide step)

### 1. `phase2_core.py` + unit tests
- New module: `Code_scripts/phase2_core.py`
  - `m3_window`, `continuum_hull`, `continuum_shoulders`, `continuum_remove`, `depth_vector`, `sam`, `pearson_corr`, `band_parameters`, `classify_mafic_from_bands`, `noise_floor`, `match_depth_library`
- Tests: `Code_scripts/test_phase2_core.py` — **ALL PASSED** (hull above spectrum, flat→zero depth, OPX/CPX/OLV/SPN separation, depth-SAM identity, depth recovery ≤2%)

### 2. Library (`phase2_spectral_library.py`)
- Default taxonomy = lunar VNIR-detectable only: orthopyroxene, clinopyroxene, olivine, spinel, plagioclase, glass
- Exotic terrestrial classes behind `include_exotic=True` / `--include-exotic`
- Synthetics **off** by default (`synthetic_fn=None`)
- Matching representation: **depth vector** `d = 1 − CR` (hull), L2 of depth
- Sample `group_id` for CV; writes `library_manifest.csv`
- Calibration build: **100** entries (16 RELAB + 84 USGS), **0 synthetic**; classes OPX/CPX/OLV/PLAG/SPINEL

### 3. SAM matcher rebuild (`phase2_mineral_identification.py`)
- Hull continuum only (`fast_envelope` removed)
- GPU SAM on **depth** vectors, not CR+L2
- Absolute depth-energy gate + band-parameter agreement
- First-class labels: `featureless`, `unresolved`
- Removed: `TERRAIN_BY_SCENE`, `IMPLAUSIBLE`, relative SPM gate, COMMON_LUNAR tie-break, beat-flat / plag terrain exemption
- Terrain is **data-derived interpretation only** (albedo + BD1000), never a label gate
- Martian SPM indices removed from default FEATURES; lunar BD1000/BD2000/olivine/spinel/albedo/slope kept
- Loads thresholds from `Phase2_outputs/phase2_calibration.json` when present

### 4. Calibration (`phase2_calibration.py`)
- Group-wise leave-one-group-out depth-SAM CV + degradation/noise
- Outputs `Phase2_outputs/phase2_calibration.json`
- Latest run: **group-wise accuracy 82.0%**, label flip @1% noise **0.0%**

### 5. Trained workflow cleanup (`phase2_trained_mineral_workflow.py`)
- Endmembers: class **medoid** (not alphabetical `idxs[0]`)
- Lunar endmember taxonomy; synthetics disabled in `build_library`
- Removed: ilmenite BD1000 hard-clamp (T6), SPM reassignment cascade (T8), crater/mare endmember exclusion by scene type, `_LIT_RANGES` pass/fail (T10), abundance SPM zeroing (T7)
- High-RMSE pixels → `*_unmodelled_mask.npy` (no forced remapping)
- Scene presence rewritten toward coherent pixel counts + SAM (not mean-abundance PRESENT)

### 6–7. Validation
- `M3_LITERATURE_TARGETS` **deleted**; `phase2_validation_proof.py` rewritten for physical stats, spatial coherence, block-bootstrap CIs only
- `_LIT_RANGES` emptied / informational only in trained validator

### 8. Patch scripts
- Deleted: `_patch_original_method.py`, `_patch_trained_library.py`

---

## Calibration snapshot

| Metric | Value |
|--------|-------|
| Library size | 100 (0 synthetic) |
| Group-wise accuracy | 82.0% |
| Global SAM max (calibrated) | ~0.45 rad |
| Min depth norm | ~0.39 |
| Label flip @ 1% noise | 0.0% |

---

## Smoke results (Jackson Crater, depth-SAM)

| Class | Fraction |
|-------|----------|
| featureless | **56.6%** |
| unresolved | **42.9%** |
| plagioclase / spinel / OPX / others | <1% combined |

Interpretation: Jackson is highland/ejecta — high **featureless** is expected (guide §3.5). Large **unresolved** means measurable depth but lab library still mismatches weathered M3 (honest abstention, not forced plag/OPX). Weathering augments added; SSA unmixing + more lunar separates remain next.

## Remaining work

- Re-run Copernicus / Mare Tranquillitatis / Highlands South with cleared stale `07_*` outputs.
- Full SSA/Hapke unmixer (guide §3.4); NN only as NNLS approximator.
- More RELAB lunar glass / grain-size separates; strengthen §3.3 augmentation.
- Site QA with cited Apollo/LSCC tables only (guide §4.2).

---

## Acceptance checklist (guide §5)

| Criterion | Status |
|-----------|--------|
| Synthetic / library CV recovery ≥70–90% | Group-wise **82%** (depth-SAM) |
| Label flip under 1% noise ≤10–15% | **0%** on calib sample |
| No unsupported classes in default taxonomy | Pass (lunar only) |
| `unresolved` / `featureless` reported | Pass (SAM matcher) |
| No scene-name / terrain gates changing labels | Pass |
| No fabricated literature % in validation | Pass |

---

## Files touched

| File | Action |
|------|--------|
| `Code_scripts/phase2_core.py` | **Added** |
| `Code_scripts/test_phase2_core.py` | **Added** |
| `Code_scripts/phase2_calibration.py` | **Added** |
| `Code_scripts/phase2_spectral_library.py` | Rewritten (lunar + depth) |
| `Code_scripts/phase2_mineral_identification.py` | Depth-SAM rebuild |
| `Code_scripts/phase2_trained_mineral_workflow.py` | Cascade/clamp/lit cleanup |
| `Code_scripts/phase2_validation_proof.py` | Honest rewrite |
| `Code_scripts/_patch_*.py` | **Deleted** |
| `Phase2_outputs/phase2_calibration.json` | Generated |
| `phase2_fix_completion_report.md` | This report |
