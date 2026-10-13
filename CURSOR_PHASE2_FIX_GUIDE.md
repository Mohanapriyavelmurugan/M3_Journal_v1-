# Phase 2 Mineral Identification — Root-Cause Audit and Fix Guide (for Cursor agent)

Scope: M3 Global-mode (85 bands, 0.461–2.976 µm) denoised cubes -> correct mineral identification.
Phase 1 (`step0`–`step5`, `run_pipeline.py`) is NOT to be modified.

Files reviewed in full: `phase2_mineral_identification.py`, `phase2_spectral_library.py`, `phase2_scientific_audit.py`,
`phase2_validation_proof.py`, `phase2a_continuum_check.py`, both `_patch_*.py`, both `.md` reports, key parts of
`phase2_trained_mineral_workflow.py` (config, library, training pool, unmixer, gating, presence, reassignment logic) and `step0_load.py` (bad-band mask, wavelengths).
Not read line-by-line: `step1`–`step5`, `run_pipeline.py`, and the figure/plot code of the trained workflow.

---------------------------------------------------------------------------
## 0. Executive summary (why minerals are not identified correctly)

The failures are not threshold-tuning problems. They are structural:

1. **The matching metric is geometrically broken.** Continuum-removed (CR) spectra are ~1.0 everywhere, so after L2 normalisation almost every spectrum points the same way (the all-ones vector). SAM then measures noise on a huge constant, not absorption shape. The 0.15 rad "good match" threshold is meaningless, which is why a "beat-flat" patch was needed.
2. **The continuum is wrong.** `fast_envelope` = 15-band running max + Gaussian. On the M3 grid that is a ~0.30 µm window at 1 µm and ~0.60 µm at 2 µm, i.e. about as wide as the pyroxene bands themselves. The envelope sags into the band, so depths are underestimated and shapes are distorted. Library "exact_hull" vs scene "fast_envelope" are also inconsistent between the two scripts.
3. **The library is wrong for the target.** Fresh terrestrial/lab minerals (USGS ASD, hematite, calcite, serpentine, gypsum...) are matched against space-weathered lunar regolith (shallow bands, red slope, dark, intimately mixed). Classes like apatite/carbonate/phyllosilicate/sulfate/water_ice/iron_oxide do not belong in a lunar M3 taxonomy.
4. **Endmember choice is arbitrary.** `select_endmembers` takes `idxs[0]` = the first file in alphabetical order for each class. Every abundance map depends on that single accident.
5. **Linear unmixing is done in the wrong space.** The unmixer mixes *continuum-removed reflectance* linearly. Lunar soil is an intimate (Hapke) mixture; CR of a mixture is not the mixture of CRs. Also: closed-set softmax (sum = 1) with no dark/featureless/weathered component and no residual-based rejection forces every pixel into the ~10 listed minerals.
6. **Gates guarantee a detection.** `spm_agree = SPM / p90(SPM of this scene)` is *relative*, so the top pixels of any scene look "olivine-supported" even if olivine is absent. Several priors (RPEAK1) saturate and never reject.
7. **Hard-coded answers are injected.** Terrain is looked up from the scene name; "featureless -> plagioclase" reassignment; ilmenite endmember is clamped to BD1000 >= 0.025 after every optimiser step; many reassignment rules fall back to `plagioclase_anorthite`. Results like "Jackson = 91% plagioclase + 8% OPX" are produced by these rules, not measured.
8. **Validation is circular / fabricated.** `M3_LITERATURE_TARGETS` in `phase2_validation_proof.py` contains invented per-scene percentages (e.g. 29% LCP, 70% glass) attributed to papers that do not publish such numbers. `_LIT_RANGES` are priors. Tier-1 "physical checks" (sum-to-one, tiny MSE) are true by construction (softmax; CR ~ 1). The Fisher F uses p(1-p)/n_pixels as within-variance -> astronomically large F for anything.
9. **Some target minerals cannot be found by CR + SAM at all.** Ilmenite and glass are dark and nearly featureless; pure plagioclase is featureless in VNIR (weak 1.25 µm only in very pure anorthosite). They need albedo/continuum-slope/TiO2-style methods, or must be reported as "featureless component", not a SAM class.

The honest, defensible product for M3 Global data is: **pyroxene (LCP vs HCP), olivine, spinel (if 2 µm w/o 1 µm), plus a "feldspathic/featureless" and "mature/dark" component**, each with calibrated uncertainty — not a 14–16 class terrestrial mineral list.

---------------------------------------------------------------------------
## 1. Detailed issues, by file

### 1.1 `phase2_spectral_library.py` (shared library) — HIGH

| # | Issue | Evidence | Fix |
|---|-------|----------|-----|
| L1 | Taxonomy includes terrestrial classes irrelevant to lunar M3 | `TARGET_MINERALS`: apatite, carbonate, phyllosilicate, sulfate, water_ice, iron_oxide, glass(obsidian) | Replace with a **lunar taxonomy** (section 3.1). Keep rare classes only behind an explicit `--include-exotic` flag, default off. |
| L2 | USGS splib07 ASD/Beckman spectra are terrestrial, coarse/fresh, high contrast | USGS loader is the bulk of entries | Prefer **RELAB lunar-relevant** spectra: Apollo/Luna mineral separates, lunar-analog pyroxene/olivine/plagioclase separates by grain size, LSCC soils. Use USGS only as fallback with caps and flag `source_quality`. |
| L3 | No grain-size / space-weathering treatment | none | Add **augmentation**: red-slope + darkening (npFe-like), and <45 µm / 45–125 µm grain classes where available (section 3.3). |
| L4 | Library keyword matching on file/folder names only | `infer_mineral_name` | Read the RELAB catalogue/metadata (sample ID, mineral, composition, grain size) instead of filename keywords. Record composition (Mg#, Wo/En/Fs) per entry. |
| L5 | `ilmenite`, `glass`, `spinel` flat-rejection gate is inconsistent | `FEATURELESS_OK = {anorthosite, plagioclase}` only | Do not put featureless classes into the SAM class set; handle via section 3.5. |
| L6 | "synthetic" Gaussians used as library fallback | `synthetic_library` | Delete synthetics from the matching/unmixing library. A synthetic Gaussian-on-slope cannot represent a mineral. |
| L7 | `lib_cr_match` built with `fast_envelope` (via callbacks) in SAM script but `exact_hull` for the full-band path in the trained script | `_patch_*` | One continuum function, one place (section 2.2). |

### 1.2 `phase2_mineral_identification.py` (SAM) — HIGH

| # | Issue | Fix |
|---|-------|-----|
| S1 | **SAM on CR+L2** (constant-dominated). Audit T2 measures this. | Replace with SAM / correlation on **band-depth vector `d = 1 - CR`** over the match window (or mean-centred CR). Add a minimum-depth requirement. See 2.3. |
| S2 | `fast_envelope` continuum (15-band max filter) | Replace with a proper **upper convex hull** or **straight-line continuum between fixed shoulders** (0.75 / 1.55 µm; 1.55 / 2.45 µm for the 2 µm band). See 2.2. |
| S3 | Absolute threshold `SAM_MAX_RAD = 0.15` | Calibrate on library leave-one-out and on noise-perturbed library (2.5). Use per-class thresholds from the null distribution, not a global 0.15. |
| S4 | **Relative SPM gate**: `ref = p90(scene SPM)`, `spm_agree = SPM/ref`, hard cut at 0.20 | Use **absolute physical thresholds** (e.g. BD1000 >= 0.03 *and* above the noise floor measured in the scene), never scene-relative normalisation. |
| S5 | Priors that cannot reject (`RPEAK1` for ilmenite/glass/iron_oxide/spinel) | Remove RPEAK1 priors. Ilmenite/glass: no SPM prior possible -> they are not SAM classes (3.5). |
| S6 | SPM computed from **raw reflectance** for the scene but library SPM from **CR spectra** (`lib_cr_full`); formulas differ in meaning | Compute scene and library SPM from the *same* representation, with the *same* continuum. |
| S7 | `OLINDEX3` is not the CRISM OLINDEX3 (it is `1 - mean(0.86,0.98,1.21,1.33,1.47)/R(1.75)`) | Implement a lunar olivine metric: broad 1.05 µm band with **no 2 µm band** (BD1000 high, BD2000 ~ 0), centre ~1.05 µm, width, and 1.25 µm shoulder shape. |
| S8 | Martian/terrestrial indices (BD1500, BD1900, BD2210, BD2300, SINDEX2) used as SPM | Remove from the default feature set (or mark diagnostics-only). They respond to thermal/residual noise at 2.3–2.5 µm. |
| S9 | `TERRAIN_BY_SCENE` hard-codes terrain from the scene name; unknown scenes default to `impact_ejecta` | Derive terrain from the data (median albedo at 0.75 µm, BD1000 median, FeO-proxy) or drop plausibility filtering. Do not decide the answer from the file name. |
| S10 | `IMPLAUSIBLE` filter removes minerals by terrain, and the near-tie rule prefers `COMMON_LUNAR` | These inject priors. Remove; report raw result + uncertainty, apply priors only in a clearly labelled post-hoc "interpretation" layer. |
| S11 | `confidence = gap_factor * quality_factor` with fixed constants (0.10, 0.15) | Replace with calibrated confidence from section 2.5. |
| S12 | Python loop over pixels for the COMMON_LUNAR tie-break | Vectorise or remove (S10). |
| S13 | Match window 0.75–2.45 µm drops 0.46–0.75 µm (Fe-oxide charge transfer, albedo/ slope info) | Keep as the *mineral-band* window but add a separate slope/albedo feature block (3.5). |

### 1.3 `phase2_trained_mineral_workflow.py` — HIGH

| # | Issue | Fix |
|---|-------|-----|
| T1 | `select_endmembers` -> `idxs[0]` (first alphabetical file) | Choose endmembers by **optimisation**: for each class pick medoid / or keep **multiple** endmembers per class (class-library unmixing, e.g. MESMA / sparse regression) and report class sum. |
| T2 | Linear unmixing of CR spectra | Unmix in **single-scattering-albedo (SSA) space** (Hapke): `w = 1 - ((-b + sqrt(b^2+4ac)) / (2a))` style inversion of reflectance, mix linearly in SSA, then compare. At minimum, mix **reflectance** (not CR) and include a continuum-slope term. |
| T3 | Softmax sum-to-one, no dark/featureless endmember, no residual rejection | Add a **dark/mature-soil (npFe) endmember** and a **featureless-bright (feldspathic) endmember**; use sum <= 1 (or add shade endmember); output per-pixel **RMSE of fit** and flag pixels with high residual as "unmodelled". |
| T4 | Training = library + synthetic linear mixes of the same endmembers | The network can only reproduce what it was fed. Use it as an *accelerator of a physically-based solver* (e.g. NNLS/FCLS ground truth on synthetic SSA mixtures) and **verify against NNLS** on real pixels. Report NN-vs-NNLS disagreement. |
| T5 | Reconstruction MSE ~1e-4 reported as "PASS" | In CR space almost any smooth guess has tiny MSE (target ~ 1). Evaluate on `1 - CR` depth, normalised by depth energy (e.g. fraction of band-depth variance explained). |
| T6 | `ilmenite` endmember clamped so BD1000 >= 0.025 after each step; `train_unmixer._ilmenite_endmember_idx` sentinel hack; `_spm_recon_w` cached on function object | Remove. Ilmenite is featureless-dark; clamping a Fe2+ band onto it makes it look like pyroxene. Remove all function-attribute globals. |
| T7 | `gate_mixture_abundances` zeroes abundances by SPM thresholds then renormalises | This makes abundances prior-driven. Replace with **fit-based** model selection (compare residual with/without the endmember; keep if residual drops > noise). |
| T8 | Dominant-map reassignment cascade (lines ~1997–2240): reassign to "preferred" minerals, "featureless -> plagioclase_anorthite", compound ilmenite gates, scene mare/highland switch by BD1000 median | Delete cascade. A pixel with no resolvable band is **"featureless / feldspathic-or-mature"**, not plagioclase. |
| T9 | `build_scene_presence`: PRESENT if mean abundance >= 5% or >=10% of pixels >= 0.15; POSSIBLE if SAM >= 100 pixels | Thresholds are arbitrary and mean abundance is dominated by whatever the endmember shapes absorb. Use detection by **band parameters + fit significance** (2.6). `sam_pix` matching by substring (`family in mn`) is fragile. |
| T10 | `_LIT_RANGES` used in `validate_outputs` | Priors, not validation. Remove from pass/fail. |
| T11 | HDBSCAN / silhouette negative | Cosmetic. Do not spend time here. |
| T12 | Training pool cr "fill" with per-band medians; `nan_to_num(train_x)` zero-fills before L2 | Mask-aware losses; no filling with medians inside the match window. |
| T13 | Encoder trained with SPM anchor built from the same SPM definitions that are wrong (S7, S8) | Re-do after SPM rewrite or drop the SPM anchor. |

### 1.4 `phase2_validation_proof.py` — CRITICAL (scientific integrity)

- `M3_LITERATURE_TARGETS` are **hand-typed numbers** presented as "derived from published mapping" (Pieters 2009, Mustard 2011, Staid & Pieters 2000...). Those papers do not give "70% basalt glass dominant pixels" or "29% LCP dominant pixels". **Delete this table.** Do not cite it in the journal manuscript.
- RMSE vs those numbers and the "OK/WARN/FAIL" bands are therefore meaningless.
- `tier3_fisher`: within-variance = p(1-p)/n_valid x 100 with n_valid ~ 10^6 -> F is huge for any difference between scenes; it proves nothing about correctness, only that scenes differ in label counts.
- `tier1`: sum-to-one and MSE < 0.001 hold by construction.
- Replace with the validation plan in section 4.

### 1.5 `phase2_scientific_audit.py` — keep, extend

This is the only honest diagnostic in the repo. Keep T1–T7 as regression tests. Extend as in 4.1 and make them fail the build when:
`LOO class accuracy (selected representation) < 70%` or `frac(best SAM < flat - margin) < 30%` on a mafic ROI, or `label flip under 1% noise > 15%`.

Known limitations of the audit itself: `loo_accuracy` uses nearest neighbour on the *same* library entries (files of the same mineral from the same collection leak); do group-wise CV by **sample ID / source file family**, not by random leave-one-out.

### 1.6 `phase2a_continuum_check.py` — minor

Fine as a Phase-1 preservation check. Notes:
- Features use Martian indices; add BD1000/BD2000 (the ones that matter on the Moon) and apply the same PASS/FAIL logic.
- Uses a linear shoulder continuum — good; that is the continuum style Phase 2 should adopt (2.2).

### 1.7 `_patch_original_method.py`, `_patch_trained_library.py` — remove

String-patch scripts with absolute paths and `text.index(...)`; they silently depend on the exact text of the target file. Once the real files are fixed, delete these and commit the resulting source.

---------------------------------------------------------------------------
## 2. Required technical changes (implementation spec)

### 2.1 One shared preprocessing module: `phase2_core.py` (new)

Single source of truth for: wavelength grid, valid-band mask, resampling to M3, continuum removal, band parameters, SAM/correlation, SPM. Both Phase 2 scripts import it. No duplicate `upper_hull_continuum`, `continuum_remove_*`, `l2_normalize`, `compute_spm`, `read_ascii_spectrum`, `_load_usgs_wavelengths` copies (currently duplicated in the two big scripts).

### 2.2 Continuum removal

- Define the **mineral window** `0.75–2.45 µm` (63 bands on the 85-band grid; keep mask `> 2.8` from Phase 1 plus drop > 2.45 for the match).
- Use the **upper convex hull on the window** (exact; 63 points -> cheap even for 10^7 pixels with a vectorised monotone-chain, or process in numba).
- Fallback if the hull is unstable at noisy shoulders: **two fixed-shoulder straight-line continua**: 1 µm band: ~0.75 to ~1.55 µm; 2 µm band: ~1.55 to ~2.45 µm (shoulder positions found as local maxima in tolerance windows, not fixed bands).
- **Never** use the 15-band `maximum_filter1d` envelope.
- Use the *same* function for library and scene. Unit test: a synthetic Gaussian of known depth on a sloped baseline must be recovered within 2% depth.

### 2.3 Matching representation

- Compute `d = 1 - CR` on the window; set `d < noise_floor` to 0 (noise floor from scene, 2.4).
- Primary matcher: **SAM or Pearson correlation on `d`** (not on CR+L2). Add the required constraint `||d|| > k * noise` before any label.
- Secondary: **band parameters** — centre, depth, width (FWHM), area, and band-area ratio (BAR) for 1 µm and 2 µm bands, with the usual Band I centre vs BAR trajectories (LCP/HCP/olivine discrimination).
- Report *both* and require agreement for a "confident" label.

### 2.4 Noise floor from the data

For each scene estimate per-band noise on **spatially smooth patches** (Phase 1 already has SNR). Propagate to `sigma_d` of the depth vector via Monte-Carlo (add noise -> recompute CR -> `d`). Use it for: minimum depth to count as an absorption, SAM null distribution, confidence.

### 2.5 Calibration (replaces the 0.15 rad threshold and the fake "confidence")

1. Take library spectra (grouped by sample family), add realistic lunar degradation (3.3) and scene noise, run the full matcher.
2. Measure accuracy / confusion matrix vs true class as a function of band depth and SNR.
3. Derive per-class decision thresholds at a fixed false-positive rate and a calibrated probability (isotonic/Platt on a held-out family split).
4. Output **probability + abstention** ("unresolved / featureless") rather than a forced label.

### 2.6 Detection logic (what "present in scene" means)

A mineral/group is **detected in a pixel** only if all hold:
- required diagnostic bands exist with depth > k·sigma_d (e.g. k = 3),
- band centres within the physically allowed range (e.g. 1 µm centre ~0.90–0.95 LCP; ~0.98–1.05 HCP; ~1.05 olivine with no 2 µm; 2 µm centre ~1.9–2.0 LCP, ~2.2–2.3 HCP),
- spectral fit (SAM on `d` or SSA-unmixing residual) is better than the featureless model by a margin.

A mineral is **present in the scene** if detected in at least N contiguous pixels (spatial coherence, e.g. >= 30 pixels forming a cluster) *and* appears in the spectra that survive the denoising check (compare with the *raw* cube: the band must be there before denoising too — Phase 1 outputs both).

### 2.7 Taxonomy of outputs

See 3.1. Never output a class that the data cannot support (apatite, sulfate, carbonate, hematite on the Moon at M3 resolution -> not in the default list).

### 2.8 Remove all scene-name / terrain hard-coding

Delete `TERRAIN_BY_SCENE`, `IMPLAUSIBLE`, `_scene_is_mare` switches, preferred-order reassignments. If a prior is wanted, compute terrain from data (median 0.75 µm reflectance, BD1000 median, 1.0/0.75 ratio) and expose it as an *interpretation* field, never as a gate that changes labels.

---------------------------------------------------------------------------
## 3. Library and model design

### 3.1 Lunar taxonomy for M3 Global (recommended default)

Detectable by bands:
- Low-Ca pyroxene (orthopyroxene / pigeonite): 0.90–0.95 + ~1.9–2.0 µm
- High-Ca pyroxene (augite/diopside): ~1.0–1.05 + ~2.2–2.3 µm
- Olivine (Mg-rich to Fe-rich): broad 1.05 µm, no 2 µm
- Spinel (Mg-Al, "pink spinel"): 2 µm only, no 1 µm
- Plagioclase (pure/shocked): 1.25 µm weak band (only highest-purity anorthosite), else featureless
- Glass/impact melt: broad 1.0 + 1.9–2.0, asymmetric, no sharp pyroxene shape

Not detectable by bands (treat as components, not classes):
- Ilmenite / opaques: dark, flat (lowers albedo)
- Mature soil (npFe): red slope + darkening
- Feldspathic "featureless bright": high albedo + no bands

Optionally add "unresolved".

### 3.2 Library sources (RELAB first)

- Apollo / Luna samples and mineral separates: pyroxene, olivine, plagioclase separates with measured grain sizes (<45, 45–125 µm).
- Terrestrial analogues only if they cover the composition (Mg#, Wo) range and are labelled as such.
- Store metadata per entry: source file, sample ID, composition, grain size, instrument. Split train/test **by sample ID** (not by spectrum).

### 3.3 Space-weathering and mixing augmentation

- Add npFe-like **continuum slope + darkening** to library spectra (parameterised, sampled per realisation), and reduce band depth accordingly.
- Linear **SSA** mixing for pairs/triples + dark component for training/evaluation sets.
- Keep the augmentation parameters in the report (reproducibility).

### 3.4 Unmixing

Preferred, in order:
1. **NNLS/FCLS in SSA space** with a dark endmember and a featureless-bright endmember, residual-based pixel rejection.
2. **MESMA-style** model selection over multiple spectra per class (BIC / residual drop).
3. Neural unmixer only as a fast approximator trained on (2)'s outputs; always report NN-vs-solver error on a held-out pixel set.

### 3.5 Featureless components

- Estimate **0.75 µm albedo**, **continuum slope** (e.g. 0.75–1.55 µm), and a **1.0/0.75 colour ratio** per pixel; map "feldspathic" vs "mature mare soil" from these.
- TiO2/FeO proxies (Lucey-type) need the visible-NIR (0.75, 0.95 µm and ~0.415/0.75 UV-VIS) — M3 starts at 0.46 µm so use M3-adapted relations from the literature; do not claim "ilmenite detected" from SAM.
- Report "ilmenite" only as "high-Ti proxy" with the method named.

---------------------------------------------------------------------------
## 4. Validation plan (replace `phase2_validation_proof.py`)

### 4.1 Synthetic truth tests (must pass before running on scenes)
- Known mixtures from held-out library families + degradation + noise -> recover class presence and band centres.
- Report precision/recall per class vs depth and SNR; report minimum detectable abundance.

### 4.2 Ground-truth sites with *real* published measurements
- Use measured data, not remembered percentages: Apollo/Luna soil compositions (Lunar Soil Characterization Consortium), published M3 band-parameter maps for the same pixels, Apollo 17 / 16 / 11 / 12 landing-site pixels with **cited** values.
- For every comparison, store: citation (paper, table/figure), what was measured (modal %, wt% oxide, band centre), and why it is comparable (scale, mixing, units). If you cannot cite a table, you do not have ground truth.

### 4.3 Cross-checks that do not need literature
- **Raw vs denoised**: detected bands must be present in raw too.
- **Spatial coherence**: detections in 3x3 neighbourhoods; isolated pixels are noise.
- **Band-parameter consistency**: Band I centre vs Band II centre vs BAR trajectory (pyroxene Ca-content) must lie on the expected trajectories.
- **Independent method agreement**: SAM/depth matching vs unmixing vs band parameters.

### 4.4 Statistics
- Replace Fisher F-on-pixel-counts with scene-level effect sizes + bootstrap CIs (resample spatial blocks, not pixels).

---------------------------------------------------------------------------
## 5. Ordered work plan for the Cursor agent

Execute in this order; do not skip the tests.

1. **Create `phase2_core.py`** (2.1) with: `m3_window()`, `continuum_hull()`, `depth_vector()`, `sam()`, `band_parameters()`, `noise_floor()`. Unit tests with synthetic Gaussian-on-slope spectra (depth recovery, centre recovery, invariance to slope).
2. **Fix library** (`phase2_spectral_library.py`): lunar taxonomy, metadata-driven selection, remove synthetics, group IDs; write `library_manifest.csv` (entry, source, composition, grain size, group). Print counts per class and per group.
3. **Matching rebuild** in `phase2_mineral_identification.py`: depth-space SAM + band-parameter test; remove relative SPM gate, terrain plausibility, COMMON_LUNAR tie-break; add `unresolved`.
4. **Calibration script** `phase2_calibration.py` (2.5): group-wise CV on the library with degradation + noise; output thresholds and confusion matrices; wire thresholds into matcher.
5. **Unmixer rebuild** (`phase2_trained_mineral_workflow.py`): SSA space, dark + featureless endmembers, multiple endmembers per class, residual output; delete reassignment cascade (T6, T7, T8, T10); keep neural net only as approximator vs NNLS.
6. **Scene presence rewrite** (2.6): detection by bands + fit significance + spatial coherence; remove mean-abundance thresholds.
7. **Replace validation**: delete `M3_LITERATURE_TARGETS`, `_LIT_RANGES`; implement section 4; extend `phase2_scientific_audit.py` with group-wise CV and CI tests.
8. **Delete** `_patch_original_method.py`, `_patch_trained_library.py`, and all stale `06_mineral_id` / `07_mineral_workflow` outputs before rerun (stale files have already misled the reports once).
9. **Re-run** on Mare Tranquillitatis (known LCP/HCP mixtures near Apollo 11), Copernicus (known central-peak olivine/noritic), Lunar Highlands South. Check against section 4, not against hand-typed numbers.

Acceptance criteria (all must hold):
- Synthetic recovery: >= 90% precision for pyroxene/olivine at depth >= 5% and SNR >= 50; documented minimum detectable depth.
- Label flip under 1% noise <= 10%.
- No class appears that the data cannot support; `unresolved` fraction reported.
- Band-centre/BAR plot of detected pyroxene pixels falls on published trajectories.
- No scene name, terrain string or literature-percentage table appears in code paths that change labels.

---------------------------------------------------------------------------
## 6. Things Cursor must NOT do

- Do not tune thresholds until the scene "looks right".
- Do not add more reassignment rules or "catch-all" suppressors.
- Do not use scene names to choose endmembers/gates.
- Do not report a mineral present because a prior says it should be.
- Do not claim literature agreement without a citable table/figure per number.
- Do not touch Phase 1 (`step0`–`step5`, `run_pipeline.py`).
