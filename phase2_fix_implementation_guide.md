# Phase 2 Fix Implementation Guide
## For the Cursor Agent: Detailed Instructions to Repair Mineral Identification

---

## Introduction

This document is the complete implementation guide for fixing Phase 2 mineral identification. Every problem identified in the Phase 2 analysis document has a corresponding fix described here. Read the Phase 2 analysis document first to understand what each problem is and why it breaks the science. This document tells you exactly how to fix it.

The Phase 1 pipeline (`step0` through `step5`) is **correct** and should not be modified. All fixes are confined to Phase 2 scripts:
- `phase2_mineral_identification.py` (the fast matcher — needs major overhaul)
- `phase2_trained_mineral_workflow.py` (the trained workflow — needs targeted fixes)

The overall goal is: after these fixes, the mineral maps should show geologically plausible mineral distributions. For example, mare basalt scenes (Mare Imbrium, Mare Tranquillitatis, Mare Serenitatis) should show significant clinopyroxene and olivine (not 99% iron_oxide). Highland scenes (Copernicus Crater, Jackson Crater, Lunar Highlands South, Bullialdus Crater) should show significant plagioclase.

---

## Fix 1 — Repair the RELAB Library Loader (Most Critical Fix)

### Problem
`build_library()` uses `relab_root.glob("*.*")` which only finds files in the immediate root directory. RELAB data is organized in subdirectories by mineral class. This single line is the reason only 27 total entries are found.

### Fix Instructions

**In `phase2_mineral_identification.py`, in the `build_library()` function:**

Replace the RELAB scanning loop. Instead of:
```
for path in sorted(relab_root.glob("*.*")):
```

Use recursive scanning:
```
for path in sorted(relab_root.rglob("*.*")):
```

The `rglob("*.*")` call recursively searches all subdirectories for any file with an extension. This is the correct way to find RELAB spectra which are stored in per-mineral-class subdirectories.

Do the same fix in `phase2_trained_mineral_workflow.py` in whatever function loads RELAB spectra there.

**Additionally:** The mineral name inference uses the filename only (`infer_mineral_name(path)` looks at `path.name.lower()`). RELAB organizes files by directory, so the directory path often carries the mineral class name even when the filename is a code. Modify `infer_mineral_name()` to also check the full path string (including parent directory names) against the keyword dictionary:
- Currently: `low = path.name.lower()`
- Should be: `low = (str(path.parent.name) + " " + path.name).lower()`

**Expected result:** RELAB should yield 150–400 valid library entries rather than ~13.

---

## Fix 2 — Add M3 Spectral Response Function Convolution

### Problem
Lab spectra are at high resolution (1–5 nm), but M3 measures at 20–30 nm FWHM. Directly interpolating lab spectra to M3 bands without SRF convolution keeps sharp lab features that M3 cannot see, making SAM matches worse than they should be.

### Fix Instructions

**Add a new function** `convolve_with_m3_srf(wave_lib, refl_lib, waves_m3)` to both scripts.

The logic of this function:
- M3 FWHM is approximately 20 nm for bands below 1000 nm (bands 0–16 in 85-band mode), and 30 nm for bands above 1000 nm
- Gaussian FWHM to sigma: `sigma = FWHM / (2 × sqrt(2 × ln(2)))` = `FWHM / 2.3548`
- For each M3 band center wavelength `w_m3[b]`:
  - Compute the Gaussian weight for every library wavelength: `G(w_lib) = exp(-0.5 × ((w_lib - w_m3[b]) / sigma)^2)`
  - If no library wavelengths fall within 3 sigma of `w_m3[b]`: mark as NaN
  - Otherwise: `refl_convolved[b] = sum(G × refl_lib) / sum(G)` (weighted average)
- Return the convolved reflectance array at M3 band centers

**Call this function** inside `resample_to_m3()`, replacing the simple `np.interp` call with the SRF-convolved version. Keep the `np.interp` fallback if the library spectrum wavelength range is too sparse for convolution (fewer than 3 points within 3 sigma of any M3 band).

---

## Fix 3 — Implement Convex Hull Continuum Removal

### Problem
Library and pixel spectra are normalized by their median, then L2-normalized. This does NOT remove the continuum slope. As a result, the SAM comparison is comparing overall brightness shape (albedo + continuum) rather than the mineral absorption features alone.

### Fix Instructions

**Add a new function** `convex_hull_continuum_removal(wave_um, refl)` to both scripts.

The algorithm:
1. Input: wavelength array (shape B,) and reflectance array (shape B,). Both must be finite (handle NaN before calling).
2. Create 2D points: `pts = array of (wave[i], refl[i]) pairs` for all finite positions
3. Compute the upper convex hull of these points. The upper hull connects points such that no other point lies above the line between any two consecutive hull vertices.
   - Efficient algorithm: sort points by x-coordinate, then scan from left to right maintaining a stack where each new point is added and the stack is trimmed until the last three points make a left-turn (upper hull = right-turns only)
4. The continuum at each wavelength is the linear interpolation between adjacent hull vertices.
5. Divide each reflectance value by the continuum value at that wavelength: `CR[i] = refl[i] / continuum[i]`
6. At hull vertices, `CR = 1.0` (the continuum). At absorption features, `CR < 1.0`.
7. Restore NaN at dead band positions after computing CR.

**Apply this continuum removal** as the FIRST preprocessing step in both the library building pipeline and the pixel preprocessing pipeline. The order should be:
1. Convolve with M3 SRF (library spectra only)
2. Resample to M3 bands (library spectra only — scene pixels already in M3 bands)
3. Apply bad-band mask (wavelengths > 2.80 µm → NaN)
4. **Apply convex hull continuum removal**
5. Median normalization (optional, may be skipped if CR already normalizes shape)
6. L2 normalization → unit vector

This must be applied identically to both library spectra and scene pixel spectra. If the preprocessing chain is different between library and pixels, the SAM comparison is meaningless.

---

## Fix 4 — Fix the Library Quality Filtering

### Problem
The current resampling rejection criterion requires the library spectrum to cover `waves_m3.min()` to `min(2.5, waves_m3.max())`. This is good but not sufficient. Also, the NaN fraction threshold (25%) is too lenient.

### Fix Instructions

**In `resample_to_m3()`, tighten the quality criteria:**

After resampling (and after continuum removal in Fix 3), apply these checks:
1. **Coverage check:** The library spectrum must cover at least the range 0.75 µm to 2.50 µm continuously. This range covers the key pyroxene 1-µm band, the pyroxene 2-µm band, and most SPM diagnostic wavelengths.
2. **NaN fraction check:** Tighten from 25% to 15%. Spectra with >15% NaN after resampling have too many gaps.
3. **SPM coverage check:** At minimum, bands corresponding to BD1000 (1.0 µm), BD2000 (2.0 µm), and RPEAK1 (0.75 µm) must all be finite.
4. **Plausibility check:** After continuum removal, the CR spectrum must have at least 5 bands with `CR < 0.98` (showing at least some absorption structure). A flat CR spectrum (all values ≈ 1.0) may indicate a bad entry or a featureless spectrum — acceptable for plagioclase but flag it.
5. **Reflectance range check:** Original reflectance must be between 0.0 and 1.1 for at least 90% of finite bands.

Remove entries that fail any check. Log the rejection reason for debugging.

---

## Fix 5 — Fix the Synthetic Library Fallback

### Problem
The synthetic library is generated using overly simple Gaussian absorptions on a linear continuum. These synthetic spectra are so different from real M3 data that they consistently attract pixels to the "wrong" mineral class.

### Fix Instructions

**The synthetic library should only be used as a last resort** when a mineral class has zero real library entries AND is scientifically expected in lunar terrain. Apply these changes:

1. **Only generate synthetics for minerals with zero real library coverage** — if RELAB/USGS provides even one valid plagioclase spectrum, do not add the synthetic plagioclase.

2. **Improve synthetic spectrum physics:** Instead of a pure linear continuum plus Gaussian dips, generate more realistic shapes by:
   - Using the 85-band M3 wavelength array directly
   - Starting from a reference spectrum more similar to typical lunar reflectance: `refl_base = 0.10 + 0.08 × (waves - 0.46) / (2.97 - 0.46)` (a gentle slope from 0.10 to 0.18)
   - Absorptions should be modeled as combinations of two Gaussians (a more realistic double-lobed pyroxene band shape for pyroxene minerals)
   - For plagioclase: use a nearly featureless spectrum (very shallow broad absorption near 1.25 µm)
   - For olivine: use a broad, asymmetric composite band centered near 1.05 µm (olivine has multiple overlapping Fe²⁺ bands)

3. **Lower the weight** of synthetic spectra during SAM matching: if a best match comes from a synthetic entry, cap the confidence at 0.1 regardless of the computed confidence. Flag the assignment as "synthetic-matched" in the output JSON.

---

## Fix 6 — Fix the SAM Thresholds and Confidence Scoring

### Problem
The current code does not have a maximum SAM threshold — if a pixel's best SAM match is 0.42 radians (~24°), it still gets assigned to that mineral with full labeling. All assignments are equally "valid" regardless of how poor the match is.

### Fix Instructions

**Add a maximum SAM threshold:** If the best SAM for any pixel exceeds **0.15 radians** (~8.6°), classify that pixel as "unclassified" (label = -1) instead of assigning it to any mineral. Rationale: an 8.6° spectral angle indicates essentially no match — the pixel's spectrum does not resemble any library entry. Forcing an assignment at SAM > 0.15 rad produces meaningless results.

**In the classification loop, after computing `best` (best SAM values):**
- Where `best > 0.15`: set `mineral_idx = -1` (unclassified), `confidence = 0.0`, `uncertainty = 1.0`
- Where `best ≤ 0.15`: proceed with the existing confidence calculation

**Fix the confidence formula:**
The current formula has a problem: the term `(0.25 - best) / 0.25` becomes negative for SAM > 0.25 rad, and gets clipped to 0. This means all pixels with best_SAM > 0.25 have confidence = 0 (correct), but pixels with SAM = 0.05 rad (a good match) have confidence = 0.80, while we want a confident match at SAM = 0.05 rad to have confidence close to 1.0.

**Replace the confidence formula** with:
```
gap_factor = clip((second - best) / 0.10, 0, 1)
quality_factor = clip(1.0 - best / 0.15, 0, 1)   # 1.0 at SAM=0, 0.0 at SAM=0.15
conf = gap_factor × quality_factor
```

Then apply the SPM agreement modulation as before: `conf_final = conf × (0.35 + 0.65 × spm_agree)`.

---

## Fix 7 — Fix Farthest-Point Endmember Selection in Trained Workflow

### Problem
In `phase2_trained_mineral_workflow.py`, the endmember selection algorithm selects 16 endmembers by maximizing spectral diversity, but it produces duplicates (clinopyroxene_hcp ×4, ilmenite ×3). This wastes endmember slots on the same mineral.

### Fix Instructions

**In the endmember selection function** (wherever farthest-point sampling is implemented):

Add a constraint: **at most one endmember per named mineral class**. After selecting a new candidate endmember, check if its mineral class already appears in the selected set. If it does, skip it and try the next-farthest candidate.

The modified algorithm:
1. Start with the library entry having highest mean SPM value across all 10 SPM indices
2. Track selected mineral classes in a set
3. For each next endmember slot:
   - Compute Euclidean distance from all remaining library entries to all already-selected entries
   - Sort remaining entries by distance (farthest first)
   - Iterate through this sorted list; select the first entry whose mineral class is NOT already in the selected set
   - Add it to the selected set; add its mineral class to the tracking set
4. Continue until the target endmember count is reached

If the library does not contain enough mineral classes to fill all endmember slots (e.g., only 10 distinct classes but 16 slots): fill remaining slots with the second-best entry per class (allowing one repeat per class), or reduce the number of endmembers.

**The target endmember list for M3 lunar analysis** should include exactly these classes in priority order:
1. plagioclase_anorthite (most common highland mineral)
2. orthopyroxene_lcp (common highland and mare mineral)
3. clinopyroxene_hcp (dominant mare mineral)
4. olivine_forsterite (common in both terrains)
5. ilmenite (titanium-rich mare indicator)
6. mare_basalt_glass (impact melt and volcanic glass)
7. plagioclase_bytownite (Ca-rich plagioclase end-member)
8. plagioclase_labradorite (intermediate plagioclase)
9. olivine_fayalite (Fe-rich olivine end-member)
10. spinel (rare highland mineral)
11. apatite_rare (volatile-bearing mineral)
12. carbonate_rare (impact-processed or volatile-related)

This gives 12 endmembers with one per class, no duplicates.

---

## Fix 8 — Fix the Encoder Training to Produce Mineralogically Separated Latent Space

### Problem
The trained encoder produces DBCV = -0.318, indicating the latent space has no mineralogically meaningful structure. The NT-Xent contrastive loss is not sufficient on its own to force mineral separation when the library is small and the SPM anchor weight is too low.

### Fix Instructions

**In `phase2_trained_mineral_workflow.py`, in the encoder training section:**

**Fix 8a — Strengthen the SPM anchor loss:**

The SPM anchor loss weight is currently 0.10. This is too weak — the contrastive loss (NT-Xent) dominates and pulls spectra together based on augmentation similarity rather than mineral identity.

Increase the SPM anchor weight to **0.30**. This means:
```
L_total = L_NT-Xent + 0.30 × L_SPM + 0.05 × L_library_consistency
```

Also, modify the SPM loss to use a weighted MSE where SPM indices with higher values (more diagnostic signal) contribute more to the loss:
- Compute the mean SPM value across all training samples for each index
- High-mean indices (pyroxene, olivine) should have higher weight in the loss
- Weight per index = `1.0 / (mean_index_value + 0.01)` (inverse weighting so rare minerals contribute more)

**Fix 8b — Add a mineral class separation loss (supervised component):**

Even though no pixel labels are available, the library spectra have known mineral class labels. Use these to add a supervised loss component:

For pairs of library spectra in the same training batch:
- If they are from the **same mineral class**: their latent vectors should be close (cosine similarity > 0.8)
- If they are from **different mineral classes**: their latent vectors should be far (cosine similarity < 0.2)

This is a "labeled pair" contrastive loss. Compute it as:
- For same-class pairs: `max(0, margin_pos - cosine_similarity)^2`, with `margin_pos = 0.8`
- For different-class pairs: `max(0, cosine_similarity - margin_neg)^2`, with `margin_neg = 0.2`

Apply this loss **only to library spectra** (not to scene pixels, which have no labels). Weight = 0.20.

**Fix 8c — Fix the augmentation to preserve absorption features:**

The current augmentations may destroy narrow mineral absorptions. Add a protection rule:
- Identify all bands that fall within ±2 bands of any SPM diagnostic wavelength (1.0 µm, 2.0 µm, 2.295 µm, 1.93 µm, 2.21 µm, 2.10 µm, 1.50 µm, 0.75–0.84 µm range, 2.14 µm)
- These "protected bands" must never be dropped, never be smoothed beyond sigma=0.5, never be scaled beyond ±5% of their original value
- Apply this constraint in the augmentation function

**Fix 8d — Increase encoder training epochs and add a warmup:**

The current configuration (100 epochs, lr=1e-3) may converge too fast to a local minimum. Use:
- Learning rate warmup: for the first 10 epochs, increase LR linearly from 0 to 1e-3
- After warmup: cosine annealing from 1e-3 to 1e-5 over the remaining 90 epochs
- Total epochs: 120 (10 warmup + 110 annealing)

---

## Fix 9 — Add Mineralogical Plausibility Post-Filtering

### Problem
Physically impossible detections (water ice in mid-latitude dry terrain, phyllosilicates in young unprocessed mare basalt) pass through the system because there is no geological reality check.

### Fix Instructions

**Add a post-classification plausibility filter** in both scripts. This runs after the label map is generated.

For each scene, define the terrain type based on the scene name:
- Mare scenes (Mare Imbrium, Mare Serenitatis, Mare Tranquillitatis): `terrain = "mare_basalt"`
- Impact crater scenes (Copernicus, Jackson, Bullialdus): `terrain = "impact_ejecta"`
- Highland scenes (Lunar Highlands South): `terrain = "highland"`

For each terrain type, define **implausible minerals** — minerals that are physically impossible or extremely improbable in this setting:
- `mare_basalt`: water_ice is implausible (unless near-polar — these are not polar scenes). carbonate, phyllosilicate are very improbable.
- `impact_ejecta`: water_ice is implausible for mid-latitude craters. phyllosilicate, carbonate require special formation conditions.
- `highland`: ilmenite in high abundance is improbable (Ti-poor highland crust). sulfate is improbable.

**For pixels assigned to implausible minerals:**
- If their second-best SAM match is a plausible mineral with `second_SAM < 0.25 rad`: re-assign to the second-best plausible mineral
- Otherwise: assign to "unclassified" (-1)

**This filter must not override strong matches:** Only apply it when `confidence < 0.05`. High-confidence implausible detections should be kept and flagged with a warning — they may represent real rare occurrences.

**Add a minimum pixel count threshold for minority minerals:**
If fewer than **50 pixels** in the scene are assigned to a minority mineral (non-dominant class), that mineral's assignment is suspect. Flag those pixels in the output but do not remove them — they may represent isolated real occurrences.

---

## Fix 10 — Implement Proper SPM-Based Validation of Results

### Problem
There is no validation step that checks whether assigned mineral pixels actually show the expected spectroscopic signature of that mineral. A pixel assigned to "clinopyroxene" should have high HCPINDEX2. The current code computes SPM agreement during classification but does not generate a proper validation report.

### Fix Instructions

**Add a `validate_mineral_assignments()` function** to both scripts. This runs after the full classification and before saving outputs.

For each mineral class that has at least 50 assigned pixels:
1. Extract all assigned pixels' SPM values from the SPM map
2. For the mineral's diagnostic SPM indices (from `MINERAL_SPM_PRIORS` or `SPM_PRIORS`):
   - Compute the mean SPM value for assigned pixels
   - Compare to the scene-wide 90th percentile of that SPM index
   - If `mean_assigned_spm < 0.5 × scene_90th_percentile`: the assigned pixels lack the expected spectroscopic signature → this mineral detection is **not validated**
3. Compute the mean best_SAM for assigned pixels. If `mean_best_SAM > 0.12 rad`: the spectral matches are too poor → flag as questionable
4. Record: `validated` (passes both checks), `questionable` (fails one check), `invalid` (fails both)

**Include this validation in the output JSON:**
```json
{
  "mineral": "clinopyroxene_hcp",
  "pixel_count": 4532,
  "fraction_pct": 0.086,
  "mean_confidence": 0.231,
  "median_sam_rad": 0.089,
  "mean_spm_hcpindex2": 0.143,
  "scene_90th_spm_hcpindex2": 0.188,
  "spm_validation": "validated",
  "sam_validation": "validated",
  "overall_validation": "validated"
}
```

This makes the quality of each detection explicit rather than implied.

---

## Fix 11 — Fix the Output Summary to Report Unclassified Pixels

### Problem
The current `summarize()` function only reports minerals that appear in the label map. If most pixels are labeled iron_oxide (99%+) and a few are unclassified (-1), the -1 pixels are silently ignored. This hides the fact that the classification failed for those pixels.

### Fix Instructions

**In `summarize()`, always include an "unclassified" row** in the output:
- Count all pixels where `label == -1`
- Compute their mean best_SAM (these should all be large — that's why they're unclassified)
- Include in the summary JSON and CSV

This gives an immediate diagnostic: if "unclassified" is the second-largest class, it tells us that many pixels could not be matched to any library entry — a signal to improve the library.

Also add to the summary JSON:
- `mean_best_sam_all_valid_pixels`: the mean SAM across all valid pixels (before unclassified filtering)
- `fraction_below_sam_threshold`: fraction of valid pixels with best_SAM < 0.15 rad (the "actually classified" fraction)
- `effective_classification_rate`: this fraction expressed as a percentage

---

## Implementation Order

Do NOT implement all fixes simultaneously and re-run — debug one fix at a time in this order:

**Step 1:** Implement Fix 1 only (recursive RELAB loading). Run the library building function on one scene. Check: how many entries are found? Should be 150–400. If still ≤ 50: the RELAB directory structure is different from expected — investigate manually.

**Step 2:** After confirming library size > 100 entries, implement Fix 2 (SRF convolution). Check: do resampled library spectra look smoother? Plot 5 library spectra before/after convolution to verify.

**Step 3:** Implement Fix 3 (continuum removal). Apply to both library and one scene's pixels. Plot 5 pixels' CR spectra. Check: are mineral absorptions visible as dips below 1.0? The 1-µm pyroxene band should be clearly visible in mare pixels.

**Step 4:** Implement Fix 6 (SAM threshold and confidence formula). Re-run classification on one scene. Check: what fraction of pixels are now "unclassified"? What does the mineral map look like?

**Step 5:** Implement Fix 5 (better synthetics, lower synthetic weight) and Fix 4 (tighter quality filtering).

**Step 6:** Run the full pipeline on all 7 scenes. Check the validation output (Fix 10) to see which minerals are validated vs questionable.

**Step 7:** Implement Fix 7 (endmember selection fix) in the trained workflow. Run Copernicus Crater with trained workflow. Check: are endmembers unique per mineral class?

**Step 8:** Implement Fix 8 (encoder training improvements). Check DBCV score — should be > 0.1 for useful clustering.

**Step 9:** Implement Fix 9 (plausibility filter). Final quality check on all scenes.

**Step 10:** Implement Fix 11 (unclassified reporting) and Fix 10 (validation reporting). Produce final output JSONs.

---

## What "Correct" Results Should Look Like

After all fixes, the results for each terrain type should show:

**Mare Basalt scenes (Mare Imbrium, Mare Tranquillitatis, Mare Serenitatis):**
- Clinopyroxene (HCP) should be the dominant mineral: expected 30–60% of pixels
- Olivine should appear at 5–20% of pixels
- Plagioclase should be present at 10–30% (mare basalts have plagioclase but less than highlands)
- Ilmenite should appear at 5–15% (especially in Ti-rich mare basalts)
- Iron_oxide as a distinct class should not dominate; if it appears, confidence must be backed by actual hematite/magnetite SPM signal
- Unclassified: < 30% of pixels

**Impact Ejecta scenes (Copernicus, Jackson, Bullialdus):**
- Plagioclase (anorthosite + plagioclase subtypes) should be dominant: 40–70%
- Pyroxene (LCP + HCP mixture): 15–35%
- Olivine: 5–15%
- Glass: 3–10% (impact melt is expected in crater ejecta)
- Unclassified: < 30%

**Highland scenes (Lunar Highlands South):**
- Plagioclase dominant: 50–80%
- Orthopyroxene significant: 10–30%
- Olivine minor: 3–10%
- Unclassified: < 20%

**Validation signal:** For any mineral claiming >1% of a scene, its assigned pixels' mean SPM value for that mineral's diagnostic index should be at least 40% of the scene-wide 90th percentile for that index. If not, the detection is physically unvalidated.

---

## Important Constraints for the Cursor Agent

1. **Do not modify any Phase 1 scripts.** `step0_load.py` through `step5_validate.py` and `run_pipeline.py` are correct and must not be changed. All modifications are confined to `phase2_mineral_identification.py` and `phase2_trained_mineral_workflow.py`.

2. **Preserve the existing function signatures** for `process_scene()`, `main()`, and `build_library()`. The argument parsing and file I/O structure should remain the same — only the internal algorithms change.

3. **Test one scene first before running all 7.** Use Copernicus Crater as the primary test scene because it has the most pixels (5.29M), making statistical patterns clear.

4. **Save intermediate outputs** during development: save the raw continuum-removed cube and the filtered library spectra as numpy files. This allows debugging individual stages without re-running the entire pipeline.

5. **Print diagnostic statistics** at each major stage:
   - After library building: print mineral class distribution and total entry count
   - After continuum removal: print mean and std of CR values for 3 sample pixels
   - After classification: print the fraction of pixels in each mineral class AND the fraction classified as "unclassified"
   - After validation: print which minerals are "validated" vs "questionable"

6. **The RELAB and USGS paths are:**
   - RELAB: `D:\spectral_libraries\RELAB\`
   - USGS: `D:\spectral_libraries\USGS\`
   - These paths are already in both scripts as constants. If the directory structure inside RELAB is different from expected after the recursive glob, investigate what subdirectories exist before proceeding.
