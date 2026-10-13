# Phase 1 Deep Analysis — M3 Hyperspectral Denoising Pipeline
## End-to-End Methodology: From Raw IMG to Denoised Cube

---

## Overview

This document explains every single step taken in Phase 1 of the M³ (Moon Mineralogy Mapper) pipeline — from loading the raw satellite data files all the way to producing a scientifically validated, denoised hyperspectral data cube ready for mineral identification. The pipeline was written across six Python scripts (`step0_load.py` through `step5_validate.py`) and orchestrated by `run_pipeline.py`.

**10 scenes** were targeted across different lunar terrains. **7 scenes** ultimately passed Phase 1 and were declared ready for mineral identification.

---

## The 10 Scenes Processed

| Scene | File Name | Mode | Bands |
|---|---|---|---|
| Aristarchus Plateau | M3G20090209T054031_V01_RFL | Global | 85 |
| Bullialdus Crater | M3T20090417T021417_V01_RFL | **Target** | **256** |
| Copernicus Crater | M3G20090207T083836_V01_RFL | Global | 85 |
| Highlands near Tycho | M3G20090416T081001_V01_RFL | Global | 85 |
| Jackson Crater | M3G20081225T074420_V01_RFL | Global | 85 |
| Lunar Highlands South | M3G20090205T153710_V01_RFL | Global | 85 |
| Mare Imbrium | M3G20090609T230815_V01_RFL | Global | 85 |
| Mare Serenitatis | M3G20090204T113444_V01_RFL | Global | 85 |
| Mare Tranquillitatis | M3G20090203T160452_V01_RFL | Global | 85 |
| South Pole Region | M3G20090610T153615_V01_RFL | Global | 85 |

> [!NOTE]
> M3 operates in two observing modes. **Global mode** = 85 spectral bands, 460–2976 nm. **Target mode** = 256 spectral bands with finer resolution. The pipeline handles both, but multi-scene ML training is only done for Global-mode scenes (85 bands) to avoid mixing incompatible band dimensions.

---

## Pipeline Architecture: Three Phases

```
Phase A (Physics)              Phase B (ML Pool)              Phase C (ML + Validation)
──────────────────             ──────────────────             ──────────────────────────
Step 0: Load raw cube  ──►     Collect 1,500 spectra     ──►  Step 4: DAE + CNN denoise
Step 1: EDA (band stats)       from each Global scene          Step 5: 5-metric validation
Step 2: Noise detection        → pool combined, shuffled
Step 3: Physics corrections    → saved to models/pool.npz
```

Each scene goes through Phase A independently. Phase B builds one global pool from all Global-mode scenes. Phase C applies ML denoising and validates each scene.

---

## STEP 0 — Raw Data Loading (`step0_load.py`)

### What this step does

This is the data ingest step. It reads the raw M3 Level-2 reflectance data from disk and converts it into a clean 3D NumPy array in memory.

### Files loaded from each scene directory

Each scene directory contains three files:
1. **`*_RFL.IMG`** — The actual binary hyperspectral data, stored as float32 values
2. **`*_L2.LBL`** — A PDS3 (Planetary Data System) label file containing image dimensions, storage order, and the invalid data constant value
3. **`*_RFL.HDR`** — An ENVI header file containing the exact wavelength centers for each band

### Step-by-step loading process

**Step 0.1 — Parse the PDS3 label file:**
- Reads the `.LBL` text file using regex patterns
- Extracts: `H` (number of lines/rows), `W` (number of samples/columns), `B` (number of bands)
- Extracts `BAND_STORAGE_TYPE` — this tells us how bands are arranged in the binary file. Three possible formats: `BIL` (Band Interleaved by Line), `BSQ` (Band Sequential), `BIP` (Band Interleaved by Pixel)
- Extracts `INVALID_CONSTANT` — the sentinel value used to mark missing data (typically `-999.0`)
- Also reads `RECORD_BYTES` and `FILE_RECORDS` for size validation

**Step 0.2 — Parse wavelength table from HDR:**
- Reads the ENVI `.HDR` file and finds the `wavelength = { ... }` block
- Extracts all values as a NumPy float64 array
- Wavelengths are stored in nanometers (nm) in the HDR file
- Converts to micrometers (µm) by dividing by 1000 before using

**Step 0.3 — Load binary IMG file:**
- Reads entire binary file using `np.fromfile(dtype=np.float32)`
- Checks if total element count matches `H × W × B` exactly
- Handles 512-byte or 1024-byte file header offsets if needed
- If dimensions are unknown or mismatched, tries to infer by testing candidate dimension combinations: B ∈ {85, 260, 84, 86} and W ∈ {304, 320, 608, 640, 512, 256}

**Step 0.4 — Auto-detect storage order:**
- The label says `BIL` but sometimes the actual data is in a different order
- Tests all three formats (BSQ, BIL, BIP) by reshaping the data and computing SNR on a 200×200 central patch at 5 key wavelengths (0.75, 1.0, 1.5, 2.0, 2.5 µm)
- The format producing the highest mean SNR is the real storage order
- **Safety rule:** If the label-stated order produces SNR within 80% of the best score, uses the label order to avoid false reordering
- Parameters: test targets = [0.75, 1.0, 1.5, 2.0, 2.5] µm; SNR fallback threshold = 0.8×best

**Step 0.5 — Reshape to (H, W, B):**
- All three formats are reshaped to produce a standard `(H, W, B)` NumPy float32 array
  - BSQ: `data.reshape(B, H, W).transpose(1, 2, 0)`
  - BIL: `data.reshape(H, B, W).transpose(0, 2, 1)`
  - BIP: `data.reshape(H, W, B)` (already correct)

**Step 0.6 — Clean invalid values:**
- Values ≥ 65534 → set to NaN (detector saturation)
- Values within ±0.5 of `invalid_constant` (-999.0) → set to NaN
- Values > 1.50 (above physical max reflectance) → set to NaN
- Values < -0.10 (below physical min) → set to NaN
- **Thresholds used:** `M3_IF_MIN = -0.10`, `M3_IF_MAX = 1.50`
- Result: NaN-masked float32 cube

**Step 0.7 — Wavelength assignment:**
- Priority: use HDR wavelengths if available and length matches B
- Fallback 1: use hard-coded 85-band canonical wavelength table (460.99 nm to 2976.20 nm)
- Fallback 2: linear interpolation between 0.460 and 2.976 µm
- Assertion: wavelength maximum must exceed 2.0 µm (sanity check that units are correct)

**Step 0.8 — SNR sanity check:**
- Computes quick SNR on central 200×200 patch at five target wavelengths
- For each target: extracts band slice → filters finite/positive values → computes μ/σ
- If mean SNR < 5.0 → prints a WARNING (possible wrong storage order, radiance vs reflectance, shadow-dominated scene)
- This is a diagnostic only — does not stop the pipeline

**Step 0.9 — Output directory setup:**
- Creates `Pipeline_outputs/` folder inside each location directory (one level above the scene directory)
- Creates five subdirectories: `01_eda/`, `02_noise/`, `03_physics/`, `04_denoised/`, `05_validation/`
- Saves directory paths in the `DIRS` dictionary passed to all downstream steps

### Outputs from Step 0
- `cube` → `np.ndarray float32 (H, W, B)` — the cleaned raw reflectance cube
- `waves` → `np.ndarray float64 (B,)` — wavelength centers in µm
- `fname` → string — base name, e.g., `M3G20090203T160452_V01_RFL`
- `meta` → dict — label metadata (H, W, B, storage order, invalid constant)
- `DIRS` → dict — output directory paths

---

## STEP 1 — Exploratory Data Analysis (`step1_eda.py`)

### What this step does

This step studies the raw cube band by band. It classifies every band as valid, dead, thermal, suspicious, or low-signal. It computes SNR per band. It produces the "valid pixel mask" that is used by every subsequent step to know which pixels have enough data to be worth processing.

### Thresholds used in this step

| Parameter | Value | Meaning |
|---|---|---|
| `DEAD_VAR_THRESHOLD` | 1×10⁻¹⁰ | A band with variance below this is considered dead (no signal variation) |
| `DEAD_NAN_THRESHOLD` | 0.95 | A band with >95% NaN pixels is considered dead |
| `THERMAL_ONSET_UM` | 2.50 µm | Bands beyond this wavelength begin to receive thermal emission from the lunar surface |
| `MIN_VALID_BANDS` | 20 | A pixel must have at least 20 finite valid-band values to be included in processing |
| `suspicious_std_threshold` | 95th percentile of per-band standard deviations | Bands with std above this threshold are flagged as suspicious |

### Step-by-step EDA process

**Step 1.1 — Per-band statistics:**
For every band `b` (0 to B-1):
- Extract all pixel values as a 1D array: `cube[:, :, b].ravel()`
- Keep only finite values
- Compute: mean, std, variance, min, max, NaN fraction (NaN count / total pixels), valid percentage

**Step 1.2 — Band classification:**
Every band is classified into exactly one category:
- **Dead band:** `variance < 1×10⁻¹⁰` OR `NaN fraction > 0.95`
- **Thermal band:** Band is valid AND wavelength > 2.50 µm
- **Low signal band:** Valid band where `|mean| < 0.001` (almost no reflectance signal)
- **Suspicious band:** Valid band where `std > 95th percentile of all valid-band stds`
- **Valid band:** All remaining valid bands (not dead, not thermal, not otherwise flagged)

**Step 1.3 — SNR computation (Local Spatial Variance method):**
For each valid band `b`:
- Compute horizontal differences: `np.diff(cube[:, :, b], axis=1)` → this gives the pixel-to-pixel variation in each row
- Compute vertical differences: `np.diff(cube[:, :, b], axis=0)` → pixel-to-pixel variation in each column
- Filter to finite values only
- `noise_h = std(diff_horizontal) / √2` (the √2 factor corrects for the differencing doubling the noise)
- `noise_v = std(diff_vertical) / √2`
- `noise = (noise_h + noise_v) / 2` — average of horizontal and vertical noise estimates
- `signal = mean(finite pixel values in this band)`
- `SNR[b] = signal / noise`
- Why this method: it estimates noise from local spatial variation rather than from temporal averaging, which is not available for single-image data. The differencing removes real spatial structure (geology) and leaves mainly noise.

**Step 1.4 — Valid pixel mask:**
- For every pixel `(r, c)`: count how many bands have finite (non-NaN) values
- `valid_pixel_mask[r, c] = True` if finite count ≥ `MIN_VALID_BANDS` (20)
- This mask is passed to all downstream steps

**Step 1.5 — Sample spectra:**
- Randomly samples 5 pixel locations from valid pixels (seed=42 for reproducibility)
- Used for plotting only

**Step 1.6 — EDA JSON output:**
Saves `{fname}_eda.json` to `01_eda/` containing:
- Cube shape, wavelengths, per-band statistics
- Dead bands list, thermal bands list, valid bands list, suspicious bands list
- Per-band SNR array
- Valid pixel count and fraction
- All threshold values used

**Step 1.7 — Two diagnostic figures:**
1. **Band Statistics figure** (6 panels): mean reflectance ±1σ, SNR bar chart (color coded green >30 / orange >10 / red <10), NaN fraction bar chart, per-band std plot, valid pixel coverage, 5 sample spectra
2. **Spatial Overview figure** (3 panels): false-color RGB (using 0.75 / 0.90 / 1.05 µm), single-band image at 1.0 µm, valid-band-count map

### Key points about EDA output
- The `valid_bands` list (all bands that are not dead) is passed to every downstream step — physics corrections, noise removal, and ML training all operate only on valid bands
- Dead bands are kept as NaN in the cube — they are never modified or interpolated
- SNR values computed here (called `bs["snr"]`) are the pre-physics-correction baseline

---

## STEP 2 — Noise Identification (`step2_noise.py`)

### What this step does

This step characterises every type of noise present in the raw data. It does not fix anything — it only identifies and quantifies. The outputs (which bands have which noise types, how severe) are used by Step 3 to apply the right correction to each band.

### Five noise types investigated

#### N1 — Gaussian (Additive) Noise
- **How detected:** Uses the per-band SNR computed in Step 1
- **Thresholds:**
  - Severe Gaussian: `SNR < 10.0` (high noise bands)
  - Moderate Gaussian: `10.0 ≤ SNR < 30.0`
- **Output:** Lists of severely and moderately noisy band indices
- **Physical meaning:** Random photon-counting and readout noise. Uniformly distributed across all pixels.

#### N2 — Stripe Noise (Column Offsets)
- **How detected:**
  - For each valid band: compute the column mean vector `col_means[c] = mean(cube[:, c, b])` (mean of all rows for each column)
  - Compute global mean: `global_mean = mean(col_means)`
  - Per-column offset: `offset[c] = col_means[c] - global_mean`
  - Stripe severity for this band: `std(offsets)` — how much columns deviate from the global mean
  - Threshold: `STRIPE_STD_FACTOR × median(stripe_severity across all valid bands)` = `1.5 × median severity`
  - A band is "striped" if its severity exceeds this adaptive threshold
- **Output:** List of striped band indices, per-column offset arrays for each striped band
- **Physical meaning:** M3's detector is a push-broom sensor — each column of pixels is measured by a different detector element. These detector elements have slightly different gain and offset, causing column-to-column brightness variations unrelated to the real scene.

#### N3 — Spectral Spikes
- **How detected:**
  - Samples up to 20% of valid pixels or 500,000 pixels (whichever is larger), but capped at total valid count
  - For each sampled pixel: compute spectral median and MAD (Median Absolute Deviation, scaled to normal distribution: ×1.4826)
  - For each valid band: if `|spec[b] - median| > 5.0 × MAD` → this band is a spike in this pixel
  - Counts total spikes per band across all sampled pixels
  - `spike_fraction[b] = spike_count[b] / n_sampled`
  - A band is "spiked" if `spike_fraction > 0.002` (0.2% of sampled pixels show a spike there)
- **Threshold parameters:** `MAD_SPIKE_FACTOR = 5.0`, spike fraction threshold = 0.002
- **Output:** Per-band spike fraction array, list of spiked band indices
- **Physical meaning:** Cosmic rays hitting the detector, electronic glitches, or sensor saturation cause single-pixel spectral anomalies that appear as extreme outliers in one or a few bands.

#### N4 — Thermal Emission
- **How detected:**
  - Extracts pixel values in the thermal window: bands between 2.5 µm and 2.8 µm
  - Samples 3,000 valid pixels (seed=42)
  - For each sampled pixel: compute `mean(diff(thermal_values))` — does reflectance systematically rise toward longer wavelengths?
  - `thermal_detected = True` if `mean_gradient > 0.001` OR `fraction_rising > 60%`
- **Threshold parameters:** `THERMAL_GRAD_THR = 0.001`, rising fraction threshold = 0.60
- **Output:** Boolean flag (detected or not), mean gradient value, fraction of rising pixels, list of thermal band indices
- **Physical meaning:** During daytime observations, the lunar surface re-emits heat as thermal infrared radiation. At M3 wavelengths above ~2.5 µm, this emission adds a positive flux to the measured reflectance that is not from reflected sunlight.

#### N5 — Spectral Smile
- **How detected:**
  - Focuses on the pyroxene 1-µm absorption region: bands between 0.85 and 1.15 µm
  - For each column `c`: computes the mean spectrum across all rows (`nanmean(cube[:, c, smile_bands], axis=0)`)
  - Finds the wavelength of minimum reflectance in this column's mean spectrum (this should be constant across columns if there is no smile)
  - Records this center wavelength position for each column
  - `smile_detected = True` if `max(center_positions) - min(center_positions) > 2.0 nm`
- **Threshold:** `SMILE_SHIFT_THR_NM = 2.0` nm
- **Output:** Boolean flag, position range in nm, per-column shift profile array
- **Physical meaning:** A push-broom spectrometer focuses light through a slit onto a 2D detector. If the optics are not perfectly aligned, the central wavelength of each band drifts across the image width — producing a "smile" distortion that makes the same absorption appear at slightly different wavelengths in different columns.

### Noise Map JSON Output
Saves `{fname}_noise_map.json` to `02_noise/` containing:
- All noise detection results (band lists, severity arrays, boolean flags)
- All threshold values used
- Per-band spike fraction array
- Per-column stripe offset arrays
- Per-column smile shift profile

---

## STEP 3 — Physics-Based Corrections (`step3_physics.py`)

### What this step does

This step applies four deterministic, physics-grounded corrections to produce the "physics-corrected cube." It uses the noise characterisation from Step 2 to decide which corrections to apply. A fifth sub-step ("mineral band guard") protects scientifically important spectral features from being accidentally damaged by the corrections.

### BLOCK_SIZE = 64 rows

For memory efficiency, all pixel-level corrections are applied in blocks of 64 rows at a time.

### C1 — Spike Removal

**When applied:** For every band where `spike_fraction[b] > 0.001` (1 in 1000 pixels is spiked)

**Detection (spectral):**
- For each band `b`, compare each pixel's value to the median of its spectral neighbors at `b-2, b-1, b+1, b+2`
- `difference = pixel_value[b] - median(neighbors)`
- Compute MAD of all finite differences: `MAD = median(|diff - median(diff)|) × 1.4826`
- A pixel is a spike if: `finite` AND `|difference| > 5 × MAD`
- **Threshold: 5 MAD**

**Replacement (spatial):**
- Replace spike pixels using the mean of their 4 spatial neighbors (up, down, left, right) in the **same band**
- Uses reflect-padding at image edges
- NaN neighbors are ignored (nanmean)
- This preserves spatial smoothness — the replaced value looks like the surrounding terrain

**Why spatial replacement instead of spectral interpolation:** The SNR metric used in validation computes noise from spatial differences (`np.diff`). If spikes were replaced with spectral interpolates, they would look smooth spectrally but might still be spatially anomalous, fooling the SNR metric. Spatial replacement ensures the repaired pixel looks correct to both spectral and spatial analysis.

### C2 — Thermal Emission Correction

**When applied:** Only if `thermal_detected = True` from Step 2

**Method:** Linear regression extrapolation per pixel
- Pre-thermal reference bands: all valid bands between 2.0 and 2.5 µm
- Thermal correction window: all valid bands from 2.5 µm to the last band
- For each pixel in a 64-row block:
  - Fit a linear trend through the pre-thermal reference values (least squares, fully vectorized)
  - For each thermal band: compute what the reflectance **would** be if the trend from 2.0–2.5 µm continued linearly (the "expected" value)
  - Compute `excess = measured - expected`
  - If `excess > 0` (measured exceeds the extrapolation → thermal emission detected at this pixel): subtract the excess, clip to physical range [0, 1.50]
  - If `excess ≤ 0`: leave the pixel unchanged (no thermal contribution visible here)
- **Physical reasoning:** Below 2.5 µm, reflected solar light dominates. The trend in that region represents the solar continuum slope. Thermal emission only adds positive flux on top of this. By extrapolating the solar trend and subtracting the difference, we remove the thermal component.

### C3 — Column-Offset Destripe (FFT Destripe)

**When applied:** Only for bands in the `striped_bands` list from Step 2

**Method:** Per-band column DC offset subtraction
1. For each striped band, compute the column median vector: `col_medians[c] = median(cube[:, c, b], ignoring NaN)`
2. Compute the scene-wide median of all column medians: `scene_med = median(col_medians)`
3. Column offset = `col_medians[c] - scene_med`
4. Identify which columns have offsets that are "significant outliers":
   - MAD of all column offsets: `mad = median(|col_offsets|)`
   - Significant if `|col_offset[c]| > STRIPE_SENSITIVITY × mad` where `STRIPE_SENSITIVITY = 3.0` sigma
5. For significant columns only: subtract `col_offset[c]` from all pixels in that column
6. **Safety check:** Before committing the correction, verify that the band mean did not shift by more than `MEAN_SHIFT_TOLERANCE × original_mean = 2%`. If it did, the "stripe" was actually a real radiance gradient (e.g., shadow edge), not an instrument artifact — in that case, the correction is **discarded** for that band.
7. Clip to physical range [0, 1.50], restore NaN at invalid positions

**Why this approach is called "FFT Destripe" in comments but is actually column-offset subtraction:** The original plan used FFT to remove periodic stripe patterns, but column-offset subtraction was found to be more robust for M3's non-periodic detector drift. The name is kept in the code comments but the actual algorithm is the column-median approach.

### C4 — Spectral Smile Correction

**When applied:** Only if `smile_detected = True` from Step 2

**Method:** Per-column spectral resampling
1. Smooth the per-column shift profile from Step 2 using a 1D Gaussian filter with `sigma=3.0` columns — this prevents column-to-column correction discontinuities
2. For each column `c`:
   - `shift_um = smooth_shift[c] / 1000` (convert nm to µm)
   - If `|shift_um| < 0.001 µm` → skip (below correction threshold)
   - Otherwise: define "corrected wavelength grid" = `waves_um - shift_um` (shift the wavelength axis by the detected offset)
   - For each pixel in this column: interpolate the spectrum from the corrected grid back onto the nominal wavelength grid using `scipy.interpolate.interp1d` (linear interpolation, not cubic — linear prevents Runge oscillations in reflectance)
   - Clip to physical range, restore NaN at dead band positions

### C5 — Mineral Band Guard

**When applied:** Always after C1–C4

**Purpose:** The corrections above (especially destripe) might inadvertently alter the depths of key mineral absorption features. This guard checks for that and reverts any affected bands to their original (pre-correction) values.

**Mineral absorption features monitored:**

| Index | Center (µm) | Left shoulder | Right shoulder | Mineral |
|---|---|---|---|---|
| BD2300 | 2.295 | 2.12 | 2.53 | Phyllosilicates/Carbonates |
| BD1900 | 1.930 | 1.85 | 2.07 | Hydrated minerals |
| BD2210 | 2.21 | 2.165 | 2.25 | Al-OH clays |
| SINDEX2 | 2.10 | 1.815 | 2.20 | Sulfates |
| BD1500 | 1.50 | 1.43 | 1.815 | Water ice |
| HCPINDEX2 | 2.14 | 1.82 | 2.46 | HCP pyroxene |

**For each feature:**
1. Compute band depth map using raw (pre-correction) cube:
   - `BD = 1 - R(center) / [R(left) + frac × (R(right) - R(left))]`
   - where `frac = (center - left) / (right - left)`
2. Identify "ROI pixels" — pixels where the feature depth exceeds the 90th percentile of all pixels with `BD ≥ 0.02` and `BD ≤ 0.60`
3. Sample up to 6,000 ROI pixels
4. Compute band depth in the physics-corrected cube on the same sample
5. Compute `change_pct = (BD_corrected - BD_raw) / |BD_raw| × 100`
6. Compute Continuum-SAM between raw and corrected spectra in the feature window (measures if spectral shape changed in the continuum around the absorption)
7. If `|change_pct| > 20%` (BAND_DEPTH_CHG_THR) OR `Continuum-SAM > 2.0°` (CONTINUUM_SAM_THR_DEG): → the correction damaged this absorption feature
8. For damaged features: revert all bands in the window [left, right] back to raw values: `cube_phys[bands] = cube_raw[bands]`
9. This guarantees that the physics corrections never worsen the scientific spectral content

**Parameters used:** `MAX_GUARD_SAMPLE = 6000`, `ROI_BAND_DEPTH_THR = 0.02`, `ROI_TOP_PERCENTILE = 90`, `MIN_ROI_PIXELS = 100`, `BAND_DEPTH_CHG_THR = 0.20`, `CONTINUUM_SAM_THR_DEG = 2.0`

### After C1–C5: Compute post-physics SNR

Same local spatial variance method as Step 1, but applied to the corrected cube. Result stored as `SNR_AFTER_PHYSICS` array. This becomes the new baseline for ML denoising improvement comparison.

### Outputs from Step 3
- `{fname}_physics_corrected.npy` → shape (H, W, B), float32, saved to `03_physics/`
- `{fname}_physics_correction.png` → before/after spectral plots for 3 sample pixels
- `{fname}_snr_after_physics.png` → SNR comparison before/after physics
- `{fname}_physics_log.json` → corrections applied, spike count, thermal pixel count, striped bands processed, mineral guard bands, SNR before/after

---

## STEP 4 — ML-Based Denoising (`step4_denoise.py`)

### What this step does

This step uses a machine learning model (a Denoising Autoencoder, or DAE) to further clean the physics-corrected cube. It goes beyond what physics corrections can do by learning the statistical structure of clean M3 spectra and using that structure to suppress residual Gaussian noise.

### The Spectral Physics Metric (SPM) Indices

Before training, the step computes 8 SPM index maps from the physics cube. These characterize what mineralogy each pixel is likely to contain, and are used to build a **stratified training set** (see below).

The 8 SPM indices and parameters:

| Index | Center (µm) | Left shoulder | Right shoulder | Mineralogy |
|---|---|---|---|---|
| BD2300 | 2.295 | 2.12 | 2.53 | Phyllosilicates |
| BD1900 | 1.930 | 1.85 | 2.07 | Hydrated minerals |
| OLINDEX3 | 1.05 (reflectance) | — | — | Olivine proxy |
| BD2210 | 2.21 | 2.165 | 2.25 | Al-OH clays |
| SINDEX2 | 2.10 | 1.815 | 2.20 | Sulfates |
| BD1500 | 1.50 | 1.43 | 1.815 | Water ice |
| RPEAK1 | 0.60–0.84 peak | — | — | Iron-bearing minerals |
| HCPINDEX2 | 2.14 | 1.82 | 2.46 | HCP pyroxene |

### Stratified Training Data Collection

**Target: 15,000 training spectra total**

For each of the 8 SPM indices:
1. Find the top 20% of valid pixels by that index value (80th percentile threshold)
2. Sample up to 400 pixels from this "high-signal stratum"
3. These pixels are guaranteed to be scientifically interesting (likely showing a mineral absorption)

Fill the remaining training quota (`15,000 - collected`) with background pixels (those not in the top 20% of any index).

For each selected pixel: extract the spectrum, fill any NaN values with the band-wise mean across the full scene, keep only fully finite spectra.

**Why stratified sampling:** If we sampled randomly, the rare pixels showing mineral absorptions would be swamped by the vast majority of "featureless" background pixels. The DAE trained on random samples would learn to fit background spectra perfectly but would average out rare absorptions. Stratified sampling forces the DAE to see and preserve mineral signatures.

### Multi-Scene Pool (Phase B)

For Global-mode (85-band) scenes:
- Each scene contributes 1,500 spectra to a shared pool
- All pools are combined and shuffled (seed=42)
- The global DAE is trained on this multi-scene pool **once** and saved to `models/m3_dae_B85.pth`
- When processing subsequent scenes, the saved model is loaded (no retraining)
- Target-mode scenes (256 bands) cannot use the global pool — they train a **local DAE** from their own spectra only

### DAE Architecture

**Model type:** 1D Convolutional Denoising Autoencoder

```
Input: (batch, B)  — one spectrum per sample
  ↓
Reshape: (batch, 1, B)  — one "channel" along wavelength axis
  ↓
Conv1d(1 → 32, kernel=5, padding=2) → BatchNorm1d(32) → ReLU
Conv1d(32 → 64, kernel=5, padding=2) → BatchNorm1d(64) → ReLU
Conv1d(64 → 32, kernel=5, padding=2) → BatchNorm1d(32) → ReLU
Conv1d(32 → 1, kernel=5, padding=2)
  ↓
Squeeze: (batch, B)  — output is the denoised spectrum
```

- kernel=5 with padding=2 means each output value depends on 5 adjacent spectral values — the model sees local spectral context
- BatchNorm improves training stability and acts as implicit regularization
- No fully connected layers — purely convolutional, preserving spatial (wavelength) structure
- **This is a direct-prediction architecture:** the model outputs the estimated clean spectrum (not the residual noise)

### Training Method: Noise2Void (N2V)

The DAE is trained using the **Noise2Void** principle. This is important: we do **not** have clean reference spectra to supervise the training. We only have noisy spectra. N2V works around this by:

1. For each spectrum in a batch, randomly mask 10% of valid bands (`MASK_FRACTION = 0.10`)
2. Replace each masked band with the value of a randomly chosen neighboring band (offset ±1 or ±2 positions)
3. Feed this masked spectrum to the DAE
4. The DAE predicts the full clean spectrum
5. **Loss is computed only at the masked positions:** MSE between `predicted[masked_bands]` and `original[masked_bands]`

**Why this works:** The masked band's value was replaced with a neighbor, so the model cannot simply copy what it was given. It must predict the masked values from the remaining unmasked context — forcing it to learn the smooth spectral structure of clean M3 spectra. Noise at any given band is uncorrelated with noise at other bands, so neighbors provide clean context.

### DAE Loss Function

```
total_loss = recon_loss + LAMBDA_SMOOTH × smooth_loss + LAMBDA_SAM × sam_loss
```

Where:
- `recon_loss` = MSE at masked positions only (the N2V objective)
- `smooth_loss` = `mean((diff(predicted, dim=1))^2)` — penalizes non-smooth predictions; `LAMBDA_SMOOTH = 0.00` (disabled in final configuration)
- `sam_loss` = spectral angle between predicted output and input spectrum; `LAMBDA_SAM = 0.005`

**Why SAM loss:** The spectral angle between a noisy and clean M3 spectrum is very small (Gaussian noise is spectrally unselective). Minimizing SAM prevents the DAE from changing the spectral shape (which would destroy mineral absorption features) while still allowing it to smooth amplitude noise.

### DAE Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 80 (with early stopping, patience=10) |
| Batch size | 512 |
| Learning rate | 1×10⁻³ |
| Optimizer | AdamW |
| Weight decay | 1×10⁻⁴ |
| LR scheduler | Cosine annealing over 80 epochs |
| Train/val split | 90%/10% |
| AMP | Enabled when CUDA available |
| Gradient clipping | max norm = 1.0 |

Early stopping: if validation loss does not improve for 10 consecutive epochs, training stops and the best checkpoint is restored.

### DAE Inference (Pixel-by-Pixel Denoising)

After training:
1. All valid pixels' spectra are extracted from the physics cube
2. NaN values (dead bands) are filled with the band-wise mean before passing to the model
3. Processed in batches of 4,096 pixels (`INFER_BATCH = 4096`)
4. If GPU runs out of memory (OOM), batch size is automatically halved until it fits
5. Model outputs are stored as `cube_dae`

### Adaptive Blending

The final denoised spectrum is a **blend** of the DAE output and the original physics spectrum:

```
denoised[pixel] = α × cube_dae[pixel] + (1-α) × cube_phys[pixel]
```

where `α` is the blend weight for this pixel, computed as:
- **Base weight from SAM:** For a sample of 12,000 pixels, compute SAM between DAE output and physics spectrum. If `mean_SAM < 0.035 rad` (BLEND_SAM_THR_RAD) → SAM is acceptably small → use higher DAE weight
- **Adjustment from Pearson correlation:** If `mean_Pearson > 0.995` (BLEND_PEARSON_THR) → the DAE output is highly correlated with the original → safe to apply more DAE → use higher weight

The final weight is clamped to: `[BLEND_MIN, BLEND_MAX] = [0.15, 0.80]`

**Key decision:** `BLEND_MAX = 0.80` (reduced from 0.95). This was intentionally lowered to prevent the DAE from dominating the output so heavily that narrow absorption features (which the DAE might average over) get erased.

For pixels where SAM or Pearson indicates poor DAE quality: use `BLEND_MIN = 0.15` — the physics cube contributes 85% of the output.

Additionally, a **spatial Gaussian blend** is applied: pixels with very high local spectral roughness (noisy neighborhoods) receive slightly more DAE weight; smooth regions receive slightly more physics weight. Parameters: `GAUSSIAN_SPATIAL_BLEND = 0.25`, `GAUSSIAN_SPATIAL_SIGMA = 0.75`.

### Mineral Band Guard (Post-Denoising)

Same logic as the mineral band guard in Step 3, applied again to verify that DAE denoising did not damage absorption depths. Parameters: same as Step 3. If damage detected, the affected spectral window reverts to the physics-corrected values.

### Outputs from Step 4
- `{fname}_denoised.npy` → shape (H, W, B), float32, saved to `04_denoised/`
- `{fname}_training_curves.png` → DAE loss vs epoch
- `{fname}_snr_comparison.png` → SNR: raw vs physics vs ML
- `{fname}_blend_weights.png` → spatial map of blend weights used
- `{fname}_spectral_comparison.png` → before/after spectra for 3 sample pixels
- `models/m3_dae_B85.pth` → trained DAE model
- `models/m3_dae_B{B}.pth` (for Target-mode scenes with B=256)

---

## STEP 5 — Validation (`step5_validate.py`)

### What this step does

This step quantitatively verifies that the denoised cube is better than the raw cube and has not lost scientific information. It uses 5 independent metrics. The overall verdict (PASS/REVIEW/FAIL) determines whether a scene is declared "ready for mineral identification."

### The 5 Validation Metrics

#### M1 — SNR Improvement
- **What is measured:** For every valid band, compute `improvement_pct = (SNR_ML - SNR_raw) / SNR_raw × 100`
- **Band-level pass:** A band passes if `SNR_ML ≥ SNR_raw`
- **Overall pass condition for M1:** Mean improvement ≥ 0% AND mean SNR_ML ≥ 10.0
- **Reports:** mean/max/min improvement %, bands passing, mean SNR at each stage (raw / physics / ML)

#### M2 — Spectral Angle Mapper (SAM)
- **What is measured:** For 15,000 randomly sampled valid pixels, compute the spectral angle between the raw spectrum and the denoised spectrum
  - `cos(angle) = dot(raw, denoised) / (||raw|| × ||denoised||)`
  - angle in radians
- **Why SAM instead of MSE:** SAM is scale-invariant. It measures pure shape change, ignoring amplitude scaling. This is the right metric for reflectance spectra where absolute brightness varies (illumination angle, albedo) but spectral shape is what carries mineral information.
- **Thresholds:**
  - PASS: angle < 0.05 radians (< ~2.9°)
  - REVIEW: 0.05 ≤ angle < 0.10 radians
  - FAIL: angle ≥ 0.10 radians (≥ ~5.7°)
- **Reports:** Mean, median, 95th percentile angle, percentage in each category

#### M3 — Band Depth Preservation
- **What is measured:** For 2,000 randomly sampled pixels, compute the band depth for each of the 6 SPM mineral absorption features in both the raw cube and the denoised cube
  - `BD = 1 - R(center) / [R(left) + frac × (R(right) - R(left))]`
- **If the feature is not present (`BD_raw < 0.015`):** marked as N/A — cannot meaningfully compute preservation for a feature that doesn't exist in this scene
- **If the feature is present:** compute `change_pct = (BD_denoised - BD_raw) / |BD_raw| × 100`
- **Thresholds:**
  - PASS: |change_pct| ≤ 20%
  - REVIEW: 20% < |change_pct| ≤ 30%
  - FAIL: |change_pct| > 30%

#### M4 — Pearson Correlation
- **What is measured:** For 2,000 randomly sampled pixels, compute Pearson `r` between raw and denoised spectra
- **Threshold:** PASS if fraction of pixels with r ≥ 0.98 is high
- **Reports:** Mean r across pixels, fraction of pixels passing

#### M5 — Noise Taxonomy Suppression
- **What is measured:** For each detected noise type, measure how much the ML step reduced it
  - N1 Gaussian: compare band-wise std before/after; suppression % = `(std_before - std_after) / std_before × 100`
  - N2 Stripe: compare per-column offset severity; suppression % of column std reduction
  - N3 Spikes: re-run spike detection on denoised cube; compare spike fraction
  - N4 Thermal: compare mean spectral gradient in thermal bands
  - N5 Smile: if smile was detected and corrected: 100% suppression
- **This metric is informational only** — does not affect the overall PASS/FAIL verdict

### ROI Validation (Supplemental)

In addition to the 5 metrics, an ROI (Region of Interest) validation is done for features that are strongly present in the scene:
- Identifies "high-confidence ROI pixels" where the feature has BD in the top 10% (90th percentile threshold)
- Only uses these scientifically interesting pixels (not random background)
- Re-computes band depth change, SAM, and Pearson specifically on these high-signal pixels
- If an ROI feature FAILS band depth preservation → this is reported as a science-critical failure

### Band-Level Decisions

Every valid band receives a per-band verdict:
- PASS: `SNR_ML ≥ SNR_raw` AND `std(denoised) ≤ 1.05 × std(raw)` (no artifactual noise amplification)
- REVIEW: SNR passes but std check fails (slight spread increase)
- FAIL: SNR_ML < SNR_raw (denoising actually hurt this band)

`band_pass_pct = number of PASS bands / total valid bands × 100`

### Overall Verdict Computation

**PASS** requires ALL of:
- `mean_SNR_ML ≥ 10.0`
- `mean_improvement_pct ≥ 0%`
- `band_pass_pct ≥ 80%`
- `mean_SAM_rad < 0.10` AND `mean_Pearson ≥ 0.90`
- Zero M3 band depth FAILs
- Zero ROI band depth FAILs

If none of the above PASS conditions are met but some partial conditions hold → REVIEW

Otherwise → FAIL

### Scene Readiness Classification

After collecting all scene validation reports, `run_pipeline.py` computes readiness:

**READY** = overall_verdict is PASS AND no additional failure reasons

**BORDERLINE** = `mean_SNR_ML ≥ 8` AND `mean_improvement_pct ≥ 0` AND `mean_Pearson ≥ 0.95` AND `mean_SAM_rad < 0.10` AND zero M3 fails

**NOT_READY** = anything else (fails critical thresholds)

### Validation Outputs
- `{fname}_validation_report.json` → all 5 metric results, per-band decisions, ROI results, overall verdict
- 6 publication-ready figures (150 DPI): SNR comparison, SAM distribution map, band depth bar chart, Pearson distribution, noise suppression summary, spatial uncertainty map

---

## Per-Scene Results

### Phase 1 Outcome Table

| Scene | Mode | Raw SNR | ML SNR | Gain % | SAM (°) | Pearson | Band Pass % | Verdict | Readiness |
|---|---|---|---|---|---|---|---|---|---|
| **Bullialdus Crater** | Target (256b) | 28.39 | **29.04** | **+2.48%** | **2.007°** | **0.9937** | **97.17%** | PASS | ✅ READY |
| **Copernicus Crater** | Global (85b) | 21.64 | **21.89** | **+1.19%** | **1.617°** | **0.9971** | **97.59%** | PASS | ✅ READY |
| **Jackson Crater** | Global (85b) | 48.64 | **49.30** | **+1.42%** | **0.451°** | **0.9997** | **97.59%** | PASS | ✅ READY |
| **Lunar Highlands South** | Global (85b) | 17.46 | **17.59** | **+0.74%** | **0.670°** | **0.9993** | **97.59%** | PASS | ✅ READY |
| **Mare Imbrium** | Global (85b) | 49.52 | **50.58** | **+2.20%** | **1.092°** | **0.9988** | **97.59%** | PASS | ✅ READY |
| **Mare Serenitatis** | Global (85b) | 9.65 | **10.35** | **+7.27%** | **1.802°** | **0.9907** | **91.57%** | PASS | ✅ READY |
| **Mare Tranquillitatis** | Global (85b) | 24.27 | **24.55** | **+1.21%** | **1.235°** | **0.9983** | **96.39%** | PASS | ✅ READY |
| Aristarchus Plateau | — | — | — | — | — | — | — | — | ❌ Not in summary |
| Highlands near Tycho | — | — | — | — | — | — | — | — | ❌ Not in summary |
| South Pole Region | — | — | — | — | — | — | — | — | ❌ Not in summary |

> [!NOTE]
> Aristarchus Plateau, Highlands near Tycho, and South Pole Region do not appear in the final `pipeline_summary.json`. These scenes either failed during pipeline execution or were excluded from the final readiness manifest.

### Notable Observations Per Scene

**Bullialdus Crater (Target Mode):**
- Only Target-mode scene in the "READY" list
- Mean SAM = 2.007° (slightly higher than Global-mode scenes, reflecting local DAE constraints without multi-scene pool benefit)
- SNR gain of +2.48% is healthy

**Jackson Crater:**
- Exceptional Pearson = 0.9997 — almost perfect spectral fidelity
- SAM = 0.451° — best SAM among all scenes
- High raw SNR (48.64) → only modest absolute gain needed

**Mare Serenitatis:**
- Lowest raw SNR (9.65) — most challenging scene
- Largest SNR gain: +7.27% — DAE works hardest where it's most needed
- Band pass = 91.57% (slightly lower than others, because low-SNR bands are harder to improve consistently)

**Mare Imbrium:**
- Highest final SNR (50.58 ML)
- +2.20% gain despite starting from high baseline

---

## How We Concluded the Denoised Cube is Ready for Mineral Identification

The readiness decision is made using the `scene_readiness_manifest.json`. A scene is declared **READY** when it satisfies all of the following simultaneously:

1. ✅ **Overall verdict = PASS** from Step 5 validation
2. ✅ **Mean SNR_ML ≥ 10** — minimum signal quality for spectral feature detection
3. ✅ **Mean SNR improvement ≥ 0%** — denoising did not hurt SNR
4. ✅ **Band pass ≥ 80%** — most bands are individually verified improved
5. ✅ **Mean SAM < 0.10 rad** — spectral shape preserved within ±5.7°
6. ✅ **Mean Pearson ≥ 0.90** — high spectral fidelity between raw and denoised
7. ✅ **Zero M3 band depth fails** — no mineral absorption feature was damaged
8. ✅ **Zero ROI fails** — specifically in mineral-rich regions, features are preserved

All 7 reported scenes satisfied all 8 conditions. The `readiness` field in the manifest is `"READY"` for all 7. No scene is `"BORDERLINE"` or `"NOT_READY"` among the reported ones.

---

## Summary of All Parameters Used Across Phase 1

| Category | Parameter | Value |
|---|---|---|
| Data limits | `M3_IF_MIN / M3_IF_MAX` | -0.10 / 1.50 |
| Data limits | `THERMAL_ONSET_UM` | 2.50 µm |
| Data limits | `MIN_VALID_BANDS` | 20 |
| Band classification | `DEAD_VAR_THRESHOLD` | 1×10⁻¹⁰ |
| Band classification | `DEAD_NAN_THRESHOLD` | 0.95 |
| Band classification | `suspicious_std_threshold` | 95th percentile |
| SNR categorization | High noise threshold | SNR < 10 |
| SNR categorization | Moderate noise threshold | SNR 10–30 |
| Stripe detection | `STRIPE_STD_FACTOR` | 1.5 |
| Spike detection | `MAD_SPIKE_FACTOR` | 5.0 |
| Spike detection | spike_frac_threshold | 0.002 |
| Thermal detection | `THERMAL_GRAD_THR` | 0.001 |
| Smile detection | `SMILE_SHIFT_THR_NM` | 2.0 nm |
| Physics corrections | `BLOCK_SIZE` | 64 rows |
| Physics corrections | `STRIPE_SENSITIVITY` | 3.0 sigma |
| Physics corrections | `MEAN_SHIFT_TOLERANCE` | 0.02 (2%) |
| Physics corrections | `SMILE_MIN_SHIFT_UM` | 0.001 µm |
| Mineral guard | `BAND_DEPTH_CHG_THR` | 0.20 (20%) |
| Mineral guard | `CONTINUUM_SAM_THR_DEG` | 2.0° |
| Mineral guard | `ROI_TOP_PERCENTILE` | 90th |
| Mineral guard | `ROI_BAND_DEPTH_THR` | 0.02 |
| ML training | `DAE_EPOCHS` | 80 |
| ML training | `BATCH_SIZE` | 512 |
| ML training | `LR_DAE` | 1×10⁻³ |
| ML training | `WEIGHT_DECAY` | 1×10⁻⁴ |
| ML training | `MASK_FRACTION` | 0.10 (10%) |
| ML training | `LAMBDA_SAM` | 0.005 |
| ML training | `N_TRAIN_SPECTRA` | 15,000 |
| ML training | Per-scene pool contribution | 1,500 spectra |
| Blending | `BLEND_MIN / BLEND_MAX` | 0.15 / 0.80 |
| Blending | `BLEND_SAM_THR_RAD` | 0.035 rad |
| Blending | `BLEND_PEARSON_THR` | 0.995 |
| Validation | SAM PASS threshold | 0.05 rad |
| Validation | SAM REVIEW threshold | 0.10 rad |
| Validation | Pearson PASS threshold | 0.98 |
| Validation | Band depth PASS | ≤ 20% change |
| Validation | Band depth REVIEW | ≤ 30% change |
| Validation | SNR ready threshold | ≥ 10.0 |
| Readiness | SNR borderline threshold | ≥ 8.0 |
| Readiness | Band pass pct threshold | ≥ 80% |
