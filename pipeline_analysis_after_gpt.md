# M3 Pipeline — Deep Analysis & Mineral Readiness Evaluation
## Scenes Covered: 7 locations, 9 total strips

---

## 1. Pipeline Architecture — What Was Built

The pipeline is a **5-stage, 3-phase system** implemented across `step0_load.py` → `step5_validate.py`:

```
Phase A (Physics)           Phase B (ML Pool)          Phase C (ML + Validation)
Step 0: Load ENVI cube  →   Build multi-scene DAE   →   Step 4: DAE+CNN denoise
Step 1: EDA (SNR/bands)     training pool from all       Step 5: 5-metric validation
Step 2: Noise taxonomy      Global-mode scenes
Step 3: Physics corrections
```

### Phase A — Physics Corrections Applied Per Scene
| Correction | Method |
|---|---|
| **C1 Spike removal** | Spectral-neighbour detection (±2 bands), spatial-neighbour replacement |
| **C2 Thermal emission** | Linear extrapolation subtraction in 64-row blocks |
| **C3 FFT Destripe** | Applied only to bands flagged as striped; column-offset correction |
| **C4 Smile correction** | Only when smile detected; sub-pixel wavelength shift per column |
| **Mineral band guard** | Bands 138, 170, 202 protected from aggressive destripe |

### Phase B — Multi-Scene DAE Pool
- Training spectra collected from **all Global-mode scenes only** (B=85 bands)
- Target-mode scenes (B=256: Bullialdus, Mare Imbrium M3T strip) train a **local DAE** instead
- 1,500 spectra per scene → combined, shuffled, saved to `models/multi_scene_pool.npz`

### Phase C — ML Denoising (Step 4)
- **DAE**: 1D Conv Autoencoder (3 encoder layers: 1→32→64, decoder: 32→1, kernel=5, padding=2) with Noise2Void masked-band loss
- **CNN**: 1D Stripe CNN trained on synthetic stripe injection (separate model `m3_cnn_H*.pth`)
- **Blending**: Per-pixel adaptive blend weight based on SAM threshold (0.035 rad) and Pearson (0.995), BLEND_MIN=0.15, BLEND_MAX=0.95
- **Output**: `{fname}_denoised.npy` (same shape as physics cube)

### Step 5 — Validation Framework (5 Metrics)
| Metric | What it Measures | Threshold |
|---|---|---|
| **M1 SNR** | Per-band SNR improvement (ML vs raw) | ≥0% mean improvement |
| **M2 SAM** | Spectral angle raw→denoised per pixel | PASS <0.05 rad, REVIEW <0.10 rad |
| **M3 Band Depth** | Preservation of 6 SPM indices (global mean) | Change <20% |
| **M4 Pearson** | Pixel-level spectral correlation raw→denoised | >0.98 |
| **M5 Noise** | Suppression % for N1–N5 noise types | Reported, informational |

---

## 2. Per-Scene Results — Quantitative Metrics Table

> [!NOTE]
> All scenes produced `overall_verdict: PASS` **except Mare Imbrium (Target strip)** which is `REVIEW`.

### 2.1 SNR Performance

| Scene / Strip | Mode | Raw SNR | Physics SNR | ML SNR | **Gain %** | Bands Pass |
|---|---|---|---|---|---|---|
| Bullialdus Crater | Target (256b) | 28.39 | 29.07 | **29.09** | +2.64% | 229/247 |
| Copernicus Crater | Global (85b) | 21.64 | 21.76 | **21.99** | +1.75% | 73/83 |
| Jackson Crater | Global (85b) | 48.64 | 48.92 | **49.46** | +1.75% | 81/83 |
| Lunar Highlands S (strip 1) | Global (85b) | 17.46 | 17.52 | **17.57** | +0.65% | 80/83 |
| Lunar Highlands S (strip 2) | Global (85b) | 15.85 | 15.89 | **15.96** | +0.69% | 76/83 |
| Mare Imbrium (Global) | Global (85b) | 49.52 | 50.09 | **50.72** | +2.63% | 75/83 |
| **Mare Imbrium (Target)** | Target (256b) | 24.27 | 24.95 | **25.15** | **+3.92%** | 235/247 |
| Mare Serenitatis | Global (85b) | 9.65 | 9.70 | **10.70** | **+10.97%** | 77/83 |
| Mare Tranquillitatis | Global (85b) | 24.27 | 24.40 | **24.61** | +1.49% | 80/83 |

**Key observations:**
- ✅ Every strip shows **positive mean SNR gain** — no regression
- ✅ Mare Serenitatis has the lowest raw SNR (9.65) and shows the largest ML gain (+10.97%), demonstrating the pipeline's greatest value exactly where SNR is lowest
- ⚠️ Lunar Highlands South strips have very low SNR (15.85–17.46), making the small gain (+0.65–0.69%) understandable — the DAE is being conservative
- ✅ Jackson Crater and Mare Imbrium Global have high SNR (48–49) — the small relative gain reflects good baseline quality, not pipeline failure

---

### 2.2 Spectral Angle Mapper (SAM) — Shape Preservation

| Scene / Strip | Mean SAM (°) | % PASS (<0.05 rad) | % REVIEW | % FAIL |
|---|---|---|---|---|
| Bullialdus Crater | 2.66° | 84.37% | 9.69% | **5.95%** |
| Copernicus Crater | 1.86° | 88.57% | 11.31% | 0.12% |
| Jackson Crater | **0.48°** | **99.99%** | 0.01% | 0.00% |
| Lunar Highlands S (strip 1) | 0.69° | 99.43% | 0.55% | 0.03% |
| Lunar Highlands S (strip 2) | 0.86° | 97.09% | 2.83% | 0.08% |
| Mare Imbrium (Global) | 1.22° | 99.88% | 0.12% | 0.00% |
| **Mare Imbrium (Target)** | **3.22°** | 40.01% | **56.99%** | 3.00% |
| Mare Serenitatis | 2.10° | 91.59% | 4.98% | **3.43%** |
| Mare Tranquillitatis | 1.36° | 98.97% | 1.02% | 0.01% |

**Key observations:**
- ✅ Jackson Crater is near-perfect: 0.48° mean SAM, essentially zero FAIL pixels
- ✅ Lunar Highlands South, Mare Imbrium Global, and Mare Tranquillitatis are excellent
- ⚠️ **Mare Imbrium Target strip** (3.22° mean, 57% REVIEW) is the major concern — the Target-mode local DAE is causing non-trivial spectral angle shifts across much of the scene
- ⚠️ **Bullialdus Crater** (2.66°, 5.95% FAIL) — Target-mode data with moderate SAM drift, suggesting the local DAE is also less tightly constrained
- ⚠️ **Mare Serenitatis** (2.10°, 3.43% FAIL) — the low raw SNR (9.65) makes the DAE work harder and introduces some angular distortion

---

### 2.3 Pearson Correlation (M4) — Spectral Fidelity

| Scene / Strip | Mean Pearson r | Pass Fraction (>0.98) |
|---|---|---|
| Bullialdus Crater | 0.9892 | 91.05% |
| Copernicus Crater | 0.9969 | 99.70% |
| Jackson Crater | **0.9996** | **100.00%** |
| Lunar Highlands S (strip 1) | 0.9992 | 99.95% |
| Lunar Highlands S (strip 2) | 0.9984 | 99.50% |
| Mare Imbrium (Global) | 0.9986 | 100.00% |
| **Mare Imbrium (Target)** | **0.9881** | **90.25%** |
| Mare Serenitatis | 0.9852 | 94.20% |
| Mare Tranquillitatis | 0.9982 | 99.95% |

- ✅ Most scenes are above 0.995, representing outstanding spectral fidelity
- ⚠️ Bullialdus (0.9892) and Mare Imbrium Target (0.9881) — still above the 0.98 threshold, but the lowest in the set
- ⚠️ Mare Serenitatis (0.9852) — just above threshold; the low-SNR scene pushes Pearson down

---

### 2.4 Band Depth (M3) — Global Mean Feature Preservation

| Scene | HCPINDEX2 (pyroxene) | BD2300 (phyllosilicates) | Other features |
|---|---|---|---|
| Bullialdus Crater | ✅ PASS (0.0%) | N/A (bd<0.015) | All N/A |
| Copernicus Crater | ⚠️ REVIEW (−23.4%) | ✅ PASS (−4.6%) | All N/A |
| Jackson Crater | N/A (bd<0.015) | N/A | All N/A |
| Lunar Highlands S (s1) | N/A | N/A | All N/A |
| Lunar Highlands S (s2) | N/A | N/A | All N/A |
| Mare Imbrium (Global) | ✅ PASS (+3.7%) | ✅ PASS (0.0%) | All N/A |
| Mare Imbrium (Target) | ✅ PASS (0.0%) | ✅ PASS (0.0%) | All N/A |
| Mare Serenitatis | N/A (0.0%) | N/A | All N/A |
| Mare Tranquillitatis | ✅ PASS (−2.8%) | N/A | All N/A |

> [!IMPORTANT]
> Most scenes report features as **N/A** (band depth < 0.015 at the **global scene mean**). This is expected — absorption features are spatially localised, not scene-wide. The ROI-based validation below is the more meaningful test.

---

### 2.5 ROI-Based Band Depth Validation — The Critical Mineral Preservation Test

This tests band depth preservation **only in pixels where the absorption was strongest** (top 90th percentile ROI):

| Scene | BD2300 (phyllosil.) | BD1900 (hydrated) | BD2210 (Al-OH) | SINDEX2 (sulfates) | BD1500 (ice) | HCPINDEX2 (pyrox.) |
|---|---|---|---|---|---|---|
| **Bullialdus** | ❌ −106% | ❌ −81.8% | ❌ −97.7% | ❌ −63.0% | ❌ −80.0% | ✅ 0.0% |
| **Copernicus** | ✅ −19.3% | ❌ −60.2% | ❌ −70.8% | ❌ −37.2% | ❌ −66.6% | ⚠️ −27.9% |
| **Jackson** | ❌ −49.0% | ❌ −59.7% | ❌ −77.1% | ❌ −37.7% | ❌ −45.4% | ❌ −34.3% |
| **Lun. Hld. S1** | ❌ −79.1% | ❌ −54.8% | ❌ −57.0% | ❌ −46.7% | ❌ −66.3% | ❌ −31.4% |
| **Lun. Hld. S2** | ⚠️ −24.5% | ❌ −64.3% | ❌ −89.8% | ❌ −45.5% | ❌ −73.3% | ⚠️ −20.3% |
| **Mare Imbrium G** | ✅ 0.0% | ❌ −67.2% | ❌ −58.1% | ✅ −19.7% | ❌ −70.4% | ✅ −2.9% |
| **Mare Imbrium T** | ✅ 0.0% | ❌ −72.7% | ❌ −87.6% | ❌ −74.3% | ❌ −79.4% | ✅ 0.0% |
| **Mare Serenitatis** | ✅ 0.0% | ❌ −73.6% | ❌ −95.3% | ❌ −90.4% | ❌ −45.4% | ✅ 0.0% |
| **Mare Tranquillitatis** | ✅ 0.0% | ❌ −58.3% | ❌ −67.7% | ❌ −37.6% | ❌ −61.0% | ✅ −1.0% |

> [!WARNING]
> **BD1900 (hydrated minerals) fails in every single scene.** BD2210 (Al-OH clays) also fails in nearly every scene with 57–97% attenuation. BD1500 (water ice) fails in every scene. SINDEX2 (sulfates) fails in most scenes. This is a **systematic over-smoothing problem in the narrow-band absorption features**.

---

### 2.6 Noise Suppression (M5)

| Scene | N1 Gaussian | N2 Stripe | N3 Spikes | N4 Thermal | N5 Smile |
|---|---|---|---|---|---|
| Bullialdus | N/A | **24.1%** | N/A | **31.5%** | **100%** |
| Copernicus | N/A | **30.6%** | N/A | **29.6%** | **100%** |
| Jackson | N/A | **28.3%** | N/A | **10.7%** | N/A |
| Lun. Hld. S1 | N/A | **38.9%** | N/A | **20.1%** | **100%** |
| Lun. Hld. S2 | N/A | **34.4%** | N/A | **22.5%** | **100%** |
| Mare Imbrium G | N/A | N/A | N/A | **12.8%** | **100%** |
| Mare Imbrium T | N/A | **54.6%** | N/A | **43.9%** | **100%** |
| **Mare Serenitatis** | **1.7%** | **32.3%** | **86.3%** | **23.1%** | **100%** |
| Mare Tranquillitatis | N/A | N/A | N/A | **26.6%** | N/A |

> [!NOTE]
> N5 (smile correction) hits **100%** wherever smile was detected — the physics correction fully eliminates the spectral smile artifact. N3 spike suppression in Mare Serenitatis is 86.3%, confirming the physics corrections are effective.

---

## 3. Band-Level Pass Rate

| Scene | PASS | FAIL | Pass % | Verdict |
|---|---|---|---|---|
| Bullialdus (256b) | 229 | 18 | **92.7%** | PASS |
| Copernicus (85b) | 73 | 10 | **88.0%** | PASS |
| Jackson (85b) | 81 | 2 | **97.6%** | PASS |
| Lun. Hld. S1 (85b) | 80 | 3 | **96.4%** | PASS |
| Lun. Hld. S2 (85b) | 76 | 7 | **91.6%** | PASS |
| Mare Imbrium G (85b) | 75 | 8 | **90.4%** | PASS |
| **Mare Imbrium T (256b)** | 235 | 12 | **95.1%** | ⚠️ **REVIEW** |
| Mare Serenitatis (85b) | 77 | 6 | **92.8%** | PASS |
| Mare Tranquillitatis (85b) | 80 | 3 | **96.4%** | PASS |

All scenes clear the 80% threshold easily. The REVIEW verdict for Mare Imbrium Target is driven by the SAM metric (40% PASS, 57% REVIEW), not band failures.

---

## 4. Critical Issues Identified

### Issue 1: Systematic Over-Smoothing of Narrow Absorption Features ⚠️ MAJOR

The most important finding: **ROI band depths drop by 40–106% after denoising for BD1900, BD2210, BD1500, and SINDEX2.** These are exactly the *narrow* (< 0.1 µm wide) features critical for distinguishing hydrated minerals, clays, and ice.

**Root cause analysis:**
- The DAE uses `kernel_size=5` with no explicit mineral band protection during inference
- The Noise2Void masked loss (`MASK_FRACTION=0.10`) smooths the spectrum at the masked positions
- `LAMBDA_SMOOTH=0.00` and `LAMBDA_SAM=0.00` — both spectral regularization penalties are **turned off**, so the only shape constraint is the reconstruction loss
- The blending strategy (`BLEND_PEARSON_THR=0.995`, `BLEND_SAM_THR=0.035 rad`) sets a very **high bar for trusting the raw spectrum**, meaning most pixels get heavy DAE output mixed in
- The MINERAL_BAND_DEFS are defined but appear to only be used in the **training stratification** (SPM indices), **not in per-band blending or loss masking during inference**

**This is the #1 blocker for mineral identification readiness.**

### Issue 2: Target-Mode Local DAE is Weaker ⚠️ MODERATE

Bullialdus (Target, 256b, SAM 2.66°, 5.95% FAIL) and Mare Imbrium Target (3.22°, 57% REVIEW) both show elevated SAM vs Global-mode scenes. The local DAE trained on a single scene with 15,000 spectra lacks the cross-scene diversity of the multi-scene pool. The large 256-band spectral space also makes the N2V task harder.

### Issue 3: Mare Serenitatis Very Low SNR (9.65) ⚠️ MODERATE

A mean SNR of 9.65 is borderline for mineral identification (threshold is 10). The +10.97% gain brings it to 10.70 — just above threshold. However, with Pearson at 0.9852 and SAM failures at 3.43%, this scene is the most fragile.

### Issue 4: pipeline_summary.json and scene_readiness_manifest.json Are Stale ⚠️ MINOR

Both files reflect only a **single Mare Serenitatis run with ERROR status** — they were not regenerated after the full multi-scene run. The manifest shows one scene as `NOT_READY` (missing validation report) — but that is the strip `M3G20090204T152951_V01_RFL-001` which exists in the denoised folder but has no validation report in `05_validation/`. This strip was processed but validation apparently failed or was not run.

---

## 5. Readiness Assessment for Mineral Identification

| Scene | Overall Verdict | SAM OK? | ROI Band Depth OK? | **Mineral Ready?** | Notes |
|---|---|---|---|---|---|
| **Jackson Crater** | PASS | ✅ | ⚠️ partial | 🟡 PARTIAL | Excellent SNR/SAM/Pearson; ROI fails but SAM so tight that the shape is good — pyroxene identification safe, hydrated minerals borderline |
| **Mare Imbrium G** | PASS | ✅ | ⚠️ partial | 🟡 PARTIAL | BD2300, SINDEX2, HCPINDEX2 pass; BD1900, BD2210, BD1500 fail |
| **Mare Tranquillitatis** | PASS | ✅ | ⚠️ partial | 🟡 PARTIAL | Only HCPINDEX2 active globally; ROI BD1900/BD2210 fail |
| **Lunar Highlands S1** | PASS | ✅ | ❌ | 🔴 NOT READY | All ROI features fail; very low SNR limits mineral detection |
| **Lunar Highlands S2** | PASS | ✅ | ⚠️ partial | 🟡 PARTIAL | BD2300 marginal (REVIEW), rest fail |
| **Copernicus Crater** | PASS | ✅ | ⚠️ partial | 🟡 PARTIAL | BD2300 passes; HCPINDEX2 review; others fail |
| **Bullialdus Crater** | PASS | ⚠️ | ❌ | 🔴 NOT READY | SAM failures + all ROI mineral bands severely attenuated (−80 to −106%) |
| **Mare Imbrium T** | ⚠️ REVIEW | ❌ | ⚠️ partial | 🔴 NOT READY | 57% REVIEW SAM; Target-mode DAE insufficient |
| **Mare Serenitatis** | PASS | ⚠️ | ⚠️ partial | 🟡 PARTIAL | SNR just above threshold; BD2300 and HCPINDEX2 pass |

### Summary
| Readiness | Scenes |
|---|---|
| 🟢 Ready (all key features preserved) | **None** |
| 🟡 Partial (broad features safe, narrow features degraded) | Jackson, Copernicus, Mare Imbrium G, Serenitatis, Tranquillitatis, Lun. Hld. S2 |
| 🔴 Not Ready | Bullialdus, Mare Imbrium T, Lunar Highlands S1 |

---

## 6. Is Noise Reduced While Preserving Mineral Absorption Features?

### Noise Reduction: ✅ YES
- SNR improves in every scene (positive mean gain in all 9 strips)
- Smile correction: 100% effective where applicable
- Stripe suppression: 24–55% across affected scenes
- Thermal correction: 11–44% across all scenes
- Spike removal: 86.3% in Mare Serenitatis (the only scene with spikes)

### Broad-band Feature Preservation: ✅ GOOD
- Pearson correlation >0.985 in all scenes
- HCPINDEX2 (pyroxene, broad ~1 µm) and BD2300 (phyllosilicates, 2.3 µm) are largely preserved
- Global-mode scenes with high SNR (Jackson, Mare Imbrium G) show excellent SAM

### Narrow-band Absorption Feature Preservation: ❌ PROBLEMATIC
- BD1900 (hydrated minerals): **fails in all 9 strips**, attenuation 45–82%
- BD2210 (Al-OH clays): **fails in 8/9 strips**, attenuation 57–97%
- BD1500 (water ice): **fails in all 9 strips**, attenuation 45–80%
- SINDEX2 (sulfates): fails in 8/9 strips, attenuation 37–90%

> [!CAUTION]
> The pipeline is **not safe** for direct mineral identification of hydrated minerals, Al-OH clays, water ice, or sulfates without further investigation or tuning. The narrow-band absorption features that distinguish these minerals are being systematically attenuated by the DAE's convolution kernels.

---

## 7. Recommended Next Steps

### Priority 1 — Fix Narrow-Band Feature Attenuation (Critical)
The root cause is the DAE kernel's smoothing effect on narrow absorptions. Options to fix:
1. **Enable `LAMBDA_SAM` regularization** (currently 0.00) — even a small value (0.001–0.01) will penalise spectral angle distortion during training
2. **Add per-band blend weighting**: compute local band depth in each pixel's spectrum before blending; pixels with detected absorption features should use a higher raw fraction
3. **Inference-time mineral band masking**: protect known absorption center bands (BD1900→~1.93 µm, BD2210→~2.21 µm, BD1500→~1.50 µm) with stronger blending toward raw during inference
4. **Reduce BLEND_MAX** from 0.95 to 0.75 to systematically keep more raw signal

### Priority 2 — Fix Mare Imbrium Target (SAM Issue)
- Retrain the 256-band local DAE with more training spectra (increase from 15,000 → 25,000) or add light SAM regularization
- Consider excluding this strip from mineral analysis until resolved

### Priority 3 — Regenerate pipeline_summary.json and Manifest
- Re-run the pipeline summary/manifest generation
- Validate the `M3G20090204T152951_V01_RFL-001` strip (Mare Serenitatis second strip) — it has a denoised .npy but no validation report

### Priority 4 — Increase Lunar Highlands South SNR
- Both strips have raw SNR ~15–17 — below comfortable mineral detection levels
- Consider spatial averaging or band grouping before mineral mapping in this scene
