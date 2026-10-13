# Hyperspectral Mineral Identification — Journal Expansion Plan
**Expanding from CVIP 2026 Conference Paper to Journal Submission**
*Prepared by V. Mohanapriya — September 2026*

---

## Overview

This document outlines the plan to expand the CVIP 2026 conference paper — *"Bridging Spectroscopy and Deep Learning for Next-Generation Mineral Identification across Martian and Lunar Surfaces"* — into a full journal article. The conference paper, currently under revision after Round 1 review, demonstrated a physics-anchored deep learning pipeline for mineral identification using hyperspectral orbital data from two planetary missions. This journal expansion will substantially strengthen the scope, rigor, and coverage of that work.

**Primary Journal Target:** GIScience & Remote Sensing (Impact Factor: 6.8)
**Secondary Journal Target:** ISPRS Open Journal of Photogrammetry and Remote Sensing (Impact Factor: 7.0)

---

## 1. Conference Paper — Current Status

The CVIP 2026 paper introduced a six-stage pipeline for planetary mineral identification from hyperspectral data, applied to two missions:

- **M³ (Moon Mineralogy Mapper)** — Chandrayaan-1, 85 spectral bands (0.43–2.96 µm), lunar surface
- **CRISM (Compact Reconnaissance Imaging Spectrometer for Mars)** — MRO, 438 spectral bands (0.362–3.92 µm), Martian surface

The core methodology combines physics-grounded spectral preprocessing, contrastive representation learning, and unsupervised mineral discovery, addressing key gaps in existing approaches:

1. Conventional Spectral Parameter Maps (SPMs) compress hundreds of bands into scalar indices, losing inter-band covariance that is diagnostic of mineral mixtures
2. No unified framework existed for simultaneous CRISM and M³ mineral identification
3. Existing unsupervised methods imposed rigid geometric assumptions without uncertainty quantification
4. Laboratory reference spectra (USGS splib07, RELAB) were not adapted to match instrument-specific Spectral Response Functions (SRFs)

The paper is currently in **Round 1 Revision** at CVIP 2026. The revised version addresses reviewer comments and strengthens the experimental analysis.

---

## 2. Current Work — What the Journal Expands

The journal article will build on the conference foundation and extend it significantly in the following directions:

### 2.1 Broader Regional Coverage on M³ and CRISM

The conference paper demonstrated the pipeline on selected orbital observations. The journal version will apply the same methodology systematically across **multiple distinct regions** of both missions:

- **M³ regions**: mare basaltic plains, highland anorthosite terrains, pyroclastic deposits, polar permanently shadowed regions — covering mineralogically diverse geological contexts
- **CRISM regions**: phyllosilicate-bearing units (Mawrth Vallis), mafic/olivine-rich sites (Nili Fossae), carbonate outcrops (Isidis Basin) — spanning key mineralogical diversity on Mars

This regional expansion will allow us to assess how well the pipeline generalises across geologically distinct areas, rather than a single scene — a requirement explicitly expected by the target journals.

### 2.2 Addressing Known Limitations

The conference paper identified three limitations that the journal expansion will directly address:

- The 8 SPM anchor indices will be expanded or refined based on a broader regional analysis
- The fixed 12-endmember spectral library assumption will be revisited using scene-adaptive endmember selection (Vertex Component Analysis)
- Per-pixel uncertainty quantification will be validated more rigorously across regions with different spectral complexity

### 2.3 Future Extension — PRISMA (Earth-Analog)

As a longer-term goal, the pipeline will be extended to **PRISMA** (PRecursore IperSpettrale della Missione Applicativa), the Italian Space Agency's Earth-observing hyperspectral satellite. PRISMA serves as an Earth-analog instrument, enabling:

- Testing cross-mission transfer of the pipeline to terrestrial hyperspectral data
- Validation against field-measured mineralogy at analog sites (e.g., arid evaporite environments, hydrothermal alteration zones)
- Establishing a foundation for cross-planetary generalisation studies

This PRISMA extension is scoped as **future work** in the journal article and is not part of the current submission scope.

---

## 3. What the Journal Adds Over the Conference Paper

| Aspect | Conference Paper (CVIP 2026) | Journal Article |
|---|---|---|
| **Datasets** | Selected M³ and CRISM scenes | Multiple regions across M³ and CRISM |
| **Validation** | SPM correlation, SAM separability | Extended regional analysis + uncertainty evaluation |
| **Scope** | Proof-of-concept pipeline | Systematic evaluation across geological diversity |
| **Future extension** | Mentioned briefly | Formally scoped PRISMA plan |
| **Length** | ~14 pages (conference format) | Full journal article (~30–40 pages) |

---

## 4. Target Journals

### Primary: GIScience & Remote Sensing
- **Publisher:** Taylor & Francis
- **Impact Factor:** 6.8
- **Scope fit:** Strong match — the journal covers remote sensing methodology, geospatial analysis, and planetary/Earth observation applications. Hyperspectral mineral mapping across orbital sensors is well within scope.
- **Rationale for primary choice:** The journal's scope directly covers the integration of deep learning with physical remote sensing, making it the strongest fit for a physics-anchored contrastive learning pipeline applied to planetary hyperspectral data.

### Secondary: ISPRS Open Journal of Photogrammetry and Remote Sensing
- **Publisher:** Elsevier / ISPRS
- **Impact Factor:** 7.0
- **Scope fit:** Strong match — ISPRS Open JPR&S covers photogrammetry, remote sensing, and geospatial information science with a focus on methodological rigor and sensor-level analysis.
- **Rationale for secondary choice:** The journal's emphasis on sensor-specific methodology and open-access publication makes it a suitable alternative, particularly for work involving multiple orbital instruments.

---

## 5. Current State and Open Items

### What Is Ready
- The core six-stage pipeline is implemented, tested, and revised based on CVIP reviewer feedback
- SAM-based spectral separability analysis across mineral classes is complete
- Per-pixel uncertainty quantification (Spearman ρ correlation with SPM indices) is implemented
- Contrastive encoder training with physics anchors is validated on M³ and CRISM scenes

### What Remains to Be Done

- **Regional expansion on M³:** Apply the pipeline to at least 3–4 distinct lunar regions; document mineralogical findings per region
- **Regional expansion on CRISM:** Apply the pipeline to targeted mineralogically diverse Martian sites; compare findings across regions
- **Manuscript drafting:** The journal manuscript has not yet been drafted; this will begin once the regional expansion is completed
- **PRISMA scoping (future work section):** Define the acquisition plan and preprocessing steps for PRISMA data to include as a formally scoped future direction
- **Response to CVIP Round 1 reviews:** Complete and submit the revised conference paper before beginning the journal draft

---

## 6. Summary

The CVIP 2026 paper established a working, physics-grounded pipeline for hyperspectral mineral identification on Mars and the Moon. The journal expansion will take this foundation and apply it more systematically — across multiple regions of both M³ and CRISM — to produce the kind of rigorous, multi-scene evaluation that the target venues require. The PRISMA extension is scoped as a clearly defined future direction. Both target journals have strong impact factors and direct scope alignment. The next concrete step is completing the CVIP revision, followed by the regional expansion experiments on M³ and CRISM.
