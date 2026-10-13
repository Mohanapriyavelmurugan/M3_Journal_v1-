# Phase 2 Mineral Identification — Status Report

**Project:** M3 Journal lunar hyperspectral mineral mapping  
**Scope:** Original Phase 2 methodology (library SAM + trained neural unmixer)  
**Report date:** 4 October 2026  
**Phase 1 status:** Immutable (completed denoising / validation; not modified)

---

## 1. Purpose of this report

This document explains, in plain language:

1. What work has been done so far on mineral identification  
2. How the mineral identification methodology works, step by step  
3. What scientific and engineering fixes were applied  
4. What problems remain  
5. What results we obtain now (and how to read them correctly)

---

## 2. Big picture (simple summary)

Moon pixels from Chandrayaan-1 M³ are almost never pure minerals. Each pixel is usually a **mixture** of plagioclase, pyroxene, olivine, etc., plus space-weathered soil.

So Phase 2 uses **two complementary products**:

| Product | Script | Output folder | Question it answers |
|--------|--------|---------------|---------------------|
| **Fast SAM matcher** | `phase2_mineral_identification.py` | `06_mineral_id` | “Which library mineral is the closest match (or unclassified)?” |
| **Trained workflow** | `phase2_trained_mineral_workflow.py` | `07_mineral_workflow` | “What mixture of minerals makes up this pixel / scene?” |

**Shared curated spectral library:** `phase2_spectral_library.py` (RELAB + USGS, cleaned and filtered).

**Important:** Do **not** replace this original methodology with band-parameter-only maps. Band parameters remain secondary diagnostic tools.

---

## 3. What has been done till now

### 3.1 Starting problem

Earlier Phase 2 runs were scientifically unreliable. Typical failures:

- Almost every pixel labeled **iron oxide** (geologically nonsense for most lunar scenes)
- Very small / contaminated spectral library (~27 entries)
- High SAM angles (~0.23–0.28 rad) meaning poor spectral matches
- Wrong cluster names (e.g. “Water_ice” on mid-latitude mare/highland)
- Neural unmixer collapsing to one catch-all endmember (glass, fayalite, etc.)
- Stale outputs from old runs mistaken for new results

### 3.2 Work completed

1. **Root-cause analysis** of SAM + trained workflow failure modes  
2. **Shared curated library** for both Phase 2 scripts  
3. **Hard gates** so featureless / unmatched spectra stay unclassified or get physics-checked  
4. **Trained workflow repairs** (unique endmembers, crater vs mare endmember sets, SPM v3, abundance gates)  
5. **Scene-level mineral presence product** (`*_scene_mineral_presence.json`)  
6. **Cursor skills / agents** for repeatable QA and debugging:
   - `.cursor/skills/m3-phase2-library`
   - `.cursor/skills/m3-phase2-run-validate`
   - `.cursor/skills/m3-phase2-debug`
   - `.cursor/skills/m3-phase2-scene-qa`
   - `.cursor/rules/m3-phase2.mdc`
7. **Validation smoke tests** on Copernicus, Mare Tranquillitatis, and Jackson Crater

### 3.3 What was deliberately *not* changed

- Phase 1 pipeline (`step0`–`step5`, denoising, validation) remains untouched  
- Original methodology remains primary (SAM + contrastive encoder + neural unmixer)  
- HDBSCAN clustering remains informational (does not decide mineral identity)

---

## 4. Mineral identification methodology (step by step)

### Stage A — Inputs from Phase 1

For each READY scene:

1. Load the **denoised reflectance cube** from `04_denoised`  
2. Use M³ wavelengths and a valid-pixel mask  
3. Do **not** re-run Phase 1

---

### Stage B — Build / reuse the spectral library

**File:** `phase2_spectral_library.py`

1. Collect candidate spectra from **RELAB** and **USGS**  
2. Keep only lunar-relevant mineral names using **whole-word matching** (not loose substring matches)  
3. Apply a **blocklist** (fiberglass, roofing, wrong “ice”/“glass” hits, etc.)  
4. Resample each library spectrum onto the M³ wavelength grid  
5. Restrict matching to a common useful window (~**0.75–2.45 µm**), avoiding bad bands  
6. Continuum-remove and normalize spectra for fair shape comparison  
7. Cap how many USGS spectra per class so one mineral does not flood the library  

**Typical healthy library size:** ~140–155 curated entries (not ~27).

---

### Stage C — Fast SAM mineral identification (`06_mineral_id`)

**File:** `phase2_mineral_identification.py`

This is the **hard-label** product.

#### Step C1 — Scene preprocessing
- Continuum removal on the scene spectra  
- Compute spectral parameter maps (SPM) such as OLINDEX3, HCPINDEX2, BD1000, BD2000 where needed  

#### Step C2 — Spectral Angle Mapper (SAM)
For every valid pixel:

1. Compare the pixel spectrum to every library spectrum  
2. Compute the **spectral angle** (SAM): smaller angle = better match  
3. Keep the **best** library mineral if it passes gates  

#### Step C3 — Quality gates (why many pixels stay unclassified)
A pixel is labeled only if:

1. **SAM ≤ 0.15 rad** (good enough shape match)  
2. **Beat-flat gate:** the best mineral must beat a flat continuum by a small margin  
   - Exception: plagioclase/anorthosite on highland/impact (they are naturally nearly featureless)  
   - On mare, that exception is **not** used (otherwise mare becomes fake plagioclase)  
3. **SPM agreement** for minerals that have diagnostic priors (e.g. olivine ↔ OLINDEX3)  
4. **Terrain plausibility** (e.g. water ice / sulfate are implausible on mid-latitude mare)

If gates fail → **unclassified** (this is honest, not a failure).

#### Step C4 — Outputs
- Mineral label map  
- Confidence / uncertainty  
- Best SAM map  
- `*_mineral_summary.json` with per-mineral fractions and validation flags  

**How to read SAM results:**  
SAM answers “confident closest pure-library match.” High unclassified fraction on mixed soils is expected.

---

### Stage D — Trained workflow + neural unmixer (`07_mineral_workflow`)

**File:** `phase2_trained_mineral_workflow.py`

This is the **mixture** product for scene-wide mineral inventory.

#### Step D1 — Scene continuum + SPM (Stage 2)
- Continuum-removed cube  
- L2-normalized CR cube  
- SPM maps from **raw reflectance** (SPM version 3), with cache invalidation if broken  

#### Step D2 — Training pool
Build training spectra from:

1. Stratified samples from the scene  
2. Library spectra  
3. Synthetic two-mineral mixtures of co-occurring lunar minerals  

#### Step D3 — Contrastive spectral encoder
- Self-supervised 1D encoder (NT-Xent + SPM anchor + class separation)  
- Learns a latent representation of spectral shape  
- Used by the unmixer; **not** the final mineral name by itself  

#### Step D4 — HDBSCAN clustering (informational)
- Clusters latents for QA / visualization  
- Cluster names come only from mafic SWIR indices (OLINDEX3, HCPINDEX2, BD1000, BD2000)  
- Mid-latitude hydration/ice indices are excluded from naming  
- Negative silhouette does **not** invalidate mineral maps  

#### Step D5 — Endmember selection (terrain-aware)
- **Mare:** broader set allowed (pyroxene, olivine, ilmenite, glass, plagioclase, …)  
- **Crater / highland:** restricted set to avoid catch-alls:
  - Keep: plagioclase (anorthite), orthopyroxene, clinopyroxene, olivine (forsterite), ilmenite  
  - Exclude: mare basalt glass, fayalite, spinel, apatite, extra Ca-plagioclase species, hydrated rares, iron oxide  

At most **one spectrum per mineral class** (no duplicate clinopyroxene×3).

#### Step D6 — Physics-constrained neural unmixer
For each pixel, estimate abundances \(a_i\) such that:

- abundances ≥ 0  
- abundances sum to ~1  
- reconstructed spectrum ≈ measured continuum-removed spectrum  

Training uses reconstruction loss, sparsity, endmember anchoring, and collapse penalties.

#### Step D7 — Post-unmixing SPM gates
After unmixing:

1. Suppress minerals that lack their diagnostic SPM signal  
2. Reassign dominant labels toward geologically preferred minerals  
3. **Gate the abundance cube itself** (zero unsupported fractions, then renormalize)  

This prevents “mean abundance still 50% fayalite” while the map looks like plagioclase.

#### Step D8 — Scene mineral presence (whole-scene inventory)
Write `*_scene_mineral_presence.json`:

| Status | Meaning (simple) |
|--------|-------------------|
| **PRESENT** | Meaningful mixture fraction across the scene (or many strong pixels) |
| **POSSIBLE** | Weak mixture signal and/or some SAM support; not a firm detection |
| **ABSENT** | No meaningful support |

**This is the correct way to answer: “Which minerals are present in the whole scene?”**

---

## 5. Fixes applied (detailed)

### 5.1 Library / matching fixes

| Fix | Why it was needed |
|-----|-------------------|
| Whole-word mineral name matching | Substring matching pulled contaminants into the library |
| Blocklist | Removed fiberglass / wrong ice / wrong glass matches |
| Common-band window 0.75–2.45 µm | Stopped comparing mismatched wavelength regions |
| Shared library module for both scripts | SAM and unmixer used inconsistent libraries |
| Unique endmembers per class | Stopped clinopyroxene×3 / ilmenite×3 collapse |

### 5.2 Fast SAM (`06`) fixes

| Fix | Why it was needed |
|-----|-------------------|
| Beat-flat gate | Featureless pixels were matching anything |
| Plagioclase exemption only on highland/impact | Highland feldspar is nearly featureless; mare is not |
| Hard SAM max 0.15 rad | Rejected weak matches |
| SPM gates + terrain plausibility | Removed geologically absurd labels |
| Honest unclassified fraction | Stopped forcing a label on every pixel |

### 5.3 Trained workflow (`07`) fixes

| Fix | Why it was needed |
|-----|-------------------|
| SPM v3 from raw reflectance + version marker | Zeroed / flat SPM channels broke mare/highland logic |
| Cap HDBSCAN sample size + progress logs | Clustering hung for hours with no output |
| `--resume` for encoder/latents | Avoided redoing expensive train/encode after small fixes |
| Exclude glass on highland/crater | Glass was a catch-all (~70% mean abundance) |
| Exclude fayalite / spinel / apatite / extra plag on crater | Next catch-alls after glass was removed |
| Abundance SPM gating | Dominant-map gates alone left junk mean abundances |
| Scene presence JSON | Dominant map cannot list mixture minerals |
| Safer saving / report-before-figures | Runs died overwriting open memmaps; report lagged maps |

### 5.4 Process / QA fixes

| Fix | Why it was needed |
|-----|-------------------|
| Timestamp checks (report vs npy maps) | Stale JSON reports were misread as latest results |
| Skills + agent prompts | Repeatable SceneDeepQA / ScenePresence / FixThenRerun |
| Prefer presence + mean abundances over argmax map | Argmax hides secondary minerals in mixtures |

---

## 6. Problems that still need to be addressed

These are remaining limitations. Some are physics; some are next engineering steps.

### 6.1 Encoder latent separation is still weak

- Silhouette scores are often negative  
- Clustering remains informational only  
- **Needed:** improve encoder for unmixer quality (SPM anchors, more mix training), but **do not** use silhouette as a pass/fail for mineral ID  

### 6.2 Dominant mineral map is not a full inventory

- On highland mixtures (~91% plagioclase + ~8% OPX), argmax is almost always plagioclase  
- **Needed (product discipline):** treat presence + gated abundances as primary; keep dominant map as “majority lithology” only  
- Optional next step: publish binary presence maps (`abund ≥ 0.10` / scene-adaptive threshold)

### 6.3 SAM unclassified fraction stays high on mixed soils

- Expected for pure-endmember SAM on weathered mixtures  
- **Needed:** keep SAM strict; use unmixer for scene-wide presence; optional SPM-guided “possible mafic” flag on unclassified SAM pixels (without forcing SAM labels)

### 6.4 Rare minerals on highland remain weak (e.g. ilmenite POSSIBLE)

- Small SAM counts or tiny abundances are not detections  
- **Needed:** compound rule for PRESENT (e.g. ilmenite only if BD1000 high **and** BD2000 low **and** HCPINDEX2 low **and** gated abundance above floor)

### 6.5 Mare scenes still need careful unmixer tuning

- Mare soils are strongly mixed; glass/pyroxene confusion can return if endmember set / gates are wrong  
- **Needed:** re-validate mare scenes with the same presence product + gated abundances after highland fixes

### 6.6 Operational pitfalls

- Always check that `phase2_workflow_report.json` is not older than abundance/dominant `.npy` files  
- Always check `library_entries` in SAM summary (≥ ~100; if ~27, the run is stale)

---

## 7. Results we are obtaining now

### 7.1 How to read results correctly

| Question | Read this |
|----------|-----------|
| Closest pure-library match? | `06_mineral_id/*_mineral_summary.json` |
| Mixture fractions? | `07` gated `mean_abundances_pct` |
| Minerals present in the whole scene? | `07/*_scene_mineral_presence.json` |
| Majority surface type map? | Dominant mineral map (secondary) |
| Cluster names? | Informational QA only |

---

### 7.2 Copernicus Crater (impact / ejecta) — Fast SAM (`06`)

Healthy curated library (~143 entries). Approximate classified fractions:

| Mineral | Approx. fraction | Comment |
|---------|------------------|---------|
| Plagioclase | ~28.6% | Dominant among classified |
| Anorthosite | ~6.5% | Highland feldspar family |
| Olivine | ~1.4% | Plausible local mafic |
| Pyroxene / others | small | Secondary |
| Iron oxide | ~0% | Monopoly removed |
| Unclassified | ~63% | Expected for mixed ejecta |

**Interpretation:** Highland/impact pattern is geologically plausible. Unclassified is not “empty.”

---

### 7.3 Jackson Crater (farside highland / impact) — current best inventory (`07`)

After crater endmember restriction + abundance gating:

| Mineral | Status | Mean mixture abundance | Comment |
|---------|--------|------------------------|---------|
| Plagioclase (anorthite) | **PRESENT** | ~91.5% | Expected for farside highland |
| Orthopyroxene (LCP) | **PRESENT** | ~8.4% | Real secondary mafic mixture |
| Clinopyroxene | ABSENT | ~0.06% | Not supported |
| Olivine | POSSIBLE | ~0% | Weak / local only |
| Ilmenite | POSSIBLE | ~0% | Not a firm highland detection |

**SAM on Jackson:** plagioclase among classified (~16%), ~83% unclassified — consistent with mixture-dominated soils.

**Interpretation:** Scene inventory is **plagioclase + orthopyroxene**. The dominant map looking ~100% plagioclase is expected because OPX is a minority mixture component, not the majority label.

---

### 7.4 Before vs after (conceptual)

| Then (broken) | Now (fixed methodology) |
|---------------|-------------------------|
| ~99% iron oxide | Iron oxide monopoly gone |
| Library ~27 contaminants | Curated ~140+ entries |
| Forced labels everywhere | Honest unclassified + mixture abundances |
| Water_ice cluster names on mare | Cluster naming limited to mafic SWIR indices |
| Glass / fayalite catch-alls on highland | Restricted highland endmember set |
| Only argmax map to judge “presence” | Presence JSON + gated mean abundances |

---

## 8. How to run (current recommended commands)

```powershell
cd "D:\M3 Journal"
$env:PYTHONIOENCODING="utf-8"

# Fast SAM (hard labels)
python Code_scripts\phase2_mineral_identification.py --scene-name "Jackson"

# Trained workflow + unmixer + presence (use --resume when encoder/latents already exist)
python Code_scripts\phase2_trained_mineral_workflow.py --scene-name "Jackson" --resume --max-cluster-pixels 40000
```

Key outputs:

- `...\Pipeline_outputs\06_mineral_id\*_mineral_summary.json`  
- `...\Pipeline_outputs\07_mineral_workflow\*_phase2_workflow_report.json`  
- `...\Pipeline_outputs\07_mineral_workflow\*_scene_mineral_presence.json`  
- `...\Pipeline_outputs\07_mineral_workflow\*_abundance_maps.npy`

---

## 9. Scientific bottom line

The original methodology is still the right one for this project:

1. **SAM** = careful hard matching to a curated library  
2. **Neural unmixer** = mixture fractions for mixed lunar regolith  
3. **Presence product** = whole-scene mineral inventory  

What changed is not the scientific idea — it is the **library quality, gates, terrain-aware endmembers, SPM correctness, and how results are reported**.

Current Jackson-style highland answer is scientifically coherent:

> The scene is plagioclase-dominated highland material with a secondary orthopyroxene mixture component. Rare mafics remain only POSSIBLE unless compound spectral evidence is strong.

Remaining work is refinement (encoder quality for unmixing, clearer presence maps, stricter rare-mineral rules, mare re-validation) — not a need to abandon the original methodology.

---

## 10. Related documents and skills

| Item | Path |
|------|------|
| This report | `phase2_mineral_identification_status_report.md` |
| Earlier analysis notes | `phase2_mineral_identification_analysis.md` |
| Implementation guide | `phase2_fix_implementation_guide.md` |
| Shared library | `Code_scripts/phase2_spectral_library.py` |
| Fast SAM | `Code_scripts/phase2_mineral_identification.py` |
| Trained workflow | `Code_scripts/phase2_trained_mineral_workflow.py` |
| Scene QA skill | `.cursor/skills/m3-phase2-scene-qa/SKILL.md` |
| Debug skill | `.cursor/skills/m3-phase2-debug/SKILL.md` |
| Run/validate skill | `.cursor/skills/m3-phase2-run-validate/SKILL.md` |
| Phase 2 rule | `.cursor/rules/m3-phase2.mdc` |

---

*End of report.*
