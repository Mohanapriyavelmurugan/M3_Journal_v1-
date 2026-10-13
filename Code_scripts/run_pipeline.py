"""
run_pipeline.py  --  M3 Phase 1 Pipeline Orchestrator (Phase A -> B -> C)
======================================================================
"""

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
JOURNAL_ROOT = Path(r"D:\M3 Journal")
MODELS_DIR   = JOURNAL_ROOT / "models"
GLOBAL_B     = 85   # Global-mode band count

SCENES = [
    JOURNAL_ROOT / "Aristarchus Plateau"     / "M3G20090209T054031_V01_RFL",
    JOURNAL_ROOT / "Bullialdus Crater"       / "M3T20090417T021417_V01_RFL",
    JOURNAL_ROOT / "Copernicus Crater"       / "M3G20090207T083836_V01_RFL",
    JOURNAL_ROOT / "Highlands near Tycho"    / "M3G20090416T081001_V01_RFL",
    JOURNAL_ROOT / "Jackson Crater"          / "M3G20081225T074420_V01_RFL",
    JOURNAL_ROOT / "Lunar Highlands South"   / "M3G20090205T153710_V01_RFL",
    JOURNAL_ROOT / "Mare Imbrium"            / "M3G20090609T230815_V01_RFL",
    JOURNAL_ROOT / "Mare Serenitatis"        / "M3G20090204T113444_V01_RFL",
    JOURNAL_ROOT / "Mare Tranquillitatis"    / "M3G20090203T160452_V01_RFL",
    JOURNAL_ROOT / "South Pole Region"       / "M3G20090610T153615_V01_RFL",
]

def _import_steps():
    import step0_load     as s0
    import step1_eda      as s1
    import step2_noise    as s2
    import step3_physics  as s3
    import step4_denoise  as s4
    import step5_validate as s5
    return s0, s1, s2, s3, s4, s5

def _hms(t0):
    s = int(time.time() - t0)
    return f"{s//3600}h {(s%3600)//60}m {s%60}s"

def _scene_fname(scene_dir):
    imgs = sorted(Path(scene_dir).glob("*_RFL*.IMG"))
    if not imgs:
        return Path(scene_dir).name
    return imgs[0].stem

def _matching_files(folder, fname, suffix):
    folder = Path(folder)
    return sorted(folder.glob(f"{fname}{suffix}")) if folder.exists() else []

def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Phase A: Physics Correction (Steps 0-3)
# ---------------------------------------------------------------------------
def run_phase_a(scene_dir, s0, s1, s2, s3, resume=False):
    sd   = str(scene_dir)
    loc  = scene_dir.parent.name
    expected_fname = _scene_fname(scene_dir)
    t0   = time.time()
    
    print(f"\n{'='*60}\n  PHASE A (Physics): {loc}\n{'='*60}")
    
    if resume:
        phys_dir = scene_dir.parent / "Pipeline_outputs" / "03_physics"
        npys = _matching_files(phys_dir, expected_fname, "_physics_corrected.npy")
        if npys:
            print(f"  [RESUME] Phase A already complete -- SKIPPING.")
            return True

    try:
        cube, waves, fname, meta, DIRS = s0.load_scene(sd)
    except Exception as e:
        print(f"  ERROR Step 0: {e}"); traceback.print_exc()
        return False

    try:
        EDA, vpm, bs, vb, db, tb, sb = s1.run_eda(cube, waves, fname, DIRS)
    except Exception as e:
        print(f"  ERROR Step 1: {e}"); traceback.print_exc()
        return False

    try:
        NM, sco, sfrac, stb, spb, sg, td, smd, spc = s2.run_noise_id(
            cube, waves, fname, DIRS, bs, vb, db, tb, vpm)
    except Exception as e:
        print(f"  ERROR Step 2: {e}"); traceback.print_exc()
        return False

    try:
        cp, snrp = s3.run_physics(
            cube, waves, fname, DIRS, bs, vb, db, tb, vpm,
            sfrac, stb, spb, td, smd, spc)
    except Exception as e:
        print(f"  ERROR Step 3: {e}"); traceback.print_exc()
        return False
    
    del cube, cp
    _free_gpu()
    print(f"  [Phase A] Done in {_hms(t0)}.")
    return True


# ---------------------------------------------------------------------------
# Phase B: Global ML Pool Construction
# ---------------------------------------------------------------------------
def build_pool_phase_b(scenes, s4, force_retrain):
    dae_path  = MODELS_DIR / f"m3_dae_B{GLOBAL_B}.pth"
    pool_path = MODELS_DIR / "multi_scene_pool.npz"

    if force_retrain:
        # Delete ALL saved model files so train_dae is forced to retrain
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for pth in MODELS_DIR.glob("*.pth"):
            pth.unlink(missing_ok=True)
            print(f"  [Retrain] Deleted old model: {pth.name}")
        pool_path.unlink(missing_ok=True)

    if dae_path.exists() and not force_retrain:
        print(f"\n{'='*60}\n  PHASE B (ML Pool)\n{'='*60}")
        print(f"  [Pool] DAE already exists ({dae_path.name}) -- skipping pool build.")
        return None

    if pool_path.exists() and not force_retrain:
        data  = np.load(str(pool_path))
        pool  = data["spectra"]
        B     = int(data["B"])
        valid = np.all(np.isfinite(pool), axis=1)
        pool  = pool[valid]
        if len(pool) > 100 and B == GLOBAL_B:
            print(f"\n{'='*60}\n  PHASE B (ML Pool)\n{'='*60}")
            print(f"  [Pool] Loaded cached: {len(pool)} spectra (B={B})")
            return pool
        pool_path.unlink(missing_ok=True)

    print(f"\n{'='*60}\n  PHASE B: Building Pool from Physics Cubes\n{'='*60}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_spectra = []
    
    for sd in scenes:
        print(f"    {sd.parent.name} ...", end="", flush=True)
        try:
            spec_arr, B = s4.collect_spectra_from_scene(str(sd), n_spectra=1500)
        except Exception as e:
            print(f" ERROR: {e}")
            continue
            
        if spec_arr is None or B != GLOBAL_B:
            print(f" SKIP (Target Mode B={B} cannot mix with Global Mode B={GLOBAL_B} pool. Will train local DAE later.)")
            continue
            
        valid = np.all(np.isfinite(spec_arr), axis=1)
        spec_arr = spec_arr[valid]
        if len(spec_arr) < 10:
            print(f" SKIP (too few valid spectra)")
            continue
            
        all_spectra.append(spec_arr)
        print(f" {len(spec_arr)} ok")
        _free_gpu()

    if not all_spectra:
        print("  [Pool] Nothing collected -- DAE will train on single-scene spectra.")
        return None

    combined = np.vstack(all_spectra)
    rng = np.random.default_rng(42)
    combined = combined[rng.permutation(len(combined))]
    print(f"  [Pool] Combined: {len(combined)} spectra (B={GLOBAL_B})")
    np.savez_compressed(str(pool_path), spectra=combined, B=np.array(GLOBAL_B))
    return combined


# ---------------------------------------------------------------------------
# Phase C: ML Denoising and Validation (Steps 4-5)
# ---------------------------------------------------------------------------
def run_phase_c(scene_dir, s0, s1, s2, s3, s4, s5, extra_spectra=None,
                resume=False, force_retrain=False, refresh_denoise=False):
    sd   = str(scene_dir)
    loc  = scene_dir.parent.name
    expected_fname = _scene_fname(scene_dir)
    t0   = time.time()
    
    print(f"\n{'='*60}\n  PHASE C (ML + Val): {loc}\n{'='*60}")

    if resume and not force_retrain and not refresh_denoise:
        den_dir = scene_dir.parent / "Pipeline_outputs" / "04_denoised"
        val_dir = scene_dir.parent / "Pipeline_outputs" / "05_validation"
        npys    = _matching_files(den_dir, expected_fname, "_denoised.npy")
        jsons   = _matching_files(val_dir, expected_fname, "_validation_report.json")
        if npys and jsons:
            print(f"  [RESUME] Phase C already complete -- SKIPPING.")
            return {"scene": loc, "status": "CACHED", "npy": str(npys[0]),
                    "verdict": json.load(open(str(jsons[0]))).get("overall_verdict","?"),
                    "elapsed": "0h 0m 0s"}

    # ── Fast path: load all Phase A results from cached disk files ────────────
    # If physics_corrected.npy + eda.json + noise_map.json + physics_log.json
    # all exist, skip the expensive Steps 0-3 re-execution (saves 15-30 min).
    out_root   = scene_dir.parent / "Pipeline_outputs"
    phys_npys  = _matching_files(out_root / "03_physics", expected_fname,
                                 "_physics_corrected.npy")
    eda_jsons  = _matching_files(out_root / "01_eda", expected_fname,
                                 "_eda.json")
    noise_jsons= _matching_files(out_root / "02_noise", expected_fname,
                                 "_noise_map.json")
    phys_jsons = _matching_files(out_root / "03_physics", expected_fname,
                                 "_physics_log.json")

    cached = phys_npys and eda_jsons and noise_jsons and phys_jsons

    if cached:
        print("  [Cache] Loading Phase A results from disk (skipping Steps 0-3)...")
        try:
            # Load waves/fname/DIRS from s0 (fast: only reads the tiny HDR file)
            _, waves, fname, _, DIRS = s0.load_scene(sd, header_only=True)
        except TypeError:
            # Fallback: load_scene doesn't support header_only yet — use the npy path
            # to derive fname, and call load_scene normally but immediately free cube
            _cube_tmp, waves, fname, _, DIRS = s0.load_scene(sd)
            del _cube_tmp
            _free_gpu()

        try:
            eda   = json.load(open(str(eda_jsons[0])))
            nm    = json.load(open(str(noise_jsons[0])))
            plog  = json.load(open(str(phys_jsons[0])))

            vb    = [int(x) for x in eda["valid_bands"]]
            db    = [int(x) for x in eda["dead_bands"]]
            tb    = [int(x) for x in eda["thermal_bands"]]

            # Rebuild SNR band_stats dict from EDA (used by run_denoise via bs["snr"])
            bs    = {"snr": np.array(eda["snr"], dtype=np.float64)}

            # Load physics cube via mmap (no extra RAM)
            cp    = np.load(str(phys_npys[0]), mmap_mode="r+")
            B     = cp.shape[2]

            # Rebuild valid_pixel_mask from the physics cube (fast)
            from step1_eda import MIN_VALID_BANDS
            vpm   = (np.sum(np.isfinite(cp), axis=2) >= MIN_VALID_BANDS)

            # Noise metadata from JSON
            NM    = nm
            stb   = [int(x) for x in nm["N2_striped_bands"]]
            spb   = [int(x) for x in nm["N3_spiked_bands"]]
            sg    = [int(x) for x in nm["N1_gaussian_severe"]]
            td    = bool(nm["N4_thermal_detected"])
            smd   = bool(nm["N5_smile_detected"])
            sfrac = np.array(nm["spike_frac_per_band"], dtype=np.float32)
            sco   = {int(k): v for k, v in nm.get("stripe_col_offsets", {}).items()}

            # SNR after physics (fallback to EDA SNR for older JSONs)
            if "snr_after_physics" in plog:
                snrp = np.array(plog["snr_after_physics"], dtype=np.float64)
            else:
                snrp = bs["snr"].copy()

            print(f"  [Cache] Loaded: B={B}  vb={len(vb)}  stb={len(stb)}  "
                  f"sg={len(sg)}  physics SNR={snrp[vb].mean():.3f}", flush=True)
        except Exception as e:
            print(f"  [Cache] Failed ({e!r}) -- falling back to full recompute")
            if 'cp' in locals():
                del cp
            _free_gpu()
            cached = False

    if not cached:
        # Run full steps 0-3 in memory to get necessary metadata
        try:
            cube, waves, fname, meta, DIRS = s0.load_scene(sd)
            B = cube.shape[2]
            EDA, vpm, bs, vb, db, tb, sb = s1.run_eda(cube, waves, fname, DIRS)
            NM, sco, sfrac, stb, spb, sg, td, smd, spc = s2.run_noise_id(
                cube, waves, fname, DIRS, bs, vb, db, tb, vpm)
            cp, snrp = s3.run_physics(
                cube, waves, fname, DIRS, bs, vb, db, tb, vpm,
                sfrac, stb, spb, td, smd, spc)
        except Exception as e:
            print(f"  ERROR Steps 0-3 setup: {e}"); traceback.print_exc()
            return {"scene": loc, "status": "FAIL Setup", "verdict": "ERROR"}
        del cube
        _free_gpu()


    pool = (extra_spectra if extra_spectra is not None and extra_spectra.shape[1] == B else None)
    cd = snrml = bw = None
    
    try:
        cd, snrml, bw = s4.run_denoise(
            cp, cp, waves, fname, DIRS,
            bs, vb, db, vpm, stb, sco, snrp, sg,
            extra_spectra=pool)
    except Exception as e:
        print(f"  ERROR Step 4: {e}"); traceback.print_exc()
        print("  Falling back to physics-corrected cube as denoised output.")
        cd    = cp
        snrml = snrp.copy()
        bw    = np.zeros(B, dtype=np.float32)
        npy_path = str(Path(DIRS["04_denoised"]) / f"{fname}_denoised.npy")
        np.save(npy_path, cd)

    _free_gpu()

    # ── Free cubes BEFORE reloading raw cube to prevent OOM ────────
    # At this point cp (2.4 GB) + cd (2.4 GB) are live. Loading cube (2.4 GB)
    # would push RAM to 7+ GB. Delete them first; Step 5 reloads them via mmap.
    npy_path_denoised = str(Path(DIRS["04_denoised"]) / f"{fname}_denoised.npy")
    npy_path_physics  = str(Path(DIRS["03_physics"]) / f"{fname}_physics_corrected.npy")
    try: del cd
    except NameError: pass
    try: del cp
    except NameError: pass
    _free_gpu()

    verdict = "ERROR"
    try:
        print("  [Step 5] Reloading raw cube for validation...")
        cube, _, _, _, _ = s0.load_scene(sd)
        report  = s5.run_validation(
            cube, 
            np.load(npy_path_physics, mmap_mode="r"),    # reload cp cheaply
            np.load(npy_path_denoised, mmap_mode="r"),   # reload cd cheaply
            waves, fname, DIRS,
            vb, db, vpm,
            bs["snr"], snrp, snrml,
            sg, stb, spb, sfrac, td, smd, NM)
        verdict = report["overall_verdict"]
    except Exception as e:
        print(f"  ERROR Step 5: {e}"); traceback.print_exc()

    # Safe cleanup — each variable may be unbound if an earlier step failed
    try: del cube
    except NameError: pass
    try: del cp
    except NameError: pass
    _free_gpu()

    elapsed = _hms(t0)
    npy_path = str(Path(DIRS["04_denoised"]) / f"{fname}_denoised.npy")
    print(f"\n  [{loc}] Done in {elapsed}. Verdict: {verdict}")
    return {
        "scene":   loc,
        "fname":   fname,
        "verdict": verdict,
        "npy":     npy_path if Path(npy_path).exists() else None,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------
def run_all(target_scenes=None, force_retrain=False, resume=False,
            refresh_denoise=False, exclude_scenes=None):
    t_total = time.time()
    s0, s1, s2, s3, s4, s5 = _import_steps()

    scenes = [Path(s) for s in (target_scenes or SCENES)]
    scenes = [s for s in scenes if s.exists()]
    if exclude_scenes:
        exclude_terms = [x.lower() for x in exclude_scenes]
        scenes = [s for s in scenes
                  if not any(term in str(s.parent).lower() for term in exclude_terms)]
    print(f"\n  {'='*60}")
    print(f"  M3 Phase 1 Pipeline -- {len(scenes)}/10 scenes ready")
    print(f"  {'='*60}")

    # --- PHASE A ---
    for sd in scenes:
        run_phase_a(sd, s0, s1, s2, s3, resume and not force_retrain)

    # --- PHASE B ---
    extra_spectra = build_pool_phase_b(scenes, s4, force_retrain)

    # --- PHASE C ---
    summary = []
    for sd in scenes:
        result = run_phase_c(sd, s0, s1, s2, s3, s4, s5,
                             extra_spectra=extra_spectra, resume=resume,
                             force_retrain=force_retrain,
                             refresh_denoise=refresh_denoise)
        summary.append(result)
        _free_gpu()

    # --- SUMMARY ---
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    print(f"  {'Location':<24} {'Verdict':<12} {'Time':<12} {'Output'}")
    print("  " + "-"*66)
    for s in summary:
        v  = s.get("verdict", s.get("status", "?"))
        el = s.get("elapsed", "--")
        n  = s.get("npy") or ""
        nm = Path(n).name[:40] if n else "--"
        print(f"  {s['scene']:<24} {v:<12} {el:<12} {nm}")

    summ_path = JOURNAL_ROOT / "pipeline_summary.json"
    with open(str(summ_path), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    manifest = []
    for item in summary:
        report_path = None
        npy = item.get("npy")
        if npy:
            val_dir = Path(npy).parents[1] / "05_validation"
            reports = _matching_files(
                val_dir, item.get("fname") or Path(npy).name.replace("_denoised.npy", ""),
                "_validation_report.json")
            report_path = str(reports[0]) if reports else None
        if not report_path:
            manifest.append({"scene": item["scene"], "readiness": "NOT_READY",
                             "reason": "missing validation report"})
            continue
        rpt = json.load(open(report_path))
        m1 = rpt["metric1_snr"]
        m2 = rpt["metric2_sam"]
        m4 = rpt["metric4_pearson"]
        band_pct = rpt["band_decisions"]["pct_pass"]
        bd_fail = sum(1 for v in rpt["metric3_band_depth"].values()
                      if isinstance(v, dict) and v.get("status") == "FAIL")
        detected_features = {
            k for k, v in rpt["metric3_band_depth"].items()
            if isinstance(v, dict) and v.get("status") in ("PASS", "REVIEW", "FAIL")
        }
        roi_fail = sum(
            1 for k, v in rpt.get("roi_validation", {}).items()
            if k in detected_features and isinstance(v, dict) and v.get("status") == "FAIL"
        )
        roi_candidate_fail = sum(
            1 for k, v in rpt.get("roi_validation", {}).items()
            if k not in detected_features and isinstance(v, dict) and v.get("status") == "FAIL"
        )
        reasons = []
        if m1["mean_snr_ml"] < 10:
            reasons.append("SNR<10")
        if m1["mean_improvement_pct"] < 0:
            reasons.append("SNR degraded")
        if band_pct < 80:
            reasons.append("band pass<80%")
        if m2["mean_sam_rad"] >= 0.10 or m4["mean_r"] < 0.90:
            reasons.append("poor spectral preservation")
        if bd_fail:
            reasons.append(f"M3 fails={bd_fail}")
        if roi_fail:
            reasons.append(f"ROI fails={roi_fail}")
        warnings = []
        if roi_candidate_fail:
            warnings.append(f"unconfirmed ROI candidate fails={roi_candidate_fail}")

        if not reasons and rpt["overall_verdict"] == "PASS":
            readiness = "READY"
        elif (m1["mean_snr_ml"] >= 8 and m1["mean_improvement_pct"] >= 0 and
              m4["mean_r"] >= 0.95 and m2["mean_sam_rad"] < 0.10 and
              bd_fail == 0):
            readiness = "BORDERLINE"
        else:
            readiness = "NOT_READY"
        manifest.append({
            "scene": item["scene"],
            "fname": item.get("fname"),
            "readiness": readiness,
            "verdict": rpt["overall_verdict"],
            "mean_snr_ml": m1["mean_snr_ml"],
            "snr_gain_pct": m1["mean_improvement_pct"],
            "sam_deg": m2["mean_sam_deg"],
            "pearson": m4["mean_r"],
            "band_pass_pct": band_pct,
            "reasons": reasons,
            "warnings": warnings,
            "denoised_cube": npy,
            "validation_report": report_path,
        })
    manifest_path = JOURNAL_ROOT / "scene_readiness_manifest.json"
    with open(str(manifest_path), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n  Summary saved: {summ_path}")
    print(f"  Readiness manifest saved: {manifest_path}")
    print(f"  Total time  : {_hms(t_total)}")
    print("="*70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 Phase 1 Pipeline (Phase Architecture)")
    parser.add_argument("--scene",   type=str, default=None,
                        help="Single scene directory path")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retraining even if saved models exist")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip scenes that already have denoised .npy + validation report")
    parser.add_argument("--refresh-denoise", action="store_true",
                        help="Rerun Phase C denoising/validation while reusing cached Phase A and existing models")
    parser.add_argument("--exclude-scene", action="append", default=[],
                        help="Substring of scene/location path to exclude; can be repeated")
    args = parser.parse_args()

    target = [args.scene] if args.scene else None
    run_all(target_scenes=target, force_retrain=args.retrain,
            resume=args.resume, refresh_denoise=args.refresh_denoise,
            exclude_scenes=args.exclude_scene)
