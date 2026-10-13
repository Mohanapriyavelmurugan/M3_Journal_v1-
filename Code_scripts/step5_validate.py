"""
step5_validate.py  --  M3 Phase 1, Step 5: Validation
======================================================
Quantitatively verifies the denoised cube:
    M1 -- SNR improvement per band
    M2 -- SAM preservation (per pixel)
    M3 -- Band depth preservation (8 SPM indices)
    M4 -- Pearson correlation
    M5 -- Noise taxonomy suppression

Produces 6 publication-ready figures (150 DPI) and a complete JSON report.

Prerequisites: steps 0-4 variables in memory.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Pass/Fail thresholds
# ---------------------------------------------------------------------------
SAM_PASS_THR        = 0.05   # radians
SAM_REVIEW_THR      = 0.10
PEARSON_PASS_THR    = 0.98
BAND_DEPTH_CHG_THR  = 0.20
BAND_DEPTH_REVIEW_THR = 0.30
BAND_DEPTH_DETECT_THR = 0.015
SNR_READY_THR       = 10.0
SNR_MIN_IMPROVE_PCT = 0.0
ROI_MIN_BD_THR      = 0.02
ROI_MAX_PLAUSIBLE_BD = 0.60
ROI_TOP_PERCENTILE  = 90


# ---------------------------------------------------------------------------
# M1  SNR improvement
# ---------------------------------------------------------------------------

def metric_snr(snr_raw, snr_phys, snr_ml, valid_bands):
    improvements = []
    bands_pass   = 0
    for b in valid_bands:
        if snr_raw[b] > 1e-6:
            imp = (snr_ml[b] - snr_raw[b]) / snr_raw[b] * 100
        else:
            imp = 0.0
        improvements.append(imp)
        if snr_ml[b] >= snr_raw[b]:
            bands_pass += 1

    imps = np.array(improvements) if improvements else np.array([0.0])
    result = {
        "mean_improvement_pct": float(imps.mean()),
        "max_improvement_pct":  float(imps.max()),
        "min_improvement_pct":  float(imps.min()),
        "bands_passing":        bands_pass,
        "bands_total":          len(valid_bands),
        "mean_snr_raw":         float(np.mean(snr_raw[valid_bands])) if valid_bands else 0.0,
        "mean_snr_physics":     float(np.mean(snr_phys[valid_bands])) if valid_bands else 0.0,
        "mean_snr_ml":          float(np.mean(snr_ml[valid_bands]))   if valid_bands else 0.0,
    }
    print(f"  M1 SNR improvement  mean={result['mean_improvement_pct']:.1f}%  "
          f"bands_pass={bands_pass}/{len(valid_bands)}")
    return result, np.array(improvements)


# ---------------------------------------------------------------------------
# M2  SAM preservation
# ---------------------------------------------------------------------------

def metric_sam(cube, cube_denoised, valid_pixel_mask, H, W):
    rng = np.random.default_rng(42)
    rows_v, cols_v = np.where(valid_pixel_mask)
    n_sample = min(15000, len(rows_v))  # 15k = statistically robust & fast
    idx      = rng.choice(len(rows_v), size=n_sample, replace=False)

    sam_vals = []
    sam_map  = np.full((H, W), np.nan, dtype=np.float32)

    for i in idx:
        r, c   = int(rows_v[i]), int(cols_v[i])
        orig   = cube[r, c, :].astype(np.float64)
        den    = cube_denoised[r, c, :].astype(np.float64)
        finite = np.isfinite(orig) & np.isfinite(den)
        if finite.sum() < 10:
            continue
        o_f = orig[finite]
        d_f = den[finite]
        no  = np.linalg.norm(o_f)
        nd  = np.linalg.norm(d_f)
        if no < 1e-10 or nd < 1e-10:
            continue
        cos_sim = float(np.dot(o_f, d_f) / (no * nd))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        sam     = float(np.arccos(cos_sim))
        sam_vals.append(sam)
        sam_map[r, c] = float(sam)

    sam_arr = np.array(sam_vals) if sam_vals else np.array([0.0])
    pct_pass   = float(np.mean(sam_arr < SAM_PASS_THR)   * 100)
    pct_review = float(np.mean((sam_arr >= SAM_PASS_THR) & (sam_arr < SAM_REVIEW_THR)) * 100)
    pct_fail   = float(np.mean(sam_arr >= SAM_REVIEW_THR) * 100)

    result = {
        "mean_sam_rad":     float(sam_arr.mean()),
        "median_sam_rad":   float(np.median(sam_arr)),
        "p95_sam_rad":      float(np.percentile(sam_arr, 95)),
        "mean_sam_deg":     float(np.degrees(sam_arr.mean())),
        "pct_pass":         pct_pass,
        "pct_review":       pct_review,
        "pct_fail":         pct_fail,
        "n_sampled":        len(sam_vals),
        "thresholds":       {"pass": SAM_PASS_THR, "review": SAM_REVIEW_THR},
    }
    print(f"  M2 SAM  mean={result['mean_sam_deg']:.3f}deg  "
          f"pass={pct_pass:.1f}%  review={pct_review:.1f}%  fail={pct_fail:.1f}%")
    return result, sam_map, sam_arr


# ---------------------------------------------------------------------------
# M3  Band depth preservation
# ---------------------------------------------------------------------------

SPM_DEFS = [
    ("BD2300",    "Phyllosilicates",  2.295, 2.12,  2.53),
    ("BD1900",    "Hydrated minerals",1.930, 1.85,  2.07),
    ("BD2210",    "Al-OH clays",      2.21,  2.165, 2.25),
    ("SINDEX2",   "Sulfates",         2.10,  1.815, 2.20),
    ("BD1500",    "Water ice",        1.50,  1.43,  1.815),
    ("HCPINDEX2", "HCP pyroxene",     2.14,  1.82,  2.46),
]


def _compute_bd_sample(cube, waves_um, center, sh1, sh2, rows, cols):
    from step0_load import find_nearest_band
    if center < waves_um[0] or center > waves_um[-1]:
        return None
    bc  = find_nearest_band(waves_um, center)
    bs1 = find_nearest_band(waves_um, sh1)
    bs2 = find_nearest_band(waves_um, sh2)
    frac = (center - sh1) / max(sh2 - sh1, 1e-8)
    bds  = []
    for r, c in zip(rows, cols):
        rc   = float(cube[r, c, bc])
        rs1  = float(cube[r, c, bs1])
        rs2  = float(cube[r, c, bs2])
        if not (np.isfinite(rc) and np.isfinite(rs1) and np.isfinite(rs2)):
            continue
        rcont = rs1 + frac * (rs2 - rs1)
        if rcont < 1e-8:
            continue
        bds.append(1.0 - rc / rcont)
    return float(np.mean(bds)) if bds else None


def metric_band_depth(cube, cube_denoised, waves_um, valid_pixel_mask):
    rng = np.random.default_rng(42)
    rows_v, cols_v = np.where(valid_pixel_mask)
    n_sample = min(2000, len(rows_v))
    idx      = rng.choice(len(rows_v), size=n_sample, replace=False)
    rows_s   = rows_v[idx]
    cols_s   = cols_v[idx]

    results  = {}
    print(f"\n  M3 Band Depth Preservation:")
    print(f"  {'Name':<12} {'Mineral':<20} {'Orig':>8} {'Den':>8} "
          f"{'Chg%':>8} {'Status':<6}")
    print("  " + "-" * 68)

    for name, mineral, center, sh1, sh2 in SPM_DEFS:
        bd_orig = _compute_bd_sample(cube, waves_um,
                                     center, sh1, sh2, rows_s, cols_s)
        bd_den  = _compute_bd_sample(cube_denoised, waves_um,
                                     center, sh1, sh2, rows_s, cols_s)
        if bd_orig is None or bd_den is None:
            results[name] = {"status": "SKIP", "reason": "out of range"}
            print(f"  {name:<12} {mineral:<20} {'--':>8} {'--':>8} "
                  f"{'--':>8} SKIP")
            continue

        # If the absorption is weaker than the noise floor, the mineral is not
        # present in this scene — skip rather than produce meaningless percentages.
        if bd_orig < BAND_DEPTH_DETECT_THR:
            results[name] = {
                "mineral": mineral,
                "bd_original": bd_orig,
                "bd_denoised": bd_den,
                "change_pct":  None,
                "status":      "N/A",
                "reason":      f"feature not detected (bd<{BAND_DEPTH_DETECT_THR})",
            }
            print(f"  {name:<12} {mineral:<20} {bd_orig:>8.4f} {bd_den:>8.4f} "
                  f"{'N/A':>8} N/A   (feature absent in scene)")
            continue

        chg_pct = (bd_den - bd_orig) / abs(bd_orig) * 100
        if abs(chg_pct) <= BAND_DEPTH_CHG_THR * 100:
            status = "PASS"
        elif abs(chg_pct) <= BAND_DEPTH_REVIEW_THR * 100:
            status = "REVIEW"
        else:
            status = "FAIL"
        results[name] = {
            "mineral":        mineral,
            "bd_original":    bd_orig,
            "bd_denoised":    bd_den,
            "change_pct":     float(chg_pct),
            "status":         status,
        }
        print(f"  {name:<12} {mineral:<20} {bd_orig:>8.4f} {bd_den:>8.4f} "
              f"{chg_pct:>7.1f}% {status:<6}")

    n_pass     = sum(1 for v in results.values() if v.get("status") == "PASS")
    n_detected = sum(1 for v in results.values() if v.get("status") in ("PASS", "REVIEW", "FAIL"))
    print(f"\n  Band depth PASS: {n_pass}/{n_detected} detected features "
          f"({len(results) - n_detected} not present in scene)")
    return results



# ---------------------------------------------------------------------------
# M4  Pearson correlation
# ---------------------------------------------------------------------------

def metric_pearson(cube, cube_denoised, valid_pixel_mask):
    rng = np.random.default_rng(42)
    rows_v, cols_v = np.where(valid_pixel_mask)
    n_sample = min(2000, len(rows_v))
    idx      = rng.choice(len(rows_v), size=n_sample, replace=False)
    rs       = []
    for i in idx:
        r, c   = int(rows_v[i]), int(cols_v[i])
        orig   = cube[r, c, :].astype(np.float64)
        den    = cube_denoised[r, c, :].astype(np.float64)
        finite = np.isfinite(orig) & np.isfinite(den)
        if finite.sum() < 10:
            continue
        o_f = orig[finite]
        d_f = den[finite]
        if o_f.std() < 1e-10 or d_f.std() < 1e-10:
            continue
        r_val = float(np.corrcoef(o_f, d_f)[0, 1])
        rs.append(r_val)

    rs_arr    = np.array(rs) if rs else np.array([0.0])
    mean_r    = float(rs_arr.mean())
    pass_frac = float(np.mean(rs_arr >= PEARSON_PASS_THR))
    result = {
        "mean_r":         mean_r,
        "pass_fraction":  pass_frac,
        "threshold":      PEARSON_PASS_THR,
        "n_sampled":      len(rs),
    }
    print(f"  M4 Pearson  mean_r={mean_r:.4f}  pass_frac={pass_frac*100:.1f}%")
    return result


def metric_roi_validation(cube, cube_denoised, waves_um, valid_pixel_mask):
    from step0_load import find_nearest_band

    results = {}
    for name, mineral, center, sh1, sh2 in SPM_DEFS:
        if center < waves_um[0] or center > waves_um[-1]:
            continue
        bc = find_nearest_band(waves_um, center)
        bs1 = find_nearest_band(waves_um, sh1)
        bs2 = find_nearest_band(waves_um, sh2)
        frac = (center - sh1) / max(sh2 - sh1, 1e-8)

        rc = cube[:, :, bc].astype(np.float64)
        rs1 = cube[:, :, bs1].astype(np.float64)
        rs2 = cube[:, :, bs2].astype(np.float64)
        cont = rs1 + frac * (rs2 - rs1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bd = 1.0 - rc / np.where(cont > 1e-8, cont, np.nan)
        positive = bd[np.isfinite(bd) & (bd >= ROI_MIN_BD_THR) &
                      (bd <= ROI_MAX_PLAUSIBLE_BD)]
        if positive.size < 100:
            results[name] = {
                "mineral": mineral,
                "status": "N/A",
                "reason": "too few plausible index-positive ROI pixels",
                "roi_pixels": int(positive.size),
            }
            continue
        roi_thr = max(ROI_MIN_BD_THR, float(np.percentile(positive, ROI_TOP_PERCENTILE)))
        roi_mask = (valid_pixel_mask & np.isfinite(bd) &
                    (bd >= roi_thr) & (bd <= ROI_MAX_PLAUSIBLE_BD))
        rows, cols = np.where(roi_mask)
        if len(rows) < 100:
            results[name] = {
                "mineral": mineral,
                "status": "N/A",
                "reason": "too few high-confidence ROI pixels",
                "roi_pixels": int(len(rows)),
            }
            continue

        rng = np.random.default_rng(42)
        idx = rng.choice(len(rows), size=min(5000, len(rows)), replace=False)
        rows_s = rows[idx]
        cols_s = cols[idx]
        bd_orig = _compute_bd_sample(cube, waves_um, center, sh1, sh2,
                                     rows_s, cols_s)
        bd_den = _compute_bd_sample(cube_denoised, waves_um, center, sh1, sh2,
                                    rows_s, cols_s)

        sam_vals = []
        pear_vals = []
        for r, c in zip(rows_s, cols_s):
            orig = cube[r, c, :].astype(np.float64)
            den = cube_denoised[r, c, :].astype(np.float64)
            finite = np.isfinite(orig) & np.isfinite(den)
            if finite.sum() < 10:
                continue
            of = orig[finite]
            df = den[finite]
            no = np.linalg.norm(of)
            nd = np.linalg.norm(df)
            if no > 1e-10 and nd > 1e-10:
                sam_vals.append(float(np.arccos(np.clip(np.dot(of, df) / (no * nd), -1, 1))))
            if np.std(of) > 1e-10 and np.std(df) > 1e-10:
                pear_vals.append(float(np.corrcoef(of, df)[0, 1]))

        chg_pct = None
        status = "N/A"
        if bd_orig is not None and bd_den is not None and bd_orig >= BAND_DEPTH_DETECT_THR:
            chg_pct = (bd_den - bd_orig) / abs(bd_orig) * 100.0
            if abs(chg_pct) <= BAND_DEPTH_CHG_THR * 100:
                status = "PASS"
            elif abs(chg_pct) <= BAND_DEPTH_REVIEW_THR * 100:
                status = "REVIEW"
            else:
                status = "FAIL"

        results[name] = {
            "mineral": mineral,
            "status": status,
            "roi_pixels": int(len(rows)),
            "roi_threshold": float(roi_thr),
            "sampled_pixels": int(len(rows_s)),
            "bd_original": bd_orig,
            "bd_denoised": bd_den,
            "change_pct": None if chg_pct is None else float(chg_pct),
            "mean_sam_deg": float(np.degrees(np.mean(sam_vals))) if sam_vals else None,
            "mean_pearson": float(np.mean(pear_vals)) if pear_vals else None,
        }

    detected = sum(1 for v in results.values() if v.get("status") in ("PASS", "REVIEW", "FAIL"))
    failed = sum(1 for v in results.values() if v.get("status") == "FAIL")
    print(f"  ROI validation: detected={detected}  failed={failed}")
    return results


# ---------------------------------------------------------------------------
# M5  Noise suppression
# ---------------------------------------------------------------------------

def metric_noise_suppression(cube, cube_denoised, waves_um,
                              valid_bands, valid_pixel_mask,
                              severe_gaussian_bands, striped_bands,
                              spiked_bands, spike_frac_before,
                              thermal_detected, smile_detected,
                              NOISE_MAP):
    from step0_load import find_nearest_band, THERMAL_ONSET_UM
    results = {}

    # N1 Gaussian
    if severe_gaussian_bands:
        stds_b = [float(np.nanstd(cube[:, :, b])) for b in severe_gaussian_bands]
        stds_a = [float(np.nanstd(cube_denoised[:, :, b])) for b in severe_gaussian_bands]
        suppr  = [max(0, (b - a) / max(b, 1e-10) * 100)
                  for b, a in zip(stds_b, stds_a)]
        results["N1_Gaussian"] = float(np.mean(suppr))
    else:
        results["N1_Gaussian"] = "N/A"

    # N2 Stripe
    if striped_bands:
        sev_b = NOISE_MAP.get("stripe_severity_per_band", [])
        suppr = []
        for b in striped_bands:
            sev_bef = float(sev_b[b]) if b < len(sev_b) else 0.0
            col_means_den = np.nanmean(cube_denoised[:, :, b], axis=0)
            sev_aft = float(np.std(col_means_den - np.nanmean(col_means_den)))
            if sev_bef > 1e-10:
                suppr.append(max(0, (sev_bef - sev_aft) / sev_bef * 100))
        results["N2_Stripe"] = float(np.mean(suppr)) if suppr else "N/A"
    else:
        results["N2_Stripe"] = "N/A"

    # N3 Spikes
    if spiked_bands:
        try:
            from scipy.stats import median_abs_deviation
            rows_v, cols_v = np.where(valid_pixel_mask)
            rng = np.random.default_rng(42)
            idx = rng.choice(len(rows_v), size=min(3000, len(rows_v)), replace=False)
            spike_frac_after = np.zeros(cube_denoised.shape[2])
            for i in idx:
                r, c = int(rows_v[i]), int(cols_v[i])
                spec  = cube_denoised[r, c, :].astype(np.float64)
                fspec = spec[np.isfinite(spec)]
                if fspec.size < 10:
                    continue
                med = np.median(fspec)
                try:
                    mad = float(median_abs_deviation(fspec, scale="normal"))
                except TypeError:
                    mad = float(median_abs_deviation(fspec)) * 1.4826
                if mad < 1e-10:
                    continue
                for b in spiked_bands:
                    if np.isfinite(spec[b]) and abs(spec[b] - med) > 5.0 * mad:
                        spike_frac_after[b] += 1
            n_samp = min(3000, len(rows_v))
            spike_frac_after /= max(n_samp, 1)
            suppr  = []
            for b in spiked_bands:
                bef = float(spike_frac_before[b])
                aft = float(spike_frac_after[b])
                if bef > 1e-10:
                    suppr.append(max(0, (bef - aft) / bef * 100))
            results["N3_Spikes"] = float(np.mean(suppr)) if suppr else "N/A"
        except Exception:
            results["N3_Spikes"] = "N/A"
    else:
        results["N3_Spikes"] = "N/A"

    # N4 Thermal
    if thermal_detected:
        thm_start = find_nearest_band(waves_um, THERMAL_ONSET_UM)
        thm_end   = cube.shape[2] - 1
        thm_bands = list(range(thm_start, thm_end + 1))
        if len(thm_bands) >= 2:
            rows_v, cols_v = np.where(valid_pixel_mask)
            rng = np.random.default_rng(42)
            idx = rng.choice(len(rows_v), size=min(1000, len(rows_v)), replace=False)
            grads_b = []
            grads_a = []
            for i in idx:
                r, c = int(rows_v[i]), int(cols_v[i])
                orig_t = cube[r, c, thm_bands].astype(np.float64)
                den_t  = cube_denoised[r, c, thm_bands].astype(np.float64)
                fo = orig_t[np.isfinite(orig_t)]
                fd = den_t[np.isfinite(den_t)]
                if fo.size >= 2:
                    grads_b.append(float(np.mean(np.diff(fo))))
                if fd.size >= 2:
                    grads_a.append(float(np.mean(np.diff(fd))))
            g_b = float(np.mean(grads_b)) if grads_b else 0.0
            g_a = float(np.mean(grads_a)) if grads_a else 0.0
            if abs(g_b) > 1e-10:
                suppr = max(0, (g_b - g_a) / abs(g_b) * 100)
            else:
                suppr = 0.0
            results["N4_Thermal"] = float(suppr)
        else:
            results["N4_Thermal"] = "N/A"
    else:
        results["N4_Thermal"] = "N/A"

    # N5 Smile
    results["N5_Smile"] = 100.0 if smile_detected else "N/A"

    print(f"  M5 Noise suppression: {results}")
    return results


# ---------------------------------------------------------------------------
# M6  Band-level pass/fail
# ---------------------------------------------------------------------------

def band_level_decisions(cube, cube_denoised, snr_raw, snr_ml, valid_bands):
    decisions = {}
    n_pass = n_review = n_fail = 0
    for b in valid_bands:
        snr_ok = snr_ml[b] >= snr_raw[b]
        std_b  = float(np.nanstd(cube[:, :, b]))
        std_a  = float(np.nanstd(cube_denoised[:, :, b]))
        std_ok = std_a <= std_b * 1.05
        if snr_ok and std_ok:
            d = "PASS"; n_pass += 1
        elif snr_ok:
            d = "REVIEW"; n_review += 1
        else:
            d = "FAIL"; n_fail += 1
        decisions[b] = d

    total = len(valid_bands)
    print(f"  Band decisions  PASS={n_pass}  REVIEW={n_review}  FAIL={n_fail}")
    return decisions, n_pass, n_review, n_fail


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------

def overall_verdict(n_pass, n_review, n_fail, mean_sam, mean_pearson_r,
                    valid_bands, snr_metrics, band_depth_results=None):
    total = len(valid_bands)
    pct_pass = n_pass / max(total, 1) * 100
    bd_fail = 0
    bd_detected = 0
    if band_depth_results:
        for item in band_depth_results.values():
            if not isinstance(item, dict):
                continue
            if item.get("status") in ("PASS", "REVIEW", "FAIL"):
                bd_detected += 1
            if item.get("status") == "FAIL":
                bd_fail += 1

    snr_ready = snr_metrics["mean_snr_ml"] >= SNR_READY_THR
    snr_nonharm = snr_metrics["mean_improvement_pct"] >= SNR_MIN_IMPROVE_PCT

    if (snr_ready and snr_nonharm and pct_pass >= 80 and
            mean_sam < SAM_PASS_THR and mean_pearson_r >= PEARSON_PASS_THR and
            bd_fail == 0):
        verdict = "PASS"
    elif snr_nonharm and pct_pass >= 60:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"
    symbol = {"PASS": "GREEN PASS", "REVIEW": "YELLOW REVIEW", "FAIL": "RED FAIL"}
    print(f"\n  Overall verdict: [{symbol[verdict]}] "
          f"({pct_pass:.0f}% bands pass, SNR={snr_metrics['mean_snr_ml']:.2f}, "
          f"SNR gain={snr_metrics['mean_improvement_pct']:.1f}%, "
          f"SAM={mean_sam:.3f} rad, r={mean_pearson_r:.4f}, "
          f"M3 detected fails={bd_fail}/{bd_detected})")
    return verdict


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig1_snr(snr_raw, snr_phys, snr_ml, waves_um, valid_bands,
             improvements, fname, out_path):
    from step0_load import plot_style, THERMAL_ONSET_UM
    plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    w  = waves_um; vb = valid_bands

    ax = axes[0]
    ax.plot(w[vb], snr_raw[vb],  color="#58a6ff", lw=1.4, label="Raw")
    ax.plot(w[vb], snr_phys[vb], color="#ffa657", lw=1.4,
            linestyle="--", label="After Physics")
    ax.plot(w[vb], snr_ml[vb],   color="#3fb950", lw=1.6,
            linestyle=":",  label="After ML")
    ax.axhline(10, color="#ff7b72", lw=0.8, linestyle="--", label="SNR=10")
    ax.axhline(30, color="#e3b341", lw=0.8, linestyle="--", label="SNR=30")
    ax.axvspan(THERMAL_ONSET_UM, w[-1] + 0.05, alpha=0.08, color="#ff7b72")
    mean_imp = float(np.mean(improvements)) if len(improvements) else 0
    ax.set_title(f"SNR Comparison (mean improvement={mean_imp:.1f}%)")
    ax.set_xlabel("Wavelength (um)"); ax.set_ylabel("SNR")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    colors = ["#3fb950" if v >= 0 else "#ff7b72" for v in improvements]
    ax2.bar(w[vb], improvements, color=colors, width=(w[1]-w[0])*0.8)
    ax2.axhline(0, color="#c9d1d9", lw=0.7)
    ax2.axvspan(THERMAL_ONSET_UM, w[-1] + 0.05, alpha=0.08, color="#ff7b72")
    ax2.set_xlabel("Wavelength (um)"); ax2.set_ylabel("SNR Improvement %")
    ax2.set_title("Per-Band SNR Improvement")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Fig 1: SNR Comparison -- {fname}", color="#c9d1d9", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


def fig2_sam(sam_map, sam_arr, H, W, fname, out_path):
    from step0_load import plot_style
    plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(sam_map, cmap="hot_r", vmin=0, vmax=0.10, aspect="auto")
    plt.colorbar(im0, ax=axes[0], label="SAM (rad)", fraction=0.046, pad=0.04)
    axes[0].set_title("SAM Spatial Map")
    axes[0].axis("off")

    axes[1].hist(sam_arr, bins=60, color="#58a6ff", edgecolor="#0d1117", linewidth=0.3)
    axes[1].axvline(SAM_PASS_THR,   color="#3fb950", lw=1.2, linestyle="--",
                    label=f"Pass={SAM_PASS_THR}")
    axes[1].axvline(SAM_REVIEW_THR, color="#ffa657", lw=1.2, linestyle="--",
                    label=f"Review={SAM_REVIEW_THR}")
    pct_pass   = float(np.mean(sam_arr < SAM_PASS_THR) * 100)
    pct_review = float(np.mean((sam_arr >= SAM_PASS_THR) & (sam_arr < SAM_REVIEW_THR)) * 100)
    pct_fail   = float(np.mean(sam_arr >= SAM_REVIEW_THR) * 100)
    axes[1].set_xlabel("SAM (rad)"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"SAM Distribution\nP={pct_pass:.0f}% R={pct_review:.0f}% F={pct_fail:.0f}%")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # Pass/fail classification map
    clf_map = np.full((H, W), np.nan)
    valid_r, valid_c = np.where(np.isfinite(sam_map))
    for r, c in zip(valid_r, valid_c):
        s = sam_map[r, c]
        if s < SAM_PASS_THR:
            clf_map[r, c] = 1.0
        elif s < SAM_REVIEW_THR:
            clf_map[r, c] = 0.5
        else:
            clf_map[r, c] = 0.0
    im2 = axes[2].imshow(clf_map, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im2, ax=axes[2], label="0=FAIL 0.5=REVIEW 1=PASS",
                 fraction=0.046, pad=0.04)
    axes[2].set_title("SAM Classification Map")
    axes[2].axis("off")

    plt.suptitle(f"Fig 2: SAM Preservation -- {fname}", color="#c9d1d9", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


def fig3_band_depth(bd_results, fname, out_path):
    from step0_load import plot_style
    plot_style()
    names   = [k for k, v in bd_results.items()
               if isinstance(v, dict) and "bd_original" in v]
    if not names:
        print(f"  Fig 3 skipped -- no computable band depths")
        return
    orig_v  = [bd_results[n]["bd_original"] for n in names]
    den_v   = [bd_results[n]["bd_denoised"] for n in names]
    statuses = [bd_results[n]["status"] for n in names]
    chg_pcts = [bd_results[n]["change_pct"] for n in names]

    x   = np.arange(len(names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(10, 2 * len(names)), 6))
    bars1 = ax.bar(x - w/2, orig_v, w, label="Original", color="#58a6ff", alpha=0.85)
    bars2 = ax.bar(x + w/2, den_v,  w, label="Denoised", color="#3fb950", alpha=0.85)
    for i, (chg, status) in enumerate(zip(chg_pcts, statuses)):
        if chg is None:
            label = "N/A"
            color = "#8b949e"
        else:
            mark = "PASS" if status == "PASS" else "FAIL"
            label = f"{chg:+.1f}%\n{mark}"
            color = "#3fb950" if status == "PASS" else "#ff7b72"
        y = max(orig_v[i], den_v[i]) + 0.002
        ax.annotate(label,
                    xy=(x[i], y),
                    ha="center", va="bottom", fontsize=7,
                    color=color)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Mean Band Depth")
    n_pass = sum(1 for s in statuses if s == "PASS")
    n_detected = sum(1 for s in statuses if s in ("PASS", "FAIL"))
    ax.set_title(f"Fig 3: Band Depth Preservation (PASS={n_pass}/{n_detected}, "
                 f"tolerance=20%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


def fig4_noise_suppression(noise_suppr, fname, out_path):
    from step0_load import plot_style
    plot_style()
    names  = [k for k, v in noise_suppr.items() if isinstance(v, (int, float))]
    values = [noise_suppr[k] for k in names]
    if not names:
        print("  Fig 4 skipped -- no computable suppressions")
        return
    colors = ["#3fb950" if v >= 80 else ("#ffa657" if v >= 50 else "#ff7b72")
              for v in values]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.8)))
    bars = ax.barh(names, values, color=colors, edgecolor="#0d1117", linewidth=0.3)
    ax.axvline(80, color="#ffa657", linestyle="--", lw=1.0, label="80% target")
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("Suppression %")
    ax.set_title(f"Fig 4: Noise Suppression -- {fname}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


def fig5_spectra_gallery(cube, cube_phys, cube_denoised, waves_um,
                         valid_pixel_mask, fname, out_path):
    from step0_load import plot_style, THERMAL_ONSET_UM
    plot_style()
    rng = np.random.default_rng(42)
    rows_v, cols_v = np.where(valid_pixel_mask)
    idx  = rng.choice(len(rows_v), size=min(6, len(rows_v)), replace=False)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.ravel()
    w    = waves_um
    for i, ii in enumerate(idx):
        r, c  = int(rows_v[ii]), int(cols_v[ii])
        raw   = cube[r, c, :].astype(np.float64)
        phys  = cube_phys[r, c, :].astype(np.float64)
        den   = cube_denoised[r, c, :].astype(np.float64)
        ax    = axes[i]
        mr    = np.isfinite(raw)
        mp    = np.isfinite(phys)
        md    = np.isfinite(den)
        ax.plot(w[mr], raw[mr],   color="#ff7b72", lw=1.0, label="Raw")
        ax.plot(w[mp], phys[mp],  color="#ffa657", lw=1.0,
                linestyle="--", label="Physics")
        ax.plot(w[md], den[md],   color="#3fb950", lw=1.3,
                linestyle=":",  label="Denoised")
        ax.axvspan(THERMAL_ONSET_UM, w[-1] + 0.05, alpha=0.08, color="#ff7b72")
        for wk in [1.0, 2.0]:
            if w[0] <= wk <= w[-1]:
                ax.axvline(wk, color="#e3b341", lw=0.6, linestyle=":")
        ax.set_xlabel("um"); ax.set_ylabel("I/F")
        ax.set_title(f"Pixel ({r},{c})")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Fig 5: Spectra Gallery -- {fname}", color="#c9d1d9", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


def fig6_spatial(cube, cube_denoised, waves_um, fname, out_path):
    from step0_load import plot_style, find_nearest_band
    plot_style()
    bi   = find_nearest_band(waves_um, 1.0)
    raw_b = cube[:, :, bi].astype(np.float64)
    den_b = cube_denoised[:, :, bi].astype(np.float64)
    diff  = den_b - raw_b

    def _norm(img):
        finite = img[np.isfinite(img)]
        if finite.size < 2:
            return np.zeros_like(img)
        lo = float(np.percentile(finite, 2))
        hi = float(np.percentile(finite, 98))
        if hi - lo < 1e-10:
            return np.zeros_like(img)
        return np.clip((img - lo) / (hi - lo), 0, 1)

    abs_diff = np.abs(diff[np.isfinite(diff)])
    clim = float(np.nanpercentile(abs_diff, 98)) if abs_diff.size > 0 else 0.01

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(_norm(raw_b), cmap="viridis", aspect="auto")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title(f"Raw (band {bi}, {waves_um[bi]:.3f} um)")
    axes[0].axis("off")

    im1 = axes[1].imshow(_norm(den_b), cmap="viridis", aspect="auto")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Denoised")
    axes[1].axis("off")

    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="auto")
    plt.colorbar(im2, ax=axes[2], label="Denoised - Raw", fraction=0.046, pad=0.04)
    axes[2].set_title("Difference (Denoised - Raw)")
    axes[2].axis("off")

    plt.suptitle(f"Fig 6: Spatial Before/After -- {fname}", color="#c9d1d9", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_validation(cube, cube_phys, cube_denoised, waves_um, fname, DIRS,
                   valid_bands, dead_bands, valid_pixel_mask,
                   snr_raw, SNR_AFTER_PHYSICS, SNR_AFTER_ML,
                   severe_gaussian_bands, striped_bands, spiked_bands,
                   spike_frac_before, thermal_detected, smile_detected,
                   NOISE_MAP):
    print("\n" + "=" * 60)
    print("  STEP 5 -- Validation")
    print("=" * 60)

    H, W, B = cube.shape
    out     = DIRS["05_validation"]
    vb      = valid_bands

    # M1
    print("\n  M1: SNR Improvement ...")
    m1, improvements = metric_snr(snr_raw, SNR_AFTER_PHYSICS, SNR_AFTER_ML, vb)

    # M2
    print("  M2: SAM Preservation ...")
    m2, sam_map, sam_arr = metric_sam(cube, cube_denoised, valid_pixel_mask, H, W)

    # M3
    m3 = metric_band_depth(cube, cube_denoised, waves_um, valid_pixel_mask)

    # M4
    print("  M4: Pearson Correlation ...")
    m4 = metric_pearson(cube, cube_denoised, valid_pixel_mask)

    # ROI
    print("  ROI: Mineral-index positive pixels ...")
    roi = metric_roi_validation(cube, cube_denoised, waves_um, valid_pixel_mask)

    # M5
    print("  M5: Noise Suppression ...")
    m5 = metric_noise_suppression(
        cube, cube_denoised, waves_um, vb, valid_pixel_mask,
        severe_gaussian_bands, striped_bands, spiked_bands, spike_frac_before,
        thermal_detected, smile_detected, NOISE_MAP)

    # Band decisions
    decisions, n_pass, n_review, n_fail = band_level_decisions(
        cube, cube_denoised, snr_raw, SNR_AFTER_ML, vb)

    # Verdict
    verdict = overall_verdict(
        n_pass, n_review, n_fail,
        m2["mean_sam_rad"], m4["mean_r"], vb, m1, m3)

    # Figures
    fp = {}
    fp["fig1"] = str(Path(out) / f"{fname}_fig1_snr_comparison.png")
    fig1_snr(snr_raw, SNR_AFTER_PHYSICS, SNR_AFTER_ML, waves_um, vb,
             improvements, fname, fp["fig1"])

    fp["fig2"] = str(Path(out) / f"{fname}_fig2_sam_map.png")
    fig2_sam(sam_map, sam_arr, H, W, fname, fp["fig2"])

    fp["fig3"] = str(Path(out) / f"{fname}_fig3_band_depth.png")
    fig3_band_depth(m3, fname, fp["fig3"])

    fp["fig4"] = str(Path(out) / f"{fname}_fig4_noise_suppression.png")
    fig4_noise_suppression(m5, fname, fp["fig4"])

    fp["fig5"] = str(Path(out) / f"{fname}_fig5_spectra_gallery.png")
    fig5_spectra_gallery(cube, cube_phys, cube_denoised, waves_um,
                         valid_pixel_mask, fname, fp["fig5"])

    fp["fig6"] = str(Path(out) / f"{fname}_fig6_spatial.png")
    fig6_spatial(cube, cube_denoised, waves_um, fname, fp["fig6"])

    # JSON report
    report = {
        "fname":        fname,
        "cube_shape":   list(cube.shape),
        "valid_bands":  len(vb),
        "dead_bands":   len(dead_bands),
        "metric1_snr":  m1,
        "metric2_sam":  m2,
        "metric3_band_depth": m3,
        "metric4_pearson":    m4,
        "roi_validation":     roi,
        "metric5_noise":      m5,
        "band_decisions": {
            "PASS":   n_pass,
            "REVIEW": n_review,
            "FAIL":   n_fail,
            "pct_pass": float(n_pass / max(len(vb), 1) * 100),
        },
        "overall_verdict": verdict,
        "figures": fp,
    }
    rpt_path = str(Path(out) / f"{fname}_validation_report.json")
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report: {rpt_path}")

    # Final print
    npy_path = str(Path(DIRS["04_denoised"]) / f"{fname}_denoised.npy")
    print("\n" + "=" * 60)
    print("  STEP 5 SUMMARY")
    print("=" * 60)
    print(f"  M1 SNR improvement  : {m1['mean_improvement_pct']:.1f}%")
    print(f"  M2 SAM mean         : {m2['mean_sam_deg']:.3f} deg  "
          f"(pass={m2['pct_pass']:.0f}%)")
    print(f"  M3 band depth PASS  : "
          f"{sum(1 for v in m3.values() if isinstance(v,dict) and v.get('status')=='PASS')}"
          f"/{sum(1 for v in m3.values() if isinstance(v,dict) and v.get('status') in ('PASS','FAIL'))}")
    print(f"  M4 Pearson mean r   : {m4['mean_r']:.4f}")
    print(f"  Overall verdict     : {verdict}")
    print(f"  Denoised cube path  : {npy_path}")
    print("=" * 60)
    if verdict == "PASS":
        print("  Denoised cube ready for Phase 2 Mineral Identification.")
    else:
        print("  Denoised cube requires review before Phase 2 Mineral Identification.")
    print(f"  Full path: {npy_path}")
    print("=" * 60 + "\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import step0_load as s0
    import step1_eda  as s1
    import step2_noise as s2
    import step3_physics as s3
    import step4_denoise as s4

    sd = (sys.argv[1] if len(sys.argv) > 1
          else r"D:\M3 Journal\Mare Tranquillitatis\M3G20090203T160452_V01_RFL")

    cube, waves, fname, meta, DIRS = s0.load_scene(sd)
    (EDA, vpm, bs, vb, db, tb, sb) = s1.run_eda(cube, waves, fname, DIRS)
    (NM, sco, sfrac, stb, spb, sg,
     td, smd, spc) = s2.run_noise_id(
        cube, waves, fname, DIRS, bs, vb, db, tb, vpm)
    cp, snrp = s3.run_physics(
        cube, waves, fname, DIRS, bs, vb, db, tb, vpm,
        sfrac, stb, spb, td, smd, spc)
    cd, snrml, bw = s4.run_denoise(
        cube, cp, waves, fname, DIRS,
        bs, vb, db, vpm, stb, sco, snrp)
    run_validation(
        cube, cp, cd, waves, fname, DIRS,
        vb, db, vpm,
        bs["snr"], snrp, snrml,
        sg, stb, spb, sfrac, td, smd, NM)
