"""
step2_noise.py  --  M3 Phase 1, Step 2: Noise Identification
=============================================================
Prerequisites: step0 + step1 must have run (cube, waves, fname, DIRS,
    EDA, valid_pixel_mask, band_stats, valid_bands, dead_bands,
    thermal_bands, suspicious_bands in memory).

Produces:
    02_noise/{fname}_noise_characterisation.png
    02_noise/{fname}_noise_map.json

Usage (standalone):
    python step2_noise.py "D:/M3 Journal/Mare Tranquillitatis/M3G20090203T160452_V01_RFL"
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
SNR_HIGH_NOISE_THR   = 10.0
SNR_MOD_NOISE_THR    = 30.0
STRIPE_STD_FACTOR    = 1.5
MAD_SPIKE_FACTOR     = 5.0
THERMAL_GRAD_THR     = 0.001
SMILE_SHIFT_THR_NM   = 2.0


# ---------------------------------------------------------------------------
# N1  Gaussian noise
# ---------------------------------------------------------------------------

def detect_gaussian_noise(snr, valid_bands):
    severe   = [b for b in valid_bands if snr[b] < SNR_HIGH_NOISE_THR]
    moderate = [b for b in valid_bands
                if SNR_HIGH_NOISE_THR <= snr[b] < SNR_MOD_NOISE_THR]
    print(f"  N1 Gaussian -- severe (<10): {len(severe)}  moderate (10-30): {len(moderate)}")
    return severe, moderate


# ---------------------------------------------------------------------------
# N2  Stripe noise
# ---------------------------------------------------------------------------

def detect_stripe_noise(cube, valid_bands, dead_bands):
    H, W, B = cube.shape
    stripe_severity  = np.zeros(B, dtype=np.float64)
    stripe_col_offsets = {}

    for b in valid_bands:
        band = cube[:, :, b].astype(np.float64)
        col_means  = np.nanmean(band, axis=0)          # (W,)
        global_mean = np.nanmean(col_means)
        offsets    = col_means - global_mean
        sev        = float(np.nanstd(offsets))
        stripe_severity[b]   = sev
        stripe_col_offsets[b] = offsets.tolist()

    valid_sev = stripe_severity[valid_bands]
    med_sev   = float(np.median(valid_sev[valid_sev > 0])) if np.any(valid_sev > 0) else 1e-8
    threshold = STRIPE_STD_FACTOR * med_sev
    striped   = [b for b in valid_bands if stripe_severity[b] > threshold]

    print(f"  N2 Stripe -- threshold={threshold:.6f}  striped bands: {len(striped)}")
    return striped, stripe_severity, stripe_col_offsets, float(threshold)


# ---------------------------------------------------------------------------
# N3  Spectral spikes
# ---------------------------------------------------------------------------

def detect_spectral_spikes(cube, valid_pixel_mask, valid_bands):
    from scipy.stats import median_abs_deviation

    H, W, B = cube.shape
    rows, cols = np.where(valid_pixel_mask)
    n_valid = len(rows)

    # Sample up to 20% or 500k pixels
    n_sample = min(n_valid, max(int(0.20 * n_valid), 500_000))
    n_sample = min(n_sample, n_valid)
    rng = np.random.default_rng(42)
    idx = rng.choice(n_valid, size=n_sample, replace=False)

    spike_count = np.zeros(B, dtype=np.int64)
    total_spikes = 0

    for i in idx:
        r, c = int(rows[i]), int(cols[i])
        spec = cube[r, c, :].astype(np.float64)
        valid_mask_spec = np.isfinite(spec) & np.array(
            [b in valid_bands for b in range(B)])
        if valid_mask_spec.sum() < 10:
            continue
        spec_vals = spec[valid_mask_spec]
        med = float(np.median(spec_vals))
        # MAD with normal scale
        try:
            mad = float(median_abs_deviation(spec_vals, scale="normal"))
        except TypeError:
            mad = float(median_abs_deviation(spec_vals)) * 1.4826
        if mad < 1e-10:
            continue
        for b in valid_bands:
            if not np.isfinite(spec[b]):
                continue
            if abs(spec[b] - med) > MAD_SPIKE_FACTOR * mad:
                spike_count[b] += 1
                total_spikes   += 1

    spike_frac = spike_count.astype(np.float64) / max(n_sample, 1)
    spiked_bands = [b for b in valid_bands if spike_frac[b] > 0.002]

    print(f"  N3 Spikes -- sampled={n_sample:,}  total spikes={total_spikes:,}  "
          f"spiked bands={len(spiked_bands)}")
    return spike_frac, spiked_bands, total_spikes, n_sample


# ---------------------------------------------------------------------------
# N4  Thermal emission
# ---------------------------------------------------------------------------

def detect_thermal_emission(cube, waves_um, valid_pixel_mask, valid_bands):
    from step0_load import find_nearest_band

    pre_start = find_nearest_band(waves_um, 2.0)
    pre_end   = find_nearest_band(waves_um, 2.5)
    thm_start = find_nearest_band(waves_um, 2.5)
    thm_end   = min(find_nearest_band(waves_um, 2.8), cube.shape[2] - 1)

    thm_valid = [b for b in range(thm_start, thm_end+1) if b in valid_bands]
    if len(thm_valid) < 2:
        print("  N4 Thermal -- fewer than 2 thermal bands -- skipped")
        return False, 0.0, 0.0, []

    rows, cols = np.where(valid_pixel_mask)
    n_sample   = min(3000, len(rows))
    rng        = np.random.default_rng(42)
    idx        = rng.choice(len(rows), size=n_sample, replace=False)

    gradients = []
    n_rising  = 0
    for i in idx:
        r, c = int(rows[i]), int(cols[i])
        thm_vals = cube[r, c, thm_valid].astype(np.float64)
        finite = thm_vals[np.isfinite(thm_vals)]
        if finite.size < 2:
            continue
        g = float(np.mean(np.diff(finite)))
        gradients.append(g)
        if g > 0:
            n_rising += 1

    if not gradients:
        print("  N4 Thermal -- no valid gradients computed")
        return False, 0.0, 0.0, thm_valid

    mean_grad  = float(np.mean(gradients))
    pct_rising = n_rising / len(gradients)
    detected   = mean_grad > THERMAL_GRAD_THR or pct_rising > 0.60
    flagged    = thm_valid

    print(f"  N4 Thermal -- mean_grad={mean_grad:.5f}  "
          f"pct_rising={pct_rising*100:.1f}%  detected={detected}  "
          f"flagged_bands={len(flagged)}")
    return detected, mean_grad, pct_rising, flagged


# ---------------------------------------------------------------------------
# N5  Spectral smile
# ---------------------------------------------------------------------------

def detect_spectral_smile(cube, waves_um, valid_bands):
    from step0_load import find_nearest_band

    # Track pyroxene Band I region 0.85-1.15 um
    b_lo = find_nearest_band(waves_um, 0.85)
    b_hi = find_nearest_band(waves_um, 1.15)
    smile_bands = [b for b in range(b_lo, b_hi+1) if b in valid_bands]

    if len(smile_bands) < 5:
        print("  N5 Smile -- fewer than 5 bands in 0.85-1.15 um range -- skipped")
        return False, 0.0, np.zeros(cube.shape[1])

    H, W, B = cube.shape
    center_positions = np.full(W, np.nan, dtype=np.float64)
    sub_waves = waves_um[smile_bands]

    for c in range(W):
        col_mean = np.nanmean(cube[:, c, :][:, smile_bands], axis=0).astype(np.float64)
        finite = np.isfinite(col_mean)
        if finite.sum() < 3:
            continue
        min_idx = int(np.argmin(col_mean[finite]))
        # Map back to wavelength
        finite_waves = sub_waves[finite]
        center_positions[c] = finite_waves[min_idx] * 1000.0  # nm

    valid_pos = center_positions[np.isfinite(center_positions)]
    if valid_pos.size < 3:
        print("  N5 Smile -- insufficient columns -- skipped")
        return False, 0.0, np.zeros(W)

    pos_range = float(valid_pos.max() - valid_pos.min())
    med_pos   = float(np.median(valid_pos))
    shift_per_col = center_positions - med_pos
    detected  = pos_range > SMILE_SHIFT_THR_NM

    print(f"  N5 Smile -- position_range={pos_range:.2f} nm  detected={detected}")
    return detected, pos_range, shift_per_col


# ---------------------------------------------------------------------------
# Plot  Noise diagnostic
# ---------------------------------------------------------------------------

def plot_noise_diagnostics(snr, waves_um, stripe_severity, stripe_threshold,
                           striped_bands, stripe_col_offsets, spike_frac,
                           thermal_detected, mean_thermal_grad,
                           smile_detected, shift_per_col,
                           valid_bands, fname, out_path):
    from step0_load import plot_style, find_nearest_band, THERMAL_ONSET_UM
    plot_style()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    w = waves_um

    # N1 SNR
    ax = axes[0]
    colors = ["#3fb950" if snr[b] > 30 else ("#ffa657" if snr[b] > 10 else "#ff7b72")
              for b in range(len(w))]
    ax.bar(w, snr, width=(w[1]-w[0])*0.8, color=colors)
    ax.axhline(10, color="#ffa657", linestyle="--", linewidth=0.9)
    ax.axhline(30, color="#3fb950", linestyle="--", linewidth=0.9)
    ax.set_xlabel(u"Wavelength (\u03bcm)"); ax.set_ylabel("SNR")
    ax.set_title("N1: SNR per Band"); ax.grid(True, alpha=0.3)

    # N2 Stripe severity
    ax = axes[1]
    ax.plot(w, stripe_severity, color="#58a6ff")
    ax.axhline(stripe_threshold, color="#ff7b72", linestyle="--",
               linewidth=0.9, label=f"Threshold={stripe_threshold:.5f}")
    if striped_bands:
        ax.scatter(w[striped_bands], stripe_severity[striped_bands],
                   color="#ff7b72", zorder=5, s=40)
    ax.set_xlabel(u"Wavelength (\u03bcm)"); ax.set_ylabel("Column-Offset Std")
    ax.set_title("N2: Stripe Severity per Band")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # N2 worst band column profile
    ax = axes[2]
    if striped_bands:
        worst = max(striped_bands, key=lambda b: stripe_severity[b])
        offsets = np.array(stripe_col_offsets[worst])
        ax.plot(offsets, color="#ffa657")
        ax.axhline(0, color="#c9d1d9", linewidth=0.6)
        ax.set_xlabel("Column index"); ax.set_ylabel("Offset")
        ax.set_title(f"N2: Column offsets (band {worst}, {w[worst]:.3f} um)")
    else:
        ax.text(0.5, 0.5, "No stripe noise detected",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#3fb950")
        ax.set_title("N2: Column Profile")
    ax.grid(True, alpha=0.3)

    # N3 Spike fraction
    ax = axes[3]
    ax.bar(w, spike_frac * 100, width=(w[1]-w[0])*0.8, color="#8957e5")
    ax.axhline(0.2, color="#ff7b72", linestyle="--", linewidth=0.9)
    ax.set_xlabel(u"Wavelength (\u03bcm)"); ax.set_ylabel("Spike fraction %")
    ax.set_title("N3: Spike Fraction per Band"); ax.grid(True, alpha=0.3)

    # N4 Mean spectrum + thermal region
    ax = axes[4]
    mean_spec = np.array([0.0] * len(w))
    for b in valid_bands:
        from step0_load import M3_IF_MAX
        mean_spec[b] = float(np.nanmean(mean_spec))  # placeholder
    # Just show gradient illustration
    thm_onset = find_nearest_band(waves_um, THERMAL_ONSET_UM)
    ax.axvspan(w[thm_onset], w[-1]+0.05, alpha=0.15, color="#ff7b72",
               label=f"Thermal (>{THERMAL_ONSET_UM} um)")
    ax.set_xlabel(u"Wavelength (\u03bcm)")
    ax.set_title(f"N4: Thermal Emission (detected={thermal_detected})")
    ax.text(0.5, 0.5,
            f"Mean gradient: {mean_thermal_grad:.5f}\n"
            f"Detected: {thermal_detected}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="#3fb950" if thermal_detected else "#ff7b72")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # N5 Smile
    ax = axes[5]
    if smile_detected:
        ax.plot(shift_per_col, color="#ffa657")
        ax.axhline(SMILE_SHIFT_THR_NM,  color="#ff7b72", linestyle="--")
        ax.axhline(-SMILE_SHIFT_THR_NM, color="#ff7b72", linestyle="--")
        ax.set_xlabel("Column"); ax.set_ylabel("Shift (nm)")
        ax.set_title("N5: Spectral Smile Shift")
    else:
        ax.text(0.5, 0.5, "No spectral smile detected (0.00 nm)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#3fb950")
        ax.set_title("N5: Spectral Smile")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Noise Characterisation -- {fname}", color="#c9d1d9", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Noise map JSON
# ---------------------------------------------------------------------------

def save_noise_json(fname, DIRS,
                    severe_gaussian, moderate_gaussian,
                    striped_bands, stripe_severity, stripe_threshold,
                    stripe_col_offsets,
                    spiked_bands, spike_frac, total_spikes, n_sampled,
                    thermal_detected, mean_grad, pct_rising, thermal_bands_flagged,
                    smile_detected, smile_shift_nm, shift_per_col,
                    waves_um):
    nm = {
        "fname": fname,
        "N1_gaussian_severe":   severe_gaussian,
        "N1_gaussian_moderate": moderate_gaussian,
        "N2_striped_bands":     striped_bands,
        "N2_stripe_threshold":  stripe_threshold,
        "stripe_col_offsets":   {str(k): v for k, v in stripe_col_offsets.items()},
        "N3_spiked_bands":      spiked_bands,
        "N3_total_spikes":      total_spikes,
        "N3_n_sampled":         n_sampled,
        "N4_thermal_detected":  thermal_detected,
        "N4_mean_gradient":     mean_grad,
        "N4_pct_rising":        pct_rising,
        "N4_thermal_bands":     thermal_bands_flagged,
        "N5_smile_detected":    smile_detected,
        "N5_max_shift_nm":      smile_shift_nm,
        "thresholds": {
            "SNR_HIGH_NOISE": SNR_HIGH_NOISE_THR,
            "SNR_MOD_NOISE":  SNR_MOD_NOISE_THR,
            "STRIPE_STD_FACTOR": STRIPE_STD_FACTOR,
            "MAD_SPIKE_FACTOR":  MAD_SPIKE_FACTOR,
            "THERMAL_GRAD_THR":  THERMAL_GRAD_THR,
            "SMILE_SHIFT_NM":    SMILE_SHIFT_THR_NM,
        },
        "spike_frac_per_band": spike_frac.tolist(),
        "stripe_severity_per_band": stripe_severity.tolist(),
        "shift_per_col": shift_per_col.tolist(),
        "wavelengths_um": waves_um.tolist(),
    }
    out = Path(DIRS["02_noise"]) / f"{fname}_noise_map.json"
    with open(out, "w") as f:
        json.dump(nm, f, indent=2)
    print(f"  Saved: {out}")
    return nm


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_noise_id(cube, waves_um, fname, DIRS,
                 band_stats, valid_bands, dead_bands,
                 thermal_bands, valid_pixel_mask):
    print("\n" + "="*60)
    print("  STEP 2 -- Noise Identification")
    print("="*60)

    snr = band_stats["snr"]

    # N1
    severe_gaussian, moderate_gaussian = detect_gaussian_noise(snr, valid_bands)

    # N2
    (striped_bands, stripe_severity,
     stripe_col_offsets, stripe_thr) = detect_stripe_noise(cube, valid_bands, dead_bands)

    # N3
    spike_frac, spiked_bands, total_spikes, n_sampled = detect_spectral_spikes(
        cube, valid_pixel_mask, valid_bands)

    # N4
    (thermal_detected, mean_grad, pct_rising,
     thermal_flagged) = detect_thermal_emission(cube, waves_um, valid_pixel_mask, valid_bands)

    # N5
    smile_detected, smile_shift_nm, shift_per_col = detect_spectral_smile(
        cube, waves_um, valid_bands)

    # Plot
    pplot = Path(DIRS["02_noise"]) / f"{fname}_noise_characterisation.png"
    plot_noise_diagnostics(snr, waves_um, stripe_severity, stripe_thr,
                           striped_bands, stripe_col_offsets, spike_frac,
                           thermal_detected, mean_grad,
                           smile_detected, shift_per_col,
                           valid_bands, fname, str(pplot))

    # JSON
    NOISE_MAP = save_noise_json(fname, DIRS,
        severe_gaussian, moderate_gaussian,
        striped_bands, stripe_severity, stripe_thr,
        stripe_col_offsets,
        spiked_bands, spike_frac, total_spikes, n_sampled,
        thermal_detected, mean_grad, pct_rising, thermal_flagged,
        smile_detected, smile_shift_nm, shift_per_col, waves_um)

    print("\n" + "="*60)
    print("  STEP 2 SUMMARY")
    print("="*60)
    print(f"  Severe Gaussian bands : {len(severe_gaussian)}")
    print(f"  Moderate Gaussian     : {len(moderate_gaussian)}")
    print(f"  Striped bands         : {len(striped_bands)}")
    print(f"  Spiked bands          : {len(spiked_bands)}")
    print(f"  Thermal detected      : {thermal_detected}")
    print(f"  Smile detected        : {smile_detected}")
    print("="*60)
    print("  Step 2 complete. Run step3_physics.py next.")
    print("="*60 + "\n")

    return (NOISE_MAP, stripe_col_offsets, spike_frac,
            striped_bands, spiked_bands, severe_gaussian,
            thermal_detected, smile_detected, shift_per_col)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import step0_load as s0
    import step1_eda  as s1
    sd = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\M3 Journal\Mare Tranquillitatis\M3G20090203T160452_V01_RFL"
    cube, waves, fname, meta, DIRS = s0.load_scene(sd)
    (EDA, valid_pixel_mask, band_stats,
     valid_bands, dead_bands, thermal_bands, suspicious_bands) = s1.run_eda(
        cube, waves, fname, DIRS)
    results = run_noise_id(cube, waves, fname, DIRS,
                           band_stats, valid_bands, dead_bands,
                           thermal_bands, valid_pixel_mask)
