"""
step3_physics.py  --  M3 Phase 1, Step 3: Physics-Based Corrections
====================================================================
Applies mask-guided, deterministic corrections:
    C1 -- Spike removal (spectral median interpolation)
    C2 -- Thermal emission correction (linear extrapolation subtraction)
    C3 -- FFT Destripe (only if striped_bands non-empty)
    C4 -- Smile correction (only if smile_detected)

Prerequisites: steps 0-2 variables in memory.

Produces:
    03_physics/{fname}_physics_corrected.npy
    03_physics/{fname}_physics_correction.png
    03_physics/{fname}_snr_after_physics.png
    03_physics/{fname}_physics_log.json
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BLOCK_SIZE          = 64      # rows per block for thermal correction
STRIPE_SENSITIVITY  = 3.0    # sigma threshold: column offset must exceed this many MADs
MEAN_SHIFT_TOLERANCE = 0.02  # max fractional mean shift allowed after destripe
SMILE_MIN_SHIFT_UM   = 0.001  # min shift (um) to bother correcting a column
BAND_DEPTH_CHG_THR   = 0.20
BAND_DEPTH_DETECT_THR = 0.01
ROI_BAND_DEPTH_THR = 0.02
ROI_TOP_PERCENTILE = 90
MIN_ROI_PIXELS = 100
MAX_GUARD_SAMPLE = 6000
CONTINUUM_SAM_THR_DEG = 2.0

MINERAL_BAND_DEFS = [
    ("BD2300",    2.295, 2.12,  2.53),
    ("BD1900",    1.930, 1.85,  2.07),
    ("BD2210",    2.21,  2.165, 2.25),
    ("SINDEX2",   2.10,  1.815, 2.20),
    ("BD1500",    1.50,  1.43,  1.815),
    ("HCPINDEX2", 2.14,  1.82,  2.46),
]


# ---------------------------------------------------------------------------
# 3.2  C1 -- Spike removal
# ---------------------------------------------------------------------------

def correct_spikes(cube_phys, valid_bands, dead_bands, spike_frac):
    """
    Spike detection uses spectral neighbors (bands b±1, b±2) — excellent at
    finding cosmic rays / saturation that are spectrally anomalous.
    
    Spike REPLACEMENT uses spatial neighbors (up/down/left/right in the same
    band, vectorised). This preserves spatial smoothness, ensuring that
    np.diff()-based SNR metrics do not see an artificial discontinuity at
    the replaced pixel.
    """
    dead_set = set(dead_bands)
    B = cube_phys.shape[2]
    total_removed = 0

    for b in range(B):
        if b in dead_set:
            continue
        if spike_frac[b] < 0.001:
            continue

        # ── Spectral detection ─────────────────────────────────────────────
        neighbors = []
        for delta in [-2, -1, 1, 2]:
            nb = b + delta
            if 0 <= nb < B and nb not in dead_set:
                neighbors.append(nb)
        if len(neighbors) < 2:
            continue

        band_slice  = cube_phys[:, :, b].astype(np.float64)
        neigh_stack = np.stack([cube_phys[:, :, nb] for nb in neighbors], axis=2)
        neigh_med   = np.nanmedian(neigh_stack, axis=2)
        diff        = band_slice - neigh_med.astype(np.float64)

        finite_diff = diff[np.isfinite(diff)]
        if finite_diff.size < 10:
            continue
        mad = float(np.median(np.abs(finite_diff - np.median(finite_diff)))) * 1.4826
        if mad < 1e-10:
            continue

        spike_mask = np.isfinite(band_slice) & (np.abs(diff) > 5 * mad)
        n_fixed = int(spike_mask.sum())
        if n_fixed == 0:
            continue

        # ── Spatial replacement (vectorised 4-neighbor mean) ───────────────
        # Reflect-pad to handle edges cleanly
        padded = np.pad(band_slice, 1, mode='reflect')
        # Mark NaN in padded so nanmean ignores invalid neighbours
        padded = np.where(np.isfinite(padded), padded, np.nan)
        spatial_mean = np.nanmean(np.stack([
            padded[:-2, 1:-1],   # up
            padded[2:,  1:-1],   # down
            padded[1:-1, :-2],   # left
            padded[1:-1, 2:],    # right
        ], axis=0), axis=0)

        cube_phys[:, :, b][spike_mask] = spatial_mean[spike_mask].astype(np.float32)
        total_removed += n_fixed

    print(f"  C1 Spikes -- total replaced: {total_removed:,}")
    return cube_phys, total_removed



# ---------------------------------------------------------------------------
# 3.3  C2 -- Thermal emission correction
# ---------------------------------------------------------------------------

def correct_thermal(cube_phys, waves_um, valid_bands, dead_bands):
    from step0_load import find_nearest_band, M3_IF_MAX

    dead_set = set(dead_bands)

    pre_start_b = find_nearest_band(waves_um, 2.0)
    pre_end_b   = find_nearest_band(waves_um, 2.5)
    thm_start_b = find_nearest_band(waves_um, 2.5)
    thm_end_b   = cube_phys.shape[2] - 1

    pre_bands = [b for b in range(pre_start_b, pre_end_b+1)
                 if b in valid_bands and b not in dead_set]
    thm_bands = [b for b in range(thm_start_b, thm_end_b+1)
                 if b not in dead_set]

    if len(pre_bands) < 3:
        print("  C2 Thermal -- fewer than 3 pre-thermal bands -- skipped")
        return cube_phys, 0

    H, W, B = cube_phys.shape
    pre_waves = waves_um[pre_bands]
    thm_waves = waves_um[thm_bands]

    x_pre  = pre_waves - pre_waves.mean()
    ss_xx  = float(np.dot(x_pre, x_pre)) if np.dot(x_pre, x_pre) > 1e-20 else 1e-20
    n_corrected = 0

    for row_start in range(0, H, BLOCK_SIZE):
        row_end   = min(H, row_start + BLOCK_SIZE)
        block     = cube_phys[row_start:row_end, :, :].astype(np.float64)
        block_H   = block.shape[0]

        # Extract pre-thermal values for all pixels in block
        pre_vals = block[:, :, pre_bands]   # (bH, W, len(pre_bands))

        for b_idx, b in enumerate(thm_bands):
            tw = thm_waves[b_idx]
            # For each pixel: linear regression on pre_bands -> extrapolate to tw
            # Vectorised: slope = sum((x-xm)*(y-ym)) / sum((x-xm)^2)
            ym   = np.nanmean(pre_vals, axis=2, keepdims=True)  # (bH, W, 1)
            y_c  = pre_vals - ym                                  # (bH, W, nP)
            x_c  = x_pre[np.newaxis, np.newaxis, :]              # (1, 1, nP)
            ss_xy = np.nansum(x_c * y_c, axis=2)                 # (bH, W)
            slope = ss_xy / ss_xx
            intercept = ym[:, :, 0] - slope * pre_waves.mean()

            expected = slope * tw + intercept
            measured = block[:, :, b]

            valid_pix = np.isfinite(measured) & np.isfinite(expected)
            excess    = measured - expected
            # Only subtract positive excess (thermal emission adds energy)
            to_correct = valid_pix & (excess > 0)
            corrected  = np.where(to_correct,
                                  np.clip(measured - excess, 0, M3_IF_MAX),
                                  measured)
            cube_phys[row_start:row_end, :, b] = corrected.astype(np.float32)
            n_corrected += int(to_correct.sum())

    print(f"  C2 Thermal -- pixels corrected: {n_corrected:,}")
    return cube_phys, n_corrected


# ---------------------------------------------------------------------------
# 3.4  C3 -- Column-Offset Destripe (physically correct for M3 detector drift)
# ---------------------------------------------------------------------------

def correct_fft_destripe(cube_phys, striped_bands, dead_bands, waves_um):
    """
    Column-Offset Destripe: for each striped band, estimate the per-column
    DC offset as the deviation between the column median and the robust
    scene median, then subtract it. This directly corrects for M3 detector
    element gain/offset drift without introducing Gibbs ringing artifacts.
    
    Only columns whose offset exceeds STRIPE_SENSITIVITY * MAD of all
    column offsets are corrected, preventing over-correction of real
    across-track radiance gradients.
    """
    from step0_load import M3_IF_MAX
    if not striped_bands:
        print("  C3 Destripe -- no striped bands -- skipped")
        return cube_phys

    n_corrected = 0
    for b in striped_bands:
        band     = cube_phys[:, :, b].astype(np.float64)
        nan_mask = ~np.isfinite(band)

        # Per-column median (robust to row-wise outliers)
        col_medians = np.full(band.shape[1], np.nan)
        for c in range(band.shape[1]):
            col_vals = band[:, c]
            finite   = col_vals[np.isfinite(col_vals)]
            if finite.size >= 10:
                col_medians[c] = np.median(finite)

        # Scene-wide median of column medians
        valid_cm  = col_medians[np.isfinite(col_medians)]
        if valid_cm.size < 3:
            continue
        scene_med = np.median(valid_cm)

        # Column offset = deviation from scene median
        col_offset = col_medians - scene_med

        # Only correct columns where offset is a significant outlier
        mad = np.median(np.abs(col_offset[np.isfinite(col_offset)]))
        if mad < 1e-12:
            continue
        significant = np.abs(col_offset) > STRIPE_SENSITIVITY * mad

        if not significant.any():
            continue

        orig_mean = float(np.nanmean(band))
        corrected = band.copy()
        for c in np.where(significant)[0]:
            if np.isfinite(col_offset[c]):
                corrected[:, c] -= col_offset[c]

        # Safety: reject if mean shifted too much (real gradient, not stripe)
        new_mean = float(np.nanmean(corrected[~nan_mask]))
        if abs(orig_mean) > 1e-12 and \
           abs(new_mean - orig_mean) > MEAN_SHIFT_TOLERANCE * abs(orig_mean):
            continue

        corrected = np.clip(corrected, 0, M3_IF_MAX)
        corrected[nan_mask] = np.nan
        cube_phys[:, :, b]  = corrected.astype(np.float32)
        n_corrected += int(significant.sum())

    print(f"  C3 Destripe -- corrected columns in {len(striped_bands)} bands")
    return cube_phys


# ---------------------------------------------------------------------------
# 3.5  C4 -- Smile correction
# ---------------------------------------------------------------------------

def correct_smile(cube_phys, waves_um, shift_per_col, smile_detected):
    """
    Smile correction: resample each column's spectrum to the nominal wavelength
    grid using linear (not cubic) interpolation. The shift profile is first
    Gaussian-smoothed across columns to eliminate column-to-column
    discontinuities that would inflate spatial high-frequency noise.
    """
    from step0_load import M3_IF_MAX
    if not smile_detected:
        print("  C4 Smile -- not detected -- skipped")
        return cube_phys

    try:
        from scipy.ndimage import gaussian_filter1d
        from scipy.interpolate import interp1d
    except ImportError:
        print("  C4 Smile -- scipy not available -- skipped")
        return cube_phys

    H, W, B = cube_phys.shape

    # Smooth the shift profile to avoid column-to-column discontinuities
    shift_arr    = np.array(shift_per_col, dtype=np.float64)
    shift_smooth = gaussian_filter1d(shift_arr, sigma=3.0)

    for c in range(W):
        shift_um = shift_smooth[c] / 1000.0
        if abs(shift_um) < SMILE_MIN_SHIFT_UM:
            continue
        corrected_waves = waves_um - shift_um
        for r in range(H):
            spec   = cube_phys[r, c, :].astype(np.float64)
            finite = np.isfinite(spec)
            if finite.sum() < 5:
                continue
            try:
                # Linear interp: no Runge oscillations, no spectral over-shoot
                f = interp1d(corrected_waves[finite], spec[finite],
                             kind="linear", bounds_error=False,
                             fill_value=np.nan)
                new_spec            = f(waves_um)
                new_spec            = np.clip(new_spec, 0, M3_IF_MAX)
                new_spec[~finite]   = np.nan
                cube_phys[r, c, :] = new_spec.astype(np.float32)
            except Exception:
                continue

    print("  C4 Smile -- correction applied (linear interp, smoothed shifts)")
    return cube_phys


def _sample_band_depth(cube, waves_um, center, sh1, sh2, rows, cols):
    from step0_load import find_nearest_band

    if center < waves_um[0] or center > waves_um[-1]:
        return None
    bc = find_nearest_band(waves_um, center)
    bs1 = find_nearest_band(waves_um, sh1)
    bs2 = find_nearest_band(waves_um, sh2)
    frac = (center - sh1) / max(sh2 - sh1, 1e-8)

    rc = cube[rows, cols, bc].astype(np.float64)
    rs1 = cube[rows, cols, bs1].astype(np.float64)
    rs2 = cube[rows, cols, bs2].astype(np.float64)
    finite = np.isfinite(rc) & np.isfinite(rs1) & np.isfinite(rs2)
    if not np.any(finite):
        return None
    rcont = rs1 + frac * (rs2 - rs1)
    good = finite & (rcont > 1e-8)
    if not np.any(good):
        return None
    bd = 1.0 - rc[good] / rcont[good]
    return float(np.median(bd)) if bd.size else None


def _band_depth_map(cube, waves_um, center, sh1, sh2):
    from step0_load import find_nearest_band

    if center < waves_um[0] or center > waves_um[-1]:
        return None
    bc = find_nearest_band(waves_um, center)
    bs1 = find_nearest_band(waves_um, sh1)
    bs2 = find_nearest_band(waves_um, sh2)
    frac = (center - sh1) / max(sh2 - sh1, 1e-8)

    rc = cube[:, :, bc].astype(np.float64)
    rs1 = cube[:, :, bs1].astype(np.float64)
    rs2 = cube[:, :, bs2].astype(np.float64)
    cont = rs1 + frac * (rs2 - rs1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - rc / np.where(cont > 1e-8, cont, np.nan)


def _continuum_sam(cube_a, cube_b, waves_um, left_idx, right_idx,
                   lo, hi, rows, cols):
    w = waves_um[lo:hi + 1]
    a = cube_a[rows, cols, lo:hi + 1].astype(np.float64)
    b = cube_b[rows, cols, lo:hi + 1].astype(np.float64)
    al = cube_a[rows, cols, left_idx].astype(np.float64)
    ar = cube_a[rows, cols, right_idx].astype(np.float64)
    bl = cube_b[rows, cols, left_idx].astype(np.float64)
    br = cube_b[rows, cols, right_idx].astype(np.float64)
    wl = float(waves_um[left_idx])
    wr = float(waves_um[right_idx])
    frac = (w[None, :] - wl) / max(wr - wl, 1e-8)
    cont_a = al[:, None] + frac * (ar - al)[:, None]
    cont_b = bl[:, None] + frac * (br - bl)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        cr_a = a / np.where(cont_a > 1e-8, cont_a, np.nan)
        cr_b = b / np.where(cont_b > 1e-8, cont_b, np.nan)

    sams = []
    for sa, sb in zip(cr_a, cr_b):
        finite = np.isfinite(sa) & np.isfinite(sb)
        if finite.sum() < 3:
            continue
        af = sa[finite]
        bf = sb[finite]
        na = np.linalg.norm(af)
        nb = np.linalg.norm(bf)
        if na > 1e-10 and nb > 1e-10:
            sams.append(float(np.arccos(np.clip(np.dot(af, bf) / (na * nb), -1.0, 1.0))))
    return float(np.degrees(np.median(sams))) if sams else None


def guard_physics_band_depths(cube_raw, cube_phys, waves_um,
                              valid_pixel_mask, valid_bands):
    from step0_load import find_nearest_band

    rows_v, cols_v = np.where(valid_pixel_mask)
    if len(rows_v) == 0:
        return []

    rng = np.random.default_rng(42)
    idx = rng.choice(len(rows_v), size=min(MAX_GUARD_SAMPLE, len(rows_v)), replace=False)
    rows_s = rows_v[idx]
    cols_s = cols_v[idx]
    valid_set = set(valid_bands)
    revert_bands = set()
    flagged = []

    for name, center, sh1, sh2 in MINERAL_BAND_DEFS:
        bd_raw_map = _band_depth_map(cube_raw, waves_um, center, sh1, sh2)
        if bd_raw_map is None:
            continue

        positive = bd_raw_map[np.isfinite(bd_raw_map) &
                              (bd_raw_map >= ROI_BAND_DEPTH_THR) &
                              (bd_raw_map <= 0.60)]
        rows_f, cols_f = rows_s, cols_s
        if positive.size >= MIN_ROI_PIXELS:
            roi_thr = max(ROI_BAND_DEPTH_THR,
                          float(np.percentile(positive, ROI_TOP_PERCENTILE)))
            roi = (valid_pixel_mask & np.isfinite(bd_raw_map) &
                   (bd_raw_map >= roi_thr) & (bd_raw_map <= 0.60))
            rr, cc = np.where(roi)
            if len(rr) >= MIN_ROI_PIXELS:
                ridx = rng.choice(len(rr), size=min(MAX_GUARD_SAMPLE, len(rr)),
                                  replace=False)
                rows_f = rr[ridx]
                cols_f = cc[ridx]

        bd_raw = _sample_band_depth(cube_raw, waves_um, center, sh1, sh2,
                                    rows_f, cols_f)
        bd_phys = _sample_band_depth(cube_phys, waves_um, center, sh1, sh2,
                                     rows_f, cols_f)
        if bd_raw is None or bd_phys is None:
            continue
        if bd_raw < BAND_DEPTH_DETECT_THR:
            continue
        chg_pct = (bd_phys - bd_raw) / abs(bd_raw) * 100.0

        band_ids = []
        for target in (center, sh1, sh2):
            if waves_um[0] <= target <= waves_um[-1]:
                b = find_nearest_band(waves_um, target)
                if b in valid_set:
                    band_ids.append(b)
        if len(band_ids) < 3:
            continue
        center_idx, left_idx, right_idx = band_ids
        lo = min(band_ids)
        hi = max(band_ids)
        cr_sam_deg = _continuum_sam(
            cube_raw, cube_phys, waves_um, left_idx, right_idx,
            lo, hi, rows_f, cols_f)
        depth_bad = abs(chg_pct) > BAND_DEPTH_CHG_THR * 100.0
        shape_bad = cr_sam_deg is not None and cr_sam_deg > CONTINUUM_SAM_THR_DEG
        if not (depth_bad or shape_bad):
            continue

        window = [b for b in range(lo, hi + 1) if b in valid_set]
        revert_bands.update(window)
        flagged.append((name, bd_raw, bd_phys, chg_pct, cr_sam_deg, window))

    if not revert_bands:
        print("  C5 Mineral guard -- all detected band depths preserved")
        return []

    bands = sorted(revert_bands)
    print(f"  C5 Mineral guard -- reverting {len(bands)} continuum-window band(s) "
          f"to raw to preserve absorptions")
    for name, bd_raw, bd_phys, chg_pct, cr_sam_deg, window in flagged:
        sam_txt = "--" if cr_sam_deg is None else f"{cr_sam_deg:.2f} deg"
        print(f"    {name}: {bd_raw:.4f} -> {bd_phys:.4f} "
              f"({chg_pct:+.1f}%), CR-SAM={sam_txt}, bands={window[0]}-{window[-1]}")

    H = cube_phys.shape[0]
    for i in range(0, H, BLOCK_SIZE):
        j = min(i + BLOCK_SIZE, H)
        for b in bands:
            cube_phys[i:j, :, b] = cube_raw[i:j, :, b]
    return bands


# ---------------------------------------------------------------------------
# 3.6  Post-physics SNR
# ---------------------------------------------------------------------------

def compute_snr_after(cube_phys, valid_bands):
    B = cube_phys.shape[2]
    snr = np.zeros(B, dtype=np.float64)
    for b in valid_bands:
        img = cube_phys[:, :, b]
        diff_h = np.diff(img, axis=1)
        diff_v = np.diff(img, axis=0)
        
        valid_h = diff_h[np.isfinite(diff_h)]
        valid_v = diff_v[np.isfinite(diff_v)]
        if valid_h.size < 100 or valid_v.size < 100:
            continue
            
        noise_h = np.std(valid_h) / 1.41421356
        noise_v = np.std(valid_v) / 1.41421356
        noise = (noise_h + noise_v) / 2.0
        
        valid_img = img[np.isfinite(img)]
        if valid_img.size > 0 and noise > 1e-12:
            signal = np.mean(valid_img)
            snr[b] = signal / noise
    return snr


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_before_after_spectra(cube, cube_phys, waves_um,
                               valid_pixel_mask, fname, out_path):
    from step0_load import plot_style, THERMAL_ONSET_UM, find_nearest_band
    plot_style()
    rng = np.random.default_rng(42)
    rows, cols = np.where(valid_pixel_mask)
    idx = rng.choice(len(rows), size=min(3, len(rows)), replace=False)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    COLORS_ORIG  = "#58a6ff"
    COLORS_CORR  = "#3fb950"

    for i, ii in enumerate(idx):
        r, c = int(rows[ii]), int(cols[ii])
        orig_spec = cube[r, c, :].astype(np.float64)
        corr_spec = cube_phys[r, c, :].astype(np.float64)
        diff      = corr_spec - orig_spec
        w = waves_um

        ax = axes[i, 0]
        m_orig = np.isfinite(orig_spec); m_corr = np.isfinite(corr_spec)
        ax.plot(w[m_orig], orig_spec[m_orig], color=COLORS_ORIG, linewidth=1.2, label="Original")
        ax.plot(w[m_corr], corr_spec[m_corr], color=COLORS_CORR, linewidth=1.2,
                linestyle="--", label="Physics")
        ax.axvspan(THERMAL_ONSET_UM, w[-1]+0.05, alpha=0.1, color="#ff7b72")
        for wk in [1.0, 2.0]:
            if w[0] <= wk <= w[-1]:
                ax.axvline(wk, color="#e3b341", linewidth=0.6, linestyle=":")
        ax.set_xlabel(u"\u03bcm"); ax.set_ylabel("I/F")
        ax.set_title(f"Pixel ({r},{c})")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax2 = axes[i, 1]
        m_diff = np.isfinite(diff)
        pos = diff.copy(); pos[diff < 0] = 0
        neg = diff.copy(); neg[diff >= 0] = 0
        ax2.fill_between(w[m_diff], np.zeros(m_diff.sum()),
                         pos[m_diff], alpha=0.5, color="#3fb950")
        ax2.fill_between(w[m_diff], np.zeros(m_diff.sum()),
                         neg[m_diff], alpha=0.5, color="#ff7b72")
        ax2.plot(w[m_diff], diff[m_diff], color="#c9d1d9", linewidth=0.8)
        ax2.axhline(0, color="#c9d1d9", linewidth=0.5)
        ax2.set_xlabel(u"\u03bcm"); ax2.set_ylabel("Delta I/F")
        ax2.set_title(f"Difference (physics - orig)")
        ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Physics Corrections Before/After -- {fname}",
                 color="#c9d1d9", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_snr_comparison(snr_before, snr_after, waves_um, valid_bands, fname, out_path):
    from step0_load import plot_style
    plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    w = waves_um
    ax.plot(w[valid_bands], snr_before[valid_bands],
            color="#58a6ff", label="Before physics", linewidth=1.4)
    ax.plot(w[valid_bands], snr_after[valid_bands],
            color="#3fb950", linestyle="--", label="After physics", linewidth=1.4)
    ax.set_xlabel(u"Wavelength (\u03bcm)"); ax.set_ylabel("SNR")
    ax.set_title(f"SNR Before vs After Physics Corrections -- {fname}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_physics(cube, waves_um, fname, DIRS,
                band_stats, valid_bands, dead_bands, thermal_bands,
                valid_pixel_mask,
                spike_frac, striped_bands, spiked_bands,
                thermal_detected, smile_detected, shift_per_col):
    from step0_load import M3_IF_MAX

    print("\n" + "="*60)
    print("  STEP 3 -- Physics-Based Corrections")
    print("="*60)

    # 3.1 Copy cube
    cube_phys = cube.copy().astype(np.float32)
    print("  cube_phys created (copy of original)")

    # C1
    cube_phys, n_spikes = correct_spikes(cube_phys, valid_bands, dead_bands, spike_frac)

    # C2
    if thermal_detected:
        cube_phys, n_thermal = correct_thermal(cube_phys, waves_um, valid_bands, dead_bands)
    else:
        print("  C2 Thermal -- not detected -- skipped")
        n_thermal = 0

    # C3
    cube_phys = correct_fft_destripe(cube_phys, striped_bands, dead_bands, waves_um)

    # C4
    cube_phys = correct_smile(cube_phys, waves_um, shift_per_col, smile_detected)

    # C5
    mineral_guard_bands = guard_physics_band_depths(
        cube, cube_phys, waves_um, valid_pixel_mask, valid_bands)

    # 3.6 SNR after
    snr_before = band_stats["snr"]
    SNR_AFTER_PHYSICS = compute_snr_after(cube_phys, valid_bands)

    if valid_bands:
        print(f"  SNR before: {float(np.mean(snr_before[valid_bands])):.3f}  "
              f"after: {float(np.mean(SNR_AFTER_PHYSICS[valid_bands])):.3f}")

    # Plots
    p1 = Path(DIRS["03_physics"]) / f"{fname}_physics_correction.png"
    plot_before_after_spectra(cube, cube_phys, waves_um, valid_pixel_mask, fname, str(p1))

    p2 = Path(DIRS["03_physics"]) / f"{fname}_snr_after_physics.png"
    plot_snr_comparison(snr_before, SNR_AFTER_PHYSICS, waves_um, valid_bands, fname, str(p2))

    # Save cube
    npy_path = Path(DIRS["03_physics"]) / f"{fname}_physics_corrected.npy"
    np.save(str(npy_path), cube_phys)
    print(f"  Saved cube: {npy_path}")

    # Log JSON
    log = {
        "fname": fname,
        "corrections_applied": {
            "spike_removal": True,
            "thermal_correction": bool(thermal_detected),
            "fft_destripe": len(striped_bands) > 0,
            "smile_correction": bool(smile_detected),
            "mineral_band_guard": len(mineral_guard_bands) > 0,
        },
        "spike_pixels_removed": n_spikes,
        "thermal_pixels_corrected": n_thermal,
        "striped_bands_processed": len(striped_bands),
        "mineral_guard_bands": mineral_guard_bands,
        "mean_snr_before": float(np.mean(snr_before[valid_bands])) if valid_bands else 0.0,
        "mean_snr_after":  float(np.mean(SNR_AFTER_PHYSICS[valid_bands])) if valid_bands else 0.0,
        "snr_after_physics": SNR_AFTER_PHYSICS.tolist(),   # full per-band array for cache loading
    }
    log_path = Path(DIRS["03_physics"]) / f"{fname}_physics_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved log: {log_path}")

    print("\n" + "="*60)
    print("  STEP 3 SUMMARY")
    print("="*60)
    print(f"  Spike pixels replaced    : {n_spikes:,}")
    print(f"  Thermal pixels corrected : {n_thermal:,}")
    print(f"  Striped bands processed  : {len(striped_bands)}")
    print(f"  Smile corrected          : {smile_detected}")
    print(f"  Mineral guard bands      : {len(mineral_guard_bands)}")
    print("="*60)
    print("  Step 3 complete. Run step4_denoise.py next.")
    print("="*60 + "\n")

    return cube_phys, SNR_AFTER_PHYSICS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import step0_load as s0, step1_eda as s1, step2_noise as s2
    sd = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\M3 Journal\Mare Tranquillitatis\M3G20090203T160452_V01_RFL"
    cube, waves, fname, meta, DIRS = s0.load_scene(sd)
    (EDA, valid_pixel_mask, band_stats,
     valid_bands, dead_bands, thermal_bands, suspicious_bands) = s1.run_eda(
        cube, waves, fname, DIRS)
    (NOISE_MAP, stripe_col_offsets, spike_frac,
     striped_bands, spiked_bands, severe_gaussian,
     thermal_detected, smile_detected, shift_per_col) = s2.run_noise_id(
        cube, waves, fname, DIRS, band_stats, valid_bands, dead_bands,
        thermal_bands, valid_pixel_mask)
    cube_phys, SNR_AFTER_PHYSICS = run_physics(
        cube, waves, fname, DIRS, band_stats, valid_bands, dead_bands,
        thermal_bands, valid_pixel_mask, spike_frac, striped_bands, spiked_bands,
        thermal_detected, smile_detected, shift_per_col)
