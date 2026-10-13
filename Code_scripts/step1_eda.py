"""
step1_eda.py  --  M3 Phase 1, Step 1: Exploratory Data Analysis
================================================================
Prerequisites: step0_load.py must have been imported first so that
    cube, waves, fname, meta, DIRS  are in memory.

Produces:
    01_eda/{fname}_band_statistics.png
    01_eda/{fname}_spatial_overview.png
    01_eda/{fname}_eda.json

Usage (standalone):
    python step1_eda.py "D:/M3 Journal/Mare Tranquillitatis/M3G20090203T160452_V01_RFL"
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
DEAD_VAR_THRESHOLD  = 1e-10
DEAD_NAN_THRESHOLD  = 0.95
THERMAL_ONSET_UM    = 2.50
MIN_VALID_BANDS     = 20

KEY_WAVES = {
    1.00: "Pyroxene I",
    2.00: "Pyroxene II",
    2.30: "Phyllosilicate",
    2.50: "Thermal onset",
}


# ---------------------------------------------------------------------------
# 1.1  Per-band statistics
# ---------------------------------------------------------------------------

def compute_band_stats(cube):
    H, W, B = cube.shape
    stats = {k: np.zeros(B, dtype=np.float64)
             for k in ("mean","std","var","vmin","vmax","nan_frac","valid_pct")}
    for b in range(B):
        sl = cube[:, :, b].ravel().astype(np.float64)
        finite = sl[np.isfinite(sl)]
        if finite.size < 10:
            continue
        stats["mean"][b]      = finite.mean()
        stats["std"][b]       = finite.std()
        stats["var"][b]       = finite.var()
        stats["vmin"][b]      = finite.min()
        stats["vmax"][b]      = finite.max()
        nan_count             = np.isnan(sl).sum()
        stats["nan_frac"][b]  = nan_count / sl.size
        stats["valid_pct"][b] = finite.size / sl.size * 100
    return stats


# ---------------------------------------------------------------------------
# 1.2  Band classification
# ---------------------------------------------------------------------------

def classify_bands(stats, waves_um):
    B = len(waves_um)
    dead_bands      = []
    thermal_bands   = []
    low_signal_bands= []
    suspicious_bands= []
    valid_bands     = []

    for b in range(B):
        if stats["var"][b] < DEAD_VAR_THRESHOLD or stats["nan_frac"][b] > DEAD_NAN_THRESHOLD:
            dead_bands.append(b)
        else:
            valid_bands.append(b)
            if waves_um[b] > THERMAL_ONSET_UM:
                thermal_bands.append(b)
            if abs(stats["mean"][b]) < 0.001:
                low_signal_bands.append(b)

    if valid_bands:
        stds = np.array([stats["std"][b] for b in valid_bands])
        susp_thr = np.percentile(stds, 95)
        for b in valid_bands:
            if stats["std"][b] > susp_thr:
                suspicious_bands.append(b)
    else:
        susp_thr = 0.0

    return (dead_bands, thermal_bands, low_signal_bands,
            suspicious_bands, valid_bands, float(susp_thr))


# ---------------------------------------------------------------------------
# 1.3  SNR per band
# ---------------------------------------------------------------------------

def compute_snr(cube, valid_bands, B):
    snr = np.zeros(B, dtype=np.float64)
    for b in valid_bands:
        img = cube[:, :, b]
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
# 1.4  Valid pixel mask
# ---------------------------------------------------------------------------

def build_valid_pixel_mask(cube):
    finite_count = np.isfinite(cube).sum(axis=2)
    mask = finite_count >= MIN_VALID_BANDS
    n    = int(mask.sum())
    total = mask.size
    print(f"  Valid pixels: {n:,} / {total:,}  ({n/total*100:.1f}%)")
    return mask, n, total


# ---------------------------------------------------------------------------
# 1.5  Sample spectra
# ---------------------------------------------------------------------------

def collect_sample_spectra(cube, valid_pixel_mask, n_samples=5, seed=42):
    rng = np.random.default_rng(seed)
    rows, cols = np.where(valid_pixel_mask)
    idx = rng.choice(len(rows), size=min(n_samples, len(rows)), replace=False)
    spectra = []
    for i in idx:
        r, c = int(rows[i]), int(cols[i])
        spectra.append({"row": r, "col": c, "spectrum": cube[r, c, :].tolist()})
    return spectra


# ---------------------------------------------------------------------------
# 1.6  Plot 1 -- Band Statistics
# ---------------------------------------------------------------------------

def plot_band_statistics(stats, snr, waves_um,
                         dead_bands, suspicious_bands, thermal_bands,
                         susp_thr, sample_spectra, out_path):
    from step0_load import plot_style
    plot_style()
    COLORS = ["#58a6ff","#3fb950","#ff7b72","#d2a8ff","#ffa657",
              "#79c0ff","#f0883e","#56d364","#ff6e96","#cae8ff"]

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_top  = fig.add_subplot(gs[0, :])
    ax_snr  = fig.add_subplot(gs[1, 0])
    ax_nan  = fig.add_subplot(gs[1, 1])
    ax_std  = fig.add_subplot(gs[2, 0])
    ax_cov  = fig.add_subplot(gs[2, 1])
    ax_spec = fig.add_subplot(gs[3, :])

    w = waves_um

    # -- Top: mean +/- 1-sigma
    mu  = stats["mean"]
    sig = stats["std"]
    ax_top.fill_between(w, mu-sig, mu+sig, alpha=0.25, color="#58a6ff", label=u"\u00b11\u03c3")
    ax_top.plot(w, mu, color="#58a6ff", linewidth=1.6, label="Mean I/F")
    for b in dead_bands:
        ax_top.scatter(w[b], mu[b], marker="x", color="#ff7b72", s=60, zorder=5)
    for b in suspicious_bands:
        ax_top.scatter(w[b], mu[b], marker="^", color="#ffa657", s=40, zorder=5)
    ax_top.axvspan(THERMAL_ONSET_UM, w[-1]+0.05, alpha=0.12, color="#ff7b72")
    for wk, label in KEY_WAVES.items():
        if w[0] <= wk <= w[-1]:
            ax_top.axvline(wk, color="#e3b341", linewidth=0.8, linestyle="--")
            ax_top.text(wk+0.01, ax_top.get_ylim()[1]*0.95, label,
                        color="#e3b341", fontsize=7, rotation=90, va="top")
    ax_top.set_xlabel(u"Wavelength (\u03bcm)"); ax_top.set_ylabel("I/F")
    ax_top.set_title("Per-Band Mean Reflectance +/- 1sigma")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # -- SNR bar
    colors_snr = ["#3fb950" if snr[b] > 30 else ("#ffa657" if snr[b] > 10 else "#ff7b72")
                  for b in range(len(w))]
    ax_snr.bar(w, snr, width=(w[1]-w[0])*0.8, color=colors_snr)
    ax_snr.axhline(10, color="#ffa657", linestyle="--", linewidth=0.9)
    ax_snr.axhline(30, color="#3fb950", linestyle="--", linewidth=0.9)
    ax_snr.set_xlabel(u"Wavelength (\u03bcm)"); ax_snr.set_ylabel("SNR")
    ax_snr.set_title("SNR per Band")
    ax_snr.grid(True, alpha=0.3)

    # -- NaN fraction
    ax_nan.bar(w, stats["nan_frac"]*100, width=(w[1]-w[0])*0.8, color="#8957e5")
    ax_nan.axhline(20,  color="#ffa657", linestyle="--", linewidth=0.9)
    ax_nan.axhline(95,  color="#ff7b72", linestyle="--", linewidth=0.9)
    ax_nan.set_xlabel(u"Wavelength (\u03bcm)"); ax_nan.set_ylabel("NaN %")
    ax_nan.set_title("NaN Fraction per Band")
    ax_nan.grid(True, alpha=0.3)

    # -- Std
    ax_std.plot(w, stats["std"], color="#3fb950")
    ax_std.axhline(susp_thr, color="#ffa657", linestyle="--", linewidth=0.9,
                   label=f"Suspicious thr={susp_thr:.4f}")
    ax_std.set_xlabel(u"Wavelength (\u03bcm)"); ax_std.set_ylabel("Std")
    ax_std.set_title("Standard Deviation per Band")
    ax_std.legend(fontsize=8)
    ax_std.grid(True, alpha=0.3)

    # -- Valid coverage
    ax_cov.fill_between(w, stats["valid_pct"], alpha=0.6, color="#58a6ff")
    ax_cov.axhline(80, color="#ffa657", linestyle="--", linewidth=0.9)
    ax_cov.set_xlabel(u"Wavelength (\u03bcm)"); ax_cov.set_ylabel("Valid Pixels %")
    ax_cov.set_title("Valid Pixel Coverage per Band")
    ax_cov.grid(True, alpha=0.3)

    # -- Sample spectra
    for i, sp in enumerate(sample_spectra):
        spec = np.array(sp["spectrum"], dtype=np.float64)
        mask_finite = np.isfinite(spec)
        ax_spec.plot(w[mask_finite], spec[mask_finite], color=COLORS[i % len(COLORS)],
                     label=f"({sp['row']},{sp['col']})", linewidth=1.2)
    ax_spec.set_xlabel(u"Wavelength (\u03bcm)"); ax_spec.set_ylabel("I/F")
    ax_spec.set_title("Sample Pixel Spectra (5 random pixels)")
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, alpha=0.3)

    plt.suptitle("M3 Band Statistics Overview", color="#c9d1d9", fontsize=13)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 1.7  Plot 2 -- Spatial Overview
# ---------------------------------------------------------------------------

def plot_spatial_overview(cube, waves_um, valid_pixel_mask, fname, out_path):
    from step0_load import find_nearest_band, plot_style
    plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _norm(img):
        lo, hi = np.nanpercentile(img[np.isfinite(img)], [2, 98])
        return np.clip((img - lo) / max(hi - lo, 1e-8), 0, 1)

    # False color RGB (0.75, 0.90, 1.05 um)
    R = _norm(cube[:, :, find_nearest_band(waves_um, 0.75)])
    G = _norm(cube[:, :, find_nearest_band(waves_um, 0.90)])
    B_ = _norm(cube[:, :, find_nearest_band(waves_um, 1.05)])
    rgb = np.stack([R, G, B_], axis=2)
    rgb[~np.isfinite(rgb).all(axis=2)] = 0
    axes[0].imshow(rgb, aspect="auto")
    axes[0].set_title("False Color (0.75/0.90/1.05 um)")
    axes[0].axis("off")

    # Single band at 1.0 um
    bi = find_nearest_band(waves_um, 1.0)
    band_img = cube[:, :, bi].astype(np.float64)
    lo, hi = np.nanpercentile(band_img[np.isfinite(band_img)], [2, 98])
    im1 = axes[1].imshow(np.clip(band_img, lo, hi), cmap="viridis", aspect="auto")
    plt.colorbar(im1, ax=axes[1], label="I/F", fraction=0.046, pad=0.04)
    axes[1].set_title(f"Single Band (1.0 um, idx={bi})")
    axes[1].axis("off")

    # Valid band count
    valid_count = np.isfinite(cube).sum(axis=2).astype(np.float32)
    im2 = axes[2].imshow(valid_count, cmap="RdYlGn", aspect="auto",
                         vmin=0, vmax=cube.shape[2])
    plt.colorbar(im2, ax=axes[2], label="Valid Bands", fraction=0.046, pad=0.04)
    axes[2].set_title(f"Valid Band Count (min={MIN_VALID_BANDS})")
    axes[2].axis("off")

    plt.suptitle(f"Spatial Overview -- {fname}", color="#c9d1d9", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 1.8  EDA JSON
# ---------------------------------------------------------------------------

def save_eda_json(fname, cube, waves_um, stats, snr,
                  dead_bands, thermal_bands, valid_bands, suspicious_bands,
                  low_signal_bands, susp_thr, valid_pixel_count, total_pixels,
                  out_path):
    H, W, B = cube.shape
    eda = {
        "fname":     fname,
        "shape":     [H, W, B],
        "wavelengths_um": waves_um.tolist(),
        "band_stats": {k: v.tolist() for k, v in stats.items()},
        "snr":       snr.tolist(),
        "dead_bands":       dead_bands,
        "thermal_bands":    thermal_bands,
        "valid_bands":      valid_bands,
        "suspicious_bands": suspicious_bands,
        "low_signal_bands": low_signal_bands,
        "thresholds": {
            "dead_var":          DEAD_VAR_THRESHOLD,
            "dead_nan_frac":     DEAD_NAN_THRESHOLD,
            "thermal_onset_um":  THERMAL_ONSET_UM,
            "suspicious_std":    susp_thr,
            "min_valid_bands":   MIN_VALID_BANDS,
        },
        "valid_pixel_count":  valid_pixel_count,
        "valid_pixel_frac":   valid_pixel_count / max(total_pixels, 1),
        "mean_snr":           float(np.mean(snr[valid_bands])) if valid_bands else 0.0,
    }
    with open(out_path, "w") as f:
        json.dump(eda, f, indent=2)
    print(f"  Saved: {out_path}")
    return eda


# ---------------------------------------------------------------------------
# 1.9/1.10  Main run function
# ---------------------------------------------------------------------------

def run_eda(cube, waves_um, fname, DIRS):
    """Run full Step 1 EDA. Returns EDA dict and band-classification lists."""
    print("\n" + "="*60)
    print("  STEP 1 -- Exploratory Data Analysis")
    print("="*60)

    # 1.1  Per-band stats
    print("  Computing per-band statistics ...")
    stats = compute_band_stats(cube)

    # 1.2  Band classification
    (dead_bands, thermal_bands, low_signal_bands,
     suspicious_bands, valid_bands, susp_thr) = classify_bands(stats, waves_um)

    print(f"  Valid bands  : {len(valid_bands)}")
    print(f"  Dead bands   : {len(dead_bands)}  -> {dead_bands}")
    print(f"  Thermal bands: {len(thermal_bands)}")
    print(f"  Suspicious   : {len(suspicious_bands)}")
    print(f"  Low signal   : {len(low_signal_bands)}")

    # 1.3  SNR (Local Spatial Variance)
    snr = compute_snr(cube, valid_bands, cube.shape[2])
    if valid_bands:
        snr_vals = snr[valid_bands]
        print(f"  SNR -- min={snr_vals.min():.2f}  max={snr_vals.max():.2f}  "
              f"mean={snr_vals.mean():.2f}")

    # 1.4  Valid pixel mask
    valid_pixel_mask, n_valid, n_total = build_valid_pixel_mask(cube)

    # 1.5  Sample spectra
    sample_spectra = collect_sample_spectra(cube, valid_pixel_mask)

    # 1.6  Plot 1
    p1 = Path(DIRS["01_eda"]) / f"{fname}_band_statistics.png"
    plot_band_statistics(stats, snr, waves_um, dead_bands, suspicious_bands,
                         thermal_bands, susp_thr, sample_spectra, str(p1))

    # 1.7  Plot 2
    p2 = Path(DIRS["01_eda"]) / f"{fname}_spatial_overview.png"
    plot_spatial_overview(cube, waves_um, valid_pixel_mask, fname, str(p2))

    # 1.8  JSON
    p3 = Path(DIRS["01_eda"]) / f"{fname}_eda.json"
    EDA = save_eda_json(fname, cube, waves_um, stats, snr,
                        dead_bands, thermal_bands, valid_bands, suspicious_bands,
                        low_signal_bands, susp_thr, n_valid, n_total, str(p3))

    # 1.10  Summary
    print("\n" + "="*60)
    print("  STEP 1 SUMMARY")
    print("="*60)
    print(f"  Valid bands        : {len(valid_bands)} / {cube.shape[2]}")
    print(f"  Dead bands         : {len(dead_bands)}")
    print(f"  Thermal bands      : {len(thermal_bands)}")
    print(f"  Valid pixels       : {n_valid:,} ({n_valid/n_total*100:.1f}%)")
    if valid_bands:
        print(f"  Mean SNR           : {float(np.mean(snr[valid_bands])):.2f}")
    print("="*60)
    print("  Step 1 complete. Run step2_noise.py next.")
    print("="*60 + "\n")

    return (EDA, valid_pixel_mask,
            {"mean": stats["mean"], "std": stats["std"], "var": stats["var"],
             "nan_frac": stats["nan_frac"], "valid_pct": stats["valid_pct"],
             "vmin": stats["vmin"], "vmax": stats["vmax"], "snr": snr},
            valid_bands, dead_bands, thermal_bands, suspicious_bands)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import step0_load as s0
    sd = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\M3 Journal\Mare Tranquillitatis\M3G20090203T160452_V01_RFL"
    cube, waves, fname, meta, DIRS = s0.load_scene(sd)
    (EDA, valid_pixel_mask, band_stats,
     valid_bands, dead_bands, thermal_bands, suspicious_bands) = run_eda(
        cube, waves, fname, DIRS)
