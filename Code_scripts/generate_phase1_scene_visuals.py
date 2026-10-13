"""
generate_phase1_scene_visuals.py
================================

Create per-scene journal visuals for Phase 1 denoising:
1. Noise identification diagnostics.
2. Raw -> physics corrected -> ML denoised spatial comparison.
3. Spatial-domain noise reduction evidence.

Figures use Times New Roman, white background, large labels, and are saved as
300-DPI PNG plus PDF.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step0_load as s0


JOURNAL_ROOT = Path(r"D:\M3 Journal")
OUT_DIR = JOURNAL_ROOT / "Paper_Figures" / "Phase1_Scene_Visuals"


def paper_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 19,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "axes.labelcolor": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def clean_stem(text):
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() else "_")
    return "".join(keep).strip("_")


def scene_dir_from_summary(item):
    npy = Path(item["npy"])
    location = npy.parents[2]
    fname = item["fname"]
    direct = location / fname
    if direct.exists():
        return direct
    matches = sorted(location.glob(fname + "*"))
    matches = [m for m in matches if m.is_dir()]
    return matches[0] if matches else direct


def paths_for_scene(item):
    npy = Path(item["npy"])
    location = npy.parents[2]
    fname = item["fname"]
    outputs = location / "Pipeline_outputs"
    return {
        "scene_dir": scene_dir_from_summary(item),
        "noise": outputs / "02_noise" / f"{fname}_noise_map.json",
        "physics": outputs / "03_physics" / f"{fname}_physics_corrected.npy",
        "denoised": outputs / "04_denoised" / f"{fname}_denoised.npy",
        "validation": outputs / "05_validation" / f"{fname}_validation_report.json",
    }


def finite_norm(img, lo=2, hi=98):
    arr = img.astype(np.float64)
    vals = arr[np.isfinite(arr)]
    if vals.size < 10:
        return np.zeros(arr.shape, dtype=np.float32)
    vmin = float(np.percentile(vals, lo))
    vmax = float(np.percentile(vals, hi))
    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1).astype(np.float32)


def downsample2d(img, max_side=900):
    h, w = img.shape
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    return img[::step, ::step]


def high_frequency_map(img):
    arr = img.astype(np.float64)
    finite = np.isfinite(arr)
    fill = np.nanmedian(arr[finite]) if finite.any() else 0.0
    arr = np.where(finite, arr, fill)
    neigh = (
        np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0) +
        np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1)
    ) / 4.0
    hf = np.abs(arr - neigh)
    hf[~finite] = np.nan
    return hf


def nearest(waves, target):
    return int(np.argmin(np.abs(waves - target)))


def choose_noise_band(noise, waves):
    for key in ("N2_striped_bands", "N1_gaussian_severe", "N4_thermal_bands", "N1_gaussian_moderate"):
        bands = noise.get(key) or []
        if bands:
            return int(bands[len(bands) // 2])
    return nearest(waves, 2.2 if waves[-1] >= 2.2 else 1.0)


def plot_noise_identification(item, noise, report, out_dir):
    fname = item["fname"]
    scene = item["scene"]
    waves = np.array(noise.get("wavelengths_um") or [], dtype=float)
    snr = np.array(report["metric1_snr"].get("mean_snr_ml", []), dtype=float)
    if waves.size == 0:
        waves = np.arange(len(report.get("metric3_band_depth", {})), dtype=float)

    stripe = np.array(noise.get("stripe_severity_per_band", []), dtype=float)
    spike = np.array(noise.get("spike_frac_per_band", []), dtype=float) * 100.0
    shifts = np.array(noise.get("shift_per_col", []), dtype=float)
    severe = set(int(x) for x in noise.get("N1_gaussian_severe", []))
    moderate = set(int(x) for x in noise.get("N1_gaussian_moderate", []))
    striped = set(int(x) for x in noise.get("N2_striped_bands", []))
    spiked = set(int(x) for x in noise.get("N3_spiked_bands", []))

    band_count = max(len(waves), len(stripe), len(spike), 1)
    if len(waves) != band_count:
        waves = np.arange(band_count, dtype=float)
        xlab = "Band index"
    else:
        xlab = "Wavelength (um)"

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    raw_snr = np.array(report["metric1_snr"].get("mean_snr_raw", 0.0), ndmin=1)
    ax.text(0.05, 0.82, f"Scene: {scene}", transform=ax.transAxes, fontsize=17)
    ax.text(0.05, 0.62, f"Gaussian severe bands: {len(severe)}", transform=ax.transAxes)
    ax.text(0.05, 0.47, f"Striped bands: {len(striped)}", transform=ax.transAxes)
    ax.text(0.05, 0.32, f"Spiked bands: {len(spiked)}", transform=ax.transAxes)
    ax.text(0.05, 0.17, f"Thermal detected: {noise.get('N4_thermal_detected')}", transform=ax.transAxes)
    ax.text(0.05, 0.04, f"Smile detected: {noise.get('N5_smile_detected')}", transform=ax.transAxes)
    ax.set_axis_off()
    ax.set_title("Detected Noise Summary")

    ax = axes[0, 1]
    valid = np.arange(band_count)
    colors = ["#c0392b" if i in severe else "#f1c40f" if i in moderate else "#7aa6c2" for i in valid]
    ax.bar(waves[:band_count], [1] * band_count, color=colors, width=(waves[1] - waves[0]) * 0.8 if band_count > 1 and xlab.startswith("Wavelength") else 0.8)
    ax.set_yticks([])
    ax.set_xlabel(xlab)
    ax.set_title("Gaussian Noise Bands")

    ax = axes[0, 2]
    if len(stripe):
        x = waves[:len(stripe)]
        ax.plot(x, stripe, color="#1f4e79", linewidth=1.6)
        thr = noise.get("N2_stripe_threshold")
        if isinstance(thr, (int, float)):
            ax.axhline(thr, color="#c0392b", linestyle="--", label="Detection threshold")
        if striped:
            ids = sorted(b for b in striped if b < len(x))
            ax.scatter(x[ids], stripe[ids], color="#c0392b", s=30, label="Flagged")
        ax.legend(frameon=False)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Stripe severity")
    ax.set_title("Stripe Detection")

    ax = axes[1, 0]
    if len(spike):
        x = waves[:len(spike)]
        ax.bar(x, spike, color="#6c5ce7", width=(x[1] - x[0]) * 0.8 if len(x) > 1 and xlab.startswith("Wavelength") else 0.8)
        if spiked:
            ids = sorted(b for b in spiked if b < len(x))
            ax.scatter(x[ids], spike[ids], color="#c0392b", s=35, label="Flagged")
            ax.legend(frameon=False)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Spike fraction (%)")
    ax.set_title("Spectral Spike Detection")

    ax = axes[1, 1]
    thermal_bands = noise.get("N4_thermal_bands") or []
    if len(waves) > 1:
        ax.plot(waves, np.arange(len(waves)), color="#666666", linewidth=1.2)
        if thermal_bands:
            ids = [b for b in thermal_bands if b < len(waves)]
            ax.scatter(waves[ids], np.array(ids), color="#c0392b", s=35, label="Thermal-region bands")
            ax.legend(frameon=False)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Band index")
    ax.set_title(f"Thermal Flag: gradient={noise.get('N4_mean_gradient', 0):.4f}")

    ax = axes[1, 2]
    if len(shifts):
        ax.plot(np.arange(len(shifts)), shifts, color="#2e8b57", linewidth=1.5)
        ax.axhline(2.0, color="#c0392b", linestyle="--", linewidth=1.0, label="2 nm threshold")
        ax.axhline(-2.0, color="#c0392b", linestyle="--", linewidth=1.0)
        ax.legend(frameon=False)
    ax.set_xlabel("Detector column")
    ax.set_ylabel("Spectral shift (nm)")
    ax.set_title("Spectral Smile Detection")

    fig.suptitle(f"Noise Identification Diagnostics: {scene}", y=0.995)
    fig.tight_layout()
    save(fig, out_dir / clean_stem(scene), f"{clean_stem(fname)}_noise_identification")


def plot_spatial_progression(item, raw, phys, den, waves, out_dir):
    scene = item["scene"]
    fname = item["fname"]
    bands = [nearest(waves, 1.0), nearest(waves, 2.2 if waves[-1] >= 2.2 else waves[-1])]
    labels = [f"{waves[b]:.3f} um" for b in bands]
    fig, axes = plt.subplots(len(bands), 4, figsize=(18, 8.5))
    if len(bands) == 1:
        axes = np.array([axes])

    for r, (b, label) in enumerate(zip(bands, labels)):
        imgs = [
            raw[:, :, b],
            phys[:, :, b],
            den[:, :, b],
            den[:, :, b].astype(np.float64) - raw[:, :, b].astype(np.float64),
        ]
        titles = ["Raw", "Physics corrected", "ML denoised", "Denoised - Raw"]
        for c, (img, title) in enumerate(zip(imgs, titles)):
            ax = axes[r, c]
            small = downsample2d(img)
            if c < 3:
                show = finite_norm(small)
                im = ax.imshow(show, cmap="gray", aspect="auto")
            else:
                vals = small[np.isfinite(small)]
                lim = float(np.percentile(np.abs(vals), 98)) if vals.size else 0.01
                im = ax.imshow(small, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            ax.set_title(f"{title}\nBand {b}, {label}")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Line")
            fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)

    fig.suptitle(f"Spatial Progression Through Phase 1: {scene}", y=0.995)
    fig.tight_layout()
    save(fig, out_dir / clean_stem(scene), f"{clean_stem(fname)}_raw_physics_denoised_spatial")


def plot_spatial_noise_reduction(item, raw, den, waves, noise, out_dir):
    scene = item["scene"]
    fname = item["fname"]
    b = choose_noise_band(noise, waves)
    raw_band = raw[:, :, b]
    den_band = den[:, :, b]
    raw_hf = high_frequency_map(raw_band)
    den_hf = high_frequency_map(den_band)
    diff_hf = raw_hf - den_hf

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    panels = [
        (raw_band, "Raw reflectance", "gray"),
        (den_band, "Denoised reflectance", "gray"),
        (den_band.astype(np.float64) - raw_band.astype(np.float64), "Denoised - Raw", "RdBu_r"),
        (raw_hf, "Raw spatial high-frequency noise", "magma"),
        (den_hf, "Denoised spatial high-frequency noise", "magma"),
        (diff_hf, "High-frequency noise reduction", "YlGn"),
    ]
    for ax, (img, title, cmap) in zip(axes.ravel(), panels):
        small = downsample2d(img)
        if cmap == "RdBu_r":
            vals = small[np.isfinite(small)]
            lim = float(np.percentile(np.abs(vals), 98)) if vals.size else 0.01
            im = ax.imshow(small, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        elif cmap == "YlGn":
            vals = small[np.isfinite(small)]
            vmax = float(np.percentile(vals, 98)) if vals.size else 0.01
            im = ax.imshow(np.clip(small, 0, vmax), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        else:
            im = ax.imshow(finite_norm(small), cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Line")
        fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)

    raw_med = np.nanmedian(raw_hf)
    den_med = np.nanmedian(den_hf)
    reduction = (raw_med - den_med) / max(raw_med, 1e-10) * 100.0
    fig.suptitle(
        f"Spatial-Domain Noise Reduction: {scene} | Band {b} ({waves[b]:.3f} um), "
        f"median reduction={reduction:.1f}%",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, out_dir / clean_stem(scene), f"{clean_stem(fname)}_spatial_noise_reduction")


def process_scene(item, out_dir):
    paths = paths_for_scene(item)
    missing = [str(p) for p in paths.values() if isinstance(p, Path) and not p.exists()]
    if missing:
        print(f"Skipping {item['scene']} due to missing files:")
        for p in missing:
            print(f"  {p}")
        return

    noise = load_json(paths["noise"])
    report = load_json(paths["validation"])
    plot_noise_identification(item, noise, report, out_dir)

    raw, waves, _, _, _ = s0.load_scene(str(paths["scene_dir"]))
    phys = np.load(str(paths["physics"]), mmap_mode="r")
    den = np.load(str(paths["denoised"]), mmap_mode="r")
    plot_spatial_progression(item, raw, phys, den, waves, out_dir)
    plot_spatial_noise_reduction(item, raw, den, waves, noise, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate per-scene Phase 1 visual evidence figures")
    parser.add_argument("--root", default=str(JOURNAL_ROOT))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--scene-name", action="append", default=[],
                        help="Optional substring of scene/location name to include; repeatable")
    args = parser.parse_args()

    paper_style()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    summary = load_json(root / "pipeline_summary.json")
    if args.scene_name:
        terms = [s.lower() for s in args.scene_name]
        summary = [s for s in summary if any(t in s["scene"].lower() or t in s["fname"].lower() for t in terms)]
    if not summary:
        raise SystemExit("No scenes selected.")

    for item in summary:
        print(f"\nGenerating scene visuals for {item['scene']} / {item['fname']}")
        process_scene(item, out_dir)

    print(f"\nScene visual figures saved in: {out_dir}")


if __name__ == "__main__":
    main()
