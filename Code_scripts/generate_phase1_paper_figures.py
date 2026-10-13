"""
generate_phase1_paper_figures.py
================================

Create journal-style Phase 1 denoising figures from pipeline outputs.

Inputs
------
- D:/M3 Journal/pipeline_summary.json
- D:/M3 Journal/scene_readiness_manifest.json
- D:/M3 Journal/Phase2_outputs/phase2a_continuum_readiness.json
- Per-scene *_validation_report.json files

Outputs
-------
PNG and PDF figures in D:/M3 Journal/Paper_Figures/Phase1_Denoising
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


JOURNAL_ROOT = Path(r"D:\M3 Journal")
OUT_DIR = JOURNAL_ROOT / "Paper_Figures" / "Phase1_Denoising"

FEATURE_ORDER = ["BD2300", "BD1900", "BD2210", "SINDEX2", "BD1500", "HCPINDEX2"]
FEATURE_LABELS = {
    "BD2300": "BD2300\nPhyllosilicates",
    "BD1900": "BD1900\nHydrated",
    "BD2210": "BD2210\nAl-OH clays",
    "SINDEX2": "SINDEX2\nSulfates",
    "BD1500": "BD1500\nWater ice",
    "HCPINDEX2": "HCPINDEX2\nHCP pyroxene",
}

STATUS_SCORE = {"N/A": -1, "SKIP": -1, "FAIL": 0, "REVIEW": 1, "PASS": 2}


def paper_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.1,
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "axes.labelcolor": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def report_path_from_summary(item):
    npy = Path(item["npy"])
    val_dir = npy.parents[1] / "05_validation"
    fname = item.get("fname") or npy.name.replace("_denoised.npy", "")
    return val_dir / f"{fname}_validation_report.json"


def load_latest_scene_data(root):
    summary = load_json(root / "pipeline_summary.json")
    manifest_path = root / "scene_readiness_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else []
    manifest_by_fname = {m.get("fname"): m for m in manifest}

    scenes = []
    for item in summary:
        rpt_path = report_path_from_summary(item)
        if not rpt_path.exists():
            print(f"WARNING: missing validation report: {rpt_path}")
            continue
        report = load_json(rpt_path)
        scenes.append({
            "scene": item["scene"],
            "fname": item["fname"],
            "summary": item,
            "report": report,
            "manifest": manifest_by_fname.get(item["fname"], {}),
        })
    return scenes


def load_phase2a(root):
    path = root / "Phase2_outputs" / "phase2a_continuum_readiness.json"
    if not path.exists():
        return {}
    data = load_json(path)
    return {s.get("fname"): s for s in data.get("scenes", [])}


def short_names(scenes):
    names = []
    used = {}
    for s in scenes:
        name = s["scene"]
        used[name] = used.get(name, 0) + 1
        if used[name] > 1:
            name = f"{name} {used[name]}"
        names.append(name)
    return names


def fig_snr_summary(scenes, out_dir):
    names = short_names(scenes)
    raw = [s["report"]["metric1_snr"]["mean_snr_raw"] for s in scenes]
    phys = [s["report"]["metric1_snr"]["mean_snr_physics"] for s in scenes]
    den = [s["report"]["metric1_snr"]["mean_snr_ml"] for s in scenes]
    gain = [s["report"]["metric1_snr"]["mean_improvement_pct"] for s in scenes]

    x = np.arange(len(scenes))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(13.5, 7.5))
    ax1.bar(x - width, raw, width, label="Raw", color="#7aa6c2", edgecolor="black", linewidth=0.5)
    ax1.bar(x, phys, width, label="Physics corrected", color="#d9a441", edgecolor="black", linewidth=0.5)
    ax1.bar(x + width, den, width, label="Denoised", color="#5aa469", edgecolor="black", linewidth=0.5)
    ax1.axhline(10, color="#b22222", linestyle="--", linewidth=1.2, label="SNR = 10")
    ax1.set_ylabel("Mean SNR")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=28, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, gain, marker="o", color="#2f2f2f", linewidth=2.0, label="SNR gain")
    ax2.set_ylabel("SNR improvement (%)")
    ax2.axhline(0, color="gray", linewidth=0.8)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, ncol=2)
    ax1.set_title("Phase 1 Denoising Performance: Mean SNR")
    fig.tight_layout()
    save_figure(fig, out_dir, "Fig01_mean_snr_raw_physics_denoised")


def fig_spectral_preservation(scenes, out_dir):
    names = short_names(scenes)
    sam = [s["report"]["metric2_sam"]["mean_sam_deg"] for s in scenes]
    pear = [s["report"]["metric4_pearson"]["mean_r"] for s in scenes]
    band_pass = [s["report"]["band_decisions"]["pct_pass"] for s in scenes]

    x = np.arange(len(scenes))
    fig, ax1 = plt.subplots(figsize=(13.5, 7.5))
    ax1.bar(x, band_pass, color="#8fb996", edgecolor="black", linewidth=0.5, label="Band pass rate")
    ax1.set_ylabel("Bands passing SNR/noise criteria (%)")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=28, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, sam, marker="o", color="#b22222", linewidth=2.0, label="Mean SAM")
    ax2.plot(x, [(1 - p) * 100 for p in pear], marker="s", color="#1f4e79",
             linewidth=2.0, label="(1 - Pearson) x 100")
    ax2.set_ylabel("Spectral distortion metric")
    ax2.set_ylim(0, max(5.0, max(sam) * 1.4))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    ax1.set_title("Spectral Preservation After Denoising")
    fig.tight_layout()
    save_figure(fig, out_dir, "Fig02_spectral_preservation_sam_pearson")


def fig_readiness_matrix(scenes, phase2a, out_dir):
    names = short_names(scenes)
    matrix = np.full((len(scenes), len(FEATURE_ORDER)), -1, dtype=float)
    labels = []
    for i, scene in enumerate(scenes):
        p2 = phase2a.get(scene["fname"], {})
        labels.append(p2.get("phase2a_readiness", "UNKNOWN").replace("_", " "))
        features = p2.get("features", {})
        for j, feature in enumerate(FEATURE_ORDER):
            status = features.get(feature, {}).get("status", "N/A")
            matrix[i, j] = STATUS_SCORE.get(status, -1)

    cmap = ListedColormap(["#d9d9d9", "#c0392b", "#f1c40f", "#2e8b57"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(FEATURE_ORDER)))
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_ORDER])
    ax.set_yticks(np.arange(len(scenes)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Continuum-removed mineral feature")
    ax.set_title("Phase 2A Mineral-Feature Preservation Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            status = {2: "PASS", 1: "REVIEW", 0: "FAIL", -1: "N/A"}.get(int(matrix[i, j]), "")
            ax.text(j, i, status, ha="center", va="center", fontsize=12,
                    color="white" if matrix[i, j] in (0, 2) else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, ticks=[-1, 0, 1, 2])
    cbar.ax.set_yticklabels(["N/A", "Fail", "Review", "Pass"])
    fig.tight_layout()
    save_figure(fig, out_dir, "Fig03_phase2a_feature_preservation_matrix")


def fig_band_depth_change_heatmap(scenes, phase2a, out_dir):
    names = short_names(scenes)
    matrix = np.full((len(scenes), len(FEATURE_ORDER)), np.nan, dtype=float)
    for i, scene in enumerate(scenes):
        features = phase2a.get(scene["fname"], {}).get("features", {})
        for j, feature in enumerate(FEATURE_ORDER):
            val = features.get(feature, {}).get("bd_change_pct")
            if isinstance(val, (int, float)):
                matrix[i, j] = float(val)

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    lim = 40
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(np.arange(len(FEATURE_ORDER)))
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_ORDER])
    ax.set_yticks(np.arange(len(scenes)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Continuum-removed mineral feature")
    ax.set_title("Median Band-Depth Change After Denoising")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}%", ha="center", va="center", fontsize=12)
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=12, color="gray")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Band-depth change (%)")
    fig.tight_layout()
    save_figure(fig, out_dir, "Fig04_band_depth_change_heatmap")


def fig_noise_suppression(scenes, out_dir):
    names = short_names(scenes)
    metrics = ["N1_Gaussian", "N2_Stripe", "N3_Spikes", "N4_Thermal", "N5_Smile"]
    labels = ["Gaussian", "Stripe", "Spikes", "Thermal", "Smile"]
    matrix = np.full((len(scenes), len(metrics)), np.nan, dtype=float)
    for i, scene in enumerate(scenes):
        noise = scene["report"].get("metric5_noise", {})
        for j, key in enumerate(metrics):
            val = noise.get(key)
            if isinstance(val, (int, float)):
                matrix[i, j] = float(val)

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(scenes)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Noise class")
    ax.set_title("Noise Suppression by Detected Noise Class")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.0f}%", ha="center", va="center", fontsize=12)
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=12, color="gray")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Suppression (%)")
    fig.tight_layout()
    save_figure(fig, out_dir, "Fig05_noise_suppression_matrix")


def fig_readiness_summary(scenes, phase2a, out_dir):
    names = short_names(scenes)
    readiness = []
    for scene in scenes:
        p2_ready = phase2a.get(scene["fname"], {}).get("phase2a_readiness", "UNKNOWN")
        manifest_ready = scene.get("manifest", {}).get("readiness", "UNKNOWN")
        if p2_ready == "READY_FOR_PHASE2" and manifest_ready == "READY":
            readiness.append(2)
        elif "BORDERLINE" in p2_ready or manifest_ready == "BORDERLINE":
            readiness.append(1)
        else:
            readiness.append(0)

    colors = ["#c0392b" if r == 0 else "#f1c40f" if r == 1 else "#2e8b57" for r in readiness]
    labels = ["Not ready" if r == 0 else "Borderline" if r == 1 else "Ready" for r in readiness]
    x = np.arange(len(scenes))
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    bars = ax.bar(x, readiness, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 2.7)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Not ready", "Borderline", "Ready"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.set_ylabel("Mineral-identification readiness")
    ax.set_title("Final Phase 1 Scene Readiness for Mineral Identification")
    ax.grid(axis="y", alpha=0.25)
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                label, ha="center", va="bottom", fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir, "Fig06_final_scene_readiness")


def main():
    parser = argparse.ArgumentParser(description="Generate journal-style Phase 1 denoising figures")
    parser.add_argument("--root", default=str(JOURNAL_ROOT), help="M3 journal root")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    paper_style()

    scenes = load_latest_scene_data(root)
    phase2a = load_phase2a(root)
    if not scenes:
        raise SystemExit("No scenes found in pipeline_summary.json")

    fig_snr_summary(scenes, out_dir)
    fig_spectral_preservation(scenes, out_dir)
    fig_readiness_matrix(scenes, phase2a, out_dir)
    fig_band_depth_change_heatmap(scenes, phase2a, out_dir)
    fig_noise_suppression(scenes, out_dir)
    fig_readiness_summary(scenes, phase2a, out_dir)

    print(f"\nGenerated {6} paper figures in: {out_dir}")


if __name__ == "__main__":
    main()
