"""
Journal figures for M3 Phase-2 mineral ID (current scene results).

Style: Times New Roman, base 16 pt, white background, no overlapping labels.
Outputs: D:\\Journal\\M3 Journal\\Paper_Figures\\mineral_id\\
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import m3_mineral_constants as C

OUT_ROOT = C.JOURNAL_ROOT / "Paper_Figures" / "mineral_id"
SCENES = [
    "Copernicus Crater",
    "Mare Tranquillitatis",
    "Lunar Highlands South",
    "Mare Imbrium",
    "Mare Serenitatis",
    "Jackson Crater",
]
CORE_GROUPS = [
    "Olivine",
    "Low-Ca Pyroxene",
    "High-Ca Pyroxene",
    "Plagioclase",
]
GROUP_SHORT = {
    "Olivine": "Ol",
    "Low-Ca Pyroxene": "LCP",
    "High-Ca Pyroxene": "HCP",
    "Plagioclase": "Plag",
    "Ilmenite": "Ilm",
    "Spinel": "Sp",
}
GROUP_COLOR = {
    "Olivine": "#2E7D32",
    "Low-Ca Pyroxene": "#1565C0",
    "High-Ca Pyroxene": "#C62828",
    "Plagioclase": "#6A1B9A",
    "Ilmenite": "#546E7A",
    "Spinel": "#EF6C00",
}
DEPTH_WL = {
    "d1.05": 1.05,
    "d1.25": 1.25,
    "d1.00": 1.00,
    "d1.90": 1.90,
    "d2.25": 2.25,
}
# claim codes for matrix
CLAIM_CODE = {
    "ABSENT": 0,
    "CANDIDATE": 1,
    "PRESENT_WEAK": 2,
    "PRESENT": 3,
    "PRESENT_RARE": 4,
}
CLAIM_LABELS = ["ABSENT", "CANDIDATE", "PRESENT_WEAK", "PRESENT", "PRESENT_RARE"]


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 18,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"  wrote {path}")


def _scene_dir(scene: str) -> Path:
    return C.JOURNAL_ROOT / scene / "Pipeline_outputs" / "06_mineral_id"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None or not np.isfinite(float(v)):
        return "n/a"
    return f"{float(v):.{nd}f}"


def _mean_cr(cube: np.ndarray, mask: np.ndarray, nmax: int = 2500) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    if len(ys) > nmax:
        rng = np.random.default_rng(0)
        ii = rng.choice(len(ys), nmax, replace=False)
        ys, xs = ys[ii], xs[ii]
    specs = np.asarray(cube[ys, xs, :], dtype=np.float64)
    ok = np.isfinite(specs).sum(axis=1) >= 20
    if not ok.any():
        return None
    with np.errstate(all="ignore"):
        m = np.nanmean(specs[ok], axis=0)
    if not np.isfinite(m).any():
        return None
    return m.astype(np.float32)


def _downsample_map(arr: np.ndarray, max_rows: int = 900) -> np.ndarray:
    h = arr.shape[0]
    if h <= max_rows:
        return arr
    step = int(np.ceil(h / max_rows))
    return arr[::step]


def load_scene(scene: str) -> dict:
    d = _scene_dir(scene)
    proof = _load_json(d / "mineral_proof.json")
    summary = _load_json(d / "run_summary.json")
    rare = _load_json(d / "rare_mineral_candidates.json") if (d / "rare_mineral_candidates.json").exists() else {}
    gate = _load_json(d / "spm_gate_info.json") if (d / "spm_gate_info.json").exists() else {}
    return {
        "scene": scene,
        "dir": d,
        "proof": proof,
        "summary": summary,
        "rare": rare.get("candidates") or summary.get("rare") or {},
        "gate": gate,
        "waves": np.asarray(C.M3_WAVELENGTHS_85_UM, dtype=np.float64),
        "spm_names": list(C.SPM_NAMES),
    }


def claim_level_for(scene_data: dict, group: str) -> str:
    minerals = scene_data["proof"].get("minerals") or {}
    if group in ("Ilmenite", "Spinel"):
        rare = scene_data["rare"].get(group) or {}
        st = str(rare.get("status") or "ABSENT")
        if st in CLAIM_CODE:
            return st
        return "ABSENT"
    ent = minerals.get(group) or {}
    lvl = str(ent.get("present_level") or "ABSENT")
    if lvl in CLAIM_CODE:
        return lvl
    if ent.get("present"):
        return "PRESENT"
    return "ABSENT"


# ---------------------------------------------------------------------------
# Figure set
# ---------------------------------------------------------------------------

def fig01_claim_matrix(all_data: List[dict]) -> None:
    """Cross-scene mineral claim matrix."""
    cols = CORE_GROUPS + ["Ilmenite", "Spinel"]
    mat = np.zeros((len(all_data), len(cols)), dtype=int)
    for i, data in enumerate(all_data):
        for j, g in enumerate(cols):
            mat[i, j] = CLAIM_CODE[claim_level_for(data, g)]

    cmap = ListedColormap(
        ["#ECEFF1", "#FFE082", "#90CAF9", "#43A047", "#00897B"]
    )
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    im = ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([GROUP_SHORT.get(c, c) for c in cols], rotation=0)
    ax.set_yticks(range(len(all_data)))
    ax.set_yticklabels([d["scene"] for d in all_data])
    ax.set_xlabel("Mineral family")
    ax.set_ylabel("Scene")
    ax.set_title("Mineral claims across scenes", pad=14, fontweight="bold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j, i, CLAIM_LABELS[mat[i, j]].replace("PRESENT_", "P_"),
                ha="center", va="center", fontsize=11, color="#111111",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(CLAIM_LABELS)
    cbar.set_label("Claim level")
    fig.subplots_adjust(left=0.28, right=0.88, top=0.88, bottom=0.14)
    _save(fig, OUT_ROOT / "F01_cross_scene_claim_matrix.png")


def _gate1_pass(summary: dict) -> bool:
    """Older refreshes may omit gate1 in run_summary; fall back to shared G1 QC."""
    g1 = summary.get("gate1")
    if isinstance(g1, dict) and "pass" in g1:
        return bool(g1.get("pass"))
    shared = C.GATES_DIR / "G1_library_qc.json"
    if shared.exists():
        try:
            return bool(_load_json(shared).get("pass"))
        except Exception:
            pass
    # Library was used if later gates exist
    return bool((summary.get("gate2") or {}).get("pass"))


def fig02_stage_gates(all_data: List[dict]) -> None:
    """G0–G4 + clustering/unmix QA per scene."""
    stages = ["G0", "G1", "G2", "G3", "G4", "Cluster", "Unmix"]
    mat = np.zeros((len(all_data), len(stages)), dtype=int)
    annot = []
    for i, data in enumerate(all_data):
        s = data["summary"]
        row_ann = []
        checks = [
            bool((s.get("gate0") or {}).get("pass", True)),
            _gate1_pass(s),
            bool((s.get("gate2") or {}).get("pass")),
            bool((s.get("gate3") or {}).get("pass")),
            bool((s.get("gate4") or {}).get("pass")),
            str(s.get("clustering_status") or "").upper() == "PASS",
            str(s.get("mixture_qa") or "").upper() == "PASS",
        ]
        for j, ok in enumerate(checks):
            mat[i, j] = 1 if ok else 0
            row_ann.append("PASS" if ok else "FAIL")
        annot.append(row_ann)

    cmap = ListedColormap(["#EF9A9A", "#A5D6A7"])
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_yticks(range(len(all_data)))
    ax.set_yticklabels([d["scene"] for d in all_data])
    ax.set_xlabel("Pipeline stage / QA")
    ax.set_ylabel("Scene")
    ax.set_title("Stage results by scene", pad=14, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=13)
    ax.legend(
        handles=[
            Patch(facecolor="#A5D6A7", edgecolor="none", label="PASS"),
            Patch(facecolor="#EF9A9A", edgecolor="none", label="FAIL"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.28, right=0.96, top=0.88, bottom=0.22)
    _save(fig, OUT_ROOT / "F02_stage_gate_results.png")


def fig03_verdict_summary(all_data: List[dict]) -> None:
    """Overall verdict + n PRESENT strong/weak/rare."""
    scenes = [d["scene"] for d in all_data]
    strong, weak, rare_n = [], [], []
    verdicts = []
    for d in all_data:
        p = d["proof"]
        strong.append(len(p.get("present_strong") or []))
        weak.append(len(p.get("present_weak") or []))
        rn = sum(
            1
            for k, v in (d["rare"] or {}).items()
            if str((v or {}).get("status")) == "PRESENT_RARE"
        )
        rare_n.append(rn)
        verdicts.append(str(p.get("overall_verdict") or d["summary"].get("overall_verdict") or ""))

    x = np.arange(len(scenes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    ax.bar(x - w, strong, width=w, color="#43A047", label="PRESENT (strong)")
    ax.bar(x, weak, width=w, color="#90CAF9", label="PRESENT_WEAK")
    ax.bar(x + w, rare_n, width=w, color="#00897B", label="PRESENT_RARE")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=25, ha="right")
    ax.set_ylabel("Mineral count")
    ax.set_title("Claim counts and overall verdict by scene", pad=14, fontweight="bold")
    ymax = max(strong + weak + rare_n + [1]) + 1.8
    ax.set_ylim(0, ymax)
    for i, v in enumerate(verdicts):
        ax.text(i, ymax - 0.55, v, ha="center", va="top", fontsize=13, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.32)
    _save(fig, OUT_ROOT / "F03_claim_counts_verdict.png")


def fig04_present_proof_panel(data: dict) -> None:
    """
    Proof figure: for each claimed mineral (strong/weak),
    SPM map + hotspot contour | hot vs bland match-CR | metric bars.
    """
    proof = data["proof"]
    minerals = proof.get("minerals") or {}
    claimed = []
    for g in CORE_GROUPS:
        ent = minerals.get(g) or {}
        if ent.get("present"):
            claimed.append((g, ent))
    # rare ilmenite if PRESENT_RARE
    rare_ilm = (data["rare"] or {}).get("Ilmenite") or {}
    if str(rare_ilm.get("status")) == "PRESENT_RARE":
        # synthesize a minimal entry for map from proof Ilmenite thr if present
        ilm_ent = minerals.get("Ilmenite") or {
            "spm_index": "R750",
            "thr_adaptive": 0.05,
            "proof_depth_key": "d1.00",
            "band_depths_hot": {},
            "hot_minus_bland": None,
            "best_sam": None,
            "best_library": "",
            "n_hot_pixels": rare_ilm.get("n_sam_hits_gated"),
            "present_level": "PRESENT_RARE",
        }
        ilm_ent = dict(ilm_ent)
        ilm_ent["present_level"] = "PRESENT_RARE"
        ilm_ent["n_hot_pixels"] = int(rare_ilm.get("n_sam_hits_gated") or 0)
        ilm_ent["_rare_examples"] = rare_ilm.get("examples") or []
        claimed.append(("Ilmenite", ilm_ent))

    if not claimed:
        print(f"  skip proof panel: {data['scene']} (no claims)")
        return

    d = data["dir"]
    spm = np.load(d / "spm_map.npy", mmap_mode="r")
    cube_cr = np.load(d / "cube_cr.npy", mmap_mode="r")
    waves = data["waves"]
    names = data["spm_names"]
    valid = np.any(np.isfinite(spm), axis=2)

    n = len(claimed)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(16.5, 4.6 * n),
        gridspec_kw={"width_ratios": [1.05, 1.35, 0.95], "wspace": 0.38, "hspace": 0.55},
        squeeze=False,
    )

    # bland mask from enabled SPM max
    gate = data["gate"]
    en_idx = [names.index(nm) for nm, g in gate.items() if g.get("enabled") and nm in names]
    if en_idx:
        mx = np.nanmax(np.asarray(spm[:, :, en_idx], dtype=np.float64), axis=2)
    else:
        mx = np.nanmax(np.asarray(spm, dtype=np.float64), axis=2)
    finite = valid & np.isfinite(mx)
    thr_b = float(np.percentile(mx[finite], 10)) if finite.any() else 0.0
    bland = finite & (mx <= thr_b)

    for row, (group, ent) in enumerate(claimed):
        axm, axs, axb = axes[row, 0], axes[row, 1], axes[row, 2]
        color = GROUP_COLOR.get(group, "#333333")
        level = str(ent.get("present_level") or "PRESENT")
        spm_name = str(ent.get("spm_index") or "")
        thr = float(ent.get("thr_adaptive") or C.SPM_DETECT_THRESH.get(spm_name, 0.01))
        j = names.index(spm_name) if spm_name in names else None

        # --- map ---
        hot = np.zeros(valid.shape, dtype=bool)
        n_hot = int(ent.get("n_hot_pixels") or 0)
        rare_mode = level == "PRESENT_RARE" and group == "Ilmenite"
        ex = ent.get("_rare_examples") or []

        if rare_mode and ex:
            # Rare path: plot gated hit locations (not saturated R750)
            bg = np.asarray(spm[:, :, names.index("BD1000")], dtype=np.float64) if "BD1000" in names else np.asarray(spm[:, :, 0], dtype=np.float64)
            show = _downsample_map(np.where(np.isfinite(bg), bg, np.nan))
            step = max(1, int(np.ceil(bg.shape[0] / max(show.shape[0], 1))))
            im = axm.imshow(show, origin="upper", cmap="Greys", aspect="auto", alpha=0.85)
            ys = [int(e["y"]) / step for e in ex[:80]]
            xs = [int(e["x"]) for e in ex[:80]]
            axm.scatter(xs, ys, s=12, c=color, marker="o", linewidths=0, label="Rare hits")
            for e in ex[:40]:
                y, x = int(e["y"]), int(e["x"])
                if 0 <= y < hot.shape[0] and 0 <= x < hot.shape[1]:
                    y0, y1 = max(0, y - 2), min(hot.shape[0], y + 3)
                    x0, x1 = max(0, x - 2), min(hot.shape[1], x + 3)
                    hot[y0:y1, x0:x1] = True
            n_hot = int(rare_ilm_n if (rare_ilm_n := ent.get("n_hot_pixels")) else len(ex))
            axm.set_xlabel(f"gated hits = {n_hot:,}   (rare SAM+depth+hmb)", fontsize=13, labelpad=6)
        elif j is not None:
            layer = np.asarray(spm[:, :, j], dtype=np.float64)
            hot = valid & np.isfinite(layer) & (layer >= thr)
            show = _downsample_map(np.where(np.isfinite(layer), layer, np.nan))
            hot_ds = _downsample_map(hot.astype(np.float32))
            vmin = float(np.nanpercentile(show, 5)) if np.isfinite(show).any() else 0.0
            vmax = float(np.nanpercentile(show, 99)) if np.isfinite(show).any() else 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-3
            im = axm.imshow(show, origin="upper", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
            if np.any(hot_ds >= 0.5):
                axm.contour(hot_ds, levels=[0.5], colors=["white"], linewidths=0.6, origin="upper")
            cbar = fig.colorbar(im, ax=axm, fraction=0.046, pad=0.03)
            cbar.set_label(spm_name, fontsize=13)
            cbar.ax.tick_params(labelsize=12)
            n_hot = int(ent.get("n_hot_pixels") or int(hot.sum()))
            axm.set_xlabel(f"n_hot = {n_hot:,}   thr = {thr:.3f}", fontsize=13, labelpad=6)
        else:
            axm.text(0.5, 0.5, "No SPM layer", ha="center", va="center", transform=axm.transAxes)
            axm.set_xlabel("")

        axm.set_xticks([])
        axm.set_yticks([])
        axm.set_title(f"{group}  ({level})", color=color, fontweight="bold", pad=8)

        # --- spectra ---

        spec_h = _mean_cr(cube_cr, hot) if hot.any() else None
        spec_b = _mean_cr(cube_cr, bland) if bland.any() else None
        if spec_h is not None:
            axs.plot(waves, spec_h, color=color, lw=2.0, label="Hot mean CR")
        if spec_b is not None:
            axs.plot(waves, spec_b, color="#616161", lw=1.6, ls="--", label="Bland mean CR")

        dkey = str(ent.get("proof_depth_key") or "")
        d_hot = None
        bd = ent.get("band_depths_hot") or {}
        if dkey in bd and bd[dkey] is not None:
            d_hot = float(bd[dkey])
        hmb = ent.get("hot_minus_bland")
        sam = ent.get("best_sam")
        if dkey in DEPTH_WL and d_hot is not None and np.isfinite(d_hot):
            axs.axvline(DEPTH_WL[dkey], color=color, ls=":", lw=1.3, label=f"{dkey}={_fmt(d_hot)}")

        axs.set_xlim(0.7, 2.6)
        axs.set_xlabel("Wavelength (µm)")
        axs.set_ylabel("Match-CR")
        axs.set_title("Spectral proof (hot vs bland)", fontweight="bold", pad=8)
        axs.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
            fontsize=12,
        )

        # --- metric bars ---
        labels = ["Depth", "Hot−bland", "SAM"]
        vals = [
            float(d_hot) if d_hot is not None and np.isfinite(d_hot) else 0.0,
            float(hmb) if hmb is not None and np.isfinite(float(hmb)) else 0.0,
            float(sam) if sam is not None and np.isfinite(float(sam)) else 0.0,
        ]
        floors = [C.PROOF_DEPTH_BD_MIN, C.PROOF_HOT_MINUS_BLAND, C.FAMILY_SAM_HIT_MAX]
        # SAM is "pass if below floor" — plot as (floor - sam) clipped for display? Better dual note.
        colors = [color, color, color]
        bars = axb.barh(labels, vals, color=colors, height=0.55)
        # reference lines for depth and hmb minima
        axb.axvline(C.PROOF_DEPTH_BD_MIN, color="#9E9E9E", ls="--", lw=1.0)
        axb.axvline(C.PROOF_HOT_MINUS_BLAND, color="#BDBDBD", ls=":", lw=1.0)
        axb.set_xlabel("Value")
        axb.set_title("Proof metrics", fontweight="bold", pad=8)
        xmax = max(vals + [C.PROOF_DEPTH_BD_MIN, C.PROOF_HOT_MINUS_BLAND, 0.05]) * 1.35
        axb.set_xlim(0, xmax)
        axb.text(
            0.0,
            -0.28,
            f"Depth≥{_fmt(C.PROOF_DEPTH_BD_MIN)}  HMB≥{_fmt(C.PROOF_HOT_MINUS_BLAND)}  "
            f"SAM≤{_fmt(C.FAMILY_SAM_HIT_MAX)} (lower better)",
            transform=axb.transAxes,
            fontsize=11,
            va="top",
        )
        for b, v in zip(bars, vals):
            axb.text(v + 0.002, b.get_y() + b.get_height() / 2, _fmt(v, 4), va="center", fontsize=12)

    fig.suptitle(
        f"{data['scene']}  |  {proof.get('fname', '')}  |  Mineral presence proof",
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.08)
    safe = data["scene"].replace(" ", "_")
    _save(fig, OUT_ROOT / "proof_panels" / f"F04_{safe}_present_proof.png")


def fig05_cross_scene_metrics(all_data: List[dict]) -> None:
    """Grouped bars of depth / HMB / SAM for each PRESENT (strong+weak) claim."""
    records = []
    for d in all_data:
        minerals = d["proof"].get("minerals") or {}
        for g in CORE_GROUPS:
            ent = minerals.get(g) or {}
            if not ent.get("present"):
                continue
            dkey = ent.get("proof_depth_key")
            dval = (ent.get("band_depths_hot") or {}).get(dkey)
            records.append(
                {
                    "scene": d["scene"],
                    "group": g,
                    "level": ent.get("present_level"),
                    "depth": float(dval) if dval is not None else np.nan,
                    "hmb": float(ent["hot_minus_bland"])
                    if ent.get("hot_minus_bland") is not None
                    else np.nan,
                    "sam": float(ent["best_sam"]) if ent.get("best_sam") is not None else np.nan,
                }
            )
    if not records:
        return

    # one figure with 3 panels
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 11.5), sharex=True)
    scenes = SCENES
    x = np.arange(len(scenes))
    width = 0.18
    metrics = [("depth", "Continuum depth (hot)"), ("hmb", "Hot − bland depth"), ("sam", "Library SAM (lower better)")]
    for ax, (key, title) in zip(axes, metrics):
        for k, g in enumerate(CORE_GROUPS):
            vals = []
            for sc in scenes:
                hit = next((r for r in records if r["scene"] == sc and r["group"] == g), None)
                vals.append(hit[key] if hit and np.isfinite(hit[key]) else np.nan)
            ax.bar(
                x + (k - 1.5) * width,
                vals,
                width=width,
                color=GROUP_COLOR[g],
                label=GROUP_SHORT[g],
            )
        ax.set_ylabel(title)
        if key in ("depth", "hmb"):
            thr = C.PROOF_DEPTH_BD_MIN if key == "depth" else C.PROOF_HOT_MINUS_BLAND
            ax.axhline(thr, color="#757575", ls="--", lw=1.0)
        if key == "sam":
            ax.axhline(C.FAMILY_SAM_HIT_MAX, color="#757575", ls="--", lw=1.0)
        ax.set_title(title, fontweight="bold", pad=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(scenes, rotation=25, ha="right")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Multi-evidence proof metrics for claimed minerals", fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.14, hspace=0.35)
    _save(fig, OUT_ROOT / "F05_cross_scene_proof_metrics.png")


def fig06_rare_summary(all_data: List[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    rows = ["Ilmenite", "Spinel"]
    mat = np.zeros((len(rows), len(all_data)), dtype=int)
    texts = []
    for j, d in enumerate(all_data):
        col_t = []
        for i, g in enumerate(rows):
            lvl = claim_level_for(d, g)
            mat[i, j] = CLAIM_CODE.get(lvl, 0)
            col_t.append(lvl.replace("PRESENT_", "P_"))
        texts.append(col_t)
    cmap = ListedColormap(["#ECEFF1", "#FFE082", "#90CAF9", "#43A047", "#00897B"])
    im = ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xticks(range(len(all_data)))
    ax.set_xticklabels([d["scene"] for d in all_data], rotation=25, ha="right")
    ax.set_title("Rare-mineral status by scene", fontweight="bold", pad=14)
    for j in range(len(all_data)):
        for i in range(len(rows)):
            ax.text(j, i, texts[j][i], ha="center", va="center", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(CLAIM_LABELS)
    fig.subplots_adjust(left=0.14, right=0.90, top=0.86, bottom=0.28)
    _save(fig, OUT_ROOT / "F06_rare_mineral_status.png")


def fig07_mixture_secondary(all_data: List[dict]) -> None:
    """Secondary unmix context (SI) — not PRESENT proof."""
    scenes, ol, pyx, plag = [], [], [], []
    for d in all_data:
        mix = d["proof"].get("mixture_secondary") or d["summary"].get("mixture_secondary") or {}
        if not mix:
            continue
        scenes.append(d["scene"])
        ol.append(float(mix.get("olivine_mean") or 0.0))
        pyx.append(float(mix.get("pyroxene_mean_LCP_plus_HCP") or 0.0))
        plag.append(float(mix.get("plagioclase_mean") or 0.0))
    if not scenes:
        return
    x = np.arange(len(scenes))
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.bar(x, ol, color=GROUP_COLOR["Olivine"], label="Olivine")
    ax.bar(x, pyx, bottom=ol, color=GROUP_COLOR["Low-Ca Pyroxene"], label="Pyroxene (LCP+HCP)")
    bottom2 = np.asarray(ol) + np.asarray(pyx)
    ax.bar(x, plag, bottom=bottom2, color=GROUP_COLOR["Plagioclase"], label="Plagioclase")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=25, ha="right")
    ax.set_ylabel("Mean abundance (train pixels)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Secondary mixture context (not PRESENT proof)", fontweight="bold", pad=14)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.34)
    _save(fig, OUT_ROOT / "F07_secondary_mixture_stacks.png")


def _mineral_entry(data: dict, group: str) -> Tuple[dict, str]:
    """Return (entry_dict, claim_level) for a mineral in one scene."""
    minerals = data["proof"].get("minerals") or {}
    if group == "Ilmenite":
        rare = (data["rare"] or {}).get("Ilmenite") or {}
        st = str(rare.get("status") or "ABSENT")
        ent = dict(minerals.get("Ilmenite") or {})
        ent["present_level"] = st
        ent["spm_index"] = ent.get("spm_index") or "R750"
        ent["proof_depth_key"] = ent.get("proof_depth_key") or "d1.00"
        if st == "PRESENT_RARE":
            ent["n_hot_pixels"] = int(rare.get("n_sam_hits_gated") or 0)
            ent["_rare_examples"] = rare.get("examples") or []
            ent["present"] = True
        else:
            ent["present"] = False
            ent["_rare_examples"] = rare.get("examples") or []
        return ent, st
    ent = dict(minerals.get(group) or {})
    lvl = str(ent.get("present_level") or "ABSENT")
    return ent, lvl


def fig09_single_mineral_across_scenes(all_data: List[dict], group: str) -> None:
    """
    One mineral × all scenes: SPM hotspot | hot vs bland CR | proof metrics.
    Includes ABSENT scenes so cross-scene comparison is complete.
    """
    color = GROUP_COLOR.get(group, "#333333")
    short = GROUP_SHORT.get(group, group)
    n = len(all_data)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(16.5, 4.35 * n),
        gridspec_kw={"width_ratios": [1.05, 1.35, 0.95], "wspace": 0.40, "hspace": 0.58},
        squeeze=False,
    )
    waves = all_data[0]["waves"]
    names = all_data[0]["spm_names"]

    for row, data in enumerate(all_data):
        axm, axs, axb = axes[row, 0], axes[row, 1], axes[row, 2]
        ent, level = _mineral_entry(data, group)
        claimed = level in ("PRESENT", "PRESENT_WEAK", "PRESENT_RARE")
        title_color = color if claimed else "#757575"

        d = data["dir"]
        spm = np.load(d / "spm_map.npy", mmap_mode="r")
        cube_cr = np.load(d / "cube_cr.npy", mmap_mode="r")
        valid = np.any(np.isfinite(spm), axis=2)
        gate = data["gate"]
        en_idx = [names.index(nm) for nm, g in gate.items() if g.get("enabled") and nm in names]
        if en_idx:
            with np.errstate(all="ignore"):
                mx = np.nanmax(np.asarray(spm[:, :, en_idx], dtype=np.float64), axis=2)
        else:
            with np.errstate(all="ignore"):
                mx = np.nanmax(np.asarray(spm, dtype=np.float64), axis=2)
        finite = valid & np.isfinite(mx)
        thr_b = float(np.percentile(mx[finite], 10)) if finite.any() else 0.0
        bland = finite & (mx <= thr_b)

        spm_name = str(ent.get("spm_index") or "")
        thr = float(ent.get("thr_adaptive") or C.SPM_DETECT_THRESH.get(spm_name, 0.01))
        j = names.index(spm_name) if spm_name in names else None
        hot = np.zeros(valid.shape, dtype=bool)
        n_hot = int(ent.get("n_hot_pixels") or 0)
        rare_mode = level == "PRESENT_RARE" and group == "Ilmenite"
        ex = ent.get("_rare_examples") or []

        if rare_mode and ex:
            bg = (
                np.asarray(spm[:, :, names.index("BD1000")], dtype=np.float64)
                if "BD1000" in names
                else np.asarray(spm[:, :, 0], dtype=np.float64)
            )
            show = _downsample_map(np.where(np.isfinite(bg), bg, np.nan))
            step = max(1, int(np.ceil(bg.shape[0] / max(show.shape[0], 1))))
            axm.imshow(show, origin="upper", cmap="Greys", aspect="auto", alpha=0.85)
            axm.scatter(
                [int(e["x"]) for e in ex[:80]],
                [int(e["y"]) / step for e in ex[:80]],
                s=12, c=color, marker="o", linewidths=0,
            )
            for e in ex[:40]:
                y, x = int(e["y"]), int(e["x"])
                if 0 <= y < hot.shape[0] and 0 <= x < hot.shape[1]:
                    y0, y1 = max(0, y - 2), min(hot.shape[0], y + 3)
                    x0, x1 = max(0, x - 2), min(hot.shape[1], x + 3)
                    hot[y0:y1, x0:x1] = True
            axm.set_xlabel(f"gated hits = {n_hot:,}", fontsize=13, labelpad=6)
        elif j is not None:
            layer = np.asarray(spm[:, :, j], dtype=np.float64)
            hot = valid & np.isfinite(layer) & (layer >= thr)
            show = _downsample_map(np.where(np.isfinite(layer), layer, np.nan))
            hot_ds = _downsample_map(hot.astype(np.float32))
            vmin = float(np.nanpercentile(show, 5)) if np.isfinite(show).any() else 0.0
            vmax = float(np.nanpercentile(show, 99)) if np.isfinite(show).any() else 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-3
            im = axm.imshow(show, origin="upper", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
            if claimed and np.any(hot_ds >= 0.5):
                axm.contour(hot_ds, levels=[0.5], colors=["white"], linewidths=0.6, origin="upper")
            cbar = fig.colorbar(im, ax=axm, fraction=0.046, pad=0.03)
            cbar.set_label(spm_name, fontsize=12)
            cbar.ax.tick_params(labelsize=11)
            n_hot = int(ent.get("n_hot_pixels") or int(hot.sum()))
            axm.set_xlabel(f"n_hot = {n_hot:,}   thr = {thr:.3f}", fontsize=13, labelpad=6)
        else:
            axm.text(0.5, 0.5, "No SPM layer", ha="center", va="center", transform=axm.transAxes)

        axm.set_xticks([])
        axm.set_yticks([])
        axm.set_title(f"{data['scene']}  |  {level}", color=title_color, fontweight="bold", pad=8)

        # Spectra — always try; grey style if not claimed
        line_c = color if claimed else "#9E9E9E"
        spec_h = _mean_cr(cube_cr, hot) if hot.any() else None
        spec_b = _mean_cr(cube_cr, bland) if bland.any() else None
        if spec_h is not None:
            axs.plot(waves, spec_h, color=line_c, lw=2.0, label="Hot mean CR")
        if spec_b is not None:
            axs.plot(waves, spec_b, color="#616161", lw=1.5, ls="--", label="Bland mean CR")
        if spec_h is None and spec_b is None:
            axs.text(0.5, 0.5, "No spectrum", ha="center", va="center", transform=axs.transAxes, color="#9E9E9E")

        dkey = str(ent.get("proof_depth_key") or "")
        bd = ent.get("band_depths_hot") or {}
        d_hot = float(bd[dkey]) if dkey in bd and bd[dkey] is not None else np.nan
        hmb = ent.get("hot_minus_bland")
        hmb_v = float(hmb) if hmb is not None and np.isfinite(float(hmb)) else np.nan
        sam = ent.get("best_sam")
        sam_v = float(sam) if sam is not None and np.isfinite(float(sam)) else np.nan
        if dkey in DEPTH_WL and np.isfinite(d_hot):
            axs.axvline(DEPTH_WL[dkey], color=line_c, ls=":", lw=1.2, label=f"{dkey}={_fmt(d_hot)}")

        axs.set_xlim(0.7, 2.6)
        axs.set_xlabel("Wavelength (µm)")
        axs.set_ylabel("Match-CR")
        axs.set_title("Spectral proof (hot vs bland)", fontweight="bold", pad=8)
        axs.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=2,
            frameon=False,
            fontsize=11,
        )

        labels = ["Depth", "Hot−bland", "SAM"]
        vals = [
            float(d_hot) if np.isfinite(d_hot) else 0.0,
            float(hmb_v) if np.isfinite(hmb_v) else 0.0,
            float(sam_v) if np.isfinite(sam_v) else 0.0,
        ]
        bar_c = color if claimed else "#BDBDBD"
        bars = axb.barh(labels, vals, color=bar_c, height=0.55)
        axb.axvline(C.PROOF_DEPTH_BD_MIN, color="#9E9E9E", ls="--", lw=1.0)
        axb.axvline(C.PROOF_HOT_MINUS_BLAND, color="#BDBDBD", ls=":", lw=1.0)
        xmax = max(vals + [C.PROOF_DEPTH_BD_MIN, C.PROOF_HOT_MINUS_BLAND, 0.05]) * 1.35
        axb.set_xlim(0, xmax)
        axb.set_xlabel("Value")
        axb.set_title("Proof metrics", fontweight="bold", pad=8)
        axb.text(
            0.0,
            -0.30,
            f"Depth≥{_fmt(C.PROOF_DEPTH_BD_MIN)}  HMB≥{_fmt(C.PROOF_HOT_MINUS_BLAND)}  "
            f"SAM≤{_fmt(C.FAMILY_SAM_HIT_MAX)}",
            transform=axb.transAxes,
            fontsize=11,
            va="top",
        )
        for b, v in zip(bars, vals):
            axb.text(v + 0.002, b.get_y() + b.get_height() / 2, _fmt(v, 4), va="center", fontsize=12)

    fig.suptitle(
        f"{group} ({short})  —  presence proof across scenes",
        fontweight="bold",
        y=0.995,
        color=color,
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.04)
    safe = short if short != group else group.replace(" ", "_").replace("-", "")
    out = OUT_ROOT / "single_mineral" / f"F09_{safe}_all_scenes_proof.png"
    _save(fig, out)


def fig08_enabled_spm(all_data: List[dict]) -> None:
    """Which SPM indices were enabled (Gate2) per scene."""
    idx_names = ["OLINDEX3", "LCPINDEX2", "HCPINDEX2", "BD1250"]
    mat = np.zeros((len(all_data), len(idx_names)), dtype=int)
    for i, d in enumerate(all_data):
        g2 = (d["summary"].get("gate2") or {})
        enabled = set(g2.get("enabled_indices") or [])
        for j, nm in enumerate(idx_names):
            mat[i, j] = 1 if nm in enabled else 0
    cmap = ListedColormap(["#FFCDD2", "#C8E6C9"])
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(idx_names)))
    ax.set_xticklabels(idx_names)
    ax.set_yticks(range(len(all_data)))
    ax.set_yticklabels([d["scene"] for d in all_data])
    ax.set_title("Enabled SPM indices (Gate 2)", fontweight="bold", pad=14)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "ON" if mat[i, j] else "off", ha="center", va="center", fontsize=13)
    ax.legend(
        handles=[
            Patch(facecolor="#C8E6C9", label="Enabled"),
            Patch(facecolor="#FFCDD2", label="Disabled"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.28, right=0.96, top=0.88, bottom=0.20)
    _save(fig, OUT_ROOT / "F08_enabled_spm_indices.png")


def run_single_mineral_pages(all_data: List[dict]) -> None:
    groups = CORE_GROUPS + ["Ilmenite"]
    print("F09 single-mineral × all-scenes proof pages...")
    (OUT_ROOT / "single_mineral").mkdir(parents=True, exist_ok=True)
    for g in groups:
        print(f"  {g}")
        fig09_single_mineral_across_scenes(all_data, g)


def main(single_mineral_only: bool = False) -> None:
    _apply_style()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "proof_panels").mkdir(parents=True, exist_ok=True)

    print("Loading scene summaries...")
    all_data = [load_scene(s) for s in SCENES if (_scene_dir(s) / "mineral_proof.json").exists()]
    print(f"  {len(all_data)} scenes")

    if single_mineral_only:
        run_single_mineral_pages(all_data)
        print(f"\nDone. Single-mineral pages in {OUT_ROOT / 'single_mineral'}")
        return

    print("F01 claim matrix")
    fig01_claim_matrix(all_data)
    print("F02 stage gates")
    fig02_stage_gates(all_data)
    print("F03 claim counts")
    fig03_verdict_summary(all_data)
    print("F05 cross-scene metrics")
    fig05_cross_scene_metrics(all_data)
    print("F06 rare summary")
    fig06_rare_summary(all_data)
    print("F07 mixture SI")
    fig07_mixture_secondary(all_data)
    print("F08 enabled SPM")
    fig08_enabled_spm(all_data)

    print("F04 proof panels (per scene; loads cubes)...")
    for data in all_data:
        print(f"  {data['scene']}")
        fig04_present_proof_panel(data)

    run_single_mineral_pages(all_data)

    # manifest
    manifest = {
        "style": {"font": "Times New Roman", "base_size_pt": 16, "background": "white"},
        "out_dir": str(OUT_ROOT),
        "scenes": [d["scene"] for d in all_data],
        "figures": [
            "F01_cross_scene_claim_matrix.png — minerals found per scene",
            "F02_stage_gate_results.png — G0–G4 / cluster / unmix PASS-FAIL",
            "F03_claim_counts_verdict.png — strong/weak/rare counts + verdict",
            "F04_*_present_proof.png — SPM hotspot + CR spectra + proof metrics",
            "F05_cross_scene_proof_metrics.png — depth / HMB / SAM comparison",
            "F06_rare_mineral_status.png — Ilmenite / Spinel",
            "F07_secondary_mixture_stacks.png — unmix SI only",
            "F08_enabled_spm_indices.png — Gate2 SPM enablement",
            "single_mineral/F09_*_all_scenes_proof.png — one mineral × all scenes",
        ],
    }
    (OUT_ROOT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. Figures in {OUT_ROOT}")


if __name__ == "__main__":
    import sys

    main(single_mineral_only=("--single-mineral" in sys.argv))
