"""
M3 Phase 2 — Stage 1 spectral library (RELAB-first + USGS lunar filter).

Builds continuum-removed, L2-normalized spectra on the Global-85 M3 grid.
Run Gate1 via:  python m3_spectral_library.py --build
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import m3_mineral_constants as C
from m3_spm import compute_spm_vector, continuum_removal_match, nearest_band


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_relab_tab(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """Return (waves_um, reflectance, sample_id) or None."""
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not text:
        return None
    try:
        npts = int(text[0].strip().split()[0])
    except ValueError:
        return None

    waves_nm, refl = [], []
    for line in text[1:]:
        s = line.strip()
        if not s:
            if waves_nm:
                break
            continue
        # metadata starts with letters / non-numeric
        parts = s.split()
        try:
            w = float(parts[0])
            r = float(parts[1])
        except (ValueError, IndexError):
            if waves_nm:
                break
            continue
        waves_nm.append(w)
        refl.append(r)
        if len(waves_nm) >= npts:
            break

    if len(waves_nm) < 20:
        return None
    waves = np.asarray(waves_nm, dtype=np.float64) / 1000.0
    refl_a = np.asarray(refl, dtype=np.float64)
    ok = np.isfinite(waves) & np.isfinite(refl_a) & (refl_a >= 0) & (refl_a <= 1.5)
    if ok.sum() < 20:
        return None
    sample_id = path.stem
    # try metadata ID
    for line in text[-40:]:
        if "LR-CMP" in line or "LS-CMP" in line:
            sample_id = line.strip().split()[0]
            break
    return waves[ok], refl_a[ok], sample_id


def parse_usgs_ascii(path: Path) -> Optional[np.ndarray]:
    """Parse USGS spectrum values (skip title); return 1-D float array."""
    vals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        next(f, None)  # title
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                v = float(s.split()[0])
            except ValueError:
                continue
            vals.append(v)
    if len(vals) < 20:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    arr[np.abs(arr - C.USGS_DELETE) < 1e20] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < -0.05) | (arr > 1.5)] = np.nan
    return arr


def load_usgs_m3t_waves() -> np.ndarray:
    vals = []
    with C.USGS_M3T_WAVES.open("r", encoding="utf-8", errors="ignore") as f:
        next(f, None)
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s.split()[0]))
            except ValueError:
                continue
    return np.asarray(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Family assignment / filters
# ---------------------------------------------------------------------------

def _whole_word(hay: str, needle: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", hay) is not None


def assign_family_from_name(name: str) -> Optional[str]:
    low = name.lower().replace("-", "_").replace(" ", "_")
    # blocklist first
    for b in C.USGS_BLOCKLIST:
        if b in low:
            return None
    # RELAB filename stems
    stem = Path(name).stem.lower()
    if stem.startswith("olivine"):
        return "Olivine"
    if stem.startswith("orthopyroxene") or stem.startswith("pigeonite"):
        return "LCP"
    if stem.startswith("clinopyroxene"):
        return "HCP"
    if stem.startswith("plagioclase") or stem.startswith("anorthosite"):
        return "Plagioclase"
    if stem.startswith("ilmenite"):
        return "Ilmenite"

    for fam, keys in C.USGS_FAMILY_KEYWORDS.items():
        for k in keys:
            if _whole_word(low.replace("_", " "), k) or k in low:
                return fam
    return None


# ---------------------------------------------------------------------------
# SRF / resample / CR
# ---------------------------------------------------------------------------

def gaussian_srf_convolve(
    src_waves: np.ndarray,
    src_refl: np.ndarray,
    tgt_waves: np.ndarray,
) -> np.ndarray:
    """Convolve high-res spectrum onto M3 band centers with Gaussian SRF."""
    out = np.full(len(tgt_waves), np.nan, dtype=np.float64)
    for i, wc in enumerate(tgt_waves):
        fwhm = C.SRF_FWHM_VNIR if wc < C.SRF_SWIR_ONSET else C.SRF_FWHM_SWIR
        sigma = fwhm / 2.355
        # local window ±3 sigma
        m = (src_waves >= wc - 3 * sigma) & (src_waves <= wc + 3 * sigma)
        if m.sum() < 3:
            # nearest interpolate fallback
            if np.isfinite(src_refl).sum() >= 5:
                out[i] = float(np.interp(wc, src_waves[np.isfinite(src_refl)],
                                         src_refl[np.isfinite(src_refl)],
                                         left=np.nan, right=np.nan))
            continue
        w = src_waves[m]
        r = src_refl[m]
        ok = np.isfinite(r)
        if ok.sum() < 3:
            continue
        w, r = w[ok], r[ok]
        ker = np.exp(-0.5 * ((w - wc) / sigma) ** 2)
        out[i] = float(np.sum(ker * r) / (np.sum(ker) + 1e-12))
    return out


def resample_linear(src_waves: np.ndarray, src_refl: np.ndarray, tgt_waves: np.ndarray) -> np.ndarray:
    ok = np.isfinite(src_refl) & np.isfinite(src_waves)
    if ok.sum() < 5:
        return np.full(len(tgt_waves), np.nan)
    return np.interp(tgt_waves, src_waves[ok], src_refl[ok], left=np.nan, right=np.nan)


def l2_normalize(spec: np.ndarray) -> np.ndarray:
    out = np.asarray(spec, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if finite.sum() < 5:
        return np.full_like(out, np.nan)
    nrm = np.linalg.norm(out[finite])
    if nrm < 1e-12:
        return np.full_like(out, np.nan)
    out[finite] = out[finite] / nrm
    out[~finite] = np.nan
    return out


def quality_ok(spec: np.ndarray, waves: np.ndarray) -> bool:
    finite = np.isfinite(spec)
    if finite.mean() < 0.80:
        return False
    # adjacent jump check
    s = spec.copy()
    # fill for jump test
    if finite.sum() >= 5:
        idx = np.where(finite)[0]
        s[~finite] = np.interp(np.where(~finite)[0], idx, spec[idx])
    d = np.abs(np.diff(s))
    if np.nanmax(d) > 0.5:
        return False
    # diagnostic coverage: key SPM wavelengths
    keys = [0.75, 1.05, 1.25, 1.50, 1.82, 2.14, 2.21]
    covered = 0
    for k in keys:
        j = nearest_band(waves, k)
        if np.isfinite(spec[j]):
            covered += 1
    return covered >= 5


# ---------------------------------------------------------------------------
# Build library
# ---------------------------------------------------------------------------

def load_relab_entries(tgt_waves: np.ndarray) -> List[dict]:
    entries = []
    for path in sorted(C.RELAB_DIR.glob("*.tab")):
        parsed = parse_relab_tab(path)
        if parsed is None:
            continue
        sw, sr, sid = parsed
        fam = assign_family_from_name(path.name)
        if fam is None:
            continue
        conv = gaussian_srf_convolve(sw, sr, tgt_waves)
        if not quality_ok(conv, tgt_waves):
            continue
        cr = continuum_removal_match(conv, tgt_waves)
        cr_l2 = l2_normalize(cr)
        if not np.isfinite(cr_l2).any():
            continue
        spm = compute_spm_vector(cr, tgt_waves, spectrum_refl=conv)
        entries.append({
            "name": f"RELAB_{path.stem}",
            "source": "RELAB",
            "sample_id": sid,
            "family": fam,
            "spectrum_cr": cr.astype(np.float32),
            "spectrum_cr_l2": cr_l2.astype(np.float32),
            "spm": spm.astype(np.float32),
        })
    return entries


def load_usgs_entries(tgt_waves: np.ndarray) -> List[dict]:
    if not C.USGS_M3T_MINERALS.is_dir():
        print(f"  [warn] USGS M3-target minerals missing: {C.USGS_M3T_MINERALS}")
        return []
    m3t_waves = load_usgs_m3t_waves()
    print(f"  USGS M3-target waves: {len(m3t_waves)} bands "
          f"({m3t_waves[0]:.3f}-{m3t_waves[-1]:.3f} µm)")

    per_fam: Dict[str, int] = {f: 0 for f in C.USGS_FAMILY_KEYWORDS}
    entries = []
    paths = sorted(C.USGS_M3T_MINERALS.glob("*.txt"))
    # Prefer AREF over RREF when both exist — keep AREF
    paths = [p for p in paths if "AREF" in p.name.upper() or "RREF" not in p.name.upper()]

    for path in paths:
        fam = assign_family_from_name(path.name)
        if fam is None or fam not in per_fam:
            continue
        if per_fam[fam] >= C.USGS_MAX_PER_FAMILY:
            continue
        raw = parse_usgs_ascii(path)
        if raw is None or len(raw) != len(m3t_waves):
            continue
        # resample M3-target → Global 85
        conv = resample_linear(m3t_waves, raw, tgt_waves)
        if not quality_ok(conv, tgt_waves):
            continue
        cr = continuum_removal_match(conv, tgt_waves)
        cr_l2 = l2_normalize(cr)
        if not np.isfinite(cr_l2).any():
            continue
        spm = compute_spm_vector(cr, tgt_waves, spectrum_refl=conv)
        entries.append({
            "name": f"USGS_{path.stem}",
            "source": "USGS_M3t",
            "sample_id": path.stem,
            "family": fam,
            "spectrum_cr": cr.astype(np.float32),
            "spectrum_cr_l2": cr_l2.astype(np.float32),
            "spm": spm.astype(np.float32),
        })
        per_fam[fam] += 1
    print(f"  USGS kept per family: {per_fam}")
    return entries


def gate1_library_qc(entries: List[dict]) -> dict:
    from collections import Counter
    fam_counts = Counter(e["family"] for e in entries)
    core_ok = all(fam_counts.get(f, 0) >= C.LIB_MIN_PER_CORE_FAMILY for f in C.CORE_FAMILIES)
    pass_g1 = (len(entries) >= C.LIB_MIN_TOTAL) and core_ok
    report = {
        "gate": "G1_Library_QC",
        "pass": bool(pass_g1),
        "n_spectra": len(entries),
        "min_required": C.LIB_MIN_TOTAL,
        "family_counts": dict(fam_counts),
        "core_families_ok": bool(core_ok),
        "relab": sum(1 for e in entries if e["source"] == "RELAB"),
        "usgs": sum(1 for e in entries if e["source"].startswith("USGS")),
        "reasons": [],
    }
    if len(entries) < C.LIB_MIN_TOTAL:
        report["reasons"].append(f"n_spectra={len(entries)} < {C.LIB_MIN_TOTAL}")
    if not core_ok:
        report["reasons"].append(f"core family shortfall: {dict(fam_counts)}")
    return report


def build_library(tgt_waves: Optional[np.ndarray] = None, save: bool = True) -> dict:
    waves = np.asarray(tgt_waves if tgt_waves is not None else C.M3_WAVELENGTHS_85_UM, dtype=np.float64)
    print("=" * 60)
    print("M3 Stage1 — Spectral library build")
    print(f"  Target bands: {len(waves)}  ({waves[0]:.3f}-{waves[-1]:.3f} µm)")
    print(f"  RELAB: {C.RELAB_DIR}")
    print(f"  USGS:  {C.USGS_M3T_MINERALS}")

    relab = load_relab_entries(waves)
    print(f"  RELAB kept: {len(relab)}")
    usgs = load_usgs_entries(waves)
    print(f"  USGS kept:  {len(usgs)}")
    entries = relab + usgs

    report = gate1_library_qc(entries)
    print(f"  Gate1: {'PASS' if report['pass'] else 'FAIL'}  "
          f"n={report['n_spectra']}  families={report['family_counts']}")
    if report["reasons"]:
        for r in report["reasons"]:
            print(f"    - {r}")

    out = {
        "waves": waves.astype(np.float64),
        "names": [e["name"] for e in entries],
        "families": [e["family"] for e in entries],
        "sources": [e["source"] for e in entries],
        "spectra_cr": np.stack([e["spectrum_cr"] for e in entries], axis=0) if entries else np.zeros((0, len(waves)), np.float32),
        "spectra_cr_l2": np.stack([e["spectrum_cr_l2"] for e in entries], axis=0) if entries else np.zeros((0, len(waves)), np.float32),
        "spm": np.stack([e["spm"] for e in entries], axis=0) if entries else np.zeros((0, C.N_SPM), np.float32),
        "gate1": report,
    }

    if save:
        C.SHARED_DIR.mkdir(parents=True, exist_ok=True)
        C.GATES_DIR.mkdir(parents=True, exist_ok=True)
        np.save(C.SHARED_DIR / "lib_m3_B85_cr.npy", out["spectra_cr"])
        np.save(C.SHARED_DIR / "lib_m3_B85_cr_l2.npy", out["spectra_cr_l2"])
        np.save(C.SHARED_DIR / "lib_m3_B85_spm.npy", out["spm"])
        np.save(C.SHARED_DIR / "lib_m3_B85_waves.npy", out["waves"])
        meta = {
            "names": out["names"],
            "families": out["families"],
            "sources": out["sources"],
            "spm_names": C.SPM_NAMES,
            "n_bands": int(len(waves)),
            "method_version": C.METHOD_VERSION,
            "cr_method": "continuum_removal_match",
            "cr_filter_size": C.CR_MATCH_FILTER_SIZE,
        }
        (C.SHARED_DIR / "lib_names.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (C.GATES_DIR / "G1_library_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        # CSV manifest
        with (C.SHARED_DIR / "lib_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "source", "family"])
            for n, s, fam in zip(out["names"], out["sources"], out["families"]):
                w.writerow([n, s, fam])
        print(f"  Saved -> {C.SHARED_DIR}")
        print(f"  Gate1 -> {C.GATES_DIR / 'G1_library_qc.json'}")

    return out


def load_built_library() -> dict:
    meta = json.loads((C.SHARED_DIR / "lib_names.json").read_text(encoding="utf-8"))
    return {
        "waves": np.load(C.SHARED_DIR / "lib_m3_B85_waves.npy"),
        "names": meta["names"],
        "families": meta["families"],
        "sources": meta["sources"],
        "spectra_cr": np.load(C.SHARED_DIR / "lib_m3_B85_cr.npy"),
        "spectra_cr_l2": np.load(C.SHARED_DIR / "lib_m3_B85_cr_l2.npy"),
        "spm": np.load(C.SHARED_DIR / "lib_m3_B85_spm.npy"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="Build and Gate1-check library")
    args = ap.parse_args()
    if args.build or True:
        build_library(save=True)


if __name__ == "__main__":
    main()
