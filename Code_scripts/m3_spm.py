"""
M3 lunar SPM indices + continuum removal + diagnostic depths.

Scientific split:
  - SPM band depths / OLINDEX: computed on reflectance (local continuum) — standard practice
  - Encoder / SAM / proof mean spectra: continuum_removal_match (same algo as library)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

import m3_mineral_constants as C

try:
    from scipy.ndimage import maximum_filter1d
except ImportError:  # pragma: no cover
    maximum_filter1d = None


def nearest_band(waves: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(waves - float(target))))


def continuum_removal(spectrum: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """Upper convex-hull continuum removal (steepest-slope walk). Kept for QA."""
    spectrum = np.asarray(spectrum, dtype=np.float64)
    waves = np.asarray(waves, dtype=np.float64)
    finite = np.isfinite(spectrum) & (spectrum > 0) & np.isfinite(waves)
    if finite.sum() < 5:
        return np.full_like(spectrum, np.nan, dtype=np.float32)

    w_valid = waves[finite]
    s_valid = spectrum[finite]
    n = len(w_valid)
    hull_idx = [0]
    while hull_idx[-1] < n - 1:
        i = hull_idx[-1]
        best_j = i + 1
        best_slope = -np.inf
        for j in range(i + 1, n):
            dw = w_valid[j] - w_valid[i]
            if dw <= 0:
                continue
            slope = (s_valid[j] - s_valid[i]) / dw
            if slope > best_slope:
                best_slope = slope
                best_j = j
        hull_idx.append(best_j)

    continuum = np.interp(waves, w_valid[hull_idx], s_valid[hull_idx],
                          left=s_valid[hull_idx[0]], right=s_valid[hull_idx[-1]])
    result = np.full_like(spectrum, np.nan, dtype=np.float32)
    valid = (continuum > 1e-8) & np.isfinite(spectrum)
    result[valid] = (spectrum[valid] / continuum[valid]).astype(np.float32)
    return np.clip(result, 0.0, 2.0)


def continuum_removal_match(
    spectrum: np.ndarray,
    waves: np.ndarray = None,
    size: int = None,
) -> np.ndarray:
    """
    Moving upper-envelope CR (same for library + scene → valid SAM).
    Fast and absorption-preserving for M3 85-band spectra.
    """
    size = int(size or C.CR_MATCH_FILTER_SIZE)
    spectrum = np.asarray(spectrum, dtype=np.float32)
    finite = np.isfinite(spectrum) & (spectrum > 0)
    if finite.sum() < 5:
        return np.full_like(spectrum, np.nan, dtype=np.float32)
    filled = spectrum.copy()
    if not finite.all():
        idx = np.arange(len(spectrum))
        filled[~finite] = np.interp(idx[~finite], idx[finite], spectrum[finite]).astype(np.float32)
    if maximum_filter1d is None:
        # fallback: local max via cumulative trick
        env = filled.copy()
        half = max(1, size // 2)
        for i in range(len(filled)):
            lo, hi = max(0, i - half), min(len(filled), i + half + 1)
            env[i] = float(np.max(filled[lo:hi]))
    else:
        env = maximum_filter1d(filled, size=size, mode="nearest")
    env = np.maximum(env, filled)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = filled / np.maximum(env, 1e-8)
    cr = np.clip(cr, 0.0, 2.0).astype(np.float32)
    cr[~np.isfinite(spectrum)] = np.nan
    return cr


def continuum_removal_match_cube(
    cube: np.ndarray,
    valid: np.ndarray,
    bad_bands: Optional[np.ndarray] = None,
    size: int = None,
    block_rows: int = 128,
) -> np.ndarray:
    """Vectorized match-CR for full cube (block rows)."""
    size = int(size or C.CR_MATCH_FILTER_SIZE)
    if maximum_filter1d is None:
        raise RuntimeError("scipy.ndimage.maximum_filter1d required for cube CR")
    H, W, B = cube.shape
    out = np.full((H, W, B), np.nan, dtype=np.float32)
    bad = np.zeros(B, dtype=bool) if bad_bands is None else bad_bands.astype(bool)
    for r0 in range(0, H, block_rows):
        r1 = min(H, r0 + block_rows)
        block = cube[r0:r1].astype(np.float32).copy()
        bh, bw, _ = block.shape
        flat = block.reshape(-1, B)
        vflat = valid[r0:r1].reshape(-1)
        for bi in range(B):
            if bad[bi]:
                flat[:, bi] = np.nan
                continue
            col = flat[:, bi]
            m = np.isfinite(col) & (col > 0) & vflat
            if m.any():
                fillv = float(np.nanmedian(col[m]))
                miss = (~np.isfinite(col) | (col <= 0)) & vflat
                col[miss] = fillv
                flat[:, bi] = col
        flat_env = np.nan_to_num(flat, nan=0.0)
        flat_env[:, bad] = 0.0
        env = maximum_filter1d(flat_env, size=size, axis=1, mode="nearest")
        env = np.maximum(env, flat_env)
        with np.errstate(divide="ignore", invalid="ignore"):
            cr = flat / np.maximum(env, 1e-8)
        cr = np.clip(cr, 0.0, 2.0).astype(np.float32)
        cr[:, bad] = np.nan
        cr[~vflat] = np.nan
        out[r0:r1] = cr.reshape(bh, bw, B)
        if (r0 // block_rows) % 10 == 0:
            print(f"    CR cube {r1}/{H} rows ({100.0 * r1 / H:.1f}%)", flush=True)
    return out


def band_depth_3pt(
    spectrum: np.ndarray, waves: np.ndarray,
    center: float, left: float, right: float,
) -> float:
    ci = nearest_band(waves, center)
    li = nearest_band(waves, left)
    ri = nearest_band(waves, right)
    rc, rl, rr = spectrum[ci], spectrum[li], spectrum[ri]
    if not (np.isfinite(rc) and np.isfinite(rl) and np.isfinite(rr)):
        return float("nan")
    if rc <= 0 or rl <= 0 or rr <= 0:
        return float("nan")
    denom = (waves[ri] - waves[li]) + 1e-9
    t = (waves[ci] - waves[li]) / denom
    cont = rl + (rr - rl) * t
    if abs(cont) < 1e-8:
        return float("nan")
    return float(1.0 - rc / cont)


def band_depth_map_3pt(
    cube: np.ndarray, waves: np.ndarray,
    center: float, left: float, right: float,
) -> np.ndarray:
    ci = nearest_band(waves, center)
    li = nearest_band(waves, left)
    ri = nearest_band(waves, right)
    rc = cube[:, :, ci].astype(np.float64)
    rl = cube[:, :, li].astype(np.float64)
    rr = cube[:, :, ri].astype(np.float64)
    denom = (waves[ri] - waves[li]) + 1e-9
    t = (waves[ci] - waves[li]) / denom
    cont = rl + (rr - rl) * t
    out = np.full(rc.shape, np.nan, dtype=np.float32)
    ok = (
        np.isfinite(rc) & np.isfinite(rl) & np.isfinite(rr)
        & (rc > 0) & (rl > 0) & (rr > 0) & (np.abs(cont) >= 1e-8)
    )
    out[ok] = (1.0 - rc[ok] / cont[ok]).astype(np.float32)
    return out


def olindex3_spectrum(spectrum_refl: np.ndarray, waves: np.ndarray) -> float:
    """
    Simplified OLINDEX3-like: drop of reflectance at olivine sample points
    relative to a VIS–SWIR continuum (0.75 / 1.55 µm).
    """
    j75 = nearest_band(waves, 0.75)
    j155 = nearest_band(waves, 1.55)
    r75, r155 = spectrum_refl[j75], spectrum_refl[j155]
    if not (np.isfinite(r75) and np.isfinite(r155) and r75 > 0 and r155 > 0):
        return float("nan")
    drops = []
    for p in C.OLINDEX3_POINTS:
        j = nearest_band(waves, p)
        rj = spectrum_refl[j]
        if not (np.isfinite(rj) and rj > 0):
            continue
        # linear continuum between 0.75 and 1.55
        t = (waves[j] - waves[j75]) / (waves[j155] - waves[j75] + 1e-9)
        cont = r75 + t * (r155 - r75)
        if cont > 1e-8:
            drops.append(1.0 - rj / cont)
    if len(drops) < 4:
        return float("nan")
    return float(max(0.0, np.mean(drops)))


def olindex3_map(cube_refl: np.ndarray, waves: np.ndarray) -> np.ndarray:
    j75 = nearest_band(waves, 0.75)
    j155 = nearest_band(waves, 1.55)
    r75 = cube_refl[:, :, j75].astype(np.float64)
    r155 = cube_refl[:, :, j155].astype(np.float64)
    H, W, _ = cube_refl.shape
    acc = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    for p in C.OLINDEX3_POINTS:
        j = nearest_band(waves, p)
        rj = cube_refl[:, :, j].astype(np.float64)
        t = (waves[j] - waves[j75]) / (waves[j155] - waves[j75] + 1e-9)
        cont = r75 + t * (r155 - r75)
        ok = (r75 > 0) & (r155 > 0) & (rj > 0) & (cont > 1e-8) & np.isfinite(rj)
        drop = np.zeros((H, W), dtype=np.float64)
        drop[ok] = 1.0 - rj[ok] / cont[ok]
        acc[ok] += drop[ok]
        cnt[ok] += 1.0
    out = np.full((H, W), np.nan, dtype=np.float32)
    m = cnt >= 4
    out[m] = np.clip(acc[m] / cnt[m], 0.0, 2.0).astype(np.float32)
    return out


def bd1250_plag_map(cube_refl: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """
    Plagioclase ~1.25 µm: depth at 1.25 relative to continuum 1.10–1.40,
    then muted if strong 1 µm mafic depth (avoids false plag on olivine/pyroxene).
    """
    bd125 = band_depth_map_3pt(cube_refl, waves, 1.250, 1.100, 1.400)
    bd1um = band_depth_map_3pt(cube_refl, waves, 1.050, 0.890, 1.300)
    out = bd125.astype(np.float32).copy()
    # if 1µm mafic is stronger, suppress plag index (crystal-field plag is weak vs mafics)
    strong_mafic = np.isfinite(bd1um) & (bd1um > 0.04)
    out[strong_mafic] = np.nan
    # keep only positive shallow features
    out[np.isfinite(out) & (out < 0)] = 0.0
    return out


def lcpindex_map(cube_refl: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """
    LCP: 2 µm band near ~1.90 µm (shorter than HCP) minus a fraction of 1 µm depth
    so olivine-dominated 1 µm does not dominate the LCP map.
    """
    bd2 = band_depth_map_3pt(cube_refl, waves, *C.SPM_TRIPLES["LCPINDEX2"])
    bd1 = band_depth_map_3pt(cube_refl, waves, *C.SPM_TRIPLES["BD1000"])
    out = np.full(bd2.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(bd2)
    # residual 2µm after soft 1µm suppression
    out[ok] = (bd2[ok] - 0.35 * np.nan_to_num(bd1[ok], nan=0.0)).astype(np.float32)
    out[np.isfinite(out) & (out < 0)] = 0.0
    return out


def hcpindex_map(cube_refl: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """
    HCP: 2 µm band near ~2.20–2.30 µm (longer than LCP), residual after 1 µm mute.
    """
    bd2 = band_depth_map_3pt(cube_refl, waves, *C.SPM_TRIPLES["HCPINDEX2"])
    bd1 = band_depth_map_3pt(cube_refl, waves, *C.SPM_TRIPLES["BD1000"])
    out = np.full(bd2.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(bd2)
    out[ok] = (bd2[ok] - 0.35 * np.nan_to_num(bd1[ok], nan=0.0)).astype(np.float32)
    out[np.isfinite(out) & (out < 0)] = 0.0
    return out


def pyx_ol_separation_spectrum(
    spectrum_refl: np.ndarray,
    waves: np.ndarray,
    spectrum_cr: Optional[np.ndarray] = None,
) -> dict:
    """Diagnostic 1 µm vs 2 µm depths for olivine vs pyroxene mixture context.

    1µm is measured on reflectance first; if non-positive, fall back to match-CR
    spectrum (often recovers muted 1µm character after continuum shaping).
    """
    d1_r = band_depth_3pt(spectrum_refl, waves, *C.SPM_TRIPLES["BD1000"])
    d1_c = (
        band_depth_3pt(spectrum_cr, waves, *C.SPM_TRIPLES["BD1000"])
        if spectrum_cr is not None else float("nan")
    )
    d1_source = "refl"
    d1 = d1_r
    if (not np.isfinite(d1) or float(d1) <= 0.0) and np.isfinite(d1_c) and float(d1_c) > 0.0:
        d1 = d1_c
        d1_source = "cr_fallback"
    d_lcp = band_depth_3pt(spectrum_refl, waves, *C.SPM_TRIPLES["LCPINDEX2"])
    d_hcp = band_depth_3pt(spectrum_refl, waves, *C.SPM_TRIPLES["HCPINDEX2"])
    # 2µm: prefer reflectance; if weak, allow CR fallback similarly
    d_lcp_c = (
        band_depth_3pt(spectrum_cr, waves, *C.SPM_TRIPLES["LCPINDEX2"])
        if spectrum_cr is not None else float("nan")
    )
    d_hcp_c = (
        band_depth_3pt(spectrum_cr, waves, *C.SPM_TRIPLES["HCPINDEX2"])
        if spectrum_cr is not None else float("nan")
    )
    if (not np.isfinite(d_lcp) or float(d_lcp) < C.PYX_D2_MIN) and np.isfinite(d_lcp_c):
        d_lcp = max(float(d_lcp) if np.isfinite(d_lcp) else -1.0, float(d_lcp_c))
    if (not np.isfinite(d_hcp) or float(d_hcp) < C.PYX_D2_MIN) and np.isfinite(d_hcp_c):
        d_hcp = max(float(d_hcp) if np.isfinite(d_hcp) else -1.0, float(d_hcp_c))
    d2 = np.nanmax([d_lcp, d_hcp])
    ratio = float(d2 / (d1 + 1e-6)) if np.isfinite(d1) and np.isfinite(d2) else float("nan")
    return {
        "d1um": None if not np.isfinite(d1) else float(d1),
        "d1um_source": d1_source,
        "d2um_lcp": None if not np.isfinite(d_lcp) else float(d_lcp),
        "d2um_hcp": None if not np.isfinite(d_hcp) else float(d_hcp),
        "d2_over_d1": None if not np.isfinite(ratio) else float(ratio),
    }


def r750_map(cube_refl: np.ndarray, waves: np.ndarray) -> np.ndarray:
    j = nearest_band(waves, 0.750)
    return cube_refl[:, :, j].astype(np.float32)


def compute_spm_vector(
    spectrum_cr: np.ndarray,
    waves: np.ndarray,
    spectrum_refl: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    SPM vector. Prefer reflectance for index formulas; fall back to CR spectrum.
    """
    src = spectrum_refl if spectrum_refl is not None else spectrum_cr
    out = np.full(C.N_SPM, np.nan, dtype=np.float32)
    for i, name in enumerate(C.SPM_NAMES):
        if name == "OLINDEX3":
            out[i] = olindex3_spectrum(src, waves)
        elif name == "LCPINDEX2":
            bd2 = band_depth_3pt(src, waves, *C.SPM_TRIPLES["LCPINDEX2"])
            bd1 = band_depth_3pt(src, waves, *C.SPM_TRIPLES["BD1000"])
            if np.isfinite(bd2):
                out[i] = float(max(0.0, bd2 - 0.35 * (bd1 if np.isfinite(bd1) else 0.0)))
        elif name == "HCPINDEX2":
            bd2 = band_depth_3pt(src, waves, *C.SPM_TRIPLES["HCPINDEX2"])
            bd1 = band_depth_3pt(src, waves, *C.SPM_TRIPLES["BD1000"])
            if np.isfinite(bd2):
                out[i] = float(max(0.0, bd2 - 0.35 * (bd1 if np.isfinite(bd1) else 0.0)))
        elif name == "BD1250":
            # single-spectrum plag: depth only (no spatial mafic mute)
            out[i] = band_depth_3pt(src, waves, 1.250, 1.100, 1.400)
        elif name == "R750":
            j = nearest_band(waves, 0.750)
            out[i] = float(src[j]) if np.isfinite(src[j]) else float("nan")
        elif name in C.SPM_TRIPLES:
            c, l, r = C.SPM_TRIPLES[name]
            out[i] = band_depth_3pt(src, waves, c, l, r)
    return out


def compute_spm_maps(
    cube_refl: np.ndarray,
    waves: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """SPM stack from reflectance (scientifically correct for 3-point indices)."""
    H, W, _ = cube_refl.shape
    stack = np.full((H, W, C.N_SPM), np.nan, dtype=np.float32)
    by_name = {}
    for i, name in enumerate(C.SPM_NAMES):
        if name == "OLINDEX3":
            m = olindex3_map(cube_refl, waves)
        elif name == "LCPINDEX2":
            m = lcpindex_map(cube_refl, waves)
        elif name == "HCPINDEX2":
            m = hcpindex_map(cube_refl, waves)
        elif name == "BD1250":
            m = bd1250_plag_map(cube_refl, waves)
        elif name == "R750":
            m = r750_map(cube_refl, waves)
        elif name in C.SPM_TRIPLES:
            c, l, r = C.SPM_TRIPLES[name]
            m = band_depth_map_3pt(cube_refl, waves, c, l, r)
        else:
            m = np.full((H, W), np.nan, dtype=np.float32)
        stack[:, :, i] = m
        by_name[name] = m
    return stack, by_name


def simple_depth(spectrum: np.ndarray, waves: np.ndarray, key: str) -> float:
    if key not in C.PROOF_DEPTH_DEFS:
        return float("nan")
    c, l, r = C.PROOF_DEPTH_DEFS[key]
    return band_depth_3pt(spectrum, waves, c, l, r)
