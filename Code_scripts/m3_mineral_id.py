"""
M3 Phase 2 mineral identification (Global 85-band).

Stages: setup -> SPM -> encoder -> HDBSCAN -> unmixer -> PRESENT proof -> rare scan.
Authority: mineral_proof.json PRESENT (not cluster/unmixer names alone).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import m3_mineral_constants as C
import m3_scene_io as sio
import m3_spm as spm
from m3_spectral_library import build_library, l2_normalize, load_built_library

try:
    import hdbscan
except ImportError:
    hdbscan = None


# ---------------------------------------------------------------------------
# Encoder / unmixer
# ---------------------------------------------------------------------------

class SpectralEncoder(nn.Module):
    def __init__(self, n_bands: int = C.N_BANDS_GLOBAL, latent_dim: int = C.LATENT_DIM, n_spm: int = C.N_SPM):
        super().__init__()
        self.n_bands = n_bands
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, latent_dim),
        )
        self.spm_head = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, n_spm), nn.Sigmoid(),
        )
        self.recon_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, n_bands),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, B)
        h = self.conv(x.unsqueeze(1))
        h = h.mean(dim=-1)
        z = self.fc(h)
        return F.normalize(z, dim=-1)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return z, self.spm_head(z), self.recon_head(z)


class AbundanceNetwork(nn.Module):
    """Softmax abundances; LayerNorm (not BatchNorm) for stable small-batch training."""

    def __init__(self, latent_dim: int = C.LATENT_DIM, n_end: int = C.N_ENDMEMBERS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_end),
        )

    def forward(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.net(z)
        t = max(float(temperature), 1e-3)
        return F.softmax(logits / t, dim=-1)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, T: float = 0.07) -> torch.Tensor:
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / T
    n = z1.shape[0]
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def augment_spectrum(x: np.ndarray, protected: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.astype(np.float32).copy()
    y *= float(rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:
        # light smooth
        k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        pad = np.pad(y, (1, 1), mode="edge")
        y = np.convolve(pad, k, mode="valid")
    drop = (~protected) & (rng.random(len(y)) < 0.05)
    y[drop] = 0.0
    return y


# ---------------------------------------------------------------------------
# SAM / library match
# ---------------------------------------------------------------------------

def spectral_angle(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)
    else:
        mask = mask & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 8:
        return float("nan")
    aa, bb = a[mask], b[mask]
    na, nb = np.linalg.norm(aa), np.linalg.norm(bb)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    cos = float(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0))
    return float(np.arccos(cos))


def match_library_family(
    spec_l2: np.ndarray,
    lib: dict,
    family: Optional[str] = None,
    band_mask: Optional[np.ndarray] = None,
) -> Tuple[str, str, float]:
    best_name, best_fam, best_sam = "", "", float("inf")
    for i, name in enumerate(lib["names"]):
        fam = lib["families"][i]
        if family is not None and fam != family:
            continue
        ang = spectral_angle(spec_l2, lib["spectra_cr_l2"][i], band_mask)
        if np.isfinite(ang) and ang < best_sam:
            best_sam, best_name, best_fam = ang, name, fam
    return best_name, best_fam, best_sam


# ---------------------------------------------------------------------------
# Stage 2 — SPM + sampling
# ---------------------------------------------------------------------------

def stage2_spm(ctx: dict) -> dict:
    print(f"\n=== Stage2 SPM + match-CR  ({C.METHOD_VERSION}) ===")
    cube = ctx["cube"]
    waves = ctx["waves"]
    valid = ctx["valid"]
    bad = ctx["bad_bands"]
    t0 = time.time()

    # 1) SPM on reflectance (3-point / OLINDEX) — scientifically standard
    spm_map, by_name = spm.compute_spm_maps(cube, waves)
    print(f"  SPM maps done in {time.time()-t0:.1f}s")

    # 2) Proper match-CR cube for encoder / SAM / proof (same algo as library)
    t1 = time.time()
    cube_cr = spm.continuum_removal_match_cube(
        cube, valid, bad_bands=bad, size=C.CR_MATCH_FILTER_SIZE, block_rows=128,
    )
    print(f"  Match-CR cube done in {time.time()-t1:.1f}s")

    # 3) Active domain with contrast + anti-saturation (max hot fraction)
    n_valid = int(valid.sum())
    active = np.zeros(valid.shape, dtype=bool)
    gate_info = {}
    for i, name in enumerate(C.SPM_NAMES):
        if name == "BD1500" and ctx["scene"] not in C.POLAR_SCENE_NAMES:
            gate_info[name] = {"enabled": False, "reason": "ice_not_gated"}
            continue
        if name in ("BD1900", "BD1000", "R750"):
            # support / diagnostic — not primary active drivers
            thr0 = float(C.SPM_DETECT_THRESH.get(name, 0.008))
            layer = spm_map[:, :, i]
            n_hot0 = int((valid & np.isfinite(layer) & (layer >= thr0)).sum())
            gate_info[name] = {
                "enabled": False,
                "reason": "diagnostic_or_support",
                "thr": thr0,
                "n_hot": n_hot0,
            }
            continue

        layer = spm_map[:, :, i]
        thr = float(C.SPM_DETECT_THRESH.get(name, 0.008))
        vals = layer[valid & np.isfinite(layer)]
        if vals.size < 100:
            gate_info[name] = {"enabled": False, "reason": "too_few_finite"}
            continue

        # If absolute thr is too broad, raise to top-TARGET percentile (keeps hot selective)
        hot = valid & np.isfinite(layer) & (layer >= thr)
        hot_frac = float(hot.sum()) / max(1, n_valid)
        thr_floor = thr
        if hot_frac > C.SPM_TARGET_HOT_FRAC:
            pct = 100.0 * (1.0 - float(C.SPM_TARGET_HOT_FRAC))
            thr = max(thr, float(np.percentile(vals, pct)))
            hot = valid & np.isfinite(layer) & (layer >= thr)
            hot_frac = float(hot.sum()) / max(1, n_valid)

        med = float(np.median(vals))
        sig = float(np.std(vals))
        n_hot = int(hot.sum())
        if n_hot >= C.SPM_MIN_COMPONENT:
            hot_med = float(np.nanmedian(layer[hot]))
            contrast_ok = hot_med > med + max(0.5 * sig, 0.5 * thr)
        else:
            contrast_ok = False
            hot_med = float("nan")

        saturated = hot_frac > C.SPM_MAX_HOT_FRAC
        enabled = bool(contrast_ok and n_hot >= C.SPM_MIN_COMPONENT and not saturated)
        gate_info[name] = {
            "enabled": enabled,
            "thr": thr,
            "thr_floor": thr_floor,
            "n_hot": n_hot,
            "hot_frac": hot_frac,
            "hot_med": hot_med,
            "scene_med": med,
            "contrast_ok": bool(contrast_ok),
            "saturated": bool(saturated),
        }
        if enabled:
            active |= hot

    # Stratified training coords
    rng = np.random.default_rng(42)
    coords = []
    for i, name in enumerate(C.SPM_NAMES):
        layer = spm_map[:, :, i]
        vals = layer[valid & np.isfinite(layer)]
        if vals.size < 50:
            continue
        p80 = float(np.percentile(vals, 80))
        ys, xs = np.where(valid & np.isfinite(layer) & (layer >= p80))
        if len(ys) == 0:
            continue
        take = min(500, len(ys))
        sel = rng.choice(len(ys), take, replace=False)
        coords.extend(list(zip(ys[sel].tolist(), xs[sel].tolist())))
    ys, xs = np.where(valid & ~active)
    need = max(0, C.TARGET_TRAINING_SAMPLES - len(coords))
    if need > 0 and len(ys) > 0:
        sel = rng.choice(len(ys), min(need, len(ys)), replace=False)
        coords.extend(list(zip(ys[sel].tolist(), xs[sel].tolist())))
    if len(coords) > C.TARGET_TRAINING_SAMPLES:
        idx = rng.choice(len(coords), C.TARGET_TRAINING_SAMPLES, replace=False)
        coords = [coords[j] for j in idx]

    ctx["spm_map"] = spm_map
    ctx["spm_by_name"] = by_name
    ctx["spm_active"] = active
    ctx["spm_gate_info"] = gate_info
    ctx["cube_cr"] = cube_cr
    ctx["cube_cr_proxy"] = cube_cr  # alias so older call sites stay safe
    ctx["training_coords"] = coords
    ctx["out_dir"].mkdir(parents=True, exist_ok=True)
    np.save(ctx["out_dir"] / "spm_map.npy", spm_map)
    np.save(ctx["out_dir"] / "spm_active_mask.npy", active)
    np.save(ctx["out_dir"] / "cube_cr.npy", cube_cr)
    (ctx["out_dir"] / "spm_gate_info.json").write_text(
        json.dumps(gate_info, indent=2), encoding="utf-8"
    )
    (ctx["out_dir"] / "method_version.txt").write_text(C.METHOD_VERSION, encoding="utf-8")
    print(f"  Active pixels: {int(active.sum()):,} / {n_valid:,}")
    print(f"  Training coords: {len(coords)}")
    return ctx


def gate2_spm_sanity(ctx: dict) -> dict:
    info = ctx["spm_gate_info"]
    enabled = [k for k, v in info.items() if v.get("enabled")]
    reasons = []
    if len(enabled) < 1:
        reasons.append("no_spm_index_enabled")
    # tautology check: active must not be almost all valid
    frac = float(ctx["spm_active"].sum()) / max(1, int(ctx["valid"].sum()))
    if frac > 0.85:
        reasons.append(f"active_frac_too_high={frac:.2f}")
    report = {
        "gate": "G2_CR_SPM_sanity",
        "scene": ctx["scene"],
        "pass": len(reasons) == 0,
        "enabled_indices": enabled,
        "active_frac": frac,
        "reasons": reasons,
    }
    C.GATES_DIR.mkdir(parents=True, exist_ok=True)
    (C.GATES_DIR / f"G2_{ctx['scene'].replace(' ', '_')}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (ctx["out_dir"] / "gate_report_G2.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"  Gate2: {'PASS' if report['pass'] else 'FAIL'} enabled={enabled}")
    return report


# ---------------------------------------------------------------------------
# Stage 3 — Encoder
# ---------------------------------------------------------------------------

def _build_train_pool(ctx: dict, lib: dict) -> Tuple[np.ndarray, np.ndarray]:
    waves = ctx["waves"]
    cube_cr = ctx["cube_cr"]
    spm_map = ctx["spm_map"]
    coords = ctx["training_coords"]
    specs, spms = [], []
    for y, x in coords:
        s = cube_cr[y, x, :]
        if np.isfinite(s).sum() < 20:
            continue
        specs.append(l2_normalize(s))
        spms.append(spm_map[y, x, :])
    rng = np.random.default_rng(0)
    if len(lib["spectra_cr_l2"]) > 0:
        idx = rng.choice(len(lib["spectra_cr_l2"]), C.LIB_SAMPLE_N, replace=True)
        for i in idx:
            specs.append(lib["spectra_cr_l2"][i])
            spms.append(lib["spm"][i])
    fam_idx = {}
    for i, fam in enumerate(lib["families"]):
        fam_idx.setdefault(fam, []).append(i)
    pairs = [("Plagioclase", "LCP"), ("Plagioclase", "HCP"), ("Olivine", "Plagioclase"),
             ("LCP", "HCP"), ("Olivine", "LCP"), ("HCP", "Ilmenite"), ("Olivine", "HCP")]
    for _ in range(C.SYNTH_MIXES):
        fa, fb = pairs[int(rng.integers(0, len(pairs)))]
        if fa not in fam_idx or fb not in fam_idx:
            continue
        ia = int(rng.choice(fam_idx[fa]))
        ib = int(rng.choice(fam_idx[fb]))
        a = float(rng.uniform(0.1, 0.9))
        mix = a * lib["spectra_cr_l2"][ia] + (1 - a) * lib["spectra_cr_l2"][ib]
        mix = l2_normalize(mix)
        specs.append(mix)
        spms.append(spm.compute_spm_vector(mix, waves))
    X = np.stack(specs, axis=0).astype(np.float32)
    S = np.stack(spms, axis=0).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    S = np.nan_to_num(S, nan=0.0)
    for j in range(S.shape[1]):
        mx = np.percentile(S[:, j], 99) if S.shape[0] else 1.0
        if mx > 1e-6:
            S[:, j] = np.clip(S[:, j] / mx, 0, 1)
    return X, S


def train_encoder_on_pool(
    X: np.ndarray,
    S: np.ndarray,
    waves: np.ndarray,
    out_path: Path,
    mode: str = "full",
    prior_path: Optional[Path] = None,
) -> SpectralEncoder:
    """Train SpectralEncoder on a (possibly multi-scene) pool; save weights to out_path."""
    device = _device()
    n_bands = X.shape[1]
    model = SpectralEncoder(n_bands=n_bands).to(device)
    if mode == "adapt" and prior_path is not None and Path(prior_path).is_file():
        state = torch.load(prior_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded prior: {prior_path}")
        epochs = C.ENCODER_EPOCHS_ADAPT
        patience = C.ENCODER_PATIENCE_ADAPT
        min_ep = 8
    else:
        epochs = C.ENCODER_EPOCHS_FULL
        patience = C.ENCODER_PATIENCE_FULL
        min_ep = 30

    protected = np.zeros(n_bands, dtype=bool)
    for w0 in (0.75, 1.05, 1.25, 1.50, 1.82, 2.14, 2.21):
        protected[spm.nearest_band(waves, w0)] = True

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(42)
    best = float("inf")
    bad = 0
    bs = 256
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        losses = []
        for i0 in range(0, len(X), bs):
            idx = perm[i0:i0 + bs]
            if len(idx) < 8:
                continue
            xb = X[idx]
            sb = S[idx]
            v1 = np.stack([augment_spectrum(xb[j], protected, rng) for j in range(len(idx))])
            v2 = np.stack([augment_spectrum(xb[j], protected, rng) for j in range(len(idx))])
            t1 = torch.from_numpy(v1).float().to(device)
            t2 = torch.from_numpy(v2).float().to(device)
            ts = torch.from_numpy(sb).float().to(device)
            z1, sp1, r1 = model(t1)
            z2, sp2, r2 = model(t2)
            loss = nt_xent(z1, z2) + 0.1 * F.mse_loss(sp1, ts) + 0.1 * F.mse_loss(sp2, ts)
            tgt = torch.from_numpy(xb).float().to(device)
            pm = torch.from_numpy(protected).to(device)
            loss = loss + 0.2 * F.mse_loss(r1[:, pm], tgt[:, pm])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        mean_l = float(np.mean(losses)) if losses else float("inf")
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{epochs}  loss={mean_l:.4f}")
        if mean_l + 1e-4 < best:
            best = mean_l
            bad = 0
            torch.save(model.state_dict(), out_path)
        else:
            bad += 1
            if ep + 1 >= min_ep and bad >= patience:
                print(f"  early stop at epoch {ep+1}")
                break

    model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _stratified_spm_coords(
    ctx: dict,
    n_total: int,
    spm_names: Optional[List[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, int]]:
    """Sample coordinates stratified across SPM hotspots (then fill from active/valid)."""
    rng = rng or np.random.default_rng(1)
    spm_map = ctx["spm_map"]
    valid = ctx["valid"]
    gate = ctx.get("spm_gate_info") or {}
    active = ctx.get("spm_active")
    if spm_names is None:
        spm_names = [
            n for n in C.CLUSTER_NAME_SPM
            if gate.get(n, {}).get("enabled") and n in C.SPM_NAMES
        ]
    if not spm_names:
        spm_names = [n for n, g in gate.items() if g.get("enabled") and n in C.SPM_NAMES]

    per = max(50, int(n_total // max(1, len(spm_names)))) if spm_names else n_total
    coords: List[Tuple[int, int]] = []
    seen = set()
    for sname in spm_names:
        i = C.SPM_NAMES.index(sname)
        thr = float(gate.get(sname, {}).get("thr", C.SPM_DETECT_THRESH.get(sname, 0.008)))
        ys, xs = np.where(valid & np.isfinite(spm_map[:, :, i]) & (spm_map[:, :, i] >= thr))
        if len(ys) == 0:
            continue
        take = min(per, len(ys))
        sel = rng.choice(len(ys), take, replace=False)
        for y, x in zip(ys[sel].tolist(), xs[sel].tolist()):
            key = (int(y), int(x))
            if key not in seen:
                seen.add(key)
                coords.append(key)
            if len(coords) >= n_total:
                return coords[:n_total]

    # fill from active, then valid
    for mask in (active, valid):
        if mask is None or len(coords) >= n_total:
            continue
        ys, xs = np.where(mask if mask is valid else (valid & mask))
        if len(ys) == 0:
            continue
        order = rng.permutation(len(ys))
        for j in order:
            key = (int(ys[j]), int(xs[j]))
            if key in seen:
                continue
            seen.add(key)
            coords.append(key)
            if len(coords) >= n_total:
                break
    return coords[:n_total]


def encode_scene_latents(ctx: dict, model: SpectralEncoder) -> dict:
    """Encode active/valid pixels with a frozen encoder; save latent arrays.

    Respects CLUSTER_ENCODE_MAX: if active pixels exceed the cap, subsample
    actives (SPM-stratified) instead of encoding millions of points.
    """
    device = _device()
    model = model.to(device)
    model.eval()
    valid = ctx["valid"]
    active = ctx["spm_active"]
    MAX_ENC = int(C.CLUSTER_ENCODE_MAX)
    rng = np.random.default_rng(1)

    ya, xa = np.where(active)
    n_active = int(len(ya))
    if n_active >= MAX_ENC:
        coords_e = _stratified_spm_coords(ctx, MAX_ENC, rng=rng)
        print(f"  Encode cap: active={n_active:,} > {MAX_ENC:,} → stratified subsample {len(coords_e):,}")
    else:
        coords_e = list(zip(ya.tolist(), xa.tolist()))
        need = MAX_ENC - len(coords_e)
        if need > 0:
            ys, xs = np.where(valid & (~active))
            if len(ys) > 0:
                take = min(need, len(ys))
                sel = rng.choice(len(ys), take, replace=False)
                coords_e.extend(list(zip(ys[sel].tolist(), xs[sel].tolist())))

    print(f"  Encoding {len(coords_e):,} pixels ...")
    cube_cr = ctx["cube_cr"]
    lat = np.zeros((len(coords_e), C.LATENT_DIM), dtype=np.float32)
    coord_arr = np.array(coords_e, dtype=np.int32)
    bs = 4096
    with torch.no_grad():
        for i0 in range(0, len(coords_e), bs):
            batch = coords_e[i0:i0 + bs]
            specs = np.stack([l2_normalize(cube_cr[y, x]) for y, x in batch]).astype(np.float32)
            specs = np.nan_to_num(specs, nan=0.0)
            z = model.encode(torch.from_numpy(specs).to(device)).cpu().numpy()
            lat[i0:i0 + len(batch)] = z
    ctx["out_dir"].mkdir(parents=True, exist_ok=True)
    np.save(ctx["out_dir"] / "latent_vectors.npy", lat)
    np.save(ctx["out_dir"] / "latent_coords.npy", coord_arr)
    ctx["latent"] = lat
    ctx["latent_coords"] = coord_arr
    ctx["encoder"] = model
    print(f"  Latent saved: {lat.shape}")
    return ctx


def stage3_encoder_frozen(ctx: dict, encoder_path: Path) -> dict:
    """Load a pretrained encoder (no fine-tune) and encode the scene."""
    print(f"\n=== Stage3 Encoder (frozen from {encoder_path}) ===")
    device = _device()
    model = SpectralEncoder(n_bands=C.N_BANDS_GLOBAL).to(device)
    state = torch.load(encoder_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    # keep a scene-local copy for provenance
    ctx["out_dir"].mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ctx["out_dir"] / "encoder_B85.pth")
    return encode_scene_latents(ctx, model)


def stage3_encoder(ctx: dict, lib: dict, mode: str = "full", prior_path: Optional[Path] = None) -> dict:
    print(f"\n=== Stage3 Encoder (mode={mode}) ===")
    X, S = _build_train_pool(ctx, lib)
    print(f"  Train pool: {X.shape[0]} spectra")
    out_enc = ctx["out_dir"] / "encoder_B85.pth"
    model = train_encoder_on_pool(
        X, S, ctx["waves"], out_enc, mode=mode, prior_path=prior_path
    )
    return encode_scene_latents(ctx, model)


def gate3_encoder_qa(ctx: dict) -> dict:
    lat = ctx["latent"]
    reasons = []
    if not np.isfinite(lat).all():
        reasons.append("latent_has_nan")
    if lat.shape[0] < 1000:
        reasons.append("too_few_encoded")
    norms = np.linalg.norm(lat, axis=1)
    if not (0.9 < float(np.median(norms)) < 1.1):
        reasons.append(f"latent_not_unit median_norm={float(np.median(norms)):.3f}")
    report = {
        "gate": "G3_Encoder_QA",
        "scene": ctx["scene"],
        "pass": len(reasons) == 0,
        "n_encoded": int(lat.shape[0]),
        "reasons": reasons,
    }
    (C.GATES_DIR / f"G3_{ctx['scene'].replace(' ', '_')}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"  Gate3: {'PASS' if report['pass'] else 'FAIL'}")
    return report


# ---------------------------------------------------------------------------
# Stage 4 — HDBSCAN (informational; never overrides PRESENT)
# ---------------------------------------------------------------------------

def _cluster_name_channels(ctx: dict) -> List[str]:
    """SPM channels allowed for cluster naming — prefer PRESENT groups only."""
    gate = ctx.get("spm_gate_info") or {}
    proof = ctx.get("proof") or {}
    minerals = proof.get("minerals") or {}
    channels: List[str] = []
    if minerals:
        for group, e in minerals.items():
            if not e.get("present"):
                continue
            spm_name = (C.PRESENT_GROUPS.get(group) or {}).get("spm")
            if (
                spm_name
                and spm_name in C.CLUSTER_NAME_SPM
                and gate.get(spm_name, {}).get("enabled", True)
                and spm_name not in channels
            ):
                channels.append(spm_name)
    if not channels:
        channels = [
            n for n in C.CLUSTER_NAME_SPM
            if gate.get(n, {}).get("enabled", True) and n in C.SPM_NAMES
        ]
    return channels or list(C.CLUSTER_NAME_SPM)


def _score_cluster_labels(labels: np.ndarray) -> Tuple[float, dict]:
    """Higher is better. Returns -inf metrics when quality gates fail."""
    labs = labels.astype(np.int32)
    n = len(set(labs.tolist()) - {-1})
    noise = float((labs < 0).mean()) if len(labs) else 1.0
    kept = labs[labs >= 0]
    if kept.size == 0:
        return -1e9, {"n": 0, "noise": noise, "dominance": 1.0, "pass": False}
    sizes = np.bincount(kept)
    sizes = sizes[sizes > 0]
    dominance = float(sizes.max() / sizes.sum())
    ok = (
        C.CLUSTER_MIN_N <= n <= C.CLUSTER_MAX_N
        and noise <= float(C.CLUSTER_MAX_NOISE)
        and dominance <= float(C.CLUSTER_MAX_DOMINANCE)
    )
    # Prefer more clusters (up to 5), lower noise, lower dominance
    score = (10.0 if ok else -100.0) + 0.4 * n - 2.0 * noise - 1.5 * dominance
    return float(score), {
        "n": int(n),
        "noise": noise,
        "dominance": dominance,
        "pass": bool(ok),
        "sizes": [int(x) for x in sizes.tolist()],
    }


def stage4_hdbscan(ctx: dict) -> dict:
    print("\n=== Stage4 HDBSCAN (informational) ===")
    ctx["clustering_status"] = "SKIPPED"
    ctx["n_clusters"] = 0
    ctx["cluster_names"] = {}
    ctx["cluster_labels"] = None

    if hdbscan is None:
        print("  hdbscan not available — skip")
        ctx["clustering_status"] = "UNAVAILABLE"
        return ctx

    lat = ctx["latent"]
    coords = ctx["latent_coords"]
    if lat is None or coords is None or len(lat) < 200:
        print("  too few encoded pixels — skip clustering")
        ctx["clustering_status"] = "TOO_FEW"
        return ctx

    name_channels = _cluster_name_channels(ctx)
    rng = np.random.default_rng(0)
    # Stratified fit sample in latent space (coords must exist in encoded set)
    coord_index = {(int(y), int(x)): i for i, (y, x) in enumerate(coords)}
    want = _stratified_spm_coords(
        ctx,
        min(int(C.CLUSTER_MAX_FIT), len(coords)),
        spm_names=name_channels,
        rng=rng,
    )
    fit_idx = [coord_index[c] for c in want if c in coord_index]
    if len(fit_idx) < 200:
        # fallback: random encoded subset
        fit_idx = rng.choice(len(coords), min(int(C.CLUSTER_MAX_FIT), len(coords)), replace=False).tolist()
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    Xfit = lat[fit_idx]
    print(f"  Fit sample={len(Xfit)}  name_channels={name_channels}")

    best = {
        "score": -1e18,
        "labels": None,
        "metrics": {},
        "mcs": None,
        "method": None,
    }
    for method in C.CLUSTER_METHODS:
        for mcs in C.CLUSTER_MCS_CANDIDATES:
            ms = max(5, int(mcs) // 10)
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=int(mcs),
                    min_samples=ms,
                    metric="euclidean",
                    cluster_selection_method=method,
                )
                labels = clusterer.fit_predict(Xfit)
            except Exception as exc:  # pragma: no cover
                print(f"  HDBSCAN fail mcs={mcs} method={method}: {exc}")
                continue
            score, metrics = _score_cluster_labels(labels)
            if score > best["score"]:
                best.update(
                    {
                        "score": score,
                        "labels": labels,
                        "metrics": metrics,
                        "mcs": int(mcs),
                        "method": method,
                    }
                )

    if best["labels"] is None or not best["metrics"].get("pass", False):
        ctx["clustering_status"] = "FAILED"
        ctx["n_clusters"] = 0
        (ctx["out_dir"] / "cluster_names.json").write_text("{}", encoding="utf-8")
        print(
            f"  clustering FAILED (best score={best['score']:.2f} "
            f"metrics={best.get('metrics')}) — PRESENT claims unchanged"
        )
        return ctx

    labels_fit = np.asarray(best["labels"], dtype=np.int32)
    n_clusters = int(best["metrics"]["n"])
    # NN-assign remaining encoded pixels from fit exemplars
    labels_all = np.full(len(coords), -1, dtype=np.int32)
    labels_all[fit_idx] = labels_fit
    kept_fit = fit_idx[labels_fit >= 0]
    if len(kept_fit) > 0:
        unlab = np.where(labels_all < 0)[0]
        if len(unlab) > 0:
            # subsample exemplars for speed
            rng2 = np.random.default_rng(1)
            ex = kept_fit
            if len(ex) > 4000:
                ex = ex[rng2.choice(len(ex), 4000, replace=False)]
            # chunked NN
            for i0 in range(0, len(unlab), 8000):
                chunk = unlab[i0:i0 + 8000]
                # squared euclidean to exemplars
                a = lat[chunk]  # (n,d)
                b = lat[ex]     # (m,d)
                # ||a||^2 + ||b||^2 - 2 a·b ; latents are ~unit so ≈ 2-2a·b
                dots = a @ b.T
                nn = np.argmax(dots, axis=1)
                labels_all[chunk] = labels_all[ex][nn]

    # Names from PRESENT-aware channels only
    spm_map = ctx["spm_map"]
    valid = ctx["valid"]
    scene_stats = {}
    for sname in name_channels:
        i = C.SPM_NAMES.index(sname)
        layer = spm_map[:, :, i]
        vv = layer[valid & np.isfinite(layer)]
        if vv.size < 50:
            scene_stats[sname] = (0.0, 1.0)
        else:
            scene_stats[sname] = (float(np.median(vv)), float(max(np.std(vv), 1e-6)))

    names = {}
    for cid in sorted(set(labels_fit.tolist()) - {-1}):
        pts = coords[fit_idx][labels_fit == cid]
        if len(pts) == 0:
            continue
        zscores = []
        for sname in name_channels:
            i = C.SPM_NAMES.index(sname)
            vals = [float(spm_map[y, x, i]) for y, x in pts[:500]]
            m = float(np.nanmean(vals)) if vals else float("nan")
            med, sig = scene_stats[sname]
            zscores.append((m - med) / sig if np.isfinite(m) else float("nan"))
        if not np.isfinite(zscores).any():
            names[int(cid)] = "Unresolved"
            continue
        order = np.argsort([-z if np.isfinite(z) else -1e9 for z in zscores])
        j0 = int(order[0])
        label = name_channels[j0]
        if len(order) > 1:
            j1 = int(order[1])
            z0, z1 = zscores[j0], zscores[j1]
            if (
                np.isfinite(z0)
                and np.isfinite(z1)
                and z0 > 0
                and z1 >= float(C.CLUSTER_NAME_SECOND_FRAC) * z0
            ):
                label = f"{name_channels[j0]}+{name_channels[j1]}"
        names[int(cid)] = label

    ctx["cluster_labels"] = labels_all
    ctx["cluster_names"] = names
    ctx["n_clusters"] = n_clusters
    ctx["clustering_status"] = "PASS"
    ctx["clustering_qa"] = {
        "mcs": best["mcs"],
        "method": best["method"],
        "score": best["score"],
        **best["metrics"],
        "name_channels": name_channels,
    }
    (ctx["out_dir"] / "cluster_names.json").write_text(
        json.dumps({str(k): v for k, v in names.items()}, indent=2), encoding="utf-8"
    )
    (ctx["out_dir"] / "clustering_qa.json").write_text(
        json.dumps(ctx["clustering_qa"], indent=2), encoding="utf-8"
    )
    print(
        f"  clusters={n_clusters} noise={best['metrics']['noise']:.2f} "
        f"dominance={best['metrics']['dominance']:.2f} "
        f"mcs={best['mcs']} method={best['method']} status=PASS"
    )
    print(f"  cluster names (PRESENT-aware): {names}")
    return ctx


# ---------------------------------------------------------------------------
# Stage 5 — Unmixer
# ---------------------------------------------------------------------------

def _family_balanced_endmembers(
    lib: dict,
    allowed_families: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Pick diverse endmembers with capped counts per lunar family (avoids olivine flood).

    If allowed_families is set (from PRESENT proof), only those families are used
    so unmix cannot invent minerals rejected by PRESENT (e.g. highlands olivine).
    """
    lib_s = lib["spectra_cr_l2"]
    fams_all = lib["families"]
    names_all = lib["names"]
    rng = np.random.default_rng(0)
    order = list(C.FAMILY_ENDMEMBER_ORDER)
    if allowed_families:
        allow = set(allowed_families)
        order = [f for f in order if f in allow]
        if not order:
            order = list(C.UNMIX_CORE_FAMILIES)
    chosen: List[int] = []
    for fam in order:
        idxs = [i for i, f in enumerate(fams_all) if f == fam]
        if not idxs:
            continue
        # farthest-point within family
        pick = [int(rng.choice(idxs))]
        while len(pick) < min(C.UNMIX_MAX_PER_FAMILY, len(idxs)):
            dmin = np.full(len(idxs), np.inf)
            for pi, p in enumerate(idxs):
                d = min(1.0 - float(np.nansum(lib_s[p] * lib_s[c])) for c in pick)
                dmin[pi] = d
            pick.append(idxs[int(np.argmax(dmin))])
        chosen.extend(pick)
    # trim / pad to N_ENDMEMBERS (pad only within allowed families when constrained)
    if len(chosen) > C.N_ENDMEMBERS:
        chosen = chosen[: C.N_ENDMEMBERS]
    allow_idx = set(chosen)
    if allowed_families:
        allow_set = set(allowed_families)
        cand = [i for i, f in enumerate(fams_all) if f in allow_set]
    else:
        cand = list(range(len(lib_s)))
    while len(chosen) < min(C.N_ENDMEMBERS, len(cand)):
        dmin = np.full(len(cand), np.inf)
        for pi, i in enumerate(cand):
            if i in allow_idx:
                dmin[pi] = -np.inf
                continue
            dmin[pi] = min(1.0 - float(np.nansum(lib_s[i] * lib_s[c])) for c in chosen)
        nxt = cand[int(np.argmax(dmin))]
        if nxt in allow_idx:
            break
        chosen.append(nxt)
        allow_idx.add(nxt)

    E = np.nan_to_num(lib_s[chosen].astype(np.float32), nan=0.0)
    for i in range(len(E)):
        nrm = np.linalg.norm(E[i])
        if nrm > 1e-8:
            E[i] /= nrm
    return E, [names_all[i] for i in chosen], [fams_all[i] for i in chosen]


def _present_families_from_proof(ctx: dict) -> List[str]:
    """Map PRESENT groups -> library family names for unmix constraint."""
    proof = ctx.get("proof") or {}
    fams = []
    for group, e in (proof.get("minerals") or {}).items():
        if not e.get("present"):
            continue
        conf = C.PRESENT_GROUPS.get(group) or {}
        fam = conf.get("family")
        if fam and fam not in fams:
            fams.append(fam)
    return fams


def _family_index_lists(fams: List[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {f: [] for f in C.FAMILY_ENDMEMBER_ORDER}
    for j, fam in enumerate(fams):
        out.setdefault(fam, []).append(j)
    return out


def _unmix_balanced_coords(ctx: dict) -> List[Tuple[int, int]]:
    """
    Prefer equal hotspot samples from each enabled PRESENT SPM so bland/plag
    spectra cannot dominate unmix training (a prior collapse mode).
    """
    rng = np.random.default_rng(7)
    spm_map = ctx["spm_map"]
    valid = ctx["valid"]
    gate = ctx.get("spm_gate_info") or {}
    coords: List[Tuple[int, int]] = []
    enabled = [n for n in C.CLUSTER_NAME_SPM if gate.get(n, {}).get("enabled")]
    if not enabled:
        enabled = [n for n, g in gate.items() if g.get("enabled")]
    for sname in enabled:
        if sname not in C.SPM_NAMES:
            continue
        i = C.SPM_NAMES.index(sname)
        thr = float(gate.get(sname, {}).get("thr", C.SPM_DETECT_THRESH.get(sname, 0.008)))
        ys, xs = np.where(valid & np.isfinite(spm_map[:, :, i]) & (spm_map[:, :, i] >= thr))
        if len(ys) == 0:
            continue
        take = min(int(C.UNMIX_PER_SPM_SAMPLES), len(ys))
        sel = rng.choice(len(ys), take, replace=False)
        coords.extend(list(zip(ys[sel].tolist(), xs[sel].tolist())))
    active = ctx.get("spm_active")
    if active is not None:
        ys, xs = np.where(valid & (~active))
    else:
        ys, xs = np.where(valid)
    if len(ys) > 0 and int(C.UNMIX_BLAND_SAMPLES) > 0:
        take = min(int(C.UNMIX_BLAND_SAMPLES), len(ys))
        sel = rng.choice(len(ys), take, replace=False)
        coords.extend(list(zip(ys[sel].tolist(), xs[sel].tolist())))
    if not coords:
        coords = list(ctx.get("training_coords") or [])
    # de-dup while preserving order
    seen = set()
    uniq = []
    for y, x in coords:
        key = (int(y), int(x))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _unmix_qa_from_A(A: np.ndarray, fams: List[str]) -> Tuple[dict, dict, str, dict]:
    fam_abund = {f: 0.0 for f in C.FAMILY_ENDMEMBER_ORDER}
    for j, fam in enumerate(fams):
        fam_abund[fam] = fam_abund.get(fam, 0.0) + float(A[:, j].mean())
    ol = float(fam_abund.get("Olivine", 0.0))
    pyx = float(fam_abund.get("LCP", 0.0) + fam_abund.get("HCP", 0.0))
    plag = float(fam_abund.get("Plagioclase", 0.0))
    mean_max = float(A.max(axis=1).mean())
    # QA over families actually in E (PRESENT-constrained), not absent zeros
    active_core = [f for f in C.UNMIX_CORE_FAMILIES if f in set(fams)]
    if not active_core:
        active_core = list(C.UNMIX_CORE_FAMILIES)
    core_means = [float(fam_abund.get(f, 0.0)) for f in active_core]
    family_max = float(max(core_means)) if core_means else 0.0
    family_min = float(min(core_means)) if core_means else 0.0
    collapsed = mean_max >= C.UNMIX_COLLAPSE_MAX_ABUND
    flat = (not collapsed) and (
        mean_max < float(C.UNMIX_FLAT_MIN_MEAN_MAX)
        or family_max < float(C.UNMIX_FLAT_MIN_FAMILY_MAX)
    )
    dom_thr = 0.85 if len(active_core) <= 2 else 0.80
    fam_dom = (not collapsed) and (family_max >= dom_thr)
    if collapsed or fam_dom:
        qa_status = "FAILED_COLLAPSE"
        mix_note = "FAILED — unmixer collapsed; do not cite fractions."
        collapsed = True
    elif flat:
        qa_status = "FAILED_FLAT"
        mix_note = "FAILED — abundances too flat/uninformative; do not cite fractions."
    else:
        qa_status = "PASS"
        mix_note = "Soft mixture context from unmixer on training pixels only."
    ovp = {
        "olivine_mean": ol,
        "pyroxene_mean_LCP_plus_HCP": pyx,
        "plagioclase_mean": plag,
        "olivine_fraction_of_mafic": float(ol / (ol + pyx + 1e-8)),
        "pyroxene_fraction_of_mafic": float(pyx / (ol + pyx + 1e-8)),
        "note": mix_note,
    }
    checks = {
        "abund_nonneg": bool((A >= -1e-5).all()),
        "abund_sum1": bool(np.allclose(A.sum(axis=1), 1.0, atol=1e-3)),
        "mean_max_abund": mean_max,
        "family_max_abund": family_max,
        "family_min_core": family_min,
        "active_core_families": active_core,
        "collapse_threshold": C.UNMIX_COLLAPSE_MAX_ABUND,
        "flat_min_mean_max": C.UNMIX_FLAT_MIN_MEAN_MAX,
        "flat_min_family_max": C.UNMIX_FLAT_MIN_FAMILY_MAX,
        "collapsed": bool(collapsed),
        "flat": bool(flat),
    }
    return fam_abund, ovp, qa_status, checks


def stage5_unmix(ctx: dict, lib: dict) -> dict:
    """
    Physics-constrained unmixer. Abundances are SECONDARY (mixture context).
    PRESENT authority remains mineral_proof.json.

    v2d: no sparsity reward (caused plag one-hot collapse); balanced SPM hotspot
    training; temperature softmax; band penalties on peakiness + family means;
    checkpoint by diversity score (not raw loss).
    """
    print("\n=== Stage5 Unmixer (secondary abundances) ===")
    device = _device()
    model = ctx["encoder"].to(device)
    model.eval()
    if len(lib["spectra_cr_l2"]) < 3:
        print("  library too small for unmix")
        return ctx

    allowed = _present_families_from_proof(ctx)
    if allowed:
        print(f"  Constraining endmembers to PRESENT families: {allowed}")
    else:
        print("  No PRESENT families yet — using core families (fallback)")
        allowed = list(C.UNMIX_CORE_FAMILIES)
    E_init, end_names, fams = _family_balanced_endmembers(lib, allowed_families=allowed)
    n_end = len(end_names)
    fam_idx = _family_index_lists(fams)
    core_fams = [f for f in C.UNMIX_CORE_FAMILIES if fam_idx.get(f)]
    # With fewer PRESENT families, allow higher batch-mean dominance
    fam_mean_hi = 0.70 if len(core_fams) <= 2 else float(C.UNMIX_FAM_MEAN_HI)
    fam_mean_lo = 0.08 if len(core_fams) <= 2 else float(C.UNMIX_FAM_MEAN_LO)
    # Soft prior from PRESENT hotspot fractions (align mixture with proof evidence)
    prior_vec = None
    proof_m = (ctx.get("proof") or {}).get("minerals") or {}
    if proof_m and core_fams and float(C.UNMIX_PRESENT_PRIOR_WEIGHT) > 0:
        group_for_fam = {conf["family"]: g for g, conf in C.PRESENT_GROUPS.items()}
        raw = []
        for f in core_fams:
            g = group_for_fam.get(f)
            e = proof_m.get(g) or {}
            hf = float(e.get("hot_frac") or 0.0) if e.get("present") else 0.0
            raw.append(max(hf, 0.0))
        sraw = float(sum(raw))
        if sraw > 1e-8:
            prior_vec = [v / sraw for v in raw]
            print(f"  Unmix PRESENT prior ({core_fams}) = {[round(v, 3) for v in prior_vec]}")
    abund_net = AbundanceNetwork(n_end=n_end).to(device)
    if C.UNMIX_FREEZE_ENDMEMBERS:
        E = torch.from_numpy(E_init.copy()).to(device)
        E.requires_grad_(False)
        params = list(abund_net.parameters())
        print("  Endmember matrix E FROZEN (library anchors)")
    else:
        E = nn.Parameter(torch.from_numpy(E_init.copy()).to(device))
        params = list(abund_net.parameters()) + [E]

    coords = _unmix_balanced_coords(ctx)
    cube_cr = ctx["cube_cr"]
    specs = []
    for y, x in coords:
        s = l2_normalize(cube_cr[y, x])
        if np.isfinite(s).sum() >= 20:
            specs.append(np.nan_to_num(s, nan=0.0).astype(np.float32))
    if len(specs) < 100:
        print("  too few specs for unmix train")
        return ctx
    X = np.stack(specs).astype(np.float32)
    print(f"  Unmix train pixels={len(X)} (balanced SPM hotspots + few bland)")
    with torch.no_grad():
        Z = model.encode(torch.from_numpy(X).to(device)).cpu().numpy()
    Z = np.nan_to_num(Z, nan=0.0)

    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    bs = 512
    temp = float(C.UNMIX_SOFTMAX_TEMP)
    best_score = -1e9
    best_state = None
    best_loss = float("inf")
    best_metrics = {}

    for ep in range(C.UNMIX_EPOCHS):
        abund_net.train()
        perm = np.random.permutation(len(Z))
        losses = []
        for i0 in range(0, len(Z), bs):
            idx = perm[i0:i0 + bs]
            if len(idx) < 8:
                continue
            zt = torch.from_numpy(Z[idx]).float().to(device)
            xt = torch.from_numpy(X[idx]).float().to(device)
            a = abund_net(zt, temperature=temp)
            recon = a @ E
            mse = F.mse_loss(recon, xt)
            ent = -(a.clamp_min(1e-8) * a.clamp_min(1e-8).log()).sum(dim=1).mean()
            mean_max_b = a.max(dim=1).values.mean()
            # peakiness band: not one-hot, not uniform
            peak_pen = (
                F.relu(mean_max_b - float(C.UNMIX_MEAN_MAX_HI)) ** 2
                + F.relu(float(C.UNMIX_MEAN_MAX_LO) - mean_max_b) ** 2
            )
            # family means on this batch
            fam_means = []
            for f in core_fams:
                idxs = fam_idx[f]
                fam_means.append(a[:, idxs].sum(dim=1).mean())
            fam_stack = torch.stack(fam_means)
            fam_pen = (
                F.relu(fam_stack.max() - fam_mean_hi) ** 2
                + F.relu(fam_mean_lo - fam_stack.min()) ** 2
            )
            ent_pen = (ent - float(C.UNMIX_ENTROPY_TARGET)) ** 2
            prior_pen = fam_stack.new_zeros(())
            if prior_vec is not None:
                pt = torch.tensor(prior_vec, device=fam_stack.device, dtype=fam_stack.dtype)
                prior_pen = F.mse_loss(fam_stack, pt)
            loss = (
                mse
                + float(C.UNMIX_MEAN_MAX_WEIGHT) * peak_pen
                + float(C.UNMIX_FAM_DIV_WEIGHT) * fam_pen
                + float(C.UNMIX_ENTROPY_BAND_WEIGHT) * ent_pen
                + float(C.UNMIX_PRESENT_PRIOR_WEIGHT) * prior_pen
            )
            if not C.UNMIX_FREEZE_ENDMEMBERS:
                loss = loss + 0.15 * F.mse_loss(E, torch.from_numpy(E_init).to(device))
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.item()))

        abund_net.eval()
        with torch.no_grad():
            A_ep = abund_net(torch.from_numpy(Z).float().to(device), temperature=temp).cpu().numpy()
        fam_abund_ep, _, qa_ep, checks_ep = _unmix_qa_from_A(A_ep, fams)
        # score: prefer PASS, then mid peakiness, then family diversity
        mean_max = float(checks_ep["mean_max_abund"])
        fmax = float(checks_ep["family_max_abund"])
        fmin = float(checks_ep["family_min_core"])
        score = 0.0
        if qa_ep == "PASS":
            score += 10.0
        elif qa_ep == "FAILED_FLAT":
            score += 1.0
        # closer to mid band is better
        score -= abs(mean_max - 0.50) * 2.0
        score -= max(0.0, fmax - 0.55) * 5.0
        score += min(fmin, 0.25)  # reward some presence of weakest core family
        mean_l = float(np.mean(losses)) if losses else float("nan")
        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"  unmix epoch {ep+1}/{C.UNMIX_EPOCHS}  loss={mean_l:.4f}  "
                f"qa={qa_ep} mean_max={mean_max:.3f} fam_max={fmax:.3f} score={score:.2f}"
            )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in abund_net.state_dict().items()}
            best_loss = mean_l
            best_metrics = {
                "qa": qa_ep,
                "mean_max": mean_max,
                "fam_abund": fam_abund_ep,
                "checks": checks_ep,
            }

    if best_state is not None:
        abund_net.load_state_dict(best_state)
    E_np = E.detach().cpu().numpy() if torch.is_tensor(E) else E_init
    abund_net.eval()
    with torch.no_grad():
        A = abund_net(torch.from_numpy(Z).float().to(device), temperature=temp).cpu().numpy()
    fam_abund, ovp, qa_status, checks = _unmix_qa_from_A(A, fams)
    checks["loss_finite"] = bool(np.isfinite(best_loss))
    checks["best_score"] = float(best_score)
    checks["softmax_temp"] = temp
    ol = float(ovp["olivine_mean"])
    pyx = float(ovp["pyroxene_mean_LCP_plus_HCP"])
    plag = float(ovp["plagioclase_mean"])
    mean_max = float(checks["mean_max_abund"])
    mixture = {
        "authority": "SECONDARY — not PRESENT proof",
        "method_version": C.METHOD_VERSION,
        "qa_status": qa_status,
        "citable": bool(qa_status == "PASS"),
        "mean_family_abundance": {k: float(v) for k, v in fam_abund.items()},
        "olivine_vs_pyroxene": ovp,
        "endmembers": [{"family": f, "name": n} for f, n in zip(fams, end_names)],
        "train_loss_best": best_loss if np.isfinite(best_loss) else None,
        "endmembers_frozen": bool(C.UNMIX_FREEZE_ENDMEMBERS),
        "train_pixels": int(len(X)),
        "allowed_families": list(allowed),
        "checks": checks,
    }
    (ctx["out_dir"] / "mixture_summary.json").write_text(
        json.dumps(mixture, indent=2), encoding="utf-8"
    )
    torch.save({
        "abund": abund_net.state_dict(),
        "E": E_np,
        "end_names": end_names,
        "families": fams,
        "method_version": C.METHOD_VERSION,
        "qa_status": mixture["qa_status"],
    }, ctx["out_dir"] / "unmixer_B85.pth")
    ctx["endmember_names"] = end_names
    ctx["endmember_families"] = fams
    ctx["E"] = E_np
    ctx["abund_net"] = abund_net
    ctx["mixture_summary"] = mixture
    print(f"  Endmembers: {list(zip(fams, end_names))}")
    print(f"  Mixture QA={mixture['qa_status']}  ol={ol:.3f} pyx={pyx:.3f} plag={plag:.3f} "
          f"mean_max={mean_max:.3f} best_score={best_score:.2f}")
    if best_metrics:
        print(f"  (best ckpt during train had qa={best_metrics.get('qa')})")
    return ctx


# ---------------------------------------------------------------------------
# PRESENT proof + rare
# ---------------------------------------------------------------------------

def pyx_soft_support(sep: dict, dkey: str) -> Tuple[bool, str]:
    """
    Soft 2µm support for LCP/HCP PRESENT.

    Scientific rule:
      - If 1µm depth is missing/non-positive, do NOT use d2/d1 ratio (undefined/misleading).
        Require absolute diagnostic 2µm depth >= PYX_D2_MIN_ABS_PATH (weak path).
      - If 1µm depth is positive, require d2/d1 >= 0.15 OR absolute d2 >= PYX_D2_MIN
        (strong path).
    Reasons encode grade: strong_* | weak_* | fail_*.
    """
    if not sep:
        return True, "strong_no_sep_skip"
    d1 = sep.get("d1um")
    if dkey == "d1.90":
        d2 = sep.get("d2um_lcp")
    elif dkey == "d2.25":
        d2 = sep.get("d2um_hcp")
    else:
        d2 = sep.get("d2um_hcp")
        if d2 is None:
            d2 = sep.get("d2um_lcp")
    ratio = sep.get("d2_over_d1")

    d2_ok = d2 is not None and np.isfinite(float(d2)) and float(d2) >= C.PYX_D2_MIN
    d2_ok_abs = (
        d2 is not None and np.isfinite(float(d2)) and float(d2) >= C.PYX_D2_MIN_ABS_PATH
    )
    d1_pos = d1 is not None and np.isfinite(float(d1)) and float(d1) > 0.0

    if not d1_pos:
        # ratio meaningless; stricter absolute 2µm = weak-grade support
        return (
            bool(d2_ok_abs),
            "weak_abs_d2_d1_nonpos" if d2_ok_abs else "fail_abs_d2_d1_nonpos",
        )

    ratio_ok = ratio is not None and np.isfinite(float(ratio)) and float(ratio) >= 0.15
    ok = bool(ratio_ok or d2_ok)
    return ok, ("strong_ratio_or_abs_d2" if ok else "fail_ratio_and_abs_d2")


def pyx_support_grade(reason: str) -> str:
    if reason.startswith("strong"):
        return "strong"
    if reason.startswith("weak"):
        return "weak"
    if reason.startswith("fail"):
        return "fail"
    return "n/a"


def write_mineral_proof(ctx: dict, lib: dict) -> dict:
    print("\n=== PRESENT proof (match-CR spectra) ===")
    waves = ctx["waves"]
    cube_cr = ctx["cube_cr"]
    cube_refl = ctx["cube"]
    valid = ctx["valid"]
    spm_map = ctx["spm_map"]
    gate = ctx["spm_gate_info"]
    enabled_idx = [C.SPM_NAMES.index(n) for n, g in gate.items() if g.get("enabled")]
    if enabled_idx:
        mx = np.nanmax(spm_map[:, :, enabled_idx], axis=2)
    else:
        mx = np.nanmax(spm_map, axis=2)
    finite = valid & np.isfinite(mx)
    thr_b = float(np.percentile(mx[finite], 10)) if finite.any() else 0.0
    bland = finite & (mx <= thr_b)

    def mean_spec(cube_src, mask, nmax=3000):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        if len(ys) > nmax:
            rng = np.random.default_rng(0)
            ii = rng.choice(len(ys), nmax, replace=False)
            ys, xs = ys[ii], xs[ii]
        specs = np.asarray(cube_src[ys, xs, :], dtype=np.float64)
        ok = np.isfinite(specs).sum(axis=1) >= 20
        if not ok.any():
            return None
        sub = specs[ok]
        if not np.isfinite(sub).any():
            return None
        with np.errstate(all="ignore"):
            mean = np.nanmean(sub, axis=0)
        if mean is None or (not np.isfinite(mean).any()):
            return None
        return mean.astype(np.float32)

    minerals = {}
    present_count = 0
    present_strong = []
    present_weak = []
    n_valid = max(1, int(valid.sum()))
    for group, conf in C.PRESENT_GROUPS.items():
        spm_name = conf["spm"]
        gi = gate.get(spm_name, {})
        dkey = conf["depth_key"]
        fam = conf["family"]
        thr = float(gi.get("thr", C.SPM_DETECT_THRESH.get(spm_name, 0.008)))
        j = C.SPM_NAMES.index(spm_name)
        layer = spm_map[:, :, j]
        hot = valid & np.isfinite(layer) & (layer >= thr)
        n_hot = int(hot.sum())
        hot_frac = float(n_hot) / float(n_valid)

        hot_cr = mean_spec(cube_cr, hot)
        bland_cr = mean_spec(cube_cr, bland)
        hot_refl = mean_spec(cube_refl, hot)
        bland_refl = mean_spec(cube_refl, bland)

        # Depths on reflectance (same domain as SPM 3-point) — more stable for BD
        d_hot = spm.simple_depth(hot_refl, waves, dkey) if hot_refl is not None else float("nan")
        d_bland = spm.simple_depth(bland_refl, waves, dkey) if bland_refl is not None else float("nan")
        hmb = (d_hot - d_bland) if np.isfinite(d_hot) and np.isfinite(d_bland) else float("nan")
        sam_name, sam_fam, sam = "", "", float("nan")
        if hot_cr is not None:
            sam_name, sam_fam, sam = match_library_family(l2_normalize(hot_cr), lib, family=fam)

        sep = (
            spm.pyx_ol_separation_spectrum(hot_refl, waves, spectrum_cr=hot_cr)
            if hot_refl is not None else {}
        )

        depth_ok = np.isfinite(d_hot) and d_hot >= C.PROOF_DEPTH_BD_MIN
        hmb_ok = np.isfinite(hmb) and hmb >= C.PROOF_HOT_MINUS_BLAND
        hot_ok = n_hot >= C.SPM_MIN_COMPONENT and bool(gi.get("enabled", False))
        if group == "Ilmenite":
            lib_ok = np.isfinite(sam) and sam <= C.SAM_STRONG_MAX
            present = bool(hot_ok and depth_ok and hmb_ok and lib_ok)
        else:
            lib_ok = (not np.isfinite(sam)) or sam <= C.FAMILY_SAM_HIT_MAX
            present = bool(hot_ok and depth_ok and hmb_ok and lib_ok)

        # Pyroxene soft 2µm support (graded strong/weak; weak = abs 2µm when d1um<=0)
        pyx_ok, pyx_reason = True, "n/a"
        pyx_grade = "n/a"
        if group in ("Low-Ca Pyroxene", "High-Ca Pyroxene"):
            pyx_ok, pyx_reason = pyx_soft_support(sep, dkey)
            pyx_grade = pyx_support_grade(pyx_reason)
            if present and not pyx_ok:
                present = False

        ol_mix = None
        if group == "Olivine" and sep:
            r = sep.get("d2_over_d1")
            if r is not None and np.isfinite(float(r)):
                ol_mix = "pyx_mixed" if float(r) >= C.OL_MIX_D2_OVER_D1 else "olivine_rich"

        # Claim level: PRESENT_WEAK when pyroxene only has abs-2µm / non-positive 1µm support
        if not present:
            present_level = "ABSENT"
            confidence = "none"
        elif pyx_grade == "weak":
            present_level = "PRESENT_WEAK"
            confidence = "weak"
        else:
            present_level = "PRESENT"
            confidence = "strong"

        if present:
            present_count += 1
            if present_level == "PRESENT_WEAK":
                present_weak.append(group)
            else:
                present_strong.append(group)
        minerals[group] = {
            "present": present,
            "present_level": present_level,
            "confidence": confidence,
            "spm_index": spm_name,
            "thr_adaptive": thr,
            "n_hot_pixels": n_hot,
            "hot_frac": hot_frac,
            "proof_depth_key": dkey,
            "band_depths_hot": {dkey: None if not np.isfinite(d_hot) else float(d_hot)},
            "band_depths_bland": {dkey: None if not np.isfinite(d_bland) else float(d_bland)},
            "hot_minus_bland": None if not np.isfinite(hmb) else float(hmb),
            "best_library": sam_name,
            "best_family": sam_fam,
            "best_sam": None if not np.isfinite(sam) else float(sam),
            "pyx_ol_diagnostic": sep,
            "olivine_mixture_flag": ol_mix,
            "gates": {
                "hot_ok": hot_ok,
                "depth_ok": depth_ok,
                "hmb_ok": hmb_ok,
                "lib_ok": lib_ok,
                "spm_enabled": bool(gi.get("enabled", False)),
                "pyx_2um_support": bool(pyx_ok),
                "pyx_support_reason": pyx_reason,
                "pyx_support_grade": pyx_grade,
            },
        }

    n_clusters = int(ctx.get("n_clusters") or 0)
    # STRONG requires enough *strong* PRESENT (weak pyroxene alone must not inflate).
    # Clustering is informational — never demote mineral verdict when clusters fail.
    n_strong = len(present_strong)
    if n_strong >= int(C.STRONG_MIN_PRESENT_STRONG):
        overall = "STRONG"
    elif present_count >= 1:
        overall = "MODERATE"
    else:
        overall = "WEAK"

    proof = {
        "scene": ctx["scene"],
        "fname": ctx["fname"],
        "n_bands": ctx["n_bands"],
        "method_version": C.METHOD_VERSION,
        "cr_method": "continuum_removal_match",
        "minerals": minerals,
        "n_present": present_count,
        "present_strong": present_strong,
        "present_weak": present_weak,
        "n_clusters": n_clusters,
        "clustering_status": ctx.get("clustering_status"),
        "overall_verdict": overall,
        "map_available": n_clusters > 0,
        "mixture_secondary": ctx.get("mixture_summary", {}).get("olivine_vs_pyroxene"),
        "mixture_qa": ctx.get("mixture_summary", {}).get("qa_status"),
        "note": (
            "PRESENT = hotspot+depth+hot-bland(+SAM). "
            "PRESENT_WEAK = pyroxene with abs-2µm only (1µm non-positive). "
            "Unmixer mixture is secondary and constrained to PRESENT families. "
            "HDBSCAN maps are informational and do not change PRESENT verdict."
        ),
    }
    (ctx["out_dir"] / "mineral_proof.json").write_text(
        json.dumps(proof, indent=2), encoding="utf-8"
    )
    ctx["proof"] = proof
    print(f"  PRESENT strong={present_strong}  weak={present_weak}")
    print(f"  overall_verdict={overall}")
    return proof


def detect_rare_minerals(ctx: dict, lib: dict) -> dict:
    """
    Rare scan. Ilmenite requires SAM + depth + hot>bland (not SAM alone).
    Spinel remains SAM-count candidate only (no dedicated SPM yet).
    """
    print("\n=== Rare mineral scan ===")
    cube_cr = ctx["cube_cr"]
    cube_refl = ctx["cube"]
    waves = ctx["waves"]
    valid = ctx["valid"]
    active = ctx["spm_active"]
    ys, xs = np.where(active)
    rng = np.random.default_rng(2)
    if len(ys) > 8000:
        ii = rng.choice(len(ys), 8000, replace=False)
        ys, xs = ys[ii], xs[ii]

    # bland reference for rare depth contrast
    spm_map = ctx["spm_map"]
    gate = ctx.get("spm_gate_info") or {}
    en_idx = [C.SPM_NAMES.index(n) for n, g in gate.items() if g.get("enabled")]
    if en_idx:
        mx = np.nanmax(spm_map[:, :, en_idx], axis=2)
    else:
        mx = np.nanmax(spm_map, axis=2)
    finite = valid & np.isfinite(mx)
    thr_b = float(np.percentile(mx[finite], 10)) if finite.any() else 0.0
    bland = finite & (mx <= thr_b)
    bland_ys, bland_xs = np.where(bland)
    if len(bland_ys) > 3000:
        bi = rng.choice(len(bland_ys), 3000, replace=False)
        bland_ys, bland_xs = bland_ys[bi], bland_xs[bi]
    bland_refl = None
    if len(bland_ys):
        bspecs = np.asarray(cube_refl[bland_ys, bland_xs, :], dtype=np.float64)
        bok = np.isfinite(bspecs).sum(axis=1) >= 20
        if bok.any() and np.isfinite(bspecs[bok]).any():
            with np.errstate(all="ignore"):
                bland_refl = np.nanmean(bspecs[bok], axis=0).astype(np.float32)
            if bland_refl is not None and (not np.isfinite(bland_refl).any()):
                bland_refl = None

    is_mare = ctx["scene"] in C.MARE_SCENE_NAMES
    if is_mare:
        ilm_depth_min = float(C.RARE_ILM_DEPTH_MIN)
        ilm_hmb_min = float(C.RARE_ILM_HMB_MIN)
        ilm_n_min = int(C.RARE_ILM_MIN_HOT)
        cell_min = int(C.RARE_ILM_MIN_CELL_HITS)
        terrain = "mare"
    else:
        ilm_depth_min = float(C.RARE_ILM_DEPTH_MIN_NONMARE)
        ilm_hmb_min = float(C.RARE_ILM_HMB_MIN_NONMARE)
        ilm_n_min = int(C.RARE_ILM_MIN_HOT_NONMARE)
        cell_min = int(C.RARE_ILM_MIN_CELL_HITS_NONMARE)
        terrain = "nonmare"

    ilm_hits = []
    spinel_hits = []
    for y, x in zip(ys, xs):
        s = l2_normalize(cube_cr[y, x])
        name, fam, ang = match_library_family(s, lib, family=None)
        if not (np.isfinite(ang) and ang <= C.SAM_STRONG_MAX):
            continue
        if fam == "Spinel":
            spinel_hits.append((int(y), int(x), name, float(ang)))
        elif fam == "Ilmenite":
            # per-pixel depth contrast vs bland
            d = spm.simple_depth(cube_refl[y, x], waves, "d1.00")
            db = spm.simple_depth(bland_refl, waves, "d1.00") if bland_refl is not None else float("nan")
            hmb = (d - db) if np.isfinite(d) and np.isfinite(db) else float("nan")
            depth_ok = np.isfinite(d) and d >= ilm_depth_min
            hmb_ok = np.isfinite(hmb) and hmb >= ilm_hmb_min
            if depth_ok and hmb_ok:
                ilm_hits.append((int(y), int(x), name, float(ang), float(d), float(hmb)))

    # Spatial clustering: require at least one coarse cell with enough hits
    spatial_ok = False
    max_cell = 0
    if ilm_hits:
        H, W = cube_cr.shape[:2]
        gy = max(1, int(C.RARE_ILM_GRID_Y))
        gx = max(1, int(C.RARE_ILM_GRID_X))
        cells = np.zeros((gy, gx), dtype=np.int32)
        for y, x, *_ in ilm_hits:
            cy = min(gy - 1, int(y * gy / max(1, H)))
            cx = min(gx - 1, int(x * gx / max(1, W)))
            cells[cy, cx] += 1
        max_cell = int(cells.max())
        spatial_ok = max_cell >= cell_min

    majors = ctx.get("proof", {}).get("minerals", {})
    candidates = {}

    # Ilmenite gated (terrain-aware + spatial cluster).
    # PRESENT_RARE is mare-only when RARE_ILM_PRESENT_RARE_MARE_ONLY (stops highland false rares).
    already_ilm = bool((majors.get("Ilmenite") or {}).get("present"))
    n_ilm = len(ilm_hits)
    mare_only = bool(C.RARE_ILM_PRESENT_RARE_MARE_ONLY)
    if already_ilm:
        st_ilm = "ALREADY_MAJOR"
    elif n_ilm >= ilm_n_min and spatial_ok and (is_mare or not mare_only):
        st_ilm = "PRESENT_RARE"
    elif n_ilm > 0:
        st_ilm = "CANDIDATE"
    else:
        st_ilm = "ABSENT"
    candidates["Ilmenite"] = {
        "n_sam_hits_gated": int(n_ilm),
        "status": st_ilm,
        "terrain": terrain,
        "spatial_max_cell": int(max_cell),
        "spatial_ok": bool(spatial_ok),
        "mare_only_present_rare": mare_only,
        "requires": (
            f"SAM + d1.00>={ilm_depth_min} + hmb>={ilm_hmb_min} "
            f"(n>={ilm_n_min}, terrain={terrain}, spatial_cell>={cell_min}"
            f"{', PRESENT_RARE mare-only' if mare_only else ''})"
        ),
        "examples": [
            {"y": h[0], "x": h[1], "library": h[2], "sam": h[3], "d1.00": h[4], "hmb": h[5]}
            for h in ilm_hits[:5]
        ],
    }

    # Spinel: SAM-count candidate only
    n_sp = len(spinel_hits)
    candidates["Spinel"] = {
        "n_sam_hits": int(n_sp),
        "status": ("CANDIDATE" if n_sp >= C.RARE_MIN_HOT else ("WEAK" if n_sp > 0 else "ABSENT")),
        "requires": "SAM only (no dedicated SPM yet) — not PRESENT_RARE",
        "examples": [
            {"y": h[0], "x": h[1], "library": h[2], "sam": h[3]} for h in spinel_hits[:5]
        ],
    }

    out = {"scene": ctx["scene"], "method_version": C.METHOD_VERSION, "candidates": candidates}
    (ctx["out_dir"] / "rare_mineral_candidates.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    ctx["rare"] = out
    print(f"  Rare: { {k:v['status'] for k,v in candidates.items()} }")
    return out


def gate4_present_audit(ctx: dict) -> dict:
    """Per-scene PRESENT audit (run for all scenes, not only the pilot)."""
    proof = ctx["proof"]
    present = [g for g, e in proof["minerals"].items() if e["present"]]
    present_strong = list(proof.get("present_strong") or [])
    present_weak = list(proof.get("present_weak") or [])
    reasons = []
    if len(present) < 1:
        reasons.append("no_PRESENT_groups")
    if present == ["Ilmenite"] and proof["n_present"] == 1:
        reasons.append("only_ilmenite_present_review")
    # Unmix must not invent families absent from PRESENT
    mix = ctx.get("mixture_summary") or {}
    allowed = set(mix.get("allowed_families") or [])
    fam_ab = mix.get("mean_family_abundance") or {}
    for fam, val in fam_ab.items():
        if fam == "Ilmenite":
            continue
        if allowed and fam not in allowed and float(val) > 0.05:
            reasons.append(f"unmix_invented_{fam}={float(val):.3f}")
    hard = [r for r in reasons if r == "no_PRESENT_groups" or r.startswith("unmix_invented_")]
    soft = [r for r in reasons if r.endswith("_review")]
    report = {
        "gate": "G4_PRESENT_audit",
        "scene": ctx["scene"],
        "pass": len(hard) == 0,
        "present_groups": present,
        "present_strong": present_strong,
        "present_weak": present_weak,
        "overall_verdict": proof["overall_verdict"],
        "mixture_qa": mix.get("qa_status"),
        "reasons": hard + soft,
    }
    (C.GATES_DIR / f"G4_{ctx['scene'].replace(' ', '_')}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"  Gate4: {'PASS' if report['pass'] else 'FAIL'} "
        f"strong={present_strong} weak={present_weak}"
    )
    return report


# backward-compatible alias
def gate4_pilot_present(ctx: dict) -> dict:
    return gate4_present_audit(ctx)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_mineral_id(
    scene_name: str,
    mode: str = "full",
    prior_dir: Optional[str] = None,
    skip_encoder: bool = False,
) -> dict:
    print("=" * 60)
    print(f"M3 mineral ID — {scene_name}  mode={mode}")
    print("=" * 60)

    g0 = sio.gate0_scene_ready(scene_name)
    print(f"Gate0: {'PASS' if g0['pass'] else 'FAIL'} {g0.get('reasons')}")
    if not g0["pass"]:
        return {"gate0": g0}

    # Library (rebuild if missing or method_version mismatch — CR must match scene)
    need_lib = True
    lib_meta_p = C.SHARED_DIR / "lib_names.json"
    if lib_meta_p.is_file():
        meta = json.loads(lib_meta_p.read_text(encoding="utf-8"))
        if meta.get("method_version") == C.METHOD_VERSION and meta.get("cr_method") == "continuum_removal_match":
            need_lib = False
    if need_lib:
        print(f"  Rebuilding library for {C.METHOD_VERSION} ...")
        build_library(save=True)
    lib = load_built_library()
    g1 = json.loads((C.GATES_DIR / "G1_library_qc.json").read_text(encoding="utf-8"))
    print(f"Gate1: {'PASS' if g1['pass'] else 'FAIL'} n={g1['n_spectra']}")
    if not g1["pass"]:
        return {"gate0": g0, "gate1": g1}

    ctx = sio.load_phase1_scene(scene_name)
    print(f"Loaded cube {ctx['cube'].shape}  valid={int(ctx['valid'].sum()):,}")

    ctx = stage2_spm(ctx)
    g2 = gate2_spm_sanity(ctx)
    if not g2["pass"]:
        return {"gate0": g0, "gate1": g1, "gate2": g2, "out_dir": str(ctx["out_dir"])}

    if not skip_encoder:
        prior = None
        if mode == "adapt":
            prior = Path(prior_dir or C.SHARED_DIR) / "encoder_B85.pth"
        ctx = stage3_encoder(ctx, lib, mode=mode, prior_path=prior)
        g3 = gate3_encoder_qa(ctx)
    else:
        g3 = {"pass": True, "skipped": True}

    # PRESENT proof BEFORE clustering/unmix:
    # - clustering names from PRESENT groups only
    # - unmix endmembers constrained to PRESENT families
    proof = write_mineral_proof(ctx, lib)

    if not skip_encoder:
        ctx = stage4_hdbscan(ctx)
        # clustering is informational — patch map fields only (never change verdict)
        proof["n_clusters"] = int(ctx.get("n_clusters") or 0)
        proof["clustering_status"] = ctx.get("clustering_status")
        proof["map_available"] = bool(proof["n_clusters"] > 0)
        ctx = stage5_unmix(ctx, lib)
        proof["mixture_secondary"] = ctx.get("mixture_summary", {}).get("olivine_vs_pyroxene")
        proof["mixture_qa"] = ctx.get("mixture_summary", {}).get("qa_status")
        proof["method_version"] = C.METHOD_VERSION
        (ctx["out_dir"] / "mineral_proof.json").write_text(
            json.dumps(proof, indent=2), encoding="utf-8"
        )
        ctx["proof"] = proof

    rare = detect_rare_minerals(ctx, lib)
    g4 = gate4_present_audit(ctx)

    summary = {
        "scene": scene_name,
        "mode": mode,
        "method_version": C.METHOD_VERSION,
        "out_dir": str(ctx["out_dir"]),
        "gate0": g0,
        "gate1": g1,
        "gate2": g2,
        "gate3": g3,
        "gate4": g4,
        "present": [g for g, e in proof["minerals"].items() if e["present"]],
        "present_strong": list(proof.get("present_strong") or []),
        "present_weak": list(proof.get("present_weak") or []),
        "overall_verdict": proof["overall_verdict"],
        "n_clusters": proof.get("n_clusters"),
        "clustering_status": proof.get("clustering_status"),
        "mixture_secondary": ctx.get("mixture_summary", {}).get("olivine_vs_pyroxene"),
        "mixture_qa": ctx.get("mixture_summary", {}).get("qa_status"),
        "rare": rare.get("candidates", {}),
    }
    (ctx["out_dir"] / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nDone.", summary["present"], summary["overall_verdict"])
    return summary
