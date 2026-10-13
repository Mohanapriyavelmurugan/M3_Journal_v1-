"""
M3 Phase 2 scene I/O — load READY denoised cubes + wavelengths + masks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import m3_mineral_constants as C
import step0_load as s0


def load_readiness_manifest() -> list:
    path = C.JOURNAL_ROOT / "scene_readiness_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    # Fix stale D:\M3 Journal paths → D:\Journal\M3 Journal
    for row in data:
        for k in ("denoised_cube", "validation_report"):
            if k in row and isinstance(row[k], str):
                row[k] = row[k].replace(r"D:\M3 Journal", str(C.JOURNAL_ROOT))
                row[k] = row[k].replace(r"D:/M3 Journal", str(C.JOURNAL_ROOT))
    return data


def get_scene_record(scene_name: str) -> dict:
    for row in load_readiness_manifest():
        if row["scene"] == scene_name:
            return row
    raise KeyError(f"Scene not in readiness manifest: {scene_name}")


def find_raw_scene_dir(scene_name: str, fname: str) -> Path:
    region = C.JOURNAL_ROOT / scene_name
    # Prefer folder matching fname
    direct = region / fname
    if direct.is_dir():
        return direct
    cands = list(region.glob("M3*_V01_RFL")) + list(region.glob("M3*_V01_RFL-*"))
    for c in cands:
        if fname in c.name or c.name.startswith(fname[:20]):
            return c
    if cands:
        return cands[0]
    raise FileNotFoundError(f"No raw scene dir under {region}")


def load_phase1_scene(scene_name: str) -> dict:
    """Load denoised cube, waves, valid mask for a READY scene."""
    rec = get_scene_record(scene_name)
    fname = rec["fname"]
    den_path = Path(rec["denoised_cube"])
    if not den_path.is_file():
        # try local relative
        den_path = (
            C.JOURNAL_ROOT / scene_name / "Pipeline_outputs" / "04_denoised"
            / f"{fname}_denoised.npy"
        )
    if not den_path.is_file():
        raise FileNotFoundError(f"Denoised cube missing: {den_path}")

    cube = np.load(den_path, mmap_mode="r")
    H, W, B = cube.shape
    raw_dir = find_raw_scene_dir(scene_name, fname)
    # wavelengths from HDR only (do not reload full IMG)
    waves = C.M3_WAVELENGTHS_85_UM.copy() if B == 85 else None
    meta = {"H": H, "W": W, "B": B, "scene_dir": str(raw_dir)}
    try:
        files = s0.find_scene_files(str(raw_dir))
        hdr_nm = s0.parse_hdr_wavelengths(files["hdr"])
        waves = s0.assign_wavelengths(hdr_nm, B)
    except Exception as exc:
        if waves is None:
            raise RuntimeError(f"Cannot resolve wavelengths for {scene_name}: {exc}") from exc
        print(f"  [warn] HDR waves failed ({exc}); using canonical {B}-band table")

    if len(waves) != B:
        if B == 85:
            waves = C.M3_WAVELENGTHS_85_UM.copy()
        else:
            raise ValueError(f"Wave/band mismatch: waves={len(waves)} B={B}")

    bad = s0.get_bad_band_mask(waves)
    # valid pixel: finite fraction across non-bad bands
    good_bands = ~bad
    # sample validity without loading full cube into RAM twice
    cube_mem = np.asarray(cube)
    finite = np.isfinite(cube_mem[:, :, good_bands]).mean(axis=2)
    valid = finite >= 0.70

    out_dir = C.JOURNAL_ROOT / scene_name / "Pipeline_outputs" / "06_mineral_id"
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "scene": scene_name,
        "fname": fname,
        "cube": cube_mem,
        "waves": np.asarray(waves, dtype=np.float64),
        "valid": valid,
        "bad_bands": bad,
        "meta": meta,
        "out_dir": out_dir,
        "readiness": rec,
        "n_bands": B,
    }


def gate0_scene_ready(scene_name: str) -> dict:
    rec = get_scene_record(scene_name)
    den = Path(str(rec["denoised_cube"]).replace(r"D:\M3 Journal", str(C.JOURNAL_ROOT)))
    if not den.is_file():
        den = (
            C.JOURNAL_ROOT / scene_name / "Pipeline_outputs" / "04_denoised"
            / f"{rec['fname']}_denoised.npy"
        )
    reasons = []
    if rec.get("readiness") != "READY":
        reasons.append(f"readiness={rec.get('readiness')}")
    if rec.get("verdict") != "PASS":
        reasons.append(f"verdict={rec.get('verdict')}")
    if not den.is_file():
        reasons.append("denoised.npy missing")
    if scene_name in C.DEFERRED_SCENES:
        reasons.append("deferred_scene")
    report = {
        "gate": "G0_Scene_READY",
        "scene": scene_name,
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "fname": rec.get("fname"),
        "denoised": str(den),
    }
    C.GATES_DIR.mkdir(parents=True, exist_ok=True)
    (C.GATES_DIR / f"G0_{scene_name.replace(' ', '_')}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
