"""
Offline unmixer smoke test for depth_v2d_unmix.
Loads last Copernicus encoder + cube_cr + SPM maps; trains ONLY the unmixer.
Does not overwrite the scene's mixture_summary.json (writes to _smoke_unmix_out/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_mineral_constants as C
from m3_mineral_id import SpectralEncoder, stage5_unmix
from m3_spectral_library import load_built_library

SRC = Path(r"D:\Journal\M3 Journal\Copernicus Crater\Pipeline_outputs\06_mineral_id")
SMOKE = Path(__file__).resolve().parent / "_smoke_unmix_out"
SMOKE.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"Unmix smoke  method={C.METHOD_VERSION}")
print("=" * 60)

assert C.METHOD_VERSION == "depth_v2d_unmix", C.METHOD_VERSION
assert C.UNMIX_SPARSITY_WEIGHT == 0.0, "sparsity must be off"
assert C.UNMIX_SOFTMAX_TEMP > 1.0, "need softened softmax"
assert C.UNMIX_FAM_DIV_WEIGHT > 0, "need family diversity penalty"

cube_cr = np.load(SRC / "cube_cr.npy", mmap_mode="r")
spm_map = np.load(SRC / "spm_map.npy")
spm_active = np.load(SRC / "spm_active_mask.npy")
gate = json.loads((SRC / "spm_gate_info.json").read_text(encoding="utf-8"))
# materialize a compact view for coord sampling (mmap ok for indexing)
valid = np.isfinite(np.asarray(cube_cr[:, :, 0])) & np.isfinite(np.asarray(cube_cr[:, :, 40]))
# stronger valid: enough finite bands
# (full all-band check is expensive; two-band proxy + later per-pixel filter)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = SpectralEncoder(n_bands=C.N_BANDS_GLOBAL).to(device)
state = torch.load(SRC / "encoder_B85.pth", map_location=device, weights_only=True)
enc.load_state_dict(state, strict=False)
enc.eval()
print(f"  encoder loaded from {SRC / 'encoder_B85.pth'}  device={device}")

# Need real cube array for training pulls (mmap fancy index can be slow but OK)
cube = np.load(SRC / "cube_cr.npy")  # ~1.8GB — required for correct spectra
valid = np.isfinite(cube).sum(axis=-1) >= 20

ctx = {
    "scene": "Copernicus Crater",
    "out_dir": SMOKE,
    "cube_cr": cube,
    "spm_map": spm_map,
    "spm_active": spm_active,
    "spm_gate_info": gate,
    "valid": valid,
    "encoder": enc,
    "training_coords": [],  # unused — balanced coords rebuilt inside stage5
}

lib = load_built_library()
print(f"  library n={len(lib['names'])}")
ctx = stage5_unmix(ctx, lib)
mix = ctx.get("mixture_summary") or {}
qa = mix.get("qa_status")
checks = mix.get("checks") or {}
fam = mix.get("mean_family_abundance") or {}
print("\n" + "=" * 60)
print(f"SMOKE RESULT: qa={qa}  citable={mix.get('citable')}")
print(f"  mean_max={checks.get('mean_max_abund')}")
print(f"  family_max={checks.get('family_max_abund')}  family_min_core={checks.get('family_min_core')}")
print(f"  fam={fam}")
print(f"  wrote {SMOKE / 'mixture_summary.json'}")
if qa != "PASS":
    print("RESULT: NOT READY — unmix still failing offline")
    sys.exit(1)
print("RESULT: READY TO RERUN (unmix PASS on smoke)")
print('  cd "D:\\Journal\\M3 Journal\\Code_scripts"')
print('  python m3_run_mineral.py --scene "Copernicus Crater" --mode full')
sys.exit(0)
