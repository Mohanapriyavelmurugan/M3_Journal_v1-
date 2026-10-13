"""
Minimum path: retrain unmixer only into scene 06_mineral_id using saved encoder.
Updates mixture_summary.json, unmixer_B85.pth, mineral_proof mixture fields,
run_summary.json, method_version.txt. Does NOT retrain the encoder.
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

OUT = Path(r"D:\Journal\M3 Journal\Copernicus Crater\Pipeline_outputs\06_mineral_id")


def main() -> int:
    print("=" * 60)
    print(f"Unmix-only rerun  method={C.METHOD_VERSION}")
    print(f"out={OUT}")
    print("=" * 60)
    if C.METHOD_VERSION != "depth_v2d_unmix":
        print(f"ERROR: expected depth_v2d_unmix, got {C.METHOD_VERSION}")
        return 1

    enc_path = OUT / "encoder_B85.pth"
    if not enc_path.is_file():
        print(f"ERROR: missing encoder {enc_path}")
        return 1

    print("Loading cube_cr / SPM artifacts...")
    cube = np.load(OUT / "cube_cr.npy")
    spm_map = np.load(OUT / "spm_map.npy")
    spm_active = np.load(OUT / "spm_active_mask.npy")
    gate = json.loads((OUT / "spm_gate_info.json").read_text(encoding="utf-8"))
    valid = np.isfinite(cube).sum(axis=-1) >= 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = SpectralEncoder(n_bands=C.N_BANDS_GLOBAL).to(device)
    enc.load_state_dict(
        torch.load(enc_path, map_location=device, weights_only=True), strict=False
    )
    enc.eval()
    print(f"Encoder loaded ({device})")

    lib = load_built_library()
    ctx = {
        "scene": "Copernicus Crater",
        "out_dir": OUT,
        "cube_cr": cube,
        "spm_map": spm_map,
        "spm_active": spm_active,
        "spm_gate_info": gate,
        "valid": valid,
        "encoder": enc,
        "training_coords": [],
    }
    ctx = stage5_unmix(ctx, lib)
    mix = ctx.get("mixture_summary") or {}
    qa = mix.get("qa_status")
    print(f"\nUnmix QA={qa}  citable={mix.get('citable')}")
    print(f"  fam={mix.get('mean_family_abundance')}")
    print(f"  checks={mix.get('checks')}")

    # Stamp method version
    (OUT / "method_version.txt").write_text(C.METHOD_VERSION + "\n", encoding="utf-8")

    # Patch mineral_proof.json mixture fields
    proof_p = OUT / "mineral_proof.json"
    if proof_p.is_file():
        proof = json.loads(proof_p.read_text(encoding="utf-8"))
        proof["method_version"] = C.METHOD_VERSION
        proof["mixture_secondary"] = mix.get("olivine_vs_pyroxene")
        proof["mixture_qa"] = qa
        proof_p.write_text(json.dumps(proof, indent=2), encoding="utf-8")
        print(f"Patched {proof_p.name}")

    # Patch run_summary.json
    sum_p = OUT / "run_summary.json"
    if sum_p.is_file():
        summary = json.loads(sum_p.read_text(encoding="utf-8"))
        summary["method_version"] = C.METHOD_VERSION
        summary["mixture_secondary"] = mix.get("olivine_vs_pyroxene")
        summary["mixture_qa"] = qa
        summary["unmix_only_rerun"] = True
        sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Patched {sum_p.name}")

    # Patch G4 if present (PRESENT unchanged; keep record of mixture)
    g4_p = C.GATES_DIR / "G4_Copernicus_Crater.json"
    if g4_p.is_file():
        g4 = json.loads(g4_p.read_text(encoding="utf-8"))
        g4["method_version"] = C.METHOD_VERSION
        g4["mixture_qa"] = qa
        g4_p.write_text(json.dumps(g4, indent=2), encoding="utf-8")

    if qa != "PASS":
        print("\nRESULT: FAIL — unmix did not PASS; do not proceed to pretrain")
        return 2
    print("\nRESULT: PASS — unmix-only complete; PRESENT left unchanged")
    print("You can proceed to pretraining when ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
