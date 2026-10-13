"""
Fast prior refresh for depth_v2g_clusters.

Reuses pooled shared encoder (NO retrain).
Order: SPM -> encode(capped) -> PRESENT -> HDBSCAN(adaptive) -> unmix -> rare -> G4

  python _refresh_priors_v2g.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_mineral_constants as C
import m3_scene_io as sio
from m3_mineral_id import (
    detect_rare_minerals,
    gate2_spm_sanity,
    gate3_encoder_qa,
    gate4_present_audit,
    stage2_spm,
    stage3_encoder_frozen,
    stage4_hdbscan,
    stage5_unmix,
    write_mineral_proof,
)
from m3_spectral_library import build_library, load_built_library


def main() -> int:
    assert C.METHOD_VERSION == "depth_v2g_clusters", C.METHOD_VERSION
    enc = C.SHARED_DIR / "encoder_B85.pth"
    if not enc.is_file():
        print(f"ERROR: missing shared encoder {enc}")
        return 1

    need_lib = True
    meta_p = C.SHARED_DIR / "lib_names.json"
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("method_version") == C.METHOD_VERSION:
            need_lib = False
    if need_lib:
        print("Rebuilding library for", C.METHOD_VERSION)
        build_library(save=True)
    lib = load_built_library()

    results = []
    for scene in C.PRIOR_SCENES:
        print("\n" + "#" * 60)
        print(f"# REFRESH {scene}  ({C.METHOD_VERSION})")
        print("#" * 60)
        g0 = sio.gate0_scene_ready(scene)
        if not g0.get("pass"):
            print("G0 FAIL", g0)
            return 2
        ctx = sio.load_phase1_scene(scene)
        ctx = stage2_spm(ctx)
        g2 = gate2_spm_sanity(ctx)
        if not g2.get("pass"):
            print("G2 FAIL", g2)
            return 2
        ctx = stage3_encoder_frozen(ctx, enc)
        g3 = gate3_encoder_qa(ctx)

        proof = write_mineral_proof(ctx, lib)
        ctx = stage4_hdbscan(ctx)
        proof["n_clusters"] = int(ctx.get("n_clusters") or 0)
        proof["clustering_status"] = ctx.get("clustering_status")
        proof["map_available"] = bool(proof["n_clusters"] > 0)

        ctx = stage5_unmix(ctx, lib)
        proof["mixture_secondary"] = ctx.get("mixture_summary", {}).get("olivine_vs_pyroxene")
        proof["mixture_qa"] = ctx.get("mixture_summary", {}).get("qa_status")
        (ctx["out_dir"] / "mineral_proof.json").write_text(
            json.dumps(proof, indent=2), encoding="utf-8"
        )
        ctx["proof"] = proof

        rare = detect_rare_minerals(ctx, lib)
        g4 = gate4_present_audit(ctx)
        summary = {
            "scene": scene,
            "mode": "frozen_shared_refresh",
            "method_version": C.METHOD_VERSION,
            "out_dir": str(ctx["out_dir"]),
            "shared_encoder": str(enc),
            "gate0": g0,
            "gate2": g2,
            "gate3": g3,
            "gate4": g4,
            "present": [g for g, e in proof["minerals"].items() if e["present"]],
            "present_strong": list(proof.get("present_strong") or []),
            "present_weak": list(proof.get("present_weak") or []),
            "overall_verdict": proof["overall_verdict"],
            "n_clusters": proof.get("n_clusters"),
            "clustering_status": proof.get("clustering_status"),
            "mixture_qa": ctx.get("mixture_summary", {}).get("qa_status"),
            "rare": rare.get("candidates", {}),
        }
        (ctx["out_dir"] / "run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (ctx["out_dir"] / "method_version.txt").write_text(
            C.METHOD_VERSION + "\n", encoding="utf-8"
        )
        results.append(summary)
        print(
            f"Done {scene}: verdict={summary['overall_verdict']} "
            f"clusters={summary['n_clusters']} ({summary['clustering_status']}) "
            f"strong={summary['present_strong']}"
        )

    man_p = C.SHARED_DIR / "pretrain_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.is_file() else {}
    man["method_version"] = C.METHOD_VERSION
    man["claims_refresh"] = "depth_v2g_clusters"
    man["results"] = [
        {
            "scene": r["scene"],
            "present_strong": r["present_strong"],
            "present_weak": r["present_weak"],
            "overall_verdict": r["overall_verdict"],
            "n_clusters": r["n_clusters"],
            "clustering_status": r["clustering_status"],
            "mixture_qa": r["mixture_qa"],
            "gate4": r.get("gate4", {}).get("pass"),
            "rare_ilm": (r.get("rare") or {}).get("Ilmenite", {}).get("status"),
        }
        for r in results
    ]
    man_p.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print("\nAll priors refreshed under depth_v2g_clusters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
