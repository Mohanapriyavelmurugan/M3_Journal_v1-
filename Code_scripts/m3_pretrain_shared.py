"""
Pooled multi-scene pretrain for shared M3 encoder.

Phase A: build SPM + train pools on all PRIOR_SCENES, concatenate, train ONE encoder.
Phase B: refresh each prior scene with mode=adapt from the shared encoder
         (PRESENT, unmix, rare, G4) so scene outputs match depth_v2e_claims.

  python m3_pretrain_shared.py
  python m3_pretrain_shared.py --skip-refresh   # pooled train only
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import m3_mineral_constants as C
import m3_scene_io as sio
from m3_mineral_id import (
    _build_train_pool,
    gate2_spm_sanity,
    run_mineral_id,
    stage2_spm,
    train_encoder_on_pool,
)
from m3_spectral_library import build_library, load_built_library


def _ensure_library() -> dict:
    need_lib = True
    lib_meta_p = C.SHARED_DIR / "lib_names.json"
    if lib_meta_p.is_file():
        meta = json.loads(lib_meta_p.read_text(encoding="utf-8"))
        if meta.get("method_version") == C.METHOD_VERSION and meta.get("cr_method") == "continuum_removal_match":
            need_lib = False
    if need_lib:
        print(f"Rebuilding library for {C.METHOD_VERSION} ...")
        build_library(save=True)
    return load_built_library()


def build_pooled_train(scenes: list, lib: dict) -> tuple:
    """Return concatenated (X, S, waves, per-scene sizes)."""
    Xs, Ss = [], []
    waves = None
    sizes = {}
    for scene in scenes:
        print("\n" + "#" * 60)
        print(f"# POOL SCENE: {scene}")
        print("#" * 60)
        g0 = sio.gate0_scene_ready(scene)
        if not g0.get("pass"):
            raise SystemExit(f"G0 fail for {scene}: {g0.get('reasons')}")
        ctx = sio.load_phase1_scene(scene)
        ctx = stage2_spm(ctx)
        g2 = gate2_spm_sanity(ctx)
        if not g2.get("pass"):
            raise SystemExit(f"G2 fail for {scene}: {g2.get('reasons')}")
        X, S = _build_train_pool(ctx, lib)
        # Cap per-scene contribution so one large scene cannot dominate
        MAX_PER = 20_000
        if len(X) > MAX_PER:
            rng = np.random.default_rng(abs(hash(scene)) % (2**31))
            sel = rng.choice(len(X), MAX_PER, replace=False)
            X, S = X[sel], S[sel]
        print(f"  pool contrib={len(X)}")
        Xs.append(X)
        Ss.append(S)
        sizes[scene] = int(len(X))
        if waves is None:
            waves = ctx["waves"]
    X = np.concatenate(Xs, axis=0)
    S = np.concatenate(Ss, axis=0)
    # shuffle pooled
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X, S = X[perm], S[perm]
    print(f"\nPooled train size={len(X)} from {sizes}")
    return X, S, waves, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=C.PRIOR_SCENES)
    ap.add_argument("--skip-refresh", action="store_true",
                    help="Only train pooled shared encoder; do not re-run scenes")
    args = ap.parse_args()

    C.SHARED_DIR.mkdir(parents=True, exist_ok=True)
    C.GATES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"POOLED PRETRAIN  method={C.METHOD_VERSION}")
    print(f"scenes={args.scenes}")
    print("=" * 60)

    lib = _ensure_library()
    X, S, waves, sizes = build_pooled_train(args.scenes, lib)

    dest = C.SHARED_DIR / "encoder_B85.pth"
    print("\n=== Training pooled shared encoder ===")
    train_encoder_on_pool(X, S, waves, dest, mode="full", prior_path=None)
    print(f"Shared encoder -> {dest}")

    results = []
    if not args.skip_refresh:
        print("\n=== Refreshing prior scenes with shared encoder (adapt) ===")
        for scene in args.scenes:
            print("\n" + "#" * 60)
            print(f"# REFRESH SCENE: {scene}")
            print("#" * 60)
            summary = run_mineral_id(
                scene_name=scene,
                mode="adapt",
                prior_dir=str(C.SHARED_DIR),
            )
            results.append(summary)
            # Also copy scene encoder from shared so scene folder stays consistent
            out_enc = Path(summary.get("out_dir", "")) / "encoder_B85.pth"
            if out_enc.parent.is_dir() and dest.is_file():
                # adapt already saved a fine-tuned scene encoder; keep it.
                # Additionally store pointer to shared source in summary.
                summary["shared_encoder_source"] = str(dest)
                (Path(summary["out_dir"]) / "run_summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )

    meta = {
        "method_version": C.METHOD_VERSION,
        "pretrain_mode": "pooled",
        "prior_scenes": args.scenes,
        "pool_sizes": sizes,
        "pooled_n": int(len(X)),
        "shared_encoder": str(dest),
        "refreshed": (not args.skip_refresh),
        "results": [
            {
                "scene": r.get("scene"),
                "present": r.get("present"),
                "present_strong": r.get("present_strong"),
                "present_weak": r.get("present_weak"),
                "overall_verdict": r.get("overall_verdict"),
                "mixture_qa": r.get("mixture_qa"),
                "gate2": r.get("gate2", {}).get("pass"),
                "gate3": r.get("gate3", {}).get("pass"),
                "gate4": r.get("gate4", {}).get("pass"),
            }
            for r in results
        ],
    }
    (C.SHARED_DIR / "pretrain_manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    # keep a backup copy named explicitly
    shutil.copy2(dest, C.SHARED_DIR / "encoder_B85_pooled.pth")
    print(f"\nPooled pretrain complete -> {dest}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
