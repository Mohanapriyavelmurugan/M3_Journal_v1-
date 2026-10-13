"""
CLI for M3 Phase 2 mineral identification.

Examples:
  python m3_run_mineral.py --scene "Copernicus Crater" --mode full
  python m3_run_mineral.py --scene "Jackson Crater" --mode adapt
  python m3_run_mineral.py --build-library
  python m3_run_mineral.py --gates-only --scene "Copernicus Crater"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import m3_mineral_constants as C
import m3_scene_io as sio
from m3_spectral_library import build_library
from m3_mineral_id import run_mineral_id


def main():
    ap = argparse.ArgumentParser(description="M3 Phase 2 mineral identification")
    ap.add_argument("--scene", type=str, default=C.PILOT_SCENE)
    ap.add_argument("--mode", choices=["full", "adapt"], default="full")
    ap.add_argument("--prior-dir", type=str, default=str(C.SHARED_DIR))
    ap.add_argument("--build-library", action="store_true")
    ap.add_argument("--gates-only", action="store_true", help="Run G0 only")
    ap.add_argument("--skip-encoder", action="store_true", help="SPM+PRESENT only (fast debug)")
    args = ap.parse_args()

    if args.build_library:
        build_library(save=True)
        return 0

    if args.gates_only:
        g0 = sio.gate0_scene_ready(args.scene)
        print(json.dumps(g0, indent=2))
        return 0 if g0["pass"] else 1

    summary = run_mineral_id(
        scene_name=args.scene,
        mode=args.mode,
        prior_dir=args.prior_dir,
        skip_encoder=args.skip_encoder,
    )
    ok = True
    for k in ("gate0", "gate1", "gate2"):
        if k in summary and isinstance(summary[k], dict) and not summary[k].get("pass", True):
            ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
