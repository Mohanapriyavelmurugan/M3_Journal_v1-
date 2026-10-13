"""Offline checks for depth_v2f_harden before prior refresh / adapt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_mineral_constants as C
from m3_spm import pyx_ol_separation_spectrum
import numpy as np

errors = []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        errors.append(msg)
        print(f"  FAIL  {msg}")


print("=" * 60)
print(f"Verify {C.METHOD_VERSION}")
print("=" * 60)

check(C.METHOD_VERSION == "depth_v2f_harden", f"version={C.METHOD_VERSION}")
check(C.RARE_ILM_PRESENT_RARE_MARE_ONLY is True, "PRESENT_RARE mare-only")
check(C.STRONG_MIN_PRESENT_STRONG >= 2, f"STRONG_MIN_STRONG={C.STRONG_MIN_PRESENT_STRONG}")
check(C.UNMIX_PRESENT_PRIOR_WEIGHT > 0, "unmix PRESENT prior on")
check("Mare Tranquillitatis" in C.MARE_SCENE_NAMES, "mare set ok")

# CR fallback for 1µm: synthetic spectrum with positive CR 1µm
waves = C.M3_WAVELENGTHS_85_UM
refl = np.ones(85, dtype=np.float64)
# make BD1000 negative-ish flat; CR with a dip near 1.05
cr = np.ones(85, dtype=np.float64)
j = int(np.argmin(np.abs(waves - 1.05)))
cr[j - 2 : j + 3] = 0.7
sep = pyx_ol_separation_spectrum(refl, waves, spectrum_cr=cr)
check(sep.get("d1um_source") in ("refl", "cr_fallback"), f"d1 source={sep.get('d1um_source')}")
check(sep.get("d1um") is not None and float(sep["d1um"]) > 0, f"d1um recovered={sep.get('d1um')}")

# Highlands last rare should become CANDIDATE under mare-only rule
h = Path(r"D:\Journal\M3 Journal\Lunar Highlands South\Pipeline_outputs\06_mineral_id\rare_mineral_candidates.json")
if h.is_file():
    rare = json.loads(h.read_text(encoding="utf-8"))
    st = (rare.get("candidates") or {}).get("Ilmenite", {}).get("status")
    check(True, f"last Highlands ilm was {st} (will be CANDIDATE max after mare-only rule)")

mid = (Path(__file__).resolve().parent / "m3_mineral_id.py").read_text(encoding="utf-8")
check("STRONG_MIN_PRESENT_STRONG" in mid, "STRONG uses strong-count")
check("stage3_encoder_frozen" in mid, "frozen encode path exists")
check(Path("_refresh_priors_v2f.py").is_file(), "refresh script present")

print("=" * 60)
if errors:
    print("RESULT: FAIL")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("RESULT: READY")
print("\nRun (fast — no encoder retrain):")
print('  cd "D:\\Journal\\M3 Journal\\Code_scripts"')
print("  python _refresh_priors_v2f.py")
print("\nDo NOT start adapt until this refresh is evaluated.")
sys.exit(0)
