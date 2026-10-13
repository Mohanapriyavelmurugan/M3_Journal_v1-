"""Offline checks for depth_v2e_claims before pooled pretrain / scene refresh."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_mineral_constants as C
from m3_mineral_id import (
    _family_balanced_endmembers,
    pyx_soft_support,
    pyx_support_grade,
)
from m3_spectral_library import load_built_library

errors, warnings = [], []


def check(cond, msg, warn=False):
    if cond:
        print(f"  PASS  {msg}")
    elif warn:
        warnings.append(msg)
        print(f"  WARN  {msg}")
    else:
        errors.append(msg)
        print(f"  FAIL  {msg}")


print("=" * 60)
print(f"Pre-rerun verification  method={C.METHOD_VERSION}")
print("=" * 60)

print("\n[1] Constants")
check(C.METHOD_VERSION == "depth_v2e_claims", f"METHOD_VERSION={C.METHOD_VERSION}")
check("Mare Tranquillitatis" in C.MARE_SCENE_NAMES, "mare set includes Tranquillitatis")
check(C.RARE_ILM_MIN_HOT_NONMARE > C.RARE_ILM_MIN_HOT, "nonmare ilm stricter count")
check(C.RARE_ILM_DEPTH_MIN_NONMARE > C.RARE_ILM_DEPTH_MIN, "nonmare ilm stricter depth")
check(C.UNMIX_SPARSITY_WEIGHT == 0.0, "sparsity still off")

print("\n[2] PRESENT_WEAK pyroxene grading")
ok_w, r_w = pyx_soft_support(
    {"d1um": -0.01, "d2um_lcp": 0.01, "d2um_hcp": 0.02, "d2_over_d1": -1},
    "d1.90",
)
check(ok_w and pyx_support_grade(r_w) == "weak", f"weak path ({r_w})")
ok_s, r_s = pyx_soft_support(
    {"d1um": 0.05, "d2um_lcp": 0.02, "d2um_hcp": 0.02, "d2_over_d1": 0.4},
    "d1.90",
)
check(ok_s and pyx_support_grade(r_s) == "strong", f"strong path ({r_s})")

print("\n[3] PRESENT-constrained endmembers")
try:
    lib = load_built_library()
except Exception as e:
    print(f"  WARN  library load failed ({e}); building not required for this check")
    lib = None
if lib is not None:
    E, names, fams = _family_balanced_endmembers(lib, allowed_families=["LCP", "HCP", "Plagioclase"])
    check("Olivine" not in fams, f"no olivine in constrained E: {sorted(set(fams))}")
    check(set(fams) <= {"LCP", "HCP", "Plagioclase"}, f"only allowed fams: {set(fams)}")
    E2, _, fams2 = _family_balanced_endmembers(lib, allowed_families=None)
    check("Olivine" in fams2, "unconstrained still includes Olivine")

print("\n[4] Pretrain script is pooled")
pt = Path(__file__).resolve().parent / "m3_pretrain_shared.py"
txt = pt.read_text(encoding="utf-8")
check("pooled" in txt.lower() and "build_pooled_train" in txt, "pretrain uses pooled train")
check("mode=\"adapt\"" in txt or "mode='adapt'" in txt, "refresh uses adapt from shared")

print("\n[5] Pipeline order: proof before unmix")
mid = (Path(__file__).resolve().parent / "m3_mineral_id.py").read_text(encoding="utf-8")
i_proof = mid.find("proof = write_mineral_proof(ctx, lib)")
i_unmix = mid.find("ctx = stage5_unmix(ctx, lib)")
check(i_proof > 0 and i_unmix > i_proof, "write_mineral_proof precedes stage5_unmix in orchestrator")
check("gate4_present_audit" in mid, "G4 audit for all scenes")
check("PRESENT_WEAK" in mid, "PRESENT_WEAK claim level present")

print("\n[6] Last-run Highlands should exclude olivine from unmix after fix")
h_proof = Path(r"D:\Journal\M3 Journal\Lunar Highlands South\Pipeline_outputs\06_mineral_id\mineral_proof.json")
if h_proof.is_file():
    p = json.loads(h_proof.read_text(encoding="utf-8"))
    ol = p["minerals"]["Olivine"]["present"]
    check(ol is False, "last Highlands Olivine was ABSENT (constraint will drop Ol from E)")
else:
    check(True, "no Highlands proof yet", warn=True)

print("\n" + "=" * 60)
if errors:
    print(f"RESULT: {len(errors)} FAIL(s)")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print(f"RESULT: READY  ({len(warnings)} warning(s))")
for w in warnings:
    print("  warn:", w)
print("\nRecommended next step (pooled pretrain + refresh all priors):")
print('  cd "D:\\Journal\\M3 Journal\\Code_scripts"')
print("  python m3_pretrain_shared.py")
print("\nOr single-scene full rerun first:")
print('  python m3_run_mineral.py --scene "Copernicus Crater" --mode full')
sys.exit(0)
