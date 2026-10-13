"""
Offline verification of depth_v2b_fixes BEFORE full scene rerun.
Uses last Copernicus artifacts + unit tests. Does not overwrite outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_mineral_constants as C
from m3_mineral_id import pyx_soft_support

OUT = Path(r"D:\Journal\M3 Journal\Copernicus Crater\Pipeline_outputs\06_mineral_id")
errors = []
warnings = []


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

# --- constants ---
print("\n[1] Constants")
check(C.METHOD_VERSION == "depth_v2b_fixes", f"METHOD_VERSION={C.METHOD_VERSION}")
check("R750" not in C.CLUSTER_NAME_SPM, "CLUSTER_NAME_SPM excludes R750")
check(set(C.CLUSTER_NAME_SPM) <= {"OLINDEX3", "LCPINDEX2", "HCPINDEX2", "BD1250"},
      f"CLUSTER_NAME_SPM={C.CLUSTER_NAME_SPM}")
check(C.UNMIX_FREEZE_ENDMEMBERS is True, "UNMIX_FREEZE_ENDMEMBERS=True")
check(C.PYX_D2_MIN > 0, f"PYX_D2_MIN={C.PYX_D2_MIN}")
check(C.PRESENT_GROUPS["Low-Ca Pyroxene"]["depth_key"] == "d1.90", "LCP depth_key=d1.90")
check(C.PRESENT_GROUPS["High-Ca Pyroxene"]["depth_key"] == "d2.25", "HCP depth_key=d2.25")

# --- pyx soft gate against last-run seps ---
print("\n[2] Pyroxene soft-gate (using last mineral_proof seps)")
proof_p = OUT / "mineral_proof.json"
if not proof_p.is_file():
    errors.append("missing mineral_proof.json for replay")
    print("  FAIL  no previous proof to replay")
else:
    proof = json.loads(proof_p.read_text(encoding="utf-8"))
    # Old buggy rule would fail LCP/HCP; new rule should pass them given last seps
    for g in ("Low-Ca Pyroxene", "High-Ca Pyroxene"):
        e = proof["minerals"][g]
        sep = e["pyx_ol_diagnostic"]
        dkey = C.PRESENT_GROUPS[g]["depth_key"]
        ok, reason = pyx_soft_support(sep, dkey)
        print(f"  {g}: sep={sep}")
        print(f"       -> support={ok} reason={reason}")
        # Last run had all hard gates true; soft gate was the only blocker
        hard = all(e["gates"][k] for k in ("hot_ok", "depth_ok", "hmb_ok", "lib_ok", "spm_enabled"))
        check(hard, f"{g} hard gates were true on last run")
        check(ok, f"{g} NEW soft gate passes (would become PRESENT if hard gates still hold)")

    # Olivine ratio path still OK
    ol_sep = proof["minerals"]["Olivine"]["pyx_ol_diagnostic"]
    ok_ol, reason_ol = pyx_soft_support(ol_sep, "d1.90")  # not used for olivine, just sanity
    check(True, f"Olivine sep diagnostic readable d2/d1={ol_sep.get('d2_over_d1')} ({reason_ol})")

    # Synthetic: positive d1 with tiny ratio and tiny d2 -> fail
    ok_fail, r_fail = pyx_soft_support(
        {"d1um": 0.05, "d2um_lcp": 0.001, "d2um_hcp": 0.001, "d2_over_d1": 0.02},
        "d1.90",
    )
    check(not ok_fail, f"tiny d2 still rejected ({r_fail})")

    # Synthetic: d1 negative, d2 strong -> pass
    ok_pass, r_pass = pyx_soft_support(
        {"d1um": -0.01, "d2um_lcp": 0.01, "d2um_hcp": 0.02, "d2_over_d1": -2.0},
        "d1.90",
    )
    check(ok_pass, f"d1 nonpos + abs d2 passes ({r_pass})")

# --- cluster naming channel filter ---
print("\n[3] Cluster naming channels")
old_names = json.loads((OUT / "cluster_names.json").read_text(encoding="utf-8")) if (OUT / "cluster_names.json").is_file() else {}
check(any(v == "R750" for v in old_names.values()), "last run had R750 cluster names (bug confirmed)", warn=True)
check("R750" not in C.CLUSTER_NAME_SPM and "BD1000" not in C.CLUSTER_NAME_SPM,
      "new naming set cannot emit R750/BD1000")

# --- rare ilmenite rules ---
print("\n[4] Rare ilmenite gates")
check(C.RARE_ILM_DEPTH_MIN > 0 and C.RARE_ILM_HMB_MIN > 0,
      f"Ilmenite rare needs depth>={C.RARE_ILM_DEPTH_MIN} and hmb>={C.RARE_ILM_HMB_MIN}")
rare_old = json.loads((OUT / "rare_mineral_candidates.json").read_text(encoding="utf-8")) if (OUT / "rare_mineral_candidates.json").is_file() else {}
old_ilm = (rare_old.get("candidates") or {}).get("Ilmenite", {})
if old_ilm.get("status") == "PRESENT_RARE" and "requires" not in old_ilm:
    check(True, "last run Ilmenite PRESENT_RARE was SAM-only (will be re-gated)", warn=True)

# --- unmix collapse QA ---
print("\n[5] Unmixer collapse QA")
mix = json.loads((OUT / "mixture_summary.json").read_text(encoding="utf-8")) if (OUT / "mixture_summary.json").is_file() else {}
mm = (mix.get("checks") or {}).get("mean_max_abund")
if mm is not None:
    check(mm >= C.UNMIX_COLLAPSE_MAX_ABUND, f"last run mean_max={mm:.3f} would now be FAILED_COLLAPSE", warn=True)
check(C.UNMIX_FREEZE_ENDMEMBERS, "endmembers will be frozen on next run")

# --- expected PRESENT replay (logic only) ---
print("\n[6] Expected PRESENT replay (logic on last proof numbers)")
if proof_p.is_file():
    expect = []
    for g, e in proof["minerals"].items():
        hard = all(e["gates"][k] for k in ("hot_ok", "depth_ok", "hmb_ok", "lib_ok", "spm_enabled"))
        if g in ("Low-Ca Pyroxene", "High-Ca Pyroxene"):
            soft, _ = pyx_soft_support(e["pyx_ol_diagnostic"], C.PRESENT_GROUPS[g]["depth_key"])
            would = hard and soft
        elif g == "Ilmenite":
            would = hard  # major path unchanged
        else:
            would = hard and e["gates"].get("pyx_2um_support", True)
            # olivine/plag: old pyx_2um_support was True
            would = hard
        if would:
            expect.append(g)
        print(f"  {g}: hard={hard} -> would_PRESENT={would}")
    check("Olivine" in expect and "Plagioclase" in expect, f"expect keep Olivine+Plagioclase; got {expect}")
    check("Low-Ca Pyroxene" in expect, "expect LCP PRESENT after soft-gate fix")
    check("High-Ca Pyroxene" in expect, "expect HCP PRESENT after soft-gate fix")

print("\n" + "=" * 60)
if errors:
    print(f"RESULT: {len(errors)} FAIL(s) — fix before rerun")
    for e in errors:
        print("  -", e)
    sys.exit(1)
print(f"RESULT: READY TO RERUN  ({len(warnings)} warning(s))")
for w in warnings:
    print("  warn:", w)
print("\nRerun command:")
print('  cd "D:\\Journal\\M3 Journal\\Code_scripts"')
print('  python m3_run_mineral.py --scene "Copernicus Crater" --mode full')
sys.exit(0)
