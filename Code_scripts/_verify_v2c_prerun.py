"""
Offline verification of depth_v2c_polish BEFORE full scene rerun.
Uses last Copernicus artifacts + unit tests. Does not overwrite outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_mineral_constants as C
from m3_mineral_id import pyx_soft_support, pyx_support_grade

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
check(C.METHOD_VERSION == "depth_v2c_polish", f"METHOD_VERSION={C.METHOD_VERSION}")
check(0 < C.SPM_TARGET_HOT_FRAC < C.SPM_MAX_HOT_FRAC,
      f"TARGET={C.SPM_TARGET_HOT_FRAC} < MAX={C.SPM_MAX_HOT_FRAC}")
check(C.UNMIX_ENTROPY_WEIGHT == 0.0, f"UNMIX_ENTROPY_WEIGHT={C.UNMIX_ENTROPY_WEIGHT}")
check(C.UNMIX_SPARSITY_WEIGHT > 0, f"UNMIX_SPARSITY_WEIGHT={C.UNMIX_SPARSITY_WEIGHT}")
check(C.UNMIX_FLAT_MIN_MEAN_MAX > 0, f"FLAT_MIN_MEAN_MAX={C.UNMIX_FLAT_MIN_MEAN_MAX}")
check(C.PYX_D2_MIN_ABS_PATH >= C.PYX_D2_MIN,
      f"ABS_PATH={C.PYX_D2_MIN_ABS_PATH} >= PYX_D2_MIN={C.PYX_D2_MIN}")
check(C.RARE_ILM_DEPTH_MIN >= 0.015, f"RARE_ILM_DEPTH_MIN={C.RARE_ILM_DEPTH_MIN}")
check(C.RARE_ILM_MIN_HOT >= 120, f"RARE_ILM_MIN_HOT={C.RARE_ILM_MIN_HOT}")
check("R750" not in C.CLUSTER_NAME_SPM, "CLUSTER_NAME_SPM excludes R750")

# --- SPM hotspot selectivity replay ---
print("\n[2] SPM hotspot selectivity (replay last gate_info)")
spm_info_p = OUT / "spm_gate_info.json"
if not spm_info_p.is_file():
    errors.append("missing spm_gate_info.json")
    print("  FAIL  no spm_gate_info")
else:
    gi = json.loads(spm_info_p.read_text(encoding="utf-8"))
    lcp = gi.get("LCPINDEX2", {})
    ol = gi.get("OLINDEX3", {})
    check(float(lcp.get("hot_frac", 0)) > C.SPM_TARGET_HOT_FRAC,
          f"last LCP hot_frac={lcp.get('hot_frac'):.3f} > TARGET "
          f"(will be raised to ~top {C.SPM_TARGET_HOT_FRAC:.0%})",
          warn=True)
    check(float(ol.get("hot_frac", 1)) <= C.SPM_TARGET_HOT_FRAC + 0.02,
          f"last Ol hot_frac={ol.get('hot_frac'):.3f} already near target")

    # unit: synthetic broad index (~20% hot) gets raised to ~top TARGET
    vals = np.concatenate([np.zeros(800), np.linspace(0.01, 0.05, 200)])
    thr0 = 0.006
    hot_frac0 = float((vals >= thr0).mean())
    check(hot_frac0 > C.SPM_TARGET_HOT_FRAC, f"synthetic broad hot_frac={hot_frac0:.2f}")
    pct = 100.0 * (1.0 - C.SPM_TARGET_HOT_FRAC)
    thr1 = max(thr0, float(np.percentile(vals, pct)))
    hot_frac1 = float((vals >= thr1).mean())
    check(hot_frac1 <= C.SPM_TARGET_HOT_FRAC + 0.02,
          f"after raise hot_frac={hot_frac1:.3f} thr={thr1:.4f}")

# --- pyx graded soft gate ---
print("\n[3] Pyroxene graded soft-gate")
proof_p = OUT / "mineral_proof.json"
proof = json.loads(proof_p.read_text(encoding="utf-8")) if proof_p.is_file() else {}
for g in ("Low-Ca Pyroxene", "High-Ca Pyroxene"):
    if g not in proof.get("minerals", {}):
        errors.append(f"missing {g} in last proof")
        continue
    e = proof["minerals"][g]
    sep = e["pyx_ol_diagnostic"]
    dkey = C.PRESENT_GROUPS[g]["depth_key"]
    ok, reason = pyx_soft_support(sep, dkey)
    grade = pyx_support_grade(reason)
    print(f"  {g}: ok={ok} grade={grade} reason={reason}")
    print(f"       sep={sep}")
    hard = all(e["gates"][k] for k in ("hot_ok", "depth_ok", "hmb_ok", "lib_ok", "spm_enabled"))
    check(hard, f"{g} hard gates true on last run")
    check(ok, f"{g} still passes soft gate under ABS_PATH={C.PYX_D2_MIN_ABS_PATH}")
    check(grade == "weak", f"{g} graded weak (d1um non-positive path)")

ok_s, r_s = pyx_soft_support(
    {"d1um": 0.05, "d2um_lcp": 0.01, "d2um_hcp": 0.02, "d2_over_d1": 0.2}, "d1.90"
)
check(ok_s and pyx_support_grade(r_s) == "strong", f"positive d1 path strong ({r_s})")

ok_f, r_f = pyx_soft_support(
    {"d1um": -0.01, "d2um_lcp": 0.0065, "d2um_hcp": 0.0065, "d2_over_d1": -1.0}, "d1.90"
)
check(not ok_f, f"d1 nonpos + d2 between PYX_D2_MIN and ABS_PATH rejected ({r_f})")

# --- unmix flat QA ---
print("\n[4] Unmixer flat / collapse QA")
mix = json.loads((OUT / "mixture_summary.json").read_text(encoding="utf-8")) if (OUT / "mixture_summary.json").is_file() else {}
mm = (mix.get("checks") or {}).get("mean_max_abund")
fam = mix.get("mean_family_abundance") or {}
core = [float(fam.get(k, 0)) for k in ("Olivine", "LCP", "HCP", "Plagioclase")]
fmax = max(core) if core else 0.0
if mm is not None:
    would_flat = (mm < C.UNMIX_FLAT_MIN_MEAN_MAX) or (fmax < C.UNMIX_FLAT_MIN_FAMILY_MAX)
    check(would_flat,
          f"last run mean_max={mm:.3f} family_max={fmax:.3f} would be FAILED_FLAT "
          f"(entropy off + sparsity on next run)")
check(C.UNMIX_FREEZE_ENDMEMBERS, "endmembers frozen")
check(C.UNMIX_ENTROPY_WEIGHT == 0.0, "entropy encouragement disabled")

# --- cluster naming ---
print("\n[5] Cluster naming")
check(C.CLUSTER_NAME_SECOND_FRAC > 0, f"SECOND_FRAC={C.CLUSTER_NAME_SECOND_FRAC}")
old_names = json.loads((OUT / "cluster_names.json").read_text(encoding="utf-8")) if (OUT / "cluster_names.json").is_file() else {}
check(all("R750" not in str(v) for v in old_names.values()),
      f"last names already R750-free: {old_names}")

# --- rare ilm tighten ---
print("\n[6] Rare ilmenite tighten")
rare = json.loads((OUT / "rare_mineral_candidates.json").read_text(encoding="utf-8")) if (OUT / "rare_mineral_candidates.json").is_file() else {}
ilm = (rare.get("candidates") or {}).get("Ilmenite", {})
n_old = int(ilm.get("n_sam_hits_gated") or 0)
ex = ilm.get("examples") or []
n_pass_new = 0
for h in ex:
    d = float(h.get("d1.00", 0))
    hmb = float(h.get("hmb", 0))
    if d >= C.RARE_ILM_DEPTH_MIN and hmb >= C.RARE_ILM_HMB_MIN:
        n_pass_new += 1
check(C.RARE_ILM_MIN_HOT > 80, f"MIN_HOT raised to {C.RARE_ILM_MIN_HOT}")
check(True, f"last gated hits={n_old}; example survivors under new depth/hmb={n_pass_new}/{len(ex)}")

# --- olivine mix flag ---
print("\n[7] Olivine mixture flag")
ol_sep = (proof.get("minerals") or {}).get("Olivine", {}).get("pyx_ol_diagnostic") or {}
r = ol_sep.get("d2_over_d1")
if r is not None and np.isfinite(float(r)):
    flag = "pyx_mixed" if float(r) >= C.OL_MIX_D2_OVER_D1 else "olivine_rich"
    check(True, f"Olivine d2/d1={float(r):.3f} -> flag={flag}")
else:
    check(False, "Olivine sep missing d2_over_d1")

# --- expected PRESENT replay ---
print("\n[8] Expected PRESENT replay (logic on last proof; hot frac may tighten)")
expect = []
for g, e in (proof.get("minerals") or {}).items():
    hard = all(e["gates"][k] for k in ("hot_ok", "depth_ok", "hmb_ok", "lib_ok", "spm_enabled"))
    if g in ("Low-Ca Pyroxene", "High-Ca Pyroxene"):
        soft, _ = pyx_soft_support(e["pyx_ol_diagnostic"], C.PRESENT_GROUPS[g]["depth_key"])
        would = hard and soft
    elif g == "Ilmenite":
        would = False  # still SPM-disabled / hmb fail on last run
    else:
        would = hard
    if would:
        expect.append(g)
    print(f"  {g}: would_PRESENT={would}")
check("Olivine" in expect and "Plagioclase" in expect, f"keep Ol+Plag: {expect}")
check("Low-Ca Pyroxene" in expect, "LCP still PRESENT under weak abs path", warn=True)
check("High-Ca Pyroxene" in expect, "HCP still PRESENT under weak abs path")
check(True, "NOTE: LCP hotspot will shrink (~10%); depth/hmb must still clear on rerun", warn=True)

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
