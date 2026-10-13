"""Deep audit of pretrain prior-scene mineral ID outputs."""
from __future__ import annotations

import json
from pathlib import Path

root = Path(r"D:\Journal\M3 Journal")
scenes = ["Mare Tranquillitatis", "Lunar Highlands South", "Copernicus Crater"]
shared = root / "models" / "shared"

print("=" * 72)
print("PRETRAIN MANIFEST")
print("=" * 72)
man = json.loads((shared / "pretrain_manifest.json").read_text(encoding="utf-8"))
print(json.dumps(man, indent=2))
enc = shared / "encoder_B85.pth"
print(f"shared encoder exists={enc.is_file()} size={enc.stat().st_size if enc.is_file() else 0}")

for s in scenes:
    p = root / s / "Pipeline_outputs" / "06_mineral_id"
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)
    mv = (p / "method_version.txt").read_text(encoding="utf-8").strip()
    summary = json.loads((p / "run_summary.json").read_text(encoding="utf-8"))
    proof = json.loads((p / "mineral_proof.json").read_text(encoding="utf-8"))
    mix = json.loads((p / "mixture_summary.json").read_text(encoding="utf-8"))
    rare = json.loads((p / "rare_mineral_candidates.json").read_text(encoding="utf-8"))
    spm = json.loads((p / "spm_gate_info.json").read_text(encoding="utf-8"))
    clusters = json.loads((p / "cluster_names.json").read_text(encoding="utf-8"))

    print(f"method={mv}  summary_method={summary.get('method_version')}")
    print(f"PRESENT={summary.get('present')}  verdict={summary.get('overall_verdict')}")
    print(f"gates: G0={summary.get('gate0',{}).get('pass')} G1={summary.get('gate1',{}).get('pass')} "
          f"G2={summary.get('gate2',{}).get('pass')} G3={summary.get('gate3',{}).get('pass')} "
          f"G4={summary.get('gate4',{}).get('pass')}")
    print(f"G2 enabled={summary.get('gate2',{}).get('enabled_indices')} "
          f"active_frac={summary.get('gate2',{}).get('active_frac')}")
    print(f"G3 n_encoded={summary.get('gate3',{}).get('n_encoded')}")
    print(f"clusters={clusters}  n_clusters={proof.get('n_clusters')}")
    print(f"mixture_qa={mix.get('qa_status')} citable={mix.get('citable')}")
    print(f"  fam={mix.get('mean_family_abundance')}")
    print(f"  checks mean_max={mix.get('checks',{}).get('mean_max_abund')} "
          f"fam_max={mix.get('checks',{}).get('family_max_abund')}")
    print("SPM enabled detail:")
    for name, g in spm.items():
        if g.get("enabled"):
            print(f"  {name}: thr={g.get('thr'):.5f} hot_frac={g.get('hot_frac'):.4f} "
                  f"n_hot={g.get('n_hot')} contrast_ok={g.get('contrast_ok')}")
        else:
            print(f"  {name}: DISABLED ({g.get('reason')})")
    print("PRESENT proof detail:")
    for gname, e in proof.get("minerals", {}).items():
        gates = e.get("gates", {})
        print(
            f"  {gname}: present={e.get('present')} hot={e.get('n_hot_pixels')} "
            f"depth={e.get('band_depths_hot')} hmb={e.get('hot_minus_bland')} "
            f"sam={e.get('best_sam')} fam={e.get('best_family')} "
            f"ol_flag={e.get('olivine_mixture_flag')} "
            f"pyx_grade={gates.get('pyx_support_grade')} reason={gates.get('pyx_support_reason')} "
            f"hot_ok={gates.get('hot_ok')} depth_ok={gates.get('depth_ok')} "
            f"hmb_ok={gates.get('hmb_ok')} lib_ok={gates.get('lib_ok')} "
            f"spm_en={gates.get('spm_enabled')}"
        )
        sep = e.get("pyx_ol_diagnostic") or {}
        if e.get("present") or gname in ("Low-Ca Pyroxene", "High-Ca Pyroxene", "Olivine"):
            print(f"    sep={sep}")
    print("Rare:")
    for k, v in (rare.get("candidates") or {}).items():
        print(f"  {k}: {v.get('status')} n={v.get('n_sam_hits_gated', v.get('n_sam_hits'))}")
        for ex in (v.get("examples") or [])[:2]:
            print(f"    ex={ex}")
