import json
from pathlib import Path

root = Path(r"D:\Journal\M3 Journal")
man = json.loads((root / "models" / "shared" / "pretrain_manifest.json").read_text(encoding="utf-8"))
print("MANIFEST", json.dumps({k: man[k] for k in man if k != "results"}, indent=2))
for s in man["prior_scenes"]:
    p = root / s / "Pipeline_outputs" / "06_mineral_id"
    summary = json.loads((p / "run_summary.json").read_text(encoding="utf-8"))
    proof = json.loads((p / "mineral_proof.json").read_text(encoding="utf-8"))
    mix = json.loads((p / "mixture_summary.json").read_text(encoding="utf-8"))
    rare = json.loads((p / "rare_mineral_candidates.json").read_text(encoding="utf-8"))
    print("\n====", s, "====")
    print("method", summary.get("method_version"), "verdict", summary.get("overall_verdict"))
    print("strong", summary.get("present_strong"), "weak", summary.get("present_weak"))
    print(
        "gates",
        summary.get("gate2", {}).get("pass"),
        summary.get("gate3", {}).get("pass"),
        summary.get("gate4", {}).get("pass"),
    )
    print("mix qa", mix.get("qa_status"), "allowed", mix.get("allowed_families"))
    print("fam", mix.get("mean_family_abundance"))
    print("mean_max", (mix.get("checks") or {}).get("mean_max_abund"))
    for g, e in proof["minerals"].items():
        bd = e.get("band_depths_hot") or {}
        dval = next(iter(bd.values()), None)
        print(
            f"  {g}: level={e.get('present_level')} hot_frac={e.get('hot_frac'):.4f} "
            f"depth={dval} hmb={e.get('hot_minus_bland')} conf={e.get('confidence')} "
            f"pyx={e.get('gates', {}).get('pyx_support_grade')} olflag={e.get('olivine_mixture_flag')}"
        )
    for k, v in (rare.get("candidates") or {}).items():
        n = v.get("n_sam_hits_gated", v.get("n_sam_hits"))
        print(
            f"  rare {k}: {v.get('status')} n={n} terrain={v.get('terrain')} "
            f"spatial_ok={v.get('spatial_ok')} max_cell={v.get('spatial_max_cell')}"
        )
