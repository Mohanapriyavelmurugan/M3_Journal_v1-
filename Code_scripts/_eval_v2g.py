import json
from pathlib import Path

root = Path(r"D:\Journal\M3 Journal")
man = json.loads((root / "models" / "shared" / "pretrain_manifest.json").read_text(encoding="utf-8"))
print("MANIFEST method=", man.get("method_version"), "refresh=", man.get("claims_refresh"))
print(json.dumps(man.get("results"), indent=2))

for s in man.get("prior_scenes") or [
    "Mare Tranquillitatis",
    "Lunar Highlands South",
    "Copernicus Crater",
]:
    p = root / s / "Pipeline_outputs" / "06_mineral_id"
    print("\n====", s, "====")
    summary = json.loads((p / "run_summary.json").read_text(encoding="utf-8"))
    proof = json.loads((p / "mineral_proof.json").read_text(encoding="utf-8"))
    mix = json.loads((p / "mixture_summary.json").read_text(encoding="utf-8"))
    rare = json.loads((p / "rare_mineral_candidates.json").read_text(encoding="utf-8"))
    clusters = json.loads((p / "cluster_names.json").read_text(encoding="utf-8")) if (p / "cluster_names.json").is_file() else {}
    cqa = json.loads((p / "clustering_qa.json").read_text(encoding="utf-8")) if (p / "clustering_qa.json").is_file() else {}
    print("method", summary.get("method_version"), "verdict", summary.get("overall_verdict"))
    print("strong", summary.get("present_strong"), "weak", summary.get("present_weak"))
    print(
        "gates",
        summary.get("gate2", {}).get("pass"),
        summary.get("gate3", {}).get("pass"),
        summary.get("gate4", {}).get("pass"),
    )
    print(
        "clusters n=",
        summary.get("n_clusters"),
        "status=",
        summary.get("clustering_status"),
        "names=",
        clusters,
    )
    if cqa:
        print(
            "  qa mcs=",
            cqa.get("mcs"),
            "method=",
            cqa.get("method"),
            "noise=",
            cqa.get("noise"),
            "dominance=",
            cqa.get("dominance"),
            "channels=",
            cqa.get("name_channels"),
        )
    print("mix", mix.get("qa_status"), "allowed", mix.get("allowed_families"))
    print("fam", mix.get("mean_family_abundance"))
    # encode size
    import numpy as np

    lat = np.load(p / "latent_vectors.npy", mmap_mode="r")
    print("encoded", lat.shape)
    for g, e in proof["minerals"].items():
        print(
            f"  {g}: {e.get('present_level')} hot_frac={e.get('hot_frac'):.4f} "
            f"conf={e.get('confidence')} pyx={e.get('gates', {}).get('pyx_support_grade')}"
        )
    ilm = (rare.get("candidates") or {}).get("Ilmenite", {})
    print(
        "rare ilm",
        ilm.get("status"),
        "n=",
        ilm.get("n_sam_hits_gated"),
        "terrain=",
        ilm.get("terrain"),
    )
