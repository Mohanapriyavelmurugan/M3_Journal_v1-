import json
from pathlib import Path
p = Path(r"D:\Journal\M3 Journal\Copernicus Crater\Pipeline_outputs\06_mineral_id")
proof = json.loads((p / "mineral_proof.json").read_text())
mix = json.loads((p / "mixture_summary.json").read_text())
spm = json.loads((p / "spm_gate_info.json").read_text())
print("SPM:", json.dumps(spm, indent=2))
print()
# scene size estimate from n_hot relative - use densest
import numpy as np
mask = np.load(p / "spm_active_mask.npy")
print("spm_active_mask", mask.shape, "dtype", mask.dtype, "mean", float(np.asarray(mask).mean()))
H, W = mask.shape[:2] if mask.ndim >= 2 else (None, None)
n_pix = int(np.prod(mask.shape[:2])) if mask.ndim >= 2 else None
print("n_pix", n_pix)
for g, e in proof["minerals"].items():
    hf = e["n_hot_pixels"] / n_pix if n_pix else None
    print(f"{g}: present={e['present']} hot={e['n_hot_pixels']} hot_frac={hf:.4f} thr={e['thr_adaptive']:.6g} hmb={e['hot_minus_bland']:.5f} depth={list(e['band_depths_hot'].values())[0]:.5f} sam={e['best_sam']:.4f} fam={e['best_family']}")
print("unmix", mix["mean_family_abundance"])
print("mean_max", mix["checks"]["mean_max_abund"], "qa", mix["qa_status"])
print("clusters", json.loads((p / "cluster_names.json").read_text()))
