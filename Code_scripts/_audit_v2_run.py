import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_mineral_constants as C
import m3_spm as spm

out = Path(r"D:\Journal\M3 Journal\Copernicus Crater\Pipeline_outputs\06_mineral_id")
spm_map = np.load(out / "spm_map.npy", mmap_mode="r")
active = np.load(out / "spm_active_mask.npy")
cube_cr = np.load(out / "cube_cr.npy", mmap_mode="r")
den = np.load(
    r"D:\Journal\M3 Journal\Copernicus Crater\Pipeline_outputs\04_denoised"
    r"\M3G20090207T083836_V01_RFL_denoised.npy",
    mmap_mode="r",
)
waves = C.M3_WAVELENGTHS_85_UM
print("shapes", den.shape, cube_cr.shape, spm_map.shape)
print("active frac", float(active.mean()))
print("CR finite frac", float(np.isfinite(cube_cr).mean()))

rng = np.random.default_rng(0)
ys, xs = np.where(np.isfinite(cube_cr[:, :, 40]))
ii = rng.choice(len(ys), 2000, replace=False)
specs = cube_cr[ys[ii], xs[ii], :]
mean_s = np.nanmean(specs, axis=0)
print("CR mean min/max", float(np.nanmin(mean_s)), float(np.nanmax(mean_s)))
for wl in (1.05, 1.25, 1.90, 2.25):
    j = spm.nearest_band(waves, wl)
    print(f"  CR@{wl}", float(mean_s[j]))

print("\nSPM percentiles:")
for i, name in enumerate(C.SPM_NAMES):
    layer = np.asarray(spm_map[:, :, i])
    fin = layer[np.isfinite(layer)]
    if fin.size == 0:
        print(name, "ALL NAN")
        continue
    print(
        f"{name:12s} p50={np.percentile(fin,50):.4f} p90={np.percentile(fin,90):.4f} "
        f"p99={np.percentile(fin,99):.4f} pos={(fin>0).mean():.3f} max={fin.max():.4f}"
    )

proof = json.loads((out / "mineral_proof.json").read_text(encoding="utf-8"))
print("\nPROOF detail:")
for g, e in proof["minerals"].items():
    print(
        f"  {g}: present={e['present']} n_hot={e['n_hot_pixels']} "
        f"d_hot={e['band_depths_hot']} hmb={e['hot_minus_bland']} sam={e['best_sam']}"
    )
    print(f"    gates={e['gates']}")
    print(f"    sep={e.get('pyx_ol_diagnostic')}")

# Overlap of olivine and plag hotspots
ol = np.asarray(spm_map[:, :, C.SPM_NAMES.index("OLINDEX3")])
pl = np.asarray(spm_map[:, :, C.SPM_NAMES.index("BD1250")])
lcp = np.asarray(spm_map[:, :, C.SPM_NAMES.index("LCPINDEX2")])
hcp = np.asarray(spm_map[:, :, C.SPM_NAMES.index("HCPINDEX2")])
ol_hot = ol >= 0.026006806641817093
pl_hot = pl >= 0.005
lcp_hot = lcp >= 0.006
hcp_hot = hcp >= 0.018459957093000412
print("\nHotspot overlaps:")
print("  ol&pl", int((ol_hot & pl_hot).sum()), "ol", int(ol_hot.sum()), "pl", int(pl_hot.sum()))
print("  ol&lcp", int((ol_hot & lcp_hot).sum()), "lcp", int(lcp_hot.sum()))
print("  ol&hcp", int((ol_hot & hcp_hot).sum()), "hcp", int(hcp_hot.sum()))
print("  lcp&hcp", int((lcp_hot & hcp_hot).sum()))

mix = json.loads((out / "mixture_summary.json").read_text(encoding="utf-8"))
print("\nUnmixer collapse: HCP mean abund", mix["mean_family_abundance"]["HCP"])
print("mean_max_abund", mix["checks"]["mean_max_abund"])

lat = np.load(out / "latent_vectors.npy", mmap_mode="r")
norms = np.linalg.norm(lat[::100], axis=1)
print("latent", lat.shape, "norm med", float(np.median(norms)))
print("clusters", json.loads((out / "cluster_names.json").read_text(encoding="utf-8")))
