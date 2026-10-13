"""Deep clustering failure diagnosis across prior scenes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import hdbscan
except ImportError:
    hdbscan = None

ROOT = Path(r"D:\Journal\M3 Journal")
SCENES = ["Mare Tranquillitatis", "Lunar Highlands South", "Copernicus Crater"]


def run_hdbscan(X, mcs, min_samples, method="eom", metric="euclidean"):
    cl = hdbscan.HDBSCAN(
        min_cluster_size=int(mcs),
        min_samples=int(min_samples),
        metric=metric,
        cluster_selection_method=method,
    )
    labels = cl.fit_predict(X)
    n = len(set(labels.tolist()) - {-1})
    noise = float((labels < 0).mean())
    sizes = [int((labels == c).sum()) for c in sorted(set(labels.tolist()) - {-1})]
    return n, noise, sizes, getattr(cl, "relative_validity_", None)


def main():
    print("hdbscan available:", hdbscan is not None)
    for s in SCENES:
        p = ROOT / s / "Pipeline_outputs" / "06_mineral_id"
        print("\n" + "=" * 72)
        print(s)
        print("=" * 72)
        proof = json.loads((p / "mineral_proof.json").read_text(encoding="utf-8"))
        gate = json.loads((p / "spm_gate_info.json").read_text(encoding="utf-8"))
        active = np.load(p / "spm_active_mask.npy")
        lat = np.load(p / "latent_vectors.npy")
        coords = np.load(p / "latent_coords.npy")
        n_valid = int(np.load(p / "spm_active_mask.npy").size)  # approx later
        print(f"proof n_clusters={proof.get('n_clusters')} verdict={proof.get('overall_verdict')}")
        print(f"latent={lat.shape} active_mask_true={int(active.sum())} active_frac={float(active.mean()):.4f}")
        is_act = np.array([bool(active[y, x]) for y, x in coords], dtype=bool)
        print(f"encoded_active={int(is_act.sum())}/{len(is_act)}")
        print(
            "ENCODE CAP BUG? active_encoded > 400000:",
            int(is_act.sum()) > 400_000,
            "(encode keeps ALL actives if active>MAX_ENC)",
        )

        Xall = lat[is_act]
        rng = np.random.default_rng(0)
        MAX_FIT = 15000
        if len(Xall) > MAX_FIT:
            sel = rng.choice(len(Xall), MAX_FIT, replace=False)
            Xfit = Xall[sel]
        else:
            Xfit = Xall
        print(f"Xfit={Xfit.shape}")

        # current pipeline params
        mcs_cur = max(50, int(0.002 * len(Xfit)))
        ms_cur = max(5, mcs_cur // 10)
        print(f"current params mcs={mcs_cur} min_samples={ms_cur} method=eom metric=euclidean")

        if hdbscan is None:
            continue

        configs = [
            ("CURRENT", mcs_cur, ms_cur, "eom", "euclidean"),
            ("mcs30", 30, 5, "eom", "euclidean"),
            ("mcs100", 100, 10, "eom", "euclidean"),
            ("leaf_mcs50", 50, 5, "leaf", "euclidean"),
            ("cosine_mcs50", 50, 5, "eom", "euclidean"),  # still euclid; note L2-normed
            ("mcs20_leaf", 20, 5, "leaf", "euclidean"),
        ]
        # Also try cosine via normalized already equates to chordal euclid roughly

        for name, mcs, ms, method, metric in configs:
            n, noise, sizes, rel = run_hdbscan(Xfit, mcs, ms, method, metric)
            print(f"  {name:12s} clusters={n} noise={noise:.2f} sizes={sizes[:8]} rel_validity={rel}")

        # diversity of fit set
        # nearest-neighbor distances
        from sklearn.metrics import pairwise_distances
        sub = Xfit[rng.choice(len(Xfit), min(2000, len(Xfit)), replace=False)]
        D = pairwise_distances(sub, metric="euclidean")
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        print(
            f"NN dist sample: mean={nn.mean():.4f} p10={np.percentile(nn,10):.4f} "
            f"p50={np.percentile(nn,50):.4f} p90={np.percentile(nn,90):.4f}"
        )
        # SPM separation of active pixels (are mineral hotspots mixed in latent?)
        spm = np.load(p / "spm_map.npy", mmap_mode="r")
        # sample active coords from encoded
        act_coords = coords[is_act]
        take = rng.choice(len(act_coords), min(5000, len(act_coords)), replace=False)
        pts = act_coords[take]
        for sname in ("OLINDEX3", "LCPINDEX2", "HCPINDEX2", "BD1250"):
            i = ["OLINDEX3", "LCPINDEX2", "HCPINDEX2", "BD1250", "BD1000", "R750", "BD1500", "BD1900"].index(sname)
            thr = float(gate.get(sname, {}).get("thr", 0))
            vals = np.array([spm[y, x, i] for y, x in pts], dtype=np.float64)
            hot = np.isfinite(vals) & (vals >= thr)
            print(f"  SPM {sname}: thr={thr:.5f} hot_frac_in_sample={hot.mean():.3f} med={np.nanmedian(vals):.5f}")

        # naming conflict: OLINDEX enabled while olivine ABSENT?
        ol_present = proof["minerals"]["Olivine"]["present"]
        ol_en = gate.get("OLINDEX3", {}).get("enabled")
        print(f"  Olivine PRESENT={ol_present} OLINDEX3 enabled={ol_en} (naming can still tag OLINDEX3)")


if __name__ == "__main__":
    main()
