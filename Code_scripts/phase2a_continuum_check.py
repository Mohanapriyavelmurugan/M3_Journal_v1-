"""
phase2a_continuum_check.py -- Phase 2A continuum-removal diagnostics
====================================================================

Checks whether denoised M3 cubes are ready for mineral identification by
measuring continuum-removed absorption preservation in mineral-index ROIs.

This script does not identify minerals. It answers a narrower question:
did Phase 1 preserve absorption geometry well enough to proceed, or is
retraining justified before Phase 2 mineral mapping?
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step0_load as s0

JOURNAL_ROOT = Path(r"D:\M3 Journal")

EXCLUDED_LOCATIONS = {
    "aristarchus plateau",
    "highlands near tycho",
    "south pole region",
}

FEATURES = [
    {"name": "BD2300", "mineral": "Phyllosilicates", "center": 2.295, "left": 2.12, "right": 2.53},
    {"name": "BD1900", "mineral": "Hydrated minerals", "center": 1.930, "left": 1.85, "right": 2.07},
    {"name": "BD2210", "mineral": "Al-OH clays", "center": 2.210, "left": 2.165, "right": 2.25},
    {"name": "SINDEX2", "mineral": "Sulfates", "center": 2.100, "left": 1.815, "right": 2.20},
    {"name": "BD1500", "mineral": "Water ice", "center": 1.500, "left": 1.43, "right": 1.815},
    {"name": "HCPINDEX2", "mineral": "HCP pyroxene", "center": 2.140, "left": 1.82, "right": 2.46},
]

ROI_MIN_BD = 0.020
ROI_TOP_PERCENTILE = 90.0
MIN_ROI_PIXELS = 100
MAX_SAMPLE_PIXELS = 8000
BD_CHANGE_PASS_PCT = 20.0
BD_CHANGE_REVIEW_PCT = 30.0
WINDOW_HALF_WIDTH_UM = 0.060


def nearest_band(waves, target):
    return int(np.argmin(np.abs(waves - target)))


def feature_bands(waves, feature):
    if feature["center"] < waves[0] or feature["center"] > waves[-1]:
        return None
    bc = nearest_band(waves, feature["center"])
    bl = nearest_band(waves, feature["left"])
    br = nearest_band(waves, feature["right"])
    lo = min(bl, bc, br)
    hi = max(bl, bc, br)
    if hi - lo < 2:
        return None
    return bl, bc, br, lo, hi


def continuum_removed_spectra(spec, waves, left_idx, right_idx, lo, hi):
    w = waves[lo:hi + 1]
    y = spec[:, lo:hi + 1].astype(np.float64)
    yl = spec[:, left_idx].astype(np.float64)
    yr = spec[:, right_idx].astype(np.float64)
    wl = float(waves[left_idx])
    wr = float(waves[right_idx])
    frac = (w[None, :] - wl) / max(wr - wl, 1e-8)
    cont = yl[:, None] + frac * (yr - yl)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = y / np.where(cont > 1e-8, cont, np.nan)
    return cr


def band_depth_map(cube, waves, feature):
    bands = feature_bands(waves, feature)
    if bands is None:
        return None
    bl, bc, br, _, _ = bands
    wl = float(waves[bl])
    wc = float(waves[bc])
    wr = float(waves[br])
    frac = (wc - wl) / max(wr - wl, 1e-8)
    left = cube[:, :, bl].astype(np.float64)
    right = cube[:, :, br].astype(np.float64)
    center = cube[:, :, bc].astype(np.float64)
    cont = left + frac * (right - left)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - center / np.where(cont > 1e-8, cont, np.nan)


def spectral_angle(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return np.nan
    aa = a[finite]
    bb = b[finite]
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(np.arccos(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0)))


def local_roughness(spec, waves, center):
    keep = np.abs(waves - center) <= WINDOW_HALF_WIDTH_UM
    if keep.sum() < 5:
        return np.nan
    y = spec[:, keep].astype(np.float64)
    med = np.nanmedian(y, axis=1, keepdims=True)
    high_freq = y - med
    return np.nanmedian(np.abs(high_freq - np.nanmedian(high_freq, axis=1, keepdims=True)), axis=1)


def infer_scene_dir(report_path):
    location = report_path.parents[2]
    fname = report_path.name.replace("_validation_report.json", "")
    candidate = location / fname
    if candidate.exists():
        return candidate
    matches = sorted(location.glob(fname + "*"))
    matches = [m for m in matches if m.is_dir()]
    if matches:
        return matches[0]
    return candidate


def discover_reports(root, include_excluded=False):
    reports = sorted(root.glob("*/*/05_validation/*_validation_report.json"))
    selected = []
    for rpt in reports:
        location = rpt.parents[2].name.lower()
        if not include_excluded and location in EXCLUDED_LOCATIONS:
            continue
        selected.append(rpt)
    return selected


def evaluate_feature(raw, den, waves, valid_mask, feature, rng):
    bd_den = band_depth_map(den, waves, feature)
    bd_raw = band_depth_map(raw, waves, feature)
    bands = feature_bands(waves, feature)
    if bd_den is None or bd_raw is None or bands is None:
        return {"status": "SKIP", "reason": "feature outside wavelength range"}

    # Select candidate ROIs from the raw cube. Selecting from the denoised cube
    # can promote denoising-created features to "candidate minerals" and then
    # falsely make validation look better or worse.
    plausible = bd_raw[np.isfinite(bd_raw) & (bd_raw >= ROI_MIN_BD) & (bd_raw <= 0.60)]
    if plausible.size < MIN_ROI_PIXELS:
        return {
            "status": "N/A",
            "reason": "too few continuum-removed candidate pixels",
            "candidate_pixels": int(plausible.size),
        }

    roi_thr = max(ROI_MIN_BD, float(np.percentile(plausible, ROI_TOP_PERCENTILE)))
    roi = valid_mask & np.isfinite(bd_raw) & (bd_raw >= roi_thr) & (bd_raw <= 0.60)
    rows, cols = np.where(roi)
    if len(rows) < MIN_ROI_PIXELS:
        return {
            "status": "N/A",
            "reason": "too few high-confidence ROI pixels",
            "candidate_pixels": int(len(rows)),
            "roi_threshold": float(roi_thr),
        }

    take = min(MAX_SAMPLE_PIXELS, len(rows))
    idx = rng.choice(len(rows), size=take, replace=False)
    rows = rows[idx]
    cols = cols[idx]

    bl, bc, br, lo, hi = bands
    raw_spec = raw[rows, cols, :]
    den_spec = den[rows, cols, :]
    cr_raw = continuum_removed_spectra(raw_spec, waves, bl, br, lo, hi)
    cr_den = continuum_removed_spectra(den_spec, waves, bl, br, lo, hi)
    center_local = bc - lo
    bd_raw_roi = 1.0 - cr_raw[:, center_local]
    bd_den_roi = 1.0 - cr_den[:, center_local]
    good = np.isfinite(bd_raw_roi) & np.isfinite(bd_den_roi)

    if good.sum() < MIN_ROI_PIXELS:
        return {
            "status": "N/A",
            "reason": "too few finite continuum-removed ROI spectra",
            "candidate_pixels": int(good.sum()),
            "roi_threshold": float(roi_thr),
        }

    bd_raw_med = float(np.nanmedian(bd_raw_roi[good]))
    bd_den_med = float(np.nanmedian(bd_den_roi[good]))
    bd_change_pct = float((bd_den_med - bd_raw_med) / max(abs(bd_raw_med), 1e-8) * 100.0)

    sams = []
    for a, b in zip(cr_raw[good], cr_den[good]):
        sams.append(spectral_angle(a, b))
    sams = np.array(sams, dtype=np.float64)
    cr_sam_deg = float(np.degrees(np.nanmedian(sams)))

    rough_raw = local_roughness(raw_spec[good], waves, feature["center"])
    rough_den = local_roughness(den_spec[good], waves, feature["center"])
    rough_good = np.isfinite(rough_raw) & np.isfinite(rough_den) & (rough_raw > 1e-10)
    rough_reduction_pct = None
    if rough_good.sum() >= 50:
        rough_reduction_pct = float(
            (np.nanmedian(rough_raw[rough_good]) - np.nanmedian(rough_den[rough_good])) /
            max(np.nanmedian(rough_raw[rough_good]), 1e-10) * 100.0
        )

    if bd_raw_med < ROI_MIN_BD:
        return {
            "status": "N/A",
            "mineral": feature["mineral"],
            "reason": "candidate ROI not confirmed in raw continuum-removed spectra",
            "roi_pixels": int(len(rows)),
            "roi_threshold": float(roi_thr),
            "bd_raw_median": bd_raw_med,
            "bd_denoised_median": bd_den_med,
            "bd_change_pct": bd_change_pct,
            "continuum_sam_deg_median": cr_sam_deg,
            "local_roughness_reduction_pct": rough_reduction_pct,
        }

    if bd_raw_med > 0.60 or bd_den_med > 0.60:
        status = "REVIEW"
    elif abs(bd_change_pct) <= BD_CHANGE_PASS_PCT and cr_sam_deg <= 2.0:
        status = "PASS"
    elif abs(bd_change_pct) <= BD_CHANGE_REVIEW_PCT and cr_sam_deg <= 4.0:
        status = "REVIEW"
    else:
        status = "FAIL"

    return {
        "status": status,
        "mineral": feature["mineral"],
        "roi_pixels": int(len(rows)),
        "roi_threshold": float(roi_thr),
        "bd_raw_median": bd_raw_med,
        "bd_denoised_median": bd_den_med,
        "bd_change_pct": bd_change_pct,
        "continuum_sam_deg_median": cr_sam_deg,
        "local_roughness_reduction_pct": rough_reduction_pct,
    }


def evaluate_scene(report_path, rng):
    with open(report_path, "r") as f:
        report = json.load(f)
    scene_dir = infer_scene_dir(report_path)
    den_path = report_path.parents[1] / "04_denoised" / f"{report['fname']}_denoised.npy"
    if not den_path.exists():
        raise FileNotFoundError(f"Missing denoised cube: {den_path}")

    raw, waves, fname, _, _ = s0.load_scene(str(scene_dir))
    den = np.load(str(den_path), mmap_mode="r")
    valid_mask = np.sum(np.isfinite(raw) & np.isfinite(den), axis=2) >= s0.MIN_VALID_BANDS

    features = {}
    for feature in FEATURES:
        features[feature["name"]] = evaluate_feature(raw, den, waves, valid_mask, feature, rng)

    detected = [v for v in features.values() if v.get("status") in ("PASS", "REVIEW", "FAIL")]
    fails = [k for k, v in features.items() if v.get("status") == "FAIL"]
    reviews = [k for k, v in features.items() if v.get("status") == "REVIEW"]
    rough_vals = [
        v.get("local_roughness_reduction_pct") for v in detected
        if isinstance(v.get("local_roughness_reduction_pct"), (int, float))
    ]
    median_rough_reduction = float(np.nanmedian(rough_vals)) if rough_vals else None

    retrain_reasons = []
    if fails:
        retrain_reasons.append("continuum feature preservation failures: " + ",".join(fails))
    if median_rough_reduction is not None and median_rough_reduction < 0:
        retrain_reasons.append("continuum-window roughness increased")
    if report.get("metric2_sam", {}).get("mean_sam_rad", 0.0) >= 0.05:
        retrain_reasons.append("whole-spectrum SAM exceeds pass threshold")

    if fails or len(reviews) >= 3:
        readiness = "BORDERLINE"
    elif detected:
        readiness = "READY_FOR_PHASE2"
    else:
        readiness = "READY_FOR_PHASE2_NO_CONFIRMED_FEATURES"

    return {
        "location": report_path.parents[2].name,
        "fname": fname,
        "phase1_verdict": report.get("overall_verdict"),
        "denoised_cube": str(den_path),
        "phase2a_readiness": readiness,
        "retraining_recommended": bool(retrain_reasons),
        "retraining_reasons": retrain_reasons,
        "median_local_roughness_reduction_pct": median_rough_reduction,
        "features": features,
    }


def write_outputs(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase2a_continuum_readiness.json"
    csv_path = out_dir / "phase2a_continuum_readiness.csv"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for scene in results["scenes"]:
        for name, item in scene["features"].items():
            rows.append({
                "location": scene["location"],
                "fname": scene["fname"],
                "phase1_verdict": scene["phase1_verdict"],
                "phase2a_readiness": scene["phase2a_readiness"],
                "retraining_recommended": scene["retraining_recommended"],
                "feature": name,
                "mineral": item.get("mineral"),
                "status": item.get("status"),
                "roi_pixels": item.get("roi_pixels") or item.get("candidate_pixels"),
                "bd_change_pct": item.get("bd_change_pct"),
                "continuum_sam_deg_median": item.get("continuum_sam_deg_median"),
                "local_roughness_reduction_pct": item.get("local_roughness_reduction_pct"),
                "reason": item.get("reason"),
            })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Phase 2A continuum-removal readiness checks")
    parser.add_argument("--root", type=str, default=str(JOURNAL_ROOT))
    parser.add_argument("--report", action="append", default=[],
                        help="Specific validation report path; can be repeated")
    parser.add_argument("--include-excluded", action="store_true",
                        help="Include Aristarchus, Highlands near Tycho, and South Pole")
    parser.add_argument("--out-dir", type=str, default=str(JOURNAL_ROOT / "Phase2_outputs"))
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    root = Path(args.root)
    reports = [Path(p) for p in args.report] if args.report else discover_reports(root, args.include_excluded)
    if not reports:
        raise SystemExit("No validation reports found.")

    scenes = []
    for report in reports:
        print(f"\n[Phase2A] {report.parents[2].name} / {report.name}")
        try:
            scenes.append(evaluate_scene(report, rng))
        except Exception as exc:
            scenes.append({
                "location": report.parents[2].name,
                "fname": report.name.replace("_validation_report.json", ""),
                "phase2a_readiness": "ERROR",
                "retraining_recommended": True,
                "retraining_reasons": [repr(exc)],
                "features": {},
            })
            print(f"  ERROR: {exc!r}")

    n_retrain = sum(1 for s in scenes if s.get("retraining_recommended"))
    results = {
        "method": "continuum-removed ROI preservation and local roughness diagnostics",
        "excluded_locations": [] if args.include_excluded else sorted(EXCLUDED_LOCATIONS),
        "scene_count": len(scenes),
        "retraining_recommended_scene_count": n_retrain,
        "global_retraining_recommended": n_retrain >= max(2, int(np.ceil(0.30 * len(scenes)))),
        "scenes": scenes,
    }
    json_path, csv_path = write_outputs(results, Path(args.out_dir))

    print("\n" + "=" * 72)
    print("PHASE 2A CONTINUUM READINESS SUMMARY")
    print("=" * 72)
    for scene in scenes:
        retrain = "YES" if scene.get("retraining_recommended") else "NO"
        print(f"{scene['location']:<24} {scene['phase2a_readiness']:<34} retrain={retrain}")
    print("-" * 72)
    print(f"Global retraining recommended: {results['global_retraining_recommended']}")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
