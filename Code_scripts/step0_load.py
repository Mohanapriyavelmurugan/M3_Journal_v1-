"""
step0_load.py  --  M3 Phase 1, Step 0: Cube Loader and Shared Utilities
========================================================================
Loads one M3 Level-2 reflectance scene from a scene directory that contains:
    <scene>_RFL.IMG          binary hyperspectral cube (float32)
    <scene>_L2.LBL           PDS3 label  (dimensions, storage order)
    <scene>_RFL.HDR          ENVI header (wavelength table)

Outputs  (all kept in memory for downstream steps):
    cube       np.ndarray float32 (H, W, B)
    waves      np.ndarray float64 (B,)  wavelengths in um
    fname      str   scene base name  e.g. "M3G20090203T160452_V01_RFL"
    meta       dict  label metadata
    DIRS       dict  paths for the 5 output subdirectories

Usage
-----
    import step0_load as s0
    cube, waves, fname, meta, DIRS = s0.load_scene(scene_dir)

Or run standalone to test one scene:
    python step0_load.py "D:/M3 Journal/Mare Tranquillitatis/M3G20090203T160452_V01_RFL"
"""

import os, sys, json, re, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Hard-coded M3 constants  (never overridden)
# ---------------------------------------------------------------------------
M3_IF_MIN = -0.10
M3_IF_MAX  =  1.50
THERMAL_ONSET_UM = 2.50
MIN_VALID_BANDS  = 20
DEAD_VAR_THRESHOLD   = 1e-10
DEAD_NAN_THRESHOLD   = 0.95

# Canonical 85-band global-mode wavelengths (nm) -- fallback only
M3_WAVELENGTHS_85_NM = np.array([
    460.99, 500.92, 540.84, 580.76, 620.69, 660.61, 700.54, 730.48,
    750.44, 770.40, 790.37, 810.33, 830.29, 850.25, 870.21, 890.17,
    910.14, 930.10, 950.06, 970.02, 989.98,1009.95,1029.91,1049.87,
   1069.83,1089.79,1109.76,1129.72,1149.68,1169.64,1189.60,1209.57,
   1229.53,1249.49,1269.45,1289.41,1309.38,1329.34,1349.30,1369.26,
   1389.22,1409.19,1429.15,1449.11,1469.07,1489.03,1508.99,1528.96,
   1548.92,1578.86,1618.79,1658.71,1698.63,1738.56,1778.48,1818.40,
   1858.33,1898.25,1938.18,1978.10,2018.02,2057.95,2097.87,2137.80,
   2177.72,2217.64,2257.57,2297.49,2337.42,2377.34,2417.26,2457.19,
   2497.11,2537.03,2576.96,2616.88,2656.81,2696.73,2736.65,2776.58,
   2816.50,2856.43,2896.35,2936.27,2976.20,
], dtype=np.float64)

CANDIDATE_B = [85, 260, 84, 86]
CANDIDATE_W = [304, 320, 608, 640, 512, 256]


# ---------------------------------------------------------------------------
# 0.10  Utility functions (imported by all downstream steps)
# ---------------------------------------------------------------------------

def find_nearest_band(waves_um, target_um):
    """Return integer index of band closest to target_um (in um)."""
    return int(np.argmin(np.abs(waves_um - target_um)))


def get_bad_band_mask(waves_um):
    """Boolean array True for bands beyond 2.80 um (strong thermal).
    The 2.50-2.80 um range is NOT masked -- those bands carry mineral info."""
    return waves_um > 2.80


def plot_style():
    """Set matplotlib rcParams for dark-background publication style."""
    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#c9d1d9",
        "xtick.color":       "#c9d1d9",
        "ytick.color":       "#c9d1d9",
        "text.color":        "#c9d1d9",
        "grid.color":        "#21262d",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "legend.facecolor":  "#161b22",
        "legend.edgecolor":  "#30363d",
        "font.family":       "DejaVu Sans",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "lines.linewidth":   1.4,
        "figure.dpi":        130,
    })


# ---------------------------------------------------------------------------
# 0.1  File finder
# ---------------------------------------------------------------------------

def find_scene_files(scene_dir):
    scene_dir = Path(scene_dir)
    img_candidates = sorted(scene_dir.glob("*_RFL*.IMG"))
    if not img_candidates:
        raise FileNotFoundError(f"No *_RFL*.IMG found in {scene_dir}")
    # Prefer non-segmented; otherwise first segment
    img_path = img_candidates[0]
    for c in img_candidates:
        if c.name.endswith("_RFL.IMG"):
            img_path = c; break

    lbl_candidates = sorted(scene_dir.glob("*_L2.LBL"))
    if not lbl_candidates:
        raise FileNotFoundError(
            f"No *_L2.LBL found in {scene_dir}. Label is _L2.LBL, not _RFL.LBL.")
    lbl_path = lbl_candidates[0]

    hdr_candidates = sorted(scene_dir.glob("*_RFL.HDR"))
    if not hdr_candidates:
        raise FileNotFoundError(f"No *_RFL.HDR found in {scene_dir}")
    hdr_path = hdr_candidates[0]

    fname = img_path.stem  # e.g. M3G20090203T160452_V01_RFL

    print(f"  IMG : {img_path.name}  ({img_path.stat().st_size/1e6:.1f} MB)")
    print(f"  LBL : {lbl_path.name}")
    print(f"  HDR : {hdr_path.name}")

    return {"img": str(img_path), "lbl": str(lbl_path),
            "hdr": str(hdr_path), "fname": fname, "scene_dir": str(scene_dir)}


# ---------------------------------------------------------------------------
# 0.2  PDS3 label parser
# ---------------------------------------------------------------------------

def parse_label(lbl_path):
    meta = {"H": None, "W": None, "B": None,
            "storage_order": "BIL", "invalid_constant": -999.0,
            "record_bytes": None, "file_records": None}

    storage_map = {
        "LINE_INTERLEAVED": "BIL", "BAND_SEQUENTIAL": "BSQ",
        "PIXEL_INTERLEAVED": "BIP", "BIL": "BIL", "BSQ": "BSQ", "BIP": "BIP"}

    with open(lbl_path, "r", errors="replace") as f:
        text = f.read()

    def _get(pattern, cast=str, default=None):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try: return cast(m.group(1).strip().strip('"'))
            except: pass
        return default

    meta["H"]  = _get(r"LINES\s*=\s*(\d+)",           int)
    meta["W"]  = _get(r"LINE_SAMPLES\s*=\s*(\d+)",    int)
    meta["B"]  = _get(r"BANDS\s*=\s*(\d+)",           int)
    if meta["W"] is None:
        meta["W"] = _get(r"SWATH_WIDTH\s*=\s*(\d+)",  int)
    if meta["H"] is None:
        meta["H"] = _get(r"SWATH_LENGTH\s*=\s*(\d+)", int)

    raw_s = _get(r"BAND_STORAGE_TYPE\s*=\s*(\S+)")
    if raw_s:
        meta["storage_order"] = storage_map.get(raw_s.upper(), "BIL")

    inv = _get(r"INVALID_CONSTANT\s*=\s*([-\d.]+)", float)
    if inv is not None:
        meta["invalid_constant"] = inv

    meta["record_bytes"] = _get(r"RECORD_BYTES\s*=\s*(\d+)", int)
    meta["file_records"] = _get(r"FILE_RECORDS\s*=\s*(\d+)",  int)

    print(f"\n  [Label] H={meta['H']}  W={meta['W']}  B={meta['B']}  "
          f"order={meta['storage_order']}  invalid={meta['invalid_constant']}")
    return meta


# ---------------------------------------------------------------------------
# 0.3  ENVI HDR wavelength reader
# ---------------------------------------------------------------------------

def parse_hdr_wavelengths(hdr_path):
    with open(hdr_path, "r", errors="replace") as f:
        text = f.read()
    # Match FIRST wavelength = {...} block only
    m = re.search(r"(?<!\w)wavelength\s*=\s*\{([^}]+)\}",
                  text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))
    if not vals:
        return None
    arr = np.array([float(v) for v in vals], dtype=np.float64)
    print(f"  [HDR] Wavelength table: {len(arr)} bands  "
          f"({arr[0]:.2f}-{arr[-1]:.2f} nm)")
    return arr


# ---------------------------------------------------------------------------
# 0.4  Storage-order auto-detector
# ---------------------------------------------------------------------------

def detect_storage_order(data_flat, H, W, B, waves_um, label_order="BIL"):
    targets = [0.75, 1.0, 1.5, 2.0, 2.5]

    def _snr(order):
        try:
            if order == "BSQ":
                cube_t = data_flat.reshape(B, H, W).transpose(1, 2, 0)
            elif order == "BIL":
                cube_t = data_flat.reshape(H, B, W).transpose(0, 2, 1)
            elif order == "BIP":
                cube_t = data_flat.reshape(H, W, B)
            else:
                return -1.0
            r0 = max(0, H//2-100); r1 = min(H, r0+200)
            c0 = max(0, W//2-100); c1 = min(W, c0+200)
            patch = cube_t[r0:r1, c0:c1, :]
            snrs = []
            for t in targets:
                bi = find_nearest_band(waves_um, t)
                sl = patch[:, :, bi].astype(np.float64).ravel()
                sl = sl[(sl > 0) & np.isfinite(sl)]
                if sl.size < 10: continue
                s = np.std(sl)
                if s < 1e-12: continue
                snrs.append(np.mean(sl) / s)
            return float(np.mean(snrs)) if snrs else -1.0
        except:
            return -1.0

    scores = {o: _snr(o) for o in ("BSQ", "BIL", "BIP")}
    best = max(scores, key=lambda k: scores[k])
    print(f"  [StorageOrder] scores => {scores}")
    chosen = label_order if scores.get(label_order, -1) >= 0.8 * scores[best] else best
    print(f"  [StorageOrder] label='{label_order}' best='{best}' using='{chosen}'")
    return chosen


# ---------------------------------------------------------------------------
# 0.5  Binary loader
# ---------------------------------------------------------------------------

def load_img_file(img_path, meta):
    print(f"\n  Reading binary file ...")
    data = np.fromfile(img_path, dtype=np.float32)
    n = data.size
    H, W, B = meta["H"], meta["W"], meta["B"]

    if H and W and B:
        expected = H * W * B
        if n == expected:
            print(f"  [Loader] Exact match: {n} = {H}x{W}x{B}")
            return data
        for hdr_bytes in [512, 1024, 0]:
            extra = hdr_bytes // 4
            if n - extra == expected:
                print(f"  [Loader] {hdr_bytes}-byte file header -- skipping")
                return data[extra:]
        print(f"  [Loader] WARNING: size mismatch got={n} expected={expected}. Inferring ...")

    for B_try in CANDIDATE_B:
        for W_try in CANDIDATE_W:
            if n % (W_try * B_try) == 0:
                H_try = n // (W_try * B_try)
                if H_try > 50:
                    print(f"  [Loader] Inferred H={H_try} W={W_try} B={B_try}")
                    meta["H"] = H_try; meta["W"] = W_try; meta["B"] = B_try
                    return data

    raise ValueError(f"Cannot infer dims for {n} elements. Tried B={CANDIDATE_B}, W={CANDIDATE_W}.")


# ---------------------------------------------------------------------------
# 0.6  Reshape
# ---------------------------------------------------------------------------

def reshape_cube(data_flat, H, W, B, order):
    if order == "BSQ":
        return data_flat.reshape(B, H, W).transpose(1, 2, 0).astype(np.float32)
    elif order == "BIL":
        return data_flat.reshape(H, B, W).transpose(0, 2, 1).astype(np.float32)
    elif order == "BIP":
        return data_flat.reshape(H, W, B).astype(np.float32)
    raise ValueError(f"Unknown order: {order}")


# ---------------------------------------------------------------------------
# 0.7  Cleaning
# ---------------------------------------------------------------------------

def clean_cube(cube, invalid_constant=-999.0):
    cube = cube.copy()
    cube[cube >= 65534] = np.nan
    cube[np.abs(cube - invalid_constant) < 0.5] = np.nan
    cube[cube > M3_IF_MAX] = np.nan
    cube[cube < M3_IF_MIN] = np.nan
    print(f"  [Clean] NaN fraction: {np.isnan(cube).mean()*100:.2f}%")
    return cube


# ---------------------------------------------------------------------------
# 0.8  Wavelength assignment
# ---------------------------------------------------------------------------

def assign_wavelengths(hdr_waves_nm, B):
    if hdr_waves_nm is not None and len(hdr_waves_nm) == B:
        waves_um = hdr_waves_nm / 1000.0; src = "HDR file"
    elif B == 85:
        waves_um = M3_WAVELENGTHS_85_NM / 1000.0; src = "hard-coded 85-band"
    else:
        waves_um = np.linspace(0.460, 2.976, B); src = f"linear interp ({B} bands)"
    assert waves_um.max() > 2.0, (
        f"Wavelength max={waves_um.max():.3f} um -- should exceed 2.0 um. "
        f"Source: {src}. Check HDR parsing.")
    print(f"  [Waves] {src}  range: {waves_um[0]:.4f}-{waves_um[-1]:.4f} um")
    return waves_um


# ---------------------------------------------------------------------------
# 0.9  SNR sanity check
# ---------------------------------------------------------------------------

def snr_sanity_check(cube, waves_um):
    H, W, B = cube.shape
    r0 = max(0, H//2-100); r1 = min(H, r0+200)
    c0 = max(0, W//2-100); c1 = min(W, c0+200)
    patch = cube[r0:r1, c0:c1, :]
    targets = [0.75, 1.0, 1.5, 2.0, 2.5]
    print(f"\n  {'Band(um)':>10}  {'Idx':>5}  {'Mean':>10}  {'Std':>10}  {'SNR':>8}  Status")
    print("  " + "-"*55)
    snrs = []
    for t in targets:
        bi = find_nearest_band(waves_um, t)
        sl = patch[:, :, bi].astype(np.float64).ravel()
        sl = sl[np.isfinite(sl) & (sl > 0)]
        if sl.size < 10:
            print(f"  {waves_um[bi]:>10.4f}  {bi:>5}  {'--':>10}  {'--':>10}  {'--':>8}  SKIP")
            continue
        mu = np.mean(sl); sig = np.std(sl)
        snr = mu/sig if sig > 1e-12 else 0.0
        snrs.append(snr)
        flag = "WARNING" if snr < 5 else "OK"
        print(f"  {waves_um[bi]:>10.4f}  {bi:>5}  {mu:>10.5f}  {sig:>10.6f}  {snr:>8.2f}  {flag}")
    mean_snr = float(np.mean(snrs)) if snrs else 0.0
    print()
    if mean_snr < 5:
        print("  WARNING: Mean SNR < 5 -- wrong storage order / radiance units / shadow scene?")
    else:
        print(f"  PASS: SNR sanity check passed (mean SNR = {mean_snr:.2f})")
    return mean_snr


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def setup_dirs(scene_dir):
    """Create Pipeline_outputs inside the location (parent of scene_dir)."""
    location_dir = Path(scene_dir).parent
    outputs_base = location_dir / "Pipeline_outputs"
    subdirs = ["01_eda", "02_noise", "03_physics", "04_denoised", "05_validation"]
    DIRS = {"base": str(outputs_base)}
    for s in subdirs:
        d = outputs_base / s
        d.mkdir(parents=True, exist_ok=True)
        DIRS[s] = str(d)
    # Models at journal root  (two levels above scene_dir)
    models_dir = Path(scene_dir).parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    DIRS["models"] = str(models_dir)
    print(f"\n  Output tree: {outputs_base}")
    return DIRS


# ---------------------------------------------------------------------------
# 0.10  Summary print
# ---------------------------------------------------------------------------

def print_step0_summary(cube, waves_um, fname, meta, DIRS):
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dev_str = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    except (ImportError, OSError):
        # OSError covers CUDA DLL version mismatch (WinError 127)
        device = "cpu"; dev_str = "CPU (torch CUDA DLL issue -- will use CPU)"
    H, W, B = cube.shape
    print("\n" + "="*60)
    print("  STEP 0 SUMMARY")
    print("="*60)
    print(f"  Scene      : {fname}")
    print(f"  Shape      : {H} x {W} x {B}")
    print(f"  Wavelengths: {waves_um[0]:.4f} - {waves_um[-1]:.4f} um")
    print(f"  Device     : {dev_str}")
    print(f"  Outputs    : {DIRS['base']}")
    print(f"\n  Scene files:")
    for f in sorted(Path(meta.get("scene_dir", ".")).iterdir()):
        sz = f.stat().st_size/1e6 if f.is_file() else 0
        tag = f"  {sz:>8.2f} MB" if f.is_file() else ""
        print(f"    {f.name}{tag}")
    print("="*60)
    print("  Step 0 complete. Run step1_eda.py next.")
    print("="*60 + "\n")
    return device


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def load_scene(scene_dir, force_storage_order=None):
    """
    Full loader for one M3 scene directory.

    Returns
    -------
    cube   : np.ndarray float32  (H, W, B)
    waves  : np.ndarray float64  (B,)  in um
    fname  : str
    meta   : dict
    DIRS   : dict  paths to output subdirs + models
    """
    print("\n" + "="*60)
    print("  STEP 0 -- M3 Cube Loader")
    print("="*60)

    files = find_scene_files(scene_dir)
    fname = files["fname"]
    meta  = {"scene_dir": scene_dir}

    label_meta = parse_label(files["lbl"])
    meta.update(label_meta)

    hdr_waves_nm = parse_hdr_wavelengths(files["hdr"])
    data_flat    = load_img_file(files["img"], meta)

    H, W, B = meta["H"], meta["W"], meta["B"]
    waves_um = assign_wavelengths(hdr_waves_nm, B)

    if force_storage_order:
        order = force_storage_order.upper()
        print(f"  [StorageOrder] Forced: '{order}'")
    else:
        order = detect_storage_order(data_flat, H, W, B, waves_um,
                                     label_order=meta["storage_order"])
    meta["storage_order_used"] = order

    cube = reshape_cube(data_flat, H, W, B, order)
    del data_flat
    print(f"  [Reshape] cube shape: {cube.shape}")

    cube = clean_cube(cube, invalid_constant=meta["invalid_constant"])

    print("\n  SNR Sanity Check (200x200 central patch):")
    snr_sanity_check(cube, waves_um)

    DIRS = setup_dirs(scene_dir)
    DIRS["scene_dir"] = scene_dir
    DIRS["fname"]     = fname

    meta["fname"] = fname
    meta["H"] = H; meta["W"] = W; meta["B"] = B

    print_step0_summary(cube, waves_um, fname, meta, DIRS)
    return cube, waves_um, fname, meta, DIRS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sd = r"D:\M3 Journal\Mare Tranquillitatis\M3G20090203T160452_V01_RFL"
    else:
        sd = sys.argv[1]
    force = sys.argv[2] if len(sys.argv) > 2 else None
    cube, waves, fname, meta, DIRS = load_scene(sd, force_storage_order=force)
    print(f"Cube loaded: {cube.shape}  dtype={cube.dtype}")
