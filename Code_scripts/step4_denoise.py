"""
step4_denoise.py  --  M3 Phase 1, Step 4: ML-Based Denoising (DAE + CNN)
=========================================================================
Two models:
    DAE  -- Spectral Denoising Autoencoder (Noise2Void, residual learning)
    CNN  -- 1D Stripe CNN (synthetic stripe injection)

MULTI-SCENE TRAINING:
    When run_pipeline.py runs in multi-scene mode, DAE is trained on spectra
    from ALL 10 scenes before any single-scene inference.
    Models are saved to D:/M3 Journal/models/ and reused for every scene.

OOM GUARD:
    Batch sizes start at nominal and halve automatically on CUDA OOM.
    Memory fraction capped at 87.5 pct leaving a 500 MB OS safety buffer.

Prerequisites: steps 0-3 variables in memory.

Produces:
    04_denoised/{fname}_denoised.npy
    04_denoised/{fname}_training_curves.png
    04_denoised/{fname}_snr_comparison.png
    04_denoised/{fname}_blend_weights.png
    04_denoised/{fname}_spectral_comparison.png
    models/m3_dae_B{B}.pth
    models/m3_cnn_H{H}.pth
"""

import json
import sys
import gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DAE_EPOCHS      = 80
CNN_EPOCHS      = 25   # early stopping keeps this from wasting epochs
BATCH_SIZE      = 512
LR_DAE          = 1e-3
LR_CNN          = 5e-4
WEIGHT_DECAY    = 1e-4
MASK_FRACTION   = 0.10
LAMBDA_SMOOTH   = 0.00
LAMBDA_SAM      = 0.005  # preserves continuum/absorption geometry during retraining
N_TRAIN_SPECTRA = 15000
BLEND_MIN       = 0.15
BLEND_MAX       = 0.80   # Reduced from 0.95 -- prevents extreme DAE-dominated blends
                         # that wipe out narrow absorption features
INFER_BATCH     = 4096   # automatically halved on OOM
BAND_DEPTH_CHG_THR = 0.20
BAND_DEPTH_DETECT_THR = 0.01
MINERAL_GUARD_ROI_THR = 0.02
MINERAL_GUARD_ROI_PERCENTILE = 90
MINERAL_GUARD_MAX_SAM_DEG = 2.0
BLEND_SAM_THR_RAD = 0.035
BLEND_PEARSON_THR = 0.995
BLEND_SAMPLE_PIXELS = 12000
GAUSSIAN_SPATIAL_BLEND = 0.25
GAUSSIAN_SPATIAL_SIGMA = 0.75

MINERAL_BAND_DEFS = [
    ("BD2300",    2.295, 2.12,  2.53),
    ("BD1900",    1.930, 1.85,  2.07),
    ("BD2210",    2.21,  2.165, 2.25),
    ("SINDEX2",   2.10,  1.815, 2.20),
    ("BD1500",    1.50,  1.43,  1.815),
    ("HCPINDEX2", 2.14,  1.82,  2.46),
]


# ---------------------------------------------------------------------------
# Device setup with OOM guard
# ---------------------------------------------------------------------------

def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            # Reserve ~500 MB for OS/overhead; use at most 87.5 % of 4 GB
            torch.cuda.set_per_process_memory_fraction(0.875)
            return torch.device("cuda")
    except (ImportError, OSError):
        # OSError: CUDA DLL version mismatch (WinError 127) -- fall back to CPU
        pass
    try:
        import torch
    except (ImportError, OSError) as e:
        raise RuntimeError(
            "PyTorch could not be imported. Install with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
        ) from e
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 4.12  SPM Indices (fully vectorised)
# ---------------------------------------------------------------------------

def compute_spm_indices_vectorized(cube_phys, waves_um, valid_pixel_mask):
    from step0_load import find_nearest_band
    H, W, B = cube_phys.shape
    spm = np.zeros((H, W, 8), dtype=np.float32)
    w = waves_um

    def safe_band(t):
        if w[0] <= t <= w[-1]:
            return find_nearest_band(w, t)
        return None

    def bd_vec(cube, center, sh1, sh2, idx):
        bc  = safe_band(center)
        bs1 = safe_band(sh1)
        bs2 = safe_band(sh2)
        if None in (bc, bs1, bs2):
            return
        rc   = cube[:, :, bc].astype(np.float64)
        rs1  = cube[:, :, bs1].astype(np.float64)
        rs2  = cube[:, :, bs2].astype(np.float64)
        frac = (center - sh1) / max(sh2 - sh1, 1e-8)
        rcont = rs1 + frac * (rs2 - rs1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bd = 1.0 - rc / np.where(rcont > 1e-8, rcont, np.nan)
        bd = np.where(np.isfinite(bd) & valid_pixel_mask, bd, 0.0)
        spm[:, :, idx] = bd.astype(np.float32)

    bd_vec(cube_phys, 2.295, 2.12,  2.53,  0)  # BD2300  phyllosilicates
    bd_vec(cube_phys, 1.930, 1.85,  2.07,  1)  # BD1900  hydrated minerals
    # OLINDEX3: reflectance at 1.05 um as proxy
    bo = safe_band(1.05)
    if bo is not None:
        ref = cube_phys[:, :, bo].astype(np.float64)
        spm[:, :, 2] = np.where(
            valid_pixel_mask & np.isfinite(ref),
            ref.astype(np.float32), 0.0)
    bd_vec(cube_phys, 2.21,  2.165, 2.25,  3)  # BD2210  Al-OH clays
    bd_vec(cube_phys, 2.10,  1.815, 2.20,  4)  # SINDEX2 sulfates
    bd_vec(cube_phys, 1.50,  1.43,  1.815, 5)  # BD1500  water ice
    # RPEAK1: max in 0.60-0.84 um
    lo6 = safe_band(0.60)
    hi8 = safe_band(0.84)
    if lo6 is not None and hi8 is not None:
        peak = np.nanmax(cube_phys[:, :, lo6:hi8 + 1], axis=2)
        spm[:, :, 6] = np.where(
            valid_pixel_mask & np.isfinite(peak),
            peak.astype(np.float32), 0.0)
    bd_vec(cube_phys, 2.14,  1.82,  2.46,  7)  # HCPINDEX2 HCP pyroxene
    return spm


# ---------------------------------------------------------------------------
# 4.4  Stratified training data collection
# ---------------------------------------------------------------------------

def collect_training_spectra(cube_phys, waves_um, valid_pixel_mask, spm,
                              n_target=N_TRAIN_SPECTRA, seed=42):
    H, W, B = cube_phys.shape
    rng = np.random.default_rng(seed)
    rows_v, cols_v = np.where(valid_pixel_mask)
    n_valid = len(rows_v)
    collected = []
    in_top = np.zeros(n_valid, dtype=bool)
    n_per_stratum = 400

    for si in range(spm.shape[2]):
        idx_vals  = spm[rows_v, cols_v, si]
        positives = idx_vals[idx_vals > 0]
        threshold = float(np.percentile(positives, 80)) if positives.size > 0 else 1.0
        top_mask  = idx_vals >= threshold
        in_top   |= top_mask
        top_idx   = np.where(top_mask)[0]
        if top_idx.size == 0:
            continue
        chosen = rng.choice(top_idx,
                             size=min(n_per_stratum, top_idx.size),
                             replace=False)
        for ci in chosen:
            r, c = int(rows_v[ci]), int(cols_v[ci])
            spec = cube_phys[r, c, :].astype(np.float64)
            if np.isnan(spec).mean() > 0.5:
                continue
            mean_val = np.nanmean(spec)
            if not np.isfinite(mean_val):   # all-NaN spectrum -- skip
                continue
            spec = np.where(np.isnan(spec), mean_val, spec)
            if np.all(np.isfinite(spec)):   # only keep fully valid spectra
                collected.append(spec)

    # Fill remaining quota with background pixels
    bg_idx    = np.where(~in_top)[0]
    bg_needed = max(0, n_target - len(collected))
    if bg_needed > 0 and bg_idx.size > 0:
        chosen_bg = rng.choice(bg_idx,
                                size=min(bg_needed, bg_idx.size),
                                replace=False)
        for ci in chosen_bg:
            r, c = int(rows_v[ci]), int(cols_v[ci])
            spec = cube_phys[r, c, :].astype(np.float64)
            if np.isnan(spec).mean() > 0.5:
                continue
            mean_val = np.nanmean(spec)
            if not np.isfinite(mean_val):
                continue
            spec = np.where(np.isnan(spec), mean_val, spec)
            if np.all(np.isfinite(spec)):
                collected.append(spec)

    spectra = np.array(collected[:n_target], dtype=np.float32)
    print(f"  Training spectra: {spectra.shape[0]} from {n_valid:,} valid pixels")
    return spectra


# ---------------------------------------------------------------------------
# 4.1  DAE Architecture
# ---------------------------------------------------------------------------

def build_dae(B, device):
    import torch.nn as nn
    import torch

    class DAE(nn.Module):
        def __init__(self, B):
            super().__init__()
            # 1D Convolutional Denoising Autoencoder preserves local spectral features (dips)
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 1, kernel_size=5, padding=2)
            )

        def forward(self, x):
            x = x.unsqueeze(1) # (batch, 1, B)
            return self.net(x).squeeze(1)

    return DAE(B).to(device)


# ---------------------------------------------------------------------------
# 4.2  DAE Loss
# ---------------------------------------------------------------------------

def dae_loss(model_out, x_input, mask_indices):
    """
    N2V loss for direct-prediction DAE (model_out IS the clean spectrum).

    Key design decisions
    --------------------
    - Reconstruction (recon): MSE at masked positions only (pure N2V principle).
      The model must predict the noisy observation from spectral neighbours;
      this is exactly Noise2Void for spectral data.
    - Smooth:  penalise spectral roughness in the prediction (physics prior:
      M3 reflectance spectra are smooth apart from narrow absorption bands).
    - SAM:     spectral angle between clean_pred and the noisy input.
      Because M3 Gaussian noise is spectrally unselective (white-noise like),
      the spectral angle of the noisy spectrum ≈ that of the clean spectrum.
      Minimising SAM preserves mineral absorption feature geometry.
    """
    import torch
    import torch.nn.functional as F

    clean_pred = model_out   # model directly predicts the clean spectrum

    # -- N2V reconstruction loss: predict noisy observation from spectral context
    recon_vals = []
    for i in range(x_input.shape[0]):
        m = mask_indices[i]
        if len(m) > 0:
            recon_vals.append(
                ((clean_pred[i, m] - x_input[i, m]) ** 2).mean())
    recon_loss = (torch.stack(recon_vals).mean()
                  if recon_vals
                  else torch.tensor(0.0, device=x_input.device))

    # -- Spectral smoothness of prediction
    smooth_loss = (torch.diff(clean_pred, dim=1) ** 2).mean()

    # -- SAM: preserve spectral angle (guards mineral absorption features)
    cp_norm  = F.normalize(clean_pred, dim=1, eps=1e-8)
    xi_norm  = F.normalize(x_input,   dim=1, eps=1e-8)
    cos_sim  = (cp_norm * xi_norm).sum(dim=1).clamp(-1.0, 1.0)
    sam_loss = torch.acos(cos_sim).mean()

    total = recon_loss + LAMBDA_SMOOTH * smooth_loss + LAMBDA_SAM * sam_loss
    return total, recon_loss, sam_loss


# ---------------------------------------------------------------------------
# 4.3  Noise2Void masking
# ---------------------------------------------------------------------------

def apply_n2v_mask(x_batch, valid_bands, mask_frac=MASK_FRACTION):
    """
    N2V blind-spot masking: for each masked band, replace its value with
    the mean of its two nearest spectral neighbours in valid_bands.
    The model cannot see the masked band but can see its context.
    This is the correct N2V 'blind-spot' substitution for spectral data.
    """
    import torch
    x_masked     = x_batch.clone()
    mask_indices = []
    vb           = list(valid_bands)       # ordered list of valid band indices
    n_vb         = len(vb)
    n_mask       = max(1, int(n_vb * mask_frac))

    for i in range(x_batch.shape[0]):
        perm    = torch.randperm(n_vb)[:n_mask].tolist()  # positions into vb
        m_bands = [vb[j] for j in perm]                   # actual band indices
        for j, b in zip(perm, m_bands):
            # Classic N2V: replace with a random pixel from the local neighborhood
            # This prevents the network from learning the identity interpolation function
            offset = int(torch.randint(1, 3, (1,)).item())
            if torch.rand(1).item() < 0.5:
                offset = -offset
            n_rand = vb[max(0, min(n_vb - 1, j + offset))]
            x_masked[i, b] = x_batch[i, n_rand]
        mask_indices.append(m_bands)
    return x_masked, mask_indices


# ---------------------------------------------------------------------------
# 4.5  DAE Training
# ---------------------------------------------------------------------------

def train_dae(spectra, valid_bands, B, device, model_path):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler

    if Path(model_path).exists():
        print(f"  [DAE] Loading saved model: {model_path}")
        ckpt  = torch.load(model_path, map_location=device, weights_only=False)
        model = build_dae(B, device)
        try:
            model.load_state_dict(ckpt["state_dict"])
            hp = ckpt.get("hyperparams", {})
            ckpt_sam = float(hp.get("lambda_sam", 0.0))
            ckpt_smooth = float(hp.get("lambda_smooth", 0.0))
            if abs(ckpt_sam - LAMBDA_SAM) > 1e-12 or abs(ckpt_smooth - LAMBDA_SMOOTH) > 1e-12:
                raise RuntimeError(
                    "checkpoint hyperparameters differ from current settings "
                    f"(ckpt lambda_sam={ckpt_sam}, current={LAMBDA_SAM}; "
                    f"ckpt lambda_smooth={ckpt_smooth}, current={LAMBDA_SMOOTH})"
                )
            model.eval()
            return model, ckpt.get("history", [])
        except RuntimeError as e:
            old_path = Path(model_path).with_suffix(".incompatible.pth")
            try:
                Path(model_path).replace(old_path)
                print(f"  [DAE] Incompatible checkpoint moved to: {old_path}")
            except OSError:
                print("  [DAE] Incompatible checkpoint could not be moved; retraining in place.")
            print(f"  [DAE] Checkpoint architecture mismatch; retraining. Details: {e}")

    use_amp = (device.type == "cuda")

    # ── 90 / 10 train / val split (reproducible) ───────────────────────────
    rng_split = np.random.default_rng(0)
    perm      = rng_split.permutation(len(spectra))
    n_val     = max(50, int(0.1 * len(spectra)))
    val_spec  = spectra[perm[:n_val]]
    trn_spec  = spectra[perm[n_val:]]

    print(f"  [DAE] Training N2V on {len(trn_spec)} spectra "
          f"(val={len(val_spec)})  {DAE_EPOCHS} epochs  AMP={use_amp} ...")

    model  = build_dae(B, device)
    opt    = torch.optim.AdamW(model.parameters(),
                                lr=LR_DAE, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DAE_EPOCHS)
    scaler = GradScaler(enabled=use_amp)

    trn_loader = DataLoader(
        TensorDataset(torch.tensor(trn_spec, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=use_amp)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_spec,  dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_amp)

    history       = []
    best_val      = float("inf")
    patience_cnt  = 0
    PATIENCE      = 10
    best_state    = None

    for epoch in range(1, DAE_EPOCHS + 1):
        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        trn_loss = 0.0; n_trn = 0
        for (x_b,) in trn_loader:
            x_b = x_b.to(device, non_blocking=use_amp)
            x_m, mask_idx = apply_n2v_mask(x_b, valid_bands)
            opt.zero_grad()
            with autocast(enabled=use_amp):
                out  = model(x_m)
                loss, _, _ = dae_loss(out, x_b, mask_idx)
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            trn_loss += loss.item(); n_trn += 1
        sched.step()

        # ── Validation pass ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0; n_val_b = 0
        with torch.no_grad():
            for (xv,) in val_loader:
                xv = xv.to(device)
                xv_m, mask_v = apply_n2v_mask(xv, valid_bands)
                with autocast(enabled=use_amp):
                    out_v = model(xv_m)
                    lv, _, _ = dae_loss(out_v, xv, mask_v)
                if not torch.isnan(lv):
                    val_loss += lv.item(); n_val_b += 1

        avg_trn = trn_loss / max(n_trn,   1)
        avg_val = val_loss / max(n_val_b, 1)
        history.append(avg_trn)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{DAE_EPOCHS}  "
                  f"train={avg_trn:.6f}  val={avg_val:.6f}", flush=True)

        # ── Early stopping on validation loss ──────────────────────────────
        if avg_val < best_val:
            best_val    = avg_val
            best_state  = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  [DAE] Early stopping at epoch {epoch} "
                      f"(best val={best_val:.6f})", flush=True)
                break

    # Restore the best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    torch.save({
        "state_dict": model.state_dict(),
        "history": history,
        "hyperparams": {
            "lambda_sam": LAMBDA_SAM,
            "lambda_smooth": LAMBDA_SMOOTH,
            "mask_fraction": MASK_FRACTION,
            "dae_epochs": DAE_EPOCHS,
            "lr_dae": LR_DAE,
            "weight_decay": WEIGHT_DECAY,
        },
    }, model_path)
    print(f"  [DAE] Saved: {model_path}  (best val={best_val:.6f})")
    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# 4.6  DAE Inference (OOM-safe)
# ---------------------------------------------------------------------------

def infer_dae(model, cube_phys, valid_pixel_mask, valid_bands, dead_bands, device):
    import torch
    from step0_load import M3_IF_MAX

    H, W, B   = cube_phys.shape
    # Memory-efficient per-band mean: avoids creating giant (H*W, B) bool mask
    band_mean = np.empty(B, dtype=np.float32)
    for _b in range(B):
        _sl = cube_phys[:, :, _b].ravel()
        _finite = _sl[np.isfinite(_sl)]
        band_mean[_b] = float(_finite.mean()) if _finite.size > 0 else 0.0
    cube_dae  = cube_phys.copy()
    rows_v, cols_v = np.where(valid_pixel_mask)
    n_valid   = len(rows_v)
    batch_size = INFER_BATCH
    model.eval()

    with torch.no_grad():
        i = 0
        while i < n_valid:
            try:
                j        = min(i + batch_size, n_valid)
                rs       = rows_v[i:j]
                cs       = cols_v[i:j]
                specs    = cube_phys[rs, cs, :].astype(np.float32)
                nan_mask = ~np.isfinite(specs)
                specs    = np.where(nan_mask, band_mean[np.newaxis, :], specs)
                x_t      = torch.tensor(specs, dtype=torch.float32, device=device)
                # Model predicts the clean spectrum directly (N2V direct-prediction)
                clean    = model(x_t).cpu().numpy()
                clean    = np.where(nan_mask, np.nan, clean)
                clean    = np.clip(clean, 0.0, M3_IF_MAX)
                cube_dae[rs, cs, :] = clean.astype(np.float32)
                i = j
            except (RuntimeError, MemoryError) as e:
                if "memory" in str(e).lower() and batch_size > 64:
                    batch_size //= 2
                    print(f"  [OOM] DAE inference batch -> {batch_size}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise

    print(f"  [DAE] Inference done, final batch_size={batch_size}")
    return cube_dae


# ---------------------------------------------------------------------------
# 4.7  Stripe CNN Architecture
# ---------------------------------------------------------------------------

def build_cnn(device):
    import torch.nn as nn
    import torch

    class StripeCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=9, padding=4), nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=9, padding=4), nn.ReLU(),
                nn.Conv1d(32, 1,  kernel_size=9, padding=4),
            )

        def forward(self, x):
            return self.net(x)

    return StripeCNN().to(device)


# ---------------------------------------------------------------------------
# 4.8  CNN Training Data -- Synthetic stripe injection
# ---------------------------------------------------------------------------

def generate_cnn_data(cube_phys, valid_bands, dead_bands,
                       stripe_col_offsets, n_pairs=15000, seed=42):
    from step0_load import M3_IF_MAX
    rng      = np.random.default_rng(seed)
    dead_set = set(dead_bands)
    # Use non-striped, non-dead bands as clean references
    clean_bands = [b for b in valid_bands
                   if b not in dead_set and b not in stripe_col_offsets]
    if not clean_bands:
        clean_bands = valid_bands

    if stripe_col_offsets:
        amps = [float(np.std(v)) for v in stripe_col_offsets.values()
                if len(v) > 1]
        amplitude = float(np.median(amps)) if amps else 0.005
    else:
        amplitude = 0.005

    H, W, B = cube_phys.shape
    noisy_list = []
    clean_list = []

    # Cap column length to avoid GPU OOM on large cubes
    MAX_COL = 2048

    for _ in range(n_pairs):
        b = int(rng.choice(clean_bands))
        c = int(rng.integers(0, W))
        col_full = cube_phys[:, c, b].astype(np.float64)

        # Random crop to MAX_COL
        if H > MAX_COL:
            start    = int(rng.integers(0, H - MAX_COL + 1))
            col_full = col_full[start:start + MAX_COL]
        h_crop = len(col_full)

        # Fill NaN so stripe injection doesn't propagate NaN
        nan_mask = ~np.isfinite(col_full)
        if nan_mask.all():
            continue
        col_fill = float(np.nanmedian(col_full))
        col_clean = np.where(nan_mask, col_fill, col_full)

        dc    = float(rng.normal(0, amplitude))
        freq  = float(rng.uniform(0.005, 0.05))
        phase = float(rng.uniform(0, 2 * np.pi))
        t     = np.arange(h_crop, dtype=np.float64)
        stripe    = dc + 0.3 * abs(dc) * np.sin(2 * np.pi * freq * t + phase)
        col_noisy = np.clip(col_clean + stripe, 0.0, M3_IF_MAX)
        noisy_list.append(col_noisy.astype(np.float32))
        clean_list.append(col_clean.astype(np.float32))

    return np.array(noisy_list), np.array(clean_list)


# ---------------------------------------------------------------------------
# 4.9  CNN Training
# ---------------------------------------------------------------------------

def train_cnn(cube_phys, valid_bands, dead_bands, stripe_col_offsets,
              H, device, model_path):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler

    if Path(model_path).exists():
        print(f"  [CNN] Loading saved model: {model_path}")
        ckpt  = torch.load(model_path, map_location=device, weights_only=False)
        model = build_cnn(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt.get("history", [])

    print("  [CNN] Generating synthetic stripe pairs ...")
    noisy_arr, clean_arr = generate_cnn_data(
        cube_phys, valid_bands, dead_bands, stripe_col_offsets)

    if noisy_arr.shape[0] < 100:
        print("  [CNN] Not enough pairs -- skipping CNN training")
        return None, []

    use_amp   = (device.type == "cuda")
    cnn_batch = min(BATCH_SIZE, 128)
    print(f"  [CNN] Training on {noisy_arr.shape[0]} pairs, "
          f"{CNN_EPOCHS} epochs  AMP={use_amp}  batch={cnn_batch} ...")
    model  = build_cnn(device)
    opt    = torch.optim.AdamW(model.parameters(),
                                lr=LR_CNN, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CNN_EPOCHS)
    scaler = GradScaler(enabled=use_amp)
    noisy_t = torch.tensor(noisy_arr[:, np.newaxis, :], dtype=torch.float32)
    clean_t = torch.tensor(clean_arr[:, np.newaxis, :], dtype=torch.float32)
    loader  = DataLoader(TensorDataset(noisy_t, clean_t),
                          batch_size=cnn_batch, shuffle=True,
                          drop_last=True, pin_memory=use_amp)
    history      = []
    nan_streak   = 0
    best_loss    = float("inf")
    patience_cnt = 0
    PATIENCE     = 5   # epochs without >=1e-5 improvement -> stop

    for epoch in range(1, CNN_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for (xn, xc) in loader:
            xn, xc = xn.to(device, non_blocking=use_amp), xc.to(device, non_blocking=use_amp)
            opt.zero_grad()
            try:
                with autocast(enabled=use_amp):
                    sp   = model(xn)
                    cp   = xn - sp
                    mse  = ((cp - xc) ** 2).mean()
                    l1   = sp.abs().mean()
                    mc   = (cp.mean(dim=2) - xc.mean(dim=2)).abs().mean()
                    loss = mse + 0.01 * l1 + 0.5 * mc
            except (RuntimeError, torch.OutOfMemoryError):
                torch.cuda.empty_cache(); gc.collect()
                continue
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1
        sched.step()
        avg = epoch_loss / max(n_batches, 1) if n_batches > 0 else float("nan")
        history.append(avg)

        if np.isnan(avg):
            nan_streak += 1
        else:
            nan_streak = 0
            if avg < best_loss - 1e-5:
                best_loss    = avg
                patience_cnt = 0
            else:
                patience_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    CNN Epoch {epoch:3d}/{CNN_EPOCHS}  loss={avg:.6f}  best={best_loss:.6f}")
        if nan_streak >= 5:
            print("  [CNN] WARNING: 5 NaN epochs -- stopping early")
            break
        if patience_cnt >= PATIENCE:
            print(f"  [CNN] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    if all(np.isnan(h) for h in history):
        print("  [CNN] WARNING: Training diverged. NOT saving model.")
        model.eval()
        return None, history

    torch.save({"state_dict": model.state_dict(), "history": history},
               model_path)
    print(f"  [CNN] Saved: {model_path}")
    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# CNN Inference
# ---------------------------------------------------------------------------

def infer_cnn(cnn_model, cube_phys, striped_bands, device):
    import torch
    from step0_load import M3_IF_MAX

    if cnn_model is None or not striped_bands:
        return {}

    cnn_bands = {}
    H, W, B  = cube_phys.shape

    with torch.no_grad():
        for b in striped_bands:
            band     = cube_phys[:, :, b].astype(np.float32)  # (H, W)
            nan_mask = ~np.isfinite(band)                      # (H, W)

            # Fill NaN with column means so the CNN sees no NaN input
            col_means = np.where(
                np.isfinite(band).any(axis=0),
                np.nanmean(band, axis=0, keepdims=True),       # (1, W) broadcastable
                0.0)
            band_fill = np.where(nan_mask, col_means, band)   # (H, W)

            # Vectorised: all W columns in one batch → (W, 1, H)
            x_t  = torch.tensor(
                band_fill.T[:, np.newaxis, :],                 # (W, 1, H)
                dtype=torch.float32, device=device)
            sp   = cnn_model(x_t).cpu().numpy()[:, 0, :]      # (W, H)
            clean = np.clip(band_fill - sp.T, 0.0, M3_IF_MAX) # (H, W)
            clean[nan_mask] = np.nan
            cnn_bands[b] = clean.astype(np.float32)

    print(f"  [CNN] Inference done on {len(striped_bands)} striped bands")
    return cnn_bands



# ---------------------------------------------------------------------------
# 4.10  Adaptive Blending
# ---------------------------------------------------------------------------

def _mean_sample_sam_pearson(phys_sample, den_sample):
    sam_vals = []
    pear_vals = []
    for p, d in zip(phys_sample, den_sample):
        finite = np.isfinite(p) & np.isfinite(d)
        if finite.sum() < 10:
            continue
        pf = p[finite].astype(np.float64)
        df = d[finite].astype(np.float64)
        pn = np.linalg.norm(pf)
        dn = np.linalg.norm(df)
        if pn > 1e-10 and dn > 1e-10:
            cos_sim = float(np.dot(pf, df) / (pn * dn))
            sam_vals.append(float(np.arccos(np.clip(cos_sim, -1.0, 1.0))))
        if np.std(pf) > 1e-10 and np.std(df) > 1e-10:
            pear_vals.append(float(np.corrcoef(pf, df)[0, 1]))
    mean_sam = float(np.mean(sam_vals)) if sam_vals else 0.0
    mean_pear = float(np.mean(pear_vals)) if pear_vals else 1.0
    return mean_sam, mean_pear


def _select_scene_blend_alpha(cube_phys, cube_dae, dict_cnn, valid_pixel_mask,
                              valid_bands, striped_bands, snr_before):
    rows_v, cols_v = np.where(valid_pixel_mask)
    if len(rows_v) == 0:
        return (BLEND_MIN + BLEND_MAX) / 2.0, 0.0, 1.0

    rng = np.random.default_rng(42)
    idx = rng.choice(len(rows_v), size=min(BLEND_SAMPLE_PIXELS, len(rows_v)),
                     replace=False)
    rs = rows_v[idx]
    cs = cols_v[idx]

    phys_sample = cube_phys[rs, cs, :].astype(np.float32)
    dae_sample = cube_dae[rs, cs, :].astype(np.float32)
    if dict_cnn:
        for b in striped_bands:
            if b in dict_cnn:
                dae_sample[:, b] = dict_cnn[b][rs, cs]

    valid_snr = snr_before[valid_bands] if valid_bands else np.array([])
    mean_snr = float(np.nanmean(valid_snr)) if valid_snr.size else 10.0
    if mean_snr < 8:
        candidates = np.array([0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15])
    elif mean_snr < 15:
        candidates = np.array([0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.15])
    else:
        candidates = np.array([0.55, 0.45, 0.35, 0.25, 0.15])

    best = (float(candidates[-1]), float("inf"), -1.0)
    for alpha in candidates:
        den_sample = alpha * dae_sample + (1.0 - alpha) * phys_sample
        mean_sam, mean_pear = _mean_sample_sam_pearson(phys_sample, den_sample)
        if mean_sam <= BLEND_SAM_THR_RAD and mean_pear >= BLEND_PEARSON_THR:
            return float(alpha), mean_sam, mean_pear
        if mean_sam < best[1] or (abs(mean_sam - best[1]) < 1e-6 and mean_pear > best[2]):
            best = (float(alpha), mean_sam, mean_pear)

    # If no candidate passes, use the least distorting candidate instead of a
    # high-SNR but spectrally unsafe blend.
    return best


def adaptive_blend(cube_phys, cube_dae, dict_cnn, waves_um,
                   valid_bands, striped_bands, snr_before, valid_pixel_mask):
    from step0_load import M3_IF_MAX

    H, W, B       = cube_phys.shape
    blend_weights = np.zeros(B, dtype=np.float32)
    alpha_scene, sample_sam, sample_pear = _select_scene_blend_alpha(
        cube_phys, cube_dae, dict_cnn, valid_pixel_mask, valid_bands,
        striped_bands, snr_before)

    # ── Mineral absorption centre protection ──────────────────────────────────
    # At these wavelengths the DAE conv kernels tend to smooth away narrow
    # absorption dips. We protect them by capping the DAE contribution so at
    # least (1 - MINERAL_PROTECT_ALPHA) of the raw physics signal is retained.
    # Window of ±MINERAL_WINDOW_BANDS adjacent bands is also protected to cover
    # the full dip shoulder, not just the single band peak.
    try:
        from step0_load import find_nearest_band as _fnb
        _MINERAL_CENTERS_UM = [
            1.930,   # BD1900  -- hydrated minerals / water
            2.210,   # BD2210  -- Al-OH clays
            1.500,   # BD1500  -- water ice
            2.100,   # SINDEX2 -- sulfates (shoulder)
            2.295,   # BD2300  -- phyllosilicates
        ]
        MINERAL_PROTECT_ALPHA  = 0.30   # max DAE weight at absorption centres
        MINERAL_WINDOW_BANDS   = 6 if B > 100 else 2  # wider window in Target mode (256 bands) to cover same nm range
        _mineral_protected = set()
        for _wum in _MINERAL_CENTERS_UM:
            if waves_um[0] <= _wum <= waves_um[-1]:
                _ci = _fnb(waves_um, _wum)
                for _delta in range(-MINERAL_WINDOW_BANDS, MINERAL_WINDOW_BANDS + 1):
                    _nb = _ci + _delta
                    if 0 <= _nb < B:
                        _mineral_protected.add(_nb)
        print(f"  [Blend] Mineral-protected bands: {len(_mineral_protected)} "
              f"(alpha capped at {MINERAL_PROTECT_ALPHA})")
    except Exception as _e:
        print(f"  [Blend] WARNING: mineral protection setup failed ({_e}); skipping.")
        _mineral_protected = set()
        MINERAL_PROTECT_ALPHA = BLEND_MAX
    # ─────────────────────────────────────────────────────────────────────────

    alpha_full = np.zeros(B, dtype=np.float32)
    striped_set = set(striped_bands)
    for b in valid_bands:
        alpha = alpha_scene
        if b in striped_set:
            alpha = min(BLEND_MAX, alpha + 0.10)
        # Mineral centre override: DAE contribution capped to preserve absorption dips
        if b in _mineral_protected:
            alpha = min(alpha, MINERAL_PROTECT_ALPHA)
        alpha_full[b] = alpha
        blend_weights[b] = alpha

    cube_denoised = cube_dae  # Re-use memory in-place

    # Process in contiguous row chunks to completely eliminate strided memory thrashing
    chunk_size = 500
    alpha_view = alpha_full.reshape(1, 1, B)
    
    for i in range(0, H, chunk_size):
        j = min(i + chunk_size, H)
        
        phys_chunk = cube_phys[i:j, :, :]
        dae_chunk  = cube_denoised[i:j, :, :]
        
        # Apply CNN overrides to the dae_chunk directly in cache
        if dict_cnn:
            for b in striped_bands:
                if b in dict_cnn:
                    dae_chunk[:, :, b] = dict_cnn[b][i:j, :]
                    
        # Blend (broadcasting alpha across H and W)
        blended = alpha_view * dae_chunk + (1.0 - alpha_view) * phys_chunk
        blended = np.clip(blended, 0.0, M3_IF_MAX)
        
        # Restore NaNs
        blended[~np.isfinite(phys_chunk)] = np.nan
        
        # Write contiguous block back
        cube_denoised[i:j, :, :] = blended.astype(np.float32)

    vb = valid_bands
    print(f"  [Blend] selected scene alpha={alpha_scene:.3f}  "
          f"sample SAM={sample_sam:.4f} rad  Pearson={sample_pear:.4f}")
    print(f"  [Blend] alpha mean={float(np.mean(blend_weights[vb])):.3f}  "
          f"range=[{float(blend_weights[vb].min()):.3f}, "
          f"{float(blend_weights[vb].max()):.3f}]", flush=True)
    return cube_denoised, blend_weights


# ---------------------------------------------------------------------------
# 4.11  Post-denoising SNR
# ---------------------------------------------------------------------------

def compute_snr_ml(cube_denoised, valid_bands):
    H, W, B = cube_denoised.shape
    snr = np.zeros(B, dtype=np.float64)
    
    # Process in row chunks to avoid strided-access thrashing
    sum_h  = np.zeros(B, dtype=np.float64)
    sum_h2 = np.zeros(B, dtype=np.float64)
    count_h= np.zeros(B, dtype=np.int64)
    
    sum_v  = np.zeros(B, dtype=np.float64)
    sum_v2 = np.zeros(B, dtype=np.float64)
    count_v= np.zeros(B, dtype=np.int64)
    
    sum_s  = np.zeros(B, dtype=np.float64)
    count_s= np.zeros(B, dtype=np.int64)
    
    chunk_size = 500
    for i in range(0, H - 1, chunk_size):
        j = min(i + chunk_size, H - 1)
        chunk = cube_denoised[i:j+1, :, :]
        
        dh = np.diff(chunk[:-1, :, :], axis=1)
        dv = np.diff(chunk, axis=0)
        c_vals = chunk[:-1, :, :]
        
        for b in valid_bands:
            # H-diff
            dh_b = dh[:, :, b]
            vh = dh_b[np.isfinite(dh_b)]
            count_h[b] += vh.size
            sum_h[b]   += np.sum(vh)
            sum_h2[b]  += np.sum(vh**2)
            
            # V-diff
            dv_b = dv[:, :, b]
            vv = dv_b[np.isfinite(dv_b)]
            count_v[b] += vv.size
            sum_v[b]   += np.sum(vv)
            sum_v2[b]  += np.sum(vv**2)
            
            # Signal
            cv_b = c_vals[:, :, b]
            vs = cv_b[np.isfinite(cv_b)]
            count_s[b] += vs.size
            sum_s[b]   += np.sum(vs)
            
    for b in valid_bands:
        if count_h[b] < 100 or count_v[b] < 100 or count_s[b] == 0:
            continue
            
        mean_h = sum_h[b] / count_h[b]
        var_h  = (sum_h2[b] / count_h[b]) - mean_h**2
        std_h  = np.sqrt(max(0, var_h))
        
        mean_v = sum_v[b] / count_v[b]
        var_v  = (sum_v2[b] / count_v[b]) - mean_v**2
        std_v  = np.sqrt(max(0, var_v))
        
        noise_h = std_h / 1.41421356
        noise_v = std_v / 1.41421356
        noise = (noise_h + noise_v) / 2.0
        
        signal = sum_s[b] / count_s[b]
        if noise > 1e-12:
            snr[b] = signal / noise
            
    return snr


def _sample_band_depth(cube, waves_um, center, sh1, sh2, rows, cols):
    from step0_load import find_nearest_band

    if center < waves_um[0] or center > waves_um[-1]:
        return None
    bc = find_nearest_band(waves_um, center)
    bs1 = find_nearest_band(waves_um, sh1)
    bs2 = find_nearest_band(waves_um, sh2)
    frac = (center - sh1) / max(sh2 - sh1, 1e-8)

    rc = cube[rows, cols, bc].astype(np.float64)
    rs1 = cube[rows, cols, bs1].astype(np.float64)
    rs2 = cube[rows, cols, bs2].astype(np.float64)
    finite = np.isfinite(rc) & np.isfinite(rs1) & np.isfinite(rs2)
    if not np.any(finite):
        return None
    rcont = rs1 + frac * (rs2 - rs1)
    good = finite & (rcont > 1e-8)
    if not np.any(good):
        return None
    bd = 1.0 - rc[good] / rcont[good]
    return float(np.mean(bd)) if bd.size else None


def _band_depth_map(cube, waves_um, center, sh1, sh2):
    from step0_load import find_nearest_band

    if center < waves_um[0] or center > waves_um[-1]:
        return None
    bc = find_nearest_band(waves_um, center)
    bs1 = find_nearest_band(waves_um, sh1)
    bs2 = find_nearest_band(waves_um, sh2)
    frac = (center - sh1) / max(sh2 - sh1, 1e-8)
    rc = cube[:, :, bc].astype(np.float64)
    rs1 = cube[:, :, bs1].astype(np.float64)
    rs2 = cube[:, :, bs2].astype(np.float64)
    cont = rs1 + frac * (rs2 - rs1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - rc / np.where(cont > 1e-8, cont, np.nan)


def _sample_continuum_sam(cube_a, cube_b, waves_um, left_idx, right_idx,
                          lo, hi, rows, cols):
    w = waves_um[lo:hi + 1]
    a = cube_a[rows, cols, lo:hi + 1].astype(np.float64)
    b = cube_b[rows, cols, lo:hi + 1].astype(np.float64)
    al = cube_a[rows, cols, left_idx].astype(np.float64)
    ar = cube_a[rows, cols, right_idx].astype(np.float64)
    bl = cube_b[rows, cols, left_idx].astype(np.float64)
    br = cube_b[rows, cols, right_idx].astype(np.float64)
    wl = float(waves_um[left_idx])
    wr = float(waves_um[right_idx])
    frac = (w[None, :] - wl) / max(wr - wl, 1e-8)
    cont_a = al[:, None] + frac * (ar - al)[:, None]
    cont_b = bl[:, None] + frac * (br - bl)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        cr_a = a / np.where(cont_a > 1e-8, cont_a, np.nan)
        cr_b = b / np.where(cont_b > 1e-8, cont_b, np.nan)

    sams = []
    for sa, sb in zip(cr_a, cr_b):
        finite = np.isfinite(sa) & np.isfinite(sb)
        if finite.sum() < 3:
            continue
        af = sa[finite]
        bf = sb[finite]
        na = np.linalg.norm(af)
        nb = np.linalg.norm(bf)
        if na > 1e-10 and nb > 1e-10:
            sams.append(float(np.arccos(np.clip(np.dot(af, bf) / (na * nb), -1.0, 1.0))))
    return float(np.degrees(np.median(sams))) if sams else None


def guard_mineral_band_depths(cube_phys, cube_denoised, waves_um,
                              valid_pixel_mask, valid_bands, snr_after_ml,
                              snr_after_physics, blend_weights):
    """
    Preserve continuum-relative mineral absorptions after ML denoising.

    SNR-only selection can flatten or invert shallow absorption bands while still
    reducing local variance. For detected features, revert the full continuum
    interval when band-depth or continuum-removed shape changes exceed tolerance.
    """
    from step0_load import find_nearest_band

    rows_v, cols_v = np.where(valid_pixel_mask)
    if len(rows_v) == 0:
        return []

    rng = np.random.default_rng(42)
    idx = rng.choice(len(rows_v), size=min(6000, len(rows_v)), replace=False)
    rows_s = rows_v[idx]
    cols_s = cols_v[idx]
    valid_set = set(valid_bands)
    revert_bands = set()
    flagged = []

    for name, center, sh1, sh2 in MINERAL_BAND_DEFS:
        bd_map = _band_depth_map(cube_phys, waves_um, center, sh1, sh2)
        if bd_map is None:
            continue
        positive = bd_map[np.isfinite(bd_map) &
                          (bd_map >= MINERAL_GUARD_ROI_THR) &
                          (bd_map <= 0.60)]
        if positive.size >= 100:
            roi_thr = max(MINERAL_GUARD_ROI_THR,
                          float(np.percentile(positive, MINERAL_GUARD_ROI_PERCENTILE)))
            roi = valid_pixel_mask & np.isfinite(bd_map) & (bd_map >= roi_thr) & (bd_map <= 0.60)
            rows_r, cols_r = np.where(roi)
            if len(rows_r) >= 100:
                ridx = rng.choice(len(rows_r), size=min(6000, len(rows_r)), replace=False)
                rows_f = rows_r[ridx]
                cols_f = cols_r[ridx]
            else:
                rows_f = rows_s
                cols_f = cols_s
        else:
            rows_f = rows_s
            cols_f = cols_s

        bd_phys = _sample_band_depth(
            cube_phys, waves_um, center, sh1, sh2, rows_f, cols_f)
        bd_den = _sample_band_depth(
            cube_denoised, waves_um, center, sh1, sh2, rows_f, cols_f)
        if bd_phys is None or bd_den is None:
            continue
        if abs(bd_phys) < BAND_DEPTH_DETECT_THR:
            continue
        chg_pct = (bd_den - bd_phys) / abs(bd_phys) * 100.0

        band_ids = []
        for target in (center, sh1, sh2):
            if waves_um[0] <= target <= waves_um[-1]:
                b = find_nearest_band(waves_um, target)
                if b in valid_set:
                    band_ids.append(b)
        if not band_ids:
            continue
        lo = min(band_ids)
        hi = max(band_ids)
        cr_sam_deg = _sample_continuum_sam(
            cube_phys, cube_denoised, waves_um, band_ids[1], band_ids[2],
            lo, hi, rows_f, cols_f) if len(band_ids) == 3 else None

        shape_bad = cr_sam_deg is not None and cr_sam_deg > MINERAL_GUARD_MAX_SAM_DEG
        depth_bad = abs(chg_pct) > BAND_DEPTH_CHG_THR * 100.0
        if not (depth_bad or shape_bad):
            continue

        window = [b for b in range(lo, hi + 1) if b in valid_set]
        revert_bands.update(window)
        flagged.append((name, bd_phys, bd_den, chg_pct, cr_sam_deg, window))

    if not revert_bands:
        print("  [Blend] Mineral guard: all detected band depths preserved")
        return []

    bands = sorted(revert_bands)
    print(f"  [Blend] Mineral guard: reverting {len(bands)} continuum-window band(s) "
          f"to physics to preserve absorptions")
    for name, bd_phys, bd_den, chg_pct, cr_sam_deg, window in flagged:
        sam_txt = "--" if cr_sam_deg is None else f"{cr_sam_deg:.2f} deg"
        print(f"    {name}: {bd_phys:.4f} -> {bd_den:.4f} "
              f"({chg_pct:+.1f}%), CR-SAM={sam_txt}, bands={window[0]}-{window[-1]}")

    H = cube_denoised.shape[0]
    for i in range(0, H, 500):
        j = min(i + 500, H)
        for b in bands:
            cube_denoised[i:j, :, b] = cube_phys[i:j, :, b]
    for b in bands:
        snr_after_ml[b] = snr_after_physics[b]
        blend_weights[b] = 0.0
    return bands


def _mineral_guard_band_set(waves_um, valid_bands):
    from step0_load import find_nearest_band

    valid_set = set(valid_bands)
    bands = set()
    for _, center, sh1, sh2 in MINERAL_BAND_DEFS:
        for target in (center, sh1, sh2):
            if waves_um[0] <= target <= waves_um[-1]:
                b = find_nearest_band(waves_um, target)
                if b in valid_set:
                    bands.add(b)
    return bands


def improve_gaussian_bands(cube_denoised, waves_um, valid_bands,
                           severe_gaussian_bands):
    """
    Conservative spatial cleanup for bands already classified as severe
    Gaussian noise. Mineral index center/shoulder bands are excluded so the
    extra smoothing cannot weaken diagnostic absorptions.
    """
    if not severe_gaussian_bands:
        print("  [Gaussian] No severe Gaussian bands -- skipped")
        return []

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("  [Gaussian] scipy unavailable -- skipped")
        return []

    mineral_bands = _mineral_guard_band_set(waves_um, valid_bands)
    target_bands = [b for b in severe_gaussian_bands
                    if b in valid_bands and b not in mineral_bands]
    if not target_bands:
        print("  [Gaussian] All severe bands overlap mineral guard bands -- skipped")
        return []

    H = cube_denoised.shape[0]
    for b in target_bands:
        band = cube_denoised[:, :, b].astype(np.float32)
        nan_mask = ~np.isfinite(band)
        if np.all(nan_mask):
            continue
        fill = float(np.nanmedian(band))
        band_fill = np.where(nan_mask, fill, band)
        smooth = gaussian_filter(band_fill, sigma=GAUSSIAN_SPATIAL_SIGMA,
                                 mode="nearest")
        cleaned = ((1.0 - GAUSSIAN_SPATIAL_BLEND) * band_fill +
                   GAUSSIAN_SPATIAL_BLEND * smooth)
        cleaned[nan_mask] = np.nan
        cube_denoised[:, :, b] = cleaned.astype(np.float32)

    print(f"  [Gaussian] Spatial cleanup applied to {len(target_bands)} "
          f"severe band(s), skipped {len(severe_gaussian_bands)-len(target_bands)} mineral band(s)")
    return target_bands


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(dae_history, cnn_history, fname, out_path):
    from step0_load import plot_style
    plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if dae_history:
        axes[0].plot(dae_history, color="#58a6ff", label="DAE loss")
    axes[0].set_title("DAE Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if cnn_history:
        axes[1].plot(cnn_history, color="#3fb950", label="CNN loss")
    axes[1].set_title("CNN Training Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(f"Training Curves -- {fname}", color="#c9d1d9", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_snr_comparison(snr_raw, snr_phys, snr_ml, waves_um,
                        valid_bands, fname, out_path):
    from step0_load import plot_style
    plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    w  = waves_um
    vb = valid_bands
    axes[0].plot(w[vb], snr_raw[vb],  color="#58a6ff", label="Raw")
    axes[0].plot(w[vb], snr_phys[vb], color="#ffa657",
                 linestyle="--", label="Physics")
    axes[0].plot(w[vb], snr_ml[vb],   color="#3fb950",
                 linestyle=":",  label="ML Denoised")
    axes[0].axhline(10, color="#ff7b72", linewidth=0.7, linestyle="--")
    axes[0].axhline(30, color="#e3b341", linewidth=0.7, linestyle="--")
    axes[0].set_xlabel("Wavelength (um)")
    axes[0].set_ylabel("SNR")
    axes[0].set_title(f"SNR Comparison -- {fname}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    improve = np.zeros(len(w))
    for b in vb:
        if snr_raw[b] > 1e-6:
            improve[b] = (snr_ml[b] - snr_raw[b]) / snr_raw[b] * 100
    colors = ["#3fb950" if improve[b] >= 0 else "#ff7b72" for b in vb]
    axes[1].bar(w[vb], improve[vb], color=colors,
                width=(w[1] - w[0]) * 0.8)
    axes[1].axhline(0, color="#c9d1d9", linewidth=0.7)
    axes[1].set_xlabel("Wavelength (um)")
    axes[1].set_ylabel("SNR Improvement %")
    axes[1].set_title("Per-Band SNR Improvement (ML vs Raw)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_blend_weights(blend_weights, waves_um, valid_bands, fname, out_path):
    from step0_load import plot_style
    plot_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    w  = waves_um
    vb = valid_bands
    ax.bar(w[vb], blend_weights[vb],
           width=(w[1] - w[0]) * 0.8, color="#58a6ff", alpha=0.8)
    ax.axhline(BLEND_MIN, color="#ff7b72", linestyle="--",
               linewidth=0.9, label=f"Min={BLEND_MIN}")
    ax.axhline(BLEND_MAX, color="#3fb950", linestyle="--",
               linewidth=0.9, label=f"Max={BLEND_MAX}")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Blend weight alpha")
    ax.set_title(f"Adaptive Blend Weights -- {fname}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_spectral_comparison(plot_pixels, plot_data_raw, plot_data_phys, plot_data_den, 
                             waves_um, fname, out_path):
    from step0_load import plot_style, THERMAL_ONSET_UM
    plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    w    = waves_um
    for i, (r, c) in enumerate(plot_pixels):
        raw  = plot_data_raw[i]
        phys = plot_data_phys[i]
        den  = plot_data_den[i]
        ax   = axes[i]
        mr = np.isfinite(raw)
        mp = np.isfinite(phys)
        md = np.isfinite(den)
        ax.plot(w[mr], raw[mr],   color="#ff7b72", lw=1.0, label="Raw")
        ax.plot(w[mp], phys[mp],  color="#ffa657", lw=1.0,
                linestyle="--", label="Physics")
        ax.plot(w[md], den[md],   color="#3fb950", lw=1.2,
                linestyle=":",  label="Denoised")
        ax.axvspan(THERMAL_ONSET_UM, w[-1] + 0.05, alpha=0.08, color="#ff7b72")
        ax.set_xlabel("um")
        ax.set_ylabel("I/F")
        ax.set_title(f"Pixel ({r},{c})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Spectral Comparison -- {fname}", color="#c9d1d9", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Multi-scene spectra collector (called by run_pipeline.py)
# ---------------------------------------------------------------------------

def collect_spectra_from_scene(scene_dir, n_spectra=1500):
    """
    Lightweight collector: load a scene, sample n_spectra random spectra.
    Returns (spectra np.array float32, B) or (None, None) on failure.
    Used by run_pipeline.py to aggregate training data across all 10 scenes.
    """
    try:
        from pathlib import Path
        sd = Path(scene_dir)
        phys_dir = sd.parent / "Pipeline_outputs" / "03_physics"
        npy_files = list(phys_dir.glob("*_physics_corrected.npy"))
        if not npy_files:
            print(f"  [MultiScene] No physics cube found in {phys_dir.name}. Run Phase A first.")
            return None, None
            
        cube = np.load(str(npy_files[0]))
        B    = cube.shape[2]
        mask = np.isfinite(cube).sum(axis=2) >= 20
        rows, cols = np.where(mask)
        if len(rows) == 0:
            del cube
            gc.collect()
            return None, None
        rng       = np.random.default_rng(42)
        idx       = rng.choice(len(rows),
                               size=min(n_spectra, len(rows)), replace=False)
        valid_pixels = cube[rows, cols, :]
        with np.errstate(invalid="ignore"):
            band_mean = np.nanmean(valid_pixels, axis=0).astype(np.float32)
        band_mean = np.nan_to_num(band_mean, nan=0.0)
        spectra   = []
        for i in idx:
            r, c = int(rows[i]), int(cols[i])
            spec  = cube[r, c, :].astype(np.float32)
            if np.isnan(spec).mean() > 0.5:
                continue
            spec = np.where(np.isnan(spec), band_mean, spec)
            spectra.append(spec)
        del cube
        gc.collect()
        if not spectra:
            return None, None
        return np.array(spectra, dtype=np.float32), B
    except Exception as e:
        print(f"  [MultiScene] Warning loading {scene_dir}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_denoise(cube, cube_phys, waves_um, fname, DIRS,
                band_stats, valid_bands, dead_bands,
                valid_pixel_mask, striped_bands, stripe_col_offsets,
                SNR_AFTER_PHYSICS, severe_gaussian_bands=None,
                extra_spectra=None):
    """
    Run full Step 4 ML denoising.

    Parameters
    ----------
    extra_spectra : np.ndarray, optional
        Pre-collected spectra from other scenes (shape N x B).
        When provided, combined with this scene's spectra for DAE training.
    """
    import torch

    print("\n" + "=" * 60)
    print("  STEP 4 -- ML-Based Denoising")
    print("=" * 60)

    device = _get_device()
    print(f"  Device: {device}")

    B = cube_phys.shape[2]
    H = cube_phys.shape[0]

    model_dae_path = str(Path(DIRS["models"]) / f"m3_dae_B{B}.pth")
    model_cnn_path = str(Path(DIRS["models"]) / f"m3_cnn_H{H}.pth")

    # Extract 4 pixels for plotting NOW, and instantly delete raw `cube` to free 2.4 GB RAM
    rng = np.random.default_rng(42)
    rows_v, cols_v = np.where(valid_pixel_mask)
    plot_idx = rng.choice(len(rows_v), size=min(4, len(rows_v)), replace=False)
    plot_pixels = [(int(rows_v[i]), int(cols_v[i])) for i in plot_idx]
    plot_data_raw = [cube[r, c, :].astype(np.float64) for r, c in plot_pixels]
    
    del cube
    gc.collect()

    # SPM indices
    print("  Computing SPM indices ...")
    spm = compute_spm_indices_vectorized(cube_phys, waves_um, valid_pixel_mask)

    # Training data
    print("  Collecting stratified training spectra ...")
    spectra_local = collect_training_spectra(
        cube_phys, waves_um, valid_pixel_mask, spm,
        n_target=N_TRAIN_SPECTRA)

    if extra_spectra is not None and extra_spectra.shape[1] == B:
        spectra_all = np.vstack([spectra_local, extra_spectra])
        print(f"  Multi-scene pool: {spectra_all.shape[0]} total spectra")
    else:
        spectra_all = spectra_local

    # Remove any rows with NaN/Inf before training (guards against all-NaN spectra)
    valid_rows = np.all(np.isfinite(spectra_all), axis=1)
    n_removed  = int((~valid_rows).sum())
    if n_removed > 0:
        print(f"  Removed {n_removed} invalid spectra ({n_removed/len(spectra_all)*100:.1f}%)")
    spectra_all = spectra_all[valid_rows]
    if len(spectra_all) < 50:
        print(f"  WARNING: Only {len(spectra_all)} valid training spectra -- falling back to physics cube")
        SNR_AFTER_ML = SNR_AFTER_PHYSICS.copy()
        npy_path = str(Path(DIRS["04_denoised"]) / f"{fname}_denoised.npy")
        np.save(npy_path, cube_phys)
        return cube_phys.copy(), SNR_AFTER_ML, np.zeros(B, dtype=np.float32)

    rng = np.random.default_rng(42)
    spectra_all = spectra_all[rng.permutation(len(spectra_all))]

    # Train / load DAE
    dae_model, dae_history = train_dae(
        spectra_all, valid_bands, B, device, model_dae_path)

    # DAE inference
    print("  DAE inference ...")
    cube_dae = infer_dae(dae_model, cube_phys, valid_pixel_mask,
                         valid_bands, dead_bands, device)
    del dae_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Train / load CNN
    cnn_model, cnn_history = train_cnn(
        cube_phys, valid_bands, dead_bands,
        stripe_col_offsets, H, device, model_cnn_path)

    # CNN inference
    print("  CNN inference ...")
    dict_cnn = infer_cnn(cnn_model, cube_phys, striped_bands, device)
    if cnn_model is not None:
        del cnn_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Adaptive blend
    print("  Adaptive blending ...")
    snr_raw = band_stats["snr"]
    cube_denoised, BLEND_WEIGHTS = adaptive_blend(
        cube_phys, cube_dae, dict_cnn, waves_um,
        valid_bands, striped_bands, SNR_AFTER_PHYSICS, valid_pixel_mask)
    del cube_dae
    gc.collect()

    improve_gaussian_bands(
        cube_denoised, waves_um, valid_bands, severe_gaussian_bands or [])

    # Post SNR
    SNR_AFTER_ML = compute_snr_ml(cube_denoised, valid_bands)
    if valid_bands:
        improvement = [
            (SNR_AFTER_ML[b] - snr_raw[b]) / max(snr_raw[b], 1e-6) * 100
            for b in valid_bands]
        print(f"  Mean SNR improvement (pre-guard): {float(np.mean(improvement)):.1f}%")

    # ── Do-no-harm: ML output must be ≥ physics SNR for every band ──────────
    # For any band where ML degraded SNR below the physics baseline,
    # revert that band to the physics cube (chunk-based to avoid thrashing).
    bad_bands = [b for b in valid_bands
                 if SNR_AFTER_ML[b] < SNR_AFTER_PHYSICS[b] * 0.99]
    if bad_bands:
        print(f"  [Blend] Do-no-harm: reverting {len(bad_bands)} band(s) "
              f"to physics (indices: {bad_bands})", flush=True)
        _H = cube_denoised.shape[0]
        for _i in range(0, _H, 500):
            _j = min(_i + 500, _H)
            for _b in bad_bands:
                cube_denoised[_i:_j, :, _b] = cube_phys[_i:_j, :, _b]
        for _b in bad_bands:
            SNR_AFTER_ML[_b] = SNR_AFTER_PHYSICS[_b]
        if valid_bands:
            improvement = [
                (SNR_AFTER_ML[b] - snr_raw[b]) / max(snr_raw[b], 1e-6) * 100
                for b in valid_bands]
            print(f"  Mean SNR improvement (post-guard): "
                  f"{float(np.mean(improvement)):.1f}%")
    else:
        print(f"  [Blend] Do-no-harm: all {len(valid_bands)} bands improved ✓")



    guard_bands = guard_mineral_band_depths(
        cube_phys, cube_denoised, waves_um, valid_pixel_mask, valid_bands,
        SNR_AFTER_ML, SNR_AFTER_PHYSICS, BLEND_WEIGHTS)
    if guard_bands and valid_bands:
        improvement = [
            (SNR_AFTER_ML[b] - snr_raw[b]) / max(snr_raw[b], 1e-6) * 100
            for b in valid_bands]
        print(f"  Mean SNR improvement (post-mineral-guard): "
              f"{float(np.mean(improvement)):.1f}%")

    # Plots
    out = DIRS["04_denoised"]
    plot_training_curves(
        dae_history, cnn_history, fname,
        str(Path(out) / f"{fname}_training_curves.png"))
    plot_snr_comparison(
        snr_raw, SNR_AFTER_PHYSICS, SNR_AFTER_ML, waves_um, valid_bands,
        fname, str(Path(out) / f"{fname}_snr_comparison.png"))
    plot_blend_weights(
        BLEND_WEIGHTS, waves_um, valid_bands, fname,
        str(Path(out) / f"{fname}_blend_weights.png"))
        
    plot_data_phys = [cube_phys[r, c, :].astype(np.float64) for r, c in plot_pixels]
    plot_data_den = [cube_denoised[r, c, :].astype(np.float64) for r, c in plot_pixels]
    plot_spectral_comparison(
        plot_pixels, plot_data_raw, plot_data_phys, plot_data_den, waves_um,
        fname, str(Path(out) / f"{fname}_spectral_comparison.png"))
    
    # We can now delete cube_phys
    del cube_phys
    gc.collect()

    # Save denoised cube
    npy_path = str(Path(out) / f"{fname}_denoised.npy")
    np.save(npy_path, cube_denoised)
    print(f"  Saved denoised cube: {npy_path}")

    print("\n" + "=" * 60)
    print("  STEP 4 SUMMARY")
    print("=" * 60)
    if valid_bands:
        vb = valid_bands
        print(f"  SNR raw     : {float(np.mean(snr_raw[vb])):.3f}")
        print(f"  SNR physics : {float(np.mean(SNR_AFTER_PHYSICS[vb])):.3f}")
        print(f"  SNR ML      : {float(np.mean(SNR_AFTER_ML[vb])):.3f}")
    print(f"  Denoised    : {npy_path}")
    print("=" * 60)
    print("  Step 4 complete. Run step5_validate.py next.")
    print("=" * 60 + "\n")

    return cube_denoised, SNR_AFTER_ML, BLEND_WEIGHTS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import step0_load as s0
    import step1_eda  as s1
    import step2_noise as s2
    import step3_physics as s3

    sd = (sys.argv[1] if len(sys.argv) > 1
          else r"D:\M3 Journal\Mare Tranquillitatis\M3G20090203T160452_V01_RFL")

    cube, waves, fname, meta, DIRS = s0.load_scene(sd)
    (EDA, vpm, bs, vb, db, tb, sb) = s1.run_eda(cube, waves, fname, DIRS)
    (NM, sco, sfrac, stb, spb, sg,
     td, smd, spc) = s2.run_noise_id(
        cube, waves, fname, DIRS, bs, vb, db, tb, vpm)
    cp, snrp = s3.run_physics(
        cube, waves, fname, DIRS, bs, vb, db, tb, vpm,
        sfrac, stb, spb, td, smd, spc)
    run_denoise(cube, cp, waves, fname, DIRS,
                bs, vb, db, vpm, stb, sco, snrp)
