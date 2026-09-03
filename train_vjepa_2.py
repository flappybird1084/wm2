"""Train a V-JEPA-style video model -- v2, with anti-collapse fixes.

The v1 checkpoint (train_vjepa.py) collapsed: the target encoder used an effective
rank of ~3/1536 and the predictor learned to output the mean embedding for every
token (in-distribution mse_pred / predict-mean ~= 1.0). Diagnosis lived in
scratchpad/diag.py. This script fixes the three root causes:

  1. VICReg variance + covariance regularization on the online representations
     (encoder output + predictor output). The variance hinge forces every feature
     dimension to keep std >= 1 across the batch (kills dimensional collapse); the
     covariance term decorrelates dimensions. This is the actual cure -- the EMA
     target + predictor alone were not enough.

  2. Tube / temporal masking. Instead of always masking the same spatial patches in
     every frame, we now also mask WHOLE trailing tubes ("give the first K tubes,
     predict the rest") -- harder, fewer shortcuts, and it makes the notebook's
     temporal-inference task in-distribution. Mask tokens now also carry the frame
     positional embedding (v1 omitted it), so the predictor knows which tube it is
     filling in.

  3. EMA momentum ramp (0.99 -> 0.9999) + LR warmup/cosine decay for stability.

A live collapse monitor prints the online representation's effective rank and
per-dim std every EVAL_EVERY steps -- if effective rank stays in the tens/hundreds
and var loss stays low, collapse is being avoided.

Run: python train_vjepa_2.py
Checkpoint (weights + config + loss components) is written to MODEL_PATH at the end.
The saved state_dict is name-compatible with the notebook's InformedPredictor.
"""
import math
import os
import random

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Hyperparameters & paths
# ----------------------------------------------------------------------------
DATA_ROOT = "./data/20bn-something-something-v2"
MODEL_PATH = "./checkpoints/vjepa_2.pth"
LOSS_PLOT_PATH = "vjepa_2_loss.png"

SEQ_LEN = 256            # spatial patches per frame (perfect square)
BASE_EMB = 768          # per-frame patch embedding dim (becomes BASE_EMB*FRAMES_PER_TUBE)
FRAMES_PER_TUBE = 2     # frames grouped into one spatiotemporal tube
TARGET_FRAMES = 30 // FRAMES_PER_TUBE   # number of tubes (= 15)
RAW_FRAMES = TARGET_FRAMES * FRAMES_PER_TUBE  # raw frames fed to the dataset (= 30)

BATCH_SIZE = 2
NUM_WORKERS = 6
LR = 1e-4
WEIGHT_DECAY = 0.04
STEPS = 45000
WARMUP = 1000
LOSS_EVERY = 100
EVAL_EVERY = 1000       # how often to log the collapse monitor (effective rank)

# EMA momentum ramp for the target encoder
EMA_BASE = 0.99
EMA_FINAL = 0.9999

# ---- masking ----
# Each step picks spatial-block masking or temporal (tube) masking.
P_TEMPORAL = 0.5                 # probability of using temporal masking this step
SPATIAL_KEEP_RATIO = 0.2         # spatial mode: keep this fraction of spatial columns
TEMPORAL_VISIBLE_CHOICES = (3, 5, 7)   # temporal mode: # of leading tubes given as context

# ---- VICReg coefficients (invariance / variance / covariance) ----
SIM_COEF = 25.0   # weight on the JEPA MSE (predictor vs EMA target)
VAR_COEF = 25.0   # weight on the variance hinge (anti dimensional-collapse)
COV_COEF = 1.0    # weight on the covariance penalty (decorrelation)
VAR_GAMMA = 1.0   # target per-dim std

USE_COMPILE = True
# We feed a few distinct mask shapes (spatial + temporal/{3,5,7}). Let dynamo
# specialize each STATICALLY -- if it tries symbolic/dynamic shapes instead,
# inductor codegen dies with "CantSplit". cache_size_limit must exceed the
# number of distinct shapes so they all stay compiled.
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.automatic_dynamic_shapes = False


# ----------------------------------------------------------------------------
# Patch / tube (un)folding utilities  (unchanged from v1)
# ----------------------------------------------------------------------------
def patchify(tensor, seq_len):
    B, C, H, W = tensor.shape
    assert H == W
    assert int(seq_len ** 0.5) ** 2 == seq_len
    assert (H * W) % seq_len == 0
    scaling_factor = int(H / (seq_len ** 0.5))
    tensor = tensor.reshape(B, C, H // scaling_factor, scaling_factor, W // scaling_factor, scaling_factor)
    tensor = tensor.permute(0, 2, 4, 1, 3, 5)
    tensor = tensor.reshape(B, H // scaling_factor * W // scaling_factor, C * scaling_factor * scaling_factor)
    return tensor


def unpatchify(tensor, height):
    width = height
    B, S, C = tensor.shape
    scaling_factor = int(height / (S ** 0.5))
    image_c = C // scaling_factor // scaling_factor
    tensor = tensor.reshape(B, height // scaling_factor, width // scaling_factor, image_c, scaling_factor, scaling_factor)
    tensor = tensor.permute(0, 3, 1, 4, 2, 5)
    tensor = tensor.reshape(B, image_c, height, width)
    return tensor


def tubeify(tensor, seq_len, frames_per_tube=2):
    B, T, C, H, W = tensor.shape
    t_shortened = T // frames_per_tube
    assert H == W
    assert int(seq_len ** 0.5) ** 2 == seq_len
    assert (H * W) % seq_len == 0
    scaling_factor = int(H / (seq_len ** 0.5))
    tensor = tensor.reshape(B, t_shortened, frames_per_tube, C, H // scaling_factor, scaling_factor, W // scaling_factor, scaling_factor)
    tensor = tensor.permute(0, 1, 4, 6, 3, 2, 5, 7)
    tensor = tensor.reshape(B, t_shortened, H // scaling_factor * W // scaling_factor, C * frames_per_tube * scaling_factor * scaling_factor)
    return tensor


def untubeify(tensor, height, frames_per_tube=2):
    width = height
    B, t_shortened, S, C = tensor.shape
    scaling_factor = int(height / (S ** 0.5))
    image_c = C // scaling_factor // scaling_factor // frames_per_tube
    tensor = tensor.reshape(B, t_shortened, height // scaling_factor, width // scaling_factor, image_c, frames_per_tube, scaling_factor, scaling_factor)
    tensor = tensor.permute(0, 1, 5, 4, 2, 6, 3, 7)
    tensor = tensor.reshape(B, t_shortened * frames_per_tube, image_c, height, width)
    return tensor


# ----------------------------------------------------------------------------
# Dataset  (unchanged from v1)
# ----------------------------------------------------------------------------
class VideoDataset(Dataset):
    def __init__(self, root, transform=None, seq_len=256, frames=30, frames_per_tube=2):
        self.data_root = root
        self.frames = frames
        self.transform = transform
        self.seq_len = seq_len
        self.frames_per_tube = frames_per_tube

    def __len__(self):
        return sum(1 for entry in os.scandir(self.data_root) if entry.is_file())

    def __getitem__(self, idx):
        frame_tensors = VideoDecoder(
            os.path.join(self.data_root, f'{idx + 1}.webm')
        )[:].to(torch.float32) / 255.0
        frame_tensors = self.transform(frame_tensors)
        frame_height = frame_tensors.size(2)
        frame_tensors = patchify(frame_tensors, self.seq_len)  # (t, s, c)

        t, s, c = frame_tensors.shape
        assert s == self.seq_len, f"patch dim {s} != seq_len {self.seq_len}"
        frame_tensors = frame_tensors.reshape(t * s, c)

        total = t * s
        target_len = self.frames * self.seq_len
        if total < target_len:
            pad_len = target_len - total
            frame_tensors = F.pad(frame_tensors, (0, 0, pad_len, 0))
        else:
            # crop on a FRAME boundary: total-target_len is a multiple of seq_len, but
            # a raw randint start straddles frames (bottom of frame k + top of k+1),
            # producing scrambled mosaic frames. Snap start to a multiple of seq_len.
            max_start_frame = (total - target_len) // self.seq_len
            start = random.randint(0, max_start_frame) * self.seq_len
            frame_tensors = frame_tensors[start:start + target_len]

        frame_tensors = frame_tensors.reshape(self.frames, self.seq_len, c)
        frame_tensors = unpatchify(frame_tensors, frame_height)
        frame_tensors = tubeify(frame_tensors.unsqueeze(0), seq_len=self.seq_len, frames_per_tube=self.frames_per_tube).squeeze(0)
        return frame_tensors


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, nhead=4, emb=768):
        super().__init__()
        self.nhead = nhead
        assert emb % nhead == 0
        self.ln1 = nn.LayerNorm(emb)
        self.qkv_proj = nn.Linear(emb, emb * 3)
        self.o_proj = nn.Linear(emb, emb)
        self.ln2 = nn.LayerNorm(emb)
        self.mlp = nn.Sequential(nn.Linear(emb, emb * 4), nn.GELU(), nn.Linear(emb * 4, emb))

    def forward(self, x):
        b, s, c = x.shape
        qkv = self.qkv_proj(self.ln1(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, s, self.nhead, c // self.nhead).transpose(2, 1)
        k = k.view(b, s, self.nhead, c // self.nhead).transpose(2, 1)
        v = v.view(b, s, self.nhead, c // self.nhead).transpose(2, 1)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(2, 1).contiguous().reshape(b, s, c)
        x = x + self.o_proj(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):
    # operates on a flat token sequence (b, n, c)
    def __init__(self, nlayer=4, nhead=4, emb=768):
        super().__init__()
        self.blocks = nn.ModuleList([Block(nhead, emb) for _ in range(nlayer)])
        self.ln_out = nn.LayerNorm(emb)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.ln_out(x)


class Predictor(nn.Module):
    # operates on a flat token sequence (b, n, c); downcasts width internally
    def __init__(self, nlayer=4, nhead=4, emb=768, dim=384):
        super().__init__()
        self.blocks = nn.ModuleList([Block(nhead, dim) for _ in range(nlayer)])
        self.down = nn.Linear(emb, dim)
        self.up = nn.Linear(dim, emb)
        self.ln_out = nn.LayerNorm(emb)

    def forward(self, x):
        x = self.down(x)
        for block in self.blocks:
            x = block(x)
        x = self.up(x)
        return self.ln_out(x)


class InformedPredictor(nn.Module):
    """I-JEPA over a flat (t*s) token grid with pluggable masking.

    forward(x, visible_idx, masked_idx) where the index tensors select tokens over
    the flattened grid (token id = tube*SEQ + spatial). Returns:
        mask_pred   : predictor output at the masked tokens   (b, n_masked, emb)
        target_pred : EMA target-encoder output at the same   (b, n_masked, emb)
        enc         : online encoder output of the context    (b, n_visible, emb)
    """
    def __init__(self, seq=SEQ_LEN, emb=BASE_EMB, frames=TARGET_FRAMES, frames_per_tube=FRAMES_PER_TUBE):
        super().__init__()
        emb = emb * frames_per_tube  # tube tokens stack frames_per_tube frames
        self.seq = seq
        self.emb = emb
        self.frames = frames

        self.encoder = Encoder(emb=emb)
        self.target_encoder = Encoder(emb=emb)
        self.predictor = Predictor(emb=emb)
        self.embed_mlp = nn.Sequential(nn.LayerNorm(emb), nn.Linear(emb, emb))
        self.embed = nn.Embedding(seq, emb)          # spatial position
        self.frame_embed = nn.Embedding(frames, emb)  # temporal position
        nn.init.trunc_normal_(self.embed.weight, std=.02)
        nn.init.trunc_normal_(self.frame_embed.weight, std=.02)
        self.register_buffer("pos_idx", torch.arange(0, seq))
        self.register_buffer("frame_idx", torch.arange(0, frames))
        self.mask_token_embedding = nn.Parameter(torch.randn(1, emb) * 0.02)

        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def full_pos(self):
        # (t*s, emb): token (t, s) -> frame_embed[t] + embed[s]
        spatial = self.embed(self.pos_idx)            # (S, emb)
        frame = self.frame_embed(self.frame_idx)      # (T, emb)
        return (frame[:, None, :] + spatial[None, :, :]).reshape(self.frames * self.seq, self.emb)

    def forward(self, x, visible_idx, masked_idx):
        b, t, s, c = x.shape
        pos = self.full_pos()
        x = self.embed_mlp(x).reshape(b, t * s, c) + pos

        with torch.no_grad():
            target = self.target_encoder(x)                       # (b, t*s, emb)

        ctx = x.index_select(1, visible_idx)                      # (b, n_vis, emb)
        enc = self.encoder(ctx)

        mask_tokens = self.mask_token_embedding + pos.index_select(0, masked_idx)   # (n_msk, emb)
        mask_tokens = mask_tokens.unsqueeze(0).expand(b, -1, -1)
        pred = self.predictor(torch.cat([enc, mask_tokens], dim=1))
        mask_pred = pred[:, visible_idx.shape[0]:]                # (b, n_msk, emb)
        target_pred = target.index_select(1, masked_idx)         # (b, n_msk, emb)
        return mask_pred, target_pred, enc


# ----------------------------------------------------------------------------
# Masking strategies -> (visible_idx, masked_idx) over the flat t*s grid
# ----------------------------------------------------------------------------
def spatial_mask(T, S, keep_ratio, device):
    n_keep = max(1, int(round(keep_ratio * S)))
    perm = torch.randperm(S, device=device)
    keep_cols, mask_cols = perm[:n_keep], perm[n_keep:]
    t_idx = torch.arange(T, device=device)[:, None]
    vis = (t_idx * S + keep_cols[None, :]).reshape(-1)
    msk = (t_idx * S + mask_cols[None, :]).reshape(-1)
    return vis, msk


def temporal_mask(T, S, n_visible_tubes, device):
    # give the first n_visible_tubes, predict the trailing ones
    s_idx = torch.arange(S, device=device)[None, :]
    vis_t = torch.arange(0, n_visible_tubes, device=device)[:, None]
    msk_t = torch.arange(n_visible_tubes, T, device=device)[:, None]
    vis = (vis_t * S + s_idx).reshape(-1)
    msk = (msk_t * S + s_idx).reshape(-1)
    return vis, msk


def sample_mask(T, S, device):
    if random.random() < P_TEMPORAL:
        k = random.choice(TEMPORAL_VISIBLE_CHOICES)
        return temporal_mask(T, S, k, device) + (f"temporal/{k}",)
    return spatial_mask(T, S, SPATIAL_KEEP_RATIO, device) + ("spatial",)


# ----------------------------------------------------------------------------
# VICReg terms + collapse monitor
# ----------------------------------------------------------------------------
def vicreg_terms(z, gamma=VAR_GAMMA, eps=1e-4):
    # z: (..., D) -> variance hinge + off-diagonal covariance penalty
    z = z.reshape(-1, z.shape[-1]).float()
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (z.shape[0] - 1)
    d = z.shape[1]
    cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d
    return var_loss, cov_loss


@torch.no_grad()
def effective_rank(z):
    z = z.reshape(-1, z.shape[-1]).float()
    z = z - z.mean(0)
    cov = (z.T @ z) / (z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    return (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12)).item()


def lr_at(step):
    if step < WARMUP:
        return LR * step / max(1, WARMUP)
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR * 0.5 * (1 + math.cos(math.pi * p))


def ema_at(step):
    p = step / max(1, STEPS)
    return EMA_FINAL - (EMA_FINAL - EMA_BASE) * 0.5 * (1 + math.cos(math.pi * p))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    T, S = TARGET_FRAMES, SEQ_LEN

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = VideoDataset(DATA_ROOT, transform, seq_len=SEQ_LEN, frames=RAW_FRAMES, frames_per_tube=FRAMES_PER_TUBE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=True)

    model = InformedPredictor(seq=SEQ_LEN, emb=BASE_EMB, frames=TARGET_FRAMES, frames_per_tube=FRAMES_PER_TUBE).to(device)
    run = torch.compile(model) if USE_COMPILE else model
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    hist = {"inv": [], "var": [], "cov": [], "total": [], "effrank": []}

    def _save(step_done):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        config = {
            "seq_len": SEQ_LEN, "base_emb": BASE_EMB, "frames_per_tube": FRAMES_PER_TUBE,
            "target_frames": TARGET_FRAMES, "raw_frames": RAW_FRAMES,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "steps": STEPS, "steps_done": step_done, "warmup": WARMUP,
            "ema_base": EMA_BASE, "ema_final": EMA_FINAL,
            "p_temporal": P_TEMPORAL, "spatial_keep_ratio": SPATIAL_KEEP_RATIO,
            "temporal_visible_choices": list(TEMPORAL_VISIBLE_CHOICES),
            "sim_coef": SIM_COEF, "var_coef": VAR_COEF, "cov_coef": COV_COEF, "var_gamma": VAR_GAMMA,
        }
        torch.save({"model_state_dict": model.state_dict(), "config": config, "hist": hist}, MODEL_PATH + ".tmp")
        os.replace(MODEL_PATH + ".tmp", MODEL_PATH)

    dl_iter = iter(dl)
    pbar = tqdm(total=STEPS)
    for step in range(STEPS):
        try:
            data = next(dl_iter).to(device)
        except StopIteration:
            dl_iter = iter(dl)
            data = next(dl_iter).to(device)

        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)

        vis, msk, _mode = sample_mask(T, S, device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            mask_pred, target_pred, enc = run(data, vis, msk)
            inv = F.mse_loss(mask_pred, target_pred)
            v_pred, c_pred = vicreg_terms(mask_pred)
            v_enc, c_enc = vicreg_terms(enc)
            var = 0.5 * (v_pred + v_enc)
            cov = 0.5 * (c_pred + c_enc)
            loss = SIM_COEF * inv + VAR_COEF * var + COV_COEF * cov

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # EMA update of the target encoder toward the online encoder
        beta = ema_at(step)
        with torch.no_grad():
            for p_t, p_o in zip(model.target_encoder.parameters(), model.encoder.parameters()):
                p_t.data.mul_(beta).add_(p_o.data, alpha=1 - beta)

        pbar.update(1)
        pbar.set_postfix(inv=f"{inv.item():.3f}", var=f"{var.item():.3f}", cov=f"{cov.item():.2f}", mode=_mode)

        if step % LOSS_EVERY == 0:
            hist["inv"].append(inv.item())
            hist["var"].append(var.item())
            hist["cov"].append(cov.item())
            hist["total"].append(loss.item())

        if step % EVAL_EVERY == 0:
            er = effective_rank(enc.detach())
            std = enc.detach().float().reshape(-1, enc.shape[-1]).std(0).mean().item()
            hist["effrank"].append((step, er))
            tqdm.write(f"[step {step}] effective_rank(enc)={er:.1f}/{model.emb}  mean per-dim std={std:.3f}  "
                       f"inv={inv.item():.3f} (collapse would be rank~3 / inv~=predict-mean)")

        if step > 0 and step % 5000 == 0:
            _save(step)
            tqdm.write(f"[step {step}] checkpoint saved -> {MODEL_PATH}")
    pbar.close()

    _save(STEPS)
    print(f"Saved checkpoint to {MODEL_PATH}")

    save_loss_plot(hist, LOSS_PLOT_PATH)
    print(f"Saved loss plot to {LOSS_PLOT_PATH}")


def save_loss_plot(hist, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(hist["inv"])) * LOSS_EVERY
    for k in ("inv", "var", "cov", "total"):
        ax1.plot(x, hist[k], label=k, alpha=0.8)
    ax1.set_yscale("log"); ax1.set_xlabel("step"); ax1.set_ylabel("loss component"); ax1.legend()
    ax1.set_title("loss components")
    if hist["effrank"]:
        ex, ey = zip(*hist["effrank"])
        ax2.plot(ex, ey, marker=".")
    ax2.set_xlabel("step"); ax2.set_ylabel("effective rank of enc")
    ax2.set_title("collapse monitor (higher = healthier; v1 sat at ~3)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
