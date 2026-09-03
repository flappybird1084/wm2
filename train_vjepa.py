"""Train a V-JEPA-style video model (port of test-1.ipynb, training parts only).

Run: python train_vjepa.py
A checkpoint (weights + config + losses) is written to MODEL_PATH at the end.
"""
import os
import random

import matplotlib
matplotlib.use("Agg")  # headless: write plot to file, no display
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
MODEL_PATH = "./checkpoints/vjepa.pth"
LOSS_PLOT_PATH = "vjepa_loss.png"

SEQ_LEN = 256            # spatial patches per frame (must be a perfect square)
BASE_EMB = 768          # per-frame patch embedding dim (becomes BASE_EMB*FRAMES_PER_TUBE)
FRAMES_PER_TUBE = 2     # frames grouped into one spatiotemporal tube
TARGET_FRAMES = 30 // FRAMES_PER_TUBE   # number of tubes (= 15)
RAW_FRAMES = TARGET_FRAMES * FRAMES_PER_TUBE  # raw frames fed to the dataset (= 30)
MASK_RATIO = 0.8        # fraction of spatial tokens masked

BATCH_SIZE = 2
NUM_WORKERS = 4
LR = 1e-4
STEPS = 45000
LOSS_EVERY = 100
EMA_BETA = 0.999

MASKED_TOKENS = int(MASK_RATIO * SEQ_LEN)


# ----------------------------------------------------------------------------
# Patch / tube (un)folding utilities
# ----------------------------------------------------------------------------
def patchify(tensor, seq_len):
    # takes tensor of b,c,h,w
    # returns b,s,c tensor
    B, C, H, W = tensor.shape
    assert H == W  # square patch
    assert int(seq_len ** 0.5) ** 2 == seq_len
    assert (H * W) % seq_len == 0

    scaling_factor = int(H / (seq_len ** 0.5))
    tensor = tensor.reshape(B, C, H // scaling_factor, scaling_factor, W // scaling_factor, scaling_factor)
    tensor = tensor.permute(0, 2, 4, 1, 3, 5)
    tensor = tensor.reshape(B, H // scaling_factor * W // scaling_factor, C * scaling_factor * scaling_factor)
    return tensor


def unpatchify(tensor, height):
    # assume h,w is the same
    width = height
    B, S, C = tensor.shape
    scaling_factor = int(height / (S ** 0.5))
    image_c = C // scaling_factor // scaling_factor
    tensor = tensor.reshape(B, height // scaling_factor, width // scaling_factor, image_c, scaling_factor, scaling_factor)
    tensor = tensor.permute(0, 3, 1, 4, 2, 5)
    tensor = tensor.reshape(B, image_c, height, width)
    return tensor


def tubeify(tensor, seq_len, frames_per_tube=2):
    # takes tensor of b,t,c,h,w
    # returns b,t_shortened,s,c tensor
    B, T, C, H, W = tensor.shape
    t_shortened = T // frames_per_tube
    assert H == W  # square patch
    assert int(seq_len ** 0.5) ** 2 == seq_len
    assert (H * W) % seq_len == 0

    scaling_factor = int(H / (seq_len ** 0.5))
    tensor = tensor.reshape(B, t_shortened, frames_per_tube, C, H // scaling_factor, scaling_factor, W // scaling_factor, scaling_factor)
    tensor = tensor.permute(0, 1, 4, 6, 3, 2, 5, 7)
    tensor = tensor.reshape(B, t_shortened, H // scaling_factor * W // scaling_factor, C * frames_per_tube * scaling_factor * scaling_factor)
    return tensor


def untubeify(tensor, height, frames_per_tube=2):
    # assume h,w is the same
    width = height
    B, t_shortened, S, C = tensor.shape
    scaling_factor = int(height / (S ** 0.5))
    image_c = C // scaling_factor // scaling_factor // frames_per_tube
    tensor = tensor.reshape(B, t_shortened, height // scaling_factor, width // scaling_factor, image_c, frames_per_tube, scaling_factor, scaling_factor)
    tensor = tensor.permute(0, 1, 5, 4, 2, 6, 3, 7)
    tensor = tensor.reshape(B, t_shortened * frames_per_tube, image_c, height, width)
    return tensor


# ----------------------------------------------------------------------------
# Dataset
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

        # (t, s, c) -> (t*s, c)
        frame_tensors = frame_tensors.reshape(t * s, c)

        total = t * s
        target_len = self.frames * self.seq_len

        if total < target_len:
            pad_len = target_len - total
            frame_tensors = F.pad(frame_tensors, (0, 0, pad_len, 0))  # front-pad seq dim
        else:
            start = random.randint(0, int(total - target_len))
            frame_tensors = frame_tensors[start:start + target_len]

        # (target_len, c) -> (frames, seq_len, c)
        frame_tensors = frame_tensors.reshape(self.frames, self.seq_len, c)
        frame_tensors = unpatchify(frame_tensors, frame_height)
        frame_tensors = tubeify(frame_tensors.unsqueeze(0), seq_len=self.seq_len, frames_per_tube=self.frames_per_tube).squeeze(0)
        return frame_tensors


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, nhead=4, seq=TARGET_FRAMES * SEQ_LEN, emb=768):
        super().__init__()
        self.seq = seq
        self.emb = emb
        self.nhead = nhead
        assert emb % nhead == 0

        self.ln1 = nn.LayerNorm(emb)
        self.qkv_proj = nn.Linear(emb, emb * 3)
        self.o_proj = nn.Linear(emb, emb)

        self.ln2 = nn.LayerNorm(emb)
        self.mlp = nn.Sequential(
            nn.Linear(emb, emb * 4),
            nn.GELU(),
            nn.Linear(emb * 4, emb),
        )

    def forward(self, x):
        b, s, c = x.shape
        qkv = self.qkv_proj(self.ln1(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, s, self.nhead, c // self.nhead).transpose(2, 1)
        k = k.view(b, s, self.nhead, c // self.nhead).transpose(2, 1)
        v = v.view(b, s, self.nhead, c // self.nhead).transpose(2, 1)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(2, 1).contiguous().reshape(b, s, c)
        attn_out = self.o_proj(attn_out)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, nlayer=4, nhead=4, seq=TARGET_FRAMES * SEQ_LEN, emb=768):
        super().__init__()
        self.blocks = nn.ModuleList([Block(nhead, seq, emb) for _ in range(nlayer)])
        self.ln_out = nn.LayerNorm(emb)

    def forward(self, x):
        b, t, s, c = x.shape
        x = x.reshape(b, t * s, c)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return x.reshape(b, t, s, c)


class Predictor(nn.Module):
    def __init__(self, nlayer=4, nhead=4, seq=TARGET_FRAMES * SEQ_LEN, emb=768, dim=384):
        super().__init__()
        self.blocks = nn.ModuleList([Block(nhead, seq, dim) for _ in range(nlayer)])  # downcast width
        self.down = nn.Linear(emb, dim)
        self.up = nn.Linear(dim, emb)
        self.ln_out = nn.LayerNorm(emb)

    def forward(self, x):
        b, t, s, c = x.shape
        x = x.reshape(b, t * s, c)
        x = self.down(x)
        for block in self.blocks:
            x = block(x)
        x = self.up(x)
        x = self.ln_out(x)
        return x.reshape(b, t, s, c)


class InformedPredictor(nn.Module):
    def __init__(self, seq=SEQ_LEN, emb=768, masked_tokens=10, frames=TARGET_FRAMES, frames_per_tube=FRAMES_PER_TUBE):
        super().__init__()
        emb = emb * frames_per_tube  # tube tokens stack frames_per_tube frames

        self.encoder = Encoder(seq=seq, emb=emb)
        self.target_encoder = Encoder(seq=seq, emb=emb)
        self.predictor = Predictor(seq=seq, emb=emb)
        assert masked_tokens <= seq
        self.seq = seq
        self.emb = emb
        self.masked_tokens = masked_tokens
        self.embed_mlp = nn.Sequential(
            nn.LayerNorm(emb),
            nn.Linear(emb, emb),
        )
        self.frames = frames
        self.embed = nn.Embedding(seq, emb)
        self.frame_embed = nn.Embedding(frames, emb)
        nn.init.trunc_normal_(self.embed.weight, std=.02)
        self.register_buffer("pos_idx", torch.arange(0, seq))
        self.register_buffer("frame_idx", torch.arange(0, frames))
        self.mask_token_embedding = nn.Parameter(torch.randn(1, emb))

        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        b, t, s, c = x.shape
        x = self.embed_mlp(x)
        x = x + self.embed(self.pos_idx)
        x = x + self.frame_embed(self.frame_idx).unsqueeze(1)
        x_orig = x.clone()

        all_perms = torch.randperm(self.seq, device=x.device)
        inv_masked_token_idx = all_perms[:self.seq - self.masked_tokens]
        filtered_x = x[:, :, inv_masked_token_idx, :]

        encoded_x = self.encoder(filtered_x)  # b,t,seq-mask,c
        mask = torch.isin(self.pos_idx, inv_masked_token_idx)
        mask_token_idx = self.pos_idx[~mask]
        mask_tokens = self.embed(mask_token_idx).unsqueeze(0).expand(b, -1, -1, -1)
        mask_tokens = mask_tokens + self.mask_token_embedding.unsqueeze(0).expand(b, t, self.masked_tokens, -1)

        predicted_x = self.predictor(torch.cat([encoded_x, mask_tokens], dim=-2))
        with torch.no_grad():
            target_x = self.target_encoder(x_orig)

        mask_pred = predicted_x[:, :, self.seq - self.masked_tokens:, :]
        target_pred = target_x[:, :, mask_token_idx, :]
        return mask_pred, target_pred


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ds = VideoDataset(DATA_ROOT, transform, seq_len=SEQ_LEN, frames=RAW_FRAMES, frames_per_tube=FRAMES_PER_TUBE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = InformedPredictor(
        seq=SEQ_LEN, emb=BASE_EMB, masked_tokens=MASKED_TOKENS,
        frames=TARGET_FRAMES, frames_per_tube=FRAMES_PER_TUBE,
    ).to(device)
    compiled_model = torch.compile(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    losses = []

    dl_iter = iter(dl)
    pbar = tqdm(total=STEPS)
    for step in range(STEPS):
        try:
            data = next(dl_iter).to(device)
        except StopIteration:
            dl_iter = iter(dl)
            data = next(dl_iter).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            mask_pred, target_pred = compiled_model(data)
            loss = loss_fn(mask_pred, target_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.update(1)
        pbar.set_postfix(loss=loss.item())

        if step % LOSS_EVERY == 0:
            losses.append(loss.item())

        # EMA update of the target encoder toward the online encoder
        for param, reference in zip(model.target_encoder.parameters(), model.encoder.parameters()):
            param.data.mul_(EMA_BETA).add_(reference.data, alpha=1 - EMA_BETA)
    pbar.close()

    config = {
        "seq_len": SEQ_LEN,
        "base_emb": BASE_EMB,
        "frames_per_tube": FRAMES_PER_TUBE,
        "target_frames": TARGET_FRAMES,
        "raw_frames": RAW_FRAMES,
        "mask_ratio": MASK_RATIO,
        "masked_tokens": MASKED_TOKENS,
        "lr": LR,
        "steps": STEPS,
        "ema_beta": EMA_BETA,
    }
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config, "losses": losses},
        MODEL_PATH,
    )
    print(f"Saved checkpoint to {MODEL_PATH}")

    save_loss_plot(losses, LOSS_PLOT_PATH)
    print(f"Saved loss plot to {LOSS_PLOT_PATH}")


def save_loss_plot(losses, path):
    losses_arr = np.array(losses)
    ema = np.zeros_like(losses_arr)
    beta, acc = 0.98, 0.0
    for i, l in enumerate(losses_arr):
        acc = beta * acc + (1 - beta) * l
        ema[i] = acc / (1 - beta ** (i + 1))  # bias-corrected EMA
    plt.figure()
    plt.plot(losses_arr, alpha=0.25, label="raw")
    plt.plot(ema, label="ema")
    plt.yscale("log")
    plt.xlabel(f"step / {LOSS_EVERY}")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
