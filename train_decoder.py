"""Train a pixel decoder for the (non-collapsed) V-JEPA-2 checkpoint.

The JEPA predicts in *representation* space. To SEE its predictions as RGB we
train a separate head that inverts the frozen target-encoder embedding back to
pixels:  target_encoder(embed(x)) -> x.  Then to GENERATE held-out tubes we swap
the encoder embedding of the future tubes for the *predictor's* guess and decode.

Imports the exact architecture from train_vjepa_2.py (the script that produced
checkpoints/vjepa_2.pth) so the state_dict loads with zero key mismatch.

Usage:
    python train_decoder.py --smoke          # tiny cache, few steps (sanity)
    python train_decoder.py                   # full run
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


def save_gif(path, frames_uint8, fps=6):
    imgs = [Image.fromarray(f) for f in frames_uint8]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, disposal=2)

import train_vjepa_2 as v2
from train_vjepa_2 import (
    InformedPredictor, VideoDataset, Block, untubeify, effective_rank,
    SEQ_LEN, BASE_EMB, FRAMES_PER_TUBE, TARGET_FRAMES, RAW_FRAMES, DATA_ROOT,
)

CKPT = "checkpoints/vjepa_2.pth"
DEC_CKPT = "checkpoints/decoder.pth"
S = SEQ_LEN                       # 256 spatial tokens / frame
T = TARGET_FRAMES                 # 15 tubes
EMB = BASE_EMB * FRAMES_PER_TUBE  # 1536


# ----------------------------------------------------------------------------
# JEPA embedding / inference helpers (mirror InformedPredictor.forward exactly)
# ----------------------------------------------------------------------------
@torch.no_grad()
def embed_tokens(model, x):
    """x: (b,t,s,c) raw tube tensor -> (b, t*s, emb) positional token embeddings."""
    b, t, s, c = x.shape
    pos = model.full_pos()                                   # (t*s, emb)
    return model.embed_mlp(x).reshape(b, t * s, c) + pos


@torch.no_grad()
def target_embed(model, x):
    """Frozen target-encoder embedding of the whole clip: (b, t*s, emb)."""
    return model.target_encoder(embed_tokens(model, x))


@torch.no_grad()
def predict_future(model, x, n_visible):
    """Give the first n_visible tubes, predict the trailing ones (temporal mask).

    Returns (full_pred_emb, target_full):
      target_full   : frozen target-encoder embedding of the whole clip (b, t*s, emb)
      full_pred_emb : same, but the hidden-tube tokens REPLACED by the predictor's
                      guess (visible tubes keep their true embedding). This is the
                      realistic generation setting: known past + predicted future.
    """
    b, t, s, c = x.shape
    pos = model.full_pos()
    emb_x = model.embed_mlp(x).reshape(b, t * s, c) + pos
    target_full = model.target_encoder(emb_x)

    n_vis = n_visible * S
    enc_vis = model.encoder(emb_x[:, :n_vis])                # online encoder, context only
    masked_idx = torch.arange(n_vis, t * s, device=x.device)
    mask_tokens = model.mask_token_embedding + pos.index_select(0, masked_idx)
    mask_tokens = mask_tokens.unsqueeze(0).expand(b, -1, -1)
    pred = model.predictor(torch.cat([enc_vis, mask_tokens], dim=1))   # (b, t*s, emb)
    mask_pred = pred[:, n_vis:]

    full_pred_emb = target_full.clone()
    full_pred_emb[:, n_vis:] = mask_pred
    return full_pred_emb, target_full, mask_pred, target_full[:, n_vis:]


# ----------------------------------------------------------------------------
# Decoder: transformer over the t*s token grid + learned positional embedding
# ----------------------------------------------------------------------------
class TubeDecoder(nn.Module):
    """Inverts a JEPA token embedding back to its pixel patch.

    Decodes per-tube: each token already carries global clip context (the encoder
    is full self-attention), so the decoder only needs the 256 spatial positions
    within a tube. Cheaper attention (256 vs t*s tokens) and far fewer positional
    params to learn, so it trains better from limited data.
    """
    def __init__(self, emb, c_pixel, seq=SEQ_LEN, nlayer=6, nhead=12):
        super().__init__()
        self.seq = seq
        self.in_ln = nn.LayerNorm(emb)
        self.pos = nn.Parameter(torch.zeros(1, seq, emb))   # spatial positions only
        nn.init.trunc_normal_(self.pos, std=.02)
        self.blocks = nn.ModuleList([Block(nhead=nhead, emb=emb) for _ in range(nlayer)])
        self.head = nn.Sequential(nn.LayerNorm(emb), nn.Linear(emb, c_pixel))

    def forward(self, z):                       # z: (b, t*s, emb)
        b, n, c = z.shape
        z = z.reshape(b * (n // self.seq), self.seq, c)
        z = self.in_ln(z) + self.pos
        for blk in self.blocks:
            z = blk(z)
        out = self.head(z)                      # (b*t, s, c_pixel)
        return out.reshape(b, n, -1)


# ----------------------------------------------------------------------------
def load_model(device):
    model = InformedPredictor(seq=SEQ_LEN, emb=BASE_EMB, frames=TARGET_FRAMES,
                              frames_per_tube=FRAMES_PER_TUBE).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_loader(batch_size, workers, shuffle):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = VideoDataset(DATA_ROOT, transform, seq_len=SEQ_LEN, frames=RAW_FRAMES,
                      frames_per_tube=FRAMES_PER_TUBE)
    return DataLoader(ds, batch_size=batch_size, num_workers=workers,
                      shuffle=shuffle, drop_last=True, persistent_workers=False,
                      prefetch_factor=4 if workers > 0 else None)


@torch.no_grad()
def cache_clips(model, dl, n_clips, device):
    """Decode videos ONCE; cache (target_embedding, pixels) as fp16 on CPU."""
    zc, pc = [], []
    it = iter(dl)
    pbar = tqdm(total=n_clips, desc="caching")
    while sum(z.shape[0] for z in zc) < n_clips:
        try:
            data = next(it)
        except StopIteration:
            it = iter(dl); data = next(it)
        data = data.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = target_embed(model, data)
        zc.append(z.half().cpu())
        pc.append(data.reshape(data.shape[0], T * S, EMB).half().cpu())
        pbar.update(data.shape[0])
    pbar.close()
    return torch.cat(zc)[:n_clips], torch.cat(pc)[:n_clips]


def diagnostics(model, dl, device, n_batches=8):
    """Confirm the checkpoint is non-collapsed at inference and probe the
    temporal-prediction quality vs trivial baselines (latent space)."""
    ers, mp, mm, mper, cs = [], [], [], [], []
    it = iter(dl)
    for _ in range(n_batches):
        data = next(it).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tgt = target_embed(model, data)
            _, _, mask_pred, target_pred = predict_future(model, data, n_visible=5)
        tgt, mask_pred, target_pred = tgt.float(), mask_pred.float(), target_pred.float()
        ers.append(effective_rank(tgt))
        mp.append(F.mse_loss(mask_pred, target_pred).item())
        mean = target_pred.mean(dim=1, keepdim=True).expand_as(target_pred)
        mm.append(F.mse_loss(mean, target_pred).item())
        cs.append(F.cosine_similarity(mask_pred, target_pred, dim=-1).mean().item())
    print(f"  effective_rank(target emb) = {np.mean(ers):.1f} / {EMB}   "
          f"(collapse would be ~3; v1 sat there)")
    print(f"  temporal predict  MSE = {np.mean(mp):.3f}   cosine = {np.mean(cs):+.3f}")
    print(f"  baseline predict-mean = {np.mean(mm):.3f}   "
          f"(pred < mean => model beats the trivial average)")
    return np.mean(ers)


def psnr(mse):  # pixels in [-1,1] -> peak-to-peak 2.0
    return 10 * math.log10(4.0 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--buf", type=int, default=512)       # replay-buffer size (clips)
    ap.add_argument("--refresh", type=int, default=8)     # rotate `bs` fresh clips every N steps
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--nlayer", type=int, default=6)
    args = ap.parse_args()
    if args.smoke:
        args.buf, args.steps, args.nlayer, args.refresh = 48, 600, 4, 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = load_model(device)

    print("== checkpoint diagnostics (is it really not collapsed?) ==")
    dl_probe = build_loader(batch_size=2, workers=6, shuffle=True)
    diagnostics(model, dl_probe, device, n_batches=4 if args.smoke else 8)
    del dl_probe

    # ---- fixed holdout set: decode once, NEVER trained on (true generalization) ----
    n_hold = 8
    hold_dl = build_loader(batch_size=2, workers=4, shuffle=True)
    z_hold, p_hold = cache_clips(model, hold_dl, n_hold, device)
    del hold_dl
    var = hold_var = p_hold.float().var().item()
    print(f"holdout: {n_hold} clips, pixel variance (predict-the-mean) = {hold_var:.4f}")

    # ---- REPLAY-BUFFER training ----------------------------------------------
    # A small fixed cache overfits (a 6-layer decoder memorizes a few hundred clips:
    # train MSE -> 0 while holdout MSE rises above variance). Pure on-the-fly (a
    # fresh clip every step) generalizes but warms up too slowly (each clip seen
    # once). A rotating replay buffer gets both: prefill BUF clips for fast,
    # revisited convergence, and continuously rotate in fresh clips so the run
    # still sees thousands of distinct videos (no memorization). Early-stop on the
    # held-out PSNR.
    decoder = TubeDecoder(EMB, EMB, seq=SEQ_LEN, nlayer=args.nlayer).to(device)
    run = torch.compile(decoder)
    opt = optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.01)

    @torch.no_grad()
    def holdout_mse():
        decoder.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            m = F.mse_loss(decoder(z_hold.to(device).float()).float(), p_hold.to(device).float()).item()
        decoder.train()
        return m

    def lr_at(s):
        warm = max(1, args.steps // 20)
        if s < warm:
            return args.lr * s / warm
        p = (s - warm) / max(1, args.steps - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    dl = build_loader(batch_size=args.bs, workers=8, shuffle=True)
    dl_iter = iter(dl)

    @torch.no_grad()
    def fetch():
        nonlocal dl_iter
        try:
            data = next(dl_iter).to(device)
        except StopIteration:
            dl_iter = iter(dl); data = next(dl_iter).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = target_embed(model, data)
        return z.half().cpu(), data.reshape(data.shape[0], T * S, EMB).half().cpu()

    # prefill the replay buffer
    zb, pb = [], []
    fill = tqdm(total=args.buf, desc="prefill buffer")
    while sum(z.shape[0] for z in zb) < args.buf:
        zf, pf = fetch(); zb.append(zf); pb.append(pf); fill.update(zf.shape[0])
    fill.close()
    zbuf, pbuf = torch.cat(zb)[:args.buf], torch.cat(pb)[:args.buf]
    BUF = zbuf.shape[0]
    print(f"replay buffer: {BUF} clips ({(zbuf.numel() + pbuf.numel()) * 2 / 1e9:.1f} GB), "
          f"refresh {args.bs} clips every {args.refresh} steps "
          f"(~{args.steps * args.bs // max(1, args.refresh)} distinct clips over the run)")

    ptr = 0
    losses = []
    holds = []
    best = {"mse": float("inf"), "state": None, "step": -1}
    pbar = tqdm(range(args.steps), desc="decoder")
    for step in pbar:
        if step % args.refresh == 0:                            # rotate fresh clips in
            zf, pf = fetch()
            n = zf.shape[0]
            sl = (torch.arange(ptr, ptr + n) % BUF)
            zbuf[sl] = zf; pbuf[sl] = pf; ptr = (ptr + n) % BUF
        idx = torch.randint(0, BUF, (args.bs,))
        z = zbuf[idx].to(device).float()
        px = pbuf[idx].to(device).float()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = F.mse_loss(run(z), px)
        loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(loss.item())
        if step % 50 == 0:
            r = np.mean(losses[-50:])
            pbar.set_postfix(mse=f"{r:.4f}", frac_var=f"{r / var:.3f}", psnr=f"{psnr(r):.1f}")
        if step > 0 and step % max(1, args.steps // 20) == 0:
            hm = holdout_mse()
            holds.append((step, hm))
            if hm < best["mse"]:
                best = {"mse": hm, "step": step,
                        "state": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}}
            tqdm.write(f"[step {step}] train MSE {np.mean(losses[-50:]):.4f}  "
                       f"holdout MSE {hm:.4f} ({hm / hold_var:.3f} var, PSNR {psnr(hm):.1f} dB)  "
                       f"best@{best['step']} {psnr(best['mse']):.1f}dB")
            import sys; sys.stdout.flush()

    # Use the FINAL (well-fit) decoder for figures -- the headline is in-distribution
    # reconstruction (the collapse test), for which we want a decoder that fits the
    # embeddings. `best` (lowest held-out) is reported below as the generalization #.
    decoder.eval()
    z_id, p_id = zbuf[:6].clone(), pbuf[:6].clone()             # in-distribution sample
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(pbuf[:8].cpu(), "checkpoints/indist_clips.pt")   # raw clips the decoder trained on
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        id_mse = F.mse_loss(decoder(z_id.to(device).float()).float(), p_id.to(device).float()).item()
        hold_mse = F.mse_loss(decoder(z_hold.to(device).float()).float(), p_hold.to(device).float()).item()
    print(f"\nIN-DIST recon MSE = {id_mse:.4f}  ({id_mse / p_id.float().var().item():.3f} of variance)  "
          f"PSNR = {psnr(id_mse):.1f} dB   <- distinct clips reconstruct => embeddings rich, NOT collapsed")
    print(f"HELDOUT recon MSE = {hold_mse:.4f}  ({hold_mse / p_hold.float().var().item():.3f} of variance)  "
          f"PSNR = {psnr(hold_mse):.1f} dB   (best held-out during training: {psnr(best['mse']):.1f} dB)")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"decoder_state_dict": decoder.state_dict(),
                "nlayer": args.nlayer, "emb": EMB, "n_tokens": T * S,
                "hold_mse": hold_mse, "var": var}, DEC_CKPT)
    print(f"saved decoder -> {DEC_CKPT}")

    # ---- loss curve: train (fresh batches) vs held-out; they should track ----
    plt.figure(figsize=(7, 4))
    plt.plot(losses, lw=.5, alpha=.7, label="train (in-distribution)")
    if holds:
        hx, hy = zip(*holds)
        plt.plot(hx, hy, "-o", ms=3, c="green", label="held-out")
    plt.axhline(var, ls="--", c="r", label=f"pixel variance ({var:.3f})")
    plt.yscale("log"); plt.xlabel("step"); plt.ylabel("recon MSE")
    plt.legend(); plt.title("decoder training: train fits embeddings; held-out gap = abstract features")
    plt.tight_layout(); plt.savefig("decoder_loss.png", dpi=130); plt.close()
    print("saved decoder_loss.png")

    make_figures(model, decoder, z_id, p_id, z_hold, p_hold, device)


def to_frames(tokens_btc):
    """(b, t*s, c_pixel) -> list over batch of (t*fpt, 3, 256, 256) in [0,1]."""
    b = tokens_btc.shape[0]
    tubes = tokens_btc.reshape(b, T, S, EMB).float().cpu()
    f = untubeify(tubes, height=256, frames_per_tube=FRAMES_PER_TUBE)  # (b, t*fpt,3,256,256)
    return (f * 0.5 + 0.5).clamp(0, 1)


def _recon_png(gt_f, rec_f, title, path, stride=3):
    """GT-over-recon rows for several clips: visual proof distinct clips -> distinct recon."""
    n_clips = gt_f.shape[0]
    frames = list(range(0, T * FRAMES_PER_TUBE, stride))
    fig, axes = plt.subplots(2 * n_clips, len(frames),
                             figsize=(len(frames) * 1.15, 2 * n_clips * 1.25))
    for ci in range(n_clips):
        for j, fr in enumerate(frames):
            a0, a1 = axes[2 * ci][j], axes[2 * ci + 1][j]
            a0.imshow(gt_f[ci][fr].permute(1, 2, 0).numpy()); a0.set_xticks([]); a0.set_yticks([])
            a1.imshow(rec_f[ci][fr].permute(1, 2, 0).numpy()); a1.set_xticks([]); a1.set_yticks([])
            if ci == 0 and j == 0:
                a0.set_title(f"frame {fr}", fontsize=7, loc="left")
            if j == 0:
                a0.set_ylabel(f"clip{ci} GT", fontsize=7)
                a1.set_ylabel("recon", fontsize=7)
    plt.suptitle(title, fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print(f"saved {path}")


def _up(img, k=2):  # (H,W,3) float[0,1] -> uint8 nearest-upscaled
    a = (img.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return np.repeat(np.repeat(a, k, 0), k, 1)


@torch.no_grad()
def make_figures(model, decoder, z_id, p_id, z_hold, p_hold, device, n_visible=5):
    decoder.eval()

    def recon(z):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return decoder(z.to(device).float()).float()

    id_gt, id_rec = to_frames(p_id.to(device).float()), to_frames(recon(z_id))
    ho_gt, ho_rec = to_frames(p_hold.to(device).float()), to_frames(recon(z_hold))

    # ---- Figure 1: IN-DISTRIBUTION recon = the collapse test ----
    _recon_png(id_gt[:5], id_rec[:5],
               "Decoder reconstruction from JEPA embeddings (in-distribution).\n"
               "Distinct clips -> distinct, recognizable frames => embeddings are rich, NOT collapsed.",
               "decoder_recon.png")

    # ---- Figure 2: HELD-OUT recon (honest generalization) ----
    _recon_png(ho_gt[:4], ho_rec[:4],
               "Held-out reconstruction (clips never trained on): blurrier -- this small JEPA's\n"
               "features are abstract, so pixel inversion generalizes only partially.",
               "decoder_recon_holdout.png")

    # ---- Figure 3: future-tube generation (world-model rollout) on in-dist clips ----
    raw = p_id.reshape(p_id.shape[0], T, S, EMB).to(device).float()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        full_pred_emb, target_full, _, _ = predict_future(model, raw, n_visible)
        gen = decoder(full_pred_emb).float()
    gen_f = to_frames(gen)
    n_hidden, hid = T - n_visible, list(range(n_visible, T))
    rows = [("ground truth", id_gt), ("recon (true emb)", id_rec), ("generated (pred emb)", gen_f)]
    fig, axes = plt.subplots(3, n_hidden, figsize=(n_hidden * 1.25, 3 * 1.4))
    for r, (label, fr) in enumerate(rows):
        for k, tube in enumerate(hid):
            ax = axes[r][k]
            ax.imshow(fr[0][tube * FRAMES_PER_TUBE].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"tube {tube}", fontsize=7)
            if k == 0:
                ax.set_ylabel(label, fontsize=8)
    plt.suptitle(f"Predict tubes {n_visible}..{T-1} from the first {n_visible} "
                 f"(rollout; bottom row limited by the JEPA predictor, not the decoder)", fontsize=9)
    plt.tight_layout(); plt.savefig("decoder_generate.png", dpi=130); plt.close()
    print("saved decoder_generate.png")

    # ---- GIFs: in-distribution recon (GT | recon), all frames, clips stacked ----
    nfr = id_gt.shape[1]
    gif = []
    for fr in range(nfr):
        cols = []
        for ci in range(min(4, id_gt.shape[0])):
            gt = id_gt[ci][fr].permute(1, 2, 0)
            rc = id_rec[ci][fr].permute(1, 2, 0)
            cols.append(torch.cat([gt, torch.ones(gt.shape[0], 3, 3), rc], dim=1))
        gap = torch.ones(3, cols[0].shape[1], 3)
        stacked = cols[0]
        for c in cols[1:]:
            stacked = torch.cat([stacked, gap, c], dim=0)
        gif.append(_up(stacked))
    save_gif("decoder_recon.gif", gif, fps=6)
    print("saved decoder_recon.gif  (per clip: left=ground truth, right=reconstruction)")


if __name__ == "__main__":
    main()
