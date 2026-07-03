"""Train WMNet to remove the learned Device Parts watermark.

Training pairs are synthesized on the fly:
  background  = random crop from real (cleaned) product photos, plus
                procedural flats/gradients that mimic screens and paper
  watermarked = alpha-composite of the learned watermark at random
                scale/opacity, followed by a JPEG round-trip
The network sees (watermarked RGB, matte) and regresses the residual.

Usage:
  python training/train_wmnet.py --model models/deviceparts_1500.npz \
      --backgrounds output9/ --out models/wmnet.pt --steps 1200
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from watermark_remover import estimate  # noqa: E402
from watermark_remover.neural import WMNet  # noqa: E402

PATCH = 192


class PairMaker:
    def __init__(self, models, bg_dir: str, seed: int = 0):
        if not isinstance(models, (list, tuple)):
            models = [models]
        self.rng = np.random.default_rng(seed)
        self.variants = []
        for model in models:
            a = np.maximum(model.alpha, 0.08)[..., None]
            Wf = model.alpha_w / a
            strong = model.alpha > 0.25
            med = np.median(model.alpha_w[strong] / a[strong[..., None]].reshape(-1, 1),
                            axis=0) if strong.any() else np.array([170, 170, 170])
            Wf[~strong] = med
            self.variants.append((model.alpha, np.clip(Wf, 0, 255).astype(np.float32)))
        self.bgs = []
        for p in sorted(glob.glob(os.path.join(bg_dir, "*"))):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None and min(img.shape[:2]) >= PATCH:
                self.bgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32))
        if not self.bgs:
            raise SystemExit(f"no backgrounds found in {bg_dir}")

    def _bg_patch(self) -> np.ndarray:
        r = self.rng
        kind = r.random()
        if kind < 0.55:  # real photo crop
            img = self.bgs[int(r.integers(len(self.bgs)))]
            y = int(r.integers(0, img.shape[0] - PATCH))
            x = int(r.integers(0, img.shape[1] - PATCH))
            return img[y:y + PATCH, x:x + PATCH].copy()
        if kind < 0.8:  # flat tone (screens, paper) + slight gradient + noise
            base = r.uniform(5, 250)
            g = np.linspace(0, r.uniform(-12, 12), PATCH, dtype=np.float32)
            patch = np.full((PATCH, PATCH, 3), base, np.float32)
            patch += g[None, :, None] if r.random() < 0.5 else g[:, None, None]
            tint = r.uniform(-6, 6, 3).astype(np.float32)
            return np.clip(patch + tint, 0, 255)
        # two-tone edge (product boundary crossing the watermark)
        a, b = r.uniform(5, 250, 2)
        patch = np.full((PATCH, PATCH, 3), a, np.float32)
        pos = int(r.integers(PATCH // 4, 3 * PATCH // 4))
        if r.random() < 0.5:
            patch[:, pos:] = b
        else:
            patch[pos:, :] = b
        angle = r.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((PATCH / 2, PATCH / 2), angle, 1.0)
        return cv2.warpAffine(patch, M, (PATCH, PATCH), borderMode=cv2.BORDER_REFLECT)

    def sample(self):
        r = self.rng
        B = self._bg_patch()
        B += r.normal(0, r.uniform(0.5, 2.0), B.shape).astype(np.float32)
        B = np.clip(B, 0, 255)

        # random variant + window of the watermark at random scale/opacity
        v_alpha, v_W = self.variants[int(r.integers(len(self.variants)))]
        scale = r.uniform(0.55, 1.15)
        op = r.uniform(0.75, 1.25)
        ah, aw_ = v_alpha.shape
        th, tw = int(ah * scale), int(aw_ * scale)
        a_s = cv2.resize(v_alpha, (tw, th), interpolation=cv2.INTER_AREA)
        W_s = cv2.resize(v_W, (tw, th), interpolation=cv2.INTER_AREA)
        # pick a window that actually contains ink most of the time
        for _ in range(8):
            wy = int(r.integers(0, max(th - PATCH, 1)))
            wx = int(r.integers(0, max(tw - PATCH, 1)))
            a_win = a_s[wy:wy + PATCH, wx:wx + PATCH]
            if a_win.mean() > 0.005 or r.random() < 0.15:
                break
        a_win = a_win[:PATCH, :PATCH]
        W_win = W_s[wy:wy + PATCH, wx:wx + PATCH][:PATCH, :PATCH]
        ph, pw = a_win.shape
        if ph < PATCH or pw < PATCH:
            a_win = np.pad(a_win, ((0, PATCH - ph), (0, PATCH - pw)))
            W_win = np.pad(W_win, ((0, PATCH - ph), (0, PATCH - pw), (0, 0)))

        a_eff = np.clip(a_win * op, 0, 0.98)[..., None]
        I = a_eff * W_win + (1 - a_eff) * B

        # JPEG round-trip like the site's exports
        q = int(self.rng.integers(78, 97))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        I = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)

        # Analytic unblend with the *assumed* model (op=1, tiny misalignment)
        # -- reproduces the residue patterns the refiner must repair.
        dx, dy = (int(self.rng.integers(-2, 3)), int(self.rng.integers(-2, 3))) \
            if self.rng.random() < 0.5 else (0, 0)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        a_asm = cv2.warpAffine(a_win, M, (PATCH, PATCH))[..., None]
        W_asm = cv2.warpAffine(W_win, M, (PATCH, PATCH))
        U = (I - a_asm * W_asm) / np.maximum(1 - a_asm, 0.02)
        U = np.clip(U, 0, 255)

        inp = np.concatenate([U / 255.0, a_asm], axis=2)
        tgt = (U - B) / 255.0  # residual the refiner must remove
        return inp.transpose(2, 0, 1), tgt.transpose(2, 0, 1), a_win

    def batch(self, n):
        xs, ys, ms = zip(*(self.sample() for _ in range(n)))
        return (torch.from_numpy(np.stack(xs)).float(),
                torch.from_numpy(np.stack(ys)).float(),
                torch.from_numpy(np.stack(ms)).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backgrounds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    models = [estimate.WatermarkModel.load(p) for p in args.model.split(",")]
    maker = PairMaker(models, args.backgrounds)
    val_x, val_y, val_m = PairMaker(models, args.backgrounds, seed=999).batch(12)

    net = WMNet()
    print(f"params: {sum(p.numel() for p in net.parameters())/1e3:.0f}k")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    t0 = time.time()
    best = 1e9
    for step in range(1, args.steps + 1):
        x, ytgt, m = maker.batch(args.batch)
        pred = net(x)
        # weight loss toward watermark pixels but keep global fidelity
        wmap = 1.0 + 9.0 * m[:, None]
        loss = (wmap * (pred - ytgt).abs()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                vp = net(val_x)
                clean_pred = val_x[:, :3] - vp
                clean_true = val_x[:, :3] - val_y
                mse = ((clean_pred - clean_true) ** 2).mean().item()
                psnr = -10 * np.log10(mse + 1e-12)
            print(f"step {step:5d}  loss {loss.item():.4f}  val-PSNR {psnr:.2f} dB  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            if mse < best:
                best = mse
                torch.save(net.state_dict(), args.out)
    print(f"done; best val PSNR {-10*np.log10(best+1e-12):.2f} dB -> {args.out}")


if __name__ == "__main__":
    main()
