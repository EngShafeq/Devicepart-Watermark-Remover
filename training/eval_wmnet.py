"""Objective evaluation of a WMNet checkpoint — the arbiter for the
self-driving training loop.

We synthesize a *fixed* held-out set of watermarked/clean pairs (same
generator as training but a reserved seed, so no model ever trains on
these exact samples), run the refiner, and score two things that matter
for mobile-spare-part photos:

  footprint_psnr : overall fidelity of the recovered watermark region.
  detail_score   : how well *real detail* survives — gradient (edge/text)
                   agreement measured only where the clean ground truth has
                   strong gradients (chip prints, connector edges, labels).
                   This is the number that guards against "removed or
                   affected" detail.

The combined `score` (higher is better) weights detail preservation
heavily, since losing a part number is worse than a faint smudge.
Usage:
  python training/eval_wmnet.py --model models/deviceparts_1500.npz,... \
      --backgrounds all_samples --net models/wmnet_deep.pt --n 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from watermark_remover import estimate  # noqa: E402
from watermark_remover.neural import load_net  # noqa: E402
from train_wmnet import PairMaker  # noqa: E402

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3)


def _grad_mag(x):
    c = x.shape[1]
    kx = _SOBEL_X.repeat(c, 1, 1, 1)
    ky = _SOBEL_Y.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-6).mean(1, keepdim=True)


def evaluate(net, models, bg_dir, n=64, seed=12345):
    maker = PairMaker(models, bg_dir, seed=seed)
    x, ytgt, m = maker.batch(n)
    with torch.no_grad():
        pred = net(x)
    clean_pred = (x[:, :3] - pred).clamp(0, 1)
    clean_true = (x[:, :3] - ytgt).clamp(0, 1)

    # overall footprint fidelity, weighted to watermark pixels
    wm = (m[:, None] > 0.02).float()
    se = ((clean_pred - clean_true) ** 2 * wm).sum() / wm.sum().clamp(min=1)
    footprint_psnr = float(-10 * torch.log10(se + 1e-12))

    # detail preservation: gradient agreement where GT has real edges/text
    gt_grad = _grad_mag(clean_true)
    edge = (gt_grad > gt_grad.mean() + gt_grad.std()).float()  # strong-edge pixels
    pr_grad = _grad_mag(clean_pred)
    gerr = (pr_grad - gt_grad).abs()
    detail_mae = float((gerr * edge).sum() / edge.sum().clamp(min=1))
    # map to a 0..1 "preservation" score (lower error -> closer to 1)
    detail_score = float(np.exp(-detail_mae * 12.0))

    score = footprint_psnr + 40.0 * detail_score   # detail weighted heavily
    return {
        "footprint_psnr": round(footprint_psnr, 3),
        "detail_mae": round(detail_mae, 5),
        "detail_score": round(detail_score, 4),
        "score": round(score, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backgrounds", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    models = [estimate.WatermarkModel.load(p) for p in args.model.split(",")]
    net = load_net(args.net)
    res = evaluate(net, models, args.backgrounds, n=args.n)
    res["net"] = args.net
    print(json.dumps(res))


if __name__ == "__main__":
    main()
