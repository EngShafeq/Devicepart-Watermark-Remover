"""Watermark detection.

Two complementary strategies:

1. ``detect_template`` — locate a known watermark (logo/text image) in a
   single photo using multi-scale, gradient-based template matching.
   Matching on gradients makes it robust to the varying backgrounds seen
   through a semi-transparent watermark.

2. ``detect_region_from_batch`` — no template needed.  Because the site
   stamps the same watermark at the same position and size on every image,
   pixels inside the watermark vary *less* across a batch of images than
   pixels outside it (the constant overlay damps background variance by
   (1-alpha)^2).  A per-pixel variance map across N images exposes the
   watermark as a coherent low-variance / consistent-value region.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float = 1.0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def _gradient_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def detect_template(
    image_rgb: np.ndarray,
    template_rgb: np.ndarray,
    scales: tuple[float, ...] = (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25),
    min_score: float = 0.20,
) -> Detection | None:
    """Find the watermark template in ``image_rgb``.

    Returns the best-scoring detection across scales, or None if nothing
    clears ``min_score``.  Matching runs on gradient magnitudes so a
    semi-transparent watermark is found regardless of what lies beneath it.
    """
    img_gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    tpl_gray = cv2.cvtColor(template_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    img_grad = _gradient_mag(img_gray)
    tpl_grad_base = _gradient_mag(tpl_gray)

    best: Detection | None = None
    for s in scales:
        th = max(4, int(round(tpl_grad_base.shape[0] * s)))
        tw = max(4, int(round(tpl_grad_base.shape[1] * s)))
        if th >= img_grad.shape[0] or tw >= img_grad.shape[1]:
            continue
        tpl = cv2.resize(tpl_grad_base, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(img_grad, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if best is None or max_val > best.score:
            best = Detection(x=max_loc[0], y=max_loc[1], w=tw, h=th, score=float(max_val), scale=s)
    if best is None or best.score < min_score:
        return None
    return best


def detect_region_from_batch(
    images_rgb: list[np.ndarray],
    z_thresh: float = 6.0,
    presence_frac: float = 0.85,
    min_area_frac: float = 0.00002,
) -> tuple[np.ndarray, Detection] | None:
    """Estimate the watermark mask from a batch of same-size images.

    Returns (mask uint8 HxW in {0,255}, bounding-box Detection) or None.
    Requires >= 4 images of identical dimensions.
    """
    if len(images_rgb) < 4:
        raise ValueError("Need at least 4 images to auto-detect the watermark region")
    shape = images_rgb[0].shape
    stack = np.stack([im for im in images_rgb if im.shape == shape]).astype(np.float32)
    if stack.shape[0] < 4:
        raise ValueError("Need at least 4 images with identical dimensions")

    gray = stack.mean(axis=3)  # N,H,W luminance

    # The per-pixel median across the batch melts away the (varying)
    # products and keeps background + the constant watermark.  The
    # watermark ink is whatever deviates from the *local* background of
    # that median image.
    med_img = np.median(gray, axis=0).astype(np.float32)
    med_u8 = np.clip(med_img, 0, 255).astype(np.uint8)
    # Multi-scale: wide logo strokes disappear under a small median window,
    # so take the strongest response across window sizes.
    ink = np.zeros_like(med_img)
    for k in (51, 101, 151):
        bg = cv2.medianBlur(med_u8, k).astype(np.float32)
        ink = np.maximum(ink, np.abs(med_img - bg))

    # Photos of similar products leave product ghosts in the median too.
    # The discriminator: the watermark deviates from the local background
    # in (nearly) EVERY image — product edges only where that product sits.
    presence = np.zeros(med_img.shape, np.float32)
    for k in range(gray.shape[0]):
        g = np.clip(gray[k], 0, 255).astype(np.uint8)
        dev = np.zeros(med_img.shape, np.float32)
        for ksz in (51, 101, 151):
            bg_k = cv2.medianBlur(g, ksz).astype(np.float32)
            dev = np.maximum(dev, np.abs(gray[k] - bg_k))
        presence += (dev > 4.0)
    presence /= gray.shape[0]

    score = ink / (ink.max() + 1e-6)
    mask = ((ink > z_thresh) & (presence >= presence_frac)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Group nearby strokes into clusters and keep the strongest one — the
    # watermark is a single coherent block of text/logo; stray blips are
    # photography artefacts that happen to repeat.
    grow = cv2.dilate(mask, np.ones((41, 41), np.uint8))
    n, labels = cv2.connectedComponents(grow, connectivity=8)
    best_lbl, best_ink = 0, -1.0
    for i in range(1, n):
        blob = (labels == i) & (mask > 0)
        if blob.sum() < int(min_area_frac * mask.size):
            continue
        blob_ink = float(ink[blob].sum())
        if blob_ink > best_ink:
            best_ink, best_lbl = blob_ink, i
    if best_lbl == 0:
        return None
    keep = np.where((labels == best_lbl) & (mask > 0), 255, 0).astype(np.uint8)

    # Complete the footprint: within the winning cluster's bounds, relax the
    # presence requirement (faint strokes over low-contrast backgrounds miss
    # a few images) and let the ink signal alone speak.
    ys, xs = np.nonzero(keep)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad_y = int(0.15 * (y1 - y0 + 1))
    pad_x = int(0.15 * (x1 - x0 + 1))
    relax = np.zeros_like(keep)
    relax[max(0, y0 - pad_y):y1 + 1 + pad_y, max(0, x0 - pad_x):x1 + 1 + pad_x] = 255
    keep = np.where((relax > 0) & (ink > z_thresh * 0.25)
                    & (presence >= presence_frac * 0.4), 255, keep).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel)
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    ys, xs = np.nonzero(keep)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    det = Detection(x=int(x0), y=int(y0), w=int(x1 - x0 + 1), h=int(y1 - y0 + 1),
                    score=float(score[keep > 0].mean()))
    return keep, det


def mask_from_template(
    template_rgba: np.ndarray,
    det: Detection,
    image_shape: tuple[int, ...],
    dilate_px: int = 2,
) -> np.ndarray:
    """Build a full-image uint8 mask by placing the template's footprint at
    the detected location.  Uses the alpha channel when present, otherwise
    the template's non-background ink."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if template_rgba.shape[2] == 4:
        foot = (template_rgba[:, :, 3] > 8).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(template_rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # ink = anything that deviates from the template's dominant flat color
        bg_val = np.bincount(gray.flatten()).argmax()
        foot = (np.abs(gray.astype(np.int16) - int(bg_val)) > 12).astype(np.uint8) * 255
    foot = cv2.resize(foot, (det.w, det.h), interpolation=cv2.INTER_NEAREST)
    mask[det.y:det.y + det.h, det.x:det.x + det.w] = foot
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask
