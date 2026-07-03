"""Estimate the watermark's alpha matte and color from a batch of images.

Model: a watermarked pixel is a linear blend
    I_k(p) = alpha(p) * W(p) + (1 - alpha(p)) * B_k(p)
where W is the (constant) watermark color, alpha its opacity matte, and
B_k the unknown clean background of image k.

Estimation is EM-style:

  E step: guess each image's clean background B̂_k — initially by
          inpainting the watermark region, later by unblending with the
          current model.
  M step: per pixel, regress the observations I_k against B̂_k across the
          batch: slope -> (1 - alpha), intercept -> alpha * W.

Two iterations converge on synthetic ground truth (~36-37 dB PSNR inside
the watermark box); more iterations start to feed the model's own errors
back into the backgrounds and diverge, so we stop early.
"""

from __future__ import annotations

import cv2
import numpy as np

EM_ITERATIONS = 2


class WatermarkModel:
    """Per-pixel alpha matte and premultiplied watermark color for a fixed
    region of the image."""

    def __init__(self, bbox: tuple[int, int, int, int],
                 alpha: np.ndarray, alpha_w: np.ndarray,
                 image_shape: tuple[int, int]):
        self.bbox = bbox              # x, y, w, h in full-image coords
        self.alpha = alpha            # h x w float32 in [0,1]
        self.alpha_w = alpha_w        # h x w x 3 float32, alpha*W in [0,255]
        self.image_shape = image_shape

    def save(self, path: str) -> None:
        np.savez_compressed(path, bbox=np.array(self.bbox), alpha=self.alpha,
                            alpha_w=self.alpha_w, image_shape=np.array(self.image_shape))

    @classmethod
    def load(cls, path: str) -> "WatermarkModel":
        d = np.load(path)
        return cls(tuple(int(v) for v in d["bbox"]), d["alpha"].astype(np.float32),
                   d["alpha_w"].astype(np.float32), tuple(int(v) for v in d["image_shape"]))

    def rescaled(self, image_shape: tuple[int, int]) -> "WatermarkModel":
        """Adapt the model to an image size with the same aspect ratio —
        the site scales the whole canvas (watermark included), so the
        matte scales proportionally."""
        if image_shape == self.image_shape:
            return self
        sy = image_shape[0] / self.image_shape[0]
        sx = image_shape[1] / self.image_shape[1]
        x, y, w, h = self.bbox
        nx, ny = int(round(x * sx)), int(round(y * sy))
        nw = min(int(round(w * sx)), image_shape[1] - nx)
        nh = min(int(round(h * sy)), image_shape[0] - ny)
        alpha = cv2.resize(self.alpha, (nw, nh), interpolation=cv2.INTER_AREA)
        alpha_w = cv2.resize(self.alpha_w, (nw, nh), interpolation=cv2.INTER_AREA)
        return WatermarkModel((nx, ny, nw, nh), alpha, alpha_w, image_shape)


def _coarse_background(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Initial clean-background guess: inpaint the watermark footprint."""
    u8 = np.clip(crop, 0, 255).astype(np.uint8)
    return cv2.inpaint(u8, mask, 7, cv2.INPAINT_TELEA).astype(np.float32)


def _flatness_weight(bg: np.ndarray) -> np.ndarray:
    """Observations over flat backgrounds (plain white / uniform product
    surfaces) constrain the blend far better than textured ones, where the
    inpainted background guess is unreliable.  Weight accordingly."""
    gray = bg.mean(axis=2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 2.0)
    return 1.0 / (1.0 + 0.05 * grad)


def estimate_watermark(
    images_rgb: list[np.ndarray],
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    template_rgb: np.ndarray | None = None,
    template_weight: float = 4.0,
) -> WatermarkModel:
    """Fit the blend model inside ``bbox`` using every image in the batch.

    ``mask`` is the full-image watermark footprint (uint8).  All images must
    share one size (others are skipped).

    ``template_rgb`` — optional full-size image of the watermark flattened
    on a white canvas (the marketing logo file).  It acts as a noiseless
    extra observation whose background is exactly 255, anchoring the fit.
    """
    x, y, w, h = bbox
    shape = images_rgb[0].shape[:2]
    mask_crop = (mask[y:y + h, x:x + w] > 0).astype(np.uint8) * 255
    crops = [im[y:y + h, x:x + w].astype(np.float32)
             for im in images_rgb if im.shape[:2] == shape]
    if len(crops) < 4 and template_rgb is None:
        raise ValueError("Need at least 4 same-size images to estimate the watermark")

    I = np.stack(crops)                      # N,h,w,3
    B = np.stack([_coarse_background(c, mask_crop) for c in crops])
    foot = cv2.GaussianBlur((mask_crop > 0).astype(np.float32), (5, 5), 1.0)

    tpl = None
    if template_rgb is not None:
        if template_rgb.shape[:2] != shape:
            template_rgb = cv2.resize(template_rgb, (shape[1], shape[0]),
                                      interpolation=cv2.INTER_AREA)
        tpl = template_rgb[y:y + h, x:x + w].astype(np.float32)

    if tpl is not None:
        # The logo file flattened on white is one extra noiseless observation.
        I = np.concatenate([I, tpl[None]], axis=0)
        B = np.concatenate([B, np.full_like(tpl, 255.0)[None]], axis=0)

    alpha = np.zeros((h, w), np.float32)
    alpha_w = np.zeros((h, w, 3), np.float32)
    n = I.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    for _ in range(EM_ITERATIONS + 1):
        # M step — pairwise slopes.  For every image pair (i, j):
        #     I_i - I_j = (1 - alpha) * (B_i - B_j)
        # A pair with very different, flat backgrounds (white sheet vs black
        # screen) pins the slope down almost exactly; the weighted median
        # over pairs is immune to the attenuation bias that plain least
        # squares suffers when the B estimates are noisy.
        Wgt = np.stack([_flatness_weight(b) for b in B])  # N,h,w
        gI = I.mean(axis=3)
        gB = B.mean(axis=3)
        dI = gI[ii] - gI[jj]                     # P,h,w
        dB = gB[ii] - gB[jj]
        pw = Wgt[ii] * Wgt[jj] * np.abs(dB)      # pair confidence
        # Keep only pairs whose background contrast is close to the best
        # available at that pixel — small-contrast pairs carry no slope
        # information and only drag the median toward "no watermark".
        db_abs = np.abs(dB)
        pw = pw * (db_abs > 12.0) * (db_abs >= 0.6 * db_abs.max(axis=0, keepdims=True))
        slope = np.where(np.abs(dB) > 1e-3, dI / np.where(np.abs(dB) > 1e-3, dB, 1.0), 1.0)
        slope = np.clip(slope, 0.0, 1.0)

        order = np.argsort(slope, axis=0)
        s_sorted = np.take_along_axis(slope, order, axis=0)
        w_sorted = np.take_along_axis(pw, order, axis=0)
        cum = np.cumsum(w_sorted, axis=0)
        total = cum[-1]
        idx = (cum >= total[None] / 2.0).argmax(axis=0)
        one_minus_a = np.take_along_axis(s_sorted, idx[None], axis=0)[0]
        alpha = np.clip(1.0 - one_minus_a, 0.0, 0.98)

        # Pixels where no pair had contrasting backgrounds are blind — fill
        # their alpha from surrounding estimated pixels.
        blind = total < 1e-3
        if blind.any() and (~blind).any():
            a_fill = cv2.inpaint(np.clip(alpha * 255, 0, 255).astype(np.uint8),
                                 blind.astype(np.uint8) * 255, 5,
                                 cv2.INPAINT_TELEA).astype(np.float32) / 255.0
            alpha = np.where(blind, a_fill, alpha)

        alpha = alpha * foot
        # Weighted mean of the intercept aW = I - (1-a) B over observations.
        w_obs = Wgt[..., None]
        alpha_w = (w_obs * (I - (1.0 - alpha[None, ..., None]) * B)).sum(axis=0) \
            / np.maximum(w_obs.sum(axis=0), 1e-6)
        alpha_w = np.clip(alpha_w, 0, 255) * (alpha > 0.01)[..., None]

        # E step: refresh backgrounds by unblending with the current model
        denom = np.maximum(1.0 - alpha[..., None], 0.02)
        B_new = np.clip((I - alpha_w) / denom, 0, 255)
        B = np.stack([cv2.GaussianBlur(b, (0, 0), 1.0) for b in B_new])
        if tpl is not None:  # the template's background is white by construction
            B[-1] = 255.0

    return WatermarkModel(bbox=(x, y, w, h), alpha=alpha.astype(np.float32),
                          alpha_w=alpha_w.astype(np.float32), image_shape=shape)


def register_model(image_rgb: np.ndarray, model: "WatermarkModel",
                   scales=(0.85, 0.92, 1.0, 1.08, 1.15),
                   search_pad: int = 60) -> "WatermarkModel":
    """Fine-align the watermark matte to one specific image.

    Different export sizes of the site stamp the watermark with small
    position/scale differences; correlating the matte against the image's
    local-background deviation ("ink") finds the exact placement before
    unblending.  Returns a shifted/rescaled copy of the model (or the
    original if no placement scores clearly)."""
    if image_rgb.shape[:2] == model.image_shape:
        # native size: the batch fit already nailed the placement; searching
        # again can only false-lock on product structure.
        return model
    work = model.rescaled(image_rgb.shape[:2])
    x, y, w, h = work.bbox
    H, W = image_rgb.shape[:2]
    gray = image_rgb.astype(np.float32).mean(axis=2)
    g_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    ink = np.zeros_like(gray)
    for k in (51, 101):
        bg = cv2.medianBlur(g_u8, k).astype(np.float32)
        ink = np.maximum(ink, np.abs(gray - bg))
    x0, y0 = max(0, x - search_pad), max(0, y - search_pad)
    x1, y1 = min(W, x + w + search_pad), min(H, y + h + search_pad)
    window = ink[y0:y1, x0:x1]

    best = None
    default_score = 0.0
    for s in scales:
        tw, th = int(round(w * s)), int(round(h * s))
        if th >= window.shape[0] or tw >= window.shape[1]:
            continue
        tpl = cv2.resize(work.alpha, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(window.astype(np.float32), tpl.astype(np.float32),
                                cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if abs(s - 1.0) < 1e-6:
            ry, rx = y - y0, x - x0
            if 0 <= ry < res.shape[0] and 0 <= rx < res.shape[1]:
                default_score = float(res[ry, rx])
        if best is None or mx > best[0]:
            best = (mx, s, x0 + ml[0], y0 + ml[1])
    # Re-place the matte only when the alternative beats the default
    # placement decisively — a weak lock elsewhere must never displace a
    # correct default (the 1500px exports always match the default).
    if best is None or best[0] < 0.10 or best[0] < 1.4 * max(default_score, 0.05):
        return work
    _, s, nx, ny = best
    tw, th = int(round(w * s)), int(round(h * s))
    alpha = cv2.resize(work.alpha, (tw, th), interpolation=cv2.INTER_AREA)
    alpha_w = cv2.resize(work.alpha_w, (tw, th), interpolation=cv2.INTER_AREA)
    tw = min(tw, W - nx)
    th = min(th, H - ny)
    return WatermarkModel((nx, ny, tw, th), alpha[:th, :tw], alpha_w[:th, :tw],
                          image_rgb.shape[:2])


def mask_from_flat_template(template_rgb: np.ndarray, ink_thresh: float = 2.5,
                            dilate_px: int = 3) -> np.ndarray:
    """Watermark footprint from a logo file flattened on white: ink is any
    pixel that deviates from the white canvas."""
    ink = 255.0 - template_rgb.astype(np.float32).mean(axis=2)
    mask = (ink > ink_thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    return cv2.dilate(mask, k)
