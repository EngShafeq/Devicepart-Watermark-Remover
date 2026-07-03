"""Watermark removal.

``remove_unblend`` inverts the alpha blend using a fitted WatermarkModel —
this *recovers* the true pixels under a semi-transparent watermark, keeping
every detail of the product photo.  Pixels the model marks as nearly opaque
cannot be recovered and are inpainted instead.

``remove_inpaint`` is the fallback for opaque watermarks or single images
without a model: edge-aware inpainting with a smooth blend back into the
photo so no seam is visible.
"""

from __future__ import annotations

import cv2
import numpy as np

from .estimate import WatermarkModel

OPAQUE_ALPHA = 0.90


def remove_inpaint(image_rgb: np.ndarray, mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """High-quality inpaint of the masked region, blended seamlessly."""
    u8 = np.clip(image_rgb, 0, 255).astype(np.uint8)
    m = (mask > 0).astype(np.uint8) * 255
    # Telea handles smooth/product-shot regions well; Navier-Stokes keeps
    # linear structures.  Blend the two: NS for high-gradient surroundings,
    # Telea elsewhere.
    telea = cv2.inpaint(u8, m, radius, cv2.INPAINT_TELEA).astype(np.float32)
    ns = cv2.inpaint(u8, m, radius, cv2.INPAINT_NS).astype(np.float32)

    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    ring = cv2.dilate(m, np.ones((9, 9), np.uint8)) & ~m
    edge_level = float(np.percentile(grad[ring > 0], 90)) if (ring > 0).any() else 0.0
    w_ns = np.clip(edge_level / 200.0, 0.0, 0.5)
    filled = (1 - w_ns) * telea + w_ns * ns

    # Feathered composite: only masked pixels change, edge blends over ~2px.
    soft = cv2.GaussianBlur((m > 0).astype(np.float32), (5, 5), 1.0)[..., None]
    return image_rgb * (1 - soft) + filled * soft


def remove_unblend(image_rgb: np.ndarray, model: WatermarkModel,
                   denoise: bool = True) -> np.ndarray:
    """Invert I = a*W + (1-a)*B  ->  B = (I - a*W) / (1-a) inside the
    model's region; inpaint only where alpha is too high to invert.

    The model is rescaled automatically for images whose aspect ratio
    matches (the site exports the same canvas at several sizes).
    """
    if image_rgb.shape[:2] != model.image_shape:
        ar_img = image_rgb.shape[1] / image_rgb.shape[0]
        ar_model = model.image_shape[1] / model.image_shape[0]
        if abs(ar_img - ar_model) > 0.01:
            raise ValueError(
                f"Image size {image_rgb.shape[:2]} does not match the watermark "
                f"model {model.image_shape}; re-run estimation for this size."
            )
        model = model.rescaled(image_rgb.shape[:2])
    x, y, w, h = model.bbox
    out = image_rgb.astype(np.float32).copy()
    crop = out[y:y + h, x:x + w]

    a = model.alpha[..., None]
    recover = (model.alpha < OPAQUE_ALPHA)[..., None]
    denom = np.maximum(1.0 - a, 0.02)
    unblended = (crop - model.alpha_w) / denom
    crop_new = np.where(recover, unblended, crop)
    crop_new = np.clip(crop_new, 0, 255)

    if denoise:
        # Unblending divides by (1-a), amplifying sensor noise where the
        # watermark was strongest.  Blend in a gentle NL-means denoise
        # weighted by that amplification so recovered pixels match the
        # noise level of the rest of the photo.
        dn = cv2.fastNlMeansDenoisingColored(
            crop_new.astype(np.uint8), None, 5, 5, 5, 15).astype(np.float32)
        wgt = np.clip(model.alpha / (1.0 - model.alpha + 1e-3), 0, 1.0)[..., None] * 0.9
        crop_new = crop_new * (1 - wgt) + dn * wgt

    out[y:y + h, x:x + w] = crop_new

    # Inpaint the unrecoverable (near-opaque) pixels, if any.
    opaque = (model.alpha >= OPAQUE_ALPHA).astype(np.uint8) * 255
    if opaque.any():
        full = np.zeros(model.image_shape, dtype=np.uint8)
        full[y:y + h, x:x + w] = opaque
        out = remove_inpaint(out, full)
    return out


def cleanup_residual(image_rgb: np.ndarray, model: WatermarkModel,
                     thresh: float | None = None) -> np.ndarray:
    """Safety net after unblending: any pixel inside the watermark footprint
    that still deviates from its local background is a leftover (alpha was
    over- or under-estimated there) — inpaint just those pixels from their
    clean surroundings.

    ``thresh=None`` adapts to the local scene: a low threshold on flat
    backgrounds (plain screens, white sheets) where any residual is
    conspicuous, a high one on textured areas where "deviation from local
    median" is normal and inpainting would erase real detail."""
    if image_rgb.shape[:2] != model.image_shape:
        model = model.rescaled(image_rgb.shape[:2])
    x, y, w, h = model.bbox
    crop = image_rgb[y:y + h, x:x + w]
    gray = crop.mean(axis=2).astype(np.float32)
    foot = cv2.dilate((model.alpha > 0.02).astype(np.uint8) * 255,
                      np.ones((5, 5), np.uint8))
    # Background from *outside* the footprint (inpainting), so residuals of
    # arbitrarily wide watermark strokes are measured against truly clean
    # pixels — a median window would swallow wide strokes whole.
    bg = cv2.inpaint(np.clip(gray, 0, 255).astype(np.uint8), foot, 7,
                     cv2.INPAINT_TELEA).astype(np.float32)
    resid = np.abs(gray - bg)
    if thresh is None:
        # Local texture level from the ring just outside the footprint,
        # brought inside by dilation so every footprint pixel sees the
        # roughness of its clean neighbourhood.
        ring_resid = np.where(foot == 0, resid, 0).astype(np.float32)
        ring = cv2.dilate(ring_resid, np.ones((61, 61), np.uint8))
        thresh_map = np.clip(3.0 * ring + 4.0, 7.0, 28.0)
    else:
        ring = None
        thresh_map = np.full_like(resid, float(thresh))
    # A leftover of the watermark can deviate from the background by at most
    # ~alpha*255; a much larger deviation is a real object crossing the
    # footprint (a cable, a label) — protect it from being "cleaned".
    a_local = cv2.dilate(model.alpha, np.ones((5, 5), np.uint8))
    resid_cap = a_local * 220.0 + 14.0
    real_object = resid > resid_cap

    ink = model.alpha > 0.10  # only true watermark strokes, not the halo
    bad = ((resid > thresh_map) & ink & ~real_object).astype(np.uint8) * 255
    out = image_rgb
    if bad.any():
        bad = cv2.dilate(bad, np.ones((3, 3), np.uint8))
        full = np.zeros(model.image_shape, np.uint8)
        full[y:y + h, x:x + w] = bad
        out = remove_inpaint(image_rgb, full)

    if ring is not None:
        # On genuinely flat surroundings (plain screens, seamless paper)
        # inpainting the whole footprint reproduces the background exactly,
        # erasing any faint leftover the unblend missed.  On textured areas
        # flatness -> 0 and the detail-preserving unblend result stands.
        flat = np.clip((6.0 - ring) / 5.0, 0.0, 1.0)
        flat = flat * (foot > 0) * ~cv2.dilate(real_object.astype(np.uint8),
                                               np.ones((9, 9), np.uint8)).astype(bool)
        flat = cv2.GaussianBlur(flat.astype(np.float32), (0, 0), 3.0)
        if flat.max() > 0.05:
            full = np.zeros(model.image_shape, np.uint8)
            full[y:y + h, x:x + w] = foot
            inp = remove_inpaint(out, full)
            blend = np.zeros(model.image_shape, np.float32)
            blend[y:y + h, x:x + w] = flat
            out = out * (1 - blend[..., None]) + inp * blend[..., None]
    return out


def region_metrics(a: np.ndarray, b: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    """PSNR / SSIM restricted to the watermark bounding box — the honest
    measure of removal quality."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    x, y, w, h = bbox
    ca = np.clip(a[y:y + h, x:x + w], 0, 255).astype(np.uint8)
    cb = np.clip(b[y:y + h, x:x + w], 0, 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(ca, cb, data_range=255)
    ssim = structural_similarity(ca, cb, channel_axis=2, data_range=255)
    return {"psnr": float(psnr), "ssim": float(ssim)}
