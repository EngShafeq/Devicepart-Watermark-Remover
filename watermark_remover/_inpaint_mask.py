"""Shared mask builder for the inpainting back-ends (LaMa, MAT).

The mask decides *where* a generic inpainter is allowed to synthesise
texture.  It must be tight: a generic inpainter has no notion that a
character belongs under the watermark, so any real print left inside the
mask gets erased and replaced with flat texture.  We therefore keep only
the near-opaque, genuinely detail-less core and subtract every pixel that
still carries high-frequency structure (recovered strokes, edges).
"""

from __future__ import annotations

import numpy as np


def build_core_mask(image_rgb: np.ndarray, model, min_alpha: float = 0.6,
                    grow: int = 3, protect_text: bool = True,
                    text_thresh: float = 10.0) -> np.ndarray:
    """Return a full-image uint8 {0,1} mask of pixels safe to inpaint.

    ``model`` is rescaled to the image if needed.  Returns an all-zero mask
    when there is nothing safe to fill (no near-opaque core, or the whole
    core is print that must be protected)."""
    import cv2

    if image_rgb.shape[:2] != model.image_shape:
        model = model.rescaled(image_rgb.shape[:2])
    x, y, w, h = model.bbox
    core = (model.alpha >= min_alpha).astype(np.uint8)
    full = np.zeros(image_rgb.shape[:2], np.uint8)
    if core.sum() == 0:
        return full
    full[y:y + h, x:x + w] = core
    if grow > 0:
        full = cv2.dilate(full, np.ones((2 * grow + 1,) * 2, np.uint8))

    if protect_text:
        gray = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32)
        hf = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0))
        hf = cv2.GaussianBlur(hf, (0, 0), 1.5)
        text = (hf > text_thresh).astype(np.uint8)
        text = cv2.dilate(text, np.ones((3, 3), np.uint8))
        full = full & (1 - text)
    return full
