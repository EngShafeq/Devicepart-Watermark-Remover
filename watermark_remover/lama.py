"""Optional LaMa (Large-Mask inpainting) final polish.

The analytic + refiner pipeline recovers the true pixels under a
semi-transparent watermark.  Where the watermark was *near-opaque* — or the
source JPEG crushed the underlying detail beyond recovery — no inversion
can restore what is gone; the honest fallback is to synthesise plausible
texture.  LaMa is the model professional erasers (Cleanup.pictures,
Dewatermark, IOPaint) use for exactly this, and it fills large masks with
coherent texture far better than classical inpainting.

This module wires LaMa in as an *optional* last step: it activates only
when the ``simple-lama-inpainting`` package and its weights are present.
In a locked-down sandbox (no access to the weight host) it degrades
gracefully to a no-op, so the pipeline still runs everywhere; on a machine
with normal network access ``pip install simple-lama-inpainting`` is all
that is needed to switch it on.
"""

from __future__ import annotations

import numpy as np

_LAMA = None
_TRIED = False


def available() -> bool:
    """True if LaMa can actually run (package importable + weights loaded)."""
    return _load() is not None


def _load():
    global _LAMA, _TRIED
    if _TRIED:
        return _LAMA
    _TRIED = True
    try:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()          # downloads weights on first use
    except Exception:
        _LAMA = None                  # package or weights unavailable
    return _LAMA


def refine_with_lama(image_rgb: np.ndarray, model, min_alpha: float = 0.6,
                     grow: int = 3) -> np.ndarray:
    """Inpaint the unrecoverable core of the watermark with LaMa.

    Only the near-opaque footprint (``alpha >= min_alpha``) is masked — the
    small region where the blend is uninvertible — so recoverable detail
    and real print outside it are preserved.  Returns the input unchanged
    if LaMa is unavailable.
    """
    lama = _load()
    if lama is None:
        return image_rgb

    import cv2
    from PIL import Image

    if image_rgb.shape[:2] != model.image_shape:
        model = model.rescaled(image_rgb.shape[:2])
    x, y, w, h = model.bbox
    core = (model.alpha >= min_alpha).astype(np.uint8)
    if core.sum() == 0:
        return image_rgb
    full = np.zeros(image_rgb.shape[:2], np.uint8)
    full[y:y + h, x:x + w] = core
    if grow > 0:
        full = cv2.dilate(full, np.ones((2 * grow + 1,) * 2, np.uint8))

    rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    result = lama(Image.fromarray(rgb), Image.fromarray(full * 255))
    out = image_rgb.astype(np.float32).copy()
    res = np.asarray(result.convert("RGB"), dtype=np.float32)
    if res.shape != out.shape:                       # LaMa pads to /8; crop back
        res = res[:out.shape[0], :out.shape[1]]
    soft = cv2.GaussianBlur(full.astype(np.float32), (0, 0), 2.0)[..., None]
    return out * (1 - soft) + res * soft
