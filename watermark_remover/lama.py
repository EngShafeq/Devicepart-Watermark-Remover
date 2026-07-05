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
                     grow: int = 3, protect_text: bool = True,
                     text_thresh: float = 10.0) -> np.ndarray:
    """Inpaint the *unrecoverable, texture-less* core of the watermark.

    Two guards keep LaMa from destroying real content:

    1. Only the near-opaque footprint (``alpha >= min_alpha``) is masked —
       the small region where the analytic blend is uninvertible.
    2. ``protect_text``: any pixel that still carries strong high-frequency
       structure (chip prints, label strokes, connector edges) is *removed*
       from the mask.  LaMa is a generic inpainter — it synthesises flat,
       plausible texture and has no notion that a character belongs there,
       so left unchecked it erases print under the watermark.  We hand it
       only the genuinely flat, JPEG-crushed pixels where no character
       survives, and leave every recoverable stroke to the refiner.

    Returns the input unchanged if LaMa is unavailable.
    """
    lama = _load()
    if lama is None:
        return image_rgb

    import cv2
    from PIL import Image

    from ._inpaint_mask import build_core_mask
    full = build_core_mask(image_rgb, model, min_alpha, grow,
                           protect_text, text_thresh)
    if full.sum() == 0:                       # nothing flat/opaque left to fill
        return image_rgb

    rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    result = lama(Image.fromarray(rgb), Image.fromarray(full * 255))
    out = image_rgb.astype(np.float32).copy()
    res = np.asarray(result.convert("RGB"), dtype=np.float32)
    if res.shape != out.shape:                       # LaMa pads to /8; crop back
        res = res[:out.shape[0], :out.shape[1]]
    soft = cv2.GaussianBlur(full.astype(np.float32), (0, 0), 2.0)[..., None]
    return out * (1 - soft) + res * soft
