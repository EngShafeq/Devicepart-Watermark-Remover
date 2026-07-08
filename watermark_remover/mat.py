"""Optional MAT (Mask-Aware Transformer) inpainting via IOPaint.

LaMa fills a masked region with plausible *flat* texture.  Where the
watermark's opaque core sits on a *structured* surface — the seam between
two keys, a connector edge, a boundary between materials — LaMa's fill
reads as a smudge because it does not reconstruct the structure that
crossed the mask.  MAT is a transformer inpainter built for exactly large,
structured masks and keeps those continuations coherent.

IOPaint (formerly lama-cleaner) ships MAT with downloadable weights.  As
with :mod:`watermark_remover.lama`, this activates only when ``iopaint``
and its weights are present, and degrades to a no-op otherwise, so the
pipeline still runs in a locked-down sandbox.  It reuses the same
text-protecting core mask, so real print is never handed to the model.
"""

from __future__ import annotations

import numpy as np

_MODEL = None
_TRIED = False


def available() -> bool:
    return _load() is not None


def _load():
    global _MODEL, _TRIED
    if _TRIED:
        return _MODEL
    _TRIED = True
    try:
        from iopaint.model_manager import ModelManager
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = ModelManager(name="mat", device=device)   # downloads weights
    except Exception:
        _MODEL = None
    return _MODEL


def refine_with_mat(image_rgb: np.ndarray, model, min_alpha: float = 0.6,
                    grow: int = 3, protect_text: bool = True,
                    text_thresh: float = 10.0) -> np.ndarray:
    """Structure-aware inpaint of the unrecoverable core with MAT.

    Same contract as :func:`watermark_remover.lama.refine_with_lama`:
    returns the input unchanged if MAT is unavailable, and only touches the
    near-opaque, detail-less core (print protected)."""
    mat = _load()
    if mat is None:
        return image_rgb

    import cv2

    from ._inpaint_mask import build_core_mask
    full = build_core_mask(image_rgb, model, min_alpha, grow,
                           protect_text, text_thresh)
    if full.sum() == 0:
        return image_rgb

    try:
        from iopaint.schema import InpaintRequest
        config = InpaintRequest()
    except Exception:
        try:                                   # older IOPaint API
            from iopaint.schema import Config
            config = Config()
        except Exception:
            return image_rgb

    rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        res_bgr = mat(bgr, (full * 255).astype(np.uint8), config)
    except Exception:
        return image_rgb
    res = cv2.cvtColor(np.clip(res_bgr, 0, 255).astype(np.uint8),
                       cv2.COLOR_BGR2RGB).astype(np.float32)
    if res.shape != image_rgb.shape:
        res = res[:image_rgb.shape[0], :image_rgb.shape[1]]

    out = image_rgb.astype(np.float32).copy()
    soft = cv2.GaussianBlur(full.astype(np.float32), (0, 0), 2.0)[..., None]
    return out * (1 - soft) + res * soft
