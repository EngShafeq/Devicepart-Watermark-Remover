"""Quality-preserving image input/output.

Images are loaded once, processed as float32 arrays, and written back at
maximum quality (JPEG 4:4:4 quality 97, or lossless PNG/WebP), preserving
the original resolution, EXIF metadata and ICC color profile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # allow very large product photos

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class LoadedImage:
    """An image plus everything needed to save it back losslessly-ish."""

    rgb: np.ndarray  # HxWx3 float32 in [0, 255]
    path: str
    info: dict = field(default_factory=dict)  # exif / icc_profile / dpi


def load_image(path: str) -> LoadedImage:
    with Image.open(path) as im:
        info = {}
        exif = im.info.get("exif")
        if exif:
            info["exif"] = exif
        icc = im.info.get("icc_profile")
        if icc:
            info["icc_profile"] = icc
        dpi = im.info.get("dpi")
        if dpi:
            info["dpi"] = dpi
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    return LoadedImage(rgb=rgb, path=path, info=info)


def save_image(img: LoadedImage, out_path: str, jpeg_quality: int = 97) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    arr = np.clip(np.round(img.rgb), 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    ext = os.path.splitext(out_path)[1].lower()
    kwargs = dict(img.info)
    if ext in (".jpg", ".jpeg"):
        kwargs.update(quality=jpeg_quality, subsampling=0, optimize=True)
    elif ext == ".webp":
        kwargs.update(lossless=True)
    elif ext in (".tif", ".tiff"):
        kwargs.update(compression="tiff_lzw")
    # PNG needs no quality args: it is lossless.
    kwargs.pop("dpi", None) if ext == ".webp" else None
    pil.save(out_path, **kwargs)


def list_images(directory: str, keyword: str | None = None) -> list[str]:
    """List image files in a directory, optionally filtered by a keyword
    appearing in the file name (case-insensitive), e.g. ``Detailed``."""
    paths = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue
        if keyword and keyword.lower() not in name.lower():
            continue
        paths.append(os.path.join(directory, name))
    return paths
