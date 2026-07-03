"""Generate synthetic product-photo fixtures with a known watermark.

Creates pairs of (clean, watermarked) images that mimic deviceparts-style
product shots: white background, phone parts (rectangles, flex cables,
screws), fine text and texture — then stamps a semi-transparent
"deviceparts.com" text+logo watermark at a fixed position and size, the
same on every image.  Ground truth lets the tests measure exactly how much
detail removal preserves.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIZE = (1000, 1000)
WM_TEXT = "deviceparts.com"
WM_BOX = (280, 430, 440, 140)  # x, y, w, h — center of image, fixed
WM_ALPHA = 0.38


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_watermark_layer() -> Image.Image:
    """RGBA layer, full image size, watermark at the fixed position."""
    layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y, w, h = WM_BOX
    a = int(WM_ALPHA * 255)
    # logo mark: two rounded rectangles like a phone outline
    d.rounded_rectangle([x + 10, y + 30, x + 70, y + 110], radius=12,
                        outline=(90, 90, 95, a), width=6)
    d.rounded_rectangle([x + 26, y + 46, x + 54, y + 94], radius=6,
                        fill=(90, 90, 95, a))
    # text
    d.text((x + 90, y + 45), WM_TEXT, font=_font(44), fill=(80, 80, 85, a))
    d.text((x + 90, y + 98), "WHOLESALE  PARTS", font=_font(20),
           fill=(110, 110, 115, int(a * 0.85)))
    return layer


def make_product_photo(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", SIZE, (255, 255, 255))
    d = ImageDraw.Draw(img)

    # slight vignette / paper tone so background is not perfectly flat
    base = np.asarray(img, np.float32)
    yy, xx = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    vign = 1 - 0.04 * (((xx - 500) / 500) ** 2 + ((yy - 500) / 500) ** 2)
    base *= vign[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)

    # main part: screen assembly / battery-like slab crossing the watermark area
    px, py = int(rng.integers(120, 380)), int(rng.integers(120, 380))
    pw, ph = int(rng.integers(320, 520)), int(rng.integers(380, 620))
    tone = tuple(int(v) for v in rng.integers(15, 70, 3))
    d.rounded_rectangle([px, py, px + pw, py + ph], radius=28, fill=tone,
                        outline=(0, 0, 0), width=3)
    # inner screen with gradient
    inner = np.zeros((ph - 60, pw - 60, 3), np.float32)
    c1 = rng.integers(40, 230, 3).astype(np.float32)
    c2 = rng.integers(40, 230, 3).astype(np.float32)
    t = np.linspace(0, 1, inner.shape[0])[:, None, None]
    inner[:] = c1 * (1 - t) + c2 * t
    img.paste(Image.fromarray(inner.astype(np.uint8)), (px + 30, py + 30))
    d = ImageDraw.Draw(img)

    # fine detail: flex cable, connector grid, tiny label text
    fx, fy = px + pw - 30, py + int(ph * 0.6)
    d.line([fx, fy, fx + 120, fy, fx + 120, fy + 90, fx + 220, fy + 90],
           fill=(180, 130, 40), width=14)
    for i in range(24):
        d.line([fx + 190 + i, fy + 80, fx + 190 + i, fy + 100],
               fill=(60, 45, 20) if i % 2 else (220, 200, 120), width=1)
    d.text((px + 20, py + ph - 26), f"MDL-{seed:04d}  REV {seed % 7}",
           font=_font(16), fill=(200, 200, 200))

    # screws / small parts scattered
    for _ in range(int(rng.integers(3, 7))):
        cx, cy = int(rng.integers(60, 940)), int(rng.integers(60, 940))
        r = int(rng.integers(8, 18))
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  fill=tuple(int(v) for v in rng.integers(90, 190, 3)),
                  outline=(50, 50, 50))

    # photo-like sensor noise
    arr = np.asarray(img, np.float32)
    arr += rng.normal(0, 1.6, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(0.4))


def build(out_dir: str, n: int = 14) -> None:
    clean_dir = os.path.join(out_dir, "clean")
    wm_dir = os.path.join(out_dir, "watermarked")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(wm_dir, exist_ok=True)
    layer = make_watermark_layer()
    for i in range(n):
        clean = make_product_photo(seed=1000 + i)
        clean.save(os.path.join(clean_dir, f"part_{i:02d}_Detailed.png"))
        wm = clean.convert("RGBA")
        wm.alpha_composite(layer)
        wm.convert("RGB").save(os.path.join(wm_dir, f"part_{i:02d}_Detailed.png"))
    # a couple of non-"Detailed" files to exercise the keyword filter
    for i in (90, 91):
        p = make_product_photo(seed=i)
        p.save(os.path.join(wm_dir, f"part_{i}_thumb.png"))
    layer.save(os.path.join(out_dir, "watermark_layer.png"))


if __name__ == "__main__":
    import sys

    build(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
