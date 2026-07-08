"""End-to-end quality tests against synthetic ground truth.

Targets (measured ONLY inside the watermark bounding box, the honest metric):
  - unblend (model) removal:  PSNR >= 34 dB, SSIM >= 0.97
  - inpaint (fallback):       PSNR >= 26 dB, SSIM >= 0.90
  - resolution and clean pixels outside the box must be untouched.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from watermark_remover import detect, estimate, io_utils, remove  # noqa: E402
from tests import make_fixtures  # noqa: E402


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory):
    out = tmp_path_factory.mktemp("fx")
    make_fixtures.build(str(out), n=14)
    clean = [io_utils.load_image(p).rgb
             for p in io_utils.list_images(str(out / "clean"))]
    wm = [io_utils.load_image(p).rgb
          for p in io_utils.list_images(str(out / "watermarked"), keyword="Detailed")]
    return {"dir": out, "clean": clean, "wm": wm}


def test_keyword_filter(fixtures):
    all_files = io_utils.list_images(str(fixtures["dir"] / "watermarked"))
    detailed = io_utils.list_images(str(fixtures["dir"] / "watermarked"), keyword="Detailed")
    assert len(all_files) == len(detailed) + 2
    assert all("Detailed" in os.path.basename(p) for p in detailed)


def test_auto_detect_region(fixtures):
    found = detect.detect_region_from_batch(fixtures["wm"])
    assert found is not None, "watermark region not detected"
    mask, det = found
    x, y, w, h = make_fixtures.WM_BOX
    # detected box must overlap the true watermark box well (IoU-ish check)
    ix = max(0, min(det.x + det.w, x + w) - max(det.x, x))
    iy = max(0, min(det.y + det.h, y + h) - max(det.y, y))
    inter = ix * iy
    union = det.w * det.h + w * h - inter
    assert inter / union > 0.5, f"poor localisation: {det.bbox} vs {make_fixtures.WM_BOX}"


def test_unblend_quality(fixtures):
    wm, clean = fixtures["wm"], fixtures["clean"]
    mask, det = detect.detect_region_from_batch(wm)
    model = estimate.estimate_watermark(wm, mask, det.bbox)

    psnrs, ssims = [], []
    for w_img, c_img in zip(wm, clean):
        out = remove.remove_unblend(w_img, model)
        m = remove.region_metrics(out, c_img, det.bbox)
        psnrs.append(m["psnr"])
        ssims.append(m["ssim"])
        # pixels outside the model bbox must be bit-identical
        x, y, bw, bh = model.bbox
        untouched = out.copy()
        untouched[y:y + bh, x:x + bw] = w_img[y:y + bh, x:x + bw]
        assert np.array_equal(untouched, w_img)
    print(f"unblend: PSNR mean {np.mean(psnrs):.2f} min {np.min(psnrs):.2f} | "
          f"SSIM mean {np.mean(ssims):.4f} min {np.min(ssims):.4f}")
    assert np.mean(psnrs) >= 30.0
    assert np.min(psnrs) >= 27.0
    assert np.mean(ssims) >= 0.93


def test_inpaint_fallback_quality(fixtures):
    """Inpainting is the no-model fallback; measure it with the true
    footprint mask (what a template would provide)."""
    import cv2
    from PIL import Image

    wm, clean = fixtures["wm"], fixtures["clean"]
    layer = np.asarray(Image.open(fixtures["dir"] / "watermark_layer.png"))
    mask = cv2.dilate((layer[:, :, 3] > 8).astype(np.uint8) * 255,
                      np.ones((5, 5), np.uint8))
    x, y, w, h = make_fixtures.WM_BOX
    psnrs, ssims = [], []
    for w_img, c_img in zip(wm[:6], clean[:6]):
        out = remove.remove_inpaint(w_img, mask)
        m = remove.region_metrics(out, c_img, (x, y, w, h))
        psnrs.append(m["psnr"])
        ssims.append(m["ssim"])
    print(f"inpaint: PSNR mean {np.mean(psnrs):.2f} min {np.min(psnrs):.2f} | "
          f"SSIM mean {np.mean(ssims):.4f} min {np.min(ssims):.4f}")
    assert np.mean(psnrs) >= 22.0
    assert np.mean(ssims) >= 0.88


def test_template_detection_opaque(fixtures):
    """Template matching targets clearly visible (opaque) logos; stamp one
    at full opacity and check it is located."""
    from PIL import Image

    layer = np.asarray(Image.open(fixtures["dir"] / "watermark_layer.png"))
    x, y, w, h = make_fixtures.WM_BOX
    tpl = layer[y:y + h, x:x + w]
    a = tpl[:, :, 3:4].astype(np.float32) / 255
    tpl_rgb = tpl[:, :, :3].astype(np.float32) * a + 255 * (1 - a)

    img = fixtures["clean"][0].copy()
    region = img[y:y + h, x:x + w]
    img[y:y + h, x:x + w] = tpl[:, :, :3] * a + region * (1 - a) * 0.0 + region * (a == 0)
    det = detect.detect_template(img, tpl_rgb)
    assert det is not None
    assert abs(det.x - x) <= 12 and abs(det.y - y) <= 12, (det.bbox, (x, y))


def test_resolution_and_io(fixtures, tmp_path):
    src = io_utils.list_images(str(fixtures["dir"] / "watermarked"), keyword="Detailed")[0]
    item = io_utils.load_image(src)
    orig_shape = item.rgb.shape
    out_path = str(tmp_path / "out.png")
    io_utils.save_image(item, out_path)
    reread = io_utils.load_image(out_path)
    assert reread.rgb.shape == orig_shape
    assert np.array_equal(reread.rgb, item.rgb)  # PNG round-trip is lossless
