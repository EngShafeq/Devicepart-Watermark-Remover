"""Command-line interface.

Typical workflow for deviceparts product photos (watermark is always the
same position and size):

    # 1. Learn the watermark once from a folder of watermarked images
    python -m watermark_remover fit --input-dir samples/ --model wm.npz

    # 2. Remove it from everything (only files whose name contains
    #    "Detailed", per the site's naming, if you pass --keyword)
    python -m watermark_remover remove --input-dir originals/ \
        --output-dir clean/ --model wm.npz --keyword Detailed

Single images with a known logo template also work:

    python -m watermark_remover remove --input-dir originals/ \
        --output-dir clean/ --template logo.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import detect, estimate, io_utils, remove


def _load_batch(input_dir: str, keyword: str | None) -> list[io_utils.LoadedImage]:
    paths = io_utils.list_images(input_dir, keyword)
    if not paths:
        sys.exit(f"No images found in {input_dir}"
                 + (f" matching keyword {keyword!r}" if keyword else ""))
    return [io_utils.load_image(p) for p in paths]


def cmd_fit(args: argparse.Namespace) -> None:
    batch = _load_batch(args.input_dir, args.keyword)
    images = [b.rgb for b in batch]
    if args.bbox:
        x, y, w, h = (int(v) for v in args.bbox.split(","))
        mask = np.zeros(images[0].shape[:2], np.uint8)
        mask[y:y + h, x:x + w] = 255
        bbox = (x, y, w, h)
    else:
        found = detect.detect_region_from_batch(images)
        if not found:
            sys.exit("Could not auto-detect a consistent watermark region; "
                     "pass --bbox x,y,w,h")
        mask, det = found
        bbox = det.bbox
        print(f"Auto-detected watermark region: x={det.x} y={det.y} "
              f"w={det.w} h={det.h} (score {det.score:.2f})")
    model = estimate.estimate_watermark(images, mask, bbox)
    model.save(args.model)
    print(f"Watermark model written to {args.model} "
          f"(mean alpha {model.alpha[model.alpha > 0.01].mean():.2f})")


def cmd_remove(args: argparse.Namespace) -> None:
    batch = _load_batch(args.input_dir, args.keyword)
    os.makedirs(args.output_dir, exist_ok=True)

    models = [estimate.WatermarkModel.load(p) for p in args.model.split(",")] \
        if args.model else []
    template = io_utils.load_image(args.template).rgb if args.template else None
    if template is not None:
        from PIL import Image
        with Image.open(args.template) as t:
            template = np.asarray(t.convert("RGBA"), dtype=np.float32)

    fixed_bbox = None
    if args.bbox:
        fixed_bbox = tuple(int(v) for v in args.bbox.split(","))

    def _pick_model(shape):
        """Native-size match wins; otherwise the first model with the same
        aspect ratio (its matte is rescaled + registered per image)."""
        for m in models:
            if shape == m.image_shape:
                return m
        for m in models:
            if abs(shape[1] / shape[0] - m.image_shape[1] / m.image_shape[0]) < 0.01:
                return m
        return None

    net = None
    if args.net:
        from . import neural
        net = neural.load_net(args.net)

    report = []
    for item in batch:
        name = os.path.basename(item.path)
        try:
            model = _pick_model(item.rgb.shape[:2])
            # Cleanup inpaints leftovers; on text-dense products (labels,
            # chip prints) it can eat real print, so it only runs for the
            # primary (first-listed) variant.
            do_cleanup = (not args.no_cleanup) and models and model is models[0]
            if net is not None and model is not None:
                from . import estimate as est
                from . import neural
                reg = est.register_model(item.rgb, model)
                out = neural.remove_neural(item.rgb, reg, net)
                if do_cleanup:
                    out = remove.cleanup_residual(out, reg)
                out = remove.suppress_chroma_residual(out, reg)
                out = remove.flatten_lowfreq_residual(out, reg)
                how = "neural"
            elif model is not None:
                out = remove.remove_unblend(item.rgb, model)
                if do_cleanup:
                    out = remove.cleanup_residual(out, model)
                out = remove.suppress_chroma_residual(out, model)
                out = remove.flatten_lowfreq_residual(out, model)
                how = "unblend"
            else:
                mask = _build_mask(item.rgb, template, fixed_bbox)
                if mask is None:
                    print(f"skip  {name}: watermark not found")
                    report.append({"file": name, "status": "not-found"})
                    continue
                out = remove.remove_inpaint(item.rgb, mask)
                how = "inpaint"
        except Exception as exc:  # keep the batch going
            print(f"error {name}: {exc}")
            report.append({"file": name, "status": f"error: {exc}"})
            continue
        item.rgb = out
        out_path = os.path.join(args.output_dir, name)
        io_utils.save_image(item, out_path, jpeg_quality=args.jpeg_quality)
        print(f"ok    {name} ({how}) -> {out_path}")
        report.append({"file": name, "status": "ok", "method": how})

    with open(os.path.join(args.output_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)


def _build_mask(image_rgb, template_rgba, fixed_bbox):
    import cv2

    if fixed_bbox is not None:
        x, y, w, h = fixed_bbox
        mask = np.zeros(image_rgb.shape[:2], np.uint8)
        mask[y:y + h, x:x + w] = 255
        return mask
    if template_rgba is not None:
        det = detect.detect_template(image_rgb, template_rgba[:, :, :3])
        if det is None:
            return None
        return detect.mask_from_template(template_rgba, det, image_rgb.shape)
    return None


def cmd_detect(args: argparse.Namespace) -> None:
    batch = _load_batch(args.input_dir, args.keyword)
    found = detect.detect_region_from_batch([b.rgb for b in batch])
    if not found:
        sys.exit("No consistent watermark region detected")
    _, det = found
    print(json.dumps({"x": det.x, "y": det.y, "w": det.w, "h": det.h,
                      "score": round(det.score, 3)}))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="watermark_remover",
                                description="Remove the fixed deviceparts watermark "
                                            "from product images at full quality.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-dir", required=True)
    common.add_argument("--keyword", default=None,
                        help="only process files whose name contains this word, "
                             "e.g. Detailed")

    f = sub.add_parser("fit", parents=[common],
                       help="learn the watermark (alpha + color) from a batch")
    f.add_argument("--model", required=True, help="output .npz model path")
    f.add_argument("--bbox", default=None, help="x,y,w,h to skip auto-detection")
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser("remove", parents=[common], help="remove the watermark")
    r.add_argument("--output-dir", required=True)
    r.add_argument("--model", default=None,
                   help="comma-separated .npz models from `fit`; the best-fitting one "
                        "is chosen per image (native size first)")
    r.add_argument("--template", default=None, help="watermark logo image (PNG w/ alpha)")
    r.add_argument("--bbox", default=None, help="fixed x,y,w,h region to inpaint")
    r.add_argument("--jpeg-quality", type=int, default=97)
    r.add_argument("--no-cleanup", action="store_true",
                   help="skip the residual cleanup pass after unblending")
    r.add_argument("--net", default=None,
                   help="trained WMNet weights (.pt) — uses the neural remover "
                        "instead of the analytic unblend (requires torch)")
    r.set_defaults(func=cmd_remove)

    d = sub.add_parser("detect", parents=[common],
                       help="report the auto-detected watermark region")
    d.set_defaults(func=cmd_detect)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
