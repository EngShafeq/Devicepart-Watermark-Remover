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

from . import detect, estimate, io_utils, neural, remove


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
    paths = io_utils.list_images(args.input_dir, args.keyword)
    if not paths:
        sys.exit(f"No images found in {args.input_dir}"
                 + (f" matching keyword {args.keyword!r}" if args.keyword else ""))
    os.makedirs(args.output_dir, exist_ok=True)
    backend = args.inpaint
    if backend == "none" and args.lama:
        backend = "lama"

    # --- parallel path: fan out across CPU cores (opt-in, big speedup) ---
    if getattr(args, "workers", 1) and args.workers > 1 and not args.template:
        import concurrent.futures as cf
        import multiprocessing as mp
        jobs = [(p, args.output_dir, args.jpeg_quality, backend) for p in paths]
        ctx = mp.get_context("spawn")           # spawn avoids torch/fork deadlocks
        report = []
        with cf.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=ctx,
                initializer=_worker_init,
                initargs=(args.model, args.net, args.template,
                          args.bbox, args.no_cleanup)) as ex:
            for r in ex.map(_worker_process, jobs):
                mark = "ok   " if r["status"] == "ok" else "skip "
                print(f"{mark} {r['file']} ({r.get('method', r['status'])})")
                report.append(r)
        with open(os.path.join(args.output_dir, "report.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        return

    # --- serial path (single process) ---
    models = [estimate.WatermarkModel.load(p) for p in args.model.split(",")] \
        if args.model else []
    template = None
    if args.template:
        from PIL import Image
        template = np.asarray(Image.open(args.template).convert("RGBA"), dtype=np.float32)
    fixed_bbox = tuple(int(v) for v in args.bbox.split(",")) if args.bbox else None
    net = neural.load_net(args.net) if args.net else None

    report = []
    for path in paths:
        item = io_utils.load_image(path)
        name = os.path.basename(path)
        try:
            out, how = _process_loaded(item.rgb, models, net, template, fixed_bbox,
                                       backend, not args.no_cleanup)
            if out is None:
                print(f"skip  {name}: watermark not found")
                report.append({"file": name, "status": "not-found"})
                continue
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


def _run_inpaint(out, model, backend):
    """Optional final inpainting polish on the unrecoverable core.

    backend: 'mat' (structure-aware), 'lama' (flat fill), or 'none'. Each is
    a graceful no-op when its package/weights are absent."""
    if backend == "mat":
        from . import mat
        return mat.refine_with_mat(out, model)
    if backend == "lama":
        from . import lama
        return lama.refine_with_lama(out, model)
    return out


def _pick_model(models, shape, image_rgb=None):
    """Choose the watermark model for an image: native-size match wins;
    otherwise, among aspect-compatible models, pick the best-fitting one by
    matte-vs-image correlation (so tilted-watermark photos aren't handed the
    straight model)."""
    for m in models:
        if shape == m.image_shape:
            return m
    cands = [m for m in models
             if abs(shape[1] / shape[0] - m.image_shape[1] / m.image_shape[0]) < 0.01]
    if not cands:
        return None
    if len(cands) == 1 or image_rgb is None:
        return cands[0]
    scored = [(estimate.model_fit_score(image_rgb, m), m) for m in cands]
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def _process_loaded(rgb, models, net, template, fixed_bbox, backend, do_cleanup_allowed):
    """Run the full removal pipeline on one loaded RGB image. Returns
    (output_rgb_or_None, how_string)."""
    model = _pick_model(models, rgb.shape[:2], rgb)
    do_cleanup = do_cleanup_allowed and models and model is models[0]
    if net is not None and model is not None:
        reg = estimate.register_model(rgb, model)
        out = neural.remove_neural(rgb, reg, net)
        if do_cleanup:
            out = remove.cleanup_residual(out, reg)
        out = remove.suppress_chroma_residual(out, reg)
        out = remove.flatten_lowfreq_residual(out, reg)
        out = remove.despeckle_residual(out, reg)
        return _run_inpaint(out, reg, backend), "neural"
    if model is not None:
        out = remove.remove_unblend(rgb, model)
        if do_cleanup:
            out = remove.cleanup_residual(out, model)
        out = remove.suppress_chroma_residual(out, model)
        out = remove.flatten_lowfreq_residual(out, model)
        out = remove.despeckle_residual(out, model)
        return _run_inpaint(out, model, backend), "unblend"
    mask = _build_mask(rgb, template, fixed_bbox)
    if mask is None:
        return None, "not-found"
    return remove.remove_inpaint(rgb, mask), "inpaint"


# ---- multiprocessing workers (opt-in via --workers) --------------------
_W: dict = {}


def _worker_init(model_paths, net_path, template_path, bbox_str, no_cleanup):
    try:
        import torch
        torch.set_num_threads(1)          # N workers x 1 thread = N cores, no oversubscription
    except Exception:
        pass
    import cv2
    cv2.setNumThreads(1)
    _W["models"] = [estimate.WatermarkModel.load(p) for p in model_paths.split(",")] \
        if model_paths else []
    _W["net"] = neural.load_net(net_path) if net_path else None
    _W["template"] = None
    if template_path:
        from PIL import Image
        _W["template"] = np.asarray(Image.open(template_path).convert("RGBA"), dtype=np.float32)
    _W["bbox"] = tuple(int(v) for v in bbox_str.split(",")) if bbox_str else None
    _W["no_cleanup"] = no_cleanup


def _worker_process(job):
    """job = (path, output_dir, jpeg_quality, backend). Runs in a subprocess."""
    path, output_dir, jpeg_quality, backend = job
    name = os.path.basename(path)
    try:
        item = io_utils.load_image(path)
        out, how = _process_loaded(item.rgb, _W["models"], _W["net"], _W["template"],
                                   _W["bbox"], backend, not _W["no_cleanup"])
        if out is None:
            return {"file": name, "status": "not-found"}
        item.rgb = out
        io_utils.save_image(item, os.path.join(output_dir, name), jpeg_quality=jpeg_quality)
        return {"file": name, "status": "ok", "method": how}
    except Exception as exc:
        return {"file": name, "status": f"error: {exc}"}


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
    r.add_argument("--workers", type=int, default=1,
                   help="process images across N parallel CPU processes "
                        "(near-linear speedup for large batches; use ~number "
                        "of CPU cores). GPU auto-used when present.")
    r.add_argument("--no-cleanup", action="store_true",
                   help="skip the residual cleanup pass after unblending")
    r.add_argument("--net", default=None,
                   help="trained WMNet weights (.pt) — uses the neural remover "
                        "instead of the analytic unblend (requires torch)")
    r.add_argument("--inpaint", choices=["none", "lama", "mat"], default="none",
                   help="final inpainting polish on the unrecoverable near-opaque "
                        "core: 'mat' is structure-aware (IOPaint, best on edges), "
                        "'lama' is flat-fill; both protect real print and are a "
                        "no-op if their weights are unavailable")
    r.add_argument("--lama", action="store_true",
                   help="alias for --inpaint lama (kept for back-compat)")
    r.set_defaults(func=cmd_remove)

    d = sub.add_parser("detect", parents=[common],
                       help="report the auto-detected watermark region")
    d.set_defaults(func=cmd_detect)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
