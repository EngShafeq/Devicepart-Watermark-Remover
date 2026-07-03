# Devicepart Watermark Remover

High-quality removal of the repeated "DEVICE PARTS — Grow for Dealers"
watermark from product photos, at **full resolution** with the underlying
detail genuinely **recovered** — not hallucinated.

أداة لإزالة العلامة المائية من صور المنتجات بجودة عالية مع الحفاظ على دقة
الصورة وتفاصيلها الأصلية.

## Why this beats generic AI erasers

Online AI erasers *paint over* the watermark area and invent replacement
pixels — fine print, connector pins and part labels under the watermark are
lost. This tool instead **learns the watermark itself** (its per-pixel
opacity `α` and color `W`) from a folder of your photos, then inverts the
blend equation

```
photo = α·W + (1−α)·original      →      original = (photo − α·W) / (1−α)
```

so the true pixels underneath are restored. A final safety pass inpaints
only what is mathematically unrecoverable (near-opaque pixels) and any
faint leftovers on flat backgrounds.

The watermark is auto-detected — no coordinates needed — by comparing many
photos: only the watermark deviates from the local background *in every
image at the same place*.

## Install

```bash
pip install -r requirements.txt
```

## Usage

**1. Learn the watermark once** from a folder with at least 4 (ideally
10+) watermarked photos of the *same pixel size*:

```bash
python -m watermark_remover fit --input-dir samples/ --model deviceparts_wm.npz
```

**2. Remove it from everything:**

```bash
python -m watermark_remover remove \
    --input-dir originals/ --output-dir clean/ --model deviceparts_wm.npz
```

Useful options:

- `--keyword Detailed` — only process files whose name contains the word
  (e.g. the site's "Detailed" images).
- `--jpeg-quality 97` — output JPEG quality (default 97, 4:4:4 chroma).
  PNG/WebP/TIFF outputs are lossless. Resolution, EXIF and ICC profiles are
  always preserved.
- `--no-cleanup` — disable the residual-cleanup pass (pure unblend).
- Images whose size differs from the model but has the same aspect ratio
  are handled automatically (the matte is rescaled — the site exports the
  same canvas at 1500×1500 and 1000×1000).

A ready-to-use model fitted on Device Parts 1500×1500 photos ships in
`models/deviceparts_1500.npz`.

**Check what was detected** (prints the watermark bounding box as JSON):

```bash
python -m watermark_remover detect --input-dir samples/
```

## How it works

| Stage | Method |
|---|---|
| Detect | Per-pixel median across the batch + multi-scale local-background deviation ("ink"), gated by a presence test: the pixel must deviate in ≥80% of the photos, which rejects product structure that repeats between shots. |
| Estimate | EM loop: backgrounds are guessed by inpainting, then per-pixel α is fitted from **pairwise slopes** between images — pairs with contrasting flat backgrounds (white sheet vs black screen) pin α down and are immune to the attenuation bias of plain least squares. |
| Remove | Invert the alpha blend (recovers true pixels), gentle noise-matched denoise where the division amplified sensor noise, inpaint only near-opaque pixels. |
| Cleanup | Residuals are measured against a background inpainted from *outside* the footprint; flat surroundings get a seamless full-footprint inpaint, textured areas keep the unblended detail. Real objects crossing the watermark (cables, labels) are protected by a physical bound: a leftover can never deviate more than α·255. |

## Quality assurance

`tests/` generates synthetic product photos with a known watermark and
verifies end-to-end quality against ground truth (PSNR/SSIM inside the
watermark box, untouched pixels outside it, lossless round-trips):

```bash
python -m pytest tests/ -q
```

## Notes

- Use this tool only on images you own or are licensed to use.
- For a new watermark (different site export), just re-run `fit` on a
  folder of the new photos.
