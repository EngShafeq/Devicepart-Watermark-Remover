# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this project is

A command-line tool that removes the fixed "DEVICE PARTS — Grow for Dealers"
watermark from product photos **at full resolution, recovering the true pixels
underneath rather than hallucinating replacements**. This is the core design
principle and it constrains every decision in the codebase: the tool must never
invent content. Fine print, connector pins, and part labels under the watermark
must survive.

The physical model is a per-pixel alpha blend:

```
photo = α·W + (1−α)·original     →     original = (photo − α·W) / (1−α)
```

The watermark's opacity matte `α` and premultiplied color `α·W` are *learned*
from a batch of watermarked photos (they sit at the same position/size in every
export), then the blend is inverted to restore the real pixels. Only
mathematically unrecoverable pixels (near-opaque core) are ever inpainted.

**Everywhere in this codebase, a correction is capped by a physical bound**: a
leftover of a semi-transparent watermark can deviate from the background by at
most ~`α·255`. Anything larger is a real object (a cable, a label, chip print)
crossing the footprint and must be protected. When adding or changing any
cleanup/refinement step, preserve this invariant — it is what keeps the tool
honest.

## Repository layout

```
watermark_remover/        # the installable package (pure library + CLI)
  __main__.py             # entry point: python -m watermark_remover
  cli.py                  # argparse CLI (fit / remove / detect) + pipeline orchestration
  io_utils.py             # quality-preserving load/save (EXIF/ICC/resolution kept)
  detect.py               # watermark localisation (batch-variance + template matching)
  estimate.py             # EM fit of the alpha matte + WatermarkModel (.npz) class
  remove.py               # analytic unblend + residual cleanup passes + metrics
  neural.py               # WMNet U-Net refiner (optional, needs torch)
  mat.py / lama.py        # optional final inpainting back-ends (IOPaint MAT / LaMa)
  _inpaint_mask.py        # shared text-protecting core mask for the inpainters
models/                   # committed model artifacts (see below)
training/                 # WMNet training + eval + self-driving loop
  train_wmnet.py          # synthesizes training pairs, trains the U-Net
  eval_wmnet.py           # objective held-out eval (arbiter for the loop)
  train_loop.sh           # self-driving loop: train → eval → adopt-if-better → push
  data/backgrounds/       # committed real product photos used as training textures
tests/                    # pytest quality tests against synthetic ground truth
  make_fixtures.py        # generates synthetic (clean, watermarked) pairs
  test_quality.py         # PSNR/SSIM + IO + detection assertions
input/                    # drop watermarked images here (CI cleans → output/)
output/                   # CI writes cleaned images here (gitignored locally)
.github/workflows/        # CI automation (see "CI / automation")
```

## Models (`models/`)

- `deviceparts_1500.npz` — primary watermark model, fitted on 1500×1500 exports.
  This is the first-listed / primary model; **only the primary model runs the
  residual-cleanup pass** (it can eat real print on text-dense products).
- `deviceparts_1000_tilted.npz` — the tilted watermark variant on 1000×1000 exports.
- `deviceparts_1500_v2.npz` — an alternate 1500 fit.
- `wmnet.pt` — the trained neural refiner (compact residual U-Net, ~260k params).

A `.npz` model is a `WatermarkModel` (see `estimate.py`): `bbox`, per-pixel
`alpha` (H×W in [0,1]), `alpha_w` (H×W×3, α·W in [0,255]), and `image_shape`.
Models auto-rescale to any image with a matching aspect ratio.

## Common commands

```bash
# Install
pip install -r requirements.txt
# torch is OPTIONAL — only needed for the neural refiner (--net):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Learn the watermark once from ≥4 (ideally 10+) same-size watermarked photos
python -m watermark_remover fit --input-dir samples/ --model wm.npz

# Remove — analytic unblend only
python -m watermark_remover remove --input-dir originals/ --output-dir clean/ \
    --model models/deviceparts_1500.npz

# Remove — full pipeline (both models + neural refiner), the production invocation
python -m watermark_remover remove --input-dir originals/ --output-dir clean/ \
    --model models/deviceparts_1500.npz,models/deviceparts_1000_tilted.npz \
    --net models/wmnet.pt

# Report the auto-detected watermark bounding box (JSON)
python -m watermark_remover detect --input-dir samples/

# Run the quality tests
python -m pytest tests/ -q

# Train / retrain the refiner (needs torch)
python training/train_wmnet.py --model models/deviceparts_1500.npz \
    --backgrounds training/data/backgrounds/ --out models/wmnet.pt --steps 3000

# Evaluate a checkpoint (prints JSON score)
python training/eval_wmnet.py --model models/deviceparts_1500.npz \
    --backgrounds training/data/backgrounds/ --net models/wmnet.pt --n 160
```

Useful `remove` flags: `--keyword Detailed` (only files whose name contains the
word), `--jpeg-quality 97` (default; 4:4:4 chroma), `--workers N` (fan out
across N CPU processes, near-linear speedup), `--no-cleanup` (skip residual
cleanup), `--inpaint {none,lama,mat}` (final polish on the unrecoverable core).

## The removal pipeline (read before touching `remove`/`neural`/`cli`)

`cli._process_loaded()` is the single orchestration point for one image:

1. **`_pick_model`** — choose the watermark model: native-size match wins;
   otherwise, among aspect-compatible models, pick the best by
   `estimate.model_fit_score` (so tilted photos aren't handed the straight model).
2. **register** (`estimate.register_model`) — fine-align the matte to the image
   (non-native sizes only; native placement is already exact from the batch fit).
3. **remove** — either `neural.remove_neural` (if `--net`, does an analytic
   unblend first, then refines) or `remove.remove_unblend` (analytic only).
4. **cleanup passes** (in order), each footprint-confined and α-bounded:
   - `remove.cleanup_residual` — inpaint leftover pixels (primary model only)
   - `remove.suppress_chroma_residual` — pull JPEG chroma noise back
   - `remove.flatten_lowfreq_residual` — flatten the low-frequency ghost on dark textures
   - `remove.despeckle_residual` — remove tiny isolated specks
5. **optional inpaint** (`--inpaint`) — MAT (structure-aware) or LaMa (flat-fill)
   on the near-opaque core, with real print protected via `_inpaint_mask`.

When there is no model, it falls back to template/bbox masking +
`remove.remove_inpaint`.

## Conventions

- **Image arrays are `float32` in `[0, 255]`, RGB, shape `H×W×3`.** Load via
  `io_utils.load_image` (returns a `LoadedImage` carrying EXIF/ICC/dpi); save via
  `io_utils.save_image`. Only clip/round to uint8 at the save boundary. Never
  degrade resolution or drop metadata.
- **Pixels outside the watermark footprint must stay bit-identical.** Every pass
  composites only inside `model.bbox`, usually with a feathered footprint weight.
  `test_quality.py` asserts this — keep it true.
- **Optional dependencies degrade to a no-op, never a crash.** `torch` (neural),
  `iopaint` (MAT), `simple-lama-inpainting` (LaMa) are all imported lazily and
  guarded so the pipeline runs in a locked-down sandbox with no network. Follow
  this pattern for any new optional backend.
- **Corrections are physically bounded by `~α·255`** (see the top section).
- Docstrings here are substantive — they explain *why* each heuristic exists.
  Match that when editing; keep the reasoning next to the code.
- Python style: `from __future__ import annotations`, type hints, stdlib
  `argparse`. Core deps are only `opencv-python-headless`, `numpy`, `Pillow`,
  `scikit-image` (see `requirements.txt`).

## Testing

`tests/make_fixtures.py` synthesizes product photos with a *known* watermark, so
`tests/test_quality.py` can measure removal against ground truth (PSNR/SSIM
inside the watermark box, bit-identical pixels outside it, lossless IO
round-trips). The neural-refiner path is not exercised by these tests (torch is
optional). Run `python -m pytest tests/ -q` before committing changes to
detection, estimation, or removal.

## Training the refiner

WMNet learns to repair exactly the residues the analytic unblend leaves. Pairs
are synthesized on the fly (`training/train_wmnet.py`, `PairMaker`): a
background patch (real product crops, procedural chip/label text, generic
structures, flats, edges) is composited with the *learned* watermark at random
scale/opacity + a JPEG round-trip, then the analytic unblend is simulated. The
network sees `(unblended RGB, α matte)` → predicts the residual to subtract. The
loss weights watermark pixels and includes a Sobel-gradient term so edges/print
stay crisp.

`training/train_loop.sh` is a self-driving loop: it trains candidate
architectures, scores each on a held-out detail-preservation eval
(`eval_wmnet.py`), and **adopts a new `models/wmnet.pt` only when it beats the
incumbent** — so the production model can only improve. It commits+pushes each
adoption and stops on plateau. This same loop runs unattended on CI via
`train.yml`.

## CI / automation (`.github/workflows/`)

All workflows check out and target the **`claude/watermark-removal-tool-kzy7ka`**
branch (this is the repository's default/main branch).

- **`remove-watermark.yml`** — triggered by pushes to `input/**` or manually.
  Runs the full pipeline (with MAT/LaMa, since runners have open network to
  download weights) and commits cleaned images to `output/`.
- **`bulk-process.yml`** — manual. Fans a large batch across N runner shards
  (from `input/` or a `urls.txt`), each with multi-core `--workers`; uploads
  cleaned images as artifacts (not committed — too large).
- **`train.yml`** — manual. Runs `train_loop.sh` under a wall-clock budget,
  commits model improvements. Self-chains job-to-job if a `CHAIN_PAT` secret is set.
- **`test-external-tool.yml`** — manual. Playwright probe of a third-party
  watermark-remover site for comparison (runs where the sandbox can't reach that host).

Note the split: heavyweight/network-dependent work (LaMa/MAT weights, external
site probes, long training) lives in CI because the Claude Code sandbox is
network-restricted; the core analytic + neural pipeline runs anywhere.

## Working in this repo

- The default branch is `claude/watermark-removal-tool-kzy7ka`. Feature work
  happens on `claude/...` branches; confirm your target branch before pushing.
- `.gitignore` excludes scratch/output dirs, session deliverables, and
  `output*/`. CI force-adds (`git add -f`) results it needs to commit.
- Keep model artifacts (`models/*.npz`, `models/wmnet.pt`) and the committed
  training backgrounds (`training/data/backgrounds/`) — they are load-bearing.
- Only process images you own or are licensed to use. For a new watermark
  (different site export), re-run `fit` on a folder of the new photos.
