# Dataset: Esri ArcGIS Reality "Packing House District"

## Purpose

This dataset is the development/model-preparation data source for the
SAM-based drone segmentation and classification pipeline (see the repo
[README](../README.md)). It is used to build and validate preprocessing,
segmentation, and classification before the pipeline is tested on unseen
DroneStock footage.

The dataset itself is **never stored in this repository**. It lives on
each contributor's machine (or shared storage) and is referenced only
through environment variables at run time.

## Dependencies

This component's dependencies are isolated in `requirements-data.txt`
rather than a shared root `requirements.txt`, to avoid colliding with
other components' dependency files once branches merge — see "Git safety
rules" below:

```bash
pip install -r requirements-data.txt
```

## Download and local placement

1. Open Esri's official ArcGIS Reality sample drone datasets page:
   https://www.esri.com/en-us/arcgis/products/arcgis-reality/resources/sample-drone-datasets
2. Find **Packing House District** and select **Download this dataset**.
3. Download the ZIP archive (approximately 6.6 GB).
4. Extract it to a local directory outside this Git repository.
5. Confirm that the extracted dataset root contains:

   ```text
   Metadata.txt
   Redlands - Packing House District.png
   Images/
   ```

6. Confirm that `Images/` contains 307 `.jpg` drone images and `Thumbs.db`.
7. Set `DRONE_DATASET_ROOT` to the extracted dataset root and
   `DRONE_OUTPUT_ROOT` to a separate output directory. The PowerShell and
   Git Bash examples later in this document show how to set them.

Each teammate may choose a different local directory. Never commit a
personal absolute path. The ZIP archive and raw images must never be
copied into or committed to the Git repository. Treat raw images as
read-only and write all generated files to the separate output root.

## Confirmed directory structure

```
<DRONE_DATASET_ROOT>/
├── Metadata.txt                          # Esri-provided text description (not an image)
├── <overview>.png                        # Location/overview image (1254x791) - not a drone photo
└── Images/
    ├── 211021_181317_983.jpg
    ├── 211021_181321_767.jpg
    ├── ...
    └── Thumbs.db                         # Windows thumbnail cache - not dataset content
```

`<DRONE_DATASET_ROOT>` is a placeholder for wherever the dataset is
extracted locally, e.g. `D:\datasets\PackingHouseDistrict` or
`/home/<user>/datasets/PackingHouseDistrict`. The actual local path is
never committed to this repository (see "Git safety rules" below).

## Confirmed counts, sizes, dimensions, and formats

Verified by direct, read-only inspection of the extracted dataset:

- **307** `.jpg` files in `Images/`, all with unique, sequential
  timestamp-style filenames.
- Total dataset size: **~6.57 GB** (matches Esri's own `Metadata.txt`,
  which states 6.56 GB).
- **Every image is exactly 9504x6336 pixels**, RGB, aspect ratio
  **3:2 (1.5)** — zero variance across the set.
- Camera: Sony ILCE-7RM4. Capture date: 2019-08-07.
- `Thumbs.db`, the overview PNG, and `Metadata.txt` are excluded from all
  image processing; they are not drone photos.

## MPO / two-frame behavior

Although every file has a `.jpg` extension, Pillow reports the format as
**MPO** (Multi-Picture Object), and each file contains **2 embedded
frames** — a Sony camera quirk where a second (lower-resolution
preview/disparity) frame is packed into the same container.

Frame 0 is the full-resolution 9504x6336 image; this is the frame that
matters for this pipeline. **Both `analyze_dataset.py` and
`preprocess.py` explicitly `seek(0)` before reading size, EXIF, or pixel
data**, rather than relying on Pillow's default frame. `preprocess.py`
exposes this as `preprocessing.mpo_frame` in `configs/data.yaml` (default
`0`) in case a future need arises to inspect the second frame, but it
must not be changed without a specific reason.

## EXIF / GPS availability

Confirmed 100% coverage across all 307 images:

- `Make`, `Model`, `Orientation` (always `1` — no rotation needed),
  `DateTime`, resolution fields, `ImageDescription`, `Software`.
- `GPSInfo` present in every image (latitude/longitude/altitude).

`preprocess.py` applies `PIL.ImageOps.exif_transpose()` before any
geometric transform, so orientation is handled safely even though the
current dataset never actually needs a rotation.

## Corruption and duplicate-check limitations

- **Corruption check**: both scripts use Pillow's `Image.verify()` by
  default, which checks file/container structural integrity without
  decoding full pixel data. Zero corrupted files were found in the real
  dataset. This is *not* an exhaustive guarantee — `verify()` does not
  decode every pixel, so pixel-level corruption deep inside a file could
  theoretically be missed. `analyze_dataset.py --full-decode` can force a
  full pixel decode of every image for a stronger guarantee, at the cost
  of reading the entire ~6.6 GB dataset — this was intentionally not run
  against the real dataset as part of this task.
- **Duplicate check**: filename and exact-byte-size matching only (no
  content hashing). Zero duplicate candidates were found in the real
  dataset by either signal. This cannot rule out two different files with
  different names/sizes that are visually identical, and does not
  guarantee true byte-for-byte non-duplication for files that happen to
  share the same size for unrelated reasons.

## Why raw images must remain unchanged

The raw dataset is Esri's original sample data, referenced by absolute
path outside the repository. Scripts in this repository:

- open every raw file in read-only mode,
- never call any Pillow method that writes back to a source file,
- write generated outputs only to a separate `DRONE_OUTPUT_ROOT` location
  that is verified (at run time) to not overlap the dataset root or the
  repository.

This keeps the dataset reusable and byte-identical for every contributor
and for future re-runs, and keeps generated/processed data out of Git.

## Environment variable setup

### PowerShell

```powershell
$env:DRONE_DATASET_ROOT = "D:\datasets\PackingHouseDistrict"
$env:DRONE_OUTPUT_ROOT  = "D:\datasets\PackingHouseDistrict_outputs"
```

### Git Bash

```bash
export DRONE_DATASET_ROOT="/d/datasets/PackingHouseDistrict"
export DRONE_OUTPUT_ROOT="/d/datasets/PackingHouseDistrict_outputs"
```

Both scripts also accept `--dataset-root` and `--output-root` CLI flags,
which take priority over the environment variables.

## How to run dataset analysis

```bash
python src/data/analyze_dataset.py
```

Optional flags:

```bash
python src/data/analyze_dataset.py --dataset-root "D:\datasets\PackingHouseDistrict" \
    --output "D:\datasets\PackingHouseDistrict_outputs\analysis_report.json" \
    --full-decode
```

`--output` must point outside both the repository and the dataset
directory; the script rejects unsafe paths before writing anything. The
script never modifies the dataset and exits non-zero on invalid paths,
an empty image set, unreadable config, or a fatal analysis error.

## How to run preprocessing in dry-run mode

By default `preprocess.py` runs in **dry-run mode**: it selects images,
computes the planned transform for each, and prints the plan — it never
writes image files or a manifest.

```bash
python src/data/preprocess.py
```

## How to execute preprocessing

Add `--execute` to actually select, transform, and write images plus a
manifest:

```bash
python src/data/preprocess.py --execute
```

Useful overrides:

```bash
python src/data/preprocess.py --execute \
    --selection-method stratified --count 30 --seed 42 \
    --preprocess-method none --manifest-format json --overwrite
```

`--preprocess-method none` is the config default and is what feeds SAM
(see "Confirmed SAM integration contract" below); `letterbox`,
`center_crop`, and `stretch` remain available as optional overrides for
other uses, but are never used for the SAM handoff.

Existing output files are not overwritten unless `preprocessing.overwrite:
true` in the config or `--overwrite` is passed on the CLI.

## Output and manifest structure

```
<DRONE_OUTPUT_ROOT>/
├── selected_images/
│   ├── 211021_181317_983.jpg
│   └── ...
└── manifest.json   (or manifest.csv)
```

`selected_images` is the configurable `preprocessing.output_subdir` in
`configs/data.yaml`. With the default `none` method, output filenames are
**the original source filenames, unchanged** (no hash suffix) — e.g.
`211021_181317_983.jpg` in, `211021_181317_983.jpg` out. Since the current
dataset's 307 filenames are already confirmed unique, this is safe; if two
selected source files would ever produce the same output filename — this
check is **case-insensitive**, so e.g. `Image01.jpg` and `image01.jpeg`
count as a collision, matching how Windows filesystems actually behave —
`preprocess.py` fails with a clear error *before* writing anything,
rather than silently overwriting one of them.

The manifest records, per selected image: source/output relative paths,
the original source filename, selection method/index/bin/seed, source and
output dimensions, the source Pillow format and frame count, the exported
frame index, output format, preprocessing method, and an `output_encoding`
note generated from those actual values (e.g. for this dataset: *"standard
single-frame JPEG re-encoded from source MPO frame 0"*) — it is not a
hardcoded string, so it stays accurate if this component is ever pointed
at a non-MPO source.

## Path and format safety

`dataset.image_subdir` and `preprocessing.output_subdir` are both
resolved through `common.resolve_safe_subdir()` before being used for any
read or write: the configured value must be a non-empty **relative**
path, and the final resolved location (after following any symlinks) must
still be contained inside its trusted root (the dataset root or the
output root, respectively). Absolute paths, `..` traversal that would
escape the root, and symlink-based escapes are all rejected with a clear
`DatasetError` — before any directory is created, image is saved, or
manifest is written. `preprocessing.output_subdir` additionally may not
resolve to the output root itself, keeping generated images confined to
their own subfolder.

`preprocessing.output_format` is restricted to `jpg`/`jpeg` — this
component's contract is standard JPEG output only, matching what SAM's
`cv2.imread`-based loader expects. Any other value (including anything
containing `/`, `\`, `.`, or path-traversal characters) is rejected before
any output is written.

## Current selection decision and reasoning

`configs/data.yaml` defaults to **stratified** selection: 307 files are
sorted deterministically by filename, split into 30 contiguous,
approximately-equal bins covering the entire ordered sequence, and one
file per bin is drawn using a local `random.Random(42)` generator
(Python's global random state is never touched). This is a
recommendation, not an immovable choice — `random` selection is also
implemented and available via `--selection-method random` /
`selection.method: random`.

Reasoning: the 307 images are one continuous nadir flight over a single
district, captured roughly every 3.3-4 seconds — sequential
photogrammetry overlap capture, not independent random scenes. Diversity
in this dataset comes almost entirely from spatial position along the
flight path. A uniform random sample (even with a fixed seed) risks
clustering in one part of the flight path; stratified selection
guarantees coverage across the full sequence for a small 30-of-307
(~10%) subset.

## Comparison of resize methods (optional, non-default)

`letterbox`, `center_crop`, and `stretch` remain implemented and
selectable via `--preprocess-method`, but **none of them are used for the
SAM handoff** (see the contract below). They stay available for other
possible uses (e.g. a future classifier that wants a fixed square input).

| Method | Effect on 9504x6336 -> 1024x1024 | Tradeoff |
|---|---|---|
| `stretch` | Non-uniform scale (x: ~0.108, y: ~0.162) | Distorts real-world object geometry — risky for mask-quality-sensitive models. |
| `center_crop` | Crops to a 6336x6336 center square (~33% of width discarded) before scaling | No distortion, but permanently discards roughly a third of each scene's horizontal coverage. |
| `letterbox` | Uniform scale to 1024x683, padded top/bottom to 1024x1024 | Preserves full scene and true aspect ratio, but wastes ~33% of the canvas as padding and still downsamples ~9.3x. |
| `none` (**default**) | No resize; output size equals source size (9504x6336) | Preserves everything at native resolution — required for the SAM handoff. Produces large files (~20-27 MB each). |

Tiling (splitting each image into a grid of native-resolution patches) is
**not implemented here and must not be** — it is `src/segmentation/tiling.py`'s
responsibility on Deniz's `feature/sam-segmentation` branch (confirmed at
commit `bd1fe6b710f2672e6a1ab42f2dd20dcad2f457dc`), not this component's.

## Confirmed SAM integration contract

Inspected directly from `origin/feature/sam-segmentation` (commit
`bd1fe6b710f2672e6a1ab42f2dd20dcad2f457dc`, read-only, no checkout):

- **Selected images must remain at original resolution (9504x6336).**
  SAM's static pipeline (`configs/sam.yaml` on that branch:
  `preprocessing.strategy: "tiling"`) tiles the image as loaded — it does
  not expect a pre-resized input, and downscaling to 1024x1024 before
  handoff would remove detail the tiling pipeline is tuned to use.
- **SAM performs 1536x1536 tiling with 256px overlap internally**
  (`tiling.tile_size: 1536`, `tiling.overlap: 256` in their config;
  implemented in `src/segmentation/tiling.py::generate_tiles()`).
- **This component's responsibility ends after selecting and exporting
  the 30 images plus a manifest.** We must not resize, letterbox, crop,
  stretch, or tile before handing off — that is exactly what `none`
  (the default) guarantees.
- **Deniz's component performs tiling, SAM 2.1 segmentation, tile-boundary
  stitching, bounding-box extraction, and 224x224 PNG crop generation** —
  all downstream of the plain full-resolution JPEG this component
  produces.
- **Their loader is OpenCV (`cv2.imread`), not Pillow**, and expects a
  normal single-frame image path. OpenCV has no explicit multi-frame/seek
  API for MPO containers the way Pillow does, so this component converts
  each selected image to a **standard single-frame JPEG** (decoded from
  MPO frame 0) rather than handing off the raw two-frame MPO file — this
  removes any dependency on undocumented OpenCV MPO behavior.
- **Their current entrypoint (`scripts/test_mask_generation.py`) hardcodes
  a single sample path** (`data/samples/sample_02.jpg`) and does not yet
  read a directory or a manifest. Wiring up batch/manifest consumption
  across our 30 exported images is **Mustafa's integration-step work**,
  not something this component can complete unilaterally.

## Remaining open items (not blocking this component's output)

1. How will the integration step actually invoke SAM per selected image —
   a loop over the manifest, a CLI batch flag added to their script, or
   something else? Depends on Mustafa's integration design.
2. Minimum object size SAM needs to reliably segment, for future tuning
   of `mask_filter.min_area` on Deniz's side — not something this
   component's output affects.
3. Whether `PyYAML` should be added explicitly to Deniz's `requirements.txt`
   (their `sam_model.py` imports `yaml` but it isn't pinned there today).

## Git safety rules for datasets and generated outputs

- The raw dataset is **read-only** and lives outside this repository. No
  script in this repo writes to the dataset directory.
- Generated outputs (analysis reports, processed images, manifests) are
  written only under `DRONE_OUTPUT_ROOT`, which both scripts verify is
  outside the repository and outside the dataset root before writing
  anything.
- `.gitignore` excludes common dataset/output directory names, `Thumbs.db`,
  archives, model weights, and other generated artifacts, without
  globally ignoring all `.jpg`/`.png` files (so small tracked
  documentation images remain possible).
- No personal absolute dataset path is committed anywhere in this
  repository; all examples above use generic placeholder paths.
- **Known future merge conflict**: `feature/sam-segmentation` also adds
  its own root `.gitignore` (globally ignoring `*.jpg`/`*.jpeg`/`*.png`,
  plus `checkpoints/`, `sam2_repo/`, `.idea/`). Since neither branch's
  common ancestor (`main`) has a `.gitignore`, merging both branches will
  produce an **add/add conflict** that must be manually reconciled by
  whoever integrates the branches (Mustafa) — this component's
  `.gitignore` is deliberately left as-is rather than pre-resolving that
  conflict unilaterally. Our Python dependencies are kept in
  `requirements-data.txt` (not `requirements.txt`) specifically so they
  don't collide with `feature/sam-segmentation`'s own root
  `requirements.txt`.
