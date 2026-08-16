# CLAUDE.md

## Project Overview
This project processes raw drone imagery using the Segment Anything Model (SAM).
SAM generates candidate object masks; each mask becomes a bounding box and image
crop. A trained classifier assigns a semantic class to each crop.

Outputs: segmentation masks, bounding boxes, class labels, confidence scores,
annotated images/videos, JSON prediction files.

Data sources:
- ArcGIS Packing House District — development and model preparation
- DroneStock footage — external testing on unseen drone scenes

## Pipeline
Raw Image → SAM → Mask Filtering → Bounding Box → Crop → Classifier →
Class + Confidence → Visualization

## Repo Conventions
- Classifier classes: building, tree, car, other (road excluded for now, handled
  in a later task once SAM's fragmented-mask issue on road regions is resolved)
- Never commit datasets, generated crops, or model checkpoints — verify .gitignore
  before every commit
- Branch-per-feature workflow; PRs target `develop`
- Config-driven scripts (no hardcoded paths/hyperparameters — use configs/*.yaml)

## Progress Log
(update this section as work happens — most recent entries at the top)

### 2026-08-17 — feature/classifier — status: tuned to the real dataset
Raw imagery is now in hand: **ArcGIS Packing House District**, 307 frames,
9504×6336 (60 MP), 6.57 GB, all geotagged, Sony ILCE-7RM4 + FE 24 mm.
Held **outside the repo** at `C:\Users\ehmed\Desktop\Redlands - Packing House
District\` (never copy it in; `data/` is gitignored). No labels, no boxes — as
expected, SAM has to produce crops first.

**Measured from the data** (drives the decisions below): ~88 m AGL → **~1.4 cm/px**,
frame footprint ~131 × 87 m, survey extent 375 × 785 m (~29 ha), shot spacing
28.8 m median → ~70–78% forward / ~65% side overlap → **every ground object
appears in ~12 frames**. A car measures ~330 × 150 px; after SAM's internal
resize to 1024 px it is only ~35 px, so tiling is worth raising with Dəniz.

**Changes made after inspecting two frames at native resolution:**
- [x] **Letterbox instead of stretch** (`LetterboxResize`, `data.preserve_aspect`).
      Crops are strongly non-square; squashing a 2.2:1 car into a square warps it
      toward roof proportions and discards a real class cue. Padding uses the
      ImageNet mean so borders normalise to ~0. The flag is stored in the
      checkpoint, so inference cannot preprocess differently from training.
- [x] **Inverse-frequency class weights** (`training.class_weights: auto`).
      The scene is mostly pavement, dirt, gravel and shadow, so `other` will
      dominate and an unweighted loss rewards always predicting it.
- [x] **Minimum crop size** (`data.min_crop_pixels: 32`, ~45 cm on the ground).
      Training-set hygiene only — inference still classifies every crop, because
      the pipeline contract needs one prediction per segment.
- [x] `load_classifier()` now returns a `LoadedClassifier` NamedTuple
      (model, classes, image_size, preserve_aspect) — **API change**, it used to
      be a 3-tuple.
- [x] Sample-crop generator emits realistic per-class aspect ratios, so the smoke
      test exercises the letterbox path rather than square placeholders.
- [x] `docs/annotation_guidelines.md`: measured-imagery table, **geographic
      block split** replacing the old split-by-frame rule, class-balance guidance,
      and 13 new edge cases observed in the actual frames (railway, construction
      machinery, mobile homes, sapling rows, dappled shadow, boulders, fences).

**Verified:** clean regenerate → train → single/batch inference; real crops cut
from frame `211021_181441_889.jpg` (car 170×350, canopy 450×420, road 490×220,
20×18 fragment) all classify and return the contract shape; min-crop guard skips
the 20×18 crop (33→32); class weights shift to 0.38/2.25 under 6:1 imbalance;
all three import layouts still work.

**Note:** predictions on real crops are wrong and near chance (~0.3 confidence) —
correct behaviour for a model trained on synthetic shapes. Real accuracy needs
labelled crops.

### 2026-08-15 — feature/classifier — status: code complete, PR blocked
Classifier stage (SAM crop → class + confidence) implemented end to end. Repo
initialized here (`main` → `develop` → `feature/classifier`); there was no
existing git history.

**Blocker:** no git remote is configured and the `gh` CLI is not installed on
this machine, so the branch could not be pushed and the PR
(feature/classifier → develop) could not be opened. Everything else is done and
verified. The PR body is drafted in `docs/pr_classifier.md` — add a remote,
push, and open the PR with it.

**How to run:**
```
python scripts/generate_sample_crops.py --config configs/classifier.yaml
python src/classification/train.py
python src/classification/inference.py outputs/crops/segment_17.png
```

**Delivered:**
- [x] `docs/annotation_guidelines.md` — per-class rules, edge-case table, and the
      `data/classifier/{train,validation}/<class>/` layout. Decision: a single
      **50% area-dominance rule** resolves mixed crops, and road/paved surfaces
      fall into `other` until the `road` class is introduced.
- [x] `scripts/generate_sample_crops.py` — writes ~8 synthetic PNGs per class per
      split into `data/classifier/`. Decision: commit the *generator*, never the
      crops, so the smoke test is reproducible on any machine.
- [x] `src/classification/dataset.py` — `CropDataset` + `create_dataloaders()`.
      Decision: label indices derive from the **sorted** class list
      (building=0, car=1, other=2, tree=3) so train and inference cannot drift
      apart if the config's class order changes. Vertical flips are included in
      augmentation because drone crops have no canonical orientation.
- [x] `src/classification/model.py` — `build_model(num_classes, backbone)`.
      Decision: **MobileNetV3-Small** as default backbone (~1.5M trainable
      params vs ResNet18's ~11M) because the classifier runs once per SAM mask,
      i.e. hundreds of times per frame; ResNet18 stays selectable via config for
      a later accuracy comparison. Pretrained-weight download failures degrade to
      random init with a warning so offline smoke tests still run.
- [x] `configs/classifier.yaml` — classes, dataset root/splits, image size,
      backbone, hyperparameters, checkpoint path, inference rounding.
- [x] `src/classification/config.py` — shared YAML loader + device resolver, so
      train and inference don't import each other.
- [x] `src/classification/train.py` — AdamW + CrossEntropyLoss loop, per-epoch
      train loss and validation accuracy, best-val checkpoint to
      `checkpoints/classifier.pt` (gitignored). Decision: the checkpoint carries
      `classes`, `backbone` and `image_size` alongside the weights so inference
      never silently drifts from a changed config.
      Smoke test: 3 epochs on the synthetic crops, loss 1.44 → 0.50, no errors.
- [x] `src/classification/inference.py` — `predict()` / `classify_crop()` +
      CLI. Verified on `outputs/crops/segment_17.png`:
      `{"segment_id": 17, "class": "building", "confidence": 0.53}`.
      Decision: `segment_id` is parsed with the regex `segment_(\d+)` so
      scene-prefixed names still work, and is `None` (not an error) when the
      filename doesn't follow the convention.

- [x] Integration hardening for `feature/integration` (Mustafa's pipeline):
      `predict_many()` / `classify_crops()` load the checkpoint once and batch
      the forward passes instead of reloading per crop; `segment_id` can be
      passed explicitly so the pipeline needn't rely on filename parsing; empty
      crop lists return `[]` without loading the model; the public API is
      exported from `classification/__init__.py`. Modules switched to relative
      imports (PEP 366 bootstrap for script runs) so they work as
      `classification.*`, `src.classification.*` **and** as direct scripts —
      this removes the need to settle the src-layout question before merging.
      `.gitignore` now allows text manifests under `data/manifests/` so the
      preprocessing task can commit reproducible image lists; imagery there is
      still ignored.

**Verified before PR:** clean-state rerun — generator → 3-epoch train run →
checkpoint saved → single and batch inference return the required JSON shape;
all three import layouts import successfully; `git status` clean with `data/`,
`outputs/` and `checkpoints/` ignored, and `git check-ignore` confirming
manifests are the only exception.

**Next:** push `feature/classifier` and open the PR (see the blocker above).

**Team coordination — open items owned elsewhere:**
- **Masked vs. plain bbox crops** (`feature/sam-segmentation`). If crops arrive
  background-zeroed, the 50%-dominance/mixed-content/occlusion rules in the
  annotation guidelines need revising and the ImageNet normalisation statistics
  shift. Recommendation: plain bbox crops, mask saved separately. Decide before
  bulk labelling.
- **Nobody owns the labelling itself.** It is the project's critical path and is
  absent from all three teammate task specs. Needs an owner, a per-class target
  (a few hundred crops each is a sane start), and a double-labelled overlap
  sample to check the guidelines actually produce annotator agreement.
- **Checkpoint distribution.** `checkpoints/classifier.pt` is gitignored, so the
  end-to-end pipeline needs an agreed source for the weights (release artifact or
  shared drive).
- **Class list ownership.** `classes` is duplicated between `classifier.yaml` and
  whatever the visualization stage uses for label colours; it should have one
  home.

Then: label real SAM crops per `docs/annotation_guidelines.md`, delete the
synthetic placeholders, retrain, and revisit the `road` class once SAM's
fragmented road masks are fixed.
