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

**Verified before PR:** clean-state rerun — generator → 3-epoch train run →
checkpoint saved → inference returns the required JSON shape; `git status` clean
with `data/`, `outputs/` and `checkpoints/` ignored.

**Next:** push `feature/classifier` and open the PR; then label real SAM crops
per `docs/annotation_guidelines.md`, delete the synthetic placeholders, retrain,
and revisit the `road` class once SAM's fragmented road masks are fixed.
