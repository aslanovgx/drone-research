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

### 2026-08-15 — feature/classifier — status: in progress
Repo initialized (`main` → `develop` → `feature/classifier`). CLAUDE.md and
.gitignore seeded.
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
Next: model.py, configs/classifier.yaml, train.py, inference.py.
