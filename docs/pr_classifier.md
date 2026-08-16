# PR: Classifier stage (feature/classifier → develop)

Draft body for the pull request from `feature/classifier` into `develop`.

## Summary

Implements the classifier stage of the pipeline: takes one crop produced by SAM
(`outputs/crops/segment_<id>.png`) and returns a class and a confidence score.

Classes: `building`, `tree`, `car`, `other`. `road` is deliberately excluded
until SAM's fragmented road masks are resolved; road-like crops fall into
`other`.

## What was implemented

| File | Purpose |
|---|---|
| `docs/annotation_guidelines.md` | Per-class inclusion/exclusion rules, edge-case decision table, dataset layout, labelling procedure |
| `scripts/generate_sample_crops.py` | Synthetic placeholder crops so the pipeline can be exercised before real labels exist |
| `src/classification/dataset.py` | `CropDataset` + `create_dataloaders()`; aspect-preserving letterbox, ImageNet normalisation, train-only augmentation, minimum-crop-size filter |
| `src/classification/model.py` | `build_model(num_classes, backbone)` — MobileNetV3-Small (default) / ResNet18 with a 4-class head |
| `src/classification/config.py` | Shared YAML config loading and device resolution |
| `src/classification/train.py` | Train/validation loop, class-weighted CrossEntropyLoss + AdamW, per-epoch logging, early stopping, best-validation checkpointing |
| `src/classification/inference.py` | Single-crop prediction + CLI |
| `configs/classifier.yaml` | All classes, paths and hyperparameters |

## Integration surface (for `feature/integration`)

The classifier output matches the agreed contract exactly:

```json
{"segment_id": 17, "class": "building", "confidence": 0.92}
```

Classify every crop of one frame, loading the checkpoint once:

```python
from classification import predict_many

predictions = predict_many(crop_paths, segment_ids=segment_ids)
```

- `segment_ids` is optional — ids are parsed from `segment_<id>.png` filenames
  when omitted — but the pipeline should pass them explicitly, since it already
  holds the SAM metadata and shouldn't depend on a naming convention.
- An empty crop list returns `[]` without loading the model, so frames where SAM
  finds nothing cost nothing and cannot crash.
- For a long-lived process, call `load_classifier()` once and then
  `classify_crops()` per frame:

  ```python
  clf = load_classifier("checkpoints/classifier.pt", device)
  predictions = classify_crops(
      crop_paths, clf.model, clf.classes, clf.image_size, device,
      preserve_aspect=clf.preserve_aspect,
  )
  ```

  `load_classifier()` returns a `LoadedClassifier` NamedTuple
  (`model, classes, image_size, preserve_aspect`) so preprocessing settings
  travel with the weights and inference cannot diverge from training.
- `predict()` remains for one-off single-crop use; it reloads the checkpoint each
  call.
- Modules use relative imports, so they work as `classification.*`, as
  `src.classification.*`, and as directly executed scripts. The pipeline can pick
  either layout without changes here.
- Masked (alpha-channel) crops are flattened via `convert("RGB")`. See the open
  question below.

## Key design decisions

- **MobileNetV3-Small as the default backbone.** The classifier runs once per SAM
  mask — hundreds of times per frame — so inference cost dominates. ~1.5M
  trainable parameters vs ResNet18's ~11M, and it runs comfortably on CPU.
  ResNet18 stays selectable via `model.backbone` for an accuracy comparison once
  real labels exist.
- **Label indices come from the *sorted* class list** (`building=0, car=1,
  other=2, tree=3`), not from the order in the config, so reordering the YAML
  cannot silently permute labels between training and inference.
- **Letterbox rather than stretch** (`data.preserve_aspect`). Measured on the real
  imagery, a car crop is about 330×150 px — 2.2:1. Squashing that into a square
  warps it toward roof proportions and discards a cue that separates the classes.
  Padding uses the ImageNet mean so borders normalise to ~0.
- **Inverse-frequency class weights** (`training.class_weights: auto`). Nadir
  urban scenes are dominated by pavement, dirt, gravel and shadow, so `other`
  will outnumber the target classes heavily and an unweighted loss rewards
  predicting it for everything.
- **Minimum crop size** (`data.min_crop_pixels: 32`, ~45 cm on the ground at this
  GSD) as training-set hygiene. Inference deliberately does *not* filter — the
  pipeline contract needs one prediction per segment.
- **25 epochs with early stopping** (patience 5). Three was a smoke-test figure;
  a real fine-tune needs more, and patience keeps a generous budget from
  overfitting the tail while the best-validation checkpoint is what gets saved.
- **The checkpoint stores `classes`, `backbone`, `image_size` and
  `preserve_aspect`** alongside the weights. Inference trusts the checkpoint over
  the config, so an edited config
  cannot mismatch a trained model.
- **A single 50% area-dominance rule** decides mixed crops in the annotation
  guidelines, with `other` as the tie-breaker. Ambiguity resolves toward `other`
  because a false `building`/`tree`/`car` costs more than an extra `other`.
- **Only the crop *generator* is committed, never crops.** Keeps the smoke test
  reproducible on any machine while `data/classifier/**` stays out of git.
- **Vertical flips are part of augmentation** — drone crops have no canonical
  orientation.
- **`segment_id` is parsed with `segment_(\d+)`**, so scene-prefixed filenames
  (`scene3_segment_17.png`) work; it is `None` rather than an error when the
  filename doesn't follow the convention.

## How it was verified

Clean-state end-to-end run (checkpoints and dataset deleted first):

```bash
python scripts/generate_sample_crops.py --config configs/classifier.yaml
python src/classification/train.py
python src/classification/inference.py outputs/crops/segment_17.png
```

- 44 synthetic crops generated across both splits and all four classes, with
  realistic per-class aspect ratios so letterboxing is exercised.
- Training ran to early stop at epoch 11 (best validation at epoch 6) in 15 s;
  checkpoint written to `checkpoints/classifier.pt`.
- Single-crop and batch inference both return the contract shape; explicit
  `segment_id` overrides filename parsing; an empty list returns `[]`.
- All three import layouts (`classification.*`, `src.classification.*`, direct
  script execution) verified.
- `git status` clean; `data/`, `outputs/` and `checkpoints/` confirmed ignored,
  with `git check-ignore` confirming `data/manifests/*.json` is the only
  exception.

**Also verified against the real imagery.** Four crops were cut from frame
`211021_181441_889.jpg` of the ArcGIS Packing House District set — a car
(170×350), a tree canopy (450×420), road surface (490×220) and a deliberate
20×18 fragment. All four flow through inference and return valid predictions;
the minimum-crop-size guard skips the 20×18 fragment during training; class
weights shift to 0.38/2.25 under a 6:1 imbalance.

Accuracy figures are meaningless — the model is trained on synthetic shapes, and
on real crops it predicts at roughly chance (~0.3 confidence), which is the
expected result. What is verified is that the loop runs, converges on separable
input, saves a loadable checkpoint, and handles real 60 MP-derived crops.

## What the real imagery told us

The ArcGIS Packing House District set is now in hand — 307 frames, 9504×6336,
6.57 GB, all geotagged, Sony ILCE-7RM4 with a 24 mm lens. Measured from it:
~88 m AGL giving **~1.4 cm/px**, a frame footprint of ~131 × 87 m over a
375 × 785 m survey area, and 28.8 m median shot spacing.

That works out to roughly 70–78% forward and ~65% side overlap, meaning **each
ground object appears in about 12 frames**. The annotation guidelines previously
said to split by source image; that is not sufficient here, because the same
physical car would land in both splits. The guidelines now specify a
**geographic block split** using the EXIF GPS positions, and note that the
unique-object count — not the crop count — is the real ceiling on dataset size.

A car also measures ~330 × 150 px in a raw frame but only ~35 px after SAM's
internal resize to 1024 px, which is worth weighing when deciding whether the
segmentation stage tiles its input.

## Open question for the team

**Masked crops or plain bounding-box crops?** `feature/sam-segmentation` says
"masked image crops". The annotation guidelines in this PR assume crops carry
*surrounding context* — the 50% dominance rule, the "mixed content" category and
the occlusion cases all presuppose that other objects can appear in the crop. If
crops arrive background-zeroed, several of those rules become dead letters, and
black padding also shifts the ImageNet normalisation statistics the backbone
expects. Recommendation: emit plain bbox crops and save the mask separately.
This needs deciding before bulk labelling starts, since it changes the rules.

## Follow-ups (not in this PR)

- Label real SAM crops per `docs/annotation_guidelines.md`, delete the synthetic
  placeholders, and retrain.
- Add a `road` class once SAM's road-mask fragmentation is addressed.
- Class-imbalance handling and a proper evaluation report (per-class precision/
  recall, confusion matrix) once real labels exist.
- Batch inference over a whole `outputs/crops/` directory for the visualization
  stage.
