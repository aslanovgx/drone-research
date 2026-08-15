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
| `src/classification/dataset.py` | `CropDataset` + `create_dataloaders()`; resize, ImageNet normalisation, train-only augmentation |
| `src/classification/model.py` | `build_model(num_classes, backbone)` — MobileNetV3-Small (default) / ResNet18 with a 4-class head |
| `src/classification/config.py` | Shared YAML config loading and device resolution |
| `src/classification/train.py` | Train/validation loop, CrossEntropyLoss + AdamW, per-epoch logging, checkpointing |
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
  `classify_crops()` per frame.
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
- **The checkpoint stores `classes`, `backbone` and `image_size`** alongside the
  weights. Inference trusts the checkpoint over the config, so an edited config
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

- 44 synthetic crops generated across both splits and all four classes.
- Training completed 3 epochs without errors; train loss 1.49 → 0.47; checkpoint
  written to `checkpoints/classifier.pt`.
- Inference returned exactly:
  `{"segment_id": 17, "class": "building", "confidence": 0.4}`
- `git status` clean; `data/`, `outputs/` and `checkpoints/` confirmed ignored.

Accuracy figures from this run are meaningless — the crops are synthetic shapes.
The purpose is that the loop runs, converges on separable input, and saves a
loadable checkpoint.

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
