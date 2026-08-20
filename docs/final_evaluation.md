# Final Evaluation Report

## Project objective

The project implements an end-to-end drone imagery pipeline:

1. SAM 2.1 automatic mask generation
2. Mask filtering and deduplication
3. Bounding-box and crop extraction
4. MobileNetV3-Small crop classification
5. Class and confidence assignment
6. JSON export and annotated-image visualization

The classifier uses four classes:

- `building`
- `car`
- `tree`
- `other`

## Dataset preparation

The ArcGIS Packing House District dataset contains 307 drone images at
9504 × 6336 resolution.

A geographically separated source split was used to reduce leakage caused by
the same physical object appearing in overlapping drone frames.

SAM crops were manually reviewed and labelled.

| Split | Building | Car | Tree | Other | Total |
|---|---:|---:|---:|---:|---:|
| Train | 49 | 63 | 57 | 118 | 287 |
| Validation | 39 | 25 | 42 | 97 | 203 |
| Total | 88 | 88 | 99 | 215 | 490 |

An exact-file duplicate check found no duplicates shared between the train and
validation splits.

## Classifier evaluation

The selected classifier baseline is MobileNetV3-Small with ImageNet
pretraining and unweighted cross-entropy loss.

| Metric | Result |
|---|---:|
| Validation samples | 203 |
| Validation accuracy | 75.86% |
| Macro F1 | 0.7717 |
| Best epoch | 2 |

### Per-class results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Building | 0.6250 | 0.6410 | 0.6329 |
| Car | 0.8000 | 0.9600 | 0.8727 |
| Other | 0.7952 | 0.6804 | 0.7333 |
| Tree | 0.7800 | 0.9286 | 0.8478 |

Cars and trees produced the strongest validation results. Building remains the
most difficult target because roof crops are visually similar to pavement,
shadows and other urban surfaces.

## Held-out ArcGIS tests

The complete SAM-to-classifier pipeline was also tested on geographically
held-out ArcGIS source images.

Observed behaviour:

- Car-heavy scenes produced high-confidence car detections.
- Tree-heavy scenes produced consistent tree detections.
- Large building roofs were less reliable and were sometimes classified as
  `other`.
- The confidence threshold suppressed uncertain building candidates.

These tests were not used for classifier training.

## DroneStock external-domain evaluation

Two independent stock drone videos were selected:

- Suburban houses surrounded by trees
- Central Park, Manhattan, viewed from above

Three frames were extracted from each video at approximately 20%, 50% and 80%
of the duration. All six frames were processed using one fixed configuration;
no frame-specific threshold tuning was performed.

| Scene | Frame | SAM segments | Final detections | Building | Car | Tree | Other | Mean confidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Central Park | 20% | 5 | 2 | 0 | 0 | 1 | 1 | 0.7150 |
| Central Park | 50% | 6 | 2 | 0 | 0 | 0 | 2 | 0.6700 |
| Central Park | 80% | 5 | 1 | 0 | 0 | 0 | 1 | 0.6000 |
| Suburbs | 20% | 13 | 6 | 0 | 0 | 6 | 0 | 0.8583 |
| Suburbs | 50% | 26 | 14 | 0 | 0 | 8 | 6 | 0.7650 |
| Suburbs | 80% | 21 | 10 | 0 | 0 | 5 | 5 | 0.7290 |

The pipeline completed successfully for all six external frames and generated:

- segmentation JSON
- crop images
- final detection JSON
- annotated output images
- execution logs

### External-test interpretation

The Suburbs frames demonstrate useful transfer for tree detection. However,
building and car detections did not transfer reliably to these stock videos.

The Central Park frames produced few SAM segments, showing that segmentation
recall is currently a major bottleneck for small objects and dense tree
canopies at 1920 × 1080 resolution.

These DroneStock frames do not have manually annotated ground truth.
Consequently, an accuracy value is not reported for this test. It is treated as
a qualitative domain-shift evaluation rather than a replacement for the
labelled ArcGIS validation set.

## Current limitations

- Only 490 real crops from a 307-image dataset were labelled.
- Building roofs are frequently confused with the `other` class.
- SAM parameters were originally developed around larger native-resolution
  ArcGIS patches.
- Small cars may disappear at stock-video resolution.
- The classifier was trained on one geographic and camera domain.
- The current pipeline processes selected frames rather than producing a
  temporally tracked final video.

## Recommended follow-up work

1. Generate and label crops from more of the 307 ArcGIS images.
2. Add more complete building-roof examples and difficult `other` negatives.
3. Add small and partially occluded car examples.
4. Evaluate SAM recall separately using manually annotated test frames.
5. Tune segmentation for 1920 × 1080 video frames.
6. Add DroneStock-domain samples only to a separate adaptation training split.
7. Keep a fully untouched final test set.
8. Report per-class precision, recall, F1 and confusion matrices after every
   retraining run.
9. Add temporal tracking and annotated-video export.

## Conclusion

The project provides a reproducible working baseline that connects SAM 2.1,
crop generation, supervised classification, confidence filtering, JSON export
and visualization.

The measured classifier baseline reaches 75.86% validation accuracy and 0.7717
macro F1. External DroneStock testing confirms that the pipeline executes on
previously unseen footage while also revealing the expected domain-shift and
small-object limitations that should guide future team work.
