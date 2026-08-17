# Pipeline Architecture

## Overview

The project detects and classifies objects in raw drone imagery.

The system uses the Segment Anything Model (SAM) to generate candidate
object masks. Each accepted mask is converted into a bounding box and
crop. A classifier assigns a semantic class to every accepted segment.

The integration pipeline combines the SAM and classifier predictions,
filters low-quality results, creates JSON output, and draws annotated
bounding boxes on the input image.

## Processing Flow

```text
Raw Drone Image
    → SAM Segmentation
    → Mask Filtering
    → Bounding Box and Crop Generation
    → Region Classification
    → Prediction Merging
    → JSON and Annotated Image
```

## Module Responsibilities

### Data preprocessing

Responsible for:

- preparing ArcGIS Packing House District images;
- extracting usable image frames when necessary;
- checking image formats and dimensions;
- organizing development and test data;
- keeping large datasets outside Git.

Expected image formats:

- `.jpg`;
- `.jpeg`;
- `.png`.

### SAM segmentation

Responsible for:

- loading the SAM model;
- generating candidate masks;
- filtering invalid or duplicate masks;
- extracting bounding boxes;
- saving optional object crops;
- returning `SegmentPrediction` objects.

Expected output:

```python
list[SegmentPrediction]
```

Example:

```python
SegmentPrediction(
    segment_id=1,
    bbox=BoundingBox(
        x=100,
        y=80,
        width=250,
        height=180,
    ),
    area=32500,
    sam_score=0.94,
    crop_path="outputs/crops/segment_1.jpg",
)
```

### Classifier

Responsible for:

- receiving the crop produced for each segment;
- predicting a semantic class;
- returning the predicted class and confidence;
- preserving the original `segment_id`.

Initial target classes may include:

- `building`;
- `tree`;
- `car`;
- `road`;
- `other`.

Expected output:

```python
list[ClassificationPrediction]
```

Example:

```python
ClassificationPrediction(
    segment_id=1,
    class_name="building",
    confidence=0.92,
)
```

### Integration pipeline

Responsible for:

- loading pipeline configuration;
- matching segmentation and classification results;
- applying quality thresholds;
- creating final `Detection` objects;
- saving JSON prediction files;
- creating annotated output images.

The integration pipeline matches predictions using `segment_id`.

List order must not be used for matching.

## Shared Data Contracts

Shared Pydantic models are defined in:

```text
src/utils/schemas.py
```

### BoundingBox

```python
BoundingBox(
    x=100,
    y=80,
    width=250,
    height=180,
)
```

Coordinates follow this format:

```text
x      = left coordinate
y      = top coordinate
width  = bounding box width
height = bounding box height
```

The bottom-right coordinate is calculated as:

```python
x2 = x + width
y2 = y + height
```

### SegmentPrediction

Represents one accepted SAM segment.

Required fields:

- `segment_id`;
- `bbox`;
- `area`;
- `sam_score`.

Optional field:

- `crop_path`.

### ClassificationPrediction

Represents the classifier result for one segment.

Required fields:

- `segment_id`;
- `class_name`;
- `confidence`.

### Detection

Represents the merged SAM and classifier result.

It contains:

- segment information from SAM;
- semantic class from the classifier;
- confidence scores;
- bounding box information.

### PipelineResult

Represents all accepted detections for one image.

Example:

```python
PipelineResult(
    image="data/image_001.jpg",
    detections=[
        building_detection,
        tree_detection,
    ],
)
```

## Matching Rule

SAM and classifier results must use the same `segment_id`.

Example:

```text
SAM segment_id=7
        +
Classifier segment_id=7
        ↓
Final Detection segment_id=7
```

If a segment has no classifier result, the integration pipeline skips it.

If the classifier produces results in a different list order, matching
still works because the pipeline uses `segment_id`.

## Filtering Rules

Filtering values are stored in:

```text
configs/pipeline.yaml
```

Current configuration:

```yaml
sam:
  min_score: 0.80
  min_mask_area: 500

classifier:
  min_confidence: 0.60

outputs:
  json_dir: outputs/json
  annotated_dir: outputs/predictions

visualization:
  box_thickness: 2
  show_confidence: true
```

A segment is excluded when:

- its SAM score is below `sam.min_score`;
- its mask area is below `sam.min_mask_area`;
- it has no matching classifier prediction;
- classifier confidence is below `classifier.min_confidence`.

## Output Files

For an input image named:

```text
test_input.jpg
```

the pipeline generates:

```text
outputs/json/test_input.json
outputs/predictions/test_input_annotated.jpg
```

Example JSON structure:

```json
{
  "image": "outputs/test_input.jpg",
  "detections": [
    {
      "segment_id": 1,
      "class_name": "building",
      "confidence": 0.92,
      "bbox": {
        "x": 100,
        "y": 80,
        "width": 250,
        "height": 180
      },
      "area": 32500,
      "sam_score": 0.94
    }
  ]
}
```

Generated datasets, model weights, crops, predictions, and other large
artifacts must not be committed to Git.

## Running Tests

Activate the virtual environment and run:

```bash
python -m pytest -v
```

The tests currently verify:

- matching predictions by `segment_id`;
- filtering using configured thresholds;
- safely skipping segments without classifier output.

All tests should pass before opening a pull request.

## Current Development Status

The integration pipeline currently works with prepared
`SegmentPrediction` and `ClassificationPrediction` objects.

Real SAM and classifier modules will be connected after their feature
branches are completed and reviewed.

DroneStock footage will be used as unseen external test data after the
development pipeline works on the ArcGIS dataset.