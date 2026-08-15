"""Classifier stage of the drone imagery pipeline.

Takes an image crop produced by the SAM stage and assigns one of the target
classes (building, tree, car, other) with a confidence score.

Public API for the pipeline
---------------------------
Classify every crop of one frame, loading the checkpoint once::

    from classification import predict_many

    predictions = predict_many(
        ["outputs/crops/segment_17.png", "outputs/crops/segment_18.png"],
        segment_ids=[17, 18],          # optional; parsed from filenames otherwise
    )
    # [{"segment_id": 17, "class": "building", "confidence": 0.92}, ...]

For a long-lived process, load the model once and reuse it::

    from classification import load_classifier, classify_crops, load_config, resolve_device

    config = load_config()
    device = resolve_device()
    model, classes, image_size = load_classifier("checkpoints/classifier.pt", device)
    predictions = classify_crops(crop_paths, model, classes, image_size, device)

:func:`predict` handles the single-crop case but reloads the checkpoint per call.
"""

from .config import load_config, resolve_device
from .dataset import CropDataset, class_to_index, create_dataloaders, index_to_class
from .inference import (
    Prediction,
    classify_crop,
    classify_crops,
    load_classifier,
    parse_segment_id,
    predict,
    predict_many,
)
from .model import build_model

__all__ = [
    "CropDataset",
    "Prediction",
    "build_model",
    "class_to_index",
    "classify_crop",
    "classify_crops",
    "create_dataloaders",
    "index_to_class",
    "load_classifier",
    "load_config",
    "parse_segment_id",
    "predict",
    "predict_many",
    "resolve_device",
]
