import json
from pathlib import Path

import cv2
import numpy as np


def mask_to_bbox_xywh(mask):
    """Return [x, y, width, height] for a SAM mask."""

    if "bbox" in mask and mask["bbox"] is not None:
        return [int(value) for value in mask["bbox"]]

    segmentation = mask.get("segmentation")

    if segmentation is None:
        return [0, 0, 0, 0]

    ys, xs = segmentation.nonzero()

    if len(xs) == 0:
        return [0, 0, 0, 0]

    x = int(xs.min())
    y = int(ys.min())
    width = int(xs.max() - x + 1)
    height = int(ys.max() - y + 1)

    offset_x, offset_y = mask.get(
        "mask_offset",
        (0, 0),
    )

    return [
        x + offset_x,
        y + offset_y,
        width,
        height,
    ]


def generate_crop(
    image,
    bbox,
    output_path,
    size=(224, 224),
):
    """Create and save a classifier-ready crop."""

    image_height, image_width = image.shape[:2]
    x, y, width, height = bbox

    x1 = max(0, min(int(x), image_width))
    y1 = max(0, min(int(y), image_height))
    x2 = max(0, min(int(x + width), image_width))
    y2 = max(0, min(int(y + height), image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bounding box for crop: {bbox}"
        )

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError(
            f"Empty crop generated for bounding box: {bbox}"
        )

    crop = cv2.resize(
        crop,
        size,
        interpolation=cv2.INTER_AREA,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    saved = cv2.imwrite(
        str(path),
        cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2BGR,
        ),
    )

    if not saved:
        raise OSError(
            f"Crop could not be saved: {path}"
        )

    return path.as_posix()


def export_segments(
    image,
    masks,
    crops_dir,
    json_path,
):
    """Export valid SAM segments using the shared schema format."""

    results = []

    for mask in masks:
        bbox_values = mask_to_bbox_xywh(mask)
        x, y, width, height = bbox_values
        area = int(mask.get("area", 0))

        if width <= 0 or height <= 0 or area <= 0:
            continue

        segment_id = len(results)

        crop_path = (
            Path(crops_dir)
            / f"segment_{segment_id}.png"
        )

        saved_crop_path = generate_crop(
            image=image,
            bbox=bbox_values,
            output_path=crop_path,
        )

        sam_score = round(
            float(mask.get("predicted_iou", 0.0)),
            4,
        )

        sam_score = max(
            0.0,
            min(sam_score, 1.0),
        )

        results.append(
            {
                "segment_id": segment_id,
                "bbox": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },
                "area": area,
                "sam_score": sam_score,
                "crop_path": saved_crop_path,
            }
        )

    output_path = Path(json_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    return results