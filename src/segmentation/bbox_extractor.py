import cv2
import json
import numpy as np
import os


def mask_to_bbox_xywh(mask):
    """Returns global [x, y, width, height] format for any mask dictionary."""
    if "bbox" in mask and mask["bbox"] is not None:
        return [int(v) for v in mask["bbox"]]

    seg = mask.get("segmentation")
    if seg is None:
        return [0, 0, 0, 0]

    ys, xs = seg.nonzero()
    if len(xs) == 0:
        return [0, 0, 0, 0]

    x, y = int(xs.min()), int(ys.min())
    w = int(xs.max() - x + 1)
    h = int(ys.max() - y + 1)

    ox, oy = mask.get("mask_offset", (0, 0))
    return [x + ox, y + oy, w, h]


def generate_crop(image, bbox, out_path, size=(224, 224)):
    """Safely crops bbox from image, handling boundary and empty cases."""
    img_h, img_w = image.shape[:2]
    x, y, w, h = bbox

    x1 = max(0, min(int(x), img_w))
    y1 = max(0, min(int(y), img_h))
    x2 = max(0, min(int(x + w), img_w))
    y2 = max(0, min(int(y + h), img_h))

    if x2 <= x1 or y2 <= y1:
        crop = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    else:
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, size)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


def export_segments(image, masks, crops_dir, json_path):
    os.makedirs(crops_dir, exist_ok=True)
    results = []
    for i, m in enumerate(masks):
        bbox = mask_to_bbox_xywh(m)
        crop_path = f"{crops_dir}/segment_{i}.png"
        generate_crop(image, bbox, crop_path)

        results.append({
            "segment_id": i,
            "bbox": bbox,
            "area": int(m.get("area", 0)),
            "sam_score": round(float(m.get("predicted_iou", 0.0)), 4),
            "crop_path": crop_path
        })

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    return results