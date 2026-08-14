import cv2
import json
import os

def mask_to_bbox_xywh(mask):
    """Returns [x, y, width, height] format as required."""
    seg = mask["segmentation"]
    ys, xs = seg.nonzero()
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x), int(ys.max() - y)
    return [x, y, w, h]

def generate_crop(image, bbox, out_path, size=(224, 224)):
    x, y, w, h = bbox
    crop = image[y:y+h, x:x+w]
    crop = cv2.resize(crop, size)
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
            "area": m["area"],
            "sam_score": round(m["predicted_iou"], 4),
            "crop_path": crop_path
        })

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    return results