import cv2
import numpy as np


def generate_tiles(image, tile_size=1536, overlap=256):
    """
    Splits a large image into overlapping tiles.

    Returns a list of dicts: {"image": tile_array, "x_offset": int, "y_offset": int}
    x_offset/y_offset = tile's top-left corner position in the ORIGINAL image.
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    tiles = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)

            tile_img = image[y_start:y_end, x_start:x_end]
            tiles.append({
                "image": tile_img,
                "x_offset": x_start,
                "y_offset": y_start,
            })

            if x_end == w:
                break
            x += stride
        if y_end == h:
            break
        y += stride

    return tiles


def mask_to_bbox(mask_array):
    """Local bbox (x, y, w, h) within the tile, from a boolean segmentation mask."""
    ys, xs = np.where(mask_array)
    if len(xs) == 0:
        return None
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x), int(ys.max() - y)
    return [x, y, w, h]


def shift_mask_to_global(mask_dict, x_offset, y_offset, full_shape):
    """
    Places a tile-local mask into a full-size (original image) boolean array,
    and shifts bbox/point_coords to global (original image) coordinates.
    """
    seg_local = mask_dict["segmentation"]
    th, tw = seg_local.shape

    global_mask = np.zeros(full_shape[:2], dtype=bool)
    global_mask[y_offset:y_offset + th, x_offset:x_offset + tw] = seg_local

    local_bbox = mask_dict.get("bbox") or mask_to_bbox(seg_local)
    global_bbox = [
        local_bbox[0] + x_offset,
        local_bbox[1] + y_offset,
        local_bbox[2],
        local_bbox[3],
    ]

    new_mask = dict(mask_dict)
    new_mask["segmentation"] = global_mask
    new_mask["bbox"] = global_bbox
    return new_mask


def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union


def deduplicate_masks(masks, iou_threshold=0.7):
    """
    Removes duplicate masks caused by tile overlap.
    Keeps the mask with the higher predicted_iou / stability_score when two
    masks (likely the same object seen in two tiles) overlap significantly.
    """
    masks_sorted = sorted(
        masks, key=lambda m: m.get("predicted_iou", 0), reverse=True
    )

    kept = []
    for m in masks_sorted:
        is_duplicate = False
        for k in kept:
            # quick reject using bbox overlap before the expensive full-mask IoU
            if not _bbox_overlaps(m["bbox"], k["bbox"]):
                continue
            if mask_iou(m["segmentation"], k["segmentation"]) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(m)

    return kept


def _bbox_overlaps(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


def generate_masks_tiled(mask_generator, image, tile_size=1536, overlap=256, iou_threshold=0.7):
    """
    Runs SAM2 automatic mask generation over tiles of a large image and
    merges results back into original image coordinates.
    """
    tiles = generate_tiles(image, tile_size=tile_size, overlap=overlap)

    all_masks = []
    for i, tile in enumerate(tiles):
        tile_masks = mask_generator.generate(tile["image"])
        print(f"  Tile {i+1}/{len(tiles)}: {len(tile_masks)} raw masks")

        for m in tile_masks:
            global_m = shift_mask_to_global(
                m, tile["x_offset"], tile["y_offset"], image.shape
            )
            all_masks.append(global_m)

    print(f"Total masks before dedup: {len(all_masks)}")
    final_masks = deduplicate_masks(all_masks, iou_threshold=iou_threshold)
    print(f"Total masks after dedup: {len(final_masks)}")

    return final_masks