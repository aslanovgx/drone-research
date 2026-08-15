import cv2
import numpy as np


def generate_tiles(image, tile_size=1536, overlap=256):
    """
    Splits a large image into overlapping tiles.

    Returns:
        [
            {
                "image": tile_array,
                "x_offset": int,
                "y_offset": int,
                "tile_w": int,
                "tile_h": int
            },
            ...
        ]
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
            actual_h, actual_w = tile_img.shape[:2]

            tiles.append({
                "image": tile_img,
                "x_offset": x_start,
                "y_offset": y_start,
                "tile_w": actual_w,
                "tile_h": actual_h,
            })

            if x_end == w:
                break

            x += stride

        if y_end == h:
            break

        y += stride

    return tiles


def mask_to_bbox(mask_array):
    """
    Calculates bbox from a boolean segmentation mask.

    Returns:
        [x, y, width, height]
    """
    ys, xs = np.where(mask_array)

    if len(xs) == 0:
        return None

    x = int(xs.min())
    y = int(ys.min())

    w = int(xs.max() - x + 1)
    h = int(ys.max() - y + 1)

    return [x, y, w, h]


def localize_mask(mask_dict, x_offset, y_offset, tile_w=1536, tile_h=1536, edge_tolerance=2, image_shape=None):
    """
    Converts a tile-local SAM mask into a memory-efficient representation.
    Crops mask to its bounding box, stores global offset, tile boundary bounds, and image shape.
    """
    seg_local = mask_dict["segmentation"]

    local_bbox = mask_dict.get("bbox") or mask_to_bbox(seg_local)

    if local_bbox is None:
        return None

    lx, ly, lw, lh = [int(v) for v in local_bbox]

    if lw <= 0 or lh <= 0:
        return None

    mask_h, mask_w = seg_local.shape[:2]

    lx = max(0, lx)
    ly = max(0, ly)

    x2 = min(lx + lw, mask_w)
    y2 = min(ly + lh, mask_h)

    if x2 <= lx or y2 <= ly:
        return None

    cropped_mask = seg_local[ly:y2, lx:x2]
    cropped_h, cropped_w = cropped_mask.shape[:2]

    if cropped_h == 0 or cropped_w == 0:
        return None

    global_x = lx + x_offset
    global_y = ly + y_offset

    global_bbox = [
        global_x,
        global_y,
        cropped_w,
        cropped_h,
    ]

    touches_edge = bool(
        lx <= edge_tolerance
        or ly <= edge_tolerance
        or x2 >= tile_w - edge_tolerance
        or y2 >= tile_h - edge_tolerance
    )

    new_mask = dict(mask_dict)
    new_mask["segmentation"] = cropped_mask
    new_mask["bbox"] = global_bbox
    new_mask["mask_offset"] = (global_x, global_y)
    new_mask["area"] = int(cropped_mask.sum())
    new_mask["tile_bounds"] = (x_offset, y_offset, tile_w, tile_h)
    new_mask["touches_tile_edge"] = touches_edge
    new_mask["is_merged"] = False
    if image_shape:
        new_mask["image_shape"] = image_shape

    return new_mask



def _bbox_overlaps(b1, b2):
    """Fast bbox overlap check. b = [x, y, width, height]"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    return not (
        x1 + w1 <= x2
        or x2 + w2 <= x1
        or y1 + h1 <= y2
        or y2 + h2 <= y1
    )


def cropped_mask_iou(mask_a, offset_a, mask_b, offset_b):
    """Calculates IoU between two bbox-cropped masks."""
    ax, ay = offset_a
    bx, by = offset_b

    ah, aw = mask_a.shape[:2]
    bh, bw = mask_b.shape[:2]

    a_x2 = ax + aw
    a_y2 = ay + ah

    b_x2 = bx + bw
    b_y2 = by + bh

    x1 = max(ax, bx)
    y1 = max(ay, by)

    x2 = min(a_x2, b_x2)
    y2 = min(a_y2, b_y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    a_sub = mask_a[y1 - ay:y2 - ay, x1 - ax:x2 - ax]
    b_sub = mask_b[y1 - by:y2 - by, x1 - bx:x2 - bx]

    intersection = np.logical_and(a_sub, b_sub).sum()

    if intersection == 0:
        return 0.0

    area_a = mask_a.sum()
    area_b = mask_b.sum()

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def cropped_mask_ios(mask_a, offset_a, mask_b, offset_b):
    """Calculates IoS (Intersection over Smaller mask) between two cropped masks."""
    ax, ay = offset_a
    bx, by = offset_b

    ah, aw = mask_a.shape[:2]
    bh, bw = mask_b.shape[:2]

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    a_sub = mask_a[y1 - ay:y2 - ay, x1 - ax:x2 - ax]
    b_sub = mask_b[y1 - by:y2 - by, x1 - bx:x2 - bx]

    intersection = float(np.logical_and(a_sub, b_sub).sum())
    if intersection == 0:
        return 0.0

    min_area = float(min(mask_a.sum(), mask_b.sum()))
    if min_area <= 0:
        return 0.0

    return float(intersection / min_area)


def merge_two_masks(mask_a, mask_b):
    """
    Combines two localized masks into a single unified localized mask.
    """
    ax, ay = mask_a["mask_offset"]
    ah, aw = mask_a["segmentation"].shape[:2]

    bx, by = mask_b["mask_offset"]
    bh, bw = mask_b["segmentation"].shape[:2]

    merged_x1 = min(ax, bx)
    merged_y1 = min(ay, by)
    merged_x2 = max(ax + aw, bx + bw)
    merged_y2 = max(ay + ah, by + bh)

    merged_w = merged_x2 - merged_x1
    merged_h = merged_y2 - merged_y1

    merged_seg = np.zeros((merged_h, merged_w), dtype=bool)

    merged_seg[ay - merged_y1 : ay - merged_y1 + ah, ax - merged_x1 : ax - merged_x1 + aw] |= mask_a["segmentation"]
    merged_seg[by - merged_y1 : by - merged_y1 + bh, bx - merged_x1 : bx - merged_x1 + bw] |= mask_b["segmentation"]

    merged_area = int(merged_seg.sum())

    merged_mask = dict(mask_a)
    merged_mask["segmentation"] = merged_seg
    merged_mask["bbox"] = [merged_x1, merged_y1, merged_w, merged_h]
    merged_mask["mask_offset"] = (merged_x1, merged_y1)
    merged_mask["area"] = merged_area
    merged_mask["predicted_iou"] = max(mask_a.get("predicted_iou", 0), mask_b.get("predicted_iou", 0))
    merged_mask["stability_score"] = max(mask_a.get("stability_score", 0), mask_b.get("stability_score", 0))
    merged_mask["is_merged"] = True
    merged_mask["touches_tile_edge"] = mask_a.get("touches_tile_edge", False) or mask_b.get("touches_tile_edge", False)

    return merged_mask


def can_stitch_candidate_pair(mask_a, mask_b, ios_threshold=0.4, max_gap=15):
    """
    Multi-condition evaluation to decide if two masks across tile boundaries should be stitched.
    Conditions:
    1. Originating from different tile bounds or at least one touches tile edge.
    2. Spatial proximity: Bounding boxes overlap or are within max_gap distance.
    3. Meaningful mask overlap/connectivity in tile overlap region.
    4. IoS exceeds threshold (or high border adjacency).
    5. Geometry check: prevents merging unrelated background regions.
    """
    tile_a = mask_a.get("tile_bounds")
    tile_b = mask_b.get("tile_bounds")

    # If both are in same tile and neither touches edge, standard deduplication handles it
    if tile_a and tile_b and tile_a == tile_b and not (mask_a.get("touches_tile_edge") or mask_b.get("touches_tile_edge")):
        return False

    ax, ay, aw, ah = mask_a["bbox"]
    bx, by, bw, bh = mask_b["bbox"]

    # Spatial proximity check with max_gap
    if ax + aw + max_gap < bx or bx + bw + max_gap < ax:
        return False
    if ay + ah + max_gap < by or by + bh + max_gap < ay:
        return False

    # Check IoS
    ios = cropped_mask_ios(mask_a["segmentation"], mask_a["mask_offset"], mask_b["segmentation"], mask_b["mask_offset"])

    # Spatial overlap or border connectivity check
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    overlap_w = max(0, x2 - x1)
    overlap_h = max(0, y2 - y1)
    has_overlap_zone = (overlap_w > 0 and overlap_h > 0)

    # Geometry check: avoid merging massive background regions unless IoS is very strong
    area_a = float(mask_a.get("area", 0))
    area_b = float(mask_b.get("area", 0))
    if min(area_a, area_b) > 0:
        area_ratio = max(area_a, area_b) / min(area_a, area_b)
        if area_ratio > 10.0 and ios < 0.6:
            return False

    if ios >= ios_threshold:
        return True

    # Check border touch connectivity (masks touching along border with IoS > 0.25)
    if has_overlap_zone and ios >= 0.25 and (mask_a.get("touches_tile_edge") or mask_b.get("touches_tile_edge")):
        return True

    return False


def stitch_tile_boundary_masks(masks, ios_threshold=0.4, max_gap=15):
    """
    Stitches split masks from adjacent overlapping tiles using multi-condition candidate matching.
    """
    n = len(masks)
    if n <= 1:
        return masks

    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, n):
            if can_stitch_candidate_pair(masks[i], masks[j], ios_threshold=ios_threshold, max_gap=max_gap):
                union(i, j)

    components = {}
    for i in range(n):
        root = find(i)
        components.setdefault(root, []).append(masks[i])

    stitched_masks = []
    for root, comp_masks in components.items():
        if len(comp_masks) == 1:
            stitched_masks.append(comp_masks[0])
        else:
            merged = comp_masks[0]
            for next_m in comp_masks[1:]:
                merged = merge_two_masks(merged, next_m)
            stitched_masks.append(merged)

    return stitched_masks


def deduplicate_masks(masks, iou_threshold=0.7):
    """
    Removes duplicate masks produced because neighboring tiles overlap.
    Masks are sorted by predicted_iou so that stronger SAM predictions are kept first.
    """
    masks_sorted = sorted(
        masks,
        key=lambda m: m.get("predicted_iou", 0),
        reverse=True
    )

    kept = []

    for m in masks_sorted:
        is_duplicate = False

        for k in kept:
            if not _bbox_overlaps(m["bbox"], k["bbox"]):
                continue

            iou = cropped_mask_iou(
                m["segmentation"],
                m["mask_offset"],
                k["segmentation"],
                k["mask_offset"],
            )

            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(m)

    return kept


def generate_masks_tiled(
    mask_generator,
    image,
    tile_size=1536,
    overlap=256,
    iou_threshold=0.7,
    ios_threshold=0.4,
    min_area=1500,
    min_stability_score=0.9,
    min_predicted_iou=0.85,
    reject_tile_edge=False,
    edge_tolerance=2,
):
    """
    Runs SAM automatic mask generation over overlapping tiles with the complete pipeline:
    1. Tile Generation & Mask Localization (preserving tile bounds and image shape).
    2. Quality Filtering (min_area, stability, predicted_iou) - boundary masks preserved.
    3. Within-tile local deduplication.
    4. Multi-condition boundary-aware stitching across adjacent tiles.
    5. Post-stitching deduplication & boundary artifact filtering.
    """
    from segmentation.mask_filter import filter_masks_quality, filter_boundary_artifacts

    import torch

    img_shape = image.shape[:2]
    tiles = generate_tiles(image, tile_size=tile_size, overlap=overlap)
    all_raw_masks = []

    print(f"Total tiles: {len(tiles)}")

    for i, tile in enumerate(tiles):
        tile_image = tile["image"]
        
        with torch.inference_mode():
            tile_masks = mask_generator.generate(tile_image)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Tile {i + 1}/{len(tiles)}: {len(tile_masks)} raw masks")

        for m in tile_masks:
            localized = localize_mask(
                m,
                tile["x_offset"],
                tile["y_offset"],
                tile_w=tile["tile_w"],
                tile_h=tile["tile_h"],
                edge_tolerance=edge_tolerance,
                image_shape=img_shape,
            )
            if localized is not None:
                all_raw_masks.append(localized)


    print(f"Total localized masks: {len(all_raw_masks)}")

    # Step 1: Initial Quality Filtering (preserves boundary masks)
    quality_masks = filter_masks_quality(
        all_raw_masks,
        min_area=min_area,
        min_stability_score=min_stability_score,
        min_predicted_iou=min_predicted_iou,
    )
    print(f"Masks after quality filter: {len(quality_masks)}")

    # Step 2: Local / Within-tile deduplication
    local_dedup_masks = deduplicate_masks(quality_masks, iou_threshold=iou_threshold)
    print(f"Masks after local deduplication: {len(local_dedup_masks)}")

    # Step 3: Boundary-aware stitching
    stitched_masks = stitch_tile_boundary_masks(local_dedup_masks, ios_threshold=ios_threshold)
    print(f"Masks after boundary stitching: {len(stitched_masks)}")

    # Step 4: Post-stitching deduplication & artifact filtering
    final_masks = deduplicate_masks(stitched_masks, iou_threshold=iou_threshold)

    if reject_tile_edge:
        final_masks = filter_boundary_artifacts(final_masks, image_shape=img_shape, edge_tolerance=edge_tolerance)

    print(f"Final masks: {len(final_masks)}")

    return final_masks
