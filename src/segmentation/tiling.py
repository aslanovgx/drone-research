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
                "y_offset": int
            },
            ...
        ]

    x_offset/y_offset are the tile's top-left coordinates
    in the ORIGINAL image.
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

            # Make sure the last tile is still tile_size when possible
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
    """
    Calculates bbox from a boolean segmentation mask.

    Returns:
        [x, y, width, height]

    Coordinates are LOCAL to the tile.
    """
    ys, xs = np.where(mask_array)

    if len(xs) == 0:
        return None

    x = int(xs.min())
    y = int(ys.min())

    # +1 is important because bbox dimensions are inclusive
    w = int(xs.max() - x + 1)
    h = int(ys.max() - y + 1)

    return [x, y, w, h]


def localize_mask(mask_dict, x_offset, y_offset):
    """
    Converts a tile-local SAM mask into a memory-efficient representation.

    Instead of storing a full tile-sized or full-image-sized mask,
    only the mask inside its own bounding box is stored.

    Example:

        Original image:
            9504 x 6336

        Object bbox:
            200 x 300

        Stored segmentation:
            200 x 300

    The global position is preserved using:
        - bbox
        - mask_offset
    """

    seg_local = mask_dict["segmentation"]

    # Get SAM bbox if available, otherwise calculate it
    local_bbox = mask_dict.get("bbox") or mask_to_bbox(seg_local)

    if local_bbox is None:
        return None

    lx, ly, lw, lh = [int(v) for v in local_bbox]

    # Invalid bbox
    if lw <= 0 or lh <= 0:
        return None

    # Make sure bbox does not go outside the segmentation array
    mask_h, mask_w = seg_local.shape[:2]

    lx = max(0, lx)
    ly = max(0, ly)

    x2 = min(lx + lw, mask_w)
    y2 = min(ly + lh, mask_h)

    if x2 <= lx or y2 <= ly:
        return None

    # Crop mask to ONLY its bounding box
    cropped_mask = seg_local[ly:y2, lx:x2]

    cropped_h, cropped_w = cropped_mask.shape[:2]

    if cropped_h == 0 or cropped_w == 0:
        return None

    # Convert local tile coordinates to global image coordinates
    global_x = lx + x_offset
    global_y = ly + y_offset

    global_bbox = [
        global_x,
        global_y,
        cropped_w,
        cropped_h,
    ]

    new_mask = dict(mask_dict)

    # Store only the small cropped mask
    new_mask["segmentation"] = cropped_mask

    # Bbox is now in ORIGINAL image coordinates
    new_mask["bbox"] = global_bbox

    # Explicit global offset of the cropped mask
    new_mask["mask_offset"] = (
        global_x,
        global_y,
    )

    return new_mask


def _bbox_overlaps(b1, b2):
    """
    Fast bbox overlap check.

    b = [x, y, width, height]
    """

    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    return not (
        x1 + w1 <= x2
        or x2 + w2 <= x1
        or y1 + h1 <= y2
        or y2 + h2 <= y1
    )


def cropped_mask_iou(mask_a, offset_a, mask_b, offset_b):
    """
    Calculates IoU between two bbox-cropped masks.

    IMPORTANT:
    The masks can have different sizes and different global positions.

    No full-image-sized mask is ever created.
    """

    ax, ay = offset_a
    bx, by = offset_b

    ah, aw = mask_a.shape[:2]
    bh, bw = mask_b.shape[:2]

    # Global bounding boxes of the cropped masks
    a_x2 = ax + aw
    a_y2 = ay + ah

    b_x2 = bx + bw
    b_y2 = by + bh

    # Find intersection rectangle
    x1 = max(ax, bx)
    y1 = max(ay, by)

    x2 = min(a_x2, b_x2)
    y2 = min(a_y2, b_y2)

    # No spatial overlap
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Extract only the overlapping area
    a_sub = mask_a[
        y1 - ay:y2 - ay,
        x1 - ax:x2 - ax
    ]

    b_sub = mask_b[
        y1 - by:y2 - by,
        x1 - bx:x2 - bx
    ]

    # Intersection
    intersection = np.logical_and(
        a_sub,
        b_sub
    ).sum()

    if intersection == 0:
        return 0.0

    # Mask areas
    area_a = mask_a.sum()
    area_b = mask_b.sum()

    # Union
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def deduplicate_masks(masks, iou_threshold=0.7):
    """
    Removes duplicate masks produced because neighboring tiles overlap.

    Masks are sorted by predicted_iou so that stronger SAM predictions
    are kept first.
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

            # Fast bbox check before expensive mask IoU
            if not _bbox_overlaps(
                m["bbox"],
                k["bbox"]
            ):
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
    iou_threshold=0.7
):
    """
    Runs SAM automatic mask generation over overlapping tiles.

    Memory-efficient version:
        - SAM works on each tile separately.
        - Masks are cropped to their bbox.
        - No full-image-sized boolean mask is created.
        - Duplicate masks from overlapping tiles are removed.
    """

    tiles = generate_tiles(
        image,
        tile_size=tile_size,
        overlap=overlap
    )

    all_masks = []

    print(f"Total tiles: {len(tiles)}")

    for i, tile in enumerate(tiles):

        tile_image = tile["image"]

        tile_masks = mask_generator.generate(tile_image)

        print(
            f"  Tile {i + 1}/{len(tiles)}: "
            f"{len(tile_masks)} raw masks"
        )

        for m in tile_masks:

            localized = localize_mask(
                m,
                tile["x_offset"],
                tile["y_offset"]
            )

            if localized is not None:
                all_masks.append(localized)

    print(
        f"Total masks before dedup: "
        f"{len(all_masks)}"
    )

    final_masks = deduplicate_masks(
        all_masks,
        iou_threshold=iou_threshold
    )

    print(
        f"Total masks after dedup: "
        f"{len(final_masks)}"
    )

    return final_masks