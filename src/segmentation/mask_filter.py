def touches_image_border(mask, image_shape=None, edge_tolerance=2):
    """
    Checks whether the mask touches the outer boundary of the entire original image.
    Objects touching outer image bounds are real scene objects at the camera frame edge.
    """
    img_shape = image_shape or mask.get("image_shape")
    if not img_shape:
        return False

    img_h, img_w = img_shape[:2]
    gx, gy, gw, gh = mask["bbox"]

    left = gx <= edge_tolerance
    top = gy <= edge_tolerance
    right = gx + gw >= img_w - edge_tolerance
    bottom = gy + gh >= img_h - edge_tolerance

    return bool(left or top or right or bottom)


def touches_tile_edge(mask, edge_tolerance=2):
    """
    Checks whether the mask touches the border of its originating tile.
    """
    if "touches_tile_edge" in mask and isinstance(mask["touches_tile_edge"], bool):
        return mask["touches_tile_edge"]

    tile_bounds = mask.get("tile_bounds")
    if not tile_bounds:
        return False

    tx, ty, tw, th = tile_bounds
    gx, gy, gw, gh = mask["bbox"]

    local_x = gx - tx
    local_y = gy - ty

    left = local_x <= edge_tolerance
    top = local_y <= edge_tolerance
    right = local_x + gw >= tw - edge_tolerance
    bottom = local_y + gh >= th - edge_tolerance

    return bool(left or top or right or bottom)


def touches_internal_tile_edge(mask, image_shape=None, edge_tolerance=2):
    """
    Checks if mask touches a tile border that lies INTERNALLY within the image grid.
    Tile borders coinciding with outer image boundaries are not internal tile edges.
    """
    tile_bounds = mask.get("tile_bounds")
    if not tile_bounds:
        return False

    img_shape = image_shape or mask.get("image_shape")
    tx, ty, tw, th = tile_bounds
    gx, gy, gw, gh = mask["bbox"]

    local_x = gx - tx
    local_y = gy - ty

    # Left tile edge is internal if tx > 0
    left = (local_x <= edge_tolerance) and (tx > 0)

    # Top tile edge is internal if ty > 0
    top = (local_y <= edge_tolerance) and (ty > 0)

    # Right tile edge is internal if tile extends further right into image grid
    right = False
    if local_x + gw >= tw - edge_tolerance:
        if img_shape:
            right = (tx + tw < img_shape[1])
        else:
            right = True

    # Bottom tile edge is internal if tile extends further down into image grid
    bottom = False
    if local_y + gh >= th - edge_tolerance:
        if img_shape:
            bottom = (ty + th < img_shape[0])
        else:
            bottom = True

    return bool(left or top or right or bottom)


def filter_masks_quality(
    masks,
    min_area=1500,
    min_stability_score=0.9,
    min_predicted_iou=0.85,
):
    """
    Initial quality filtering stage: removes tiny, unstable, or low-confidence masks.
    IMPORTANT: Preserves boundary-touching masks so they can be stitched across tiles.
    """
    filtered = []
    for m in masks:
        if m.get("area", 0) < min_area:
            continue
        if m.get("stability_score", 0.0) < min_stability_score:
            continue
        if m.get("predicted_iou", 0.0) < min_predicted_iou:
            continue
        filtered.append(m)
    return filtered


def filter_boundary_artifacts(masks, image_shape=None, edge_tolerance=2):
    """
    Post-stitching artifact filtering:
    1. Preserves successfully stitched/merged masks (is_merged=True).
    2. Preserves real scene objects touching outer image boundaries.
    3. Rejects unmerged suspicious internal tile-edge artifacts.
    """
    filtered = []
    for m in masks:
        # Rule 1: Always preserve merged/stitched masks
        if m.get("is_merged", False):
            filtered.append(m)
            continue

        # Rule 2: Always preserve objects touching outer image boundary
        if touches_image_border(m, image_shape=image_shape, edge_tolerance=edge_tolerance):
            filtered.append(m)
            continue

        # Rule 3: Reject unmerged internal tile edge artifacts
        if touches_internal_tile_edge(m, image_shape=image_shape, edge_tolerance=edge_tolerance):
            continue

        filtered.append(m)

    return filtered


def filter_masks(
    masks,
    min_area=500,
    min_stability_score=0.9,
    min_predicted_iou=0.85,
    reject_tile_edge=False,
    edge_tolerance=2,
    image_shape=None,
):
    """
    Pipeline-compatible filtering wrapper.
    First applies quality filters, then optionally applies boundary artifact rejection.
    """
    quality_filtered = filter_masks_quality(
        masks,
        min_area=min_area,
        min_stability_score=min_stability_score,
        min_predicted_iou=min_predicted_iou,
    )

    if reject_tile_edge:
        return filter_boundary_artifacts(quality_filtered, image_shape=image_shape, edge_tolerance=edge_tolerance)

    return quality_filtered


def filter_masks_from_config(masks, config, image_shape=None):
    """Reads mask filtering thresholds from configs/sam.yaml."""
    cfg = config.get("mask_filter", {})
    return filter_masks(
        masks,
        min_area=cfg.get("min_area", 1500),
        min_stability_score=cfg.get("min_stability_score", 0.9),
        min_predicted_iou=cfg.get("min_predicted_iou", 0.85),
        reject_tile_edge=cfg.get("reject_tile_edge", False),
        edge_tolerance=cfg.get("edge_tolerance", 2),
        image_shape=image_shape,
    )


def print_filter_stats(masks_before, masks_after):
    removed = len(masks_before) - len(masks_after)
    print(f"Masks before filtering: {len(masks_before)}")
    print(f"Masks after filtering:  {len(masks_after)}")
    print(
        f"Removed: {removed} "
        f"({removed / max(len(masks_before), 1) * 100:.1f}%)"
    )


def print_filter_breakdown(
    masks,
    min_area=1500,
    min_stability_score=0.9,
    min_predicted_iou=0.85,
    reject_tile_edge=False,
    edge_tolerance=2,
    image_shape=None,
):
    area_failed = sum(m.get("area", 0) < min_area for m in masks)
    stability_failed = sum(m.get("stability_score", 0) < min_stability_score for m in masks)
    iou_failed = sum(m.get("predicted_iou", 0) < min_predicted_iou for m in masks)
    internal_edge_failed = sum(touches_internal_tile_edge(m, image_shape=image_shape, edge_tolerance=edge_tolerance) and not m.get("is_merged", False) for m in masks)
    merged_count = sum(m.get("is_merged", False) for m in masks)

    print("=" * 60)
    print("FILTER BREAKDOWN")
    print("=" * 60)
    print(f"Total masks:                   {len(masks)}")
    print(f"Merged masks (is_merged=True): {merged_count}")
    print(f"Area < {min_area}:                  {area_failed}")
    print(f"Stability < {min_stability_score}:              {stability_failed}")
    print(f"Predicted IoU < {min_predicted_iou}:         {iou_failed}")
    print(f"Unmerged internal tile edges:  {internal_edge_failed}")
    print("=" * 60)
