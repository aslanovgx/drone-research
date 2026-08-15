def compute_solidity(mask):
    x, y, w, h = mask["bbox"]
    bbox_area = w * h
    if bbox_area == 0:
        return 0.0
    return mask["area"] / bbox_area


def filter_masks(masks, min_area=500, min_stability_score=0.9,
                  min_predicted_iou=0.85, max_solidity=0.92,
                  max_area_ratio=None, tile_area=None):
    """
    max_solidity: rejects near-perfect-rectangle masks, which are usually
    tile-boundary artifacts from segmenting flat/textured surfaces
    (road, dirt, uniform rooftops) rather than actual objects.
    """
    filtered = []
    for m in masks:
        if m["area"] < min_area:
            continue
        if m["stability_score"] < min_stability_score:
            continue
        if m["predicted_iou"] < min_predicted_iou:
            continue

        solidity = compute_solidity(m)
        if solidity > max_solidity:
            continue

        filtered.append(m)
    return filtered


def filter_masks_from_config(masks, config):
    """Convenience wrapper that reads filter thresholds from configs/sam.yaml"""
    cfg = config.get("mask_filter", {})
    return filter_masks(
        masks,
        min_area=cfg.get("min_area", 500),
        min_stability_score=cfg.get("min_stability_score", 0.9),
        min_predicted_iou=cfg.get("min_predicted_iou", 0.85),
        max_solidity=cfg.get("max_solidity", 0.92),
    )


def print_filter_stats(masks_before, masks_after, max_solidity=0.92):

    area_removed = sum(
        m["area"] < 1500
        for m in masks_before
    )

    stability_removed = sum(
        m["stability_score"] < 0.9
        for m in masks_before
    )

    iou_removed = sum(
        m["predicted_iou"] < 0.85
        for m in masks_before
    )

    solidity_removed = sum(
        compute_solidity(m) > max_solidity
        for m in masks_before
    )

    removed = len(masks_before) - len(masks_after)

    print(f"Masks before filtering: {len(masks_before)}")
    print(f"Masks after filtering:  {len(masks_after)}")
    print(
        f"Removed: {removed} "
        f"({removed / max(len(masks_before), 1) * 100:.1f}%)"
    )

    print()
    print("=" * 60)
    print("FILTER BREAKDOWN")
    print("=" * 60)

    print(f"Area < 1500:             {area_removed}")
    print(f"Stability < 0.9:         {stability_removed}")
    print(f"Predicted IoU < 0.85:    {iou_removed}")
    print(f"Solidity > {max_solidity}:          {solidity_removed}")
    print("=" * 60)

def print_filter_breakdown(masks, min_area=1500,
                           min_stability_score=0.9,
                           min_predicted_iou=0.85):

    total = len(masks)

    area_failed = sum(
        m["area"] < min_area
        for m in masks
    )

    stability_failed = sum(
        m["stability_score"] < min_stability_score
        for m in masks
    )

    iou_failed = sum(
        m["predicted_iou"] < min_predicted_iou
        for m in masks
    )

    combined_failed = sum(
        m["area"] < min_area
        or m["stability_score"] < min_stability_score
        or m["predicted_iou"] < min_predicted_iou
        for m in masks
    )

    print()
    print("=" * 60)
    print("FILTER BREAKDOWN")
    print("=" * 60)

    print(f"Total masks:              {total}")
    print(f"Area < {min_area}:             {area_failed}")
    print(f"Stability < {min_stability_score}:        {stability_failed}")
    print(f"Predicted IoU < {min_predicted_iou}:    {iou_failed}")
    print(f"Failed at least one:       {combined_failed}")
    print("=" * 60)