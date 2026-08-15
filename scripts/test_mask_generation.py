import sys
sys.path.append("src")

import cv2
import numpy as np

from segmentation.sam_model import (
    load_config,
    load_sam_model,
    generate_masks
)

from segmentation.mask_filter import (
    filter_masks_from_config,
    print_filter_stats,
    print_filter_breakdown
)


def show_masks_overlay(image, masks, output_path, max_masks=None):
    """
    Draw bbox-cropped SAM masks back onto the original image.
    """

    overlay = image.copy()

    if max_masks is not None:
        masks_to_show = masks[:max_masks]
    else:
        masks_to_show = masks

    for i, mask in enumerate(masks_to_show):

        segmentation = mask["segmentation"]

        x, y = mask["mask_offset"]

        h, w = segmentation.shape[:2]

        image_h, image_w = image.shape[:2]

        x2 = min(x + w, image_w)
        y2 = min(y + h, image_h)

        if x >= image_w or y >= image_h:
            continue

        actual_w = x2 - x
        actual_h = y2 - y

        if actual_w <= 0 or actual_h <= 0:
            continue

        segmentation = segmentation[:actual_h, :actual_w]

        color = np.random.randint(
            0,
            256,
            size=3,
            dtype=np.uint8
        )

        region = overlay[y:y2, x:x2]

        region[segmentation] = (
            0.5 * region[segmentation].astype(np.float32)
            + 0.5 * color.astype(np.float32)
        ).astype(np.uint8)

        overlay[y:y2, x:x2] = region

        cv2.rectangle(
            overlay,
            (x, y),
            (x2 - 1, y2 - 1),
            color.tolist(),
            2
        )

        cv2.putText(
            overlay,
            str(i + 1),
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color.tolist(),
            2
        )

    cv2.imwrite(output_path, overlay)

    print(f"Overlay saved: {output_path}")


def print_mask_statistics(masks, name="Masks"):
    """
    Prints area, predicted IoU and stability score statistics.
    """

    if not masks:
        print(f"{name}: no masks found.")
        return


    areas = np.array([
        m["area"]
        for m in masks
        if "area" in m
    ])

    predicted_ious = np.array([
        m["predicted_iou"]
        for m in masks
        if "predicted_iou" in m
    ])

    stability_scores = np.array([
        m["stability_score"]
        for m in masks
        if "stability_score" in m
    ])

    print()
    print("=" * 60)
    print(f"{name.upper()} STATISTICS")
    print("=" * 60)

    # -------------------------------
    # AREA
    # -------------------------------

    if len(areas) > 0:
        print("\nArea:")
        print(f"  Min:    {areas.min():.0f}")
        print(f"  Max:    {areas.max():.0f}")
        print(f"  Mean:   {areas.mean():.0f}")
        print(f"  Median: {np.median(areas):.0f}")

        print("\nArea percentiles:")
        print(f"  P10:    {np.percentile(areas, 10):.0f}")
        print(f"  P25:    {np.percentile(areas, 25):.0f}")
        print(f"  P50:    {np.percentile(areas, 50):.0f}")
        print(f"  P75:    {np.percentile(areas, 75):.0f}")
        print(f"  P90:    {np.percentile(areas, 90):.0f}")

    # -------------------------------
    # PREDICTED IOU
    # -------------------------------

    if len(predicted_ious) > 0:
        print("\nPredicted IoU:")
        print(f"  Min:    {predicted_ious.min():.4f}")
        print(f"  Max:    {predicted_ious.max():.4f}")
        print(f"  Mean:   {predicted_ious.mean():.4f}")
        print(f"  Median: {np.median(predicted_ious):.4f}")

        print("\nPredicted IoU percentiles:")
        print(f"  P10:    {np.percentile(predicted_ious, 10):.4f}")
        print(f"  P25:    {np.percentile(predicted_ious, 25):.4f}")
        print(f"  P50:    {np.percentile(predicted_ious, 50):.4f}")
        print(f"  P75:    {np.percentile(predicted_ious, 75):.4f}")
        print(f"  P90:    {np.percentile(predicted_ious, 90):.4f}")

    # -------------------------------
    # STABILITY SCORE
    # -------------------------------

    if len(stability_scores) > 0:
        print("\nStability score:")
        print(f"  Min:    {stability_scores.min():.4f}")
        print(f"  Max:    {stability_scores.max():.4f}")
        print(f"  Mean:   {stability_scores.mean():.4f}")
        print(f"  Median: {np.median(stability_scores):.4f}")

        print("\nStability percentiles:")
        print(f"  P10:    {np.percentile(stability_scores, 10):.4f}")
        print(f"  P25:    {np.percentile(stability_scores, 25):.4f}")
        print(f"  P50:    {np.percentile(stability_scores, 50):.4f}")
        print(f"  P75:    {np.percentile(stability_scores, 75):.4f}")
        print(f"  P90:    {np.percentile(stability_scores, 90):.4f}")


# --------------------------------------------------
# 1. CONFIG
# --------------------------------------------------

config = load_config()


# --------------------------------------------------
# 2. LOAD SAM MODEL
# --------------------------------------------------

mask_generator = load_sam_model(config)


# --------------------------------------------------
# 3. GENERATE MASKS
# --------------------------------------------------

image_path = "data/samples/sample_02.jpg"

image, masks = generate_masks(
    mask_generator,
    image_path,
    config=config
)


# --------------------------------------------------
# 4. MASK INFORMATION
# --------------------------------------------------

print()
print("=" * 60)
print("MASK GENERATION")
print("=" * 60)

print(f"Total masks found: {len(masks)}")

if len(masks) > 0:
    print("First mask keys:", list(masks[0].keys()))
    print("First mask area:", masks[0]["area"])
    print("First mask shape:", masks[0]["segmentation"].shape)
    print("First mask offset:", masks[0]["mask_offset"])
    print("First mask bbox:", masks[0]["bbox"])

# --------------------------------------------------
# 5. FILTER MASKS
# --------------------------------------------------

img_shape = image.shape[:2]

filtered = filter_masks_from_config(
    masks,
    config,
    image_shape=img_shape
)

print()
print("=" * 60)
print("FILTER RESULTS")
print("=" * 60)

print_filter_stats(
    masks,
    filtered
)
print_filter_breakdown(
    masks,
    min_area=config["mask_filter"]["min_area"],
    min_stability_score=config["mask_filter"]["min_stability_score"],
    min_predicted_iou=config["mask_filter"]["min_predicted_iou"],
    reject_tile_edge=config["mask_filter"].get("reject_tile_edge", False),
    edge_tolerance=config["mask_filter"].get("edge_tolerance", 2),
    image_shape=img_shape,
)


# --------------------------------------------------
# 6. STATISTICS
# --------------------------------------------------

print_mask_statistics(
    masks,
    "Before filtering"
)

print_mask_statistics(
    filtered,
    "After filtering"
)


# --------------------------------------------------
# 7. LOAD ORIGINAL IMAGE
# --------------------------------------------------

original_image = cv2.imread(image_path)

if original_image is None:
    raise FileNotFoundError(
        f"Image tapılmadı: {image_path}"
    )


# --------------------------------------------------
# 8. VISUALIZATION
# --------------------------------------------------

print()
print("=" * 60)
print("VISUALIZATION")
print("=" * 60)

show_masks_overlay(
    original_image,
    filtered,
    "data/samples/sample_02_masks.jpg",
    max_masks=None
)

print()
print("=" * 60)
print("AREA THRESHOLD EXPERIMENT")
print("=" * 60)

for area_threshold in [500, 1000, 1500, 3000, 5000]:

    test_filtered = [
        m for m in masks
        if m["area"] >= area_threshold
        and m["stability_score"] >= 0.9
        and m["predicted_iou"] >= 0.85
    ]

    print(
        f"min_area={area_threshold:5d} "
        f"-> {len(test_filtered):4d} masks"
    )