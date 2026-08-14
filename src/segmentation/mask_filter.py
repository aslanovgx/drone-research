def filter_masks(masks, min_area=500, min_stability=0.9, min_iou=0.85):
    filtered = []
    for m in masks:
        if (m["area"] >= min_area and
            m["stability_score"] >= min_stability and
            m["predicted_iou"] >= min_iou):
            filtered.append(m)
    return filtered