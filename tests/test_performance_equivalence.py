import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.tiling import (
    localize_mask,
    deduplicate_masks,
    stitch_tile_boundary_masks,
    cropped_mask_iou,
    can_stitch_candidate_pair,
    merge_two_masks,
    _bbox_overlaps,
)


def baseline_deduplicate_masks(masks, iou_threshold=0.7):
    """Original O(N^2) baseline deduplication for result equivalence testing."""
    masks_sorted = sorted(
        masks,
        key=lambda m: m.get("predicted_iou", 0),
        reverse=True
    )
    kept = []
    checks = 0
    for m in masks_sorted:
        is_duplicate = False
        for k in kept:
            checks += 1
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
    return kept, checks


def baseline_stitch_tile_boundary_masks(masks, ios_threshold=0.4, max_gap=15):
    """Original O(N^2) baseline stitching for result equivalence testing."""
    n = len(masks)
    if n <= 1:
        return masks, 0

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

    checks = 0
    for i in range(n):
        for j in range(i + 1, n):
            checks += 1
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

    return stitched_masks, checks


class TestPerformanceEquivalence(unittest.TestCase):

    def setUp(self):
        # Generate a synthetic grid of 50 masks across 4 tiles (9504x6336 space)
        np.random.seed(42)
        self.masks = []

        # Tile 1: [0, 0]
        for idx in range(15):
            seg = np.zeros((1536, 1536), dtype=bool)
            x_start = 100 + (idx * 60)
            y_start = 200 + (idx * 40)
            seg[y_start:y_start + 100, x_start:x_start + 100] = True
            m = localize_mask({
                "segmentation": seg,
                "predicted_iou": 0.80 + (idx % 10) * 0.015,
                "stability_score": 0.95,
            }, x_offset=0, y_offset=0, tile_w=1536, tile_h=1536)
            if m:
                self.masks.append(m)

        # Tile 2: [1280, 0] overlapping Tile 1
        for idx in range(15):
            seg = np.zeros((1536, 1536), dtype=bool)
            x_start = (idx * 50)
            y_start = 200 + (idx * 40)
            seg[y_start:y_start + 100, x_start:x_start + 100] = True
            m = localize_mask({
                "segmentation": seg,
                "predicted_iou": 0.82 + (idx % 10) * 0.012,
                "stability_score": 0.95,
            }, x_offset=1280, y_offset=0, tile_w=1536, tile_h=1536)
            if m:
                self.masks.append(m)

        # Tile 3: [5000, 3000] distant tile
        for idx in range(20):
            seg = np.zeros((1536, 1536), dtype=bool)
            x_start = 300 + (idx * 45)
            y_start = 300 + (idx * 30)
            seg[y_start:y_start + 80, x_start:x_start + 80] = True
            m = localize_mask({
                "segmentation": seg,
                "predicted_iou": 0.85 + (idx % 8) * 0.01,
                "stability_score": 0.96,
            }, x_offset=5000, y_offset=3000, tile_w=1536, tile_h=1536)
            if m:
                self.masks.append(m)

    def test_01_deduplication_equivalence_and_candidate_reduction(self):
        base_kept, base_checks = baseline_deduplicate_masks(self.masks, iou_threshold=0.7)
        opt_kept, stats = deduplicate_masks(self.masks, iou_threshold=0.7, return_stats=True)

        # Result equivalence check
        self.assertEqual(len(opt_kept), len(base_kept))
        for m_base, m_opt in zip(base_kept, opt_kept):
            self.assertEqual(m_base["bbox"], m_opt["bbox"])
            self.assertEqual(m_base["area"], m_opt["area"])
            self.assertAlmostEqual(m_base["predicted_iou"], m_opt["predicted_iou"])

        # Reduction check
        self.assertLess(stats["candidate_checks"], base_checks)
        print(f"\n[Dedup Performance] Baseline Checks: {base_checks} | Optimized Candidate Checks: {stats['candidate_checks']} | Reduction: {(1 - stats['candidate_checks']/max(1,base_checks))*100:.1f}%")

    def test_02_stitching_equivalence_and_candidate_reduction(self):
        base_stitched, base_checks = baseline_stitch_tile_boundary_masks(self.masks, ios_threshold=0.4, max_gap=15)
        opt_stitched, stats = stitch_tile_boundary_masks(self.masks, ios_threshold=0.4, max_gap=15, return_stats=True)

        # Result equivalence check
        self.assertEqual(len(opt_stitched), len(base_stitched))

        base_bboxes = sorted([m["bbox"] for m in base_stitched])
        opt_bboxes = sorted([m["bbox"] for m in opt_stitched])
        self.assertEqual(base_bboxes, opt_bboxes)

        base_merged = sorted([m["is_merged"] for m in base_stitched])
        opt_merged = sorted([m["is_merged"] for m in opt_stitched])
        self.assertEqual(base_merged, opt_merged)

        # Candidate reduction check
        self.assertLess(stats["actual_stitch_checks"], base_checks)
        print(f"\n[Stitching Performance] Baseline Pairwise Checks: {base_checks} | Optimized Candidate Pair Checks: {stats['actual_stitch_checks']} | Reduction: {(1 - stats['actual_stitch_checks']/max(1,base_checks))*100:.1f}%")


if __name__ == "__main__":
    unittest.main()
