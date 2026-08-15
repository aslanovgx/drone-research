import sys
import os
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.tiling import (
    localize_mask,
    mask_to_bbox,
    cropped_mask_iou,
    cropped_mask_ios,
    merge_two_masks,
    can_stitch_candidate_pair,
    stitch_tile_boundary_masks,
    deduplicate_masks,
)

from segmentation.mask_filter import (
    touches_tile_edge,
    filter_masks_quality,
    filter_boundary_artifacts,
    filter_masks,
)

from segmentation.bbox_extractor import (
    mask_to_bbox_xywh,
    generate_crop,
)


class TestTilingPipeline(unittest.TestCase):

    def setUp(self):
        self.tile_w = 1536
        self.tile_h = 1536

    # -------------------------------------------------------------------------
    # Test 1: A mask completely inside a tile
    # -------------------------------------------------------------------------
    def test_01_mask_completely_inside_tile(self):
        seg = np.zeros((1536, 1536), dtype=bool)
        seg[500:700, 500:700] = True  # 200x200 square inside tile
        mask_dict = {
            "segmentation": seg,
            "area": 40000,
            "predicted_iou": 0.95,
            "stability_score": 0.98,
        }
        localized = localize_mask(mask_dict, x_offset=0, y_offset=0, tile_w=1536, tile_h=1536)

        self.assertIsNotNone(localized)
        self.assertFalse(localized["touches_tile_edge"])
        self.assertEqual(localized["bbox"], [500, 500, 200, 200])
        self.assertEqual(localized["mask_offset"], (500, 500))

    # -------------------------------------------------------------------------
    # Test 2: A mask touching one tile edge (Preserved during quality filter)
    # -------------------------------------------------------------------------
    def test_02_mask_touching_one_tile_edge(self):
        seg = np.zeros((1536, 1536), dtype=bool)
        seg[500:700, 0:100] = True  # Touches left edge (x=0)
        mask_dict = {
            "segmentation": seg,
            "area": 20000,
            "predicted_iou": 0.92,
            "stability_score": 0.95,
        }
        localized = localize_mask(mask_dict, x_offset=0, y_offset=0, tile_w=1536, tile_h=1536)

        self.assertIsNotNone(localized)
        self.assertTrue(localized["touches_tile_edge"])

        # Quality filter MUST preserve boundary-touching masks before stitching
        filtered = filter_masks_quality([localized], min_area=1500, min_stability_score=0.9, min_predicted_iou=0.85)
        self.assertEqual(len(filtered), 1)

    # -------------------------------------------------------------------------
    # Test 3: A real object split across two overlapping tiles (Successfully stitched)
    # -------------------------------------------------------------------------
    def test_03_real_object_split_across_two_tiles(self):
        # Object spans from x=1200 to x=1600 in global coordinates.
        # Tile 1: [0..1536] (x_offset=0). Left part of object: [1200..1536] (local: 1200..1536)
        seg1 = np.zeros((1536, 1536), dtype=bool)
        seg1[500:600, 1200:1536] = True
        m1 = localize_mask({"segmentation": seg1, "predicted_iou": 0.9, "stability_score": 0.95}, 0, 0, 1536, 1536)

        # Tile 2: [1280..2816] (x_offset=1280). Right part of object: [1280..1600] (local: 0..320)
        seg2 = np.zeros((1536, 1536), dtype=bool)
        seg2[500:600, 0:320] = True
        m2 = localize_mask({"segmentation": seg2, "predicted_iou": 0.92, "stability_score": 0.96}, 1280, 0, 1536, 1536)

        self.assertIsNotNone(m1)
        self.assertIsNotNone(m2)

        # Multi-condition stitching check
        can_stitch = can_stitch_candidate_pair(m1, m2, ios_threshold=0.4)
        self.assertTrue(can_stitch)

        stitched = stitch_tile_boundary_masks([m1, m2], ios_threshold=0.4)
        self.assertEqual(len(stitched), 1)
        merged = stitched[0]
        self.assertTrue(merged["is_merged"])
        # Global object spans x=1200 to x=1600 (width=400), y=500 to y=600 (height=100)
        self.assertEqual(merged["bbox"], [1200, 500, 400, 100])

    # -------------------------------------------------------------------------
    # Test 4: Two unrelated masks near the same tile boundary (NOT merged)
    # -------------------------------------------------------------------------
    def test_04_unrelated_masks_near_tile_boundary(self):
        # Mask A at y=100..200
        seg1 = np.zeros((1536, 1536), dtype=bool)
        seg1[100:200, 1400:1536] = True
        m1 = localize_mask({"segmentation": seg1, "predicted_iou": 0.9, "stability_score": 0.9}, 0, 0, 1536, 1536)

        # Mask B at y=800..900 (far away vertically, no overlap)
        seg2 = np.zeros((1536, 1536), dtype=bool)
        seg2[800:900, 0:136] = True
        m2 = localize_mask({"segmentation": seg2, "predicted_iou": 0.9, "stability_score": 0.9}, 1280, 0, 1536, 1536)

        can_stitch = can_stitch_candidate_pair(m1, m2, ios_threshold=0.4)
        self.assertFalse(can_stitch)

        stitched = stitch_tile_boundary_masks([m1, m2], ios_threshold=0.4)
        self.assertEqual(len(stitched), 2)

    # -------------------------------------------------------------------------
    # Test 5: Two identical masks from overlapping tiles (Deduplicated)
    # -------------------------------------------------------------------------
    def test_05_two_identical_masks_from_overlapping_tiles(self):
        # Identical global object at x=1350..1450, y=500..600
        seg1 = np.zeros((1536, 1536), dtype=bool)
        seg1[500:600, 1350:1450] = True
        m1 = localize_mask({"segmentation": seg1, "predicted_iou": 0.95, "stability_score": 0.95}, 0, 0, 1536, 1536)

        # In tile 2 (offset 1280), object is at local x=70..170
        seg2 = np.zeros((1536, 1536), dtype=bool)
        seg2[500:600, 70:170] = True
        m2 = localize_mask({"segmentation": seg2, "predicted_iou": 0.88, "stability_score": 0.90}, 1280, 0, 1536, 1536)

        iou = cropped_mask_iou(m1["segmentation"], m1["mask_offset"], m2["segmentation"], m2["mask_offset"])
        self.assertAlmostEqual(iou, 1.0, places=3)

        dedup = deduplicate_masks([m1, m2], iou_threshold=0.7)
        self.assertEqual(len(dedup), 1)
        self.assertEqual(dedup[0]["predicted_iou"], 0.95)

    # -------------------------------------------------------------------------
    # Test 6: A mask whose bbox touches image boundary (x=0 or y=0)
    # -------------------------------------------------------------------------
    def test_06_mask_bbox_touches_image_boundary(self):
        seg = np.zeros((1536, 1536), dtype=bool)
        seg[0:50, 0:50] = True  # Top-left image corner
        m = localize_mask({"segmentation": seg, "predicted_iou": 0.9, "stability_score": 0.9}, 0, 0, 1536, 1536)

        self.assertIsNotNone(m)
        self.assertEqual(m["bbox"], [0, 0, 50, 50])
        self.assertEqual(m["mask_offset"], (0, 0))

    # -------------------------------------------------------------------------
    # Test 7: Empty segmentation array
    # -------------------------------------------------------------------------
    def test_07_empty_segmentation(self):
        empty_seg = np.zeros((100, 100), dtype=bool)
        bbox = mask_to_bbox(empty_seg)
        self.assertIsNone(bbox)

        localized = localize_mask({"segmentation": empty_seg}, 0, 0)
        self.assertIsNone(localized)

    # -------------------------------------------------------------------------
    # Test 8: Zero-area mask
    # -------------------------------------------------------------------------
    def test_08_zero_area_mask(self):
        m = {
            "segmentation": np.zeros((10, 10), dtype=bool),
            "bbox": [10, 10, 10, 10],
            "area": 0,
            "stability_score": 0.95,
            "predicted_iou": 0.90,
        }
        filtered = filter_masks_quality([m], min_area=1500)
        self.assertEqual(len(filtered), 0)

    # -------------------------------------------------------------------------
    # Test 9: Cropping near image boundaries (bbox_extractor safety)
    # -------------------------------------------------------------------------
    def test_09_cropping_near_image_boundaries(self):
        dummy_img = np.ones((500, 500, 3), dtype=np.uint8) * 128
        out_crop_path = "outputs/test_crop.png"

        # Bbox going outside image boundaries
        out_of_bounds_bbox = [450, 450, 100, 100]  # extends to 550x550
        generate_crop(dummy_img, out_of_bounds_bbox, out_crop_path, size=(64, 64))

        self.assertTrue(os.path.exists(out_crop_path))
        if os.path.exists(out_crop_path):
            os.remove(out_crop_path)

    # -------------------------------------------------------------------------
    # Test 10: Merged mask bbox/area correctness
    # -------------------------------------------------------------------------
    def test_10_merged_mask_bbox_and_area_correctness(self):
        seg1 = np.ones((50, 50), dtype=bool)
        m1 = {
            "segmentation": seg1,
            "bbox": [100, 100, 50, 50],
            "mask_offset": (100, 100),
            "area": 2500,
            "predicted_iou": 0.9,
            "stability_score": 0.9,
        }
        seg2 = np.ones((50, 50), dtype=bool)
        m2 = {
            "segmentation": seg2,
            "bbox": [130, 100, 50, 50],
            "mask_offset": (130, 100),
            "area": 2500,
            "predicted_iou": 0.95,
            "stability_score": 0.95,
        }
        merged = merge_two_masks(m1, m2)
        self.assertEqual(merged["bbox"], [100, 100, 80, 50])
        self.assertEqual(merged["area"], 4000)
        self.assertEqual(merged["area"], int(merged["segmentation"].sum()))
        self.assertTrue(merged["is_merged"])

    # -------------------------------------------------------------------------
    # Test 11: Merged boundary object is preserved
    # -------------------------------------------------------------------------
    def test_11_merged_boundary_object_preserved(self):
        merged_mask = {
            "segmentation": np.ones((50, 50), dtype=bool),
            "bbox": [1280, 500, 100, 50],
            "mask_offset": (1280, 500),
            "area": 5000,
            "tile_bounds": (1280, 0, 1536, 1536),
            "touches_tile_edge": True,
            "is_merged": True,
        }
        res = filter_boundary_artifacts([merged_mask], image_shape=(3000, 3000))
        self.assertEqual(len(res), 1)
        self.assertTrue(res[0]["is_merged"])

    # -------------------------------------------------------------------------
    # Test 12: Unrelated internal tile-edge artifact is rejected
    # -------------------------------------------------------------------------
    def test_12_unrelated_tile_edge_artifact_rejected(self):
        artifact_mask = {
            "segmentation": np.ones((50, 50), dtype=bool),
            "bbox": [1280, 500, 50, 50],
            "mask_offset": (1280, 500),
            "area": 2500,
            "tile_bounds": (1280, 0, 1536, 1536),
            "touches_tile_edge": True,
            "is_merged": False,
        }
        res = filter_boundary_artifacts([artifact_mask], image_shape=(3000, 3000))
        self.assertEqual(len(res), 0)

    # -------------------------------------------------------------------------
    # Test 13: Image-border object is preserved
    # -------------------------------------------------------------------------
    def test_13_image_border_object_preserved(self):
        img_border_mask = {
            "segmentation": np.ones((50, 50), dtype=bool),
            "bbox": [0, 500, 50, 50],
            "mask_offset": (0, 500),
            "area": 2500,
            "tile_bounds": (0, 0, 1536, 1536),
            "touches_tile_edge": True,
            "is_merged": False,
        }
        res = filter_boundary_artifacts([img_border_mask], image_shape=(3000, 3000))
        self.assertEqual(len(res), 1)

    # -------------------------------------------------------------------------
    # Test 14: Normal interior object is preserved
    # -------------------------------------------------------------------------
    def test_14_normal_interior_object_preserved(self):
        interior_mask = {
            "segmentation": np.ones((50, 50), dtype=bool),
            "bbox": [500, 500, 50, 50],
            "mask_offset": (500, 500),
            "area": 2500,
            "tile_bounds": (0, 0, 1536, 1536),
            "touches_tile_edge": False,
            "is_merged": False,
        }
        res = filter_boundary_artifacts([interior_mask], image_shape=(3000, 3000))
        self.assertEqual(len(res), 1)

    # -------------------------------------------------------------------------
    # Test 15: Export segments JSON format and crop generation
    # -------------------------------------------------------------------------
    def test_15_export_segments_json_format(self):
        from segmentation.bbox_extractor import export_segments
        import json

        dummy_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 200
        masks = [{
            "segmentation": np.ones((50, 50), dtype=bool),
            "bbox": [120, 85, 240, 190],
            "mask_offset": (120, 85),
            "area": 32840,
            "predicted_iou": 0.94,
            "stability_score": 0.96,
        }]

        crops_dir = "outputs/test_crops"
        json_path = "outputs/test_segmentation_results.json"

        results = export_segments(dummy_img, masks, crops_dir=crops_dir, json_path=json_path)

        self.assertEqual(len(results), 1)
        item = results[0]

        # Verify exact required keys
        self.assertEqual(item["segment_id"], 0)
        self.assertEqual(item["bbox"], [120, 85, 240, 190])
        self.assertEqual(item["area"], 32840)
        self.assertEqual(item["sam_score"], 0.94)
        self.assertEqual(item["crop_path"], "outputs/test_crops/segment_0.png")

        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists("outputs/test_crops/segment_0.png"))

        # Clean up test output
        if os.path.exists(json_path):
            os.remove(json_path)
        if os.path.exists("outputs/test_crops/segment_0.png"):
            os.remove("outputs/test_crops/segment_0.png")


if __name__ == "__main__":
    unittest.main()



