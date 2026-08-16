import os
import sys
import unittest
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.video_processing import (
    convert_and_clamp_bbox,
    extract_sequential_frames,
    discover_initial_boxes_frame_0,
    process_video_with_predictor,
)


class TestVideoPipeline(unittest.TestCase):

    def test_01_frame_index_and_timestamp_math(self):
        fps = 30
        for idx in range(0, 90, 30):
            ts = round(idx / fps, 4)
            if idx == 0:
                self.assertEqual(ts, 0.0)
            elif idx == 30:
                self.assertEqual(ts, 1.0)
            elif idx == 60:
                self.assertEqual(ts, 2.0)

    def test_02_bbox_coordinate_conversion_and_scaling(self):
        processed_shape = (1000, 1000, 3)
        original_shape = (2160, 3840, 3)  # h, w
        bbox_processed = [100, 100, 200, 200]  # x, y, w, h

        orig_bbox = convert_and_clamp_bbox(bbox_processed, processed_shape, original_shape)
        self.assertEqual(orig_bbox, [384, 216, 768, 432])

    def test_03_bbox_clamping_out_of_bounds(self):
        processed_shape = (1000, 1000, 3)
        original_shape = (1000, 1000, 3)

        out_bbox = [900, 900, 300, 300]
        clamped = convert_and_clamp_bbox(out_bbox, processed_shape, original_shape)
        self.assertEqual(clamped, [900, 900, 100, 100])

    def test_04_none_and_empty_bbox_handling(self):
        processed_shape = (1000, 1000, 3)
        original_shape = (1000, 1000, 3)

        res = convert_and_clamp_bbox(None, processed_shape, original_shape)
        self.assertEqual(res, [0, 0, 0, 0])

    def test_05_frame_0_auto_discovery(self):
        class MockMaskGenerator:
            def generate(self, img):
                # Return 2 raw masks
                seg1 = np.zeros(img.shape[:2], dtype=bool)
                seg1[50:250, 50:250] = True

                seg2 = np.zeros(img.shape[:2], dtype=bool)
                seg2[300:500, 300:500] = True

                return [
                    {
                        "segmentation": seg1,
                        "area": 40000,
                        "predicted_iou": 0.95,
                        "stability_score": 0.98,
                        "bbox": [50, 50, 200, 200],
                    },
                    {
                        "segmentation": seg2,
                        "area": 40000,
                        "predicted_iou": 0.92,
                        "stability_score": 0.96,
                        "bbox": [300, 300, 200, 200],
                    }
                ]

        dummy_dir = "outputs/test_frame_0"
        os.makedirs(dummy_dir, exist_ok=True)
        frame_0_path = os.path.join(dummy_dir, "000000.jpg")

        import cv2
        dummy_img = np.ones((600, 600, 3), dtype=np.uint8) * 128
        cv2.imwrite(frame_0_path, dummy_img)

        mock_gen = MockMaskGenerator()
        initial_objects = discover_initial_boxes_frame_0(mock_gen, frame_0_path)

        self.assertEqual(len(initial_objects), 2)
        self.assertEqual(initial_objects[0]["track_id"], 1)
        self.assertEqual(initial_objects[0]["box"], [50, 50, 250, 250])  # x1, y1, x2, y2

        self.assertEqual(initial_objects[1]["track_id"], 2)
        self.assertEqual(initial_objects[1]["box"], [300, 300, 500, 500])

        if os.path.exists(frame_0_path):
            os.remove(frame_0_path)

    def test_06_frame_0_auto_discovery_raises_error_on_zero_objects(self):
        class EmptyMaskGenerator:
            def generate(self, img):
                return []

        dummy_dir = "outputs/test_frame_0_empty"
        os.makedirs(dummy_dir, exist_ok=True)
        frame_0_path = os.path.join(dummy_dir, "000000.jpg")

        import cv2
        dummy_img = np.ones((600, 600, 3), dtype=np.uint8) * 128
        cv2.imwrite(frame_0_path, dummy_img)

        empty_gen = EmptyMaskGenerator()

        with self.assertRaises(ValueError):
            discover_initial_boxes_frame_0(empty_gen, frame_0_path)

        if os.path.exists(frame_0_path):
            os.remove(frame_0_path)

    def test_07_process_video_raises_error_if_no_boxes_nor_generator(self):
        dummy_dir = "outputs/test_dummy_frames"
        os.makedirs(dummy_dir, exist_ok=True)

        class MockPredictor:
            def init_state(self, video_path, **kwargs):
                return {}

        predictor = MockPredictor()

        # Dummy video frame
        import cv2
        dummy_path = os.path.join(dummy_dir, "000000.jpg")
        cv2.imwrite(dummy_path, np.zeros((100, 100, 3), dtype=np.uint8))

        with self.assertRaises(ValueError):
            process_video_with_predictor(
                predictor=predictor,
                video_path=dummy_path,
                frames_dir=dummy_dir,
                initial_boxes=None,
                mask_generator=None
            )

        if os.path.exists(dummy_path):
            os.remove(dummy_path)


if __name__ == "__main__":
    unittest.main()
