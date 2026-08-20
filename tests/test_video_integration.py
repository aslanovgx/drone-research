import os
import sys
import unittest
import json
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.video_processing import (
    build_sam2_video_predictor_from_config,
    process_video_with_predictor,
    discover_initial_boxes_frame_0,
)
from segmentation.sam_model import load_config, load_sam_model


class TestVideoIntegration(unittest.TestCase):

    def setUp(self):
        self.config_path = "configs/sam.yaml"
        self.checkpoint_path = "checkpoints/sam2.1_hiera_small.pt"
        self.test_video_path = "outputs/test_fixtures/sample_drone_clip.mp4"
        self.frames_dir = "outputs/test_fixtures/frames"
        self.json_out_path = "outputs/test_fixtures/video_results.json"

        # Create a minimal 10-frame OpenCV MP4 video with a moving white rectangle
        os.makedirs("outputs/test_fixtures", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.test_video_path, fourcc, 30.0, (640, 480))

        for f_idx in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Moving rectangle across frames
            x_pos = 100 + (f_idx * 10)
            y_pos = 100 + (f_idx * 5)
            cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 120, y_pos + 120), (255, 255, 255), -1)
            out.write(frame)

        out.release()

    def tearDown(self):
        import shutil
        # Cleanup generated video fixture files
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
        if os.path.exists(self.json_out_path):
            os.remove(self.json_out_path)
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir, ignore_errors=True)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.path.exists("checkpoints/sam2.1_hiera_small.pt"),
        "Requires CUDA GPU and checkpoints/sam2.1_hiera_small.pt checkpoint file"
    )
    def test_real_sam2_video_predictor_integration(self):
        config = load_config(self.config_path)
        mask_generator = load_sam_model(config)
        predictor = build_sam2_video_predictor_from_config(config)

        result = process_video_with_predictor(
            predictor=predictor,
            video_path=self.test_video_path,
            frames_dir=self.frames_dir,
            config=config,
            mask_generator=mask_generator,
            initial_boxes=None,  # Triggers Frame 0 auto-discovery!
            output_interval=5,
            max_frames=10,
            json_path=self.json_out_path
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["video_path"], self.test_video_path)
        self.assertEqual(result["original_width"], 640)
        self.assertEqual(result["original_height"], 480)
        self.assertTrue(os.path.exists(self.json_out_path))

        with open(self.json_out_path) as fp:
            data = json.load(fp)

        self.assertIn("frames", data)
        self.assertGreater(len(data["frames"]), 0)
        # Verify track_id exists
        first_frame_segments = data["frames"][0]["segments"]
        self.assertGreater(len(first_frame_segments), 0)
        self.assertIn("track_id", first_frame_segments[0])
        self.assertIn("bbox", first_frame_segments[0])


if __name__ == "__main__":
    unittest.main()
