from pathlib import Path

import cv2
import numpy as np

import src.run_pipeline as runner
from src.utils.schemas import ClassificationPrediction


def test_end_to_end_runner_connects_all_pipeline_stages(
    tmp_path: Path,
    monkeypatch,
):
    image_path = tmp_path / "demo.jpg"

    image = np.full(
        (100, 120, 3),
        180,
        dtype=np.uint8,
    )

    assert cv2.imwrite(
        str(image_path),
        image,
    )

    crops_dir = tmp_path / "crops"
    segmentation_json = tmp_path / "segmentation.json"
    final_json_dir = tmp_path / "json"
    annotated_dir = tmp_path / "predictions"
    pipeline_config_path = tmp_path / "pipeline.yaml"

    pipeline_config_path.write_text(
        f"""
sam:
  min_score: 0.0
  min_mask_area: 1

classifier:
  min_confidence: 0.0

outputs:
  json_dir: "{final_json_dir}"
  annotated_dir: "{annotated_dir}"

visualization:
  box_thickness: 2
  show_confidence: true
""".strip(),
        encoding="utf-8",
    )

    sam_config = {
        "preprocessing": {
            "strategy": "tiling",
        },
        "output": {
            "crops_dir": str(crops_dir),
            "json_path": str(segmentation_json),
        },
    }

    fake_masks = [
        {
            "bbox": [10, 15, 30, 25],
            "area": 750,
            "predicted_iou": 0.95,
        }
    ]

    monkeypatch.setattr(
        runner,
        "load_sam_config",
        lambda path: sam_config,
    )
    monkeypatch.setattr(
        runner,
        "load_sam_model",
        lambda config: object(),
    )
    monkeypatch.setattr(
        runner,
        "generate_masks",
        lambda mask_generator, image_path, config: (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            fake_masks,
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_classifier_config",
        lambda path: {},
    )

    def fake_predict_many(
        crop_paths,
        config,
        checkpoint_path,
        segment_ids,
    ):
        assert len(crop_paths) == 1
        assert Path(crop_paths[0]).is_file()
        assert segment_ids == [0]

        return [
            ClassificationPrediction(
                segment_id=0,
                class_name="car",
                confidence=0.91,
            )
        ]

    monkeypatch.setattr(
        runner,
        "predict_many",
        fake_predict_many,
    )

    result = runner.run_end_to_end_pipeline(
        image_path=str(image_path),
        pipeline_config_path=str(pipeline_config_path),
    )

    assert len(result.detections) == 1

    detection = result.detections[0]

    assert detection.segment_id == 0
    assert detection.class_name == "car"
    assert detection.confidence == 0.91
    assert detection.sam_score == 0.95

    assert segmentation_json.is_file()
    assert (crops_dir / "segment_0.png").is_file()
    assert (final_json_dir / "demo.json").is_file()
    assert (
        annotated_dir / "demo_annotated.jpg"
    ).is_file()