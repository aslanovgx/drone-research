"""End-to-end SAM segmentation and classification pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.classification.config import (
    load_config as load_classifier_config,
)
from src.classification.inference import predict_many
from src.pipeline import run_pipeline
from src.segmentation.bbox_extractor import export_segments
from src.segmentation.sam_model import (
    generate_masks,
    load_config as load_sam_config,
    load_sam_model,
)
from src.utils.schemas import SegmentPrediction


def run_end_to_end_pipeline(
    image_path: str,
    sam_config_path: str = "configs/sam.yaml",
    classifier_config_path: str = "configs/classifier.yaml",
    pipeline_config_path: str = "configs/pipeline.yaml",
    checkpoint_path: str | None = None,
):
    """Run SAM, classifier, JSON export and visualization."""

    image = Path(image_path)

    if not image.is_file():
        raise FileNotFoundError(
            f"Input image could not be found: {image}"
        )

    print(f"Input image: {image}")

    # 1. Load SAM configuration.
    sam_config = load_sam_config(sam_config_path)

    strategy = (
        sam_config
        .get("preprocessing", {})
        .get("strategy", "resize")
    )

    if strategy != "tiling":
        raise ValueError(
            "The end-to-end pipeline currently requires the SAM "
            "'tiling' strategy so bounding boxes remain in the "
            "original image coordinate system."
        )

    # 2. Load SAM 2.1 and generate masks.
    print("Loading SAM 2.1 model...")
    mask_generator = load_sam_model(sam_config)

    print("Generating SAM masks...")
    processed_image, masks = generate_masks(
        mask_generator=mask_generator,
        image_path=str(image),
        config=sam_config,
    )

    print(f"Generated masks: {len(masks)}")

    # 3. Export bounding boxes, crops and segmentation JSON.
    output_config = sam_config.get("output", {})

    crops_dir = output_config.get(
        "crops_dir",
        "outputs/crops",
    )
    segmentation_json_path = output_config.get(
        "json_path",
        "outputs/segmentation_results.json",
    )

    exported_segments = export_segments(
        image=processed_image,
        masks=masks,
        crops_dir=crops_dir,
        json_path=segmentation_json_path,
    )

    segments = [
        SegmentPrediction.model_validate(segment)
        for segment in exported_segments
    ]

    print(f"Exported valid segments: {len(segments)}")

    # 4. Classify every SAM crop.
    classifier_config = load_classifier_config(
        classifier_config_path
    )

    crop_paths = [
        segment.crop_path
        for segment in segments
        if segment.crop_path is not None
    ]
    segment_ids = [
        segment.segment_id
        for segment in segments
        if segment.crop_path is not None
    ]

    print(f"Classifying crops: {len(crop_paths)}")

    classifications = predict_many(
        crop_paths=crop_paths,
        config=classifier_config,
        checkpoint_path=checkpoint_path,
        segment_ids=segment_ids,
    )

    print(f"Classifier predictions: {len(classifications)}")

    # 5. Merge SAM and classifier predictions.
    result = run_pipeline(
        image_path=str(image),
        segments=segments,
        classifications=classifications,
        config_path=pipeline_config_path,
    )

    print(f"Final detections: {len(result.detections)}")
    print("End-to-end pipeline: PASSED")

    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM segmentation, crop classification and "
            "final prediction visualization."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the input drone image.",
    )
    parser.add_argument(
        "--sam-config",
        default="configs/sam.yaml",
        help="Path to the SAM configuration.",
    )
    parser.add_argument(
        "--classifier-config",
        default="configs/classifier.yaml",
        help="Path to the classifier configuration.",
    )
    parser.add_argument(
        "--pipeline-config",
        default="configs/pipeline.yaml",
        help="Path to the integration configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional classifier checkpoint override.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    run_end_to_end_pipeline(
        image_path=arguments.image,
        sam_config_path=arguments.sam_config,
        classifier_config_path=arguments.classifier_config,
        pipeline_config_path=arguments.pipeline_config,
        checkpoint_path=arguments.checkpoint,
    )


if __name__ == "__main__":
    main()