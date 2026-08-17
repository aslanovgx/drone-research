from pathlib import Path

from src.utils.config import load_config
from src.utils.schemas import (
    ClassificationPrediction,
    Detection,
    PipelineResult,
    SegmentPrediction,
)
from src.visualization.draw_predictions import save_annotated_image

def merge_predictions(
    image: str,
    segments: list[SegmentPrediction],
    classifications: list[ClassificationPrediction],
    min_sam_score: float = 0.0,
    min_mask_area: int = 1,
    min_classifier_confidence: float = 0.0,
) -> PipelineResult:
    classification_by_id = {
        classification.segment_id: classification
        for classification in classifications
    }

    detections: list[Detection] = []

    for segment in segments:
        if segment.sam_score < min_sam_score:
            continue

        if segment.area < min_mask_area:
            continue

        classification = classification_by_id.get(segment.segment_id)

        if classification is None:
            continue

        if classification.confidence < min_classifier_confidence:
            continue

        detection = Detection(
            segment_id=segment.segment_id,
            class_name=classification.class_name,
            confidence=classification.confidence,
            bbox=segment.bbox,
            area=segment.area,
            sam_score=segment.sam_score,
        )

        detections.append(detection)

    return PipelineResult(
        image=image,
        detections=detections,
    )

def save_pipeline_result(
    result: PipelineResult,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

def run_pipeline(
    image_path: str,
    segments: list[SegmentPrediction],
    classifications: list[ClassificationPrediction],
    config_path: str = "configs/pipeline.yaml",
) -> PipelineResult:
    config = load_config(config_path)

    result = merge_predictions(
        image=image_path,
        segments=segments,
        classifications=classifications,
        min_sam_score=config.sam.min_score,
        min_mask_area=config.sam.min_mask_area,
        min_classifier_confidence=config.classifier.min_confidence,
    )

    image_stem = Path(image_path).stem

    json_output_path = (
        Path(config.outputs.json_dir) /
        f"{image_stem}.json"
    )

    annotated_output_path = (
        Path(config.outputs.annotated_dir) /
        f"{image_stem}_annotated.jpg"
    )

    save_pipeline_result(
        result=result,
        output_path=str(json_output_path),
    )

    save_annotated_image(
        result=result,
        output_path=str(annotated_output_path),
        box_thickness=config.visualization.box_thickness,
        show_confidence=config.visualization.show_confidence,
    )

    return result