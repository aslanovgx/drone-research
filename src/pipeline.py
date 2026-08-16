from pathlib import Path

from src.utils.schemas import (
    ClassificationPrediction,
    Detection,
    PipelineResult,
    SegmentPrediction,
)


def merge_predictions(
    image: str,
    segments: list[SegmentPrediction],
    classifications: list[ClassificationPrediction],
) -> PipelineResult:
    classification_by_id = {
        classification.segment_id: classification
        for classification in classifications
    }

    detections: list[Detection] = []

    for segment in segments:
        classification = classification_by_id.get(segment.segment_id)

        if classification is None:
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