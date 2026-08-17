from src.pipeline import merge_predictions
from src.utils.schemas import (
    BoundingBox,
    ClassificationPrediction,
    SegmentPrediction,
)


def create_segment(
    segment_id: int,
    sam_score: float = 0.90,
    area: int = 1000,
) -> SegmentPrediction:
    return SegmentPrediction(
        segment_id=segment_id,
        bbox=BoundingBox(
            x=10,
            y=20,
            width=100,
            height=80,
        ),
        area=area,
        sam_score=sam_score,
    )


def create_classification(
    segment_id: int,
    class_name: str = "building",
    confidence: float = 0.90,
) -> ClassificationPrediction:
    return ClassificationPrediction(
        segment_id=segment_id,
        class_name=class_name,
        confidence=confidence,
    )


def test_merge_predictions_matches_using_segment_id() -> None:
    segments = [
        create_segment(segment_id=1),
        create_segment(segment_id=2),
    ]

    classifications = [
        create_classification(
            segment_id=2,
            class_name="tree",
        ),
        create_classification(
            segment_id=1,
            class_name="building",
        ),
    ]

    result = merge_predictions(
        image="test_image.jpg",
        segments=segments,
        classifications=classifications,
    )

    assert len(result.detections) == 2
    assert result.detections[0].segment_id == 1
    assert result.detections[0].class_name == "building"
    assert result.detections[1].segment_id == 2
    assert result.detections[1].class_name == "tree"


def test_merge_predictions_applies_thresholds() -> None:
    segments = [
        create_segment(
            segment_id=1,
            sam_score=0.95,
            area=1000,
        ),
        create_segment(
            segment_id=2,
            sam_score=0.50,
            area=1000,
        ),
        create_segment(
            segment_id=3,
            sam_score=0.95,
            area=300,
        ),
        create_segment(
            segment_id=4,
            sam_score=0.95,
            area=1000,
        ),
    ]

    classifications = [
        create_classification(
            segment_id=1,
            confidence=0.90,
        ),
        create_classification(
            segment_id=2,
            confidence=0.90,
        ),
        create_classification(
            segment_id=3,
            confidence=0.90,
        ),
        create_classification(
            segment_id=4,
            confidence=0.40,
        ),
    ]

    result = merge_predictions(
        image="test_image.jpg",
        segments=segments,
        classifications=classifications,
        min_sam_score=0.80,
        min_mask_area=500,
        min_classifier_confidence=0.60,
    )

    assert len(result.detections) == 1
    assert result.detections[0].segment_id == 1


def test_merge_predictions_skips_missing_classification() -> None:
    segments = [
        create_segment(segment_id=1),
        create_segment(segment_id=2),
    ]

    classifications = [
        create_classification(
            segment_id=1,
            class_name="building",
        ),
    ]

    result = merge_predictions(
        image="test_image.jpg",
        segments=segments,
        classifications=classifications,
    )

    assert len(result.detections) == 1
    assert result.detections[0].segment_id == 1