from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SegmentPrediction(BaseModel):
    segment_id: int = Field(ge=0)
    bbox: BoundingBox
    area: int = Field(gt=0)
    sam_score: float = Field(ge=0.0, le=1.0)
    crop_path: str | None = None


class ClassificationPrediction(BaseModel):
    segment_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class Detection(BaseModel):
    segment_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    area: int = Field(gt=0)
    sam_score: float = Field(ge=0.0, le=1.0)


class PipelineResult(BaseModel):
    image: str
    detections: list[Detection]