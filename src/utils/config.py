from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SAMConfig(BaseModel):
    min_score: float = Field(ge=0.0, le=1.0)
    min_mask_area: int = Field(gt=0)


class ClassifierConfig(BaseModel):
    min_confidence: float = Field(ge=0.0, le=1.0)


class OutputConfig(BaseModel):
    json_dir: str
    annotated_dir: str


class VisualizationConfig(BaseModel):
    box_thickness: int = Field(gt=0)
    show_confidence: bool


class PipelineConfig(BaseModel):
    sam: SAMConfig
    classifier: ClassifierConfig
    outputs: OutputConfig
    visualization: VisualizationConfig


def load_config(
    config_path: str = "configs/pipeline.yaml",
) -> PipelineConfig:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    return PipelineConfig.model_validate(config_data)