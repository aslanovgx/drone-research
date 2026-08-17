from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from src.classification.inference import (
    classify_crop,
    parse_segment_id,
)
from src.classification.model import build_model
from src.utils.schemas import ClassificationPrediction


class FixedClassifier(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            (images.shape[0], 4),
            dtype=torch.float32,
            device=images.device,
        )
        logits[:, 0] = 5.0
        return logits


def test_model_produces_expected_logits_shape() -> None:
    model = build_model(
        num_classes=4,
        backbone="mobilenet_v3_small",
        pretrained=False,
    )

    images = torch.randn(2, 3, 64, 64)
    logits = model(images)

    assert logits.shape == (2, 4)


def test_classify_crop_returns_shared_prediction_schema(
    tmp_path: Path,
) -> None:
    crop_path = tmp_path / "segment_17.png"

    Image.new(
        mode="RGB",
        size=(80, 40),
        color=(120, 180, 90),
    ).save(crop_path)

    prediction = classify_crop(
        crop_path=crop_path,
        model=FixedClassifier(),
        classes=["building", "car", "other", "tree"],
        image_size=64,
        device=torch.device("cpu"),
    )

    assert isinstance(prediction, ClassificationPrediction)
    assert prediction.segment_id == 17
    assert prediction.class_name == "building"
    assert 0.0 <= prediction.confidence <= 1.0


def test_parse_segment_id_rejects_unknown_filename() -> None:
    with pytest.raises(ValueError, match="segment_id"):
        parse_segment_id("crop_without_identifier.png")