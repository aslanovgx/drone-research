from pathlib import Path

import cv2
import numpy as np

from src.utils.schemas import Detection, PipelineResult


CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "building": (255, 0, 0),
    "tree": (0, 180, 0),
    "car": (0, 0, 255),
    "road": (0, 165, 255),
    "other": (128, 128, 128),
}

DEFAULT_COLOR = (255, 255, 255)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    annotated_image = image.copy()
    image_height, image_width = annotated_image.shape[:2]

    for detection in detections:
        bbox = detection.bbox

        x1 = max(bbox.x, 0)
        y1 = max(bbox.y, 0)
        x2 = min(bbox.x + bbox.width, image_width - 1)
        y2 = min(bbox.y + bbox.height, image_height - 1)

        if x1 >= x2 or y1 >= y2:
            continue

        color = CLASS_COLORS.get(
            detection.class_name.lower(),
            DEFAULT_COLOR,
        )

        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            font_thickness,
        )

        label_top = max(
            y1 - text_height - baseline - 8,
            0,
        )
        label_bottom = label_top + text_height + baseline + 8
        label_right = min(
            x1 + text_width + 8,
            image_width - 1,
        )

        cv2.rectangle(
            annotated_image,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            thickness=-1,
        )

        cv2.putText(
            annotated_image,
            label,
            (x1 + 4, label_bottom - baseline - 4),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated_image


def save_annotated_image(
    result: PipelineResult,
    output_path: str,
) -> Path:
    image = cv2.imread(result.image)

    if image is None:
        raise FileNotFoundError(
            f"Image could not be loaded: {result.image}"
        )

    annotated_image = draw_detections(
        image=image,
        detections=result.detections,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    saved = cv2.imwrite(
        str(path),
        annotated_image,
    )

    if not saved:
        raise OSError(
            f"Annotated image could not be saved: {path}"
        )

    return path