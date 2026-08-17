import cv2
import os
import json
import numpy as np
import torch

from .mask_filter import filter_masks_quality
from .tiling import deduplicate_masks, mask_to_bbox


def extract_sequential_frames(video_path, output_dir, max_frames=None):
    """
    Extracts ALL consecutive frames from a video into output_dir.
    Naming: frame_000000.jpg, frame_000001.jpg, frame_000002.jpg, ...
    This preserves unbroken temporal continuity for SAM2VideoPredictor.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Purge any stale JPG files from a previous run so SAM2's frame sorter
    # never encounters files with an incompatible naming format.
    for _f in os.listdir(output_dir):
        if _f.lower().endswith(".jpg") or _f.lower().endswith(".jpeg"):
            try:
                os.remove(os.path.join(output_dir, _f))
            except OSError:
                pass

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_idx = 0
    saved = []

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved.append({
            "frame_index": frame_idx,
            "path": out_path,
            "timestamp_sec": round(frame_idx / fps, 4),
        })

        frame_idx += 1

    cap.release()
    print(f"Extracted {len(saved)} sequential frames for video tracking.")
    return saved


def extract_sampled_frames(video_path, output_dir, frame_interval=30):
    """
    Backward-compatibility helper: extracts sampled frames every Nth frame.
    """
    return extract_sequential_frames(video_path, output_dir, max_frames=None)[::frame_interval]


def convert_and_clamp_bbox(bbox, processed_shape, original_shape):
    """
    Converts bounding box from processed frame resolution (h_p, w_p)
    back to original video frame resolution (h_orig, w_orig),
    and clamps to original video boundaries.
    """
    if bbox is None:
        return [0, 0, 0, 0]

    x, y, w, h = bbox
    h_p, w_p = processed_shape[:2]
    h_orig, w_orig = original_shape[:2]

    scale_x = w_orig / float(w_p) if w_p > 0 else 1.0
    scale_y = h_orig / float(h_p) if h_p > 0 else 1.0

    x_orig = int(round(x * scale_x))
    y_orig = int(round(y * scale_y))
    w_orig_box = int(round(w * scale_x))
    h_orig_box = int(round(h * scale_y))

    x1 = max(0, min(x_orig, w_orig))
    y1 = max(0, min(y_orig, h_orig))
    x2 = max(0, min(x_orig + w_orig_box, w_orig))
    y2 = max(0, min(y_orig + h_orig_box, h_orig))

    final_w = max(0, x2 - x1)
    final_h = max(0, y2 - y1)

    return [x1, y1, final_w, final_h]


def build_sam2_video_predictor_from_config(config):
    """
    Instantiates SAM2VideoPredictor model using paths from sam.yaml config.
    """
    from sam2.build_sam import build_sam2_video_predictor

    model_cfg = config["model"]["config_path"]
    checkpoint = config["model"]["checkpoint_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    return predictor


def discover_initial_boxes_frame_0(mask_generator, frame_0_path, config=None):
    """
    Performs automatic object discovery on Frame 0 using SAM 2 automatic mask generator.
    Filters raw masks using quality criteria and returns valid initial boxes with track IDs.
    Raises ValueError if no valid objects are discovered on Frame 0.
    """
    from segmentation.sam_model import generate_masks_from_image

    img0 = cv2.imread(frame_0_path)
    if img0 is None:
        raise ValueError(f"Could not read Frame 0 image from {frame_0_path}")

    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    # Force resize strategy for video frame 0 to match processed frame resolution
    video_config = dict(config or {})
    video_config["preprocessing"] = {
        **video_config.get("preprocessing", {}),
        "strategy": "resize",
    }

    _, raw_masks = generate_masks_from_image(mask_generator, img0_rgb, video_config)

    filter_cfg = (config or {}).get("mask_filter", {})
    min_area = filter_cfg.get("min_area", 1500)
    min_stability = filter_cfg.get("min_stability_score", 0.9)
    min_iou = filter_cfg.get("min_predicted_iou", 0.85)

    quality_masks = filter_masks_quality(
        raw_masks,
        min_area=min_area,
        min_stability_score=min_stability,
        min_predicted_iou=min_iou,
    )
    dedup_masks = deduplicate_masks(quality_masks, iou_threshold=0.7)

    initial_objects = []
    track_id = 1

    for m in dedup_masks:
        bbox = m.get("bbox") or mask_to_bbox(m["segmentation"])
        if bbox is None:
            continue
        x, y, w, h = [int(v) for v in bbox]
        if w <= 5 or h <= 5:  # Minimum valid size check
            continue

        # SAM2 add_new_points_or_box expects box prompt as [x1, y1, x2, y2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = x1 + w, y1 + h

        initial_objects.append({
            "track_id": track_id,
            "box": [x1, y1, x2, y2],
            "bbox_xywh": [x1, y1, w, h],
            "sam_score": float(m.get("predicted_iou", 0.90)),
        })
        track_id += 1

    if len(initial_objects) == 0:
        raise ValueError("No valid initial objects discovered on Frame 0 for video tracking.")

    print(f"Frame 0 Auto-Discovery: {len(initial_objects)} valid initial objects registered for tracking.")
    return initial_objects


def process_video_with_predictor(
    predictor,
    video_path,
    frames_dir,
    config=None,
    mask_generator=None,
    initial_boxes=None,
    output_interval=30,
    max_frames=None,
    json_path="outputs/video_results.json"
):
    """
    Temporal video processing pipeline using native SAM2VideoPredictor.
    1. Extracts ALL consecutive frames into frames_dir for unbroken temporal sequence.
    2. Performs Frame 0 automatic mask discovery if initial_boxes is None.
    3. Initializes SAM2 video state with CPU offloading (offload_video_to_cpu, offload_state_to_cpu).
    4. Registers initial object boxes on Frame 0 with track IDs.
    5. Propagates masks across ALL sequential frames via propagate_in_video.
    6. Filters output records according to output_interval (e.g. every 30 frames).
    7. Rescales predicted bboxes/masks back to original video resolution.
    8. Exports structured video metadata to JSON.
    """
    cap = cv2.VideoCapture(video_path)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Read output interval from config if provided
    if config:
        output_interval = (config.get("video", {}).get("output_interval") or
                           config.get("video", {}).get("frame_interval") or
                           output_interval)

    # Fail fast: if auto-discovery will be needed but no generator is provided,
    # raise immediately before doing any I/O.
    if initial_boxes is None and mask_generator is None:
        raise ValueError(
            "Neither initial_boxes nor mask_generator was provided. "
            "Supply explicit initial_boxes or a mask_generator for Frame 0 auto-discovery."
        )

    # Step 1: Extract ALL consecutive frames (unbroken sequence)
    frames = extract_sequential_frames(video_path, frames_dir, max_frames=max_frames)
    if not frames:
        return {"video_path": video_path, "frames": []}

    # Step 2: Discover Frame 0 initial boxes if not explicitly provided
    if initial_boxes is None:
        frame_0_path = frames[0]["path"]
        initial_boxes = discover_initial_boxes_frame_0(mask_generator, frame_0_path, config=config)

    offload_video = (config or {}).get("video", {}).get("offload_video_to_cpu", True)
    offload_state = (config or {}).get("video", {}).get("offload_state_to_cpu", True)

    # Initialize video state from extracted frames directory
    inference_state = predictor.init_state(
        video_path=frames_dir,
        offload_video_to_cpu=offload_video,
        offload_state_to_cpu=offload_state,
    )

    try:
        # Step 3: Register initial boxes on Frame 0
        with torch.inference_mode():
            for obj in initial_boxes:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj["track_id"],
                    box=np.array(obj["box"], dtype=np.float32),
                )

        # Step 4: Propagate masks across ALL sequential frames
        video_segments = {}
        with torch.inference_mode():
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    obj_id: out_mask_logits[i]
                    for i, obj_id in enumerate(out_obj_ids)
                }

        # Step 5: Format structured results for output_interval frames
        frame_results = []
        for idx, f in enumerate(frames):
            frame_idx = f["frame_index"]

            # Only export frames matching output_interval or the last frame
            if frame_idx % output_interval != 0 and idx != len(frames) - 1:
                continue

            timestamp = f["timestamp_sec"]
            seg_items = []

            if idx in video_segments:
                for obj_id, mask_logit in video_segments[idx].items():
                    mask_np = (mask_logit[0] > 0.0).cpu().numpy()
                    ys, xs = np.where(mask_np)

                    if len(xs) > 0:
                        lx, ly = int(xs.min()), int(ys.min())
                        lw, lh = int(xs.max() - lx + 1), int(ys.max() - ly + 1)
                        processed_shape = mask_np.shape

                        orig_bbox = convert_and_clamp_bbox(
                            [lx, ly, lw, lh],
                            processed_shape=processed_shape,
                            original_shape=(h_orig, w_orig)
                        )
                        area_px = int(mask_np.sum())

                        seg_items.append({
                            "track_id": int(obj_id),
                            "bbox": orig_bbox,
                            "area": area_px,
                            "sam_score": 0.90,  # Temporal propagation score
                        })

            frame_results.append({
                "frame_index": frame_idx,
                "timestamp_sec": timestamp,
                "segments": seg_items,
            })

        output_data = {
            "video_path": video_path,
            "original_width": w_orig,
            "original_height": h_orig,
            "total_processed_frames": len(frames),
            "exported_frames": len(frame_results),
            "output_interval": output_interval,
            "frames": frame_results,
        }

        # Export JSON
        if json_path:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as fp:
                json.dump(output_data, fp, indent=2)

        return output_data

    finally:
        # Clean up predictor state
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(inference_state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def process_video_frames(mask_generator, video_path, frames_dir, config, frame_interval=30):
    """
    Fallback frame-by-frame processing for non-temporal execution.
    """
    from segmentation.sam_model import generate_masks_from_image

    frames = extract_sequential_frames(video_path, frames_dir)[::frame_interval]

    results = []
    for f in frames:
        image = cv2.imread(f["path"])
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        video_config = dict(config or {})
        video_config["preprocessing"] = {
            **video_config.get("preprocessing", {}),
            "strategy": "resize",
        }

        image_out, masks = generate_masks_from_image(mask_generator, image, video_config)

        results.append({
            "frame_index": f["frame_index"],
            "timestamp_sec": f["timestamp_sec"],
            "masks": masks,
        })

    return results