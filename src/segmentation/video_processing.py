import cv2
import os
import json


def extract_sampled_frames(video_path, output_dir, frame_interval=30):
    """
    Extracts every Nth frame from a video and saves as images.
    frame_interval=30 means: at 30fps video, roughly 1 frame per second.
    Returns list of dicts: {"frame_index": int, "path": str, "timestamp_sec": float}
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_idx = 0
    saved = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved.append({
                "frame_index": frame_idx,
                "path": out_path,
                "timestamp_sec": round(frame_idx / fps, 2),
            })

        frame_idx += 1

    cap.release()
    print(f"Extracted {len(saved)} frames out of {frame_idx} total (interval={frame_interval})")
    return saved


def process_video_frames(mask_generator, video_path, frames_dir, config, frame_interval=30):
    """
    Full pipeline for a video: sample frames -> run SAM2 (resize strategy) on each.
    Returns list of {"frame_index", "timestamp_sec", "image", "masks"}
    """
    from segmentation.sam_model import generate_masks_from_image  # resize-based function

    frames = extract_sampled_frames(video_path, frames_dir, frame_interval=frame_interval)

    results = []
    for f in frames:
        image = cv2.imread(f["path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # force resize strategy for video, regardless of global config
        video_config = dict(config)
        video_config["preprocessing"] = {
            **config.get("preprocessing", {}),
            "strategy": "resize",
        }

        image_out, masks = generate_masks_from_image(mask_generator, image, video_config)

        results.append({
            "frame_index": f["frame_index"],
            "timestamp_sec": f["timestamp_sec"],
            "masks": masks,
        })
        print(f"Frame {f['frame_index']} ({f['timestamp_sec']}s): {len(masks)} masks")

    return results