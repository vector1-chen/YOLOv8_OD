"""
Run YOLOv8 inference on an MP4 file and save an annotated MP4.

Features:
- Draws detection boxes + labels.
- Overlays semi-transparent colored masks.
  - If the model outputs segmentation masks, uses true instance masks.
  - If the model is detection-only, uses box-area translucent fills as mask-like overlays.

Example:
python infer_video.py \
  --model runs/detect/fire_01/weights/best.pt \
  --source data_video/input.mp4 \
  --output runs/predict/output.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8 video inference with translucent overlays")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--source", type=str, required=True, help="Input .mp4 file")
    parser.add_argument("--output", type=str, required=True, help="Output .mp4 file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help="Inference device, e.g. 0 or cpu")
    parser.add_argument("--alpha", type=float, default=0.35, help="Transparency for colored mask overlay")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per frame")
    return parser


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    # Deterministic vivid color mapping in BGR space.
    hue = (class_id * 37) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def blend_binary_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> None:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if not np.any(mask):
        return

    color_arr = np.array(color, dtype=np.float32)
    region = image[mask].astype(np.float32)
    image[mask] = np.clip(region * (1.0 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)


def blend_box_region(image: np.ndarray, box: np.ndarray, color: Tuple[int, int, int], alpha: float) -> None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    roi = image[y1:y2, x1:x2].astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    blended = np.clip(roi * (1.0 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)
    image[y1:y2, x1:x2] = blended


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    y = max(th + baseline + 2, y)
    x2 = x + tw + 6
    y2 = y + baseline + 4

    cv2.rectangle(image, (x, y - th - baseline - 4), (x2, y2), color, -1)
    cv2.putText(image, text, (x + 3, y - 3), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def process_video(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    source_path = Path(args.source)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    class_names: Dict[int, str] = model.names

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            retina_masks=True,
            verbose=False,
        )
        res = results[0]
        out = frame.copy()

        boxes = res.boxes
        masks = res.masks

        # Use real segmentation masks when available.
        if masks is not None and boxes is not None and masks.data is not None:
            mask_data = masks.data.cpu().numpy() > 0.5
            box_xyxy = boxes.xyxy.cpu().numpy()
            box_cls = boxes.cls.cpu().numpy().astype(int)

            n = min(len(mask_data), len(box_xyxy), len(box_cls))
            for i in range(n):
                cls_id = int(box_cls[i])
                color = color_for_class(cls_id)
                blend_binary_mask(out, mask_data[i], color, args.alpha)

        if boxes is not None and boxes.xyxy is not None:
            box_xyxy = boxes.xyxy.cpu().numpy()
            box_conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(box_xyxy),))
            box_cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((len(box_xyxy),), dtype=int)

            # For detection-only models, use box-area translucent fill as a mask-like overlay.
            if masks is None:
                for i, xyxy in enumerate(box_xyxy):
                    cls_id = int(box_cls[i])
                    color = color_for_class(cls_id)
                    blend_box_region(out, xyxy, color, args.alpha)

            for i, xyxy in enumerate(box_xyxy):
                cls_id = int(box_cls[i])
                conf = float(box_conf[i])
                color = color_for_class(cls_id)
                x1, y1, x2, y2 = xyxy.astype(int)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
                draw_label(out, label, x1, y1, color)

        writer.write(out)
        frame_idx += 1

        if frame_idx % 30 == 0:
            if total > 0:
                print(f"Processed {frame_idx}/{total} frames")
            else:
                print(f"Processed {frame_idx} frames")

    cap.release()
    writer.release()
    print(f"Done. Output saved to: {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
