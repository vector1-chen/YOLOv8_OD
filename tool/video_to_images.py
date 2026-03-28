#!/usr/bin/env python3
"""
video_to_images.py
将视频按固定帧间隔导出为图片。

用法示例:
  # 每 5 帧导出 1 张图片，输出到指定目录
  python3 tool/video_to_images.py --video-path ./input.mp4 --output-dir ./images --frame-step 5

  # 按时间间隔导出（每 0.5 秒一张），最多导出 500 张
  python3 tool/video_to_images.py --video-path ./input.mp4 --output-dir ./images --time-step 0.5 --max-images 500
"""

import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="将视频抽帧保存为图片。")
    parser.add_argument(
        "--video-path",
        "-v",
        required=True,
        help="输入视频路径",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="图片输出目录",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="按帧抽取间隔（每 N 帧保存 1 张，默认: 1）",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=None,
        help="按时间抽取间隔（单位秒）；指定后优先于 --frame-step",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="输出文件起始编号（默认: 1）",
    )
    parser.add_argument(
        "--prefix",
        default="image_",
        help="输出文件名前缀（默认: image_）",
    )
    parser.add_argument(
        "--ext",
        default="png",
        choices=["jpg", "jpeg", "png", "bmp", "webp"],
        help="输出图片格式（默认: png）",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG 质量（1-100，默认: 95）",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="最多导出图片数量（默认不限制）",
    )
    return parser.parse_args()


def validate_args(args):
    if args.frame_step < 1:
        raise ValueError("--frame-step 必须 >= 1")
    if args.time_step is not None and args.time_step <= 0:
        raise ValueError("--time-step 必须 > 0")
    if args.start_index < 0:
        raise ValueError("--start-index 必须 >= 0")
    if not (1 <= args.jpg_quality <= 100):
        raise ValueError("--jpg-quality 必须在 1 到 100 之间")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images 必须 > 0")


def build_output_path(output_dir: Path, prefix: str, index: int, ext: str, width: int) -> Path:
    file_name = f"{prefix}{str(index).zfill(width)}.{ext}"
    return output_dir / file_name


def main():
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"参数错误: {exc}")
        raise SystemExit(2)

    video_path = Path(args.video_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists() or not video_path.is_file():
        print(f"输入视频不存在: {video_path}")
        raise SystemExit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        raise SystemExit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    if args.time_step is not None:
        frame_step = max(1, int(round(args.time_step * fps)))
    else:
        frame_step = args.frame_step

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        estimated = (total_frames + frame_step - 1) // frame_step
        if args.max_images is not None:
            estimated = min(estimated, args.max_images)
    else:
        estimated = args.max_images if args.max_images is not None else 10000

    width = max(6, len(str(args.start_index + max(estimated - 1, 0))))

    print(f"[INFO] 输入视频: {video_path}")
    print(f"[INFO] 输出目录: {output_dir}")
    print(f"[INFO] 采样间隔: 每 {frame_step} 帧导出 1 张")

    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        should_save = (frame_idx % frame_step == 0)
        if should_save:
            out_idx = args.start_index + saved
            out_path = build_output_path(output_dir, args.prefix, out_idx, args.ext, width)

            if args.ext in {"jpg", "jpeg"}:
                write_ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
            else:
                write_ok = cv2.imwrite(str(out_path), frame)

            if not write_ok:
                print(f"[WARNING] 保存失败: {out_path}")
            else:
                saved += 1

            if args.max_images is not None and saved >= args.max_images:
                break

        frame_idx += 1

    cap.release()
    print(f"[INFO] 导出完成，共保存 {saved} 张图片")


if __name__ == "__main__":
    main()
