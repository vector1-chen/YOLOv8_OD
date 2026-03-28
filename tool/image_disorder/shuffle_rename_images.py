#!/usr/bin/env python3
"""
shuffle_rename_images.py
将指定目录中的图片（及同名 .json 文件）打乱顺序并重新命名为连续编号文件名。
图片与同名 .json 文件作为一对同步处理：打乱顺序、统一重命名。

用法示例:
  # 预览（不实际操作）
  python3 tool/image_disorder/shuffle_rename_images.py --source-dir images_e --seed 42 --dry-run

  # 原地重命名（在源目录中操作）
  python3 tool/image_disorder/shuffle_rename_images.py --source-dir images_e --seed 42

  # 输出到新目录（保留原文件）
  python3 tool/image_disorder/shuffle_rename_images.py --source-dir images_e --output-dir images_shuffled --seed 42
"""

import os
import random
import shutil
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="打乱图片顺序并重新命名为连续编号。"
    )
    parser.add_argument(
        "--source-dir", "-s",
        required=True,
        help="包含原始图片的源目录"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="输出目录（若不指定则原地重命名；若指定则复制/移动到该目录）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，保证可复现（默认不固定）"
    )
    parser.add_argument(
        "--prefix",
        default="img_",
        help="新文件名前缀（默认: img_）"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="起始编号（默认: 1）"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="输出到 output-dir 时使用复制而非移动（默认: 移动）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式：只打印操作计划，不实际重命名/移动文件"
    )
    parser.add_argument(
        "--save-mapping",
        action="store_true",
        help="将原文件名与新文件名的映射保存为 JSON 文件"
    )
    parser.add_argument(
        "--fix-json-paths",
        action="store_true",
        help="扫描 source-dir 中所有 JSON 文件，将 imagePath 字段修正为与 JSON 同名的图片文件名（用于修复已重命名但 JSON 内容未更新的情况）"
    )
    parser.add_argument(
        "--remove-unlabeled",
        action="store_true",
        help="打乱时仅保留有同名 JSON 的图片；原地模式和 move 模式下会删除无标注图片，copy 模式下仅跳过无标注图片"
    )
    return parser.parse_args()


def collect_image_pairs(
    source_dir: Path,
) -> List[Tuple[Path, Optional[Path]]]:
    """
    收集目录中所有支持格式的图片文件（不递归），
    同时检测同名 .json 文件，返回 (图片路径, json路径或None) 列表。
    """
    images = [
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    images.sort()  # 先排序，保证不同系统下初始顺序一致

    pairs: List[Tuple[Path, Optional[Path]]] = []
    for img in images:
        json_path = img.with_suffix(".json")
        pairs.append((img, json_path if json_path.exists() else None))
    return pairs


def build_rename_plan(
    pairs: List[Tuple[Path, Optional[Path]]],
    output_dir: Path,
    prefix: str,
    start_index: int,
) -> List[Tuple[Path, Path, Optional[Path], Optional[Path]]]:
    """
    构造重命名计划：返回 (src_img, dst_img, src_json, dst_json) 列表。
    dst 的编号位数自动根据总数决定。若 src_json 为 None，则对应 dst_json 也为 None。
    """
    total = len(pairs)
    width = len(str(total + start_index - 1))  # 补零位数

    plan: List[Tuple[Path, Path, Optional[Path], Optional[Path]]] = []
    for i, (src_img, src_json) in enumerate(pairs):
        new_stem = f"{prefix}{str(start_index + i).zfill(width)}"
        dst_img = output_dir / f"{new_stem}{src_img.suffix.lower()}"
        dst_json = (output_dir / f"{new_stem}.json") if src_json is not None else None
        plan.append((src_img, dst_img, src_json, dst_json))
    return plan


def update_json_image_path(json_path: Path, new_image_name: str, dry_run: bool) -> None:
    """
    将 xanylabeling/labelme 格式 JSON 文件中的 imagePath 字段更新为新的图片文件名。
    只修改文件名部分，保留原来的扩展名（以应对图片格式不变的情况）。
    """
    if dry_run:
        return
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "imagePath" in data:
            data["imagePath"] = new_image_name
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [WARNING] 更新 imagePath 失败: {json_path}  原因: {e}")


def _do_file_op(
    src: Path, dst: Path, inplace: bool, use_copy: bool, dry_run: bool
) -> None:
    """对单个文件执行实际的重命名/复制/移动操作（已处于临时名阶段时不调用此函数）。"""
    if not dry_run:
        if use_copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), dst)


def execute_plan(
    plan: List[Tuple[Path, Path, Optional[Path], Optional[Path]]],
    inplace: bool,
    use_copy: bool,
    dry_run: bool,
) -> Dict[str, str]:
    """
    执行重命名计划（图片 + 同名 JSON 同步处理）。
    - inplace=True: 在源目录中重命名
    - inplace=False + use_copy=True: 复制到输出目录
    - inplace=False + use_copy=False: 移动到输出目录
    返回 {原文件名: 新文件名} 映射字典（图片和 JSON 均记录）。
    """
    mapping: Dict[str, str] = {}

    if inplace:
        # 原地重命名：先全部改为临时名，再改为目标名，避免命名冲突
        # 构建所有需要操作的文件的临时名计划（图片 + JSON）
        tmp_plan: List[Tuple[Path, Path, Path]] = []
        for src_img, dst_img, src_json, dst_json in plan:
            tmp_img = src_img.parent / (src_img.name + ".shuffle_tmp")
            tmp_plan.append((src_img, tmp_img, dst_img))
            if src_json is not None and dst_json is not None:
                tmp_json = src_json.parent / (src_json.name + ".shuffle_tmp")
                tmp_plan.append((src_json, tmp_json, dst_json))

        if not dry_run:
            for src, tmp, _ in tmp_plan:
                src.rename(tmp)
            for _, tmp, dst in tmp_plan:
                tmp.rename(dst)

        # 更新 JSON 内的 imagePath（在最终目标名阶段更新）
        for src_img, dst_img, src_json, dst_json in plan:
            mapping[src_img.name] = dst_img.name
            print(f"  [RENAME(in-place)] {src_img.name}  →  {dst_img.name}")
            if src_json is not None and dst_json is not None:
                mapping[src_json.name] = dst_json.name
                print(f"  [RENAME(in-place)] {src_json.name}  →  {dst_json.name}")
                update_json_image_path(dst_json, dst_img.name, dry_run)
    else:
        action = "COPY" if use_copy else "MOVE"
        for src_img, dst_img, src_json, dst_json in plan:
            mapping[src_img.name] = dst_img.name
            if not dry_run:
                _do_file_op(src_img, dst_img, inplace, use_copy, dry_run)
            print(f"  [{action}] {src_img}  →  {dst_img}")

            if src_json is not None and dst_json is not None:
                mapping[src_json.name] = dst_json.name
                if not dry_run:
                    _do_file_op(src_json, dst_json, inplace, use_copy, dry_run)
                    update_json_image_path(dst_json, dst_img.name, dry_run)
                print(f"  [{action}] {src_json}  →  {dst_json}")

    return mapping


def fix_json_paths(source_dir: Path, dry_run: bool) -> int:
    """
    扫描 source_dir 中所有 JSON 文件，将 imagePath 字段修正为与 JSON 同名的图片文件名。
    用于修复"图片已重命名但 JSON 内容未更新"的情况。
    """
    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        print(f"[WARNING] 在 {source_dir} 中未找到任何 JSON 文件。")
        return 0

    if dry_run:
        print("[INFO] *** DRY-RUN 模式，不会修改任何文件 ***\n")

    fixed = 0
    skipped = 0
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARNING] 读取失败: {jf.name}  原因: {e}")
            continue

        if "imagePath" not in data:
            skipped += 1
            continue

        old_image_path = data["imagePath"]
        # 保留原来的扩展名，只替换文件名主干
        old_suffix = Path(old_image_path).suffix  # e.g. ".png"
        new_image_name = jf.stem + old_suffix      # e.g. "merge01_0005.png"

        if old_image_path == new_image_name:
            skipped += 1
            continue

        print(f"  [FIX] {jf.name}: imagePath  {old_image_path!r}  →  {new_image_name!r}")
        if not dry_run:
            data["imagePath"] = new_image_name
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        fixed += 1

    status = "（预览）" if dry_run else "（完成）"
    print(f"\n[INFO] 共修复 {fixed} 个 JSON，跳过 {skipped} 个（无需修改）{status}")
    return 0


def main():
    args = parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"[ERROR] 源目录不存在: {source_dir}")
        return 1

    # --fix-json-paths 模式：修复目录中已重命名但 JSON 内容未更新的文件
    if args.fix_json_paths:
        return fix_json_paths(source_dir, args.dry_run)

    # 收集图片及同名 JSON
    pairs = collect_image_pairs(source_dir)
    if not pairs:
        print(f"[WARNING] 在 {source_dir} 中未找到任何支持的图片文件。")
        return 0

    # 确定输出目录
    inplace = args.output_dir is None
    if inplace:
        output_dir = source_dir
        print(f"[INFO] 模式: 原地重命名")
    else:
        output_dir = Path(args.output_dir)
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        op = "复制" if args.copy else "移动"
        print(f"[INFO] 模式: {op} 到 {output_dir}")

    # 可选：去除无标注图片（无同名 JSON）
    if args.remove_unlabeled:
        unlabeled_images = [img for img, js in pairs if js is None]
        pairs = [(img, js) for img, js in pairs if js is not None]

        if unlabeled_images:
            if inplace or (not inplace and not args.copy):
                if args.dry_run:
                    for p in unlabeled_images:
                        print(f"  [REMOVE-UNLABELED(dry-run)] {p}")
                else:
                    for p in unlabeled_images:
                        try:
                            p.unlink()
                            print(f"  [REMOVE-UNLABELED] {p}")
                        except Exception as e:
                            print(f"  [WARNING] 删除失败: {p}  原因: {e}")
            else:
                print(
                    f"[INFO] copy 模式下将跳过 {len(unlabeled_images)} 张无标注图片（不会删除源文件）"
                )

        if not pairs:
            status = "（预览）" if args.dry_run else "（完成）"
            print(f"[INFO] 去除无标注图片后没有可处理的带标注图片 {status}")
            return 0

    json_count = sum(1 for _, j in pairs if j is not None)
    print(f"[INFO] 找到 {len(pairs)} 张图片（其中 {json_count} 张有同名 .json），源目录: {source_dir}")

    # 打乱
    if args.seed is not None:
        random.seed(args.seed)
        print(f"[INFO] 随机种子: {args.seed}")
    else:
        print("[INFO] 未指定随机种子，结果不可复现。")

    random.shuffle(pairs)

    if args.dry_run:
        print("[INFO] *** DRY-RUN 模式，不会修改任何文件 ***\n")

    # 构造重命名计划
    plan = build_rename_plan(pairs, output_dir, args.prefix, args.start_index)

    # 执行
    mapping = execute_plan(plan, inplace, args.copy, args.dry_run)

    # 保存映射
    if args.save_mapping and not args.dry_run:
        mapping_path = output_dir / "shuffle_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 映射文件已保存: {mapping_path}")

    img_count = len(plan)
    json_count = sum(1 for _, _, sj, _ in plan if sj is not None)
    status = "（预览）" if args.dry_run else "（完成）"
    print(f"\n[INFO] 共处理 {img_count} 张图片、{json_count} 个 JSON 文件 {status}")
    return 0


if __name__ == "__main__":
    exit(main())
