"""
YOLOv8 目标检测模型训练脚本。
"""
import shutil
from pathlib import Path

import matplotlib
import matplotlib.ft2font as ft2font
from ultralytics import YOLO
from ultralytics.utils import USER_CONFIG_DIR

# ─── 训练参数（根据需要修改）────────────────────────────────────────────────
MODEL_WEIGHTS = str(Path(__file__).parent / "pre_model/yolov8n.pt")  # 本地权重
# 模型选择建议：样本量 ~2000 张，yolov8s 是精度与速度的较好平衡点
# 若 mAP 不理想可升级为 yolov8m.pt；若追求推理速度可降为 yolov8n.pt

DATA_YAML     = str(Path(__file__).parent / "dataset.yaml")
EPOCHS        = 150              # 训练轮数

IMGSZ         = 1280              # 输入图像尺寸（与采集分辨率 640×480 匹配，无需修改）
BATCH         = 4                # 批大小

WORKERS       = 4                 # 数据加载线程数（与 CPU 核心数匹配即可）
PROJECT       = str((Path(__file__).resolve().parent / "runs" / "detect").resolve())  # 输出目录（绝对路径，避免重复拼接）
NAME          = "fire_02"        # 实验名称（每次重训改名，保留历史结果）
DEVICE        = 0                # 0 = 第一块 GPU；"cpu" 使用 CPU 训练
# ─────────────────────────────────────────────────────────────────────────────


def ensure_ultralytics_font() -> None:
    """修复损坏或不兼容的 Ultralytics 字体，避免验证阶段绘图崩溃。"""
    target = Path(USER_CONFIG_DIR) / "Arial.ttf"

    def _can_load_font(font_path: Path) -> bool:
        try:
            ft2font.FT2Font(str(font_path))
            return True
        except Exception:
            return False

    if target.exists() and _can_load_font(target):
        return

    fallback = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    if not fallback.exists():
        raise FileNotFoundError(f"未找到可用后备字体: {fallback}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fallback, target)

    if not _can_load_font(target):
        raise RuntimeError(f"字体修复失败，仍无法加载: {target}")

    print(f"已修复 Ultralytics 字体: {target}")

def main():
    ensure_ultralytics_font()

    # 加载预训练模型（首次运行时会自动下载权重）
    model = YOLO(MODEL_WEIGHTS)

    # 开始训练
    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        workers   = WORKERS,
        project   = PROJECT,
        name      = NAME,
        device    = DEVICE,
        patience  = 30,          # 早停：30 轮无提升则停止（样本少时适当放宽）
        save      = True,
        cache     = "disk",      # 8GB 显存建议用 disk 缓存而非 ram，加速数据读取且不占显存
        rect      = True,           # 使用矩形训练（更快收敛，适合小数据集）
    )

    print("\n训练完成！")
    print(f"最佳模型权重：{results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
