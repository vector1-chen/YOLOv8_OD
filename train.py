"""
YOLOv8 目标检测模型训练脚本。

运行前准备：
  1. pip install ultralytics
  2. python split_dataset.py   （生成 train.txt / val.txt）
  3. python train.py
"""
from pathlib import Path
from ultralytics import YOLO

# ─── 训练参数（根据需要修改）────────────────────────────────────────────────
MODEL_WEIGHTS = str(Path(__file__).parent / "pre_model/yolov8s.pt")  # 本地权重
# 模型选择建议：样本量 ~2000 张，yolov8s 是精度与速度的较好平衡点
# 若 mAP 不理想可升级为 yolov8m.pt；若追求推理速度可降为 yolov8n.pt

DATA_YAML     = str(Path(__file__).parent / "dataset.yaml")
EPOCHS        = 150              # 训练轮数

IMGSZ         = 640              # 输入图像尺寸（与采集分辨率 640×480 匹配，无需修改）
BATCH         = 16                # 批大小

WORKERS       = 4                # 数据加载线程数（与 CPU 核心数匹配即可）
PROJECT       = "runs/detect"    # 输出目录
NAME          = "fire_01"        # 实验名称（每次重训改名，保留历史结果）
DEVICE        = 0                # 0 = 第一块 GPU；"cpu" 使用 CPU 训练
# ─────────────────────────────────────────────────────────────────────────────

def main():
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
    )

    print("\n训练完成！")
    print(f"最佳模型权重：{results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
