"""
将数据集划分为 train / val 两部分，并在项目根目录生成对应的 txt 文件列表。
默认比例：train 80%，val 20%（随机打乱后分配）。
"""
import random
from pathlib import Path

# ─── 配置 ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent  # 项目根目录
IMAGES_DIR = ROOT_DIR / "images"
LABELS_DIR = ROOT_DIR / "labels"
TRAIN_RATIO = 0.8                           # 训练集比例
SEED        = 42                            # 随机种子，保证可复现
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset():
    # 取所有在 labels/ 中存在对应 txt 的图片
    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    valid_pairs = [
        p for p in image_paths
        if (LABELS_DIR / (p.stem + ".txt")).exists()
    ]

    print(f"[INFO] 共找到有效样本：{len(valid_pairs)} 张")

    random.seed(SEED)
    random.shuffle(valid_pairs)

    split_idx  = int(len(valid_pairs) * TRAIN_RATIO)
    train_imgs = valid_pairs[:split_idx]
    val_imgs   = valid_pairs[split_idx:]

    print(f"[INFO] train: {len(train_imgs)}，val: {len(val_imgs)}")

    # 写入 train.txt / val.txt（相对路径，便于跨平台）
    for name, subset in [("train.txt", train_imgs), ("val.txt", val_imgs)]:
        out_path = ROOT_DIR / name
        with open(out_path, "w") as f:
            for p in subset:
                f.write(f"./images/{p.name}\n")
        print(f"[INFO] 已生成：{out_path}")

if __name__ == "__main__":
    split_dataset()
