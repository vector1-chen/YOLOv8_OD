### download pre-trained model
```bash
mkdir -p model
cd model
# Download YOLOv8n (nano) model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O model/yolov8n.pt
# Download other YOLOv8 models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

### example of running pipeline:
1. 视频转图片
```bash
conda activate yolov8
python3 tool/video_to_image.py -v ./data/video.mp4 -o ./images --frame_step 30
```

2. 数据标注（label.txt, dataset.yaml）
- 修改labels.txt，添加类别。
- 修改dataset.yaml，添加类别和数据集路径。
- 标注数据集，生成labels文件夹。
```bash
conda activate xanylabeling
xanylabeling
```
open images folder
upload labels.txt
open ai function on the left dock
>>> 有用快捷键
>>> |---|---|
>>> |Ctrl + S|保存|
>>> |Ctrl + J|编辑模式（拖拉缩放矩形，双击修改标签）|
>>> |R|绘制矩形|
>>> |A|上一张|
>>> |D|下一张|
>>> ai模式下（记得调整输出类型为矩形）
>>> |Q|在想要包围的区域打点|
>>> |E|在想要去除的区域打点|
>>> |F|完成标注，选择类别|
>>> |B|取消|
- 导出yolo_hbb格式的标签文件，选择dataset.yaml，放在labels文件夹下。

3. 打乱数据
```bash
python3 tool/image_disorder/shuffle_rename_images.py -s ./images --seed 1 --remove-unlabeled
```
4. 划分训练集和验证集
```bash
python3 tool/split_dataset.py
```
5. 训练模型
修改参数，启动训练
```bash
python3 train.py
```

6. 导出 NCNN 模型（.param + .bin，含 FP16 量化）

> **前提**：确保 `pnnx` 已安装到 conda 环境。首次使用需执行一次：
> ```bash
> # 下载 pnnx（Linux x86_64）
> wget https://github.com/pnnx/pnnx/releases/download/20260409/pnnx-20260409-linux.zip -O /tmp/pnnx-linux.zip
> unzip /tmp/pnnx-linux.zip -d /tmp/pnnx-linux
> cp /tmp/pnnx-linux/pnnx-20260409-linux/pnnx $CONDA_PREFIX/bin/pnnx
> chmod +x $CONDA_PREFIX/bin/pnnx
> ```
> 其中 `$CONDA_PREFIX` 为当前激活环境路径（如 `/home/bug/miniconda3/envs/yolov8`）。

```bash
conda activate yolov8
python3 -c "
from ultralytics import YOLO
model = YOLO('runs/detect/fire_02/weights/best.pt')  # 替换为实际路径
model.export(
    format='ncnn',   # 导出为 NCNN 格式
    imgsz=1280,       # 与训练时保持一致
    half=True,       # 开启 FP16 量化，模型体积减半、推理更快
    simplify=True,   # 简化 ONNX 中间图（提升兼容性）
)
"
yolo export model=runs/detect/fire_02/weights/best.pt format=ncnn imgsz=1280 half=True simplify=True
```

导出成功后，输出目录为 `runs/detect/<实验名>/weights/best_ncnn_model/`，包含：

| 文件 | 说明 |
|------|------|
| `model.ncnn.param` | 网络结构描述文件 |
| `model.ncnn.bin`   | FP16 量化权重文件 |
| `model_ncnn.py`    | Python 推理示例  |

**NCNN 推理验证**（用 ultralytics）：
```bash
yolo predict task=detect \
  model=runs/detect/fire_02/weights/best_ncnn_model \
  imgsz=1280 half \
  source=images/img_005.jpg
```

> **注意事项**
> - `imgsz` 必须与训练时一致，否则推理结果异常。
> - `half=True`（FP16）在纯 CPU 上精度损失极小，推荐开启。
> - 若需部署到 ARM 设备（如 RK3588、RaspberryPi），请下载对应架构的 pnnx：
>   `pnnx-20260409-linux-aarch64.zip`，其余步骤相同。
> - `.param` 和 `.bin` 必须同名且放在同一目录，部署时一并拷贝。
