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