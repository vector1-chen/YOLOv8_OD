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