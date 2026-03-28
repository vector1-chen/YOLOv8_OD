```bash
mkdir -p model
cd model
# Download YOLOv8n (nano) model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O model/yolov8n.pt
# Download other YOLOv8 models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

运行：
  1. pip install ultralytics
  2. python tool/split_dataset.py   （生成 train.txt / val.txt）
  3. python train.py