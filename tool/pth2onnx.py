from ultralytics import YOLO
model = YOLO('/home/bug/learn_ws/lift_ws/YOLOv8_OD/runs/detect/fire_02/weights/best.pt')
success = model.export(format='onnx', imgsz=1280, simplify=True, opset=11)
print('ONNX export success:', success)