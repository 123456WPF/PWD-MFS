from ultralytics import YOLO

# 加载模型（移除不支持的fuse参数）
model = YOLO('runs/train/exp217/weights/best.pt')

# 检测参数（保持不变）
source = r'E:\lsy\ultralytics-yolo11-main\img'
model.predict(source, save=True)