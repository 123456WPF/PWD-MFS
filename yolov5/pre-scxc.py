from ultralytics import YOLO
import os

# 加载YOLOv5模型
model = YOLO(r'E:\lsy\yolov5-master\runs\train\exp8\weights\best.pt')
# 手动融合（无参数，解决内部调用冲突）
model.fuse()

# 配置参数
conf_threshold = 0.3
source = r'E:\lsy\yolov5-master\img'
save_dir = r'E:\lsy\yolov5-master\imgout'

# 检查源路径
if not os.path.exists(source):
    print(f"无效的源路径: {source}")
    exit(1)

# 获取图片列表
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.tif')
image_paths = []
if os.path.isfile(source) and source.lower().endswith(image_extensions):
    image_paths = [source]
elif os.path.isdir(source):
    image_paths = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(image_extensions)]
if not image_paths:
    print(f"未找到图片文件: {source}")
    exit(1)

# 推理并保存结果（移除fuse参数）
for img_path in image_paths:
    model.predict(
        img_path,
        conf=conf_threshold,
        save=True,
        project=os.path.dirname(save_dir),
        name=os.path.basename(save_dir),
        exist_ok=True,
        boxes=True
    )

print(f"结果已保存至: {save_dir}")