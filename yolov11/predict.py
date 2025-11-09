from ultralytics import YOLO
# 加载训练好的模型，改为自己的路径
model = YOLO('runs/train/exp17/weights/best.pt')
# 修改为自己的图像或者文件夹的路径
source = r'E:\lsy\ultralytics-main-11\img'
# 运行推理，并附加参数
model.predict(source, save=True)