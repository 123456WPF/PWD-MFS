import warnings
import sys
import os
warnings.filterwarnings('ignore')

# 添加YOLOv5根目录到环境变量
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从官方train.py中导入run函数
from train import run

if __name__ == '__main__':
    run(
        weights='yolov5s.pt',  # YOLOv5s预训练权重
        cfg='models/yolov5s.yaml',  # 模型配置文件
        data='data2.yaml',  # 数据集配置文件
        epochs=300,
        batch_size=16,
        imgsz=640,
        workers=8,
        device=0,  # 0为GPU，'cpu'为CPU
        patience=0,
        optimizer='SGD',
        cache=False
    )