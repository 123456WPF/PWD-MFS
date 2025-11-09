
import warnings


warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 定义要使用的配置文件列表
    config_files = [

        # 'ultralytics/cfg/models/11/yolo11-vanillanet.yaml',
        # 'ultralytics/cfg/models/11/yolo11-LSKNet.yaml',
        # 'ultralytics/cfg/models/11/yolo11-swintransformer.yaml',
        # 'ultralytics/cfg/models/11/yolo11-repvit.yaml'
        'ultralytics\cfg\models\cfg2024\YOLOv8-Attention\yolov8.yaml',
    ]

    for config_file in config_files:
        # 提取配置文件的名称，用于命名训练实验
        config_name = config_file.split('/')[-1].replace('.yaml', '')

        model = YOLO(config_file)
        # model.load('yolo11n.pt') # loading pretrain weights
        model.train(data='data_congcai.yaml',
                    # resume=True,
                    cache=False,
                    imgsz=640,
                    epochs=300,
                    batch=32,
                    close_mosaic=0,
                    workers=4,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                    # device='0',
                    optimizer='SGD',  # using SGD
                    patience=0, # set 0 to close earlystop.
                    #resume=True, # 断点续训,YOLO初始化时选择last.pt
                    # amp=False, # close amp
                    # fraction=0.2,
                    project='runs/train',
                    name='exp'  # 使用配置文件名称命名实验
                    )
#