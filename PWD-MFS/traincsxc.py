import warnings


warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 指定显卡和多卡训练问题 统一都在<YOLOV11配置文件.md>下方常见错误和解注释去掉。
# 训练时候输出的AMP Check使用的YOLO11n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11-GhostHGNetV2_MPCA_LSCD.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='data_congcaiThree.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0',

                optimizer='SGD', # using SGD
                patience=0, # set 0 to close earlystop.1
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp'
                )
