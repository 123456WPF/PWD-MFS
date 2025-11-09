import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 定义要使用的 exp 目录编号列表
    # exp_numbers =[101,102,103,104,105,1
    # ] #据需要添加更多编
    exp_numbers =[180] #据需要添加更多编

    # 预测参数设置
    predict_params = {
        'source': 'img/',
        'imgsz': 640,
        'project': 'runs/detect',
        'name': 'exp',
        'save': True,
        'augment': True,  # 开启数据增强
        'conf': 0.2,  # 置信度阈值，可根据实际情况调整
        'iou': 0.45  # 交并比阈值，可根据实际情况调整
    }

    # 定义保存结果的目录
    result_dir = 'loop_results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 遍历每个 exp 目录
    for exp_num in exp_numbers:
        model_path = f'runs/train/exp{exp_num}/weights/best.pt'
        try:
            # 加载模型
            model = YOLO(model_path)

            print(f"正在使用模型 {model_path} 进行检测...")

            # 进行预测
            results = model.predict(**predict_params)

            # 为每个模型创建一个结果文件
            result_file_name = os.path.join(result_dir, f'exp{exp_num}_results.txt')
            with open(result_file_name, 'w') as result_file:
                for result in results:
                    class_counts = {}
                    for cls in result.boxes.cls:
                        class_id = int(cls)
                        if class_id not in class_counts:
                            class_counts[class_id] = 1
                        else:
                            class_counts[class_id] += 1

                    for class_id, count in class_counts.items():
                        result_line = f"使用模型 {model_path} 检测时，类别 {class_id} 的检测框数量: {count}\n"
                        result_file.write(result_line)
                        print(result_line.strip())

        except Exception as e:
            print(f"加载模型 {model_path} 时出错: {e}")
