import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
import cv2

if __name__ == '__main__':
    # 定义要使用的 exp 目录编号列表
    exp_numbers = [180]  # 可根据需要添加更多编号

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
                    img = cv2.imread(result.path)
                    # 为每个图像创建一个裁剪图片保存目录
                    img_name = os.path.basename(result.path).split('.')[0]
                    crop_dir = os.path.join(result_dir, f'exp{exp_num}_{img_name}_crops')
                    if not os.path.exists(crop_dir):
                        os.makedirs(crop_dir)

                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls)
                        if class_id not in class_counts:
                            class_counts[class_id] = 1
                        else:
                            class_counts[class_id] += 1

                        # 获取检测框的坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        # 裁剪检测框对应的图像区域
                        cropped_image = img[y1:y2, x1:x2]
                        # 保存裁剪后的图片
                        crop_img_name = os.path.join(crop_dir, f'class_{class_id}_box_{i}.jpg')
                        cv2.imwrite(crop_img_name, cropped_image)

                    for class_id, count in class_counts.items():
                        result_line = f"使用模型 {model_path} 检测时，类别 {class_id} 的检测框数量: {count}\n"
                        result_file.write(result_line)
                        print(result_line.strip())

        except Exception as e:
            print(f"加载模型 {model_path} 时出错: {e}")
