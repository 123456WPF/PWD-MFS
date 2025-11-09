import os
import numpy as np
from PIL import Image


def convert_16bit_to_8bit(input_path, output_path):
    """
    将16位图像转换为8位图像

    参数:
        input_path: 输入16位图像路径（支持.tif, .png等格式）
        output_path: 输出8位图像路径
    """
    try:
        # 打开16位图像
        with Image.open(input_path) as img:
            # 转换为numpy数组（获取原始16位数据）
            img_array = np.array(img, dtype=np.uint16)  # 16位数据类型

            # 检查是否为16位图像
            if img_array.dtype != np.uint16:
                print(f"警告：{input_path} 不是16位图像，可能无需转换")
                return

            # 处理多通道图像（如RGB）：对每个通道单独归一化
            if len(img_array.shape) == 3:  # 多通道（高度, 宽度, 通道数）
                channels = img_array.shape[2]
                normalized = np.zeros_like(img_array, dtype=np.uint8)  # 存储8位结果

                for c in range(channels):
                    channel = img_array[:, :, c]
                    min_val = channel.min()
                    max_val = channel.max()

                    # 避免除以0（如果图像所有像素值相同）
                    if max_val == min_val:
                        normalized[:, :, c] = 0
                    else:
                        # 线性归一化到0-255
                        normalized[:, :, c] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            else:  # 单通道（灰度图）
                min_val = img_array.min()
                max_val = img_array.max()

                if max_val == min_val:
                    normalized = np.zeros_like(img_array, dtype=np.uint8)
                else:
                    normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            # 转换为PIL图像并保存
            output_img = Image.fromarray(normalized)
            output_img.save(output_path)
            print(f"转换成功：{input_path} -> {output_path}")

    except Exception as e:
        print(f"转换失败：{input_path}，错误：{str(e)}")


# 批量转换文件夹中的16位图像
def batch_convert_16to8(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg','.tif', '.tiff', '.png', '.bmp')):  # 支持的16位格式
            input_path = os.path.join(input_folder, filename)
            # 生成输出文件名（保留原名称，替换后缀为.png或.jpg）
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base_name}_8bit.png")  # 8位推荐用png保存
            convert_16bit_to_8bit(input_path, output_path)


# 使用示例
if __name__ == "__main__":
    input_folder = r"E:\data\peizhunRGB\huiduhua"  # 替换为你的16位图像文件夹路径
    output_folder = r"E:\data\peizhunRGB\16-8"  # 替换为输出文件夹路径
    batch_convert_16to8(input_folder, output_folder)
