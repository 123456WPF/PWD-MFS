import os
from PIL import Image


def convert_to_grayscale(input_folder, output_folder):
    """
    将输入文件夹中的图像批量转换为灰度图并保存到输出文件夹

    参数:
        input_folder: 输入图像所在文件夹（如之前转换的JPG图片文件夹）
        output_folder: 灰度图输出文件夹
    """
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 过滤支持的图像格式（可根据需要补充）
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            # 生成输出文件名（在原文件名后加"_gray"标识）
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_gray.jpg"  # 灰度图保存为JPG
            output_path = os.path.join(output_folder, output_filename)

            try:
                # 打开图像并转换为灰度图
                with Image.open(input_path) as img:
                    # 转换为灰度模式（'L'表示8位灰度）
                    gray_img = img.convert('L')
                    # 保存灰度图
                    gray_img.save(output_path)
                    print(f"已生成灰度图：{output_filename}")
            except Exception as e:
                print(f"处理失败：{filename}，错误信息：{e}")


# 使用示例
if __name__ == "__main__":
    # 输入文件夹：之前转换后的图片所在路径（如你之前的"chongjian1"）
    input_folder = r"E:\data\peizhunRGB\huiduhua"
    # 输出文件夹：灰度图保存路径（可自定义）
    output_folder = r"E:\data\peizhunRGB\16-8"
    convert_to_grayscale(input_folder, output_folder)