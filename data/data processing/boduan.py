from PIL import Image

# 读取图像
image = Image.open("DJI_20241206125017_0001_D_bottom_left_8bit.png")

# 获取模式
print(f"📊 图像模式: {image.mode}")

# 判断通道数
mode_to_channels = {
    "1": 1,    # 二值图像
    "L": 1,    # 灰度图
    "RGB": 3,  # 3 通道彩色图
    "RGBA": 4, # 4 通道（带透明度）
    "CMYK": 4, # 4 通道（印刷）
    "I;16": 1  # 16 位灰度图
}
channels = mode_to_channels.get(image.mode, "未知")
print(f"🟢 该图像的通道数: {channels}")
