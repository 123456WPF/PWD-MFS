import os
import cv2
import numpy as np
from osgeo import gdal

# 显式启用GDAL异常处理
gdal.UseExceptions()


def process_bands_in_folder(input_dir, output_dir, target_width=5280, target_height=3956):
    """
    处理输入目录中的单波段图像，上采样至目标尺寸并组合为3波段图像，保存到输出目录

    参数:
        input_dir: 单波段图像所在目录（E:\lsy\datasets\path-8\images\train）
        output_dir: 输出3波段图像的目录（E:\lsy\datasets\path-8\images\train\sahngcaiyang）
        target_width: 目标宽度（5280）
        target_height: 目标高度（3956）
    """
    # 创建输出目录（若不存在）
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录已准备: {output_dir}")

    # 收集输入目录中的所有单波段图像（筛选常见图像格式）
    band_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')  # 支持的图像格式
    band_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(band_extensions)
    ]

    # 检查是否有足够的波段
    if len(band_paths) < 3:
        raise ValueError(f"输入目录至少需要3个波段图像，当前仅找到{len(band_paths)}个")
    print(f"找到{len(band_paths)}个波段图像，将使用前3个组合为3波段图像")

    # 读取并上采样前3个波段
    upsampled_bands = []
    for i in range(3):
        band_path = band_paths[i]
        band_name = os.path.basename(band_path)

        # 读取单波段图像（保留原始数据类型）
        band = cv2.imread(band_path, -1)
        if band is None:
            raise IOError(f"无法读取波段图像: {band_path}")

        # 上采样至目标尺寸（双三次插值，适合放大并保留细节）
        upsampled = cv2.resize(
            band,
            (target_width, target_height),  # cv2.resize参数：(width, height)
            interpolation=cv2.INTER_CUBIC
        )

        # 验证上采样尺寸
        if upsampled.shape != (target_height, target_width):
            raise RuntimeError(
                f"波段{i + 1}上采样失败！实际尺寸: {upsampled.shape}，目标尺寸: ({target_height}, {target_width})"
            )

        upsampled_bands.append(upsampled)
        print(f"波段{i + 1}（{band_name}）上采样完成，尺寸: {upsampled.shape}")

    # 组合为3波段图像并保存
    output_filename = "3_band_upsampled.tif"  # 输出文件名
    output_path = os.path.join(output_dir, output_filename)

    # 确定数据类型（根据实际波段类型自动适配）
    dtype = upsampled_bands[0].dtype
    gdal_dtype = gdal.GDT_UInt16 if dtype == np.uint16 else gdal.GDT_Byte

    # 创建GDAL数据集保存3波段图像
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, target_width, target_height, 3, gdal_dtype)

    # 写入每个波段（GDAL波段索引从1开始）
    for i in range(3):
        out_band = out_ds.GetRasterBand(i + 1)
        out_band.WriteArray(upsampled_bands[i])

    # 释放资源
    out_ds.FlushCache()
    del out_ds

    print(f"\n处理完成！3波段图像已保存至: {output_path}")
    print(f"图像尺寸: {target_width}x{target_height}，波段数: 3")


# 主程序
if __name__ == "__main__":
    # 输入目录（你的单波段图像所在路径）
    input_directory = 'A001.jpg'
    # 输出目录（上采样后的3波段图像保存路径）
    output_directory = r"E:\lsy\datasets\path-8\images\train\shangcaiyang"

    try:
        process_bands_in_folder(
            input_dir=input_directory,
            output_dir=output_directory,
            target_width=5280,
            target_height=3956
        )
    except Exception as e:
        print(f"处理失败: {str(e)}")