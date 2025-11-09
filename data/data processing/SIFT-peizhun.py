import cv2
import os
import numpy as np
#SIFT配准

def align_nir_to_rgb(rgb_path, nir_path, output_nir_path, visualize=False):
    """
    将 NIR 图像配准到 RGB 图像上。

    :param rgb_path: RGB 图像的路径 (作为基准图像)
    :param nir_path: NIR 图像的路径 (需要被配准的图像)
    :param output_nir_path: 配准后的 NIR 图像输出路径
    :param visualize: 是否可视化配准过程和结果 (True/False)
    """
    print(f"\n--- 正在处理: {os.path.basename(rgb_path)} 和 {os.path.basename(nir_path)} ---")

    # 1. 读取图像
    rgb_image = cv2.imread(rgb_path)
    if rgb_image is None:
        print(f"❌ 无法加载 RGB 图像: {rgb_path}")
        return

    nir_image = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)
    if nir_image is None:
        print(f"❌ 无法加载 NIR 图像: {nir_path}")
        return

    # 2. 预处理用于特征匹配 (将 NIR 归一化到 8-bit)
    # 注意：这只是为了特征提取，最终配准将使用原始16位NIR图像以保留精度
    nir_image_8bit = cv2.normalize(nir_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # 3. 使用 SIFT 提取特征点和描述符
    sift = cv2.SIFT_create()
    keypoints_rgb, descriptors_rgb = sift.detectAndCompute(gray_rgb, None)
    keypoints_nir, descriptors_nir = sift.detectAndCompute(nir_image_8bit, None)

    print(f"📊 提取到的特征点: RGB={len(keypoints_rgb)}, NIR={len(keypoints_nir)}")

    # 4. 使用 FLANN 匹配器进行快速匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_rgb, descriptors_nir, k=2)

    # 5. 应用比率测试筛选好的匹配点 (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"📊 筛选后的好匹配点数量: {len(good_matches)}")

    if len(good_matches) < 10:  # 如果好匹配点太少，无法进行可靠的配准
        print(f"⚠️  好匹配点数量不足 (仅{len(good_matches)}个)，无法完成配准。")
        return

    # 6. 准备用于计算单应性矩阵的点
    src_pts = np.float32([keypoints_rgb[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_nir[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 7. 使用 RANSAC 鲁棒估计单应性矩阵 H
    # H 描述了如何将 NIR 图像上的点 (dst_pts) 变换到 RGB 图像上的点 (src_pts)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        print(f"⚠️  无法计算单应性矩阵，配准失败。")
        return

    print("✅ 成功估计单应性矩阵 H。")

    # 8. 对原始 NIR 图像执行透视变换，使其与 RGB 图像对齐
    # 使用原始的16位NIR图像进行变换，以保留最大的动态范围和精度
    height, width = rgb_image.shape[:2]
    registered_nir = cv2.warpPerspective(nir_image, H, (width, height))

    # 9. 保存配准后的 NIR 图像
    cv2.imwrite(output_nir_path, registered_nir)
    print(f"✅ 配准完成！配准后的 NIR 图像已保存至: {output_nir_path}")

    # 10. (可选) 可视化结果
    if visualize:
        # 绘制匹配点
        matches_mask = mask.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
        img_matches = cv2.drawMatches(rgb_image, keypoints_rgb, nir_image_8bit, keypoints_nir, good_matches, None,
                                      **draw_params)

        # 创建配准结果的对比图
        # 将配准后的NIR转为伪彩色，以便与RGB对比
        registered_nir_8bit = cv2.normalize(registered_nir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        nir_pseudo_color = cv2.applyColorMap(registered_nir_8bit, cv2.COLORMAP_JET)

        # 将RGB和伪彩色NIR并排显示
        comparison_img = np.hstack((rgb_image, nir_pseudo_color))

        # 显示
        cv2.imshow('Feature Matches (Inliers marked in Green)', img_matches)
        cv2.imshow('Registration Result (Left: RGB, Right: Registered NIR)', comparison_img)

        print("\n请按任意键关闭可视化窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def batch_process(rgb_folder, nir_folder, output_folder_nir, visualize=False):
    """
    批量处理文件夹中的所有 RGB 和 NIR 图像对。
    """
    if not os.path.exists(output_folder_nir):
        os.makedirs(output_folder_nir)
        print(f"📁 创建输出文件夹: {output_folder_nir}")

    # 获取并排序文件列表，确保一一对应
    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    nir_files = sorted([f for f in os.listdir(nir_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])

    if len(rgb_files) != len(nir_files):
        print(f"❌ 错误：RGB 图像数量 ({len(rgb_files)}) 和 NIR 图像数量 ({len(nir_files)}) 不匹配！")
        return

    print(f"\n📋 找到 {len(rgb_files)} 对图像，开始批量配准...")

    for rgb_file, nir_file in zip(rgb_files, nir_files):
        rgb_path = os.path.join(rgb_folder, rgb_file)
        nir_path = os.path.join(nir_folder, nir_file)
        output_nir_path = os.path.join(output_folder_nir, f"aligned_{nir_file}")

        align_nir_to_rgb(rgb_path, nir_path, output_nir_path, visualize=visualize)

    print(f"\n🎉 所有图像对处理完毕！配准后的 NIR 图像保存在: {output_folder_nir}")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 设置输入输出文件夹路径
    # !!! 请确保这两个文件夹中的图像文件名是一一对应的 !!!
    # 例如: JPG/1.jpg 对应 NIR/1.tif, JPG/2.jpg 对应 NIR/2.tif
    rgb_folder = r"E:\lsy\datasets\path-8\images\RGB"  # RGB 图像文件夹路径
    nir_folder = r"E:\lsy\datasets\path-8\images\NIR"  # NIR 图像文件夹路径 (请根据你的实际情况修改)
    output_folder_nir = r"E:\lsy\datasets\path-8\images\peizhun"   # 配准后 NIR 图像的输出文件夹路径

    # 2. 执行批量处理
    # 设置 visualize=True 可以在处理每一对图像时看到匹配点和配准结果
    batch_process(rgb_folder, nir_folder, output_folder_nir, visualize=False)

