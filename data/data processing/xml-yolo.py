import os
import xml.etree.ElementTree as ET
import glob


def xml_to_yolo(xml_dir, yolo_dir, class_mapping):
    """
    批量将XML标签文件转换为YOLO格式TXT文件
    :param xml_dir: 输入XML文件所在文件夹路径
    :param yolo_dir: 输出YOLO格式TXT文件的文件夹路径
    :param class_mapping: 类别名称→ID的映射字典（如{"cat":0, "dog":1}）
    """
    # 1. 创建输出目录（不存在则创建）
    os.makedirs(yolo_dir, exist_ok=True)
    print(f"✅ 输出目录已准备：{yolo_dir}")

    # 2. 获取文件夹内所有XML文件
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    if len(xml_files) == 0:
        raise FileNotFoundError(f"❌ 在{xml_dir}中未找到任何XML文件")
    print(f"📊 共找到{len(xml_files)}个XML文件，开始转换...")

    # 3. 逐个处理XML文件
    for xml_path in xml_files:
        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图片宽高（XML中<size>标签下的width和height）
        size_node = root.find("size")
        if not size_node:
            print(f"⚠️  {xml_path}缺少<size>标签，跳过该文件")
            continue
        img_width = int(size_node.find("width").text)
        img_height = int(size_node.find("height").text)
        if img_width == 0 or img_height == 0:
            print(f"⚠️  {xml_path}图片宽高为0，跳过该文件")
            continue

        # 提取目标标注信息（<object>标签）
        yolo_content = []
        for obj_node in root.findall("object"):
            # 获取类别名称并映射为ID
            class_name = obj_node.find("name").text.strip()
            if class_name not in class_mapping:
                print(f"⚠️  {xml_path}中存在未定义类别：{class_name}，跳过该目标")
                continue
            class_id = class_mapping[class_name]

            # 获取边界框坐标（XML中为左上角xmin,ymin + 右下角xmax,ymax）
            bndbox = obj_node.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # 转换为YOLO格式：归一化x_center, y_center, width, height
            # 公式：x_center = (xmin + xmax)/(2*img_width)
            #       y_center = (ymin + ymax)/(2*img_height)
            #       width = (xmax - xmin)/img_width
            #       height = (ymax - ymin)/img_height
            x_center = (xmin + xmax) / (2 * img_width)
            y_center = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 确保坐标在[0,1]范围内（避免标注错误导致超出）
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            # 添加到TXT内容（保留6位小数，平衡精度与简洁）
            yolo_content.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 4. 保存为YOLO格式TXT文件（与XML同名，保存到输出目录）
        xml_filename = os.path.basename(xml_path)
        yolo_filename = os.path.splitext(xml_filename)[0] + ".txt"
        yolo_path = os.path.join(yolo_dir, yolo_filename)

        with open(yolo_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_content))

        # 打印进度
        print(f"✅ 已转换：{xml_filename} → {yolo_filename}（目标数：{len(yolo_content)}）")

    print(f"\n🎉 转换完成！所有YOLO格式TXT文件已保存至：{yolo_dir}")


# -------------------------- 配置参数（根据你的数据集修改） --------------------------
if __name__ == "__main__":
    # 1. 输入XML文件夹路径（替换为你的XML文件所在文件夹）
    XML_DIR = r"E:\data\train\label-xml"  # 示例：Windows路径
    # XML_DIR = "/home/user/dataset/labels/xml"  # 示例：Linux/Mac路径

    # 2. 输出YOLO-TXT文件夹路径（可自定义，不存在会自动创建）
    YOLO_DIR = r"E:\data\train\label-yolo"

    # 3. 类别映射字典（关键！需与你的数据集类别完全一致）
    # 格式：{"类别名称": 类别ID}，ID从0开始递增
    CLASS_MAPPING = {
        "late": 0,  # 示例：人→ID=0
        "early": 1,  # 示例：车→ID=1
        "mid": 2,  # 示例：狗→ID=2
    # 示例：猫→ID=3
        # 请根据你的数据集类别补充/修改
    }

    # 执行转换
    try:
        xml_to_yolo(XML_DIR, YOLO_DIR, CLASS_MAPPING)
    except Exception as e:
        print(f"❌ 转换失败：{str(e)}")