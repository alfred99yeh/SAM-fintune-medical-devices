import json
import os

def polygon_to_bbox(polygon):
    """
    将多边形坐标转换为边界框。
    Args:
        polygon (list): 多边形的 [x1, y1, x2, y2, ...] 坐标列表。
    Returns:
        list: Bounding Box 的 [xmin, ymin, xmax, ymax]。
    """
    x_coords = [point[0] for point in polygon]  # 提取所有 x 坐标
    y_coords = [point[1] for point in polygon]  # 提取所有 y 坐标

    xmin, ymin = min(x_coords), min(y_coords)
    xmax, ymax = max(x_coords), max(y_coords)
    return [xmin, ymin, xmax, ymax]  # 返回 [xmin, ymin, xmax, ymax]

def convert_polygon_to_bbox(json_file, output_folder):
    """
    将 Labelme JSON 文件中的多边形标注转换为边界框标注，保存新的 JSON 文件，仅处理 `NG` 类别。
    Args:
        json_file (str): 输入的 Labelme JSON 文件路径。
        output_folder (str): 输出的标注文件文件夹路径。
    """
    # 加载原始的 Labelme JSON 文件
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 解析图像信息
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    image_name = os.path.splitext(data["imagePath"])[0]  # 去掉文件扩展名

    # 初始化新的 JSON 结构
    new_shapes = []
    new_annotations = []

    # 转换每个多边形标注为边界框
    for shape in data["shapes"]:
        label = shape["label"]
        if label == "NG":  # 仅处理 `NG` 类别
            # 计算多边形的边界框
            points = shape["points"]
            bbox = polygon_to_bbox(points)

            # 将新标注（边界框）添加到 shapes 中
            new_shapes.append({
                "label": label,
                "points": [  # Labelme 需要的矩形坐标
                    [bbox[0], bbox[1]],  # 左上角 (xmin, ymin)
                    [bbox[2], bbox[3]]   # 右下角 (xmax, ymax)
                ],
                "group_id": None,
                "shape_type": "rectangle",  # 记录为矩形
                "flags": {}
            })

            # 更新新的 annotations
            new_annotations.append({
                "bbox": bbox,  # Bounding Box
                "category_id": 1,  # `NG` 类别对应 ID 为 1
                "iscrowd": 0
            })

    # 创建新的 JSON 结构，将新的 annotations 放入其中
    new_data = {
        "version": data["version"],
        "flags": data["flags"],
        "shapes": new_shapes,  # 将边界框放入 shapes
        "annotations": new_annotations,  # 添加 annotations
        "imagePath": data["imagePath"],
        "imageData": data["imageData"],
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 保存转换后的 JSON 文件
    new_json_file = os.path.join(output_folder, f"{image_name}.json")
    with open(new_json_file, "w") as f:
        json.dump(new_data, f, indent=4)

    print(f"已保存转换后的 JSON 文件：{new_json_file}")

def process_labelme_folder(json_folder, output_folder):
    """
    批量处理文件夹中的所有 Labelme JSON 文件，将多边形标注转换为边界框标注，仅处理 `NG` 类别。
    Args:
        json_folder (str): 包含 Labelme JSON 文件的文件夹路径。
        output_folder (str): 输出标注文件的文件夹路径。
    """
    # 遍历文件夹中的所有 JSON 文件
    for root, _, files in os.walk(json_folder):
        for json_file in files:
            if json_file.endswith(".json"):  # 仅处理 JSON 文件
                json_path = os.path.join(root, json_file)
                convert_polygon_to_bbox(json_path, output_folder)

# 使用示例
json_folder = "yellow_original_all_data/yellow_original_BMP&json"  # 包含 Labelme JSON 文件的文件夹路径
output_folder = "yellow_original_all_data/bbox"  # 输出标注文件夹

# 批量处理所有 JSON 文件，仅保留 `NG` 类别
process_labelme_folder(json_folder, output_folder)
