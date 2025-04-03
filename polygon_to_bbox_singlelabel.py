import json
import os

def polygon_to_bbox(polygon):
    """
    將多邊形轉換成 bounding box。
    Args:
        polygon (list): 多邊形的 [x1, y1, x2, y2, ...] 座標。
    Returns:
        list: Bounding Box 的 [xmin, ymin, xmax, ymax]。
    """
    x_coords = [point[0] for point in polygon]  # 提取所有 x 座標
    y_coords = [point[1] for point in polygon]  # 提取所有 y 座標
    xmin, ymin = min(x_coords), min(y_coords)
    xmax, ymax = max(x_coords), max(y_coords)
    return [xmin, ymin, xmax, ymax]  # 返回 [xmin, ymin, xmax, ymax]

def convert_polygon_to_bbox(json_file, output_folder):
    """
    將 Labelme JSON 文件中的多邊形標註轉換為 bbox 標註，保存新的 JSON 文件，僅處理 `NG` 類別。
    Args:
        json_file (str): 輸入的 Labelme JSON 文件路徑。
        output_folder (str): 輸出的標註文件夾路徑。
    """
    # 原始的 Labelme JSON 文件
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 獲取圖像的寬度和高度
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    image_name = os.path.splitext(data["imagePath"])[0]

    # 初始化新的 JSON 結構
    new_shapes = []
    new_annotations = []

    # 轉換每個多邊形標註為 bbox
    for shape in data["shapes"]:
        label = shape["label"]
        if label == "NG":  # 僅處理 `NG` 類別
            # 計算多邊形的邊界框
            points = shape["points"]
            bbox = polygon_to_bbox(points)

            # 將邊界框轉換為 Labelme 所需的格式
            new_shapes.append({
                "label": label,
                "points": [  # 將邊界框轉換為 Labelme 所需的格式
                    [bbox[0], bbox[1]],  # 左上角 (xmin, ymin)
                    [bbox[2], bbox[3]]   # 右下角 (xmax, ymax)
                ],
                "group_id": None,
                "shape_type": "rectangle",  # 矩形類型
                "flags": {}
            })

            # 更新新的 annotations
            new_annotations.append({
                "bbox": bbox,  # Bounding Box
                "category_id": 1,  # 類別 ID，這裡假設 `NG` 類別的 ID 為 1
                "iscrowd": 0
            })

    # 組裝新的 JSON 結構
    new_data = {
        "version": data["version"],
        "flags": data["flags"],
        "shapes": new_shapes, # 添加新的 shapes
        "annotations": new_annotations, # 添加 annotations
        "imagePath": data["imagePath"],
        "imageData": data["imageData"],
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # 創建輸出文件夾（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 保存新的 JSON 文件
    new_json_file = os.path.join(output_folder, f"{image_name}.json")
    with open(new_json_file, "w") as f:
        json.dump(new_data, f, indent=4)

    print(f"已保存转换后的 JSON 文件：{new_json_file}")

def process_labelme_folder(json_folder, output_folder):
    """
    批量處理文件夾中的所有 Labelme JSON 文件，將多邊形標註轉換為邊界框標註，僅處理 `NG` 類別。
    Args:
        json_folder (str): 包含 Labelme JSON 文件的文件夾路徑。
        output_folder (str): 輸出標註文件的文件夾路徑。
    """
    # 遍歷文件夾中的所有 JSON 文件
    for root, _, files in os.walk(json_folder):
        for json_file in files:
            if json_file.endswith(".json"):  # 檢查文件是否為 JSON 文件
                json_path = os.path.join(root, json_file)
                convert_polygon_to_bbox(json_path, output_folder)

# 使用示例
json_folder = "yellow_original_all_data/yellow_original_BMP&json"  # 輸入的 JSON 文件夾
output_folder = "yellow_original_all_data/bbox"  # 輸出的標註文件夾

# 批量處理文件夾中的所有 Labelme JSON 文件
process_labelme_folder(json_folder, output_folder)
