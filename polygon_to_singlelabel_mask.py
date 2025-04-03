import json
import os
import numpy as np
from PIL import Image, ImageDraw

def create_single_channel_mask_from_labelme(json_file, output_folder, label_mapping):
    """
    將單個 Labelme JSON 文件轉換為單通道 Ground Truth Mask。
    僅處理 `NG` 類别，其他類別忽略。
    
    Args:
        json_file (str): Labelme JSON 文件的路徑。
        output_folder (str): 输出遮罩保存的文件夾路徑。
        label_mapping (dict): 類别與像素值的mapping。
    """
    # 讀取 JSON 文件
    with open(json_file, "r") as f:
        data = json.load(f)

    # 獲取圖像的寬度和高度
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    image_name = os.path.splitext(data["imagePath"])[0]  # 去掉文件擴展名

    # 初始化單通道掩碼
    single_channel_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # 遍歷所有標註的多邊形
    for shape in data["shapes"]:
        label = shape["label"]  # 獲取標註的類別
        if label == "NG":  # 僅處理 `NG` 類別
            points = shape["points"]  # 獲取多邊形的點
            polygon = [(int(x), int(y)) for x, y in points]  # 將點轉換為整數

            # 使用 PIL 的 ImageDraw 繪製多邊形
            mask_channel = Image.fromarray(single_channel_mask)  # 創建一個新的掩碼圖像
            ImageDraw.Draw(mask_channel).polygon(polygon, outline=1, fill=255)  # 填充多邊形
            single_channel_mask = np.array(mask_channel)  # 更新遮罩圖像

    # 將單通道掩碼轉換為指定的像素值
    os.makedirs(output_folder, exist_ok=True)

    # 將像素值映射到指定的類別
    mask_output_path = os.path.join(output_folder, f"{image_name}_mask.png")
    mask_image = Image.fromarray(single_channel_mask)
    mask_image.save(mask_output_path)
    print(f"遮罩已保存至: {mask_output_path}")

def process_labelme_folder(json_folder, output_folder, label_mapping):
    """
    批量處理文件夾中的所有 Labelme JSON 文件，生成單標籤單通道 Ground Truth Mask。
    
    Args:
        json_folder (str): 包含 Labelme JSON 文件的文件夾路徑。
        output_folder (str): Ground Truth Mask 的輸出文件夹。
        label_mapping (dict): 類别與像素值的 mapping 關係。
    """
    # 遍歷文件夾中的所有 JSON 文件
    for root, _, files in os.walk(json_folder):
        for json_file in files:
            if json_file.endswith(".json"):  # 僅處理 JSON 文件
                json_path = os.path.join(root, json_file)
                create_single_channel_mask_from_labelme(json_path, output_folder, label_mapping)

# 設置文件夾路徑
json_folder = "yellow_original_all_data/yellow_original_barry/yellow_all_json"  # 輸入的 JSON 文件夾
output_folder = "yellow_original_all_data/yellow_original_barry/yellow_all_masks"  # 輸出的遮罩文件夾

# 設置類別與像素值的映射關係
label_mapping = {
    "NG": 1  # 將 `NG` 類別映射到像素值 1
}

# 處理文件夾中的所有 JSON 文件
process_labelme_folder(json_folder, output_folder, label_mapping)
