import json
import os
import numpy as np
from PIL import Image, ImageDraw

def create_single_channel_mask_from_labelme(json_file, output_folder, label_mapping):
    """
    将单个 Labelme JSON 文件转换为单通道 Ground Truth Mask。
    仅处理 `NG` 类别，其他类别被忽略。
    
    Args:
        json_file (str): Labelme JSON 文件的路径。
        output_folder (str): 输出遮罩保存的文件夹路径。
        label_mapping (dict): 类别与像素值的对应关系。
    """
    # 加载 JSON 文件
    with open(json_file, "r") as f:
        data = json.load(f)

    # 解析图像信息
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    image_name = os.path.splitext(data["imagePath"])[0]  # 去掉文件的扩展名

    # 创建一个单通道掩码 (H, W)
    single_channel_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # 遍历每个标注形状，绘制属于 `NG` 类别的多边形
    for shape in data["shapes"]:
        label = shape["label"]  # 类别名称
        if label == "NG":  # 仅处理 `NG` 类别
            points = shape["points"]  # 获取多边形的点
            polygon = [(int(x), int(y)) for x, y in points]  # 转换为整数坐标

            # 在掩码中绘制多边形
            mask_channel = Image.fromarray(single_channel_mask)  # 转换为 PIL Image
            ImageDraw.Draw(mask_channel).polygon(polygon, outline=1, fill=255)  # 填充多边形
            single_channel_mask = np.array(mask_channel)  # 更新掩码

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 保存单通道掩码
    mask_output_path = os.path.join(output_folder, f"{image_name}_mask.png")
    mask_image = Image.fromarray(single_channel_mask)
    mask_image.save(mask_output_path)
    print(f"遮罩已保存至: {mask_output_path}")

def process_labelme_folder(json_folder, output_folder, label_mapping):
    """
    批量处理文件夹中的所有 Labelme JSON 文件，生成单标签单通道 Ground Truth Mask。
    
    Args:
        json_folder (str): 包含 Labelme JSON 文件的文件夹路径。
        output_folder (str): Ground Truth Mask 的输出文件夹。
        label_mapping (dict): 类别与像素值的对应关系。
    """
    # 遍历文件夹中的所有 JSON 文件
    for root, _, files in os.walk(json_folder):
        for json_file in files:
            if json_file.endswith(".json"):  # 仅处理 JSON 文件
                json_path = os.path.join(root, json_file)
                create_single_channel_mask_from_labelme(json_path, output_folder, label_mapping)

# 使用示例
json_folder = "yellow_original_all_data/yellow_original_barry/yellow_all_json"  # 包含 Labelme JSON 文件的文件夹路径
output_folder = "yellow_original_all_data/yellow_original_barry/yellow_all_masks"  # Ground Truth Mask 的输出文件夹

# 定义类别与像素值的对应关系，仅 `NG` 被标记为前景
label_mapping = {
    "NG": 1  # `NG` 类别对应像素值为 1
}

# 批量处理所有 JSON 文件，仅保留 `NG` 类别
process_labelme_folder(json_folder, output_folder, label_mapping)
