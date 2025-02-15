import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SamProcessor
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import SamModel
from tqdm import tqdm  # 引入 tqdm 库

# Custom Dataset class
class SAMDataset(Dataset):
    def __init__(self, mask_folder, bbox_folder, image_folder, processor):
        self.mask_folder = mask_folder
        self.bbox_folder = bbox_folder
        self.image_folder = image_folder
        self.processor = processor
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.bmp')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Load corresponding mask
        mask_path = os.path.join(self.mask_folder, os.path.basename(image_path).replace('.bmp', '_mask.png'))
        mask = Image.open(mask_path).convert("L")

        # Load bounding box
        bbox_path = os.path.join(self.bbox_folder, os.path.basename(image_path).replace('.bmp', '.json'))
        with open(bbox_path, 'r') as f:
            bbox_data = json.load(f)

        bboxes = []
        labels = []
        for ann in bbox_data['annotations']:
            bboxes.append(ann['bbox'])
            labels.append(1 if ann['category_id'] == 1 else 0)  # 假设 `NG` 类别的 ID 为 1 # category_id, class_id

        if len(bboxes) == 0:
            bboxes = [[0, 0, 1, 1]]
            labels = [0]

        inputs = self.processor(images=image, return_tensors="pt")
        mask_array = np.array(mask)
        binary_mask = torch.tensor((mask_array > 0).astype(np.float32)).unsqueeze(0)

        inputs['input_boxes'] = torch.tensor(bboxes).float()
        inputs['labels'] = torch.tensor(labels)
        inputs['binary_masks'] = binary_mask

        return inputs
    
# Evaluation function
def evaluate_segmentation_with_cm(pred_masks, true_masks, num_classes):
    pred_masks = pred_masks.view(-1).cpu().numpy()
    true_masks = true_masks.view(-1).cpu().numpy()

    assert pred_masks.shape == true_masks.shape, (
        f"Shape mismatch: pred_masks {pred_masks.shape}, true_masks {true_masks.shape}"
    )

    cm = confusion_matrix(true_masks, pred_masks, labels=range(num_classes))

    iou_per_class = []
    f1_per_class = []
    accuracy_per_class = []
    recall_per_class = []

    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        iou = tp / (tp + fp + fn + 1e-7)
        iou_per_class.append(iou)

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        f1_per_class.append(f1)

        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-7)
        accuracy_per_class.append(accuracy)

        recall_per_class.append(recall)

    # Compute mAP@50 and mAP@50:95
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # IoU thresholds from 0.5 to 0.95 with step 0.05
    ap_per_threshold = []

    for threshold in iou_thresholds:
        aps = []
        for iou in iou_per_class:
            if iou >= threshold:
                aps.append(1)  # IoU satisfies threshold
            else:
                aps.append(0)
        ap_per_threshold.append(np.mean(aps))

    map50 = ap_per_threshold[0]  # mAP@50
    map5095 = np.mean(ap_per_threshold)  # mAP@50:95

    metrics = {
        "IoU": iou_per_class,
        "F1-score": f1_per_class,
        "Accuracy": accuracy_per_class,
        "Recall": recall_per_class,
        "mAP@50": map50,
        "mAP@50:95": map5095
    }

    return metrics, cm

# Confusion matrix plot
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# Confusion matrix plot
def plot_normalized_confusion_matrix(cm, classes):
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(norm_cm, annot=True, fmt='.4f', cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Normalized Confusion Matrix")
    plt.show()

# Metrics plot
def plot_metrics(metrics):
    # 将指标转换为 DataFrame，设置指标名为索引
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    metrics_df.plot(kind="bar", figsize=(10, 6), legend=False)
    plt.title("Evaluation Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.tight_layout()  # 调整布局以防止标题或标签被裁剪
    plt.axis(True)
    plt.show()

def visualize_masks(image, pred_mask, true_mask, save_path=None):
    """
    使用 Matplotlib 可视化原图、预测掩码和真实掩码。
    
    Args:
        image (PIL.Image): 原始图像。
        pred_mask (torch.Tensor): 预测掩码。
        true_mask (torch.Tensor): 真实掩码。
        save_path (str): 如果指定路径，将保存图片。
    """
    plt.figure(figsize=(12, 4))

    # 绘制原图
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # 绘制预测掩码
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask.cpu().numpy(), cmap="jet")
    plt.title("Predicted Mask")
    plt.axis("off")

    # 绘制真实掩码
    plt.subplot(1, 3, 3)
    plt.imshow(true_mask.cpu().numpy(), cmap="jet")
    plt.title("True Mask")
    plt.axis("off")

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label mapping
    label_mapping = {
        "NG": 1  # NG class label set to 1
    }

    # Data preparation
    mask_folder = "yellow_original_all_data/yellow_original_barry/yellow_all_masks" 
    bbox_folder = "yellow_original_all_data/bbox_annotations"
    image_folder = "yellow_original_all_data/yellow_original_processed"

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    full_dataset = SAMDataset(mask_folder, bbox_folder, image_folder, processor)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 使用多线程加载数据
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.load_state_dict(torch.load(
        "sam_finetuned_singlelabel_yellow.pth",
        map_location=device,
        weights_only=True
    ))
    model.to(device)

    # Inference and evaluation
    num_classes = len(label_mapping) + 1  # Add 1 for background class
    all_metrics = {"IoU": [], "F1-score": [], "Accuracy": [], "Recall": [], "mAP@50": [], "mAP@50:95": []}
    total_cm = np.zeros((num_classes, num_classes), dtype=int)

    # 用 tqdm 包裹 val_loader，显示进度条
    for idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        pixel_values = batch['pixel_values'].squeeze(1).to(device)
        input_boxes = batch['input_boxes'].to(device)
        true_masks = batch['binary_masks'].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes)

        pred_masks = outputs.pred_masks.sigmoid()  # 获取预测掩码
        pred_masks = pred_masks.max(dim=1)[0]  # 将 3 通道压缩为 1 通道
        pred_masks = (pred_masks > 0.5).long()  # 二值化

        # 将 3 通道的 pred_masks 转换为 1 通道
        pred_masks, _ = pred_masks.max(dim=1, keepdim=True)  # 按通道取最大值
        pred_masks = pred_masks.squeeze(1)

        # 调整 true_masks 的尺寸
        resized_true_masks = F.interpolate(
            input=true_masks,
            size=pred_masks.shape[-2:],
            mode="nearest"  # 最近邻插值
        ).squeeze(1)  # 移除多余的通道维度

        resized_true_masks = (resized_true_masks > 0.5).long()

        # 确保形状一致
        assert resized_true_masks.shape == pred_masks.shape, (
            f"Shape mismatch after resizing: true_masks {resized_true_masks.shape}, pred_masks {pred_masks.shape}"
        )

        # 从 val_dataset 获取原始索引
        original_idx = val_dataset.indices[idx]  # 使用 random_split 时保留的索引
        original_image_path = full_dataset.image_paths[original_idx]
        original_image = Image.open(original_image_path).convert("RGB")

        # 可视化并保存
        visualize_masks(original_image, pred_masks[0], resized_true_masks[0],
                        save_path=f"output/mask_comparison_{idx}.png")

        # 评估
        batch_metrics, batch_cm = evaluate_segmentation_with_cm(pred_masks, resized_true_masks, num_classes)

        for metric, values in batch_metrics.items():
            all_metrics[metric].append(np.mean(values))

        total_cm += batch_cm



    final_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
    print("Final Evaluation Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(total_cm, classes=["Background", "NG"])
    plot_normalized_confusion_matrix(total_cm, classes=["Background", "NG"])

    # 绘制评估指标
    plot_metrics(final_metrics)

    print("Confusion Matrix:")
    print(total_cm)
    print(total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis])
