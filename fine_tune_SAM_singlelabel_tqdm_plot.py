import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SamProcessor
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from transformers import SamModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# SAMDataset
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

        # 加载对应的单通道 Mask
        mask_path = os.path.join(self.mask_folder, os.path.basename(image_path).replace('.bmp', '_mask.png'))
        mask = Image.open(mask_path).convert("L")  # 單通道灰階

        # 加载 Bounding Box
        bbox_path = os.path.join(self.bbox_folder, os.path.basename(image_path).replace('.bmp', '.json'))
        with open(bbox_path, 'r') as f:
            bbox_data = json.load(f)

        bboxes = []
        labels = []
        for ann in bbox_data['annotations']:
            bboxes.append(ann['bbox'])
            # 所有标注均为 NG 类别，背景为 0
            labels.append(1 if ann['category_id'] == 1 else 0)  # 'NG' ID 為 1 # category_id, class_id

        # 检查是否存在有效的 bboxes
        if len(bboxes) == 0:
            bboxes = [[0, 0, 1, 1]]  # 添加虚拟框
            labels = [0]

        inputs = self.processor(images=image, return_tensors="pt")

        mask_array = np.array(mask)
        binary_mask = torch.tensor((mask_array > 0).astype(np.float32)).unsqueeze(0)  # 單通道

        inputs['input_boxes'] = torch.tensor(bboxes).float()
        inputs['labels'] = torch.tensor(labels)
        inputs['binary_masks'] = binary_mask

        return inputs

# Train and Validation
def train_and_evaluate():
    mask_folder = "yellow_original_all_data/yellow_original_barry/yellow_all_masks" 
    bbox_folder = "yellow_original_all_data/bbox_annotations" 
    image_folder = "yellow_original_all_data/yellow_original_processed"

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    full_dataset = SAMDataset(mask_folder, bbox_folder, image_folder, processor)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # multithreads
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.load_state_dict(torch.load(
        "sam_finetuned_singlelabel_yellow_5.pth",
        map_location="cuda",
        weights_only=True
    ))
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-6)

    def compute_loss(pred_masks, true_masks):
        # 選擇第一個通道對應true_masks
        pred_masks = pred_masks[:, 0, :, :]  # 選擇第一個通道
        pred_masks = pred_masks[:, 0, :].unsqueeze(1)
        
        # 确保 true_masks 和 pred_masks的shape一致
        true_masks = F.interpolate(true_masks, size=pred_masks.shape[-2:], mode='bilinear', align_corners=False).to(pred_masks.device)

        # 使用二元交叉熵损失计算
        loss = F.binary_cross_entropy_with_logits(pred_masks, true_masks)
        
        return loss

    train_losses, val_losses = [], []
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        # training process
        with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}") as t:
            for batch in t:
                optimizer.zero_grad()

                pixel_values = batch['pixel_values'].squeeze(1).to(device)
                input_boxes = batch['input_boxes'].to(device)
                true_masks = batch['binary_masks'].to(device)

                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes)
                pred_masks = outputs.pred_masks

                loss = compute_loss(pred_masks, true_masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                t.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss}")

        val_loss = 0.0
        model.eval()

        # validating process
        with tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}") as t:
            with torch.no_grad():
                for batch in t:
                    pixel_values = batch['pixel_values'].squeeze(1).to(device)
                    input_boxes = batch['input_boxes'].to(device)
                    true_masks = batch['binary_masks'].to(device)

                    outputs = model(pixel_values=pixel_values, input_boxes=input_boxes)
                    pred_masks = outputs.pred_masks
                    loss = compute_loss(pred_masks, true_masks)
                    val_loss += loss.item()
                    t.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}")
        # torch.save(model.state_dict(), f"sam_finetuned_singlelabel_yellow_{epoch}.pth")

    torch.save(model.state_dict(), "sam_finetuned_singlelabel_yellow.pth")

    # 顯示Loss圖
    def plot_metrics(train_losses, val_losses):
        plt.figure(figsize=(12, 6))

        # Loss
        plt.plot(train_losses, label="Train Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig("training_metrics_singlelabel_yellow.png")
        plt.show()

    plot_metrics(train_losses, val_losses)

# 在主程序中运行
if __name__ == "__main__":
    train_and_evaluate()
