import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# 参数配置
DATA_DIR = r"frames"
MODEL_PATH = r"hello_model.pth"
IMAGE_SIZE = (128, 128)
MAX_FRAMES = 16
BATCH_SIZE = 4
EPOCHS = 100
THRESHOLD = 0.5
VAL_SPLIT = 0.2
USE_KFOLD = False
NUM_FOLDS = 5

# Tensor-based 数据增强管道
train_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x).float().div(255)),  # NumPy -> Tensor [0,1]
    transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # HWC -> CHW
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x).float().div(255)),  # NumPy -> Tensor [0,1]
    transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # HWC -> CHW
])

class HelloDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        pos_dir = os.path.join(data_dir, "pos")
        neg_dir = os.path.join(data_dir, "neg")

        for d, label in [(pos_dir, 1), (neg_dir, 0)]:
            for video_dir in os.listdir(d):
                self.samples.append((os.path.join(d, video_dir), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = []

        for i in range(MAX_FRAMES):
            frame_path = os.path.join(path, f"frame_{i:03d}.jpg")
            if not os.path.exists(frame_path):
                break
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, IMAGE_SIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).float().div(255).permute(2, 0, 1)  # HWC -> CHW
            frames.append(frame)

        # 填充不足帧数
        while len(frames) < MAX_FRAMES:
            zero_frame = torch.zeros_like(frames[0]) if frames else torch.zeros(3, 128, 128)
            frames.append(zero_frame)

        # 堆叠帧并转换为 (C, T, H, W)
        video = torch.stack(frames)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # -> (C, T, H, W)
        return video, torch.tensor(label, dtype=torch.float32)

# Temporal Shift Module (TSM)
class TSM(nn.Module):
    def __init__(self, num_segments, fold_div=3):
        super().__init__()
        self.num_segments = num_segments
        self.fold_div = fold_div

    def forward(self, x):
        B_T, C, H, W = x.size()
        B = B_T // self.num_segments
        x = x.view(B, self.num_segments, C, H, W)

        fold = C // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # 向前移位
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # 向后移位
        out[:, :, 2*fold:] = x[:, :, 2*fold:]  # 保留不变部分

        return out.view(B_T, C, H, W)

# MobileNetV3 + TSM 模型
class MobileNetV3TSM(nn.Module):
    def __init__(self, num_segments=16, pretrained=True):
        super().__init__()
        self.num_segments = num_segments

        # 使用 MobileNetV3 Small 预训练模型
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])  # 去掉最后的分类层

        self.tsm = TSM(num_segments=num_segments)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)

        x = self.tsm(x)
        x = self.feature_extractor(x)  # (B*T, 576, H/8, W/8)

        x = self.classifier(x)
        x = x.view(B, T, -1).mean(dim=1)  # 时间轴平均池化
        return x  # 不再 squeeze

def train_model_with_cv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HelloDataset(DATA_DIR, transform=train_transform)
    labels = [label for _, label in dataset.samples]

    if USE_KFOLD:
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), labels)):
            print(f"\n--- Fold {fold+1}/{NUM_FOLDS} ---")
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))
            model = MobileNetV3TSM(num_segments=MAX_FRAMES).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
            optimizer = optim.AdamW(model.parameters(), lr=3e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
            scaler = GradScaler() if torch.cuda.is_available() else None
            best_loss = float('inf')
            patience_counter = 0
            EARLY_STOPPING_PATIENCE = 15

            for epoch in range(EPOCHS):
                model.train()
                running_loss = 0.0
                all_train_preds, all_train_labels = [], []

                for videos, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                    videos = videos.to(device)
                    labels_batch = labels_batch.to(device).view(-1, 1)  # 保证形状 (B, 1)

                    if scaler:
                        with autocast():
                            outputs = model(videos)  # 保留原始形状
                            loss = criterion(outputs, labels_batch)
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(videos)
                        loss = criterion(outputs, labels_batch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    preds = torch.sigmoid(outputs) > THRESHOLD
                    all_train_preds.extend(preds.cpu().numpy())
                    all_train_labels.extend(labels_batch.cpu().numpy())

                train_acc = np.mean(np.array(all_train_preds) == np.array(all_train_labels))
                train_prec = precision_score(all_train_labels, all_train_preds, zero_division=0)
                train_rec = recall_score(all_train_labels, all_train_preds, zero_division=0)
                train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)

                print(f"Epoch {epoch+1}/{EPOCHS} Loss: {running_loss/len(train_loader):.4f}")
                print(f"Train - Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")

                # 验证阶段
                model.eval()
                val_loss = 0.0
                all_val_preds, all_val_labels = [], []

                with torch.no_grad():
                    for videos, labels_batch in val_loader:
                        videos = videos.to(device)
                        labels_batch = labels_batch.to(device).view(-1, 1)
                        outputs = model(videos)
                        loss = criterion(outputs, labels_batch)
                        val_loss += loss.item()

                        preds = torch.sigmoid(outputs) > THRESHOLD
                        all_val_preds.extend(preds.cpu().numpy())
                        all_val_labels.extend(labels_batch.cpu().numpy())

                val_acc = np.mean(np.array(all_val_preds) == np.array(all_val_labels))
                val_prec = precision_score(all_val_labels, all_val_preds, zero_division=0)
                val_rec = recall_score(all_val_labels, all_val_preds, zero_division=0)
                val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
                avg_val_loss = val_loss / len(val_loader)

                print(f"Val   - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}\n")
                scheduler.step()

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model.state_dict(), f"{MODEL_PATH}_fold{fold+1}.pth")
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

            print(f"模型已保存至 {MODEL_PATH}_fold{fold+1}.pth")
    else:
        indices = np.random.permutation(len(dataset))
        split = int((1 - VAL_SPLIT) * len(dataset))
        train_idx, val_idx = indices[:split], indices[split:]
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))

        model = MobileNetV3TSM(num_segments=MAX_FRAMES).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = GradScaler() if torch.cuda.is_available() else None
        best_loss = float('inf')
        patience_counter = 0
        EARLY_STOPPING_PATIENCE = 15

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            all_train_preds, all_train_labels = [], []

            for videos, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                videos = videos.to(device)
                labels_batch = labels_batch.to(device).view(-1, 1)  # 保证标签形状为 (B, 1)

                if scaler:
                    with autocast():
                        outputs = model(videos)
                        loss = criterion(outputs, labels_batch)
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(videos)
                    loss = criterion(outputs, labels_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                preds = torch.sigmoid(outputs) > THRESHOLD
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels_batch.cpu().numpy())

            train_acc = np.mean(np.array(all_train_preds) == np.array(all_train_labels))
            train_prec = precision_score(all_train_labels, all_train_preds, zero_division=0)
            train_rec = recall_score(all_train_labels, all_train_preds, zero_division=0)
            train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)

            print(f"Epoch {epoch+1}/{EPOCHS} Loss: {running_loss/len(train_loader):.4f}")
            print(f"Train - Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")

            # 验证阶段
            model.eval()
            val_loss = 0.0
            all_val_preds, all_val_labels = [], []

            with torch.no_grad():
                for videos, labels_batch in val_loader:
                    videos = videos.to(device)
                    labels_batch = labels_batch.to(device).view(-1, 1)  # 保证标签形状为 (B, 1)
                    outputs = model(videos)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()

                    preds = torch.sigmoid(outputs) > THRESHOLD
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(labels_batch.cpu().numpy())

            val_acc = np.mean(np.array(all_val_preds) == np.array(all_val_labels))
            val_prec = precision_score(all_val_labels, all_val_preds, zero_division=0)
            val_rec = recall_score(all_val_labels, all_val_preds, zero_division=0)
            val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Val   - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}\n")
            scheduler.step()

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"模型已保存至 {MODEL_PATH}")

if __name__ == "__main__":
    train_model_with_cv()