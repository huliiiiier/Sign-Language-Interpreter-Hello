import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# 配置路径
POSITIVE_VIDEO_DIR = r"data set\你好"
NEGATIVE_VIDEO_DIR = r"data set\负样本"
VAL_POS_DIR = r"data set\验证集\你好"
VAL_NEG_DIR = r"data set\验证集\非你好"
OUTPUT_DIR = r"frames"
MAX_FRAMES = 16  # 每个视频提取帧数

# 数据增强管道（离线增强）
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
])

def extract_frames(video_path, output_dir):
    """从视频中均匀采样帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, total_frames // MAX_FRAMES)

    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    saved_count = 0

    while saved_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            resized = cv2.resize(frame, (128, 128))
            cv2.imwrite(f"{output_dir}/frame_{saved_count:03d}.jpg", resized)
            saved_count += 1
        frame_count += 1
    cap.release()
    return saved_count

def augment_frames(input_dir, output_dir, num_augments=2):
    """对单个视频帧进行数据增强"""
    frames = []
    for i in range(MAX_FRAMES):
        frame_path = os.path.join(input_dir, f"frame_{i:03d}.jpg")
        if not os.path.exists(frame_path):
            break
        img = Image.open(frame_path)
        frames.append(img)

    for aug_idx in range(num_augments):
        aug_dir = os.path.join(output_dir, f"aug{aug_idx}")
        os.makedirs(aug_dir, exist_ok=True)
        for i, img in enumerate(frames):
            augmented = transform(img)
            augmented.save(os.path.join(aug_dir, f"frame_{i:03d}.jpg"))

def extract_validation_frames():
    """提取验证集帧"""
    val_pos_output = os.path.join(OUTPUT_DIR, "val_pos")
    val_neg_output = os.path.join(OUTPUT_DIR, "val_neg")

    # 提取验证集正样本
    for video_file in os.listdir(VAL_POS_DIR):
        if not video_file.endswith(".mp4"):
            continue
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(VAL_POS_DIR, video_file)
        output_subdir = os.path.join(val_pos_output, video_id)
        extract_frames(video_path, output_subdir)

    # 提取验证集负样本
    for video_file in os.listdir(VAL_NEG_DIR):
        if not video_file.endswith(".mp4"):
            continue
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(VAL_NEG_DIR, video_file)
        output_subdir = os.path.join(val_neg_output, video_id)
        extract_frames(video_path, output_subdir)

def generate_extra_val_samples():
    """从训练集中抽取部分样本作为新验证集样本"""
    extra_pos_output = os.path.join(OUTPUT_DIR, "val_pos")
    extra_neg_output = os.path.join(OUTPUT_DIR, "val_neg")

    # 新增正样本：从 dev16-dev18
    for i in range(16, 19):  # dev16, dev17, dev18
        video_file = f"dev{i:02d}.mp4"
        video_path = os.path.join(POSITIVE_VIDEO_DIR, video_file)
        if not os.path.exists(video_path):
            continue
        output_subdir = os.path.join(extra_pos_output, f"dev{i:02d}")
        extract_frames(video_path, output_subdir)
        augment_frames(output_subdir, output_subdir, num_augments=2)  # 每个视频生成2个增强版本

    # 新增负样本：从 f17-f19
    for i in range(17, 20):  # f17, f18, f19
        video_file = f"f{i}.mp4"
        video_path = os.path.join(NEGATIVE_VIDEO_DIR, video_file)
        if not os.path.exists(video_path):
            continue
        output_subdir = os.path.join(extra_neg_output, f"f{i}")
        extract_frames(video_path, output_subdir)
        augment_frames(output_subdir, output_subdir, num_augments=1)  # 每个视频生成1个增强版本

if __name__ == "__main__":
    # 1. 提取正样本帧
    for video_file in os.listdir(POSITIVE_VIDEO_DIR):
        if not video_file.endswith(".mp4"):
            continue
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(POSITIVE_VIDEO_DIR, video_file)
        output_subdir = os.path.join(OUTPUT_DIR, "pos", video_id)
        extract_frames(video_path, output_subdir)

    # 2. 提取负样本帧
    for video_file in os.listdir(NEGATIVE_VIDEO_DIR):
        if not video_file.startswith("f") or not video_file.endswith(".mp4"):
            continue
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(NEGATIVE_VIDEO_DIR, video_file)
        output_subdir = os.path.join(OUTPUT_DIR, "neg", video_id)
        extract_frames(video_path, output_subdir)

    # 3. 数据增强（正样本生成3个增强版本，负样本生成1个）
    for video_id in [f"dev{i:02d}" for i in range(1, 16)]:
        original_dir = os.path.join(OUTPUT_DIR, "pos", video_id)
        if not os.path.exists(original_dir):
            continue
        augment_frames(original_dir, original_dir, num_augments=3)  # 正样本生成3个增强版本

    for video_id in [f"f{i}" for i in range(1, 17)]:
        original_dir = os.path.join(OUTPUT_DIR, "neg", video_id)
        if not os.path.exists(original_dir):
            continue
        augment_frames(original_dir, original_dir, num_augments=1)  # 负样本生成1个增强版本

    # 4. 提取验证集帧
    extract_validation_frames()

    # 5. 生成更多验证集样本
    generate_extra_val_samples()