import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# 配置参数
MODEL_PATH = "hello_model.pth"
IMAGE_SIZE = (128, 128)
MAX_FRAMES = 16
THRESHOLD = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out.view(B_T, C, H, W)

# MobileNetV3 + TSM 模型
class MobileNetV3TSM(nn.Module):
    def __init__(self, num_segments=16, pretrained=True):
        super().__init__()
        self.num_segments = num_segments

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])

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
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.tsm(x)
        x = self.feature_extractor(x)

        x = self.classifier(x)
        x = x.view(B, T, -1).mean(dim=1)
        return x.squeeze()

def extract_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, total_frames // max_frames)

    frames = []
    frame_count = 0
    saved_count = 0

    try:
        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_interval == 0:
                resized = cv2.resize(frame, IMAGE_SIZE)
                frames.append(resized)
                saved_count += 1
            frame_count += 1
    finally:
        cap.release()

    while len(frames) < max_frames:
        frames.append(np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32))

    return np.stack(frames)

def predict_video(model, video_path):
    try:
        frames = extract_frames(video_path)
    except Exception as e:
        return None, str(e)

    video_tensor = torch.from_numpy(frames).float().div(255).permute(3, 0, 1, 2)
    video_tensor = video_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(video_tensor)
        prob = torch.sigmoid(output).item()

    result = "✅ 检测结果：你好" if prob > THRESHOLD else "❌ 检测结果：非“你好”手势"
    return result, None

# GUI 应用
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("你好识别模型:hello in hand")
        self.root.geometry("700x500")

        # 加载模型
        try:
            self.model = MobileNetV3TSM(num_segments=MAX_FRAMES, pretrained=False).to(device)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            self.model.eval()
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {e}")
            root.destroy()
            return

        # 输入区域
        input_frame = tk.Frame(root)
        input_frame.pack(pady=10, fill=tk.X, padx=20)

        self.path_var = tk.StringVar()
        self.entry = tk.Entry(input_frame, textvariable=self.path_var, width=50)
        self.entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.select_btn = tk.Button(input_frame, text="选择文件", command=self.select_files)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.run_btn = tk.Button(input_frame, text="开始检测", command=self.run_predictions)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        # 输出区域
        output_frame = tk.Frame(root)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.output_text = tk.Text(output_frame, wrap='word', state='disabled')
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)

    def select_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("MP4 files", "*.mp4")])
        if paths:
            self.file_paths = paths
            self.path_var.set(f"{len(paths)} 个文件已选中")

    def show_output(self, msg):
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, msg)
        self.output_text.config(state='disabled')
        self.output_text.see(tk.END)

    def run_predictions(self):
        if not hasattr(self, 'file_paths'):
            self.show_output("请先选择一个或多个 MP4 视频文件。\n")
            return

        for path in self.file_paths:
            filename = os.path.basename(path)
            self.show_output(f"正在处理：{filename}\n")

            if not os.path.exists(path) or not path.endswith(".mp4"):
                self.show_output(f"❌ 路径无效：{path}\n\n")
                continue

            self.show_output("正在分析视频，请稍候...\n")
            result, error = predict_video(self.model, path)

            if error:
                self.show_output(f"❌检测过程中出错：{error}\n\n")
            else:
                self.show_output(f"   {result}\n\n")

# 主程序入口
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()