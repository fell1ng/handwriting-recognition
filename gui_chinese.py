import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw
import tkinter as tk

# ============ 配置 ============
BASE_DIR = r"D:\code\基于深度学习的手写体识别系统设计"
MODELS_DIR = os.path.join(BASE_DIR, "models")
CANVAS_SIZE = 320
BRUSH_SIZE = 8
IMG_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 1. 定义三种模型结构 ============

# 数字模型（和 train_mnist.py 一致）
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 字母模型（和 train_letters.py 一致）
class LetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 26)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 汉字模型
def create_chinese_model(num_classes):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, num_classes)
    return model

# ============ 2. 加载所有模型 ============
print("加载模型中...")

# 数字模型
mnist_model = MNIST_CNN().to(device)
mnist_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "mnist_cnn.pth"), map_location=device))
mnist_model.eval()
print("  数字模型 OK")

# 字母模型
letter_model = LetterCNN().to(device)
letter_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "emnist_letters_cnn.pth"), map_location=device))
letter_model.eval()
print("  字母模型 OK")

# 汉字模型
with open(os.path.join(MODELS_DIR, "chinese_classes.json"), "r", encoding="utf-8") as f:
    chinese_classes = json.load(f)
chinese_model = create_chinese_model(len(chinese_classes)).to(device)
chinese_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "chinese_resnet18.pth"), map_location=device))
chinese_model.eval()
print(f"  汉字模型 OK（{len(chinese_classes)} 类）")

# ============ 3. 预处理 ============
# 数字和字母模型：28x28
transform_mnist = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 汉字模型：64x64
transform_chinese = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ============ 4. 推理函数 ============

def crop_to_content(img):
    """裁到最小外接矩形"""
    arr = np.array(img)
    mask = arr < 200
    if mask.sum() == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    pad = 8
    y1 = max(0, y1 - pad)
    y2 = min(arr.shape[0], y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(arr.shape[1], x2 + pad)
    return img.crop((x1, y1, x2, y2))


def predict_digit(img):
    """数字识别，返回 (标签, 置信度)"""
    # MNIST 是 28x28，白底黑字（背景0，笔迹255？不对，MNIST 是黑底白字）
    # MNIST 数据：黑底(0)白字(255)
    # 我们画板：白底黑字
    # 需要反色
    img_inv = Image.eval(img, lambda x: 255 - x)
    # 但 MNIST 图需要白字黑底 → 反转后是黑字白底，还要再反一次
    # 实际上 MNIST 是：背景=0，笔迹=255
    # 我们画板：背景=255，笔迹=0
    # 所以要反转成背景=0，笔迹=255
    img_mnist = Image.eval(img, lambda x: 255 - x)
    t = transform_mnist(img_mnist).unsqueeze(0).to(device)
    with torch.no_grad():
        out = mnist_model(t)
        prob = torch.softmax(out, dim=1)[0]
        conf, idx = prob.max(0)
    return str(idx.item()), conf.item()


def predict_letter(img):
    """字母识别，返回 (标签, 置信度)"""
    # EMNIST 是黑底白字，要反色
    img_inv = Image.eval(img, lambda x: 255 - x)
    # ★ 去掉这一行：img_rot = img_inv.rotate(-90, expand=True)
    img_input = img_inv  # 直接用反色图，不旋转

    t = transform_mnist(img_input).unsqueeze(0).to(device)
    with torch.no_grad():
        out = letter_model(t)
        prob = torch.softmax(out, dim=1)[0]
        conf, idx = prob.max(0)
    letter = chr(ord('A') + idx.item())
    return letter, conf.item()


def predict_chinese(img):
    """汉字识别，返回 (标签, 置信度)"""
    # CASIA 是白底黑字，和画板一致
    t = transform_chinese(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = chinese_model(t)
        prob = torch.softmax(out, dim=1)[0]
        conf, idx = prob.max(0)
    return chinese_classes[idx.item()], conf.item()


# ============ 5. 界面 ============

class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写识别系统（数字/字母/汉字）")
        self.root.geometry("860x680")
        self.root.configure(bg='#f0f0f0')

        title = tk.Label(root, text="手写识别系统",
                         font=('Microsoft YaHei', 20, 'bold'),
                         bg='#f0f0f0', fg='#333')
        title.pack(pady=10)

        # 类型选择
        type_frame = tk.Frame(root, bg='#f0f0f0')
        type_frame.pack(pady=5)

        tk.Label(type_frame, text="识别类型：",
                 font=('Microsoft YaHei', 12), bg='#f0f0f0').pack(side='left')

        self.mode_var = tk.StringVar(value='auto')
        for text, val in [("自动", "auto"), ("数字", "digit"),
                          ("字母", "letter"), ("汉字", "chinese")]:
            tk.Radiobutton(type_frame, text=text, variable=self.mode_var,
                           value=val, font=('Microsoft YaHei', 11),
                           bg='#f0f0f0').pack(side='left', padx=8)

        # 主区域
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(pady=10)

        # 左：画板
        left = tk.Frame(main_frame, bg='#f0f0f0')
        left.pack(side='left', padx=20)

        tk.Label(left, text="请在下方书写",
                 font=('Microsoft YaHei', 12), bg='#f0f0f0').pack(pady=5)

        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg='white', cursor='cross',
                                highlightthickness=2, highlightbackground='#999')
        self.canvas.pack()

        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

        btn_frame = tk.Frame(left, bg='#f0f0f0')
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="识  别", font=('Microsoft YaHei', 14, 'bold'),
                  bg='#4CAF50', fg='white', width=10, height=2,
                  command=self.recognize).pack(side='left', padx=10)
        tk.Button(btn_frame, text="清  空", font=('Microsoft YaHei', 14),
                  bg='#FF9800', fg='white', width=10, height=2,
                  command=self.clear).pack(side='left', padx=10)

        # 右：结果
        right = tk.Frame(main_frame, bg='#f0f0f0')
        right.pack(side='left', padx=20)

        tk.Label(right, text="识别结果",
                 font=('Microsoft YaHei', 14, 'bold'), bg='#f0f0f0').pack(pady=5)

        self.result_char = tk.Label(right, text="?",
                                    font=('Microsoft YaHei', 100, 'bold'),
                                    bg='#f0f0f0', fg='#2196F3',
                                    width=2, height=1)
        self.result_char.pack(pady=10)

        self.result_type = tk.Label(right, text="",
                                    font=('Microsoft YaHei', 12),
                                    bg='#f0f0f0', fg='#555')
        self.result_type.pack()

        self.result_conf = tk.Label(right, text="",
                                    font=('Microsoft YaHei', 12),
                                    bg='#f0f0f0', fg='#555')
        self.result_conf.pack(pady=5)

        tk.Label(right, text="Top 5 候选：",
                 font=('Microsoft YaHei', 12, 'bold'),
                 bg='#f0f0f0').pack(pady=(20, 5), anchor='w')

        self.top5_frame = tk.Frame(right, bg='#f0f0f0')
        self.top5_frame.pack(anchor='w')
        self.top5_labels = []
        for i in range(5):
            row = tk.Frame(self.top5_frame, bg='#f0f0f0')
            row.pack(anchor='w', pady=2)
            tk.Label(row, text=f"{i+1}.", font=('Microsoft YaHei', 12),
                     bg='#f0f0f0', width=3, anchor='w').pack(side='left')
            char = tk.Label(row, text="-", font=('Microsoft YaHei', 16, 'bold'),
                            bg='#f0f0f0', fg='#333', width=3)
            char.pack(side='left')
            prob = tk.Label(row, text="", font=('Microsoft YaHei', 11),
                            bg='#f0f0f0', fg='#888')
            prob.pack(side='left', padx=5)
            self.top5_labels.append((char, prob))

    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_SIZE
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_char.config(text="?")
        self.result_type.config(text="")
        self.result_conf.config(text="")
        for char, prob in self.top5_labels:
            char.config(text="-")
            prob.config(text="")

    def recognize(self):
        cropped = crop_to_content(self.image)
        if cropped is None:
            self.result_char.config(text="?")
            self.result_conf.config(text="请先写字")
            return

        mode = self.mode_var.get()
        results = []

        if mode in ('auto', 'digit'):
            label, conf = predict_digit(cropped)
            results.append(('数字', label, conf))
        if mode in ('auto', 'letter'):
            label, conf = predict_letter(cropped)
            results.append(('字母', label, conf))
        if mode in ('auto', 'chinese'):
            label, conf = predict_chinese(cropped)
            results.append(('汉字', label, conf))

        # 选置信度最高的
        results.sort(key=lambda x: x[2], reverse=True)
        best_type, best_label, best_conf = results[0]

        self.result_char.config(text=best_label)
        self.result_type.config(text=f"类型：{best_type}")
        self.result_conf.config(text=f"置信度：{best_conf*100:.2f}%")

        # Top5 显示所有候选中置信度最高的 5 个
        all_candidates = sorted(results, key=lambda x: x[2], reverse=True)
        for i in range(5):
            if i < len(all_candidates):
                t, l, c = all_candidates[i]
                self.top5_labels[i][0].config(text=l)
                self.top5_labels[i][1].config(text=f"[{t}] {c*100:.2f}%")
            else:
                self.top5_labels[i][0].config(text="-")
                self.top5_labels[i][1].config(text="")


if __name__ == '__main__':
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()