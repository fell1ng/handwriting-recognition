import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 模拟：在 320x320 画板上写一个 "A"
img = Image.new('L', (320, 320), 255)
draw = ImageDraw.Draw(img)
# 画一个 A
draw.line([(160, 60), (100, 260)], fill=0, width=8)   # 左斜
draw.line([(160, 60), (220, 260)], fill=0, width=8)   # 右斜
draw.line([(120, 180), (200, 180)], fill=0, width=8)  # 横

# ---- 方案1：不裁剪，直接 Resize ----
img1 = img.resize((28, 28))
arr1 = np.array(img1)

# ---- 方案2：裁剪 + 反转 + 旋转 ----
arr = np.array(img)
mask = arr < 200
rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
y1, y2 = np.where(rows)[0][[0, -1]]; x1, x2 = np.where(cols)[0][[0, -1]]
pad = 8
y1 = max(0, y1-pad); y2 = min(arr.shape[0], y2+pad)
x1 = max(0, x1-pad); x2 = min(arr.shape[1], x2+pad)
cropped = img.crop((x1, y1, x2, y2))

img2 = Image.eval(cropped, lambda x: 255 - x)     # 反色
img2 = img2.resize((28, 28))
arr2 = np.array(img2)

# ---- 方案3：裁剪 + 反转 + 旋转 ----
img3 = Image.eval(cropped, lambda x: 255 - x)
img3 = img3.rotate(-90, expand=True)
img3 = img3.resize((28, 28))
arr3 = np.array(img3)

# 显示 4 张
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(np.array(img), cmap='gray'); axes[0].set_title("原始画板")
axes[1].imshow(arr1, cmap='gray'); axes[1].set_title("方案1：直接Resize")
axes[2].imshow(arr2, cmap='gray'); axes[2].set_title("方案2：裁剪+反色")
axes[3].imshow(arr3, cmap='gray'); axes[3].set_title("方案3：裁剪+反色+旋转")
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.show()