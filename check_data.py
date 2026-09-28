import os
import matplotlib.pyplot as plt
from PIL import Image
import random

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 正确路径（多了 CASIA_246 一层）============
BASE_DIR = r"D:\code\基于深度学习的手写体识别系统设计"
TRAIN_DIR = os.path.join(BASE_DIR, "CASIA_246", "train")
TEST_DIR = os.path.join(BASE_DIR, "CASIA_246", "test")

print("TRAIN_DIR:", TRAIN_DIR, "->", os.path.exists(TRAIN_DIR))
print("TEST_DIR :", TEST_DIR, "->", os.path.exists(TEST_DIR))

# ============ 1. 列出类别 ============
classes = sorted(os.listdir(TRAIN_DIR))
print(f"\n训练集类别数: {len(classes)}")
print(f"前 20 个类别: {classes[:20]}")

# ============ 2. 单类样本数 ============
sample_class = classes[0]
sample_imgs = os.listdir(os.path.join(TRAIN_DIR, sample_class))
print(f"\n类别 '{sample_class}' 的图片数: {len(sample_imgs)}")

# ============ 3. 图片属性 ============
img_path = os.path.join(TRAIN_DIR, sample_class, sample_imgs[0])
img = Image.open(img_path)
print(f"\n图片路径: {img_path}")
print(f"图片模式: {img.mode}")
print(f"图片尺寸: {img.size}")

# ============ 4. 总数统计 ============
train_total = sum(len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in classes)
test_classes = os.listdir(TEST_DIR)
test_total = sum(len(os.listdir(os.path.join(TEST_DIR, c))) for c in test_classes)
print(f"\n训练集总图片数: {train_total}")
print(f"测试集总图片数: {test_total}")

# ============ 5. 抽 8 张图看看 ============
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()
for i in range(8):
    c = random.choice(classes)
    imgs = os.listdir(os.path.join(TRAIN_DIR, c))
    img = Image.open(os.path.join(TRAIN_DIR, c, random.choice(imgs)))
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"类别: {c}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()