import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

BASE_DIR = r"D:\code\基于深度学习的手写体识别系统设计"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 1. 看看 MNIST 和 EMNIST 的真图 ============
print("=" * 50)
print("MNIST 真图（数字）")
print("=" * 50)

mnist_test = torchvision.datasets.MNIST(
    root='./data', train=False, download=False, transform=None)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i in range(5):
    img, label = mnist_test[i]
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f"MNIST 数字: {label}\n模式: {img.mode}, 尺寸: {img.size}")
    axes[0, i].axis('off')

# 找到数字 9 的前 5 张
count = 0
for i in range(len(mnist_test)):
    img, label = mnist_test[i]
    if label == 9 and count < 5:
        axes[1, count].imshow(img, cmap='gray')
        axes[1, count].set_title(f"数字 9（第{count+1}张）")
        axes[1, count].axis('off')
        count += 1
    if count >= 5:
        break

plt.tight_layout()
plt.savefig("results/mnist_check.png", dpi=120)
plt.show()

print("\n" + "=" * 50)
print("EMNIST 真图（字母）")
print("=" * 50)

emnist_test = torchvision.datasets.EMNIST(
    root='./data', split='letters', train=False, download=False, transform=None)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i in range(10):
    img, label = emnist_test[i]
    true_letter = chr(ord('A') + label - 1)
    ax = axes[i // 5, i % 5]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"EMNIST 字母: {true_letter}\n标签值: {label}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("results/emnist_check.png", dpi=120)
plt.show()

# ============ 2. 看看 EMNIST 的旋转 ============
print("\n" + "=" * 50)
print("EMNIST 旋转对比")
print("=" * 50)

img, label = emnist_test[0]
true_letter = chr(ord('A') + label - 1)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(img, cmap='gray'); axes[0].set_title("原图")
axes[1].imshow(img.rotate(-90, expand=True), cmap='gray'); axes[1].set_title("旋转 -90°")
axes[2].imshow(img.rotate(90, expand=True), cmap='gray'); axes[2].set_title("旋转 +90°")
axes[3].imshow(img.rotate(180, expand=True), cmap='gray'); axes[3].set_title("旋转 180°")
for ax in axes: ax.axis('off')
plt.suptitle(f"标签：{true_letter}（EMNIST 字母标签值 {label}）")
plt.tight_layout()
plt.savefig("results/emnist_rotation.png", dpi=120)
plt.show()

print(f"\n第一张 EMNIST 图：标签值 = {label}，对应字母 = {true_letter}")
print("看上面的图，哪种旋转方向让字母看起来是正的，就说明需要那种旋转。")