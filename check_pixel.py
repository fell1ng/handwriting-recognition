import os
from PIL import Image
import numpy as np

BASE_DIR = r"D:\code\基于深度学习的手写体识别系统设计"
TRAIN_DIR = os.path.join(BASE_DIR, "CASIA_246", "train")

# 看几个字的图
for char in ['一', '二', '三']:
    char_dir = os.path.join(TRAIN_DIR, char)
    f = os.listdir(char_dir)[0]
    img = Image.open(os.path.join(char_dir, f))

    # 转成灰度看看
    gray = img.convert('L')
    arr = np.array(gray)

    print(f"字: {char}")
    print(f"  原图模式: {img.mode}")
    print(f"  灰度后形状: {arr.shape}")
    print(f"  最小值: {arr.min()}, 最大值: {arr.max()}, 均值: {arr.mean():.1f}")

    # 看边缘（应该是背景）和中心（应该是笔迹）
    h, w = arr.shape
    print(f"  左上角像素(背景): {arr[0,0]}")
    print(f"  中心区域均值: {arr[h//3:2*h//3, w//3:2*w//3].mean():.1f}")
    print()