import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

BASE_DIR = r"D:\code\基于深度学习的手写体识别系统设计"
TRAIN_DIR = os.path.join(BASE_DIR, "CASIA_246", "train")
TEST_DIR = os.path.join(BASE_DIR, "CASIA_246", "test")
OUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)

# 一次性预处理：转灰度 + Resize 64x64
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
])

def convert_split(split_dir, out_path):
    classes = sorted(os.listdir(split_dir))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    images_list = []
    labels_list = []

    for c in tqdm(classes, desc=f"处理 {os.path.basename(split_dir)}"):
        class_dir = os.path.join(split_dir, c)
        for f in os.listdir(class_dir):
            try:
                img = Image.open(os.path.join(class_dir, f))
                img = transform(img)
                arr = np.array(img, dtype=np.uint8)   # [64, 64]
                images_list.append(arr)
                labels_list.append(class_to_idx[c])
            except Exception as e:
                print(f"跳过 {f}: {e}")

    images = torch.from_numpy(np.stack(images_list))   # [N, 64, 64]
    labels = torch.tensor(labels_list)
    torch.save({
        'images': images,
        'labels': labels,
        'classes': classes,
    }, out_path)
    print(f"已保存 {out_path}，共 {len(images)} 张")

if __name__ == '__main__':
    convert_split(TRAIN_DIR, os.path.join(OUT_DIR, "chinese_train.pt"))
    convert_split(TEST_DIR, os.path.join(OUT_DIR, "chinese_test.pt"))
    print("预处理完成")