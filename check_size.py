import os
from PIL import Image
from collections import Counter

BASE_DIR = r"D:\code\基于深度学习的手写体识别系统设计"
TRAIN_DIR = os.path.join(BASE_DIR, "CASIA_246", "train")

classes = sorted(os.listdir(TRAIN_DIR))
sizes = []
modes = []

for c in classes[:30]:   # 抽前 30 类看看
    class_dir = os.path.join(TRAIN_DIR, c)
    for f in os.listdir(class_dir)[:5]:  # 每类抽 5 张
        img = Image.open(os.path.join(class_dir, f))
        sizes.append(img.size)
        modes.append(img.mode)

print("图片模式统计:", Counter(modes))
print("\n图片尺寸分布（前 30 类抽样）:")
for size, count in Counter(sizes).most_common(10):
    print(f"  {size}: {count} 张")

print(f"\n最大尺寸: {max(sizes)}")
print(f"最小尺寸: {min(sizes)}")
