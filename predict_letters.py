import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 定义模型（结构必须和训练时一致）============
class LetterCNN(nn.Module):
    def __init__(self):
        super(LetterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LetterCNN().to(device)
model.load_state_dict(torch.load("models/emnist_letters_cnn.pth", map_location=device))
model.eval()
print("模型加载完成")

# ============ 2. 加载测试集（同样要做标签减1）============
target_transform = transforms.Lambda(lambda y: y - 1)
test_dataset = torchvision.datasets.EMNIST(
    root='./data', split='letters', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    target_transform=target_transform)

# 0~25 映射回 A~Z
idx_to_letter = [chr(ord('A') + i) for i in range(26)]

# ============ 3. 随机抽 8 张做识别 ============
import random
random.seed(42)

fig, axes = plt.subplots(2, 4, figsize=(12, 7))
axes = axes.flatten()

with torch.no_grad():
    for i in range(8):
        idx = random.randint(0, len(test_dataset) - 1)
        img, true_label = test_dataset[idx]
        img_input = img.unsqueeze(0).to(device)
        output = model(img_input)
        pred_label = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()

        # 显示时旋转回来（EMNIST 图像是旋转90度存的）
        img_show = img.squeeze().numpy()
        img_show = np.rot90(img_show, k=-1)  # 逆时针转回

        true_letter = idx_to_letter[true_label]
        pred_letter = idx_to_letter[pred_label]

        axes[i].imshow(img_show, cmap='gray')
        axes[i].set_title(
            f"真实: {true_letter} | 预测: {pred_letter}\n置信度: {confidence:.2%}",
            color='green' if pred_label == true_label else 'red')
        axes[i].axis('off')

plt.tight_layout()
plt.savefig("results/emnist_letters_predictions.png", dpi=150)
plt.show()
print("预测结果已保存到 results/emnist_letters_predictions.png")