import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 定义和训练时一样的模型结构 ============
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
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

# ============ 2. 加载模型 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
model.eval()
print("模型加载完成")

# ============ 3. 从测试集里随机抽 8 张图做识别 ============
import torchvision
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

with torch.no_grad():
    for i in range(8):
        img, true_label = test_dataset[i]
        img_input = img.unsqueeze(0).to(device)  # 加 batch 维度
        output = model(img_input)
        pred_label = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()

        # 显示
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f"真实: {true_label} | 预测: {pred_label}\n置信度: {confidence:.2%}",
                          color='green' if pred_label == true_label else 'red')
        axes[i].axis('off')

plt.tight_layout()
plt.savefig("results/mnist_predictions.png", dpi=150)
plt.show()
print("预测结果已保存到 results/mnist_predictions.png")
