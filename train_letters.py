import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============ 1. 加载 EMNIST letters 数据集 ============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 关键：EMNIST letters 标签是 1~26，减 1 变成 0~25
target_transform = transforms.Lambda(lambda y: y - 1)

train_dataset = torchvision.datasets.EMNIST(
    root='./data', split='letters', train=True, download=True,
    transform=transform, target_transform=target_transform)
test_dataset = torchvision.datasets.EMNIST(
    root='./data', split='letters', train=False, download=True,
    transform=transform, target_transform=target_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 验证标签范围
sample_labels = [train_dataset[i][1] for i in range(100)]
print(f"标签范围: {min(sample_labels)} ~ {max(sample_labels)}")

# ============ 2. 模型 ============
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

model = LetterCNN().to(device)
print(model)

# ============ 3. 训练 ============
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(avg_loss)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | "
          f"训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")

# ============ 4. 保存 ============
torch.save(model.state_dict(), "models/emnist_letters_cnn.pth")
print("模型已保存到 models/emnist_letters_cnn.pth")

# ============ 5. 画曲线 ============
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, epochs+1), train_losses, marker='o')
axes[0].set_title("训练损失曲线")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

axes[1].plot(range(1, epochs+1), test_accuracies, marker='o', color='green')
axes[1].set_title("测试准确率曲线")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("results/emnist_letters_training.png", dpi=150)
plt.show()