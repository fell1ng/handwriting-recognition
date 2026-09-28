import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

BASE_DIR = r"D:\code\基于深度学习的手写体识别系统设计"

# ============ 1. 加载预处理好的数据 ============
class PTDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.images = data['images']   # [N, 64, 64] uint8
        self.labels = data['labels']
        self.classes = data['classes']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 归一化到 [-1, 1]
        img = self.images[idx].float() / 255.0
        img = (img - 0.5) / 0.5
        return img.unsqueeze(0), self.labels[idx]

train_dataset = PTDataset(os.path.join(BASE_DIR, "models", "chinese_train.pt"))
test_dataset = PTDataset(os.path.join(BASE_DIR, "models", "chinese_test.pt"))
num_classes = len(train_dataset.classes)
print(f"类别数: {num_classes}, 训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")

# ============ 2. DataLoader ============
BATCH_SIZE = 256
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============ 3. 模型 ============
def create_model(num_classes):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, num_classes)
    return model

model = create_model(num_classes).to(device)

# ============ 4. 训练 ============
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

epochs = 15   # 改成 15 轮
train_losses, test_accuracies = [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(avg_loss)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | "
          f"训练: {train_acc:.2f}% | 测试: {test_acc:.2f}%")

    scheduler.step()

torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "chinese_resnet18.pth"))
print("模型已保存")

# ============ 5. 画曲线 ============
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, epochs+1), train_losses, marker='o')
axes[0].set_title("训练损失"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True)
axes[1].plot(range(1, epochs+1), test_accuracies, marker='o', color='green')
axes[1].set_title("测试准确率"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)"); axes[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "results", "chinese_training.png"), dpi=150)
plt.show()