# 基于深度学习的手写体识别系统

支持手写数字、字母、汉字识别的桌面应用，涵盖数据集构建、模型训练、推理部署、图形界面开发全流程。

## 功能特性

- 支持手写数字（0-9）、字母（A-Z）、汉字（246类）识别
- 基于 PyTorch + ResNet18 + CNN 实现
- 图形界面支持手写输入、实时推理、Top-5 候选展示
- GPU 加速训练，14 万样本 15 分钟完成训练

## 技术栈

- Python 3.10
- PyTorch 2.x + Torchvision
- CUDA 12.6
- OpenCV、PIL、NumPy
- tkinter

## 数据集

| 数据集 | 任务 | 类别数 | 训练样本 |
| :--- | :--- | :--- | :--- |
| MNIST | 数字识别 | 10 | 60000 |
| EMNIST Letters | 字母识别 | 26 | 124800 |
| CASIA-HWDB | 汉字识别 | 246 | 147265 |

## 模型性能

| 模型 | 测试准确率 |
| :--- | :--- |
| 数字 CNN | 99.06% |
| 字母 CNN | 94.5% |
| 汉字 ResNet18 | 95.2% |

## 使用方法

### 环境安装

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python numpy matplotlib pillow


## 训练模型
python train_mnist.py      # 数字
python train_letters.py    # 字母
python train_chinese.py    # 汉字

## 运行界面
python gui_all.py


