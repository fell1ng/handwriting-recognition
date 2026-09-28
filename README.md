# 基于深度学习的手写体识别系统

支持手写数字、字母、汉字识别的桌面应用。

## 功能特性

- 支持手写数字（0-9）、字母（A-Z）、汉字（246类）识别
- 基于 PyTorch + ResNet18 + CNN 实现
- 图形界面支持手写输入、实时推理、Top-5 候选展示
- GPU 加速训练，14 万样本 15 分钟完成训练

## 技术栈

Python 3.10、PyTorch 2.x、CUDA 12.6、OpenCV、tkinter

## 数据集

| 数据集 | 任务 | 类别数 |
| :--- | :--- | :--- |
| MNIST | 数字 | 10 |
| EMNIST | 字母 | 26 |
| CASIA-HWDB | 汉字 | 246 |

## 模型性能

| 模型 | 测试准确率 |
| :--- | :--- |
| 数字 CNN | 99.06% |
| 字母 CNN | 94.5% |
| 汉字 ResNet18 | 95.2% |

## 使用方法

安装依赖：

pip install torch torchvision opencv-python numpy matplotlib

训练模型：

python train_mnist.py
python train_letters.py
python train_chinese.py

运行界面：

python gui_all.py

## 作者

陈德强 - 人工智能专业
