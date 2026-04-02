hw05/
├── data/                # 数据集目录 (自动下载)
├── model.py             # 任务一：极简 CNN 模型定义
├── train.py             # 任务一：训练脚本
├── lenet_model.py       # 任务二：LeNet-5 模型定义
├── lenet_train.py       # 任务二：训练脚本
├── report.md            # 实验报告
├── debug_notes.md       # 调试记录
├── requirements.txt     # 环境依赖
└── README.md            # 说明文档
## 2. 环境依赖安装
### 2.1 安装方式
确保已安装 Python 3.8+，执行以下命令安装依赖：
```bash
pip install -r requirements.txt
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
tqdm>=4.65.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
tqdm>=4.65.0
训练极简 CNN 模型：python train.py
训练经典 LeNet-5 模型：python lenet_train.py
4. 实验目标

• 实现并训练一个极简 CNN 模型，达到 ~98.5% 测试准确率。

• 实现并训练 LeNet-5 模型，达到 ~99.0% 测试准确率。

• 对比两种模型结构与性能。
---

### 4. ⚙️ 依赖文件：requirements.txt
**保存为**：`hw05/requirements.txt`
```text
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
tqdm>=4.65.0
