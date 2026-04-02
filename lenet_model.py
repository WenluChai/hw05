import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层：输入1通道，输出6通道，5×5卷积核，步长1，无填充
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 卷积层：输入6通道，输出16通道，5×5卷积核，步长1，无填充
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 池化层：2×2最大池化，步长2
        self.pool = nn.MaxPool2d(2, 2)
        # 修正全连接层维度：16 * 5 * 5（32×32输入经过两次卷积+池化后尺寸为5×5）
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 第一次卷积+ReLU+池化：32×32 → 28×28 → 14×14
        x = self.pool(F.relu(self.conv1(x)))
        # 第二次卷积+ReLU+池化：14×14 → 10×10 → 5×5
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图：(batch_size, 16, 5, 5) → (batch_size, 16*5*5=400)
        x = x.view(-1, 16 * 5 * 5)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x