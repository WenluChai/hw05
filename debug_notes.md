---

### 2. 📋 调试记录文件：debug_notes.md
**保存为**：`hw05/debug_notes.md`
```markdown
# HW05 调试记录 (Debug Notes)
本次实验主要遇到 1 个核心报错，已成功解决。

## 报错 1：批次大小不匹配 (Batch Size Mismatch)
### 现象
```text
ValueError: Expected input batch_size (100) to match target batch_size (64).
原因分析

LeNet-5 模型定义中的全连接层输入维度计算错误。

• 输入图像尺寸为 32×32。

• 经过两次卷积与池化后，最终特征图尺寸为 5×5（而非代码中误写的 4×4）。

• 维度 16 * 4 * 4 与实际 16 * 5 * 5 不符，导致展平后的张量形状与目标张量形状不一致，引发批次大小计算错误。

解决方案

将 lenet_model.py 中全连接层的输入维度从 16 * 4 * 4 修改为 16 * 5 * 5，并同步修正 view 函数参数。
# 修改前 (错误)
self.fc1 = nn.Linear(16 * 4 * 4, 120)
x = x.view(-1, 16 * 4 * 4)

# 修改后 (正确)
self.fc1 = nn.Linear(16 * 5 * 5, 120)
x = x.view(-1, 16 * 5 * 5)
