import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义超参数
input_size = 28  #图像的总尺寸28*28
num_classes = 10  #标签的种类数
num_epochs = 3  #训练的总循环周期
batch_size = 64  #一个撮（批次）的大小，64张图片

# 训练集
train_dataset = datasets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# 测试集
test_dataset = datasets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# 构建batch数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)




#卷积网络模块构建
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # 输入大小 (1, 28, 28) 第一个卷积模块conv1
            nn.Conv2d(
                in_channels=1,              # 灰度图 当前输入的特征图个数（颜色通道）
                out_channels=16,            # 要得到几多少个特征图 = 多少卷积核进行特征提取
                kernel_size=5,              # 卷积核大小
                stride=1,                   # 步长
                padding=2,                  # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
            ),                              # 输出的特征图为 (16, 28, 28)
            nn.ReLU(),                      # relu层
            nn.MaxPool2d(kernel_size=2),    # 进行池化操作（2x2 区域）, 输出结果为： (16, 14, 14)
        )

        self.conv2 = nn.Sequential(         # 下一个套餐的输入 (16, 14, 14) 第二个卷积模块
            nn.Conv2d(16, 32, 5, 1, 2),     # 输出 (32, 14, 14)
            nn.ReLU(),                      # relu层
            nn.MaxPool2d(2),                # 输出 (32, 7, 7)
        )

        self.out = nn.Linear(32 * 7 * 7, 10)   # 全连接层得到的结果


#前向传播层
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten操作，结果为：(batch_size, 32 * 7 * 7) 转成向量 一维
        output = self.out(x)                # 全连接层
        return output


#评估函数 计算当前准确率 预测正确的数量和总数
def accuracy(predictions, labels):
        pred = torch.max(predictions.data, 1)[1]   #torch.max(...,1)表示在第一维取最大值
        rights = pred.eq(labels.data.view_as(pred)).sum()   #逐元素比较pred和labels是否相等，相等返回True 不相等返回False 得到总正确预测的数量rights
        return rights, len(labels)




# 实例化一个网络对象net
net = CNN()
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器 用Adam算法来更新net网络的参数，学习率位0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器，普通的随机梯度下降算法

# 开始训练循环
for epoch in range(num_epochs): #外层循环，遍历训练的轮数（epoch）
    # 当前epoch的结果保存下来
    # 定义一个空列表 收集本次epoch内所有batch的准确率信息，之后把每个batch的rights和样本数存进去
    train_rights = []

    for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
        net.train()                                            #将网络设置为训练模式
        output = net(data)                                     #将当前批次的数据data为给网络net，得到网络输出output
        loss = criterion(output, target)                       #计算这个批次上的损失值
        optimizer.zero_grad()                                  #清空梯度缓存 每一次backward之前先将梯度清零
        loss.backward()                                        #反向传播 计算梯度
        optimizer.step()                                       #用前一步计算得到的梯度 更新网络的参数 Adam优化器优化
        right = accuracy(output, target)                       #获取这个批次中预测正确的数量和总数
        train_rights.append(right)                             #把这个批次的预测正确的数保存到列表

        if batch_idx % 100 == 0:

            net.eval()                                         #将网络设置为验证模式
            val_rights = []                                    #新建列表，用来收集测试集上预测正确数

            for (data, target) in test_loader:                 #遍历测试集
                output = net(data)                             #把测试集输入到训练好的net 得到预测输出
                right = accuracy(output, target)               #计算正确数和总数
                val_rights.append(right)                       #把这个批次的预测正确的数保存到列表

            # 准确率计算    #把 train_rights 里面的所有 (rights, total) 元组的 rights 累加起来
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights])) #训练过程正确和总数
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))       #测试过程正确和总数

            print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data,
                       100. * train_r[0].numpy() / train_r[1],
                       100. * val_r[0].numpy() / val_r[1]))