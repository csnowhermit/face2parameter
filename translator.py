import torch
import torch.nn as nn

import config

'''
    Translator网络，从face-recognition到params
    网络结构参考第二篇论文：《Fast and Robust Face-to-Parameter Translation for Game Character Auto-Creation》
    keypoint：
    1.use the Adam optimizer to train T with the learning rate of 1e-4 and max-iteration of 20 epochs.
    2.the learning rate decay is set to 10% per 50 epochs.（参照imitator训练策略，但一般用不上，epoch=40时就可以停止训练了）
'''
class Translator(nn.Module):
    def __init__(self, isBias=False):
        super(Translator, self).__init__()
        self.fc1 = nn.Linear(in_features=256, out_features=512, bias=isBias)
        self.resatt1 = ResAttention(isBias=isBias)
        self.resatt2 = ResAttention(isBias=isBias)
        self.resatt3 = ResAttention(isBias=isBias)
        self.fc2 = nn.Linear(in_features=512, out_features=config.continuous_params_size, bias=isBias)
        self.bn = nn.BatchNorm1d(config.continuous_params_size)

    def forward(self, x):
        y = self.fc1(x)
        y_ = y + self.resatt1(y)
        y_ = y_ + self.resatt2(y_)
        y_ = y_ + self.resatt3(y_)
        return self.bn(self.fc2(y_))


'''
    带attention的res模块
'''
class ResAttention(nn.Module):
    def __init__(self, isBias=False):
        super(ResAttention, self).__init__()
        self.fc1 = nn.Linear(512, 1024, bias=isBias)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(1024, 512, bias=isBias)
        self.bn2 = nn.BatchNorm1d(512)

        # 这里开始分支
        self.fc3 = nn.Linear(512, 16, bias=isBias)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(16, 512, bias=isBias)
        self.sigmoid4 = nn.Sigmoid()

        # 这里开始做点乘

        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.fc1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.fc2(y)
        y = self.bn2(y)

        # 这里开始分支

        y_ = self.fc3(y)
        y_ = self.relu3(y_)

        y_ = self.fc4(y_)
        y_ = self.sigmoid4(y_)

        # 做点乘
        y_ = torch.mul(y, y_)
        return self.relu5(y_)

if __name__ == '__main__':
    trans = Translator()
    x = torch.randn([2, 256], dtype=torch.float32)    # face-recognition的结果
    print(x.shape)
    y = trans(x)
    print(y.shape)