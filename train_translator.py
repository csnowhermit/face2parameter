from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import json
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

'''
    训练translator网络：只需用这一个脚本即可
'''

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Batch size during training
batch_size = 16
image_size = 512
num_epochs = 1000
lr = 0.01
ngpu = 2
root_path = "F:/dataset/face_2021_1130_20000_0.2-0.8/"

# root_path = "/root/dataset/face/"

class Translator_Dataset(Dataset):
    def __init__(self, params_root, feature_root, mode="train"):
        self.mode = mode
        with open(params_root, encoding='utf-8') as f:
            self.params = json.load(f)
        with open(feature_root, encoding='utf-8') as f:
            self.features = json.load(f)

    def __getitem__(self, index):
        if self.mode == "val":
            features = torch.tensor(self.features['%d.png' % (index + 18000)])
            param = torch.tensor(self.params['%d.png' % (index + 18000)])
        else:
            features = torch.tensor(self.features['%d.png' % index])
            param = torch.tensor(self.params['%d.png' % index])

        return features, param

    def __len__(self):
        if self.mode == "train":
            return 18000
        else:
            return 2000

'''
    Translator网络，从face-recognition到params
'''
class Translator(nn.Module):
    def __init__(self, isBias=False):
        super(Translator, self).__init__()
        self.fc1 = nn.Linear(in_features=256, out_features=512, bias=isBias)
        self.resatt1 = ResAttention(isBias=isBias)
        self.resatt2 = ResAttention(isBias=isBias)
        self.resatt3 = ResAttention(isBias=isBias)
        self.fc2 = nn.Linear(in_features=512, out_features=223, bias=isBias)
        self.bn = nn.BatchNorm1d(223)

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

        y_ = self.fc3(y)
        y_ = self.relu3(y_)

        y_ = self.fc4(y_)
        y_ = self.sigmoid4(y_)

        # 做点乘
        y_ = torch.mul(y, y_)
        return self.relu5(y_)

train_dataset = Translator_Dataset(root_path + "param.json", root_path + "lightcnn_feature.json", mode="train")
val_dataset = Translator_Dataset(root_path + "param.json", root_path + "lightcnn_feature.json", mode="val")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


translator = Translator(isBias=False)
if device.type == 'cuda':
    translator = nn.DataParallel(translator)
translator.to(device)

# Initialize BCELoss function
criterion = nn.L1Loss()

# optimizer = optim.SGD(translator.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adam(translator.parameters(), lr=1e-4)
# 每50个epoch衰减10%
scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) * 50, gamma=0.9)

total_step = len(train_dataloader)
translator.train()
train_loss_list = []
val_loss_list = []
for epoch in range(num_epochs):
    start = time.time()
    for i, (features, params) in enumerate(train_dataloader):
        optimizer.zero_grad()
        features = features.to(device)
        params = params.to(device)
        outputs = translator(features)
        loss = criterion(outputs, params)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i % 10) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, spend time: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), time.time() - start))
            start = time.time()

    train_loss_list.append(loss.item())

    translator.eval()
    with torch.no_grad():
        val_loss = 0
        for i, (features, params) in enumerate(val_dataloader):
            features = features.to(device)
            params = params.to(device)
            outputs = translator(features)
            loss = criterion(outputs, params)
            val_loss += loss.item()
        val_loss_list.append(val_loss / len(val_dataloader))

        print('Epoch [{}/{}], val_loss: {:.6f}'
              .format(epoch + 1, num_epochs, val_loss))
        if (epoch % 10) == 0 or (epoch+1) == num_epochs:
            torch.save(translator.state_dict(),
                       root_path + 'model/epoch_{}_{:.6f}.pt'.format(epoch, val_loss))
        if epoch >= 1:
            plt.figure()
            plt.subplot(121)
            plt.plot(np.arange(0, len(train_loss_list)), train_loss_list)
            plt.subplot(122)
            plt.plot(np.arange(0, len(val_loss_list)), val_loss_list)
            plt.savefig(root_path + "metrics.jpg")
            plt.close("all")

    translator.train()