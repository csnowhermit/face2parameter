import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import config
from imitator import Imitator
from dataset import Translator_Dataset, split_dataset
from lightcnn import LightCNN_29Layers_v2
from face_parser import load_model
from translator import Translator

# root_path = "./data/"
root_path = "F:/dataset/CelebAMask-HQ/align/"    # 训练前要先做对齐
r1 = 0.01
r2 = 1
r3 = 1

def criterion_lightcnn(x1, x2):
    distance = torch.cosine_similarity(x1, x2)[0]
    return 1 - distance

'''
    完整版训练translator
'''

if __name__ == '__main__':
    # 1.初始化并加载imitator
    imitator = Imitator()
    if len(config.imitator_model) > 0:
        if config.use_gpu:
            imitator_model = torch.load(config.imitator_model)
        else:
            imitator_model = torch.load(config.imitator_model, map_location=torch.device('cpu'))
        print("load pretrained model success!")
    else:
        print("No pretrained model...")

    imitator = imitator.to(config.device)
    for param in imitator.parameters():
        param.requires_grad = False

    # 2.加载lightcnn
    lightcnn = LightCNN_29Layers_v2(num_classes=80013)
    lightcnn = lightcnn.to(config.device)
    lightcnn.eval()
    if config.use_gpu:
        checkpoint = torch.load(config.lightcnn_checkpoint)
        model = torch.nn.DataParallel(lightcnn).cuda()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load(config.lightcnn_checkpoint, map_location="cpu")
        new_state_dict = lightcnn.state_dict()
        for k, v in checkpoint['state_dict'].items():
            _name = k[7:]  # remove `module.`
            new_state_dict[_name] = v
        lightcnn.load_state_dict(new_state_dict)

    # 冻结lightcnn
    for param in lightcnn.parameters():
        param.requires_grad = False

    # 3.T网络
    translator = Translator(isBias=False)
    if config.device.type == 'cuda':
        translator = nn.DataParallel(translator)
    translator.to(config.device)

    # 4.加载face_parser
    deeplab = load_model('mobilenetv2', num_classes=config.num_classes, output_stride=config.output_stride)
    checkpoint = torch.load(config.faceparse_checkpoint, map_location=config.device)
    if config.faceparse_backbone == 'resnet50':
        deeplab.load_state_dict(checkpoint)
    else:
        deeplab.load_state_dict(checkpoint["model_state"])
    deeplab.to(config.device)
    deeplab.eval()

    for param in deeplab.parameters():
        param.requires_grad = False

    trainlist, vallist = split_dataset(root_path)

    # 损失
    criterion_param = nn.L1Loss()
    criterion_parser = nn.L1Loss()
    # criterion_lightcnn = torch.cosine_similarity

    train_dataset = Translator_Dataset(trainlist)
    val_dataset = Translator_Dataset(vallist)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(translator.parameters(), lr=1e-4)

    total_step = len(train_dataloader)
    translator.train()
    train_loss_list = []
    val_loss_list = []
    for epoch in range(config.total_epochs):
        start = time.time()
        for i, imgs in enumerate(train_dataloader):
            optimizer.zero_grad()
            # features = features.to(config.device)
            # params = params.to(config.device)
            imgs = imgs.to(config.device)

            # 先做语义分割
            parse = deeplab(imgs)

            imgs = T.Grayscale()(F.interpolate(imgs, (128, 128), mode='bilinear'))
            _, features = lightcnn(imgs)
            outputs = translator(features)    # 223

            gen_img = imitator(outputs)    # 生成图像
            gen_parse = deeplab(gen_img)
            gen_img = T.Grayscale()(F.interpolate(gen_img, (128, 128), mode='bilinear'))

            _, gen_features = lightcnn(gen_img)
            gen_outputs = translator(gen_features)

            loss1 = criterion_lightcnn(gen_features, features)
            loss2 = criterion_parser(gen_parse, parse)
            loss3 = criterion_param(outputs, gen_outputs)

            loss = r1 * loss1 + r2 * loss2 + r3 * loss3

            loss.backward()
            optimizer.step()

            if (i % 10) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, spend time: {:.4f}'
                      .format(epoch + 1, config.total_epochs, i + 1, total_step, loss.item(), time.time() - start))
                start = time.time()

        train_loss_list.append(loss.item())

        translator.eval()
        with torch.no_grad():
            val_loss = 0
            for i, imgs in enumerate(val_dataloader):
                # features = features.to(config.device)
                # params = params.to(config.device)

                imgs = imgs.to(config.device)

                imgs = T.Grayscale()(F.interpolate(imgs, (128, 128), mode='bilinear'))
                _, features = lightcnn(imgs)
                outputs = translator(features)  # 223

                gen_img = imitator(outputs)  # 生成图像
                gen_img = T.Grayscale()(F.interpolate(gen_img, (128, 128), mode='bilinear'))

                _, gen_features = lightcnn(gen_img)
                gen_outputs = translator(gen_features)

                loss1 = criterion_lightcnn(gen_features, features)
                loss3 = criterion_param(outputs, gen_outputs)

                loss = r1 * loss1 + r3 * loss3

                val_loss += loss.item()
            val_loss_list.append(val_loss / len(val_dataloader))

            print('Epoch [{}/{}], val_loss: {:.6f}'
                  .format(epoch + 1, config.total_epochs, val_loss))
            if (epoch % 1) == 0 or (epoch + 1) == config.total_epochs:
                torch.save(translator.state_dict(), './checkpoint/translator_{}_{:.6f}.pt'.format(epoch, val_loss))
            if epoch >= 1:
                plt.figure()
                plt.subplot(121)
                plt.plot(np.arange(0, len(train_loss_list)), train_loss_list)
                plt.subplot(122)
                plt.plot(np.arange(0, len(val_loss_list)), val_loss_list)
                plt.savefig(root_path + "metrics.jpg")
                plt.close("all")

        translator.train()
