import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

'''
    LSGAN：最小二乘生成对抗损失
    GAN存在的通用问题：生成图片质量不高；训练过程不稳定
    传统做法：传统gan里用的BCELoss，见tradition_gan.py
        1.采用交叉熵损失（只看True/False）。使得生成器不再优化被判别器识别为True的fake image，即使生成的图片离判别器的决策边界很远，即离真实数据很远。
          这意味着生成器生成的图片质量并不高。为什么生成器不再优化生成图片？因为已经完成了目标——即骗过判别器，所以这时交叉熵已经很小了。
          而最小二乘法要求，骗过判别器的前提下还得让生成器把离决策边界比较远的图片拉向决策边界。
        2.sigmoid函数，输入过大或过小时，都会造成梯度消失；而最小二乘只有x=1时梯度为0。
    传统GAN与LSGAN做法类似，只需将损失函数换成torch.nn.BCELoss()即可，如下：
        adversarial_loss = torch.nn.BCELoss()，若报错RuntimeError: all elements of input should be between 0 and 1，可采用如下：
        adversarial_loss = torch.nn.BCEWithLogitsLoss()
'''

# 配置项
epochs=200
batch_size=64
learning_rate=0.0002
latent_dim=100    # 从100维向量开始生成图片
image_size=32
sample_interval=1000    # 每隔1000个batch保存一下
save_sample_path = "../output/images/"    # 样例图片保存路径
if os.path.exists("../output/images/") is False:
    os.makedirs("../output/images/")

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),    # 最后生成1维的图像
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = image_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# 这里使用MSE损失，符合LSGAN最小二乘损失的思想
# MSE：均方误差损失函数；BCE：二分类下的交叉熵损失（传统gan里用的BCELoss）
adversarial_loss = torch.nn.MSELoss()
# adversarial_loss = torch.nn.BCEWithLogitsLoss()    # 若报错RuntimeError: all elements of input should be between 0 and 1，可采用torch.nn.BCEWithLogitsLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../output/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../output/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 训练过程
g_loss_list = []
d_loss_list = []

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):    # images [batch_size, 1, image_size, image_size]

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)    # [batch_size, 1]
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)    # [batch_size, 0]

        real_imgs = Variable(imgs.type(Tensor))    # 真实图片

        # 训练生成器
        optimizer_G.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))    # [batch_size, latent_dim]
        gen_imgs = generator(z)    # [batch_size, 1, image_size, image_size]

        # 生成器优化方向是D(x)越来越靠近1
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)    # 判别器判断的结果与真做MSE

        g_loss.backward()
        optimizer_G.step()

        # 训练判别器
        optimizer_D.zero_grad()

        # LSGAN，要求不能简简单单算交叉熵损失，而是计算两个分布之间的均方误差损失
        real_loss = adversarial_loss(discriminator(real_imgs), valid)    # 要求把真的识别为真
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)    # 把假的识别为假
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        if i % sample_interval == 0:
            save_image(gen_imgs.data[:25], "%s/%d_%d.png" % (save_sample_path, epoch, i), nrow=5, normalize=True)
    g_loss_list.append(g_loss.item())
    d_loss_list.append(d_loss.item())

    if epoch >= 0:
        plt.figure()
        plt.subplot(121)
        plt.plot(np.arange(0, len(g_loss_list)), g_loss_list)
        plt.subplot(122)
        plt.plot(np.arange(0, len(d_loss_list)), d_loss_list)
        plt.savefig("loss.jpg")
        plt.close("all")