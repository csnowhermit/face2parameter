from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
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

import copy
import math

'''
    在服务器上训练只需上传这一个文件即可
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

image_root = "F:/dataset/face_20211203_20000_nojiemao/"

class Imitator_Dataset(Dataset):
    def __init__(self, params_root, image_root, mode="train"):
        self.image_root = image_root
        self.mode = mode
        with open(params_root, encoding='utf-8') as f:
            self.params = json.load(f)

    def __getitem__(self, index):
        if self.mode == "val":
            img = Image.open(os.path.join(self.image_root, '%d.png' % (index + 18000))).convert("RGB")
            param = torch.tensor(self.params['%d.png' % (index + 18000)])
        else:
            img = Image.open(os.path.join(self.image_root, '%d.png' % index)).convert("RGB")
            param = torch.tensor(self.params['%d.png' % index])
        img = T.ToTensor()(img)
        return param, img

    def __len__(self):
        if self.mode == "train":
            return 18000
        else:
            return 2000


train_dataset = Imitator_Dataset(image_root + "param.json", image_root + "face_train/", mode="train")
val_dataset = Imitator_Dataset(image_root + "param.json", image_root + "face_val/", mode="val")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# real_batch = next(iter(val_dataloader))
# plt.figure(figsize=(4, 4))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[1].to(device)[:16], nrow=4, padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.show()
# vutils.save_image(vutils.make_grid(real_batch[1].to(device)[:16], nrow=4, padding=2, normalize=True).cpu(), "./a.jpg")



'''
    自定义Imitator
    1.conv，linear，embedding后加上sn
    2.指定层加上self-attention
    3.自定义bn
'''

# 采用sn做 normalization
def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)

def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)

def sn_embedding(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Embedding(**kwargs), eps=eps)

# self-attention层
class SelfAttn(nn.Module):
    def __init__(self, in_channels, eps=1e-12):
        super(SelfAttn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8,
                                        kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8,
                                      kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2,
                                    kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_o_conv = snconv2d(in_channels=in_channels//2, out_channels=in_channels,
                                         kernel_size=1, bias=False, eps=eps)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g - o_conv
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_o_conv(attn_g)
        # Out
        out = x + self.gamma*attn_g
        return out

# 自定义bn
class BigGANBatchNorm(nn.Module):
    """ This is a batch norm module that can handle conditional input and can be provided with pre-computed
        activation means and variances for various truncation parameters.

        We cannot just rely on torch.batch_norm since it cannot handle
        batched weights (pytorch 1.0.1). We computate batch_norm our-self without updating running means and variances.
        If you want to train this model you should add running means and variance computation logic.
    """
    def __init__(self, num_features, condition_vector_dim=None, n_stats=51, eps=1e-4, conditional=True):
        super(BigGANBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional

        # We use pre-computed statistics for n_stats values of truncation between 0 and 1
        self.register_buffer('running_means', torch.zeros(n_stats, num_features))
        self.register_buffer('running_vars', torch.ones(n_stats, num_features))
        self.step_size = 1.0 / (n_stats - 1)

        if conditional:
            assert condition_vector_dim is not None
            self.scale = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
            self.offset = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))

    def forward(self, x, truncation, condition_vector=None):
        # Retreive pre-computed statistics associated to this truncation
        coef, start_idx = math.modf(truncation / self.step_size)
        start_idx = int(start_idx)
        if coef != 0.0:  # Interpolate
            running_mean = self.running_means[start_idx] * coef + self.running_means[start_idx + 1] * (1 - coef)
            running_var = self.running_vars[start_idx] * coef + self.running_vars[start_idx + 1] * (1 - coef)
        else:
            running_mean = self.running_means[start_idx]
            running_var = self.running_vars[start_idx]

        if self.conditional:
            running_mean = running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            running_var = running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            weight = 1 + self.scale(condition_vector).unsqueeze(-1).unsqueeze(-1)
            bias = self.offset(condition_vector).unsqueeze(-1).unsqueeze(-1)

            out = (x - running_mean) / torch.sqrt(running_var + self.eps) * weight + bias
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias,
                               training=False, momentum=0.0, eps=self.eps)
        return out

class GenBlock(nn.Module):
    def __init__(self, in_size, out_size, condition_vector_dim, reduction_factor=4, up_sample=False,
                 n_stats=51, eps=1e-12):
        super(GenBlock, self).__init__()
        self.up_sample = up_sample
        self.drop_channels = (in_size != out_size)
        middle_size = in_size // reduction_factor

        self.bn_0 = BigGANBatchNorm(in_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_0 = snconv2d(in_channels=in_size, out_channels=middle_size, kernel_size=1, eps=eps)

        self.bn_1 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_1 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)

        self.bn_2 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_2 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)

        self.bn_3 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_3 = snconv2d(in_channels=middle_size, out_channels=out_size, kernel_size=1, eps=eps)

        self.relu = nn.ReLU()

    def forward(self, x, cond_vector, truncation):
        x0 = x

        x = self.bn_0(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_0(x)

        x = self.bn_1(x, truncation, cond_vector)
        x = self.relu(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv_1(x)

        x = self.bn_2(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_2(x)

        x = self.bn_3(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_3(x)

        if self.drop_channels:
            new_channels = x0.shape[1] // 2
            x0 = x0[:, :new_channels, ...]
        if self.up_sample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest')

        out = x + x0
        return out

class MyImitator(nn.Module):
    def __init__(self):
        super(MyImitator, self).__init__()

        # 1.加载配置文件
        with open("./checkpoint/myimitator-512.json", "r", encoding='utf-8') as reader:
            text = reader.read()
        self.conf = BigGANConfig()
        for key, value in json.loads(text).items():
            self.conf.__dict__[key] = value

        # 定义网络结构
        # self.embeddings = nn.Linear(config.num_classes, config.continuous_params_size, bias=False)

        ch = self.conf.channel_width
        condition_vector_dim = 223

        self.gen_z = snlinear(in_features=condition_vector_dim, out_features=4*4*16*ch, eps=self.conf.eps)
        layers = []
        for i, layer in enumerate(self.conf.layers):
            if i == self.conf.attention_layer_position:    # 在指定层加上self-attention
                layers.append(SelfAttn(ch * layer[1], eps=self.conf.eps))
            layers.append(GenBlock(ch * layer[1],
                                   ch * layer[2],
                                   condition_vector_dim,
                                   up_sample=layer[0],
                                   n_stats=self.conf.n_stats,
                                   eps=self.conf.eps))
        self.layers = nn.ModuleList(layers)

        self.bn = BigGANBatchNorm(ch, n_stats=self.conf.n_stats, eps=self.conf.eps, conditional=False)
        self.relu = nn.ReLU()
        self.conv_to_rgb = snconv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, eps=self.conf.eps)
        self.tanh = nn.Tanh()

    def forward(self, cond_vector, truncation=0.4):
        # cond_vector = cond_vector.unsqueeze(2).unsqueeze(3)
        z = self.gen_z(cond_vector)    # cond_cector [batch_size, config.continuous_params_size], z [1, 4*4*16*self.conf.channel_width]

        # We use this conversion step to be able to use TF weights:
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        z = z.view(-1, 4, 4, 16 * self.conf.channel_width)    # [batch_size, 4, 4, 2048]
        z = z.permute(0, 3, 1, 2).contiguous()    # [batch_size, 2048, 4, 4]

        for i, layer in enumerate(self.layers):
            if isinstance(layer, GenBlock):
                z = layer(z, cond_vector, truncation)
            else:
                z = layer(z)

        z = self.bn(z, truncation)    # [1, 128, 512, 512]
        z = self.relu(z)    # [1, 128, 512, 512]
        z = self.conv_to_rgb(z)    # [1, 128, 512, 512]
        z = z[:, :3, ...]    # [1, 3, 512, 512]
        z = self.tanh(z)    # [1, 3, 512, 512]
        return z

'''
    自定义Imitator的config
'''
class BigGANConfig(object):
    """ Configuration class to store the configuration of a `BigGAN`.
        Defaults are for the 128x128 model.
        layers tuple are (up-sample in the layer ?, input channels, output channels)
    """
    def __init__(self,
                 output_dim=512,
                 z_dim=512,
                 class_embed_dim=512,
                 channel_width=512,
                 num_classes=1000,
                 # (是否上采样，input_channels，output_channels)
                 layers=[(False, 16, 16),
                         (True, 16, 16),
                         (False, 16, 16),
                         (True, 16, 8),
                         (False, 8, 8),
                         (True, 8, 4),
                         (False, 4, 4),
                         (True, 4, 2),
                         (False, 2, 2),
                         (True, 2, 1)],
                 attention_layer_position=8,
                 eps=1e-4,
                 n_stats=51):
        """Constructs BigGANConfig. """
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.class_embed_dim = class_embed_dim
        self.channel_width = channel_width
        self.num_classes = num_classes
        self.layers = layers
        self.attention_layer_position = attention_layer_position
        self.eps = eps
        self.n_stats = n_stats

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BigGANConfig` from a Python dictionary of parameters."""
        config = BigGANConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BigGANConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


imitator = MyImitator()
if device.type == 'cuda':
    imitator = nn.DataParallel(imitator)
imitator.to(device)

# Initialize BCELoss function
criterion = nn.L1Loss()

# optimizer = optim.SGD(imitator.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(params=imitator.parameters(), lr=5e-5,
                           betas=(0.0, 0.999), weight_decay=0,
                           eps=1e-8)

# 每50个epoch衰减10%
# scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) * 50, gamma=0.9)

total_step = len(train_dataloader)
imitator.train()
train_loss_list = []
val_loss_list = []
for epoch in range(num_epochs):
    start = time.time()
    for i, (params, img) in enumerate(train_dataloader):
        optimizer.zero_grad()
        params = params.to(device)
        img = img.to(device)
        outputs = imitator(params)
        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        train_loss_list.append(loss.item())

        if (i % 10) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, spend time: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), time.time() - start))
            start = time.time()

    imitator.eval()
    with torch.no_grad():
        val_loss = 0
        for i, (params, img) in enumerate(val_dataloader):
            params = params.to(device)
            img = img.to(device)
            outputs = imitator(params)
            loss = criterion(outputs, img)
            val_loss += loss.item()
            if i == 1:
                vutils.save_image(
                    vutils.make_grid(outputs.to(device)[:16], nrow=4, padding=2, normalize=True).cpu(),
                    image_root + "gen_image/%d.jpg" % epoch)
        val_loss_list.append(val_loss)

        print('Epoch [{}/{}], val_loss: {:.6f}'
              .format(epoch + 1, num_epochs, val_loss))
        if (epoch % 10) == 0 or (epoch+1) == num_epochs:
            torch.save(imitator.state_dict(),
                       image_root + 'model/epoch_{}_val_loss_{:.6f}_file.pt'.format(
                           epoch, val_loss))
        if epoch >= 1:
            plt.figure()
            plt.subplot(121)
            plt.plot(np.arange(0, len(train_loss_list)), train_loss_list)
            plt.subplot(122)
            plt.plot(np.arange(0, len(val_loss_list)), val_loss_list)
            plt.savefig(image_root + "metrics.jpg")
            plt.close("all")

    imitator.train()
