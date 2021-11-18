import os
import cv2
import torch
import numpy as np
import torch.nn as nn

import utils
import config


'''
    模拟器：使用神经网络代替游戏引擎
'''

class Imitator(nn.Module):
    def __init__(self, is_bias=True):
        super(Imitator, self).__init__()

        self.model = nn.Sequential(
            # 1. (batch, 512, 4, 4)
            nn.ConvTranspose2d(config.continuous_params_size, 512, kernel_size=4, bias=is_bias),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 2. (batch, 512, 8, 8)
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=is_bias),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 3. (batch, 512, 16, 16)
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=is_bias),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 4. (batch, 256, 32, 32)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=is_bias),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 5. (batch, 128, 64, 64)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=is_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 6. (batch, 64, 128, 128)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=is_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 7. (batch, 64, 256, 256)
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=is_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 8. (batch, 3, 512, 512)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=is_bias),
            # nn.Tanh(),     # DCGAN和论文中用的tanh
            nn.Sigmoid()    # 开源版本用的sigmoid
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    '''
        :param params [batch, params_cnt]
        :return [batch, 3, 512, 512]
    '''
    def forward(self, params):
        # batch = params.size(0)    # 1
        # length = params.size(1)    # 95
        # _params = params.reshape((batch, length, 1, 1))    # [1, 95, 1, 1]

        # 把连续参数的size从[batch, continuous_params_size]扩展成[batch, continuous_params_size, 1, 1]
        _params = params.unsqueeze(2).unsqueeze(3)
        return self.model(_params)

