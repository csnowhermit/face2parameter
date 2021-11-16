import os
import cv2
import torch
import numpy as np
import torch.nn as nn

import utils


'''
    模拟器：使用神经网络代替游戏引擎
'''

class Imitator(nn.Module):
    def __init__(self, name, args):
        super(Imitator, self).__init__()
        self.name = name
        self.args = args
        self.init_steps = 0

        self.model = nn.Sequential(
            utils.deconv_layer(args.params_cnt, 512, kernel_size=4),  # 1. (batch, 512, 4, 4)
            utils.deconv_layer(512, 512, kernel_size=4, stride=2, pad=1),  # 2. (batch, 512, 8, 8)
            utils.deconv_layer(512, 512, kernel_size=4, stride=2, pad=1),  # 3. (batch, 512, 16, 16)
            utils.deconv_layer(512, 256, kernel_size=4, stride=2, pad=1),  # 4. (batch, 256, 32, 32)
            utils.deconv_layer(256, 128, kernel_size=4, stride=2, pad=1),  # 5. (batch, 128, 64, 64)
            utils.deconv_layer(128, 64, kernel_size=4, stride=2, pad=1),  # 6. (batch, 64, 128, 128)
            utils.deconv_layer(64, 64, kernel_size=4, stride=2, pad=1),  # 7. (batch, 64, 256, 256)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 8. (batch, 3, 512, 512)
            nn.Sigmoid(),
        )

    '''
        :param params [batch, params_cnt]
        :return [batch, 3, 512, 512]
    '''
    def forward(self, params):
        batch = params.size(0)    # 1
        length = params.size(1)    # 95
        _params = params.reshape((batch, length, 1, 1))    # [1, 95, 1, 1]
        return self.model(_params)

