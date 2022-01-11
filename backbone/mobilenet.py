import torch.nn as nn
import torch.nn.functional as F


'''
    backbone: mobilenetV2
'''
class MobileNetV2(nn.Module):
    '''
        :param num_classes 类别个数
        :param output_stride
        :param width_mult 通过该参数控制每一层的通道数量
        :param inverted_residual_setting
        :param round_neatest 将每层的通道数四舍五入为该数字的倍数，设为1则关闭四舍五入
    '''
    def __init__(self, num_classes=1000, output_stride=8, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.output_stride = output_stride
        current_stride = 1
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # 构建第一层
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        current_stride *= 2
        dilation = 1
        previous_dilation = 1

        # 构建中间的倒置残差块
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            previous_dilation = dilation
            if current_stride == output_stride:
                stride = 1
                dilation *= s
            else:
                stride = s
                current_stride *= s
            output_channel = int(c * width_mult)

            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, stride, previous_dilation, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, dilation, expand_ratio=t))
                input_channel = output_channel

        # 构建最后一层
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # 构建最后的分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

"""
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
"""
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:    # 确保四舍五入的下降幅度不超过10%
        new_v += divisor
    return new_v

'''
    CBR结构：Conv+BN+Relu
'''
class ConvBNReLU(nn.Sequential):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, dilation=1, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU6(inplace=True)
        )

'''
    倒置残差模块
'''
class InvertedResidual(nn.Module):
    def __init__(self, input_channel, output_channel, stride, dilation, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = int(round(input_channel * expand_ratio))
        self.use_residual_connect = self.stride == 1 and input_channel == output_channel    # 判断是否采用残差连接
        layers = []

        if expand_ratio != 1:
            layers.append(ConvBNReLU(input_channel, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),     # 采用深度可分离卷积，各通道各卷积各的
            nn.Conv2d(hidden_dim, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel)
        ])
        self.conv = nn.Sequential(*layers)
        self.input_padding = fix_padding(3, dilation)

    def forward(self, x):
        x_pad = F.pad(x, self.input_padding)
        if self.use_residual_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)


def fix_padding(kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)
