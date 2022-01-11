import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
from collections import OrderedDict

import config
from backbone.resnet import ResNet50, Bottleneck
from backbone.mobilenet import MobileNetV2

'''
    指定backbone，加载模型
'''
def load_model(backbone, num_classes, output_stride):
    if backbone == 'resnet50':
        model = segment_resnet(num_classes=num_classes, output_stride=output_stride)
    elif backbone == 'mobilenetv2':
        model = segment_mobilenetv2(num_classes=num_classes, output_stride=output_stride)
    else:
        raise NotImplementedError
    return model

'''
    语义分割网络，使用resnet作为backbone
    :param num_classes 分割的类别个数
    :param output_stride 
'''
def segment_resnet(num_classes, output_stride):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]    # 是否用空洞卷积代替stride
        aspp_dilate = [12, 24, 36]    # ASPP结构空洞卷积的dilate大小
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    # 加载backbone（这里resnet50最后两层仅为了加载预训练模型，需设置num_classes为1000）
    backbone = ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=1000, replace_stride_with_dilation=replace_stride_with_dilation)
    if config.pretrained:
        state_dict = load_state_dict_from_url(config.model_urls, progress=config.progress)
        backbone.load_state_dict(state_dict)
        del state_dict

    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}    # layer4作为resnet的最后输出，作为encoder端ASPP结构的输入；layer1作为低等级特征，直接输入到decoder端
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    # 提取网络的第几层输出结果并给一个别名
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # 组装成完整的分割模型
    # model = Simple_Segmentation_Model(backbone, classifier)
    model = FaceSegmentation(backbone, classifier)
    return model

'''
    语义分割网络，使用mobilenetv2作为backbone
    :param num_classes 分割的类别个数
    :param output_stride 
'''
def segment_mobilenetv2(num_classes, output_stride):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    # 这里num_classes要写1000，为加载上预训练模型
    backbone = MobileNetV2(num_classes=1000, output_stride=output_stride, width_mult=1.0, inverted_residual_setting=None, round_nearest=8)
    if config.pretrained:
        state_dict = load_state_dict_from_url(config.model_urls, progress=config.progress)
        backbone.load_state_dict(state_dict)
        del state_dict

    # 将backbone的特征分为高阶特征和低阶特征
    backbone.low_level_features = backbone.features[0:4]    # 倒数第四层之前的全为低等级特征
    backbone.high_level_features = backbone.features[4:-1]    # 第四层开始，到倒数第二层的为高等级特征
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)    # 这里要写传入参数的num_classes，为加载voc预训练模型考虑

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = FaceSegmentation(backbone, classifier)
    return model

'''
    deeplabv3+
'''
class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        # print(feature.shape)
        low_level_feature = self.project(
            feature['low_level'])  # return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        # print(low_level_feature.shape)
        output_feature = self.aspp(feature['out'])
        # print(output_feature.shape)
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        # print(output_feature.shape)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

'''
    ASPP: atrous conv spp，空洞卷积+SPP
'''
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            #print(conv(x).shape)
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

'''
    ASPP模块中conv结构
'''
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

'''
    ASPP模块中pooling结构
'''
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

'''
    完整的分割模型
'''
class Simple_Segmentation_Model(nn.Module):
    def __init__(self, backbone, classifier):
        super(Simple_Segmentation_Model, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class FaceSegmentation(Simple_Segmentation_Model):
    pass

if __name__ == '__main__':
    model = load_model('resnet50', num_classes=config.num_classes, output_stride=config.output_stride)
    model = model.to(config.device)
    # print(model)
    input = torch.randn([16, 3, 513, 513])
    output = model(input)
    print(output.shape)    # torch.Size([16, 10, 513, 513])

