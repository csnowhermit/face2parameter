import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.utils as vutils
from PIL import Image

import utils
import config
from imitator import Imitator
from lightcnn import LightCNN_29Layers_v2
from face_parser import load_model
from face_align import template_path, face_Align
from faceswap import template_path, faceswap

'''
    evaluate with torch tensor
    :param input: torch tensor [B, H, W, C] rang: [0-1], not [0-255]
    :param image: 做mask图底色用
    :param bsnet: BiSeNet model
    :param w: tuple len=6 [eyebrow，eye，nose，teeth，up lip，lower lip]
    :return 带权重的tensor；脸部mask；带mask的图
'''
def faceparsing_tensor(input, image, face_parse, is_label=False):
    out = face_parse(input)    # [1, 19, 512, 512]
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    mask_img = utils.vis_parsing_maps(image, parsing, 1)
    # # 7类
    # {
    #     0: 'background',
    #     1: 'face',
    #     2: 'brow',
    #     3: 'eye',
    #     4: 'nose',
    #     5: 'up_lip',
    #     6: 'down_lip'
    # }

    if is_label:    # 如果是标签，则做成[1, 512, 512]的
        out = out.max(1)[1]
    else:
        out = out.squeeze()    # [7, 512, 512]
        # tmp = out[4]
    return out, mask_img

if __name__ == '__main__':
    eval_imagepath = "./dat/16.jpg"  # 要评估的图片

    # 1.加载lightcnn
    lightcnn = LightCNN_29Layers_v2(num_classes=80013)
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

    losses = []

    # 2.加载语义分割网络
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    deeplab = load_model(backbone=config.faceparse_backbone, num_classes=config.num_classes, output_stride=config.output_stride)
    checkpoint = torch.load(config.faceparse_checkpoint, map_location=torch.device('cpu'))
    deeplab.load_state_dict(checkpoint['model_state'])
    deeplab.eval()

    for param in deeplab.parameters():
        param.requires_grad = False

    # 3.加载imitator
    imitator = Imitator(is_bias=True)

    l2_c = (torch.ones((512, 512)), torch.ones((512, 512)))
    if config.use_gpu:
        imitator.cuda()
    imitator.eval()
    imitator_model = torch.load(config.imitator_model, map_location=torch.device('cpu'))
    imitator.load_state_dict(imitator_model)  # 这里加载已经处理过的参数

    # 冻结imitator
    for param in imitator.parameters():
        param.requires_grad = False

    # 读取图片
    # 先对齐
    warped_mask = face_Align(template_path, eval_imagepath)
    warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2RGB)
    # img_upload = Image.open(eval_imagepath)
    img_upload = Image.fromarray(np.uint8(warped_mask))  # 对齐后的

    image_mask = img_upload.convert('RGB').resize((512, 512), Image.BILINEAR)    # 做mask图用

    image_F1 = img_upload.convert('L').resize((128, 128), Image.BILINEAR)  # F1损失：身份验证损失
    image_F1 = T.ToTensor()(image_F1).unsqueeze(0)    # [1, 1, 128, 128]

    image_F2 = img_upload.convert('RGB').resize((512, 512), Image.BILINEAR)    # F2损失：内容损失
    image_F2 = T.ToTensor()(image_F2).unsqueeze(0)    # [1, 3, 512, 512]
    image_F2 = transform(image_F2)  # 训练mobilenet时做了normalize，推理时也得做

    t_params = torch.full([1, config.continuous_params_size], 0.5, dtype=torch.float32)  # 从平均人脸初始化
    optimizer = torch.optim.SGD([t_params], lr=config.eval_learning_rate, momentum=0.9)  # SGD带动量的
    if config.use_gpu:
        t_params = t_params.cuda()

    t_params.requires_grad = True
    losses.clear()  # 清空损失

    # 做total_eval_steps次训练，取最后一次
    m_progress = tqdm(range(1, config.total_eval_steps + 1))
    for i in m_progress:
        gen_img = imitator(t_params)  # [1, 3, 512, 512]
        gen_img_copy = gen_img.clone()  # 复制出一份来展示用

        # 1.身份验证损失
        trans = T.Compose([
            T.Resize((128, 128)),
            T.Grayscale()
        ])

        F1_Loss = utils.discriminative_loss(image_F1, trans(gen_img), lightcnn)    # 身份验证损失

        # 2.内容损失
        gen_img = transform(gen_img)
        upload_F2_feature, mask_img_upload = faceparsing_tensor(image_F2, image_mask, deeplab, is_label=True)  # 参照
        gen_F2_feature, mask_img_gen = faceparsing_tensor(gen_img, image_mask, deeplab, is_label=False)  # 生成

        # tmp = upload_F2_feature.view(-1, 512*512).softmax(dim=0).max(0)[0].long()

        # F2_Loss = F.l1_loss(upload_F2_feature, gen_F2_feature)
        F2_Loss = F.cross_entropy(gen_F2_feature.view(-1, config.num_classes, 512*512), upload_F2_feature.view(-1, 512*512).long())    # 这里改用交叉熵损失
        # F2_Loss = F.cross_entropy(gen_F2_feature.view(-1, config.num_classes, 512*512), upload_F2_feature.view(-1, 512*512).softmax(dim=0).max(0)[0].view(-1, 512*512).long())    # 采用带条件概率的交叉熵损失

        # 计算综合损失：Ls = alpha * L1 + L2
        Ls = config.eval_alpha * (1 - F1_Loss) + F2_Loss
        # Ls = F1_Loss
        # Ls = F2_Loss
        info = "1-F1:{0:.6f} F2:{1:.6f} Ls:{2:.6f}".format(1 - F1_Loss, F2_Loss, Ls)
        print(info)
        losses.append((1 - F1_Loss.item(), F2_Loss.item(), Ls.item()))

        optimizer.zero_grad()
        Ls.backward()
        optimizer.step()

        if i % 5 == 0 and i != 0:  # 评估时学习率，每5轮衰减20%
            for p in optimizer.param_groups:
                p["lr"] *= 0.8
        t_params.data = t_params.data.clamp(0.05, 0.95)
        t_params.grad.zero_()
        m_progress.set_description(info)

        if i % config.eval_prev_freq == 0 or i == 1:
            # image_upload, gen_img, mask_img_upload, mask_img_gen
            # tmp = T.PILToTensor()(T.ToPILImage()(gen_img[0]))
            # tmp.show()

            show_img_list = [T.PILToTensor()(img_upload), T.PILToTensor()(T.ToPILImage()(gen_img_copy[0])), T.ToTensor()(mask_img_upload) * 255., T.ToTensor()(mask_img_gen) * 255.]

            label_show = vutils.make_grid(show_img_list, nrow=2, padding=2, normalize=True).cpu()
            vutils.save_image(label_show, os.path.join(config.prev_path, "eval_%d.png" % i))
