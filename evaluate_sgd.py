import os
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
from faceparse import BiSeNet

'''
    evaluate with torch tensor
    :param input: torch tensor [B, H, W, C] rang: [0-1], not [0-255]
    :param image: 做mask图底色用
    :param bsnet: BiSeNet model
    :param w: tuple len=6 [eyebrow，eye，nose，teeth，up lip，lower lip]
    :return 带权重的tensor；脸部mask；带mask的图
'''
def faceparsing_tensor(input, image, bsnet, is_cartoon=False):
    out = bsnet(input)    # [1, 19, 512, 512]
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    mask_img = utils.vis_parsing_maps(image, parsing, 1)

    out = out.squeeze()    # [19, 512, 512]
    prob_mat = F.softmax(out, dim=0)
    tmp1 = out[2].mul(prob_mat[2]) + out[3].mul(prob_mat[3])
    tmp2 = out[4].mul(prob_mat[4]) + out[5].mul(prob_mat[5])
    tmp3 = torch.cat([tmp1.view(1, 512, 512), tmp2.view(1, 512, 512), out[[1, 10, 11, 12, 13]]], dim=0)

    return tmp3, out[1], mask_img

if __name__ == '__main__':
    eval_imagepath = "./dat/rand_cut_result_5.jpg"  # 要评估的图片

    # 加载lightcnn
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

    # 加载BiSeNet
    bsnet = BiSeNet(n_classes=19)
    if config.use_gpu:
        bsnet.cuda()
        bsnet.load_state_dict(torch.load(config.faceparse_checkpoint))
    else:
        bsnet.load_state_dict(torch.load(config.faceparse_checkpoint, map_location="cpu"))
    bsnet.eval()

    # 冻结BiSeNet
    for param in bsnet.parameters():
        param.requires_grad = False

    # 加载imitator
    imitator = Imitator()

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
    img_upload = Image.open(eval_imagepath)

    image_mask = img_upload.convert('RGB').resize((512, 512), Image.BILINEAR)    # 做mask图用

    image_F1 = img_upload.convert('L').resize((128, 128), Image.BILINEAR)  # F1损失：身份验证损失
    image_F1 = T.ToTensor()(image_F1).unsqueeze(0)    # [1, 1, 128, 128]

    image_F2 = img_upload.convert('RGB').resize((512, 512), Image.BILINEAR)    # F2损失：内容损失
    image_F2 = T.ToTensor()(image_F2).unsqueeze(0)    # [1, 3, 512, 512]

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

        # 1.身份验证损失
        trans = T.Compose([
            T.Resize((128, 128)),
            T.Grayscale()
        ])
        # gen_img_copy = gen_img.clone()
        # tmp = trans(gen_img)
        F1_Loss = utils.discriminative_loss(image_F1, trans(gen_img), lightcnn)[0]    # 身份验证损失

        # 2.内容损失
        upload_F2_feature, _, mask_img_upload = faceparsing_tensor(image_F2, image_mask, bsnet, is_cartoon=False)  # 参照
        gen_F2_feature, _, mask_img_gen = faceparsing_tensor(gen_img, image_mask, bsnet, is_cartoon=False)  # 生成

        F2_Loss = F.l1_loss(upload_F2_feature, gen_F2_feature)

        # 计算综合损失：Ls = alpha * L1 + L2
        # Ls = config.eval_alpha * (1 - F1_Loss) + F2_Loss
        Ls = F2_Loss
        info = "1-F1:{0:.6f} F2:{1:.6f} Ls:{2:.6f}".format(1 - F1_Loss, F2_Loss, Ls)
        print(info)
        losses.append((1 - F1_Loss.item(), F2_Loss.item() / 3, Ls.item()))

        optimizer.zero_grad()
        Ls.backward()
        optimizer.step()

        if i % 5 == 0 and i != 0:  # 评估时学习率，每5轮衰减20%
            for p in optimizer.param_groups:
                p["lr"] *= 0.8
        t_params.data = t_params.data.clamp(0.2, 0.8)
        t_params.grad.zero_()
        m_progress.set_description(info)

        if i % config.eval_prev_freq == 0 or i == 1:
            # image_upload, gen_img, mask_img_upload, mask_img_gen
            # tmp = T.PILToTensor()(T.ToPILImage()(gen_img[0]))
            # tmp.show()

            show_img_list = [T.PILToTensor()(img_upload), T.PILToTensor()(T.ToPILImage()(gen_img[0])), T.ToTensor()(mask_img_upload) * 255., T.ToTensor()(mask_img_gen) * 255.]

            label_show = vutils.make_grid(show_img_list, nrow=2, padding=2, normalize=True).cpu()
            vutils.save_image(label_show, os.path.join(config.prev_path, "eval_%d.png" % i))
