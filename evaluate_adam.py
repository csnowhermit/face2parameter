import os
import cv2
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

import utils
import config
from imitator import Imitator
from lightcnn import LightCNN_29Layers_v2
from faceparse import BiSeNet

'''
    评估：t_params封装起来，做Adam优化
'''

'''
    evaluate with torch tensor
    :param input: torch tensor [B, H, W, C] rang: [0-1], not [0-255]
    :param image: 做mask图底色用
    :param bsnet: BiSeNet model
    :param w: tuple len=6 [eyebrow，eye，nose，teeth，up lip，lower lip]
    :return 带权重的tensor；脸部mask；带mask的图
'''
def faceparsing_tensor(input, image, bsnet, w):
    out = bsnet(input)    # [1, 19, 512, 512]
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)

    mask_img = utils.vis_parsing_maps(image, parsing, 1)
    out = out.squeeze()

    return w[0] * out[3] + w[1] * out[4] + w[2] * out[10] + out[11] + out[12] + out[13], out[1], mask_img

if __name__ == '__main__':
    eval_image = "./dat/gen.jpg"    # 要评估的图片

    # 加载lightcnn
    model = LightCNN_29Layers_v2(num_classes=80013)
    model.eval()
    if config.use_gpu:
        checkpoint = torch.load(config.lightcnn_checkpoint)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load(config.lightcnn_checkpoint, map_location="cpu")
        new_state_dict = model.state_dict()
        for k, v in checkpoint['state_dict'].items():
            _name = k[7:]  # remove `module.`
            new_state_dict[_name] = v
        model.load_state_dict(new_state_dict)

    # 冻结lightcnn
    for param in model.parameters():
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

    # 冻结imitator
    for param in imitator.parameters():
        param.requires_grad = False

    imitator_model = torch.load(config.imitator_model, map_location=torch.device('cpu'))
    imitator.load_state_dict(imitator_model)    # 这里加载已经处理过的参数

    # 图片读取
    img = cv2.imread(eval_image)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32)

    img1 = Image.open(eval_image)
    image = img1.resize((512, 512), Image.BILINEAR)

    # inference
    # t_params = 0.5 * torch.ones((1, config.continuous_params_size), dtype=torch.float32)
    t_params = torch.rand((1, config.continuous_params_size), dtype=torch.float32)    # 论文中用均匀分布，torch.randn()为正态分布
    optimizer = torch.optim.Adam([t_params], lr=config.eval_learning_rate)
    if config.use_gpu:
        t_params = t_params.cuda()
    t_params.requires_grad = True
    losses.clear()    # 清空损失

    # 用于计算L1损失的：L1_y [B, C, W, H]
    img_resized = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
    img_resized = np.swapaxes(img_resized, 0, 2).astype(np.float32)
    img_resized = np.mean(img_resized, axis=0)[np.newaxis, np.newaxis, :, :]

    # cv2.imwrite("l1_tmp.jpg", cv2.resize(img_resized[0].transpose(1, 2, 0), (512, 512)))
    img_resized = torch.from_numpy(img_resized)
    if config.use_gpu:
        img_resized = img_resized.cuda()
    L1_y = img_resized

    # 用于计算L2损失的参照：L2_y [B, C, H, W]
    img = img[np.newaxis, :, :, ]
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 1, 3)
    # print(img.shape)
    # cv2.imwrite("l2_tmp.jpg", cv2.resize(img[0].transpose(1, 2, 0), (512, 512)))

    img = torch.from_numpy(img)
    if config.use_gpu:
        img = img.cuda()
    L2_y = img / 255.



    # 做total_eval_steps次训练，取最后一次
    m_progress = tqdm(range(1, config.total_eval_steps + 1))
    for i in m_progress:
        y_ = imitator(t_params)    # [1, 3, 512, 512], [batch_size, c, w, h]
        # tmp = y_.detach().cpu().numpy()[0]
        # tmp = tmp.transpose(2, 1, 0)
        # tmp = tmp * 255.0
        # # cv2.imshow("y_", tmp)
        # # cv2.waitKey()
        # print(type(tmp), tmp.shape)
        # cv2.imwrite("./output/gen.jpg", tmp)
        # print("已保存")
        # break
        # loss, info = self.evaluate_ls(y_)
        y_copy = y_.clone()    # 复制出一份算L2损失

        # 衡量人脸相似度
        # L1损失:表示余弦距离的损失（判断是否为同一个人），大致身份的一致性。y_ [B, C, W, H]
        y_ = F.max_pool2d(y_, kernel_size=(4, 4), stride=4)  # 512->128, [1, 3, 128, 128]
        y_ = torch.mean(y_, dim=1).view(1, 1, 128, 128)  # gray

        # 计算L1损失时，BCWH改为BCHW
        # L1 = utils.discriminative_loss(torch.from_numpy(L1_y.detach().cpu().numpy().transpose(0, 1, 3, 2)),
        #                                torch.from_numpy(y_.detach().cpu().numpy().transpose(0, 1, 3, 2)),
        #                                model)

        L1 = utils.discriminative_loss(L1_y, y_, model)    # BCWH
        # if i % config.eval_prev_freq == 0:
        #     cv2.imwrite("L1_y_%d.jpg" % i, cv2.resize(L1_y.detach().cpu().numpy()[0].transpose(1, 2, 0), (512, 512)))
        #     cv2.imwrite("y_%d.jpg" % i, cv2.resize(y_.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255., (512, 512)))

        # L2损失：面部语义损失（关键部位加权计算损失）
        w_r = [1.1, 1., 1., 0.7, 1., 1.]
        w_g = [1.1, 1., 1., 0.7, 1., 1.]
        part1, _, mask_img1 = faceparsing_tensor(L2_y, image, bsnet, w_r)
        y_copy = y_copy.transpose(2, 3)    # [1, 3, 512, 512]
        part2, _, mask_img2 = faceparsing_tensor(y_copy, image, bsnet, w_g)
        # if i % config.eval_prev_freq == 0:
        #     cv2.imwrite("L2_y_%d.jpg" % i, L2_y.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.)
        #     cv2.imwrite("y_copy_%d.jpg" % i, y_copy.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.)

        L2_c = (part1 * 10, part2 * 10)
        L2 = F.l1_loss(part1, part2)

        L2_mask = (mask_img1, mask_img2)  # 用于在第二行显示mask的图

        # 计算综合损失：Ls = alpha * L1 + L2
        Ls = config.eval_alpha * (1 - L1) + L2
        # Ls = L2
        info = "L1:{0:.6f} L2:{1:.6f} Ls:{2:.6f}".format(1-L1, L2, Ls)
        print(info)
        losses.append((1-L1.item(), L2.item()/3, Ls.item()))

        optimizer.zero_grad()
        Ls.backward()
        optimizer.step()

        if i == 1:
            # utils.eval_output(imitator, t_params, img[0].cpu().detach().numpy(), 0, config.prev_path, L2_c)
            utils.eval_output(imitator, t_params, img[0].cpu().detach().numpy(), 0, config.prev_path, L2_mask)

        # eval_learning_rate = config.eval_learning_rate
        # if i % 10 == 0 and i != 0:    # 评估时学习率，每5轮衰减20%
        #     eval_learning_rate = (1 - 0.20) * eval_learning_rate

        # t_params.data = t_params.data - eval_learning_rate * t_params.grad.data
        # t_params.data = t_params.data.clamp(0., 1.)
        # print(i, t_params.grad, t_params.data)

        # one-hot编码：argmax处理（这里没搞清楚定义方法但没返回值的作用，直接写方法的内容在这里处理）
        # def argmax_params(params, start, count)
        # utils.argmax_params(t_params.data, 96, 3)sss    #这行有没有用？待定

        # start = 96
        # count = 3
        # dims = t_params.size()[0]
        # for dim in range(dims):
        #     tmp = t_params[dim, start]
        #     mx = start
        #     for idx in range(start + 1, start + count):
        #         if t_params[dim, idx] > tmp:
        #             mx = idx
        #             tmp = t_params[dim, idx]
        #     for idx in range(start, start + count):
        #         t_params[dim, idx] = 1. if idx == mx else 0

        # one-hot编码：argmax处理结束

        t_params.grad.zero_()
        m_progress.set_description(info)
        if i % config.eval_prev_freq == 0:
            x = i / float(config.total_eval_steps)
            lr = config.eval_learning_rate * (1 - x) + 1e-2
            # utils.eval_output(imitator, t_params, img[0].cpu().detach().numpy(), i, config.prev_path, L2_c)
            utils.eval_output(imitator, t_params, img[0].cpu().detach().numpy(), i, config.prev_path, L2_mask)
            utils.eval_plot(losses)
    utils.eval_plot(losses)

    # 输出
    # utils.eval_output(imitator, t_params, img[0].cpu().detach().numpy(), config.total_eval_steps+1, config.prev_path, L2_c)
    utils.eval_output(imitator, t_params, img[0].cpu().detach().numpy(), config.total_eval_steps+1, config.prev_path, L2_mask)
