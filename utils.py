import os
import cv2
import struct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import config
from faceparse import BiSeNet

# 反卷积
def deconv_layer(in_chanel, out_chanel, kernel_size, stride=1, pad=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride, padding=pad),
        nn.BatchNorm2d(out_chanel),
        nn.ReLU())

# 自定义异常
class NeuralException(Exception):
    def __init__(self, message):
        print("neural error: " + message)
        self.message = "neural exception: " + message

'''
    提取原始图像的边缘
    :param img: input image
    :return: edge image
'''
def img_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x_grad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    y_grad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    return cv2.Canny(x_grad, y_grad, 40, 130)

'''
    将tensor转numpy array 给cv2使用
    :param tensor: [batch, c, w, h]
    :return: [batch, h, w, c]
'''
def tensor_2_image(tensor):

    batch = tensor.size(0)
    images = []
    for i in range(batch):
        img = tensor[i].cpu().detach().numpy()
        img = np.swapaxes(img, 0, 2)  # [h, w, c]
        img = np.swapaxes(img, 0, 1)  # [w, h, c]
        images.append(img * 255)
    return images

'''
    [W, H, 1] -> [W, H, 3] or [W, H]->[W, H, 3]
    :param image: input image
    :return: transfer image
'''
def fill_gray(image):
    shape = image.shape
    if len(shape) == 2:
        image = image[:, :, np.newaxis]
        shape = image.shape
    if shape[2] == 1:
        return np.pad(image, ((0, 0), (0, 0), (1, 1)), 'edge')
    elif shape[2] == 3:
        return np.mean(image, axis=2)
    return image

'''
    imitator 快照
    :param path: save path
    :param tensor1: input photo
    :param tensor2: generated image
    :param parse: parse checkpoint's path
'''
def capture(path, tensor1, tensor2, parse, cuda):
    img1 = tensor_2_image(tensor1)[0].swapaxes(0, 1).astype(np.uint8)
    img2 = tensor_2_image(tensor2)[0].swapaxes(0, 1).astype(np.uint8)
    img1 = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_LINEAR)
    img3 = faceparsing_ndarray(img1, parse, cuda)
    img4 = img_edge(img3)
    img4 = 255 - fill_gray(img4)
    image = merge_4image(img1, img2, img3, img4, transpose=False)
    cv2.imwrite(path, image)

def merge_4image(image1, image2, image3, image4, size=512, show=False, transpose=True):
    """
    拼接图片
    :param image1: input image1, numpy array
    :param image2: input image2, numpy array
    :param image3: input image3, numpy array
    :param image4: input image4, numpy array
    :param size: 输出分辨率
    :param show: 窗口显示
    :param transpose: 转置长和宽 cv2顺序[H, W, C]
    :return: merged image
    """
    size_ = (int(size / 2), int(size / 2))
    img_1 = cv2.resize(image1, size_)
    # cv2.imshow("img1", img_1)
    # cv2.waitKey()

    img_2 = cv2.resize(image2, size_)
    # cv2.imshow("img2", img_2)
    # cv2.waitKey()

    img_3 = cv2.resize(image3, size_)
    # cv2.imshow("img3", img_3)
    # cv2.waitKey()

    img_4 = cv2.resize(image4, size_)
    # cv2.imshow("img4", img_4)
    # cv2.waitKey()

    image1_ = np.append(img_1, img_2, axis=1)    # axis=1,行
    image2_ = np.append(img_3, img_4, axis=1)
    image = np.append(image1_, image2_, axis=0)
    if transpose:
        image = image.swapaxes(0, 1)
    if show:
        cv2.imshow("contact", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return image

'''
    evaluate with numpy array
    :param input: numpy array, 注意一定要是np.uint8, 而不是np.float32 [H, W, C]
    :param cp: args.parsing_checkpoint, str
    :param cuda: use gpu to speedup
'''
def faceparsing_ndarray(input, checkpoint, cuda=False):
    # 构建BiSeNet并加载模型
    bsnet = BiSeNet(n_classes=19)
    if cuda:
        bsnet.cuda()
        bsnet.load_state_dict(torch.load(checkpoint))
    else:
        bsnet.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    bsnet.eval()

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),  # [H, W, C]->[C, H, W]
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    # input_ = _to_tensor_(input)
    input_ = to_tensor(input)
    input_ = torch.unsqueeze(input_, 0)

    if cuda:
        input_ = input_.cuda()
    out = bsnet(input_)
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    return vis_parsing_maps(input, parsing, stride=1)

'''
    结果可视化
'''
def vis_parsing_maps(im, parsing, stride):
    """
    # 显示所有部位
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
                   [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
                   [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    """
    # 只显示脸 鼻子 眼睛 眉毛 嘴巴
    part_colors = [[255, 255, 255], [255, 85, 0], [25, 170, 0], [255, 170, 0], [254, 0, 170], [254, 0, 170],
                   [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [0, 0, 254], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
    """
    part_colors = [[255, 255, 255], [脸], [左眉], [右眉], [左眼], [右眼],
                   [255, 255, 255],
                   [左耳], [右耳], [255, 255, 255], [鼻子], [牙齿], [上唇], 
                   [下唇],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
    """

    im = np.array(im)
    vis_parsing = parsing.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing.shape[0], vis_parsing.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing)
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    return vis_parsing_anno_color

'''
    论文里的判别损失, 判断真实照片和由模拟器生成的图像是否属于同一个身份
    Discriminative Loss 使用余弦距离
    https://www.cnblogs.com/dsgcBlogs/p/8619566.html
    :param lightcnn_inst: lightcnn model instance
    :param img1: generated by engine, type: list of Tensor
    :param img2: generated by imitator, type: list of Tensor
    :return tensor scalar，余弦相似度
'''
def discriminative_loss(img1, img2, lightcnn_inst):
    x1 = batch_feature256(img1, lightcnn_inst)    # [1, 256]
    x2 = batch_feature256(img2, lightcnn_inst)    # [1, 256]
    distance = torch.cosine_similarity(x1, x2)
    return torch.mean(distance)

def batch_feature256(img, lightcnn_inst):
    """
       使用light cnn提取256维特征参数
       :param lightcnn_inst: lightcnn model instance
       :param img: tensor 输入图片 shape:(batch, 1, 512, 512)
       :return: 256维特征参数 tensor [batch, 256]
       """
    _, features = lightcnn_inst(img)
    # log.debug("features shape:{0} {1} {2}".format(features.size(), features.requires_grad, img.requires_grad))
    return features

'''
    capture for result
    :param x: generated image with grad, torch tensor [b,params]
    :param refer: reference picture: [3, 512, 512]
    :param step: train step
'''
def eval_output(imitator, x, refer, step, prev_path, L2_c):
    eval_write(x)
    y_ = imitator(x)
    y_ = y_.cpu().detach().numpy()
    y_ = np.squeeze(y_, axis=0)
    y_ = np.swapaxes(y_, 0, 2) * 255
    y_ = y_.astype(np.uint8)    # [512, 512, 3]
    im1 = L2_c[0]    # [512, 512]
    im2 = L2_c[1]    # [512, 512]
    # np_im1 = im1.cpu().detach().numpy()
    # np_im2 = im2.cpu().detach().numpy()
    # f_im1 = fill_gray(np_im1)    # [512, 512, 3]，灰度图
    # f_im2 = fill_gray(np_im2)    # [512, 512, 3]
    f_im1 = im1  # [512, 512, 3]，这里直接显示原图
    f_im2 = im2  # [512, 512, 3]

    # refer 改为channel last的
    refer = np.transpose(refer, [1, 2, 0])    # [512, 512, 3]
    # print("f_im1:", type(f_im1), f_im1.shape)
    image_ = merge_4image(refer, y_, f_im1, f_im2, transpose=False)
    path = os.path.join(prev_path, "eval_{0}.jpg".format(step))
    cv2.imwrite(path, image_)

'''
    生成二进制文件 能够在unity里还原出来
    :param params: 捏脸参数 tensor [batch, params_cnt]
'''
def eval_write(params):
    np_param = params.cpu().detach().numpy()
    np_param = np_param[0]
    list_param = np_param.tolist()
    dataset = config.train_set
    shape = curr_roleshape(dataset)
    path = os.path.join(config.model_path, "eval.bytes")
    f = open(path, 'wb')
    write_layer(f, shape, list_param)
    f.close()

'''
    判断当前运行的是roleshape (c# RoleShape)
    :param dataset: args path_to_dataset
    :return: RoleShape
'''
def curr_roleshape(dataset):
    if dataset.find("female") >= 0:
        return 4
    else:
        return 3

def write_layer(f, shape, args):
    f.write(struct.pack('i', shape))
    for it in args:
        byte = struct.pack('f', it)
        f.write(byte)

'''
    One-hot编码 argmax 处理
    :param params: 处理params
    :param start: One-hot 偏移起始地址
    :param count: One-hot 编码长度
'''
def argmax_params(params, start, count):
    dims = params.size()[0]
    for dim in range(dims):
        tmp = params[dim, start]
        mx = start
        for idx in range(start + 1, start + count):
            if params[dim, idx] > tmp:
                mx = idx
                tmp = params[dim, idx]
        for idx in range(start, start + count):
            params[dim, idx] = 1. if idx == mx else 0

# plot loss
def eval_plot(losses):
    count = len(losses)
    if count > 0:
        plt.style.use('seaborn-whitegrid')
        x = range(count)
        y1 = []
        y2 = []
        for it in losses:
            y1.append(it[0])
            y2.append(it[1])
        plt.plot(x, y1, color='r', label='1-L1')
        plt.plot(x, y2, color='g', label='L2')
        plt.ylabel("loss")
        plt.xlabel('step')
        plt.legend()
        path = os.path.join(config.prev_path, "loss.png")
        plt.savefig(path)
        plt.close('all')

'''
    dlib检测68个关键点
    :param img BGR三通道图
'''
def detect_face_keypoint(img):
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = config.detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in config.predictor(img, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            print(idx, pos)

            cv2.circle(img, pos, 5, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("img", img)
    cv2.waitKey(0)
