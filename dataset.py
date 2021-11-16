import os
import cv2
import struct
import random
import torch
import numpy as np

import config
from utils import NeuralException


# face数据集：由unity生成
class FaceDataset:
    def __init__(self, args, mode="train"):
        self.names = []    # 这里图片和param是成对出现的
        self.params = []
        if mode == "train":
            self.path = args.train_set
        elif mode == "test":
            self.path = args.test_set
        else:
            raise NeuralException("not such mode for dataset")

        self.args = args
        if os.path.exists(self.path):
            name = "db_description"
            path = os.path.join(self.path, name)
            f = open(path, "rb")
            self.cnt = struct.unpack("i", f.read(4))[0]
            for it in range(self.cnt):
                kk = f.read(11)[1:]  # 第一个是c#字符串的长度
                self.names.append(str(kk, encoding='utf-8'))
                v = []
                for i in range(args.params_cnt):
                    v.append(struct.unpack("f", f.read(4))[0])
                self.params.append(v)
            f.close()
        else:
            print("can't be found path %s. Skip it.", self.path)

    '''
        获取一张图片
        :param idx 指定下标，-1表示随机取
    '''
    def get_image(self, idx=-1):
        count = len(self.names)
        if idx >= count:
            raise NeuralException("dataset override array bounds")
        if idx == -1:
            idx = random.randint(0, count)

        name = self.names[idx]
        param = self.params[idx]
        image = cv2.imread(os.path.join(self.path, name + ".jpg"))
        return name, param, image

    '''
        批量获取图片
        :param batch_size batch_size
        :param edge 
    '''
    def get_batch(self, batch_size, edge):
        names = []
        cnt = self.cnt
        param_cnt = config.params_cnt
        size = 64 if edge else 512
        deep = 1 if edge else 3    # 单通道/三通道

        np_params = np.zeros((batch_size, param_cnt), dtype=np.float32)
        np_images = np.zeros((batch_size, deep, size, size), dtype=np.float32)
        read_mode = cv2.IMREAD_GRAYSCALE if edge else cv2.IMREAD_COLOR

        for i in range(batch_size):
            idx = random.randint(0, cnt-1)
            name = self.names[idx]
            np_params[i] = self.params[idx]
            name = name + ".jpg"
            names.append(name)
            if edge:
                path = os.path.join(self.path + "2", name)
            else:
                path = os.path.join(self.path, name)
            image = cv2.imread(path, read_mode)
            if not edge:
                np_images[i] = np.swapaxes(image, 0, 2) / 255.  # [C, W, H]
            else:
                np_images[i] = image[np.newaxis, :, :] / 255.  # [1, H, W]
        params = torch.from_numpy(np_params)
        images = torch.from_numpy(np_images)
        return names, params, images




