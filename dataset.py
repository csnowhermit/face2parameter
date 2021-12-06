import os
import json
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

import config


class Imitator_Dataset(Dataset):
    def __init__(self, params_root, image_root):
        self.image_root = image_root
        with open(params_root, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        self.fileList = [k for k in self.params.keys()]

    def __getitem__(self, index):
        file = self.fileList[index]
        img = Image.open(os.path.join(self.image_root, file)).convert("RGB")
        img = T.ToTensor()(img)
        param = torch.tensor(self.params['%s' % (file)])  # 一般做法是init里获取文件列表，按照index下标从文件列表中取文件名

        return (param, img)

    def __len__(self):
        return len(self.params.keys())


if __name__ == '__main__':
    train_imitator_Dataset = Imitator_Dataset(config.train_params_root, config.image_root)
    train_imitator_dataloader = DataLoader(train_imitator_Dataset, batch_size=16, shuffle=True)
    for i, content in enumerate(train_imitator_dataloader):
        x, y = content[:]
        print(i, x.shape, y.shape)

