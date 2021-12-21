import os
import json
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

import config


class Imitator_Dataset(Dataset):
    def __init__(self, params_root, image_root, mode="train"):
        self.image_root = image_root
        self.mode = mode
        with open(params_root, encoding='utf-8') as f:
            self.params = json.load(f)

    def __getitem__(self, index):
        if self.mode == "val":
            img = Image.open(os.path.join(self.image_root, '%d.png' % (index + 54000))).convert("RGB")
            param = torch.tensor(self.params['%d.png' % (index + 54000)])
        else:
            img = Image.open(os.path.join(self.image_root, '%d.png' % index)).convert("RGB")
            param = torch.tensor(self.params['%d.png' % index])
        img = T.ToTensor()(img)
        return param, img

    def __len__(self):
        if self.mode == "train":
            return 54000
        else:
            return 6000


if __name__ == '__main__':
    train_imitator_Dataset = Imitator_Dataset(config.train_params_root, config.image_root)
    train_imitator_dataloader = DataLoader(train_imitator_Dataset, batch_size=16, shuffle=True)
    for i, content in enumerate(train_imitator_dataloader):
        x, y = content[:]
        print(i, x.shape, y.shape)

