import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import utils
import config
from imitator import Imitator
from dataset import Imitator_Dataset

'''
    训练
    数据预处理：Face alignment, dlib
'''

if __name__ == '__main__':
    imitator = Imitator("imitator model")
    if len(config.imitator_model) > 0:
        if config.use_gpu:
            imitator_model = torch.load(config.imitator_model)
        else:
            imitator_model = torch.load(config.imitator_model, map_location=torch.device('cpu'))
        print("load pretrained model success!")
    else:
        print("No pretrained model...")

    imitator = imitator.to(config.device)
    if config.device.type == 'cuda' and config.num_gpu > 1:
        imitator = nn.DataParallel(imitator, list(range(config.num_gpu)))

    imitator.train()
    optimizer = optim.Adam(imitator.parameters(), lr=config.learning_rate)

    train_imitator_Dataset = Imitator_Dataset(config.train_params_root, config.image_root)
    train_imitator_dataloader = DataLoader(train_imitator_Dataset, batch_size=config.batch_size, shuffle=True)

    test_imitator_Dataset = Imitator_Dataset(config.test_params_root, config.image_root)
    test_imitator_dataloader = DataLoader(test_imitator_Dataset, batch_size=config.batch_size, shuffle=True)

    progress = tqdm(range(config.init_step, config.total_steps + 1), initial=config.init_step, total=config.total_steps)

    train_loss = 999999  # 记录当前train loss

    for step in progress:
        for i, content in enumerate(train_imitator_dataloader):
            x, y = content[:]
            if config.use_gpu:
                x = x.to(config.device)
                y = y.to(config.device)

            optimizer.zero_grad()
            y_ = imitator(x)
            loss = F.l1_loss(y, y_)
            loss.backward()
            optimizer.step()

            info = "step:{0:d} batch:{1:d} Loss:{2:.6f}".format(step, i, loss)
            print(info)
            progress.set_description(info)
        curr_train_loss = loss.item()
        if curr_train_loss < train_loss:
            train_loss = curr_train_loss

        # if (step + config.init_step + 1) % config.save_freq == 0:
        if True:
            imitator.eval()
            test_losses = []
            for i, content in enumerate(test_imitator_dataloader):
                x, y = content[:]
                if config.use_gpu:
                    x = x.to(config.device)
                    y = y.to(config.device)

                y_ = imitator(x)    # BCHW
                test_loss = F.l1_loss(y, y_)
                test_losses.append(test_loss.item())

                if i == 0:
                    if os.path.exists(config.prev_path) is False:
                        os.makedirs(config.prev_path)
                    # utils.capture(path, x, y_, config.faceparse_checkpoint, config.use_gpu)

                    label_show = np.transpose(vutils.make_grid(y.to(config.device), nrow=4, padding=2, normalize=True).cpu().numpy(),(1,2,0)) * 255.
                    cv2.imwrite(os.path.join(config.prev_path, "show_label_%d.jpg" % step), cv2.cvtColor(label_show, cv2.COLOR_RGB2BGR))

                    fake_show = np.transpose(vutils.make_grid(y_.to(config.device), nrow=4, padding=2, normalize=True).detach().cpu().numpy(),(1,2,0)) * 255.
                    cv2.imwrite(os.path.join(config.prev_path, "show_fake_%d.jpg" % step), cv2.cvtColor(fake_show, cv2.COLOR_RGB2BGR))

            test_avg_loss = np.mean(test_losses)
            torch.save(imitator.state_dict(), os.path.join(config.model_path, "imitator-%d-%.3f-%.3f.pth" % (step, curr_train_loss, test_avg_loss)))

