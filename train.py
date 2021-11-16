import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import utils
import config
from imitator import Imitator
from dataset import FaceDataset

'''
    训练
    数据预处理：Face alignment, dlib
'''

if __name__ == '__main__':
    imitator = Imitator("imitator model", args=config)

    rand_input = torch.randn(config.batch_size, config.params_cnt)
    if config.use_gpu:
        rand_input = rand_input.cuda()

    writer = SummaryWriter(comment='imitator', log_dir=config.path_tensor_log)

    imitator.train()
    optimizer = optim.Adam(imitator.parameters(), lr=config.learning_rate)

    dataset = FaceDataset(config, mode="train")
    init_step = config.init_step
    total_steps = config.total_steps
    progress = tqdm(range(init_step, total_steps + 1), initial=init_step, total=total_steps)
    for step in progress:
        names, params, images = dataset.get_batch(batch_size=config.batch_size, edge=False)
        if config.use_gpu:
            params = params.cuda()
            images = images.cuda()

        # 开始训练
        optimizer.zero_grad()
        y_ = imitator(params)
        loss = F.l1_loss(images, y_)
        loss.backward()
        optimizer.step()

        loss_ = loss.cpu().detach().numpy()    # 转到cpu上
        progress.set_description("loss: {:.3f}".format(loss_))
        writer.add_scalar('imitator/loss', loss_, step)

        # 每隔prev_step步显示
        if (step + 1) % config.prev_freq == 0:
            path = "{1}/imit_{0}.jpg".format(step + 1, config.prev_path)

            # 保存快照
            utils.capture(path, images, y_, config.faceparse_checkpoint, config.use_gpu)
            x = step / float(total_steps)
            lr = config.learning_rate * (x ** 2 - 2 * x + 1) + 2e-3

            # 动态更新lr，加快训练速度
            for group in optimizer.param_groups:
                group['lr'] = lr

            writer.add_scalar('imitator/learning rate', lr, step)

            # 把imitator的权重以图片的方式上传到tensorboard
            for module in imitator._modules.values():
                if isinstance(module, nn.Sequential):
                    for it in module._modules.values():
                        if isinstance(it, nn.ConvTranspose2d):
                            if it.in_channels == 64 and it.out_channels == 64:
                                name = "weight_{0}_{1}".format(it.in_channels, it.out_channels)
                                weights = it.weight.reshape(4, 64, -1)
                                writer.add_image(name, weights, step)
        # 每隔save_freq步保存
        if (step + 1) % config.save_freq == 0:
            state = {'net': imitator.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': step}
            if os.path.exists(config.model_path) is False:
                os.makedirs(config.model_path)
            ext = "cuda" if config.use_gpu else "cpu"
            torch.save(state, '{1}/imitator_{0}_{2}.pth'.format(step + 1, config.model_path, ext))
    writer.close()
