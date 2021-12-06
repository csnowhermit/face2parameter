import torch

import config

'''
    模型key的处理
    多GPU训练的模型以module.开头，需改成单GPU模型的格式
'''

if __name__ == '__main__':
    # checkpoint = torch.load(config.imitator_model, map_location=config.device)
    # for new_key, old_key in checkpoint.items():
    #     checkpoint[new_key] = checkpoint.pop(old_key)
    # torch.save(checkpoint, './checkpoint/epoch_950.pt')

    new_model = {k.replace('module.', ''): v for k, v in torch.load(config.imitator_model, map_location=config.device).items()}
    torch.save(new_model, './checkpoint/epoch_950.pt')