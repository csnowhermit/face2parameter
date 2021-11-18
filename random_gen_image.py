import cv2
import torch
import random

import config
from imitator import Imitator


if __name__ == '__main__':
    imitator = Imitator()
    imitator_model = torch.load(config.imitator_model, map_location=torch.device('cpu'))
    imitator.load_state_dict(imitator_model)  # 这里加载已经处理过的参数

    for i in range(10):
        if random.randint(1, 10) % 2 == 0:
            t_params = torch.rand((1, config.continuous_params_size), dtype=torch.float32)
        else:
            # t_params = torch.randn((1, config.continuous_params_size), dtype=torch.float32)
            t_params = torch.normal(0.5, 1, (1, config.continuous_params_size))
            print("2.1.", t_params)
            t_params.data = t_params.data.clamp(0., 1.)
            print("2.2.", t_params)
        # t_params = torch.rand((1, config.continuous_params_size), dtype=torch.float32)

        y_ = imitator(t_params)  # [1, 3, 512, 512], [batch_size, c, w, h]
        tmp = y_.detach().cpu().numpy()[0]
        tmp = tmp.transpose(2, 1, 0)
        tmp = tmp * 255.0
        # cv2.imshow("y_", tmp)
        # cv2.waitKey()
        print(type(tmp), tmp.shape)
        cv2.imwrite("./dat/gen_%d.jpg" % i, tmp)
        print("已保存：", i)

