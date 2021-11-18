import cv2
import torch
import utils
import config
from imitator import Imitator

imitator = Imitator()

# model_dict = {"model.0.weight": "0.0.weight",
#               "model.0.bias": "0.0.bias",
#               "model.1.weight": "0.1.weight",
#               "model.1.bias": "0.1.bias",
#               "model.1.running_mean": "0.1.running_mean",
#               "model.1.running_var": "0.1.running_var",
#               "model.1.num_batches_tracked": "0.1.num_batches_tracked",
#               "model.3.weight": "1.0.weight",
#               "model.3.bias": "1.0.bias",
#               "model.4.weight": "1.1.weight",
#               "model.4.bias": "1.1.bias",
#               "model.4.running_mean": "1.1.running_mean",
#               "model.4.running_var": "1.1.running_var",
#               "model.4.num_batches_tracked": "1.1.num_batches_tracked",
#               "model.6.weight": "2.0.weight",
#               "model.6.bias": "2.0.bias",
#               "model.7.weight": "2.1.weight",
#               "model.7.bias": "2.1.bias",
#               "model.7.running_mean": "2.1.running_mean",
#               "model.7.running_var": "2.1.running_var",
#               "model.7.num_batches_tracked": "2.1.num_batches_tracked",
#               "model.9.weight": "3.0.weight",
#               "model.9.bias": "3.0.bias",
#               "model.10.weight": "3.1.weight",
#               "model.10.bias": "3.1.bias",
#               "model.10.running_mean": "3.1.running_mean",
#               "model.10.running_var": "3.1.running_var",
#               "model.10.num_batches_tracked": "3.1.num_batches_tracked",
#               "model.12.weight": "4.0.weight",
#               "model.12.bias": "4.0.bias",
#               "model.13.weight": "4.1.weight",
#               "model.13.bias": "4.1.bias",
#               "model.13.running_mean": "4.1.running_mean",
#               "model.13.running_var": "4.1.running_var",
#               "model.13.num_batches_tracked": "4.1.num_batches_tracked",
#               "model.15.weight": "5.0.weight",
#               "model.15.bias": "5.0.bias",
#               "model.16.weight": "5.1.weight",
#               "model.16.bias": "5.1.bias",
#               "model.16.running_mean": "5.1.running_mean",
#               "model.16.running_var": "5.1.running_var",
#               "model.16.num_batches_tracked": "5.1.num_batches_tracked",
#               "model.18.weight": "6.0.weight",
#               "model.18.bias": "6.0.bias",
#               "model.19.weight": "6.1.weight",
#               "model.19.bias": "6.1.bias",
#               "model.19.running_mean": "6.1.running_mean",
#               "model.19.running_var": "6.1.running_var",
#               "model.19.num_batches_tracked": "6.1.num_batches_tracked",
#               "model.21.weight": "7.weight",
#               "model.21.bias": "7.bias"
#             }
#
# checkpoint = torch.load("./checkpoint/model_imitator_100000_cuda.pth", map_location=torch.device('cpu'))
# # imitator.load_state_dict(checkpoint['net'])
# for k1, k2 in zip(imitator.state_dict(), checkpoint['net']):
#     # print("%s\t%s\t%s\t%s" % (k1, imitator.state_dict()[k1].shape, k2, checkpoint['net'][k2].shape))
#     content = '"%s": "%s",' % (k1, k2)
#     print(content)
#
# state_dict = torch.load(config.imitator_model, map_location=torch.device('cpu'))['net']
# #
# # op_model = {}
# # for k in state_dict['net'].keys():
# #     op_model["model." + str(k)[2:]] = imitator_model['net'][k]
# for new_key, old_key in model_dict.items():
#     state_dict[new_key] = state_dict.pop(old_key)
#
# imitator.load_state_dict(state_dict)
# torch.save(state_dict, "./imitator.pth")


model = torch.load(config.imitator_model)
imitator.load_state_dict(model)
