import torch

import config
from imitator import Imitator

imitator = Imitator("neural imitator", config)

# checkpoint = torch.load("./checkpoint/model_imitator_100000_cuda.pth", map_location=torch.device('cpu'))
# # imitator.load_state_dict(checkpoint['net'])
# for k1, k2 in zip(imitator.state_dict(), checkpoint['net']):
#     print("%s\t%s\t%s\t%s" % (k1, imitator.state_dict()[k1].shape, k2, checkpoint['net'][k2].shape))

imitator_model = torch.load(config.imitator_model, map_location=torch.device('cpu'))

op_model = {}
for k in imitator_model['net'].keys():
    op_model["model." + str(k)] = imitator_model['net'][k]

imitator.load_state_dict(op_model)
print()

# oldDict = {"tmpA": 3, "tmpB":4, "tmpC":5}
# newDict = oldDict.copy()
# oldDict.clear()
# print(oldDict)
# for k in newDict.keys():
#     k1 = k.replace("tmp", "")
#     oldDict[k1] = newDict[k]
#
# print(oldDict)
