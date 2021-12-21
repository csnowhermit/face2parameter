import os
from torch.utils.data import Dataset, DataLoader
from torchvision.models.inception import Inception3
from torchvision.models.utils import load_state_dict_from_url

mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]

class FID_Dataset(Dataset):
    def __init__(self, imgpath, transform):
        self.imgpath = imgpath
        self.fileList = [os.path.join(imgpath, file) for file in os.listdir(imgpath)]
        self.transform = transform


'''
    计算FID指标
'''
def compute_FID():
    pass


if __name__ == '__main__':
    model = Inception3()
    print(model)
    state_dict = load_state_dict_from_url("https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth", progress=True)
    model.load_state_dict(state_dict)
