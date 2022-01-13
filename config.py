import dlib
import torch

# 通用配置项
continuous_params_size = 159    # 连续参数个数，github开源的是95个参数
# image_root = "F:/dataset/face_simple/face/"
# train_params_root = "F:/dataset/face_simple/train_param.json"
# test_params_root = "F:/dataset/face_simple/test_param.json"

# train_set = "./face_data/trainset_female"
# test_set = "./face_data/testset_female"


image_root = "F:/dataset/face_20211203_20000_nojiemao/"
params_root = "F:/dataset/face_20211203_20000_nojiemao/param.json"

use_gpu = False    # 是否使用gpu
num_gpu = 1    # gpu的个数
device = torch.device('cuda:0') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
path_tensor_log = "./logs/"


# imitator配置项
total_epochs = 500
batch_size = 16
save_freq = 10
prev_freq = 10
learning_rate = 1
# imitator_model = "./checkpoint/imitator.pth"    # 不做finetune，就直接写空字符串
imitator_model = "./checkpoint/epoch_340_0.434396.pt"
# imitator_model = ""

prev_path = "./output/preview"
# prev_path = "E:/nielian/"
model_path = "./output/imitator"

# 评估时
total_eval_steps = 50
eval_alpha = 0.1    # Ls = alpha * L1 + L2
eval_learning_rate = 1
eval_prev_freq = 1


# 人脸语义分割
faceparse_backbone = 'mobilenetv2'
faceparse_checkpoint = "./checkpoint/faceseg_179_0.050777_0.065476_0.842724_withface7.pth"
num_classes = 7
output_stride = 16
pretrained = True
progress = True
model_urls = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"    # mobilenetv2
# model_urls = "https://download.pytorch.org/models/resnet50-19c8e357.pth"    # resnet

# light-cnn
lightcnn_checkpoint = "./checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar"

# 人脸关键点检测及摆正
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./checkpoint/shape_predictor_68_face_landmarks.dat')

# 自定义imitator
config_jsonfile = "./checkpoint/myimitator-512.json"


