import dlib
import torch

# 通用配置项
continuous_params_size = 223    # 连续参数个数，github开源的是95个参数

image_root = "F:/dataset/face_2021_1130_20000_0.2-0.8/face/"
train_params_root = "F:/dataset/face_2021_1130_20000_0.2-0.8/train_param.json"
test_params_root = "F:/dataset/face_2021_1130_20000_0.2-0.8/test_param.json"

use_gpu = False    # 是否使用gpu
num_gpu = 1    # gpu的个数
device = torch.device('cuda:0') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
path_tensor_log = "./logs/"


# imitator配置项
total_steps = 500
batch_size = 16
save_freq = 50
prev_freq = 10
learning_rate = 0.1
# imitator_model = "./checkpoint/imitator.pth"    # 不做finetune，就直接写空字符串
imitator_model = ""

init_step = 0

prev_path = "./output/preview"
model_path = "./output/imitator"

# 评估时
total_eval_steps = 1000
eval_alpha = 0.01    # Ls = alpha * L1 + L2
eval_learning_rate = 1
eval_prev_freq = 50


# 人脸语义分割
faceparse_checkpoint = "./checkpoint/79999_iter.pth"

# light-cnn
lightcnn_checkpoint = "./checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar"

# 人脸关键点检测及摆正
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./checkpoint/shape_predictor_68_face_landmarks.dat')