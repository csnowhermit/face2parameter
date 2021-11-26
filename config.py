import dlib

# 通用配置项
continuous_params_size = 95    # 连续参数个数
train_set = "./face_data/trainset_female"
test_set = "./face_data/testset_female"
use_gpu = False
path_tensor_log = "./logs/"


# imitator配置项
total_steps = 600000
batch_size = 1
save_freq = 10000
prev_freq = 1000
learning_rate = 0.1
imitator_model = "./checkpoint/imitator_gitopen.pth"    # 不做finetune，就直接写空字符串

init_step = 0

prev_path = "./output/preview"
model_path = "./output/imitator"

# 评估时
total_eval_steps = 1000
eval_alpha = 0.01    # Ls = alpha * L1 + L2
eval_learning_rate = 0.1
eval_prev_freq = 50


# 人脸语义分割
faceparse_checkpoint = "./checkpoint/79999_iter.pth"

# light-cnn
lightcnn_checkpoint = "./checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar"

# 人脸关键点检测及摆正
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./checkpoint/shape_predictor_68_face_landmarks.dat')