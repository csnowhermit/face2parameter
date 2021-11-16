
# 通用配置项
params_cnt = 99
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
imitator_model = "./checkpoint/model_imitator_400000_compete_cuda.pth"    # 不做finetune，就直接写空字符串

init_step = 0

prev_path = "./output/preview"
model_path = "./output/imitator"

# 评估时
total_eval_steps = 1000
eval_alpha = 0.01    # Ls = alpha * L1 + L2
eval_learning_rate = 10**10
eval_prev_freq = 50


# 人脸语义分割
faceparse_checkpoint = "./checkpoint/79999_iter.pth"

# light-cnn
lightcnn_checkpoint = "./checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar"
