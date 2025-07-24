# KF_tracker_config.py

# 这个文件用于存储卡尔曼滤波器相关的配置参数

# 状态向量的维度 (x, y, z, vx, vy, vz)
STATE_DIM = 6

# 测量向量的维度 (x, y, z)
MEASUREMENT_DIM = 3

# 过程噪声协方差
# 控制模型对真实世界不确定性的估计。值越大，代表模型认为物体的运动越不稳定。
PROCESS_NOISE_COV = 1e-3

# 测量噪声协方差
# 控制模型对测量值（来自检测器）不确定性的估计。值越大，代表模型认为测量值越不可靠。
MEASUREMENT_NOISE_COV = 1e-1

# 初始状态协方差
# 控制模型对初始状态不确定性的估计。一个较大的值表示对初始状态的估计非常不确定。
INITIAL_STATE_COVARIANCE = 10.0
