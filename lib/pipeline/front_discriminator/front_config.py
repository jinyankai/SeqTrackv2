# front_discriminator_config.py

# 这个文件用于存储前端判别器相关的配置参数

# 使用的CLIP模型名称
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# CLIP模型能够处理的最小ROI（Region of Interest）尺寸
# 如果预测的BBox小于这个尺寸，将跳过CLIP验证以避免报错
MIN_ROI_SIZE = 32

# 当目标丢失后，根据KF预测位置扩展BBox的比例因子
# 大于1.0的值会包含更多上下文信息，有助于CLIP判断
BBOX_EXPANSION_FACTOR = 2.0

# CLIP打分的阈值
# 如果分数低于此阈值，则认为目标已丢失或跟错了
CLIP_SCORE_THRESHOLD = 0.25
