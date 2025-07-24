import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from lib.pipeline.front_discriminator.front_config import *
from lib.pipeline.KF_tracker.kf_tracker_manager import kf_tracker_manager

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

class clip_scorer:
    """
    封装CLIP模型，用于计算图像ROI和文本描述的匹配分数。
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        try:
            print(f"[CLIP Scorer] 正在加载模型 {CLIP_MODEL_NAME} 到 {self.device}...")
            self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            print("[CLIP Scorer] 模型加载成功。")
        except Exception as e:
            print(f"[CLIP Scorer] 错误: 无法加载CLIP模型。 {e}")

    def score(self, image_roi, text):
        """
        计算图像ROI与文本的匹配分数。
        Args:
            image_roi (PIL.Image): 从图像中裁剪出的ROI区域。
            text (str): 文本描述，例如类别名称。
        Returns:
            float: 匹配分数 (0到1之间)。
        """
        if not self.model:
            print("[CLIP Scorer] 警告: 模型未加载，无法评分。")
            return 0.0

        try:
            inputs = self.processor(text=[text], images=image_roi, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            score = probs[0, 0].item()
            print(f"[CLIP Scorer] 文本 '{text}' 与图像的匹配分数为: {score:.4f}")
            return score
        except Exception as e:
            print(f"[CLIP Scorer] 错误: 评分过程中发生异常: {e}")
            return 0.0



