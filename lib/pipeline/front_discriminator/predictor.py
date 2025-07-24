import cv2
from PIL import Image
from lib.pipeline.KF_tracker.kf_tracker_manager import *
from lib.pipeline.front_discriminator.clip_scorer import clip_scorer
from lib.pipeline.front_discriminator.front_config import *

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


class Predictor:
    """
    作为前端判别器的核心调度器。
    管理来自SeqTrackV2的BBox，更新KF追踪器，并在目标丢失时进行预测和验证。
    """

    def __init__(self):
        self.kf_manager = kf_tracker_manager()
        self.clip_scorer = clip_scorer()

    def _get_xyz_from_detection(self, bbox_xywh, depth_map):
        """
        辅助函数：从检测框和深度图中估算目标的3D中心点。
        这是一个简化的实现。
        """
        x, y, w, h = bbox_xywh
        center_x = x + w / 2
        center_y = y + h / 2

        # 从深度图中获取深度值 (z)
        # 实际应用中，应取ROI内深度的中位数或均值以提高鲁棒性
        # 这里简化为取中心点的深度
        try:
            depth_z = depth_map[int(center_y), int(center_x)]
            # 假设深度图单位是毫米，转换为米
            depth_z_meters = depth_z / 1000.0
        except IndexError:
            depth_z_meters = 0  # 如果越界，则深度为0

        return [center_x, center_y, depth_z_meters]

    def process_detection(self, bbox_xywh, depth_map):
        """
        当SeqTrackV2给出有效检测时调用此方法。
        Args:
            bbox_xywh (list): [x, y, w, h] 格式的检测框。
            depth_map (np.ndarray): 当前帧的深度图。
        """
        xyz = self._get_xyz_from_detection(bbox_xywh, depth_map)
        self.kf_manager.process_update(xyz)
        print("[Predictor] 已处理新的检测结果并更新KF。")

    def predict_and_verify(self, next_image_path, class_name):
        """
        当SeqTrackV2丢失目标时调用此方法。
        它会使用KF进行预测，并使用CLIP进行验证。

        Args:
            next_image_path (str): 下一帧的RGB图像路径。
            class_name (str): 目标的类别名称，用于CLIP验证。

        Returns:
            tuple: (is_found, predicted_bbox)
                   is_found (bool): 是否认为目标被找回。
                   predicted_bbox (list): 预测的BBox [x1, y1, x2, y2]。
        """
        print("[Predictor] 目标丢失，启动预测与验证流程...")

        # 1. 使用KF预测BBox
        image = cv2.imread(next_image_path)
        if image is None:
            print(f"[Predictor] 错误: 无法读取图像 {next_image_path}")
            return False, None

        predicted_bbox = self.kf_manager.get_predicted_bbox(image.shape[:2])
        if predicted_bbox is None:
            print("[Predictor] KF未初始化，无法预测。")
            return False, None

        # 2. 扩展BBox并检查尺寸
        x1, y1, x2, y2 = predicted_bbox
        w, h = x2 - x1, y2 - y1

        center_x, center_y = x1 + w / 2, y1 + h / 2
        new_w = w * BBOX_EXPANSION_FACTOR
        new_h = h * BBOX_EXPANSION_FACTOR

        ex_x1 = int(center_x - new_w / 2)
        ex_y1 = int(center_y - new_h / 2)
        ex_x2 = int(center_x + new_w / 2)
        ex_y2 = int(center_y + new_h / 2)

        if (ex_x2 - ex_x1) < MIN_ROI_SIZE or (ex_y2 - ex_y1) < MIN_ROI_SIZE:
            print(f"[Predictor] 警告: 预测的ROI过小，跳过验证。")
            return False, predicted_bbox

        # 3. 裁剪ROI并送入CLIP打分
        roi_cv = image[ex_y1:ex_y2, ex_x1:ex_x2]
        if roi_cv.size == 0:
            print("[Predictor] 警告: 裁剪的ROI为空。")
            return False, predicted_bbox

        roi_pil = Image.fromarray(cv2.cvtColor(roi_cv, cv2.COLOR_BGR2RGB))

        score = self.clip_scorer.score(roi_pil, class_name)

        # 4. 根据分数判断是否找回目标
        if score >= CLIP_SCORE_THRESHOLD:
            print(f"[Predictor] 验证成功！分数 {score:.2f} >= {CLIP_SCORE_THRESHOLD}。认为目标已找回。")
            # 找到后，可以用这个预测的BBox的中心点来更新KF
            # (这是一个简化的更新，更精确的做法是运行一个检测器来精确定位)
            # self.kf_manager.process_update([center_x, center_y, self.kf_manager.tracker.kf.x[2]])
            return True, predicted_bbox
        else:
            print(f"[Predictor] 验证失败。分数 {score:.2f} < {CLIP_SCORE_THRESHOLD}。认为目标仍丢失。")
            # 连续多次失败后，可以考虑终止KF
            # self.kf_manager.kill()
            return False, predicted_bbox
