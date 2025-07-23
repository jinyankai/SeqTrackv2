import numpy as np
from enum import Enum
import cv2
from PIL import Image
import torch
# 确保 transformers 版本 >= 4.20.0
from transformers import CLIPProcessor, CLIPModel

# --- 定义追踪状态 ---
class TrackingStatus(Enum):
    TRACKING = "TRACKING"          # 状态良好，正在追踪
    LOST = "LOST"                  # 目标丢失 (置信度低或完全消失)
    WRONG_TARGET = "WRONG_TARGET"  # 目标错误 (可能跟上了干扰物)

class TargetDiscriminator:
    """
    一个多模态目标判别器类。
    它综合追踪器分数、几何变化、语义信息和深度信息来判断目标的追踪状态。
    这个版本使用了真实的CLIP模型和OpenCV进行图像处理。
    """
    def __init__(self, 
                 confidence_thresh=0.5, 
                 area_change_thresh=3.0, 
                 clip_similarity_thresh=0.28, 
                 depth_change_thresh_m=0.5,
                 uint8_to_meters_scale=0.1, #! 重要参数：需要您根据设备设定
                 min_roi_size=32): #! 新增：CLIP模型能处理的最小ROI尺寸
        """
        初始化判别器。

        Args:
            confidence_thresh (float): 追踪器置信度阈值。
            area_change_thresh (float): 边界框面积变化率阈值。
            clip_similarity_thresh (float): CLIP图文相似度阈值。
            depth_change_thresh_m (float): 深度跳变阈值（单位：米）。
            uint8_to_meters_scale (float): 将uint8深度值转换为米的比例因子。
            min_roi_size (int): 语义检查时，ROI的最小边长。
        """
        self.confidence_thresh = confidence_thresh
        self.area_change_thresh = area_change_thresh
        self.clip_similarity_thresh = clip_similarity_thresh
        self.depth_change_thresh_m = depth_change_thresh_m
        self.uint8_to_meters_scale = uint8_to_meters_scale
        self.min_roi_size = min_roi_size
        
        self.prev_info = None # 用于存储上一帧的信息

        # --- 初始化CLIP模型 (一次性加载) ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[判别器] 正在加载CLIP模型到 {self.device} ...")
        try:
            # 使用最基础的CLIP模型
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("[判别器] CLIP模型加载成功。")
        except Exception as e:
            print(f"[判别器] 错误: 无法加载CLIP模型。请检查网络连接和transformers库。 {e}")
            self.clip_model = None
            self.clip_processor = None

    def _check_basic_disappearance(self, current_info):
        """1. 判断消失：基础检查"""
        current_confidence = current_info["confidence"]
        if current_confidence < self.confidence_thresh:
            print(f"[判别器] 状态: 丢失. 原因: 置信度过低 ({current_confidence:.2f} < {self.confidence_thresh}).")
            return True

        if self.prev_info:
            prev_area = self._get_bbox_area(self.prev_info["bbox_xyxy"])
            current_area = self._get_bbox_area(current_info["bbox_xyxy"])
            if prev_area > 0 and current_area > 0:
                area_ratio = max(prev_area / current_area, current_area / prev_area)
                if area_ratio > self.area_change_thresh:
                    print(f"[判别器] 状态: 丢失. 原因: 面积变化剧烈 (变化率: {area_ratio:.2f}).")
                    return True
        return False

    def _check_semantic_match(self, current_info):
        """2. 判断错误：语义检查 (真实CLIP实现)"""
        if not self.clip_model:
            print("[判别器] 警告: CLIP模型未加载，跳过语义检查。")
            return True # 如果模型加载失败，则跳过检查

        try:
            # 从路径加载RGB图像
            rgb_image = cv2.imread(current_info["rgb_image_path"])
            if rgb_image is None:
                print(f"[判别器] 错误: 无法读取RGB图像路径 {current_info['rgb_image_path']}")
                return False # 图像读取失败，视为异常

            # 裁剪ROI
            x1, y1, x2, y2 = map(int, current_info["bbox_xyxy"])
            
            # ! 新增：安全检查，防止ROI过小导致报错
            roi_width = x2 - x1
            roi_height = y2 - y1
            if roi_width < self.min_roi_size or roi_height < self.min_roi_size:
                print(f"[判别器] 警告: ROI区域过小 ({roi_width}x{roi_height})，跳过语义检查。")
                return True # 尺寸过小，无法判断，默认通过，让其他检查模块决定

            roi_image_cv = rgb_image[y1:y2, x1:x2]

            # ! 新增：安全检查，防止ROI为空
            if roi_image_cv.size == 0:
                print("[判别器] 警告: 裁剪后的ROI为空，跳过语义检查。")
                return True

            # 将OpenCV图像 (BGR) 转换为PIL图像 (RGB)
            roi_image_pil = Image.fromarray(cv2.cvtColor(roi_image_cv, cv2.COLOR_BGR2RGB))
            
            # 准备文本
            prompts = [
                current_info["nlp_sentence"],
                f"a photo of a {current_info['class_name']}"
            ]
            
            # 使用CLIP模型进行处理和计算
            inputs = self.clip_processor(text=prompts, images=roi_image_pil, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            
            # 计算相似度 (logits_per_image 形状为 [1, N_prompts])
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            clip_score = probs[0][0].item() # 取第一个prompt（即详细描述）的概率作为分数

            print(f"[判别器] CLIP图文匹配分数: {clip_score:.4f}")

            if clip_score < self.clip_similarity_thresh:
                print(f"[判别器] 状态: 错误目标. 原因: 图文不匹配 (CLIP分数: {clip_score:.2f} < {self.clip_similarity_thresh}).")
                return False
            return True

        except Exception as e:
            print(f"[判别器] 错误: 在语义检查过程中发生异常: {e}")
            return False # 出现异常时，保守地认为可能出错

    def _get_representative_depth(self, depth_map_path, bbox_xyxy):
        """从深度图中提取代表性深度值（米）"""
        try:
            # 以单通道形式读取uint8深度图
            depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
            if depth_map is None:
                print(f"[判别器] 错误: 无法读取深度图路径 {depth_map_path}")
                return None
            
            # 裁剪ROI
            x1, y1, x2, y2 = map(int, bbox_xyxy)
            depth_roi = depth_map[y1:y2, x1:x2]

            # 提取所有非零的深度值
            non_zero_depths = depth_roi[depth_roi > 0]
            
            if non_zero_depths.size == 0:
                print("[判别器] 警告: 深度ROI内没有有效的深度值。")
                return None # 没有有效值

            # 计算中位数并转换为米
            median_uint8 = np.median(non_zero_depths)
            median_meters = median_uint8 * self.uint8_to_meters_scale
            return median_meters

        except Exception as e:
            print(f"[判别器] 错误: 在提取深度过程中发生异常: {e}")
            return None

    def _check_depth_consistency(self, current_info):
        """3. 判断错误：深度检查 (真实OpenCV实现)"""
        if not self.prev_info:
            return True # 没有上一帧，跳过

        prev_depth = self._get_representative_depth(
            self.prev_info["depth_map_path"], self.prev_info["bbox_xyxy"]
        )
        current_depth = self._get_representative_depth(
            current_info["depth_map_path"], current_info["bbox_xyxy"]
        )

        if prev_depth is None or current_depth is None:
            print("[判别器] 警告: 无法获取有效的深度值，跳过深度一致性检查。")
            return True # 如果无法获取深度，则跳过检查

        depth_diff = abs(current_depth - prev_depth)
        print(f"[判别器] 深度变化: {depth_diff:.2f}m (上一帧: {prev_depth:.2f}m, 当前帧: {current_depth:.2f}m)")
        
        if depth_diff > self.depth_change_thresh_m:
            print(f"[判别器] 状态: 错误目标. 原因: 深度发生剧烈跳变 ({depth_diff:.2f}m > {self.depth_change_thresh_m}m).")
            return False
        return True

    def check_status(self, current_info):
        """主检查函数"""
        # 阶段一：基础消失判断
        if self._check_basic_disappearance(current_info):
            self.prev_info = current_info # 即使丢失，也更新信息以便下一帧比较
            return TrackingStatus.LOST

        # 阶段二：语义错误判断
        if not self._check_semantic_match(current_info):
            self.prev_info = current_info
            return TrackingStatus.WRONG_TARGET

        # 阶段三：深度跳变判断
        if not self._check_depth_consistency(current_info):
            self.prev_info = current_info
            return TrackingStatus.WRONG_TARGET

        # 如果所有检查都通过
        self.prev_info = current_info
        print("[判别器] 状态: 追踪中. 所有检查通过。")
        return TrackingStatus.TRACKING

    def _get_bbox_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

# --- Example Usage ---
def create_dummy_data():
    """创建用于测试的虚拟图像和深度图"""
    rgb_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(rgb_img, (100, 100), (150, 200), (0, 0, 255), -1)
    cv2.imwrite("frame1.png", rgb_img)
    rgb_img_distractor = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(rgb_img_distractor, (100, 100), (150, 200), (255, 0, 0), -1)
    cv2.imwrite("frame_distractor.png", rgb_img_distractor)
    depth_map1 = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(depth_map1, (100, 100), (150, 200), 50, -1)
    cv2.imwrite("depth1.png", depth_map1)
    depth_map2 = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(depth_map2, (100, 100), (150, 200), 80, -1)
    cv2.imwrite("depth2.png", depth_map2)

if __name__ == '__main__':
    create_dummy_data()
    discriminator = TargetDiscriminator(uint8_to_meters_scale=0.1)

    # --- 场景1: 正常追踪 ---
    print("\n--- Frame 1 (正常追踪) ---")
    frame1_info = {
        "bbox_xyxy": [100, 100, 150, 200], "confidence": 0.98,
        "nlp_sentence": "the person wearing a red hat", "class_name": "person",
        "rgb_image_path": "frame1.png", "depth_map_path": "depth1.png"
    }
    status = discriminator.check_status(frame1_info)
    print(f"最终状态: {status}\n")

    # --- 场景2: ROI过小 ---
    print("--- Frame 2 (ROI过小 -> 跳过语义检查) ---")
    frame2_info = frame1_info.copy()
    frame2_info["bbox_xyxy"] = [100, 100, 110, 110] # 一个20x20的小框
    status = discriminator.check_status(frame2_info)
    print(f"最终状态: {status}\n") # 应该会通过，因为其他检查项都正常

    # --- 场景3: 语义不匹配 ---
    print("--- Frame 3 (语义不匹配 -> 错误目标) ---")
    frame3_info = frame1_info.copy()
    frame3_info["rgb_image_path"] = "frame_distractor.png"
    status = discriminator.check_status(frame3_info)
    print(f"最终状态: {status}\n")
