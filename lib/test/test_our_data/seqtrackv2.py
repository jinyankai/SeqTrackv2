import random
import glob
import cv2
import torch
import sys
import time
import shutil
import numpy as np
import math

# Assume these imports are from your project structure
# We will use mock objects for these in the __main__ block
sys.path.append('../..')  # Adjust the path as necessary
from lib.test.evaluation import Tracker
from lib.train.dataset.depth_utils import get_rgbd_frame
from lib.utils.box_ops import box_cxcywh_to_xyxy

# ====================================================================================
# Part 0: Evaluation Functions
# ====================================================================================

def calculate_iou(box_a, box_b):
    """计算两个边界框的IoU (Intersection over Union)。格式: [x, y, w, h]。"""
    x1_a, y1_a, w_a, h_a = box_a
    x2_a, y2_a = x1_a + w_a, y1_a + h_a
    x1_b, y1_b, w_b, h_b = box_b
    x2_b, y2_b = x1_b + w_b, y1_b + h_b
    x1_inter, y1_inter = max(x1_a, x1_b), max(y1_a, y1_b)
    x2_inter, y2_inter = min(x2_a, x2_b), min(y2_a, y2_b)
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    area_a, area_b = w_a * h_a, w_b * h_b
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_auc(pred_boxes, gt_boxes, thresholds=np.arange(0.0, 1.05, 0.05)):
    """根据预测框和真实框计算成功率曲线下的面积 (AUC)。"""
    num_frames = len(pred_boxes)
    if num_frames == 0: return 0.0
    ious = [calculate_iou(pred, gt) for pred, gt in zip(pred_boxes, gt_boxes)]
    success_rates = [np.mean([1 if iou >= th else 0 for iou in ious]) for th in thresholds]
    return np.mean(success_rates)

# ====================================================================================
# Part 1: Enhanced TestDataset Class
# ====================================================================================
class TestDataset:
    def __init__(self, root_dir):
        if not os.path.isdir(root_dir): raise ValueError(f"提供的根目录不存在: {root_dir}")
        self.root_dir = root_dir
        self.categories = sorted([cat for cat in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, cat))])
        print(f"成功初始化数据集，找到 {len(self.categories)} 个类别: {self.categories}")

    def _parse_bbox(self, bbox_str):
        try: return [float(coord) for coord in bbox_str.strip().split(',')]
        except: return None

    def get_all_categories(self): return self.categories

    def get_sequence(self, category_name):
        if category_name not in self.categories: return None
        category_path = os.path.join(self.root_dir, category_name)
        color_dir, depth_dir = os.path.join(category_path, 'color'), os.path.join(category_path, 'depth')
        gt_path, nlp_path = os.path.join(category_path, 'groundtruth_rect.txt'), os.path.join(category_path, 'nlp.txt')
        if not all([os.path.isdir(color_dir), os.path.isdir(depth_dir), os.path.exists(gt_path), os.path.exists(nlp_path)]): return None
        try:
            color_images, depth_images = sorted(glob.glob(os.path.join(color_dir, '*'))), sorted(glob.glob(os.path.join(depth_dir, '*')))
            with open(gt_path, 'r') as f: gt_bboxes_str = f.readlines()
            with open(nlp_path, 'r') as f: nlp_text = f.read().strip()
            if not (len(color_images) == len(depth_images) == len(gt_bboxes_str)): return None
            gt_bboxes = [self._parse_bbox(line) for line in gt_bboxes_str]
            if any(b is None for b in gt_bboxes): return None
            return {'color_images': color_images, 'depth_images': depth_images, 'gt_bboxes': gt_bboxes, 'nlp': nlp_text}
        except: return None

# ====================================================================================
# Part 2: Tracker Wrapper Class
# ====================================================================================
class SeqTrackV2Wrapper:
    def __init__(self, tracker_name='seqtrackv2', para_name=''):
        tracker_info = Tracker(tracker_name, para_name, "MY_DATA", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def initialize(self, img_rgbd, bbox, nlp_sentence):
        init_info = {'init_bbox': bbox, 'init_nlp': nlp_sentence if nlp_sentence else ""}
        _ = self.tracker.initialize(img_rgbd, init_info)

    def track(self, img_rgbd):
        outputs = self.tracker.track(img_rgbd)
        return outputs['target_bbox'], outputs['best_score']

# ====================================================================================
# Part 3: Main Execution Logic with Periodic Re-initialization
# ====================================================================================
def run_tracker_on_dataset(tracker_name, para_name, data_root, output_dir, vis=True, reset_interval=100):
    torch.set_num_threads(1)
    output_txt_dir, output_img_dir = os.path.join(output_dir, 'output_txt'), os.path.join(output_dir, 'output_images')
    os.makedirs(output_txt_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    
    dataset = TestDataset(root_dir=data_root)
    tracker = SeqTrackV2Wrapper(tracker_name=tracker_name, para_name=para_name)
    all_aucs = []
    
    for category in dataset.get_all_categories():
        print(f"\n{'='*20} 开始处理类别: {category} {'='*20}")
        sequence_data = dataset.get_sequence(category)
        if not sequence_data or len(sequence_data['color_images']) < 2: continue

        sequence_pred_boxes, sequence_gt_boxes = [], []
        output_txt_path = os.path.join(output_txt_dir, f'{category}_output.txt')
        print(f"结果将写入: {output_txt_path}")
        
        with open(output_txt_path, 'w') as f:
            # --- Frame 1: Initialization with Ground Truth ---
            last_init_frame_idx = 0
            gt_bbox_first_frame = sequence_data['gt_bboxes'][0]
            f.write(','.join(map(str, gt_bbox_first_frame)) + '\n')
            sequence_pred_boxes.append(gt_bbox_first_frame)
            sequence_gt_boxes.append(gt_bbox_first_frame)
            
            # Store previous frame state (initialized with frame 1 GT)
            prev_image_rgbd = get_rgbd_frame(sequence_data['color_images'][0], sequence_data['depth_images'][0])
            prev_pred_bbox = gt_bbox_first_frame
            
            tracker.initialize(prev_image_rgbd, prev_pred_bbox, sequence_data['nlp'])
            print("跟踪器在第1帧初始化完成。")

            # --- Subsequent Frames: Tracking with Periodic Reset Logic ---
            for frame_idx in range(1, len(sequence_data['color_images'])):
                was_reset = False
                current_image_rgbd = get_rgbd_frame(sequence_data['color_images'][frame_idx], sequence_data['depth_images'][frame_idx])
                
                # Step 1: Check if reset interval is reached
                if (frame_idx - last_init_frame_idx) > reset_interval:
                    was_reset = True
                    print(f"---! 帧 {frame_idx+1}: 达到重置间隔({reset_interval}帧)。正在使用上一帧的预测结果重置跟踪器 !---")
                    
                    # Re-initialize tracker with the PREVIOUS frame's PREDICTION
                    tracker.initialize(prev_image_rgbd, prev_pred_bbox, sequence_data['nlp'])
                    
                    # Update the last init frame index
                    last_init_frame_idx = frame_idx - 1

                # Step 2: Track on the current frame
                pred_bbox, best_score = tracker.track(current_image_rgbd)
                
                # Record final results for this frame
                f.write(','.join(map(str, pred_bbox)) + '\n')
                sequence_pred_boxes.append(pred_bbox)
                sequence_gt_boxes.append(sequence_data['gt_bboxes'][frame_idx])
                
                # Step 3: Update previous state for the next iteration
                prev_image_rgbd = current_image_rgbd
                prev_pred_bbox = pred_bbox
                
                # --- Visualization ---
                if vis and ((frame_idx + 1) % 10 == 0 or was_reset):
                    vis_image = cv2.imread(sequence_data['color_images'][frame_idx])
                    # Draw GT box (Green)
                    x_gt, y_gt, w_gt, h_gt = [int(v) for v in sequence_data['gt_bboxes'][frame_idx]]
                    cv2.rectangle(vis_image, (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), (0, 255, 0), 2)
                    
                    # Draw Prediction box (Yellow if reset, Red otherwise)
                    x, y, w, h = [int(v) for v in pred_bbox]
                    pred_color = (0, 255, 255) if was_reset else (0, 0, 255) # BGR: Yellow or Red
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), pred_color, 2)
                    
                    # Save image
                    vis_category_dir = os.path.join(output_img_dir, category)
                    os.makedirs(vis_category_dir, exist_ok=True)
                    save_path = os.path.join(vis_category_dir, f"{frame_idx+1:08d}_bbox.jpg")
                    cv2.imwrite(save_path, vis_image)

        # --- AUC Calculation for the sequence ---
        category_auc = calculate_auc(sequence_pred_boxes, sequence_gt_boxes)
        all_aucs.append(category_auc)
        print(f"\n{'*'*10} 类别 '{category}' 评估结果 {'*'*10}\nAUC 得分: {category_auc:.4f}\n{'*'*40}")

    # --- Final Average AUC ---
    if all_aucs:
        print(f"\n{'='*20} 数据集总体评估结果 {'='*20}\n处理的总类别数: {len(all_aucs)}\n平均 AUC 得分: {np.mean(all_aucs):.4f}\n{'='*60}")
