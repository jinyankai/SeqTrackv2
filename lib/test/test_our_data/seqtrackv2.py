import os
import random
import glob
import cv2
import torch
import sys
import time
import shutil
import numpy as np
from lib.test.evaluation import Tracker
from lib.train.dataset.depth_utils import get_rgbd_frame
from lib.utils.box_ops import box_cxcywh_to_xyxy

# ====================================================================================
# Part 0: Evaluation Functions 
# ====================================================================================

def calculate_iou(box_a, box_b):
    """
    计算两个边界框的IoU (Intersection over Union)。
    边界框格式为 [x, y, w, h]。

    Args:
        box_a (list or np.array): 第一个边界框。
        box_b (list or np.array): 第二个边界框。

    Returns:
        float: 两个边界框的IoU值。
    """
    # 转换为 [x1, y1, x2, y2] 格式
    x1_a, y1_a, w_a, h_a = box_a
    x2_a, y2_a = x1_a + w_a, y1_a + h_a
    
    x1_b, y1_b, w_b, h_b = box_b
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    # 计算交集矩形的坐标
    x1_inter = max(x1_a, x1_b)
    y1_inter = max(y1_a, y1_b)
    x2_inter = min(x2_a, x2_b)
    y2_inter = min(y2_a, y2_b)

    # 计算交集面积
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # 计算并集面积
    area_a = w_a * h_a
    area_b = w_b * h_b
    union_area = area_a + area_b - inter_area

    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_auc(pred_boxes, gt_boxes, thresholds=np.arange(0.0, 1.05, 0.05)):
    """
    根据预测框和真实框计算成功率曲线下的面积 (AUC)。

    Args:
        pred_boxes (list of lists): 预测的边界框列表。
        gt_boxes (list of lists): 真实的边界框列表。
        thresholds (np.array): 用于计算成功率的IoU阈值。

    Returns:
        float: AUC得分。
    """
    if len(pred_boxes) != len(gt_boxes):
        raise ValueError("预测框和真实框的数量必须相等。")

    num_frames = len(pred_boxes)
    if num_frames == 0:
        return 0.0

    ious = [calculate_iou(pred, gt) for pred, gt in zip(pred_boxes, gt_boxes)]
    
    success_rates = []
    for th in thresholds:
        success_count = sum(1 for iou in ious if iou >= th)
        success_rates.append(success_count / num_frames)
        
    # AUC是所有成功率的平均值
    auc = np.mean(success_rates)
    return auc


# ====================================================================================
# Part 1: Enhanced TestDataset Class
# ====================================================================================

class TestDataset:
    """
    用于加载和采样特定结构的测试数据集的类。
    """
    def __init__(self, root_dir):
        if not os.path.isdir(root_dir):
            raise ValueError(f"提供的根目录不存在: {root_dir}")
        self.root_dir = root_dir
        self.categories = sorted([cat for cat in os.listdir(root_dir) 
                                  if os.path.isdir(os.path.join(root_dir, cat))])
        if not self.categories:
            print(f"警告: 在目录 {root_dir} 中没有找到任何类别文件夹。")
        else:
            print(f"成功初始化数据集，找到 {len(self.categories)} 个类别: {self.categories}")

    def _parse_bbox(self, bbox_str):
        try:
            return [float(coord) for coord in bbox_str.strip().split(',')]
        except (ValueError, AttributeError):
            try:
                return [float(coord) for coord in bbox_str.strip().split()]
            except Exception as e:
                print(f"错误: 无法解析bbox字符串 '{bbox_str}'. 错误: {e}")
                return None

    def get_all_categories(self):
        return self.categories

    def get_sequence(self, category_name):
        if category_name not in self.categories:
            print(f"错误: 类别 '{category_name}' 不在数据集中。")
            return None
        category_path = os.path.join(self.root_dir, category_name)
        color_dir = os.path.join(category_path, 'color')
        depth_dir = os.path.join(category_path, 'depth')
        gt_path = os.path.join(category_path, 'groundtruth_rect.txt')
        nlp_path = os.path.join(category_path, 'nlp.txt')

        if not all([os.path.isdir(color_dir), os.path.isdir(depth_dir), os.path.exists(gt_path), os.path.exists(nlp_path)]):
            print(f"警告: 类别 '{category_name}' 数据不完整，跳过。")
            return None
        try:
            color_images = sorted(glob.glob(os.path.join(color_dir, '*')))
            depth_images = sorted(glob.glob(os.path.join(depth_dir, '*')))
            with open(gt_path, 'r') as f:
                gt_bboxes_str = f.readlines()
            with open(nlp_path, 'r') as f:
                nlp_text = f.read().strip()
            num_frames = len(color_images)
            if not (num_frames == len(depth_images) and num_frames == len(gt_bboxes_str)):
                print(f"警告: 类别 '{category_name}' 的数据文件数量不匹配，跳过。")
                return None
            gt_bboxes = [self._parse_bbox(line) for line in gt_bboxes_str]
            if any(bbox is None for bbox in gt_bboxes):
                print(f"警告: 类别 '{category_name}' 的 groundtruth 文件中存在格式错误的行，跳过。")
                return None
            return {
                'color_images': color_images,
                'depth_images': depth_images,
                'gt_bboxes': gt_bboxes,
                'nlp': nlp_text,
                'category': category_name
            }
        except Exception as e:
            print(f"错误: 处理类别 '{category_name}' 时发生异常: {e}，跳过。")
            return None

# ====================================================================================
# Part 2: Tracker Wrapper Class
# ====================================================================================

class SeqTrackV2Wrapper(object):
    def __init__(self, tracker_name='seqtrackv2', para_name=''):
        tracker_info = Tracker(tracker_name, para_name, "MY_DATA", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def initialize(self, img_rgbd, bbox, nlp_sentence):
        init_info = {'init_bbox': bbox}
        if nlp_sentence:
            init_info['init_nlp'] = nlp_sentence
        _ = self.tracker.initialize(img_rgbd, init_info)
        print("跟踪器初始化完成。")

    def track(self, img_rgbd):
        outputs = self.tracker.track(img_rgbd)
        pred_bbox = outputs['target_bbox']
        max_score = outputs['best_score']
        return pred_bbox, max_score

# ====================================================================================
# Part 3: Main Execution Logic
# ====================================================================================

def run_tracker_on_dataset(tracker_name, para_name, data_root, output_dir, vis=True):
    torch.set_num_threads(1)
    
    output_txt_dir = os.path.join(output_dir, 'output_txt')
    output_img_dir = os.path.join(output_dir, 'output_images')
    os.makedirs(output_txt_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    
    dataset = TestDataset(root_dir=data_root)
    tracker = SeqTrackV2Wrapper(tracker_name=tracker_name, para_name=para_name)
    
    all_aucs = []
    
    for category in dataset.get_all_categories():
        print(f"\n{'='*20} 开始处理类别: {category} {'='*20}")
        
        sequence_data = dataset.get_sequence(category)
        if not sequence_data:
            continue
        if len(sequence_data['color_images']) < 1:
            print(f"类别 '{category}' 没有足够的帧，跳过。")
            continue

        # 用于存储当前序列的所有结果以计算AUC
        sequence_pred_boxes = []
        sequence_gt_boxes = []

        output_txt_path = os.path.join(output_txt_dir, 'output.txt')
        print(f"结果将写入: {output_txt_path}")
        
        with open(output_txt_path, 'w') as f:
            # --- 处理第一帧 (初始化) ---
            first_frame_idx = 0
            gt_bbox_first_frame = sequence_data['gt_bboxes'][first_frame_idx]
            
            # 写入第一帧的真实GT
            f.write(','.join(map(str, gt_bbox_first_frame)) + '\n')
            
            # 记录第一帧结果 (预测=真实)
            sequence_pred_boxes.append(gt_bbox_first_frame)
            sequence_gt_boxes.append(gt_bbox_first_frame)
            
            # 加载图像并初始化跟踪器
            image_rgbd = get_rgbd_frame(sequence_data['color_images'][first_frame_idx], sequence_data['depth_images'][first_frame_idx])
            tracker.initialize(image_rgbd, gt_bbox_first_frame, sequence_data['nlp'])

            # --- 循环处理后续帧 (跟踪) ---
            for frame_idx in range(1, len(sequence_data['color_images'])):
                image_rgbd = get_rgbd_frame(sequence_data['color_images'][frame_idx], sequence_data['depth_images'][frame_idx])
                
                start_time = time.time()
                pred_bbox, best_score = tracker.track(image_rgbd)
                elapsed_time = time.time() - start_time
                
                print(f"帧 {frame_idx+1}/{len(sequence_data['color_images'])}: "
                      f"预测BBox={pred_bbox}, Score={best_score:.4f}, "
                      f"耗时={elapsed_time:.4f}s")
                
                f.write(','.join(map(str, pred_bbox)) + '\n')
                
                # 记录结果
                sequence_pred_boxes.append(pred_bbox)
                sequence_gt_boxes.append(sequence_data['gt_bboxes'][frame_idx])
                
                # --- 可视化 ---
                if vis and (frame_idx + 1) % 10 == 0:
                    vis_image = cv2.imread(sequence_data['color_images'][frame_idx])
                    x, y, w, h = [int(v) for v in pred_bbox]
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    gt_bbox = sequence_data['gt_bboxes'][frame_idx]
                    x_gt, y_gt, w_gt, h_gt = [int(v) for v in gt_bbox]
                    cv2.rectangle(vis_image, (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), (0, 255, 0), 2)
                    vis_category_dir = os.path.join(output_img_dir, category)
                    os.makedirs(vis_category_dir, exist_ok=True)
                    base_filename = os.path.basename(sequence_data['color_images'][frame_idx])
                    save_filename = base_filename.rsplit('.', 1)[0] + '_bbox.jpg'
                    save_path = os.path.join(vis_category_dir, save_filename)
                    cv2.imwrite(save_path, vis_image)
                    print(f"可视化结果已保存至: {save_path}")

        # --- 计算并打印当前类别的AUC ---
        category_auc = calculate_auc(sequence_pred_boxes, sequence_gt_boxes)
        all_aucs.append(category_auc)
        print(f"\n{'*'*10} 类别 '{category}' 评估结果 {'*'*10}")
        print(f"AUC 得分: {category_auc:.4f}")
        print(f"{'*'*40}")

    # --- 计算并打印所有类别的平均AUC ---
    if all_aucs:
        mean_auc = np.mean(all_aucs)
        print(f"\n{'='*20} 数据集总体评估结果 {'='*20}")
        print(f"处理的总类别数: {len(all_aucs)}")
        print(f"平均 AUC 得分: {mean_auc:.4f}")
        print(f"{'='*60}")

if __name__ == '__main__':
    
    # --- 正式运行 ---
    # 在您的环境中，您应该使用真实的路径
    # DATA_ROOT = '/data/our_data'
    # OUTPUT_DIR = '/home/jzuser/Work_dir/SeqTrackv2'
    run_tracker_on_dataset(
        tracker_name='seqtrackv2',
        para_name='seqtrackv2_b256',
        data_root='/home/jzuser/Work_dir/SeqTrackv2/data/our_data',
        output_dir='/home/jzuser/Work_dir/SeqTrackv2',
        vis=True
    )
