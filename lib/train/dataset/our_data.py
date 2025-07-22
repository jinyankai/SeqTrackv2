import os
import torch.utils.data
from lib.train.data.image_loader import jpeg4py_loader,opencv_loader
# 假设 base_image_dataset.py 与此文件在同一目录或在 Python 路径中
import sys
sys.path.append('/home/jinyankai/PycharmProject/SeqTrackv2/lib/train/dataset')
from base_video_dataset import  BaseVideoDataset
from depth_utils import merge_img
from PIL import Image
import numpy as np
import re
from collections import OrderedDict


class MyDataset(BaseVideoDataset):
    """
    用于处理自定义多序列RGB+深度+文本数据集的类。
    此类现在面向序列（视频），而不是单个图像。
    """

    def __init__(self, name, root_path, image_loader=opencv_loader,
                 multi_modal_vision=False, multi_modal_language=False):
        """
        Args:
            name (str): 数据集名称。
            root_path (str): 数据集根目录的路径, 包含多个类别子文件夹。
            image_loader: 加载图像的函数。
        """
        super().__init__(name, root_path, image_loader)

        # 加载所有序列的数据
        # self.sequence_list 现在是一个列表，每个元素代表一个序列(一个类别文件夹)
        self.sequence_list = self._build_sequence_list()

        # 从加载的数据中构建类别列表
        # self.class_list 中的顺序与 self.sequence_list 的顺序一一对应
        self.class_list = [seq['class'] for seq in self.sequence_list]
        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language

    def _build_sequence_list(self):
        """
        扫描根目录，加载所有序列（类别）的数据。
        返回一个序列列表，其中每个序列都是一个包含其所有帧信息的字典。
        """
        sequences = []
        class_names = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])

        for class_name in class_names:
            class_path = os.path.join(self.root, class_name)
            img_folder_path = os.path.join(class_path, 'color')
            depth_folder_path = os.path.join(class_path, 'depth')
            bbox_file_path = os.path.join(class_path, 'groundtruth_rect.txt')
            desc_file_path = os.path.join(class_path, 'nlp.txt')

            if not all(os.path.exists(p) for p in [img_folder_path, depth_folder_path, bbox_file_path, desc_file_path]):
                print(f"Warning: Skipping sequence '{class_name}' due to missing files/folders.")
                continue

            bboxes = self._load_bboxes(bbox_file_path)
            descriptions = []
            with open(desc_file_path, 'r', encoding='utf-8') as f:
                description = f.readline().strip()
            descriptions.append(description)

            image_files = sorted(os.listdir(img_folder_path))

            if len(image_files) != len(bboxes):
                print(
                    f"Warning: Skipping sequence '{class_name}'. Mismatch between image count ({len(image_files)}) and bbox count ({len(bboxes)}).")
                continue

            # 为这个序列创建一个帧列表
            frames = []
            for i, img_name in enumerate(image_files):
                depth_img_name = os.path.splitext(img_name)[0] + '.png'
                frame_entry = {
                    'rgb_path': os.path.join(img_folder_path, img_name),
                    'depth_path': os.path.join(depth_folder_path, depth_img_name),
                    'bbox': bboxes[i],
                }
                frames.append(frame_entry)

            # 将整个序列的信息作为一个字典添加到列表中
            sequences.append({
                'class': class_name,
                'description': descriptions,
                'frames': frames,
            })

        return sequences

    def _load_bboxes(self, path):
        """更稳健地加载bbox文件，处理带[]和不同分隔符的情况。"""
        bboxes = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if len(parts) == 4:
                    bboxes.append([float(p) for p in parts])
        return torch.tensor(bboxes, dtype=torch.float32)



    def get_name(self):
        return self.name

    def get_num_sequences(self):
        """返回数据集中序列（视频）的总数。"""
        return len(self.sequence_list)

    def get_num_images(self):
        """为与基类兼容，返回序列总数。"""
        return self.get_num_sequences()

    def has_class_info(self):
        return True

    def get_class_name(self, seq_id):
        """根据序列ID返回类别名称。"""
        return self.sequence_list[seq_id]['class']

    def is_video_sequence(self):
        """
        告诉采样器这是一个基于视频/序列的数据集。
        这是必须的。
        """
        return True

    def get_sequence_info(self, seq_id):
        """
        返回关于一个序列的元信息，最重要的是 `visible` 标志。
        采样器使用 `visible` 来决定可以从哪些帧中进行采样。
        """
        sequence = self.sequence_list[seq_id]
        num_frames = len(sequence['frames'])

        # 在这个数据集中，我们假设所有帧中的目标都是可见且有效的。
        # 如果您有遮挡等信息，可以在这里提供。
        visible = torch.ones(num_frames, dtype=torch.bool)

        return {'visible': visible, 'valid': visible}

    def get_frames(self, seq_id, frame_ids, anno=None):
        """
        获取一个序列中的指定帧。
        Args:
            seq_id (int): 序列的ID (在self.sequence_list中的索引)。
            frame_ids (list[int]): 需要从此序列中获取的帧的ID列表。
            anno (None): 此处未使用，但为保持API兼容性而保留。
        Returns:
            tuple: (frame_list, anno_frames, object_meta)
        """
        sequence = self.sequence_list[seq_id]
        class_name = sequence['class']

        frame_list = []
        anno_list = []
        for f_id in frame_ids:
            frame_info = sequence['frames'][f_id]

            # 加载RGB和深度图像，并进行堆叠
            rgb_image = self.image_loader(frame_info['rgb_path'])
            depth_image = np.array(Image.open(frame_info['depth_path']).convert('L'))
            rgbd_image = merge_img(rgb_image, depth_image)
            # rgbd_image_processed = rgbd_image.astype(np.float32).transpose((2, 0, 1))
            frame_list.append(rgbd_image)
            anno_list.append(frame_info['bbox'])

        # 格式化标注
        anno_frames = {'bbox': anno_list,
                       'nlp' : sequence['description']}

        object_meta = OrderedDict({
            'object_class_name': class_name,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        return frame_list, anno_frames, object_meta

    def __len__(self):
        return self.get_num_sequences()

    def __getitem__(self, index):
        """
        对于序列数据集，此方法通常不直接使用。
        应使用 get_frames() 来获取数据。
        """
        return None


# --- 使用示例 ---
if __name__ == '__main__':
    dataset_root = '/home/jinyankai/PycharmProject/SeqTrackv2/data'  # <--- 请将这里替换为您的数据集根目录

    if not os.path.isdir(dataset_root) or 'path/to/your' in dataset_root:
        print("=" * 50)
        print("错误：请将 'dataset_root' 变量设置为您的数据集的有效路径！")
        print("该路径下应包含多个序列（类别）的子文件夹。")
        print(f"当前路径: '{dataset_root}'")
        print("=" * 50)
    else:
        my_dataset = MyDataset(name='my_multi_sequence_dataset', root_path=dataset_root)

        print(f"数据集名称: {my_dataset.get_name()}")
        print(f"序列总数: {my_dataset.get_num_sequences()}")
        print(f"检测到的类别: {my_dataset.get_class_list()}")

        if my_dataset.get_num_sequences() > 0:
            print("\n测试 get_frames() 方法:")
            # 获取第一个序列 (seq_id=0) 的第 0, 2, 4 帧
            seq_id_to_test = 0
            frame_ids_to_test = [0, 2, 4]

            # 确保请求的帧存在
            num_frames_in_seq = len(my_dataset.sequence_list[seq_id_to_test]['frames'])
            if max(frame_ids_to_test) < num_frames_in_seq:
                frames, annos, meta = my_dataset.get_frames(seq_id_to_test, frame_ids_to_test)

                print(f"  - 成功获取序列 ID: {seq_id_to_test}")
                print(f"  - 类别: '{meta['object_class_name']}'")
                print(f"  - 获取到的帧数: {len(frames)}")
                print(f"  - 第一帧的尺寸 (HxWxC): {frames[0].shape}")
                print(f"  - 获取到的标注数量: {len(annos['bbox'])}")
                print(f"  - 第一个标注: {annos['bbox'][0]}")
            else:
                print(
                    f"  - 无法测试，序列 {seq_id_to_test} 的帧数 ({num_frames_in_seq}) 少于请求的最大帧ID ({max(frame_ids_to_test)})")

        else:
            print("\n数据集中没有找到样本。请检查您的文件夹结构和文件内容。")
