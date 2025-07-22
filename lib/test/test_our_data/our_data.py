"""
\file vot.py
@brief Python utility functions for VOT integration
@author Luka Cehovin, Alessio Dore
@date 2016
"""
import os
import sys
import copy
import collections
import torch
import numpy as np
import re
from lib.train.dataset.depth_utils import merge_img



Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])

class Our_Data(object):
    """ Base class for Python VOT integration """
    def __init__(self, region_format, channels='rgbd'):
        """ Constructor
        Args:
            region_format: Region format options
        """
        assert(region_format in ['RECTANGLE' , 'POLYGEN' , 'MASK'])

        if channels is None:
            channels = ['color']
        elif channels == 'rgbd':
            channels = ['color', 'depth']
        elif channels == 'rgbt':
            channels = ['color', 'ir']
        elif channels == 'ir':
            channels = ['ir']
        else:
            raise Exception('Illegal configuration {}.'.format(channels))
        self.data_base = '/home/jinyankai/PycharmProjects/data/our_data/'


    def frame(self):
        img_folder = os.path.join(self.data_base, 'color')
        depth_folder = os.path.join(self.data_base, 'depth')
        img_files = os.listdir(img_folder)
        frames = []
        for img_file in img_files:
            depth_path = img_file.split('.')[0] + '.png'
            frames.append([img_file, depth_path])
        return frames

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

    def region(self):
        anno_path = os.path.join(self.data_base, 'groundtruth_rect.txt')
        bboxes = self._load_bboxes(anno_path)
        return bboxes[0]

    def nlp(self):
        description_path = os.path.join(self.data_base, 'nlp.txt')
        descriptions  = []
        with open(description_path, 'r') as f:
            description = f.readline().strip()
        descriptions.append(description)
        return descriptions



