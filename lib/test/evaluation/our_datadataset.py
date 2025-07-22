import os

import numpy as np
import torch

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text,load_str


class MY_DATA(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.our_data_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i]
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _get_frame_list(self,path_color:str, path_depth:str):
        img_folder = path_color
        depth_folder = path_depth
        image_files = sorted(os.listdir(img_folder))

        for i, img_name in enumerate(image_files):
            depth_img_name = os.path.splitext(img_name)[0] + '.png'






    def _construct_sequence(self, sequence_name):
        # FIXME : according to our datasets structure
        class_name = sequence_name
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        target_visible = torch.ones(ground_truth_rect.shape[0], dtype=torch.bool)

        frames_path = '{}/{}/color'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        language_query_path = '{}/{}/nlp.txt'.format(self.base_path, sequence_name)
        language_query = load_str(language_query_path)

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'our_data', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible,language_query=language_query)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['basketball1',
                         ]
        return sequence_list
