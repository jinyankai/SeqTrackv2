from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pdb
import cv2
import torch
# import vot
import sys
import time
import os
from lib.test.evaluation import Tracker
import our_data
from lib.test.vot.vot22_utils import *
from lib.train.dataset.depth_utils import get_rgbd_frame,merge_img
from lib.utils.box_ops import box_cxcywh_to_xyxy


class seqtrackv2(object):
    def __init__(self, tracker_name='seqtrackv2', para_name=''):
        # create tracker
        tracker_info = Tracker(tracker_name, para_name, "our_data", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def write(self, str):
        txt_path = ""
        file = open(txt_path, 'a')
        file.write(str)

    def initialize(self, img_rgb, selection, nlp = None):
        # init on the 1st frame
        # region = rect_from_mask(mask)
        x, y, w, h = selection
        bbox = [x,y,w,h]
        self.H, self.W, _ = img_rgb.shape
        init_info = {'init_bbox': bbox}
        if nlp is not None:
            init_info['init_nlp'] = nlp[0]
        _ = self.tracker.initialize(img_rgb, init_info)

    def track(self, img_rgb):
        # track
        outputs = self.tracker.track(img_rgb)
        pred_bbox = outputs['target_bbox']
        self.extract_and_save_boxes(pred_bbox)
        max_score = outputs['best_score']  #.max().cpu().numpy()
        return pred_bbox, max_score

    def extract_and_save_boxes(self, boxes_pred_cxcywh,
                               filepath:str = '/home/jinyankai/PycharmProject/SeqTrackv2/output_txt' ,
                               save_format:str='cxcywh'):
        """
        Extracts predicted bounding boxes from model outputs and saves them to a text file.

        Each box is written to a new line in the file.

        Args:
            boxes_pred_cxcywh (torch.Tensor): The raw output tensor from the model.
            filepath (str): The path to the text file where boxes should be saved.
            save_format (str): The format to save the boxes in ('xyxy' or 'cxcywh').
                               Defaults to 'cxcywh'.
        """
        print(f"Extracting boxes and saving to {filepath}...")
        # Disable gradient calculation for this operation as it's not for training
        with torch.no_grad():
            if save_format == 'xyxy':
                boxes_to_save = box_cxcywh_to_xyxy(boxes_pred_cxcywh)
            elif save_format == 'cxcywh':
                boxes_to_save = boxes_pred_cxcywh
            else:
                raise ValueError("save_format must be either 'xyxy' or 'cxcywh'")

            # --- File writing operation ---
            try:
                # Open the file in append mode ('a') to add new lines without
                # overwriting existing content. Use 'w' to overwrite the file each time.
                with open(filepath, 'a') as f:
                    # Iterate over each predicted box in the batch
                    for box in boxes_to_save:
                        # Convert tensor elements to a list of numbers, then to strings
                        # We round to a few decimal places for cleaner output
                        box_coords = [f"{coord.item():.4f}" for coord in box]
                        # Join coordinates with a comma and write to file with a newline
                        line = ",".join(box_coords)
                        f.write(line + '\n')
                print(f"Successfully wrote {len(boxes_to_save)} boxes.")
            except IOError as e:
                print(f"Error: Could not write to file {filepath}. Reason: {e}")


def run_our_data(tracker_name, para_name = 'seqtrackv2_b256', vis=False, out_conf=False, channel_type='rgbd'):

    torch.set_num_threads(1)
    save_root = os.path.join('', para_name)
    if vis and (not os.path.exists(save_root)):
        os.mkdir(save_root)
    tracker = seqtrackv2(tracker_name=tracker_name, para_name=para_name)

    if channel_type=='rgb':
        channel_type=None
    handle = our_data.Our_Data('RECTANGLE' , channels=channel_type)

    selections = handle.region()
    imagefiles = handle.frame()
    nlp = handle.nlp()

    if not imagefiles:
        sys.exit(0)
    if vis:
        '''for vis'''
        seq_name = imagefiles[0].split('/')[-3]
        save_v_dir = os.path.join(save_root,seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # read rgbd data
    if isinstance(imagefiles[0], list) and len(imagefiles[0])==2:
        image = merge_img(imagefiles[0][0], imagefiles[0][1])
    else:
        image = cv2.cvtColor(cv2.imread(imagefiles[0]), cv2.COLOR_BGR2RGB) # Right

    tracker.initialize(image, selections[0])

    for idx , items in enumerate(imagefiles):
        if idx == 0:
            continue
        img_list = items
        if not img_list:
            continue
        if isinstance(img_list, list) and len(img_list) == 2:
            image = merge_img(img_list[0], img_list[1])
        else:
            image = cv2.cvtColor(cv2.imread(img_list), cv2.COLOR_BGR2RGB)
        b1 , max_score = tracker.track(image)

        if vis:
            '''Visualization'''
            # original image
            image_ori = image[:,:,::-1].copy() # RGB --> BGR
            image_name = img_list.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg','_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)





