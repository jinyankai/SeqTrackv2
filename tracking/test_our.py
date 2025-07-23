import os
import sys
import argparse

env_path = '/home/jzuser/Work_dir/SeqTrackv2/'

sys.path.append(env_path)

from lib.test.test_our_data.our_data import run_tracker_on_dataset

if __name__ == "__main__":
  run_tracker_on_dataset(
        tracker_name='seqtrackv2',
        para_name='seqtrackv2_b256',
        data_root='/home/jzuser/Work_dir/SeqTrackv2/data/our_data',
        output_dir='/home/jzuser/Work_dir/SeqTrackv2/output',
        vis=True
    )