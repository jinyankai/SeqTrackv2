import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.pipeline.seqtrackv2_wrapper.seqtrackv2 import run_our_data

run_our_data('seqtrackv2', 'seqtrackv2_b256', vis=True, out_conf=True, channel_type='rgbd')

