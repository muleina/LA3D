# =============================================================================
# LA3D: Lightweight Anonymization (AN) and Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides inference configuration for the MGFN VAD model.
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: May 2024
# =============================================================================
import numpy as np
import os, sys
current_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.dirname(current_path))
sys.path.append(current_path)

class Config(object):
    def __init__(self, args):
        self.lr = eval(args.lr)
        self.lr_str = args.lr

    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')

def build_config(dataset):
    print("dataset: ", dataset)
    cfg = type('', (), {})()
    if dataset in ['ucf', 'ucf-crime']:
        cfg.dataset = 'ucf-crime'
        cfg.metrics = 'AUC'
        cfg.ckpt_path = rf'{current_path}/ckpt/mgfn_ucf.pkl'
        cfg.enc_vid_model_filepath = rf"{model_path}/VIDEO_ENCODER_RESNET_2048/ckpt/i3d_r50_nl_kinetics.pth"
        cfg.enc_vid_feature_dim = 2048
        cfg.enc_vid_num_crops = 10
    elif dataset in ['xd', 'xd-violence']:
        cfg.dataset = 'xd-violence'
        cfg.metrics = 'AP'
        cfg.ckpt_path = rf'{current_path}/ckpt/mgfn_xd.pkl'
        cfg.enc_vid_model_filepath = rf"{model_path}/VIDEO_ENCODER_RESNET_1024/models/i3d/ckpt/i3d_rgb.pt"
        cfg.enc_vid_feature_dim = 1024
        cfg.enc_vid_num_crops = 5
    return cfg
