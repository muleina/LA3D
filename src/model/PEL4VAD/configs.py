import os, sys
current_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.dirname(current_path))
sys.path.append(current_path)

def build_config(dataset):
    print("dataset: ", dataset)
    cfg = type('', (), {})()
    if dataset in ['ucf', 'ucf-crime']:
        cfg.dataset = 'ucf-crime'
        cfg.model_name = 'ucf_'
        cfg.metrics = 'AUC'
        # cfg.feat_prefix = rf'{data_path}/data/pyj/feat/ucf-i3d'
        # cfg.train_list = rf'{data_path}/list/ucf/train.list'
        # cfg.test_list = rf'{data_path}/list/ucf/test.list'
        # cfg.token_feat = rf'{data_path}/list/ucf/ucf-prompt.npy'
        # cfg.gt = rf'{data_path}/list/ucf/ucf-gt.npy'
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        # CC settings
        cfg.t_step = 9
        # training settings
        cfg.temp = 0.09
        cfg.lamda = 1
        cfg.seed = 9
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'slide'  # ['fixed': 10, slide': 7]
        cfg.kappa = 7  # smooth window
        cfg.ckpt_path = rf'{current_path}/ckpt/ucf__8636.pkl'
        cfg.enc_vid_model_filepath = rf"{model_path}/VIDEO_ENCODER_RESNET_1024/models/i3d/ckpt/i3d_rgb.pt"
        cfg.enc_vid_feature_dim = 1024
        cfg.enc_vid_num_crops = 10

    elif dataset in ['xd', 'xd-violence']:
        cfg.dataset = 'xd-violence'
        cfg.model_name = 'xd_'
        cfg.metrics = 'AP'
        # cfg.feat_prefix = rf'{data_path}/i3d-features'
        # cfg.train_list = rf'{current_path}/list/xd/train.list'
        # cfg.test_list = rf'{current_path}/list/xd/test.list'
        # cfg.token_feat = rf'{current_path}/list/xd/xd-prompt.npy'
        # cfg.gt = rf'{current_path}/list/xd/xd-gt.npy'
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.06
        cfg.bias = 0.02
        cfg.norm = False
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.05
        cfg.lamda = 1
        cfg.seed = 4
        # test settings
        cfg.test_bs = 5 #, must be 5 to include all 5-clip features. clip features are stored in separate files.
        cfg.smooth = 'fixed'  # ['fixed': 8, slide': 3] 
        cfg.kappa = 8  # smooth window
        cfg.ckpt_path = rf'{current_path}/ckpt/xd__8526.pkl'
        cfg.enc_vid_model_filepath = rf"{model_path}/VIDEO_ENCODER_RESNET_1024/models/i3d/ckpt/i3d_rgb.pt"
        cfg.enc_vid_feature_dim = 1024
        cfg.enc_vid_num_crops = 5

    # base settings
    cfg.feat_dim = 1024
    cfg.head_num = 1
    cfg.hid_dim = 128
    cfg.out_dim = 300
    cfg.lr = 5e-4
    cfg.dropout = 0.1
    cfg.train_bs = 128
    cfg.max_seqlen = 200
    cfg.max_epoch = 50
    cfg.workers = 8
    # cfg.save_dir = rf'{current_path}/ckpt/'
    # cfg.logs_dir = rf'{current_path}/log_info.log'

    return cfg
