# =============================================================================
# LA3D: Lightweight Anonymization (AN) and Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides VAD functions for LA3D
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: October 2024
# =============================================================================
import sys, os
import torch
import numpy as np
import time
from tqdm import tqdm
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from torch.autograd import Variable

sys.path.append(".")
current_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.dirname(current_path)
sys.path.append(main_path)

from model.VIDEO_ENCODER_RESNET_2048 import resnet
from model.VIDEO_ENCODER_RESNET_1024.models.i3d.i3d_src.i3d_net import I3D

class VideoFeatureExtractor():
    
    def __init__(self, pretrainedpath, feature_dim=1024, device="cuda", **kwargs) -> None:
        print(f"VideoFeatureExtractor: feature_dim={feature_dim}, pretrainedpath={pretrainedpath}")
        print(kwargs)
        self.pretrainedpath = pretrainedpath
        self.feature_dim = feature_dim
        self.frequency = kwargs.get("frequency", 16)
        self.target_resolution = (340, 256) # WxH
        self.sample_T = kwargs.get("sample_T", 10)
        self.device = device

        self.load_extractor()
    
    def load_extractor(self):
        # load and setup the model
        if isinstance(self.pretrainedpath, str):
            if self.feature_dim == 2048:
                if "_nl_" in self.pretrainedpath.lower() or "_nonlocal_" in self.pretrainedpath.lower():
                    i3d = resnet.i3_res50_nl(400, self.pretrainedpath)
                else:
                    i3d = resnet.i3_res50(400, self.pretrainedpath)
            elif self.feature_dim == 1024:
                i3d = I3D(num_classes=400, modality='rgb')
                i3d.load_state_dict(torch.load(self.pretrainedpath, map_location='cpu'))
            else:
                raise Exception("Undefined feature size. choose 2048 or 1024")
        else:
            i3d = self.pretrainedpath

        self.extractor_model = i3d
        self.extractor_model.to(self.device)
        self.extractor_model.eval()

    def load_frame(self, frame_file):
        """
        frame_file: image filepath or array of # HxWx3
        """
        
        data = F.to_pil_image(frame_file) # handles tensor and numpy with different expected input dims
        data = data.resize(self.target_resolution, Image.LANCZOS) # WxH
        data = np.asarray(data)
        data = data.astype(float)

        if isinstance(self.extractor_model, I3D): # for d=1024
            data = (data * 2.0/255.0) - 1
            assert(data.max()<=1.0)
            assert(data.min()>=-1.0)
        else: # for d=2048
            data = data/255.0
            # # # pytorchvideo.org kinetics
            mean = [0.450, 0.450, 0.450] # or [114.75, 114.75, 114.75] on raw
            std = [0.225, 0.225, 0.225] # or [57.375, 57.375, 57.375] on raw
            normalizer_mean = np.array(mean)[np.newaxis, np.newaxis,...]
            normalizer_std = np.array(std)[np.newaxis, np.newaxis,...]
            data = (data - normalizer_mean)/normalizer_std
           
        return data

    def load_rgb_batch(self, rgb_files, frame_indices):
        """
        rgb_files: [b, h, w, 3]: [b, 256, 340, 3]
        batch_data:  [b, h, w, 3]: [b, 256 ,340, 3]
        """
        # batch_data = np.zeros(frame_indices.shape + (256,340,3))
        batch_data = np.zeros(frame_indices.shape + self.target_resolution[::-1] + (3,))
        
        for i in range(frame_indices.shape[0]):
            for j in range(frame_indices.shape[1]):
                batch_data[i,j,:,:,:] = self.load_frame(rgb_files[frame_indices[i][j]])
                # break
        return batch_data

    def oversample_data(self, data):
    
        data_1 = np.array(data[:, :, :224, :224, :])
        data_2 = np.array(data[:, :, :224, -224:, :])
        data_3 = np.array(data[:, :, 16:240, 58:282, :])
        data_4 = np.array(data[:, :, -224:, :224, :])
        data_5 = np.array(data[:, :, -224:, -224:, :])

        if self.sample_T == 10:
            data_flip = np.array(data[:,:,:,::-1,:])
            data_f_1 = np.array(data_flip[:, :, :224, :224, :])
            data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
            data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
            data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
            data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

            return [data_1, data_2, data_3, data_4, data_5,
                data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]
        else:
            return [data_1, data_2, data_3, data_4, data_5]

    def run(self, batch_size, rgb_files=[], device="cuda"):
        print(f"run...device: {device}, batch_size: {batch_size}...")
        chunk_size = 16

        def forward_batch(b_data):
            b_data = b_data.transpose([0, 4, 1, 2, 3])
            b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224
            with torch.no_grad():
                b_data = Variable(b_data.to(device)).float()
                if isinstance(self.extractor_model, I3D):
                    features = self.extractor_model(b_data, features=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    inp = {'frames': b_data}
                    features = self.extractor_model(inp)
            return features.cpu().numpy()

        frame_cnt = len(rgb_files)
        print(f"frame_cnt: {frame_cnt}")

        # chunk frames
        assert(frame_cnt > chunk_size)
        clipped_length = frame_cnt - chunk_size
        clipped_length = (clipped_length // self.frequency) * self.frequency  # The start of last chunk
        frame_indices = [] # Frames to chunks
        for i in range(clipped_length // self.frequency + 1):
            frame_indices.append([j for j in range(i * self.frequency, i * self.frequency + chunk_size)])
        frame_indices = np.array(frame_indices)
        chunk_num = frame_indices.shape[0]

        batch_size = min(chunk_num, batch_size)
        batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)
        print(f"frame_indices:  {len(frame_indices)}, batch_num: {batch_num}")

        full_features = [[] for i in range(self.sample_T)]

        for batch_id in tqdm(range(batch_num)): 
            batch_data = self.load_rgb_batch(rgb_files, frame_indices[batch_id])
            batch_data_ten_crop = self.oversample_data(batch_data)
            for i in range(self.sample_T):
                assert(batch_data_ten_crop[i].shape[-2]==224)
                assert(batch_data_ten_crop[i].shape[-3]==224)
                temp = forward_batch(batch_data_ten_crop[i])
                full_features[i].append(temp)

        full_features = [np.concatenate(i, axis=0) for i in full_features]
        full_features = [np.expand_dims(i, axis=0) for i in full_features]
        full_features = np.concatenate(full_features, axis=0)
        full_features = full_features[:,:,:,0,0,0] 
        full_features = np.array(full_features).transpose([1,0,2])      
      
        return full_features

    def get_extracted_feature(self, rgb_files, batch_size=1):
        print("extracting features...")

        startime = time.time()

        features = self.run(
                            batch_size,
                            rgb_files=rgb_files,
                            device=self.device)
        
        print("obtained features of size: ", features.shape)

        if features.ndim < 3:
            features = np.expand_dims(features, axis=1)
        mag = np.linalg.norm(features, axis=2)[:,:, np.newaxis]
        features = np.concatenate((features, mag),axis = 2)
        
        print("done in {} sec, features shape: {}.".format(time.time() - startime, features.shape))

        return features

