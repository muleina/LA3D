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
import cv2
import time
from tqdm import tqdm
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from torch.autograd import Variable

# Add project paths for module imports
sys.path.append(".")
current_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.dirname(current_path)
sys.path.append(main_path)

import utilities as util
from config import config as cfg
from model.VIDEO_ENCODER_RESNET_2048 import resnet
from model.VIDEO_ENCODER_RESNET_1024.models.i3d.i3d_src.i3d_net import I3D
from model.PEL4VAD import configs as pel4vad_configs
from model.PEL4VAD import model_interface as pel4vad_interface
from model.MGFN import config as mgfn_configs
from model.MGFN import model_interface as mgfn_interface

# -----------------------------------------------------------------------------
# Visualization Parameters for VAD Panel
# -----------------------------------------------------------------------------
AD_TEXT = "AD: {0:0.2f} Alert: "
TOP_PAD = 10
R = 4
TEXT_OFFSET = len(AD_TEXT)+R
FLAG_COORDS = [(TEXT_OFFSET*R, TOP_PAD), (TEXT_OFFSET*R+2*R, TOP_PAD), (TEXT_OFFSET*R+4*R, TOP_PAD)]
TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.09*R
TEXT_THICKNESS = 0
AD_TEXT_COORD = [0, FLAG_COORDS[0][1]+1*R]
STATUS_REC_COORDS = (0, FLAG_COORDS[0][1]-2*R), (FLAG_COORDS[-1][0]+2*R, FLAG_COORDS[0][1]+2*R)

# -----------------------------------------------------------------------------
# Timer Utility
# -----------------------------------------------------------------------------
timer = util.ProcTimer()

class VideoFeatureExtractor():
    """
    VideoFeatureExtractor extracts spatio-temporal features from video frames using
    pretrained I3D or ResNet models for anomaly detection.
    """
    def __init__(self, pretrainedpath: str, feature_dim: int = 1024, device: str = "cuda", **kwargs) -> None:
        """
        Initialize the feature extractor.

        Args:
            pretrainedpath (str): Path to the pretrained model weights or model object.
            feature_dim (int): I3D Feature dimension (1024 for I3D, 2048 for resnet).
            device (str): Device to run the model on ('cuda' or 'cpu').
            **kwargs: Additional parameters (frequency, sample_T, etc.).
        """        
        print(f"VideoFeatureExtractor: feature_dim={feature_dim}, pretrainedpath={pretrainedpath}")
        print(kwargs)
        self.pretrainedpath = pretrainedpath
        self.feature_dim = feature_dim
        self.frequency = kwargs.get("frequency", 16)
        self.target_resolution = (340, 256) # WxH
        self.sample_T = kwargs.get("sample_T", 10)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.load_extractor()
    
    def load_extractor(self):
        """
        Load and set up the feature extraction model (I3D or ResNet).
        """
        # load and setup the model
        if isinstance(self.pretrainedpath, str):
            # Load I3D VIDEO_ENCODER_RESNET_2048-based video encoder
            if self.feature_dim == 2048:
                if "_nl_" in self.pretrainedpath.lower() or "_nonlocal_" in self.pretrainedpath.lower():
                    i3d = resnet.i3_res50_nl(400, self.pretrainedpath)
                else:
                    i3d = resnet.i3_res50(400, self.pretrainedpath)
            elif self.feature_dim == 1024:
                # Load I3D VIDEO_ENCODER_RESNET_1024-based video encoder
                i3d = I3D(num_classes=400, modality='rgb')
                i3d.load_state_dict(torch.load(self.pretrainedpath, map_location='cpu'))
            else:
                raise Exception("Undefined feature size. choose 2048 or 1024")
        else:
            i3d = self.pretrainedpath

        self.extractor_model = i3d
        self.extractor_model.to(self.device)
        self.extractor_model.eval()

    def load_frame(self, frame_file: np.ndarray) -> np.ndarray:
        """
        Load and preprocess a single frame for feature extraction.

        Args:
            frame_file (np.ndarray): Image array (HxWx3).

        Returns:
            np.ndarray: Preprocessed image array.
        """
        data = F.to_pil_image(frame_file)
        data = data.resize(self.target_resolution, Image.LANCZOS) # WxH
        data = np.asarray(data).astype(np.float64)
        if isinstance(self.extractor_model, I3D): # for d=1024
            data = (data * 2.0/255.0) - 1
            assert(data.max()<=1.0)
            assert(data.min()>=-1.0)
        else: # for d=2048
            data = data/255.0
            # pytorchvideo.org kinetics
            mean = [0.450, 0.450, 0.450] 
            std = [0.225, 0.225, 0.225]
            normalizer_mean = np.array(mean)[np.newaxis, np.newaxis,...]
            normalizer_std = np.array(std)[np.newaxis, np.newaxis,...]
            data = (data - normalizer_mean)/normalizer_std
        return data.astype("float32")

    def prepare_rgb_batch(self, rgb_files: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
        """
        Load and preprocess a batch of frames for feature extraction.

        Args:
            rgb_files (np.ndarray): Array of frames [b, h, w, 3].
            frame_indices (np.ndarray): Indices for batch selection.

        Returns:
            np.ndarray: Batch data [batch, frequency, h, w, 3].
        """
        batch_data = np.zeros(frame_indices.shape + self.target_resolution[::-1] + (3,), dtype=np.float32)
        for i in range(frame_indices.shape[0]):
            for j in range(frame_indices.shape[1]):
                batch_data[i,j,:,:,:] = self.load_frame(rgb_files[frame_indices[i][j]])
        return batch_data

    def oversample_crop_augmentation(self, data: np.ndarray) -> list:
        """
        Perform spatial oversampling (ten-crop) on the batch data.

        Args:
            data (np.ndarray): Input batch data.

        Returns:
            list: List of oversampled and cropped data arrays.
        """
        data_1 = data[:, :, :224, :224, :].copy()
        data_2 = data[:, :, :224, -224:, :].copy()
        data_3 = data[:, :, 16:240, 58:282, :].copy()
        data_4 = data[:, :, -224:, :224, :].copy()
        data_5 = data[:, :, -224:, -224:, :].copy()
        if self.sample_T == 10:
            data_flip = data[:,:,:,::-1,:].copy()
            data_f_1 = data_flip[:, :, :224, :224, :].copy()
            data_f_2 = data_flip[:, :, :224, -224:, :].copy()
            data_f_3 = data_flip[:, :, 16:240, 58:282, :].copy()
            data_f_4 = data_flip[:, :, -224:, :224, :].copy()
            data_f_5 = data_flip[:, :, -224:, -224:, :].copy()
            return [data_1, data_2, data_3, data_4, data_5,
                data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]
        else:
            return [data_1, data_2, data_3, data_4, data_5]

    def run(self, rgb_files: list = [], batch_size: int = 1) -> np.ndarray:
        """
        Extract features from a sequence of frames in batches.

        Args:
            rgb_files (list): List of frames (as arrays).
            batch_size (int): Number of chunks to process per batch.
    
        Returns:
            np.ndarray: Extracted features array.
        """
        def forward_batch(b_data: np.ndarray) -> np.ndarray:
            """
            Forward pass for a batch of data.

            Args:
                b_data (np.ndarray): Batch data [batch, frequency, h, w, 3].

            Returns:
                np.ndarray: Extracted features.
            """
            b_data = b_data.transpose([0, 4, 1, 2, 3])
            b_data = torch.from_numpy(b_data)   # b,c,t,h,w 
            with torch.no_grad():
                b_data = Variable(b_data.to(self.device)).float()
                if isinstance(self.extractor_model, I3D):
                    features = self.extractor_model(b_data, features=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    inp = {'frames': b_data}
                    features = self.extractor_model(inp)
            return features.cpu().numpy()

        print(f"run...device: {self.device}, batch_size: {batch_size}...")
        chunk_size = 16
        frame_cnt = len(rgb_files)
        print(f"frame_cnt: {frame_cnt}")
        # chunk frames for batch processing
        assert(frame_cnt > chunk_size)
        clipped_length = frame_cnt - chunk_size
        clipped_length = (clipped_length // self.frequency) * self.frequency  # the start of last chunk
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
            batch_data = self.prepare_rgb_batch(rgb_files, frame_indices[batch_id])
            batch_data_ten_crop = self.oversample_crop_augmentation(batch_data)
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

    def get_extracted_feature(self, rgb_files: list, batch_size: int = 1) -> np.ndarray:
        """
        Public interface to extract features from a list of frames.

        Args:
            rgb_files (list): List of frames (as arrays).
            batch_size (int): Batch size for processing.

        Returns:
            np.ndarray: Extracted features with magnitude appended.
        """
        print("extracting features...")

        startime = time.time()
        features = self.run(rgb_files=rgb_files, batch_size=batch_size)
        print("obtained features of size: ", features.shape)
        if features.ndim < 3:
            features = np.expand_dims(features, axis=1)
        mag = np.linalg.norm(features.astype("float64"), axis=2)[:,:, np.newaxis].astype("float32")
        features = np.concatenate((features, mag),axis = 2)
        print("done in {} sec, features shape: {}.".format(time.time() - startime, features.shape))
        return features

class VADetector():
    """
    VAD Integration Engine.
    Handles initialization and inference for video anomaly detection models.
    """
    def __init__(self, ad_method: str = "pel", kwargs_ad_method: dict = {}, device : str = "cuda", **kwargs):
        self.ad_method = ad_method
        self.ad_model_src = kwargs_ad_method.get("ad_model_src", cfg.ad_model_src)
        self.enc_frequency = kwargs_ad_method.get("enc_frequency", cfg.enc_vid_enc_frame_seq_size)
        self.ad_thr = kwargs_ad_method.get("ad_thr", cfg.ad_thr)
        self.device = device if torch.cuda.is_available() else "cpu"
        
        if self.ad_method in ["pel"]:
            # Initialize PEL4VAD model and feature extractor
            _pel4vad_cfg = pel4vad_configs.build_config(self.ad_model_src)
            self.objVADModel = pel4vad_interface.load_model(_pel4vad_cfg.ckpt_path, datasource=self.ad_model_src).to(self.device)
            self.objVADModel.eval()
            
            self.vidfeature_i3d_extractor = VideoFeatureExtractor(pretrainedpath=rf"{_pel4vad_cfg.enc_vid_model_filepath}", 
                                                            frequency=self.enc_frequency, feature_dim=_pel4vad_cfg.enc_vid_feature_dim, sample_T=_pel4vad_cfg.enc_vid_num_crops)
            
            self.predict = lambda x: pel4vad_interface.predict(torch.tensor(x[:, :, :-1], dtype=torch.float32).to(self.device), self.objVADModel, frequency=self.vidfeature_i3d_extractor.frequency, device=self.device)

        elif self.ad_method in ["mgfn"]:
            # Initialize MGFN model and feature extractor            
            _mgfn_cfg = mgfn_configs.build_config(self.ad_model_src)
            self.objVADModel = mgfn_interface.load_model(filepath=_mgfn_cfg.ckpt_path, channels=_mgfn_cfg.enc_vid_feature_dim).to(self.device)
            self.objVADModel.eval()
            self.vidfeature_i3d_extractor = VideoFeatureExtractor(pretrainedpath=rf"{_mgfn_cfg.enc_vid_model_filepath}", 
                                                            frequency=self.enc_frequency, feature_dim=_mgfn_cfg.enc_vid_feature_dim, sample_T=_mgfn_cfg.enc_vid_num_crops)
            
            self.predict = lambda x: mgfn_interface.predict(torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device), self.objVADModel, frequency=self.vidfeature_i3d_extractor.frequency, device=self.device)
        else:
            raise f"Undefined ad_method: {self.ad_method}!"

    def inference(self, video_frames: np.ndarray) -> tuple[bool, np.ndarray | None, np.ndarray]:
        """
        Run anomaly detection on video frames.
        Args:
            video_frames: RGB frames [b, h, w, 3].
        Returns:
            success: Boolean indicating success.
            ad_scores: Anomaly score(s) [b, 1].
            video_frames: Input frames (unchanged).
        """ 
        success = False

        if video_frames is None:
            return success, None, video_frames
        elif len(video_frames) < self.enc_frequency:
            return success, None, video_frames

        features = self.vidfeature_i3d_extractor.get_extracted_feature(video_frames, batch_size=20)
        ad_scores = self.predict(features)
        success = True
        return success, ad_scores, video_frames

class ADInference():
    """
    VAD Inference Engine.
    Provides online and offline anomaly detection interfaces.
    """   
    def __init__(self, *args, **kwargs):
        """
        Initialize the VAD inference engine with the specified model and parameters.

        Args:
            *args: Arguments for the VAD model.
            **kwargs: Keyword arguments for the VAD model.
        """
        self.ObjVADetector = VADetector(*args, **kwargs)
        self.ad_thr = self.ObjVADetector.ad_thr
        self.ad_proc_time = 0.0

    def get_ad_status_offline(self, frames = tuple[bool, np.ndarray | None]) -> tuple[bool, np.ndarray | None]:
        """
        Offline anomaly detection for batch video data.
        Args:
            frames: RGB frames [b, h, w, 3].
        Returns:
            ad_success: Boolean indicating success.
            ad_scores: Anomaly score(s) [b, 1].
        """        
        print("VAD Inference...")
        try:
            if frames is not None:
                timer.restart()
                ad_success, ad_scores, frames_snippets = self.ObjVADetector.inference(frames)
                self.ad_proc_time = np.round(timer.get_proctime(time_format="s"), 5)
                timer.stop()
                if not ad_success:
                    if len(frames_snippets) >= self.ObjVADetector.enc_frequency:
                        print("AD ERROR!") 
                    else:
                        print("AD processing, filling sequence of frames...!")
                    return False, None
                return ad_success, ad_scores
            else:
                return False, None
        except Exception as ex:
            print(f"{ex}")
            return False, None
    
    def add_ad_status_to_frame(self, frame: np.array, ad_score: float) -> np.ndarray:
        """
        Overlay anomaly detection status and score on the frame.
        Args:
            frame: RGB frame.
            ad_score: Anomaly score.
        Returns:
            frame: Frame with overlay.
        """
        frame = util.image_resize(frame, [320, 240], mode="cv", ref="wh")
        frame = cv2.rectangle(frame, STATUS_REC_COORDS[0], STATUS_REC_COORDS[1], (127, 127, 127), -1) # add status panel
        ad_flag_color = (255,0,0) if ad_score >= self.ad_thr else (0,255,0)
        for i in range(3):
           _ = cv2.circle(frame, FLAG_COORDS[i], R, ad_flag_color, -1) 
        _ = cv2.putText(frame, AD_TEXT.format(ad_score) , AD_TEXT_COORD, TEXT_FACE, TEXT_SCALE, (0,0,255), TEXT_THICKNESS, cv2.LINE_AA)
        return frame
