# =============================================================================
# LA3D: Lightweight Anonymization (AN) and Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides Object Detection and Segmentation functions for LA3D
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: October 2024
# =============================================================================
import os
import torch
import numpy as np
import cv2
import copy
import shutil
from ultralytics import YOLO

import utilities as util
from config import config as cfg

src_path = os.path.abspath(os.path.dirname(__file__))

class PerFrameObjectDetector():
    """
    PerFrameObjectDetector provides per-frame object detection and segmentation
    using configurable detection models (currently supports YOLO).
    """
    def __init__(self, detector_name:str='yolo', **kwargs) -> None:
        """
        Initialize the object detector with the specified detector name and parameters.

        Args:
            detector_name (str): Name of the detection tool/model (default: 'yolo').
            **kwargs: Additional configuration parameters for the detector.
        """
        self.detector_name = detector_name
        self.kwargs = kwargs

    def load_detector(self):
        """
        Load and initialize the object detection model based on configuration.

        This method sets up the segmentation tool, model name, and model parameters.
        Currently, only YOLO is supported. Raises an error for undefined models.

        Kwargs Args:
            detector_name (str): Name of the detection tool/model (default: 'yolo').
            detector_version (str): Version of the detection model to use (default: None).
            detector_version_kwargs (dict): Additional keyword arguments for the detection model (default: {}).
            seg_type (str): Type of segmentation to perform (default: 'instant').
        """        
        print('loading object detector...')
        print(self.kwargs)
        if self.detector_name is None:
            print("detector_name: ", self.detector_name)
            return
        
        self.detector_name = self.kwargs.get('detector_name', 'yolo')
        self.seg_type = self.kwargs.get('seg_type', 'instant')
        self.detector_version = self.kwargs.get('detector_version', None)
        self.detector_version_kwargs = self.kwargs.get('detector_version_kwargs', {})
        self.seg_tool_dict = cfg.object_detection_model_dict[self.detector_name]
        self.seg_model_name = self.seg_tool_dict[self.seg_type]['model'] if self.detector_version is None else self.detector_version
        self.seg_model_kwargs = self.seg_tool_dict[self.seg_type]['kwargs'] if self.detector_version_kwargs is None else self.detector_version_kwargs
        self.seg_model_kwargs.update(self.kwargs)
        self.sem_class_to_idx = {}

        if self.detector_name in ['yolo']:
            print("#"*30)
            print("model: ", self.seg_model_name)
            # COCO-pretrained Instance segmentation model
            try:
                self.model = YOLO(rf'{src_path}/model/OBJECT_DETECTOR/{self.detector_name}/{self.seg_model_name}')
            except Exception as ex:
                print(f"Error loading model {self.seg_model_name}: {ex}")
                self.model = YOLO(f'{self.seg_model_name}') 
                shutil.move(rf'{src_path}/{self.seg_model_name}', rf'{src_path}/model/OBJECT_DETECTOR/{self.detector_name}')
        else:
            raise "Undefined model. please choose yolo"

    def get_detection(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Perform object detection and segmentation on a single frame.

        Args:
            frame (np.ndarray): Input image frame (BGR).

        Returns:
            tuple: (processed_frame, boolean_masks)
                processed_frame (np.ndarray): Possibly resized/restored frame.
                boolean_masks (np.ndarray or None): Boolean masks for detected objects.
        """
        if self.detector_name is None:
            return frame, None
        
        detection_thr = self.seg_model_kwargs.get('detection_thr', 0.25)
        classes = self.seg_model_kwargs.get('classes', [0])
        imgsz = self.seg_model_kwargs.get('imgsz', None)
        device = self.seg_model_kwargs.get('device', 'cuda')
        device = device if torch.cuda.is_available() else 'cpu'

        if self.detector_name in ['yolo']:
            if imgsz is not None:
                _imgsz_org = frame.shape[:-1]
                frame = util.image_resize(frame, imgsz, ref="w") # preserves dim ratio
                
            results = self.model.predict(frame, 
                                        # imgsz=imgsz, # resize the whole image without keeping the dimension.
                                        conf=detection_thr, retina_masks=True,half=False,
                                        classes=classes, device=device, save=False, verbose=False
                                        )
            boolean_masks = None
            if results[0].masks: 
                boolean_masks = results[0].masks.data.type(torch.uint8).cpu().detach().numpy()
                if imgsz is not None:
                    boolean_masks = np.array([[cv2.resize(mask, _imgsz_org[::-1]).astype(bool) for mask in boolean_masks]])
                    frame = util.image_resize(frame, _imgsz_org[::-1], ref="w")
                else:
                    boolean_masks = np.array([[cv2.resize(mask, results[0].orig_shape[::-1]).astype(bool) for mask in boolean_masks]])
            return frame.astype("uint8"), boolean_masks
        else:
            frame_res = copy.copy(frame)
            boolean_masks = [np.zeros(frame.shape, dtype=np.uint8)]
        return frame_res, boolean_masks
    
