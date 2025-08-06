# =============================================================================
# LA3D: Lightweight Anonymization (AN) and Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides Object Detection and Segmentation functions for LA3D
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: October 2024
# =============================================================================
import torch
import numpy as np
import cv2
import copy
import utilities as util

from config import config as cfg


class PerFrameObjectDetector():
    def __init__(self, detector_name:str='yolo', **kwargs) -> None:
        self.run_tool = detector_name
        self.kwargs = kwargs

    def load_detector(self):
        
        """ segemtation tools."""

        print('loading object detector...')
        print(self.kwargs)

        if self.run_tool is None:
            print("run_tool: ", self.run_tool)
            return

        self.run_tool = self.kwargs.get('run_tool', 'yolo')
        self.seg_type = self.kwargs.get('seg_type', 'instant')
        self.anony_type = self.kwargs.get('anony_type', 'body')
        self.detector_version = self.kwargs.get('detector_version', None)
        self.detector_version_kwargs = self.kwargs.get('detector_version_kwargs', {})
        self.seg_tool_dict = cfg.object_detection_model_dict[self.run_tool]
        self.seg_model_name = self.seg_tool_dict[self.seg_type]['model'] if self.detector_version is None else self.detector_version
        self.seg_model_kwargs = self.seg_tool_dict[self.seg_type]['kwargs'] if self.detector_version_kwargs is None else self.detector_version_kwargs
        self.seg_model_kwargs.update(self.kwargs)

        self.sem_class_to_idx = {}

        if self.run_tool in ['yolo']:
            from ultralytics import YOLO
            print("#"*30)
            print("model: ", self.seg_model_name)
            # COCO-pretrained Instance segmentation model
            self.model = YOLO(f'{self.seg_model_name}')
        else:
            raise "Undefined model. please choose yolo"

    def get_detection(self, frame):

        if self.run_tool is None:
            return frame, None
        
        detection_thr =  self.seg_model_kwargs.get('detection_thr', 0.25)
        classes = self.seg_model_kwargs.get('classes', [0])
        imgsz =  self.seg_model_kwargs.get('imgsz', None)
        device = self.seg_model_kwargs.get('device', 'cuda')

        if self.run_tool in ['yolo']:
            if imgsz is not None:
                imgsz_org = frame.shape[:-1]
                frame = util.image_resize(frame, imgsz, ref="w") # preserves dim ratio

            results = self.model.predict(frame, 
                                        # imgsz=imgsz, # 640 better detection, resize the whole image without keeping the dimension.
                                        conf=detection_thr,
                                        retina_masks=True,	# use high-resolution segmentation masks
                                        half=False,
                                        classes=classes, # person
                                        device=device,
                                        save=False, verbose=False
                                        )

            boolean_masks = None
            if results[0].masks: 
                boolean_masks = results[0].masks.data.type(torch.uint8).cpu().detach().numpy()
                boolean_masks = np.array([[cv2.resize(mask, results[0].orig_shape[::-1]).astype(bool) for mask in boolean_masks]])

            boolean_masks = None
            if results[0].masks: 
                boolean_masks = results[0].masks.data.type(torch.uint8).cpu().detach().numpy()
                if imgsz is not None:
                    boolean_masks = np.array([[cv2.resize(mask, imgsz_org[::-1]).astype(bool) for mask in boolean_masks]])
                    frame = util.image_resize(frame, imgsz_org[::-1], ref="w")
                else:
                    boolean_masks = np.array([[cv2.resize(mask, results[0].orig_shape[::-1]).astype(bool) for mask in boolean_masks]])

            return frame.astype("uint8"), boolean_masks

        else:
            frame_res = copy.copy(frame)
            boolean_masks = [np.zeros(frame.shape, dtype=np.uint8)]
            
        return frame_res, boolean_masks
    
