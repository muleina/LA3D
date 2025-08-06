# =============================================================================
# LA3D: Lightweight Anonymization (AN) and Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides AN functions for LA3D
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: October 2024
# =============================================================================
import cv2
import torch
import numpy as np
import copy
from tqdm import tqdm
import torchvision.transforms.functional as F
from PIL import Image

import utilities as util
from detector_interface import *


def _masks_to_boxes_np(masks: np.ndarray) -> np.ndarray:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """

    if masks.size == 0:
        return np.zeros((0, 4), dtype="int16")

    n = masks.shape[0]
    bounding_boxes = np.zeros((n, 4), dtype="int16")
    for index, mask in enumerate(masks):
        y, x = np.where(mask != 0)
        bounding_boxes[index, 0] = np.min(x)
        bounding_boxes[index, 1] = np.min(y)
        bounding_boxes[index, 2] = np.max(x)
        bounding_boxes[index, 3] = np.max(y)
    return bounding_boxes

def apply_anony_mask(frame, masker_func=None, detector=None, mask=None, **kwargs):
    use_adaptive_mask = kwargs.get("use_adaptive_mask", False)

    if masker_func is not None:
        frame_res = copy.deepcopy(frame) 
        if mask is None:
            frame_res = masker_func(frame, **kwargs) if detector is None else frame
        else:

            if isinstance(mask, torch.Tensor):
                mask = mask.detach().numpy()
            
            if mask.ndim == 2:
                frame_res_ = masker_func(frame, mask=mask, **kwargs)
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                frame_res[mask] = frame_res_[mask]      

                frame_res_ = copy.deepcopy(frame)
                frame_res[mask] = frame_res_[mask]  

            elif mask.ndim == 3 and mask.shape[0] > 0:
              
                max_num_pixels = np.prod(mask[0].shape)

                if use_adaptive_mask:
    
                    try:
                        boxes = _masks_to_boxes_np(np.array(list(filter(lambda x:x.sum()>0, mask))))
                    except Exception as ex:
                        print(ex)
                        print(f"mask shape = {mask.shape}, dtype = {mask.dtype}")
    
                        for index, mask_ in enumerate(torch.tensor(mask)):
                            print(f"mask shape = {mask_.shape}, dtype = {mask_.dtype}, mask_sum: {mask_.sum()}")
                            y, x = torch.where(mask_ != 0)
                            print(f"x= {x.shape}, dtype = {x.dtype}, y= {y.shape}, dtype = {y.dtype}")
                            print(torch.min(x), torch.min(y), torch.max(x), torch.max(y))


                    for i, (mask_i, box_i) in enumerate(zip(mask, boxes)):
                        mask_scale = int(100*mask_i[:].sum()/max_num_pixels)
                        mask_scale = max(mask_scale, 1)

                        mask_i_ = np.repeat(mask_i[:, :, np.newaxis], 3, axis=2)    

                        frame_res_ = copy.deepcopy(frame)            
                        frame_res_[box_i[1]:box_i[3]+1, box_i[0]:box_i[2]+1, :] = masker_func(frame[box_i[1]:box_i[3]+1, box_i[0]:box_i[2]+1, :], 
                                                                                          mask_scale=mask_scale, image_size=frame_res.shape[:-1], **kwargs)
                        frame_res[mask_i_] = frame_res_[mask_i_]                     
                else:
                    frame_res_ = masker_func(frame, **kwargs)
                    mask = mask.sum(axis=0) > 0
                    mask = np.repeat(mask[..., np.newaxis], 3, axis=2)
                    frame_res[mask] = frame_res_[mask] 

                    # for i, mask_i in enumerate(mask):
                    #     mask_i_ = np.repeat(mask_i[:, :, np.newaxis], 3, axis=2)
                    #     frame_res__ = frame_res_ if postprocessor is None else postprocessor(frame_res_, mask_i, **kwargs) 
                    #     frame_res[mask_i_] = frame_res__[mask_i_]                 

        return frame_res

    return frame

class NonAnonymizer():

    def __init__(self) -> None:
        self.anony_type = "image"
        self.detector_method = None
        self.anonymizer = {"method": None,  "kwargs":{}}
    
    @classmethod
    def get_anonymizers(cls):
        return None 
     
    def get_anonymized(self, x:np.array):
        return x
    
    @property
    def get_anonymizer_settings(self):
        return self.anonymizer

class ConventionalImageAnonymizer():

    anonymizer_dict = {"guassian_blur": {"kwargs": dict(kernelsize=(13, 13))},
                        "pixelization": {"kwargs":dict(downsize=4)},
                        "silhoutte": {"kwargs": dict()},
                        "edge": {"kwargs": dict(thr1=100, thr2=200)},
                        }  
    def __init__(self, method:str, detector_name:str="body__torvis", **kwargs) -> None:
        self.kwargs = kwargs
        self.detector_name = detector_name if detector_name is not None else "image"
        self.kwargs_anony_method = kwargs.get("kwargs_anony_method", {}) 
        self.kwargs_detector_method = kwargs.get("kwargs_detector_method", {"seg_type": "instant"}) 

        if self.detector_name.startswith("image"):
            self.anony_type = "image"
            self.detector_method = None
        else:
            detector_args = self.detector_name.split("__")
            self.anony_type = detector_args[0]
            self.detector_method = detector_args[1] 
        
        if self.anony_type not in ["face", "head", "body", "image"]:
            raise Exception(f"Undefined anonymize type: {self.anony_type}!")

        anonymizer_dispatcher_dict = {f"{alg}": {"method": getattr(self, alg), "kwargs": {**copy.deepcopy(v["kwargs"]), **self.kwargs_anony_method}} for alg, v in self.get_anonymizers().items()}  

        self.anonymizer = anonymizer_dispatcher_dict.get(method, None)
        if self.anonymizer is None:
            raise f"Undefined anonymize method: {self.anonymizer}!"

        self.load_detector()
    
    @classmethod
    def get_anonymizers(cls):
        return cls.anonymizer_dict
    
    def adapt_to_image_size(self, x, base_resolution):
        if base_resolution is not None:
            x = F.to_pil_image(x)
            x_input_size = (x.width, x.height)
            w = base_resolution[0]
            alpha_size_scale = w/x_input_size[0]
            if v['anony_method'] != "raw": self.kwargs["alpha_size_scale"] = alpha_size_scale if alpha_size_scale > 1 else 1
            h = int(x_input_size[1]*alpha_size_scale)
            anony_base_size = (w, h)
            x = x.resize((w, h), Image.LANCZOS)
            x = np.asarray(x)
        else:
            self.kwargs["alpha_size_scale"] = 1
            x_input_size = None
        return x, x_input_size

    def load_detector(self):
        self.detector = PerFrameObjectDetector(detector_name=self.detector_method, 
                                               anony_type=self.anony_type, **self.kwargs_detector_method)
        self.detector.load_detector()

    def get_anonymized(self, x:np.array, **kwargs):
    
        if self.anonymizer is None:
            # non-anonymization
            return x
        
        if self.detector_method is None:
            # image-level anonymization
            y_anony = self.anonymizer["method"](x, **self.anonymizer["kwargs"], **kwargs)
        else:
            # body-level anonymization
            base_resolution = kwargs.get("base_resolution", None)
            x, x_input_size = self.adapt_to_image_size(x, base_resolution)
            y, boolean_masks = self.detector.get_detection(x)

            if boolean_masks is not None:
                frames_with_masks = [apply_anony_mask(frame, masker_func=self.anonymizer["method"], mask=mask, **self.anonymizer["kwargs"], **kwargs)
                                for frame, mask in zip([y], boolean_masks) ]
                y_anony = frames_with_masks[0]
            else:
                y_anony = y
            
            if base_resolution is not None:
                y_anony = F.to_pil_image(y_anony)
                y_anony = y_anony.resize(x_input_size, Image.LANCZOS)

        return y_anony
    
    @property
    def get_anonymizer_settings(self):
        return self.anonymizer

    @staticmethod
    def silhoutte(x, **kwargs):
        alpha = kwargs.get("alpha", 0.95) # mask alpha
        color_format = kwargs.get("color_format", 'rgb')
        color = kwargs.get("color", '#190fd4')

        height, width = x.shape[:2]
        hex2rgb = util.convertcolor_hexstr2tuple(color, color_format=color_format)

        silhoute_image = np.concatenate((np.full(x.shape[:2], hex2rgb[0], np.uint8)[:, :, np.newaxis], 
                            np.full(x.shape[:2], hex2rgb[1], np.uint8)[:, :, np.newaxis],
                            np.full(x.shape[:2], hex2rgb[2], np.uint8)[:, :, np.newaxis]), 
                            axis=2)
        
        x = cv2.addWeighted(x, 1-alpha, silhoute_image, alpha, 1.0)

        return x

    @staticmethod
    def edge(x, **kwargs):
        """
        The smallest value between threshold1 and threshold2 is used for edge linking. The largest value is used to find initial segments of strong edges.
        Those who lie between these two thresholds are classified edges or non-edges based on their connectivity. 
        If they are connected to "sure-edge" pixels, they are considered to be part of edges. Otherwise, they are also discarded. 
        """
        thr1 = kwargs.get("thr1", 100) # thr1
        thr2 = kwargs.get("thr2", 200) # thr2
        use_adaptive_mask = kwargs.get("use_adaptive_mask", False)

        # Convert the frame to grayscale for edge detection 
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) 
        
        # # Apply Gaussian blur to reduce noise and smoothen edges 
        # x = cv2.GaussianBlur(src=x, ksize=(5, 5), sigmaX=0.5) 

        # Perform Canny edge detection 
        x = cv2.Canny(x, thr1, thr2)

        x = x[:, :, np.newaxis]
        x = np.concatenate(([x]*3), axis=2)
    
        return x

    @staticmethod
    def guassian_blur(x, **kwargs):
        """
        """
        kernelsize = kwargs.get("kernelsize", (13, 13))
        sigma = kwargs.get("sigma", (0, 0)) # (x, y) direction, if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively 
        mask_scale = kwargs.get("mask_scale", 0)
        use_log_scaler = kwargs.get("use_log_scaler", False)
        adaptive_sigma = kwargs.get("adaptive_sigma", False)
        box_dims = kwargs.get("box_dims", None)

        alpha_mask_scale = kwargs.get("alpha_mask_scale", 1.0)
        alpha_size_scale = kwargs.get("alpha_size_scale", 1.0)
        alpha_dim_scale = kwargs.get("alpha_dim_scale", 0.5)   # v4, alpha_dim_scale=0.5, v5, alpha_dim_scale=1

        height, width = x.shape[:2] if box_dims is None else box_dims

        if (kernelsize[0]==0) | (kernelsize[1]==0):
            #  max guassian_blur 
            try:
                kernelsize = [width, height]
                kernelsize = tuple([k-1 if k%2==0 else k for k in kernelsize]) # make kernel is odd number
                x = cv2.GaussianBlur(x, kernelsize, 0, 0) # sigma is auto estimated by opencv
            except Exception as ex:
                print(ex)
                print(f"x: {x.shape}, kernelsize: {kernelsize}, sigma:{sigma}")
                raise ex
        else:
            if mask_scale > 0: # adaptive 
                im_box_scaler = np.log(mask_scale) if use_log_scaler else mask_scale

                im_box_scaler = max(im_box_scaler, 1)

                r = alpha_mask_scale*alpha_size_scale*im_box_scaler
                kernelsize = [int(max(r*k, alpha_mask_scale*k)) for k in kernelsize]
                kernelsize = [min(max(int(alpha_dim_scale*d), 1), k) for k, d in zip(kernelsize, [width, height])]
                
                if adaptive_sigma:
                    sigma = [int(max(r*s, alpha_mask_scale*s)) for s in sigma]

            kernelsize = tuple([k-1 if k%2==0 else k for k in kernelsize]) # make kernel is odd number

            try:
                sigma = min(sigma[0], kernelsize[0]), min(sigma[1], kernelsize[1])

                x = cv2.GaussianBlur(x, kernelsize, 
                                    sigma[0], sigma[1])
            except Exception as ex:
                print(ex)
                print(f"x: {x.shape}, kernelsize: {kernelsize}, sigma:{sigma}")
                raise ex
        
        return x

    @staticmethod
    def pixelization(x, **kwargs):
        downsize = int(kwargs.get("downsize", 4))
        mask_scale = kwargs.get("mask_scale", 0)
        box_dims = kwargs.get("box_dims", None)
        use_log_scaler = kwargs.get("use_log_scaler", False)
        alpha_mask_scale = kwargs.get("alpha_mask_scale", 1.0)
        alpha_size_scale = kwargs.get("alpha_size_scale", 1.0)
        alpha_dim_scale = kwargs.get("alpha_dim_scale", 0.5)

        height, width = x.shape[:2] if box_dims is None else box_dims

        if downsize == 0:
            #  max pixelization 
            try:
                downsize_dims = (1, 1)
                # Resize input to "pixelated" size
                tmp = cv2.resize(x, downsize_dims, interpolation=cv2.INTER_LINEAR)
                # restore image to original size
                x = cv2.resize(tmp, (width, height), interpolation=cv2.INTER_NEAREST)
            except Exception as ex:
                print(f"input_width_height:{width}x{height}, last x.shape[hxwxc]:{x.shape}, mask_scale:{mask_scale}, downsize: {downsize}, downsize_dims: {downsize_dims}")
                raise ex
  
        else:
            assert downsize >= 1

            height, width = x.shape[:2] if box_dims is None else box_dims

            if mask_scale > 0:
      
                im_box_scaler = np.log(mask_scale) if use_log_scaler else mask_scale
                im_box_scaler = max(im_box_scaler, 1)
     
                r = alpha_mask_scale*alpha_size_scale*im_box_scaler

                downsize_target = [int(max(r*downsize, alpha_mask_scale*downsize))]*2
                downsize = tuple([min(max(int(alpha_dim_scale*d), 1), dt) for d, dt in zip([width, height], downsize_target)])

            else:
                downsize =  tuple([downsize if d > downsize else d for d in [width, height]])

            try:
                downsize_dims = (width//downsize[0], height//downsize[1])
                # Resize input to "pixelated" size
                tmp = cv2.resize(x, downsize_dims, interpolation=cv2.INTER_LINEAR)
                # restore image to original size
                x = cv2.resize(tmp, (width, height), interpolation=cv2.INTER_NEAREST)
            except Exception as ex:
                print(f"input_width_height:{width}x{height}, last x.shape[hxwxc]:{x.shape}, mask_scale:{mask_scale}, downsize: {downsize}, downsize_dims: {downsize_dims}")
                raise ex
            
        return x

class AnonymizerHub():
    
    def __init__(self, method, **kwargs) -> None:
        self.method = method
        self.kwargs = kwargs
        self.detector_name = kwargs.get("detector_name", None) 
        self.detector_name = self.detector_name if self.detector_name is not None else "image__None"
        print(self.kwargs)
        self.load_anonymizer()

    def load_anonymizer(self):
        if self.method in ConventionalImageAnonymizer.get_anonymizers().keys():
            self.anonymizer = ConventionalImageAnonymizer(method=self.method, **self.kwargs)
        else:
            print(f"Undefined anonymize method: {self.method}. Thus, enforcing using raw footage.")
            self.anonymizer = NonAnonymizer()
        
        print(self.anonymizer.get_anonymizer_settings)
        
    def anonymize_frame(self, frame, **kwargs):
        frame_res = self.anonymizer.get_anonymized(frame, **kwargs)
        return frame_res

    def anonymize(self, frames:list, **kwargs):
        """
        frames: [b, h, w, 3] or [h, w, 3]
        """
        islist = isinstance(frames, list)
        if not islist:
            frames = [frames] 
        f = lambda x: self.anonymize_frame(x, **kwargs)
        frames = list(map(f, tqdm(frames)))

        if not islist:
            return frames[0]
        return frames
