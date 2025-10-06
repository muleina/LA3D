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

import utilities as util
from detector_interface import PerFrameObjectDetector
from config import config as cfg

# -----------------------------------------------------------------------------
# Visualization Parameters for VAD Panel
# -----------------------------------------------------------------------------
AN_TEXT = "AN Time: {0:0.3f}s"
R = 4
TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.09*R
TEXT_THICKNESS = 0
AN_TEXT_COORD = [0, 30+1*R]

# -----------------------------------------------------------------------------
# Timer Utility
# -----------------------------------------------------------------------------
timer = util.ProcTimer()

def _masks_to_boxes_np(masks: np.ndarray) -> np.ndarray:
    """
    Compute the bounding boxes around the provided masks.

    Args:
        masks (np.ndarray): Boolean masks of shape [n, h, w].

    Returns:
        np.ndarray: Bounding boxes [n, 4] in (x1, y1, x2, y2) format.
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

def apply_anony_mask(frame, masker_func=None, detector=None, mask=None, **kwargs) -> np.ndarray:
    """
    Apply an anonymization mask to a frame.

    Args:
        frame (np.ndarray): Input image.
        masker_func (callable): Function to apply for anonymization.
        detector: Not used here, for compatibility.
        mask (np.ndarray): Mask(s) to apply.
        **kwargs: Additional arguments for the masker_func.

    Returns:
        np.ndarray: Anonymized image.
    """
    use_adaptive_mask = kwargs.get("use_adaptive_mask", False)

    if masker_func is not None:
        frame_res = copy.deepcopy(frame) 
        if mask is None:
            # No mask: apply masker_func to the whole image
            frame_res = masker_func(frame, **kwargs) if detector is None else frame
        else:
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().numpy()
            if mask.ndim == 2:
                # for sematic segmentation mask
                frame_res_ = masker_func(frame, mask=mask, **kwargs)
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                frame_res[mask] = frame_res_[mask]      
                frame_res_ = copy.deepcopy(frame)
                frame_res[mask] = frame_res_[mask]  
            elif mask.ndim == 3 and mask.shape[0] > 0:
                # for instance segmentation mask
                max_num_pixels = np.prod(mask[0].shape)
                if use_adaptive_mask:
                    try:
                        boxes = _masks_to_boxes_np(np.array(list(filter(lambda x:x.sum()>0, mask))))
                    except Exception as ex:
                        print(ex)
                        print(f"mask shape = {mask.shape}, dtype = {mask.dtype}")
                        raise ex
                    for _, (mask_i, box_i) in enumerate(zip(mask, boxes)):
                        mask_scale = int(100*mask_i[:].sum()/max_num_pixels)
                        mask_scale = max(mask_scale, 1)
                        mask_i_ = np.repeat(mask_i[:, :, np.newaxis], 3, axis=2)    
                        frame_res_ = copy.deepcopy(frame)            
                        frame_res_[box_i[1]:box_i[3]+1, box_i[0]:box_i[2]+1, :] = masker_func(frame[box_i[1]:box_i[3]+1, box_i[0]:box_i[2]+1, :], 
                                                                                          mask_scale=mask_scale, image_size=frame_res.shape[:-1], 
                                                                                          **kwargs)
                        frame_res[mask_i_] = frame_res_[mask_i_]                     
                else:
                    frame_res_ = masker_func(frame, **kwargs)
                    mask = mask.sum(axis=0) > 0
                    mask = np.repeat(mask[..., np.newaxis], 3, axis=2)
                    frame_res[mask] = frame_res_[mask] 
        return frame_res

    return frame

class NonAnonymizer():
    """
    Dummy anonymizer that returns the input image unchanged.
    This class is used to return the original image without any anonymization, used as a placeholder when no anonymization is required.
    """
    def __init__(self) -> None:
        """
        Initialize the NonAnonymizer. 
        """
        self.anony_type = "image"
        self.detector_method = None
        self.anonymizer = {"method": None,  "kwargs":{}}
    
    @classmethod
    def get_anonymizers(cls):
        """
        Returns None, as no anonymizers are defined for NonAnonymizer.
        """
        return None 
     
    def get_anonymized(self, x: np.ndarray) -> np.ndarray:
        """
        Return the input image unchanged.

        Args:
            x (np.ndarray): Input image.

        Returns:
            np.ndarray: Unchanged image.
        """
        return x
    
    @property
    def get_anonymizer_settings(self):
        """
        Get anonymizer settings.

        Returns:
            dict: Anonymizer settings.
        """
        return self.anonymizer

class ConventionalImageAnonymizer():
    """
    Provides conventional image anonymization methods (blur, pixelization, silhouette, edge).
    Supports both image-level and body-level anonymization.
    """
    anonymizer_dict = {
                        "guassian_blur": {"kwargs": dict(kernelsize=(13, 13))},
                        "pixelization": {"kwargs": dict(downsize=4)},
                        "silhoutte": {"kwargs": dict()},
                        "edge": {"kwargs": dict(thr1=100, thr2=200)},
                    }

    def __init__(self, method: str, detector_name: str = "body__yolo", **kwargs) -> None:
        """
        Initialize the anonymizer.

        Args:
            method (str): Anonymization method name.
            detector_name (str): Detector name (default: "body__yolo").
            **kwargs: Additional arguments for anonymization and detection.
        """
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

        if self.anony_type not in ["body", "image"]:
            raise Exception(f"Undefined anonymize type: {self.anony_type}!")
        
        anonymizer_dispatcher_dict = {
            f"{alg}": {
                "method": getattr(self, alg),
                "kwargs": {**copy.deepcopy(v["kwargs"]), **self.kwargs_anony_method}
            }
            for alg, v in self.get_anonymizers().items()
        }
        self.anonymizer = anonymizer_dispatcher_dict.get(method, None)

        if self.anonymizer is None:
            raise f"Undefined anonymize method: {self.anonymizer}!"
        
        self.load_detector()
    
    @classmethod
    def get_anonymizers(cls):
        """
        Get available anonymizer methods.

        Returns:
            dict: Dictionary of anonymizer methods and their default kwargs.
        """
        return cls.anonymizer_dict
    
    def load_detector(self):
        """
        Load the object detector for body-level anonymization.
        """
        self.detector = PerFrameObjectDetector(detector_name=self.detector_method, 
                                               anony_type=self.anony_type, **self.kwargs_detector_method)
        self.detector.load_detector()

    def get_anonymized(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply the selected anonymization method to the input image.

        Args:
            x (np.ndarray): Input image.
            **kwargs: Additional arguments for the anonymizer.

        Returns:
            np.ndarray: Anonymized image.
        """    
        if self.anonymizer is None:
            # non-anonymization
            return x
        
        if self.detector_method is None:
            # image-level anonymization
            y_anony = self.anonymizer["method"](x, **self.anonymizer["kwargs"], **kwargs)
        else:
            # body-level anonymization
            y, boolean_masks = self.detector.get_detection(x)

            if boolean_masks is not None:
                frames_with_masks = [
                    apply_anony_mask(frame, 
                                    masker_func=self.anonymizer["method"], 
                                    mask=mask, **self.anonymizer["kwargs"], 
                                    **kwargs
                                )
                                for frame, mask in zip([y], boolean_masks) 
                            ]
                y_anony = frames_with_masks[0]
            else:
                y_anony = y
            
        return y_anony
    
    @property
    def get_anonymizer_settings(self):
        """
        Get anonymizer settings.

        Returns:
            dict: Anonymizer settings.
        """
        return self.anonymizer

    @staticmethod
    def silhoutte(x, **kwargs):
        """
        Apply silhouette anonymization to the image.

        Args:
            x (np.ndarray): Input image.
            **kwargs: alpha, color_format, color.

        Returns:
            np.ndarray: Silhouette-anonymized image.
        """
        alpha = kwargs.get("alpha", 1.0) # mask alpha
        color_format = kwargs.get("color_format", 'rgb')
        color = kwargs.get("color", '#190fd4')

        hex2rgb = util.convertcolor_hexstr2tuple(color, color_format=color_format)
        silhoute_image = np.concatenate(
            (
                np.full(x.shape[:2], hex2rgb[0], np.uint8)[:, :, np.newaxis],
                np.full(x.shape[:2], hex2rgb[1], np.uint8)[:, :, np.newaxis],
                np.full(x.shape[:2], hex2rgb[2], np.uint8)[:, :, np.newaxis]
            ),
            axis=2
        )
        x = cv2.addWeighted(x, 1 - alpha, silhoute_image, alpha, 1.0)
        return x

    @staticmethod
    def edge(x, **kwargs):
        """
        Apply edge detection anonymization to the image.
        The smallest value between thr1 and thr2 is used for edge linking. The largest value is used to find initial segments of strong edges.
        Those who lie between these two thresholds are classified edges or non-edges based on their connectivity. 
        If they are connected to "sure-edge" pixels, they are considered to be part of edges. Otherwise, they are also discarded. 

        Args:
            x (np.ndarray): Input image.
            **kwargs: thr1, thr2.

        Returns:
            np.ndarray: Edge-anonymized image.
        """
        thr1 = kwargs.get("thr1", 100) # thr1
        thr2 = kwargs.get("thr2", 200) # thr2
        assert thr1 < thr2, "thr1 should be less than thr2 for edge detection!"

        # convert the frame to grayscale for edge detection 
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) 
        
        # apply Gaussian blur to reduce noise and smoothen edges 
        # x = cv2.GaussianBlur(src=x, ksize=(5, 5), sigmaX=0.5) 

        # perform Canny edge detection 
        x = cv2.Canny(x, thr1, thr2)
        x = x[:, :, np.newaxis]
        x = np.concatenate(([x]*3), axis=2)
    
        return x

    @staticmethod
    def guassian_blur(x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Gaussian blur anonymization to the image.

        Args:
            x (np.ndarray): Input image.
            **kwargs: kernelsize, sigma, mask_scale, use_log_scaler, adaptive_sigma, alpha_mask_scale, alpha_dim_scale.

        Returns:
            np.ndarray: Blurred image.
        """
        kernelsize = kwargs.get("kernelsize", (13, 13)) # (x, y) direction
        sigma = kwargs.get("sigma", (0, 0)) # (x, y) direction, if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively 
        mask_scale = kwargs.get("mask_scale", 0) # 0 means no adaptive mask scaling, > 0 means mask scaling
        use_log_scaler = kwargs.get("use_log_scaler", "log") # use log scaling for mask scaling
        alpha_mask_scale = kwargs.get("alpha_mask_scale", 1.0) # mask scale for bounded adaptive AN
        alpha_dim_scale = kwargs.get("alpha_dim_scale", 0.5)  # limit for bounded adaptive AN
        adaptive_sigma = kwargs.get("adaptive_sigma", False) # True if sigma should be adapted to the mask size, False if sigma is fixed

        height, width = x.shape[:2]
        if (kernelsize[0]==0) | (kernelsize[1]==0):
            #  max adaptive Guassian blur AN
            try:
                kernelsize = [width, height]
                kernelsize = tuple([k - 1 if k % 2 == 0 else k for k in kernelsize]) # make kernel is odd number
                x = cv2.GaussianBlur(x, kernelsize, 0, 0) # sigma is auto estimated by opencv
            except Exception as ex:
                print(ex)
                print(f"x: {x.shape}, kernelsize: {kernelsize}, sigma:{sigma}")
                raise ex
        else:
            if mask_scale > 0: # bounded adaptive Guassian blur AN
                # im_box_scaler = np.log(mask_scale) if use_log_scaler else mask_scale
                mask_scale_max = 100 # max mask scale
                if isinstance(use_log_scaler, bool):
                    im_box_scaler = np.log(mask_scale) if use_log_scaler else mask_scale
                elif use_log_scaler == "log":
                    im_box_scaler = np.log(mask_scale) # ln(x)
                elif use_log_scaler == "exp": # e^x
                    im_box_scaler = np.exp(mask_scale)
                elif use_log_scaler == "quadratic":
                    im_box_scaler = mask_scale^2 # x^2
                elif use_log_scaler == "exp_norm": # e^x
                    norm_scaler = np.log(mask_scale_max) / np.exp(mask_scale_max)
                    im_box_scaler = norm_scaler * np.exp(mask_scale)
                elif use_log_scaler == "quadratic_norm":
                    norm_scaler = np.log(mask_scale_max) / (mask_scale_max^2)
                    im_box_scaler = norm_scaler  * (mask_scale^2) # x^2
                elif use_log_scaler == "linear_norm":
                    norm_scaler = np.log(mask_scale_max) / mask_scale_max
                    im_box_scaler = norm_scaler * mask_scale
                else:
                    im_box_scaler = mask_scale # use_log_scaler == "linear"
                    
                im_box_scaler = max(im_box_scaler, 1)
                r = alpha_mask_scale * im_box_scaler
                kernelsize = [int(max(r * k, alpha_mask_scale * k)) for k in kernelsize]
                kernelsize = [min(max(int(alpha_dim_scale * d), 1), k) for k, d in zip(kernelsize, [width, height])]
                if adaptive_sigma:
                    sigma = [int(max(r * s, alpha_mask_scale * s)) for s in sigma]
            kernelsize = tuple([k - 1 if k % 2 == 0 else k for k in kernelsize]) # make kernel is odd number

            try:
                sigma = min(sigma[0], kernelsize[0]), min(sigma[1], kernelsize[1])
                x = cv2.GaussianBlur(x, kernelsize, sigma[0], sigma[1])
            except Exception as ex:
                print(ex)
                print(f"x: {x.shape}, kernelsize: {kernelsize}, sigma:{sigma}")
                raise ex
        return x

    @staticmethod
    def pixelization(x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply pixelization anonymization to the image.

        Args:
            x (np.ndarray): Input image.
            **kwargs: downsize, mask_scale, use_log_scaler, alpha_mask_scale, alpha_dim_scale.

        Returns:
            np.ndarray: Pixelized image.
        """
        downsize = int(kwargs.get("downsize", 4))
        mask_scale = kwargs.get("mask_scale", 0) # 0 means no adaptive mask scaling, > 0 means mask scaling
        use_log_scaler = kwargs.get("use_log_scaler", "log") # use log scaling for mask scaling
        alpha_mask_scale = kwargs.get("alpha_mask_scale", 1.0) # mask scale for bounded adaptive AN
        alpha_dim_scale = kwargs.get("alpha_dim_scale", 0.5) # limit for bounded adaptive AN

        height, width = x.shape[:2]
        if downsize == 0:
            # max pixelization 
            try:
                downsize_dims = (1, 1)
                # Resize input to pixelated size by downsize_dims
                tmp = cv2.resize(x, downsize_dims, interpolation=cv2.INTER_LINEAR)
                # restore image to original size
                x = cv2.resize(tmp, (width, height), interpolation=cv2.INTER_NEAREST)
            except Exception as ex:
                print(f"input_width_height:{width}x{height}, last x.shape[hxwxc]:{x.shape}, mask_scale:{mask_scale}, downsize: {downsize}, downsize_dims: {downsize_dims}")
                raise ex
        else:
            assert downsize >= 1
            if mask_scale > 0: # bounded adaptive Guassian blur AN
                # im_box_scaler = np.log(mask_scale) if use_log_scaler else mask_scale
                mask_scale_max = 100 # max mask scale
                if isinstance(use_log_scaler, bool):
                    im_box_scaler = np.log(mask_scale) if use_log_scaler else mask_scale
                elif use_log_scaler == "log":
                    im_box_scaler = np.log(mask_scale) # ln(x)
                elif use_log_scaler == "exp": # e^x
                    im_box_scaler = np.exp(mask_scale)
                elif use_log_scaler == "quadratic":
                    im_box_scaler = mask_scale^2 # x^2
                elif use_log_scaler == "exp_norm": # e^x
                    norm_scaler = np.log(mask_scale_max) / np.exp(mask_scale_max)
                    im_box_scaler = norm_scaler * np.exp(mask_scale)
                elif use_log_scaler == "quadratic_norm":
                    norm_scaler = np.log(mask_scale_max) / (mask_scale_max^2)
                    im_box_scaler = norm_scaler  * (mask_scale^2) # x^2
                elif use_log_scaler == "linear_norm":
                    norm_scaler = np.log(mask_scale_max) / mask_scale_max
                    im_box_scaler = norm_scaler * mask_scale
                else:
                    im_box_scaler = mask_scale # use_log_scaler == "linear"
                    
                im_box_scaler = max(im_box_scaler, 1)
                r = alpha_mask_scale * im_box_scaler
                downsize_target = [int(max(r*downsize, alpha_mask_scale * downsize))] * 2
                downsize = tuple([min(max(int(alpha_dim_scale*d), 1), dt) for d, dt in zip([width, height], downsize_target)])
            else:
                downsize =  tuple([downsize if d > downsize else d for d in [width, height]])

            try:
                downsize_dims = (width // downsize[0], height // downsize[1])
                # Resize input to pixelated size  by downsize_dims
                tmp = cv2.resize(x, downsize_dims, interpolation=cv2.INTER_LINEAR)
                # restore image to original size
                x = cv2.resize(tmp, (width, height), interpolation=cv2.INTER_NEAREST)
            except Exception as ex:
                print(f"input_width_height:{width}x{height}, last x.shape[hxwxc]:{x.shape}, mask_scale:{mask_scale}, downsize: {downsize}, downsize_dims: {downsize_dims}")
                raise ex
            
        return x

class AnonymizerHub():
    """
    Main interface for anonymization. Selects and applies the appropriate anonymizer.
    """
    def __init__(self, method: str, **kwargs) -> None:
        """
        Initialize the AnonymizerHub.

        Args:
            method (str): Anonymization method name.
            **kwargs: Additional arguments for the anonymizer.
        """
        self.method = method
        self.kwargs = kwargs
        self.detector_name = kwargs.get("detector_name", None) 
        self.detector_name = self.detector_name if self.detector_name is not None else "image__None"
        print(self.kwargs)
        self.load_anonymizer()

    def load_anonymizer(self) -> None:
        """
        Load the appropriate anonymizer based on the method.
        """
        if self.method in ConventionalImageAnonymizer.get_anonymizers().keys():
            self.anonymizer = ConventionalImageAnonymizer(method=self.method, **self.kwargs)
        else:
            print(f"Undefined anonymize method: {self.method}. Thus, enforcing using raw footage.")
            self.anonymizer = NonAnonymizer()
        
        print(self.anonymizer.get_anonymizer_settings)
        
    def anonymize_frame(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Anonymize a single frame.

        Args:
            frame (np.ndarray): Input image.
            **kwargs: Additional arguments for the anonymizer.

        Returns:
            np.ndarray: Anonymized image.
        """
        frame_res = self.anonymizer.get_anonymized(frame, **kwargs)
        return frame_res

    def anonymize(self, frames: list, **kwargs) -> list:
        """
        Anonymize a list of frames or a single frame.

        Args:
            frames (list or np.ndarray): List of images or a single image. [n, h, w, 3] or [h, w, 3]
            **kwargs: Additional arguments for the anonymizer.

        Returns:
            list or np.ndarray: Anonymized images.
        """
        islist = isinstance(frames, list)
        if not islist:
            frames = [frames] 
        f = lambda x: self.anonymize_frame(x, **kwargs)
        frames = list(map(f, tqdm(frames)))
        if not islist:
            return frames[0]
        return frames

class ANInference():
    """
    AN Integration and Inference Engine.
    Handles initialization and inference for anonymization methods.
    """
    def __init__(self, anony_method: str, **kwargs):
        """
        Initialize the ANInference object.
        
        Args:
            anony_method (str): Name of the anonymization method to use.
            **kwargs: Additional configuration parameters for the anonymizer.
        """
        self.preprocessors_list = cfg.preprocessors_list
        self.postprocessors_list = cfg.postprocessors_list
        self.anony_method = anony_method

        self.init_anonymizer(**kwargs)

        self.input_frames_raw = []
        self.input_frames_transformed = []
        self.an_proc_time = 0.0

    def init_anonymizer(self, kwargs_anony_method: dict = {}, kwargs_detector_method: dict = {}, **kwargs):
        """
        Initialize the anonymizer object with the selected method and parameters.

        Args:
            kwargs_anony_method (dict): Additional keyword arguments for the anonymization method.
            kwargs_detector_method (dict): Additional keyword arguments for the detector method.
            **kwargs: Additional configuration parameters for the anonymizer.
        """
        print("init_anonymizer...")
        self.anony_method_kwargs = cfg.anny_transform_dict[cfg.anony_method_name_mapper_dict[self.anony_method]]
        self.anony_method_kwargs['kwargs_anony_method']["preprocessors_list"] = self.preprocessors_list
        self.anony_method_kwargs['kwargs_anony_method']["postprocessors_list"] = self.postprocessors_list
        self.anonyObj = AnonymizerHub(method=self.anony_method_kwargs['anony_method'], 
                                detector_name=self.anony_method_kwargs['detector_name'],
                                kwargs_anony_method={**self.anony_method_kwargs['kwargs_anony_method'], **kwargs_anony_method}, 
                                kwargs_detector_method={**self.anony_method_kwargs['kwargs_detector_method'], **kwargs_detector_method},
                                target_resolution=None,
                                isreset_cache=False,
                                )

    def inference(self, frames: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """
        Run anonymization on input frames.
        Args:
            frames: Input image or video frames (BGR).
        Returns:
            success: Boolean indicating success.
            frames_transformed: Anonymized frames.
        """
        success = False 
        timer.restart()
        try:
            frames_transformed = self.anonyObj.anonymize(frames)
            success = True
        except Exception as ex:
            print(f"AN ERROR: {ex}")
            frames_transformed = None
            
        self.an_proc_time = np.round(timer.get_proctime(time_format="s"), 5)
        timer.stop()

        return success, frames_transformed 
    
    def add_an_status_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay anonymization status on the frame.
        Args:
            frame: RGB frame.
        Returns:
            frame: Frame with overlay.
        """
        _ = cv2.putText(frame, AN_TEXT.format(self.an_proc_time), AN_TEXT_COORD, TEXT_FACE, TEXT_SCALE, (255,0,0), TEXT_THICKNESS, cv2.LINE_AA)
        return frame
