# =============================================================================
# LA3D: Lightweight Anonymization (AN) and Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides Configs for LA3D
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: October 2024
# =============================================================================
input_data_source_formats = ["webcam", "image", "video"]
input_imgsz = (320, 240) # w

object_detector_model_version = "yolov8m-seg.pt"
# object_detector_model_version = yolo11m-seg.pt"
object_detection_model_dict = dict(yolo={ 
                                        'semantic': {'model':f'{object_detector_model_version}', 'kwargs':{}},
                                        'instant': {'model':f'{object_detector_model_version}', 'kwargs':{}} 
                                        })
object_detection_classid_mapper_dict = {"person":0, "car":2, "motorcycle":3, "bicycle":1, 
                            "cell phone":67, "laptop":63, "tv":62, 
                            "handbag":26, "backpack":24, "umbrella":25, "suitcase":28}
object_detection_classname_list = ["person", "cell phone", "laptop", "tv"]
object_detection_thr = 0.15
object_detection_imgsz = (320, 240)
color_format = "bgr"
detector_name = "body__yolo"
kwargs_detector_method = {"seg_type": "instant", "detector_version": object_detector_model_version, "detection_thr":object_detection_thr, 
                                                        "classes":[object_detection_classid_mapper_dict[item] for item in object_detection_classname_list]}

anny_transform_dict = {
                        "RAW_IMAGE": dict(anony_method="raw", 
                                    kwargs_anony_method={"color_format":color_format, }, 
                                    detector_name="image", 
                                    kwargs_detector_method=kwargs_detector_method
                                    ),
                        f"MASKED": dict(anony_method="silhoutte",    
                                        kwargs_anony_method={"color_format":color_format, "alpha": 1.0, "color":"#000000"}, 
                                        detector_name=detector_name, 
                                        kwargs_detector_method=kwargs_detector_method
                                        ),
                        f"EDGED": dict(anony_method="edge", 
                                    kwargs_anony_method={"color_format":color_format, "thr1": 100, "thr2":200}, 
                                    detector_name=detector_name, 
                                    kwargs_detector_method=kwargs_detector_method
                                    ),
                        "BLURRED_NA": dict(anony_method = "guassian_blur", 
                                    kwargs_anony_method={"color_format":color_format, "kernelsize": (13, 13), "sigma": (10,10), "use_adaptive_mask":False}, 
                                    detector_name=detector_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                ),
                        "BLURRED_L_A": dict(anony_method = "guassian_blur", 
                                    kwargs_anony_method={"color_format":color_format, "kernelsize": (13, 13), "sigma":(10,10), "use_adaptive_mask":True, "adaptive_sigma":False, "alpha_mask_scale":1.0, "alpha_dim_scale":0.5, "use_log_scaler":True}, 
                                    detector_name=detector_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                ),
                        "BLURRED_L_A_FULL": dict(anony_method = "guassian_blur", 
                                    kwargs_anony_method={"color_format":color_format, "kernelsize": (13, 13), "sigma":(10,10), "use_adaptive_mask":True, "adaptive_sigma":True, "alpha_mask_scale":1.0, "alpha_dim_scale":0.5, "use_log_scaler":True}, 
                                    detector_name=detector_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                ),
                        "BLURRED_L_A_MAX": dict(anony_method = "guassian_blur", 
                                    kwargs_anony_method={"color_format":color_format, "kernelsize": (0, 0), "sigma":(0,0), "use_adaptive_mask":True}, 
                                    detector_name=detector_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                ),
                        "PIXELIZED_NA": dict(anony_method="pixelization",    
                                    kwargs_anony_method={"color_format":color_format, "downsize": 4, "use_adaptive_mask":False}, 
                                    detector_name=detector_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                    ),
                        "PIXELIZED_L_A": dict(anony_method="pixelization",    
                                    kwargs_anony_method={"color_format":color_format, "downsize": 4, "use_adaptive_mask":True, "alpha_mask_scale":1.0, "alpha_dim_scale":0.5, "use_log_scaler":True}, 
                                    detector_name=detector_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                    ),
                        "PIXELIZED_L_A_MAX": dict(anony_method="pixelization",    
                                        kwargs_anony_method={"color_format":color_format, "downsize": 0, "use_adaptive_mask":True}, 
                                        detector_name=detector_name, 
                                        kwargs_detector_method=kwargs_detector_method,  
                                        )
                        }
anony_method_name_mapper_dict = {
                                "no-an":"RAW_IMAGE",
                                "mask":"MASKED", 
                                "edge":"EDGED", 
                                "blur":"BLURRED_NA", 
                                "adaptive_blur":"BLURRED_L_A", 
                                "adaptive_full_blur":"BLURRED_L_A_FULL", 
                                "adaptive_max_blur":"BLURRED_L_A_MAX", 
                                "pixelization":"PIXELIZED_NA",
                                "adaptive_pixelization":"PIXELIZED_L_A",
                                "adaptive_max_pixelization":"PIXELIZED_L_A_MAX",
                            }
anony_method_options_list = list(anony_method_name_mapper_dict.keys())
preprocessors_list = []
postprocessors_list = []

enc_vid_enc_frame_seq_size = 16
ad_method_options_list = ["pel", "mgfn"]
ad_model_src = "xd"
ad_thr = 0.5

result_path = "results/"
