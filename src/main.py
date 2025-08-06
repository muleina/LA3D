# =============================================================================
# LA3D: Lightweight Anonymization (AN) and Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides an integrated pipeline for computationally lightweight video anonymization for video anomaly detection application.
# It supports multiple anonymization methods (masking, blurring, pixelization, etc.)
# and anomaly detection models (PEL4VAD, MGFN) for processing images, videos, or webcam streams.
#
# Main Features:
#   - Anonymization of persons in images/videos using various methods.
#   - Anomaly detection in video streams using deep learning models.
#   - Visualization and saving of results.
#   - Flexible command-line interface for configuration.
#
# Author: Mulugeta Weldezgina Asres
# Date: October 2024
# =============================================================================

import os, sys, gc
import numpy as np
import cv2
import argparse
import torch
# torch.classes.__path__ = []
import time 
import matplotlib.pyplot as plt
# import moviepy.editor as mpy
from pathlib import Path

import utilities as util
from anony_interface import AnonymizerHub
from ad_interface import VideoFeatureExtractor, pel4vad_interface, pel4vad_configs, mgfn_interface, mgfn_configs
from config import config as cfg

# -----------------------------------------------------------------------------
# Path and Device Setup
# -----------------------------------------------------------------------------
src_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.dirname(src_path)
sys.path.append(src_path)
sys.path.append(main_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# Visualization Parameters for Anomaly Detection (AD) Panel
# -----------------------------------------------------------------------------
AD_TEXT = "AD: {0:0.2f} Alert: "
R = 4
# AD_CENTER = [(4*R, 30), (6*R, 30), (8*R, 30)]
TEXT_OFFSET = len(AD_TEXT)+R
FLAG_COORDS = [(TEXT_OFFSET*R, 30), (TEXT_OFFSET*R+2*R, 30), (TEXT_OFFSET*R+4*R, 30)]
TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.09*R
TEXT_THICKNESS = 0
# AD_TEXT_COORD = AD_CENTER[0][0]-4*R, AD_CENTER[0][1] +1*R# x, y
AD_TEXT_COORD = [0, FLAG_COORDS[0][1]+1*R]
STATUS_REC_COORDS = (0, FLAG_COORDS[0][1]-2*R), (FLAG_COORDS[-1][0]+2*R, FLAG_COORDS[0][1]+2*R)

# -----------------------------------------------------------------------------
# Timer Utility
# -----------------------------------------------------------------------------
timer = util.ProcTimer()

class ANInference():
    """
    AN Integration and Inference Engine.
    Handles initialization and inference for anonymization methods.
    """
    def __init__(self, anony_method, **kwargs):
        self.preprocessors_list = cfg.preprocessors_list
        self.postprocessors_list = cfg.postprocessors_list
        self.anony_method = anony_method

        self.init_anonymizer(**kwargs)

        self.input_frames_raw = []
        self.input_frames_transformed = []
        self.an_proc_time = 0.0

    def init_anonymizer(self, kwargs_anony_method={}, kwargs_detector_method={}, **kwargs):
        """
        Initialize the anonymizer object with the selected method and parameters.
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

    def inference(self, frames):
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
    
class VADetector():
    """
    VAD Integration Engine.
    Handles initialization and inference for video anomaly detection models.
    """
    def __init__(self, ad_modelname, kwargs_ad_method={}, **kwargs):
        self.ad_modelname = ad_modelname
        self.ad_model_src = kwargs_ad_method.get("ad_model_src", cfg.ad_model_src)
        self.enc_frequency = kwargs_ad_method.get("enc_frequency", cfg.enc_vid_enc_frame_seq_size)
        self.ad_thr = kwargs_ad_method.get("ad_thr", cfg.ad_thr)
        
        if ad_modelname in ["pel"]:
            # Initialize PEL4VAD model and feature extractor
            _pel4vad_cfg = pel4vad_configs.build_config(self.ad_model_src)
            self.objVADModel = pel4vad_interface.load_model(_pel4vad_cfg.ckpt_path, datasource=self.ad_model_src).to(device)
            self.objVADModel.eval()
            
            self.vidfeature_i3d_extractor = VideoFeatureExtractor(pretrainedpath=rf"{_pel4vad_cfg.enc_vid_model_filepath}", 
                                                            frequency=self.enc_frequency, feature_dim=_pel4vad_cfg.enc_vid_feature_dim, sample_T=_pel4vad_cfg.enc_vid_num_crops)
            
            self.predict = lambda x: pel4vad_interface.predict(torch.tensor(x[:, :, :-1], dtype=torch.float32), self.objVADModel, frequency=self.vidfeature_i3d_extractor.frequency)

        elif ad_modelname in ["mgfn"]:
            # Initialize MGFN model and feature extractor            
            _mgfn_cfg = mgfn_configs.build_config(self.ad_model_src)
            self.objVADModel = mgfn_interface.load_model(filepath=_mgfn_cfg.ckpt_path, channels=_mgfn_cfg.feature_dim).to(device)
            self.objVADModel.eval()
            self.vidfeature_i3d_extractor = VideoFeatureExtractor(pretrainedpath=rf"{_mgfn_cfg.enc_vid_model_filepath}", 
                                                            frequency=self.enc_frequency, feature_dim=_mgfn_cfg.enc_vid_feature_dim, sample_T=_mgfn_cfg.enc_vid_num_crops)
            
            self.predict = lambda x: mgfn_interface.predict(torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0), self.objVADModel, frequency=self.vidfeature_i3d_extractor.frequency, device=device)
        else:
            raise "Undefined ad_modelname!"

    def inference(self, video_frames):
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
    def __init__(self, ObjVADetector, *args, **kwargs):

        self.ObjVADetector = ObjVADetector
        self.ad_thr = ObjVADetector.ad_thr
        self.ad_proc_time = 0.0

    def get_ad_status_offline(self, frames=None):
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
    
    def add_ad_status_to_frame(self, frame, ad_score):
        """
        Overlay anomaly detection status and score on the frame.
        Args:
            frame: RGB frame.
            ad_score: Anomaly score.
        Returns:
            frame: Frame with overlay.
        """
        frame = cv2.rectangle(frame, STATUS_REC_COORDS[0], STATUS_REC_COORDS[1], (127, 127, 127), -1) # add status panel
        ad_flag_color = (255,0,0) if ad_score >= self.ad_thr else (0,255,0)
        for i in range(3):
           _ = cv2.circle(frame, FLAG_COORDS[i], R, ad_flag_color, -1) 
        
        _ = cv2.putText(frame, AD_TEXT.format(ad_score) , AD_TEXT_COORD, TEXT_FACE, TEXT_SCALE, (0,0,255), TEXT_THICKNESS, cv2.LINE_AA)

        return frame

def an_inference_engine(*args, **kwargs):
    """
    Init inference engine object for AN
    """
    return ANInference(*args, **kwargs)

def video_ad_model_inference_engine(*args, **kwargs):
    """
    Init inference engine for VAD
    """
    objVADModel = VADetector(*args, **kwargs)
    return ADInference(objVADModel, *args, **kwargs)

def app_init(app_mode, anony_method="no-an", ad_method="pel", **kwargs):
    """
    Initialize anonymization and anomaly detection engines based on app mode.
    Args:
        app_mode: "an", "ad", or "an-ad"
        anony_method: Anonymization method.
        ad_method: Anomaly detection method.
    Returns:
        axisANObj: ANInference object.
        axisADObj: ADInference object.
    """    
    print("app_init...")

    axisANObj = None
    axisADObj = None
    
    if "ad" in app_mode:
        axisADObj = video_ad_model_inference_engine(ad_modelname=ad_method, **kwargs)
    if "an" in app_mode:
        axisANObj = an_inference_engine(anony_method, **kwargs)
    else:
        anony_method = "no-an"
        axisANObj = an_inference_engine(anony_method, **kwargs)
       
    return axisANObj, axisADObj

if device == 'cuda':
    gc.collect()
    torch.cuda.empty_cache()

def main(**kwargs):
    """
    Main entry point for the LA3D application.
    Handles argument parsing, initialization, and processing for different input formats.
    Args:
        kwargs: Configuration parameters.
    """
    
    app_mode = kwargs.get("app_mode", "an")

    input_format = kwargs.get("input_format", "webcam")
    input_datapath = kwargs.get("input_datapath", None)
    input_imgsz = kwargs.get("input_imgsz", [320, 240])
    fps = kwargs.get("fps", None)
    

    object_detection_classes = kwargs.get("object_detection_classes", cfg.object_detection_classname_list)
    object_detection_imgsz = kwargs.get("object_detection_imgsz", cfg.object_detection_imgsz)
    object_detection_thr = kwargs.get("object_detection_thr", cfg.object_detection_thr)
    
    anony_method = kwargs.get("anony_method", "mask")
    
    alpha_mask_scale = kwargs.get("alpha_mask_scale", 1.0)
    alpha_dim_scale = kwargs.get("alpha_dim_scale", 0.5)

    ad_method = kwargs.get("ad_method", "pel")
    ad_model_src = kwargs.get("ad_model_src", "xd")
    ad_thr = kwargs.get("ad_thr", cfg.ad_thr)

    visualize = kwargs.get("visualize", True)
    issave = kwargs.get("issave", False)
    output_dir = kwargs.get("output_dir", "results")

    print(f"APP MODE: {app_mode.upper()}")

    if issave:
        output_dirpath = Path(rf"{main_path}/{output_dir}")
        output_dirpath.mkdir(exist_ok=True, parents=True)

    AN_TEXT = (f"ANONYMIZED: {anony_method.upper()}" if anony_method!="no-an" else "NON-ANONYMIZED")    

    kwargs_detector_method = {"classes":[cfg.object_detection_classid_mapper_dict[classname] for classname in object_detection_classes], "detection_thr":object_detection_thr, "imgsz":[int(s) for s in object_detection_imgsz]}
    kwargs_anony_method = {"target_resolution":None, "alpha_dim_scale":alpha_dim_scale, "alpha_mask_scale":alpha_mask_scale, "color_format":"bgr"}
    kwargs_ad_method = {"ad_model_src":ad_model_src, "ad_thr":ad_thr}

    axisANObj, axisADObj = app_init(app_mode, anony_method=anony_method, ad_method=ad_method,
                                    kwargs_detector_method=kwargs_detector_method, 
                                    kwargs_anony_method=kwargs_anony_method,
                                    kwargs_ad_method=kwargs_ad_method,
                                    )

    ########################################################################################
    # Start main processor


    image_width, image_height = None, None
    if input_imgsz is not None:
        input_imgsz = [int(d) for d in input_imgsz]
        assert len(input_imgsz) == 2, "the dimension of input_imgsz must be 2: [wxh]"
        image_width, image_height = input_imgsz[0], input_imgsz[1]

    if input_format == "webcam":
        print("initializing webcam...")
        fps = 5 if fps is None else fps
        camObj = None
        try:
            webcam_id = 0
            camObj = util.VideoCameraThreadQueue(webcam_id, width=image_width, height=image_height, fps=fps) # change this if webcam not on 0 incase of multiple webcams on the system

            print("initializing webcam done!")
        except Exception as ex:
            camObj = None
            print(f"Webcam {webcam_id} connection failed! {ex}")

        while True and (camObj is not None):
            img = camObj.read() # BGR
            if img is None:
                print(f"Waiting for webcam data stream...")
                time.sleep(1.0)
                continue

            print(f"webcam is streaming...")
            
            an_success, frame = axisANObj.inference(img)
            _ = cv2.putText(frame, f"PT:{axisANObj.an_proc_time:0.03f}s", AD_TEXT_COORD, TEXT_FACE, TEXT_SCALE, (255,0,0), TEXT_THICKNESS, cv2.LINE_AA)

            cv2.imshow(f"{AN_TEXT} (press q to exit)", frame)
            if cv2.waitKey(1) == ord('q'):
                camObj.stop()
                break
    
    elif input_format == "image":
        img = cv2.imread(rf"{input_datapath}") # BGR
        if input_imgsz is not None:
            print(f"input image size: {img.shape}")
            img = util.image_resize(img, input_imgsz, mode="cv", ref="wh")
            print(f"input image resize: {img.shape}")

        # AN inference
        an_success, frame = axisANObj.inference(img)
        if an_success:
            print("AN: {:0.03f} secs".format(axisANObj.an_proc_time))

            if issave:
                input_datapath = Path(input_datapath)
                filename = ".".join(input_datapath.name.split(".")[:-1])
                print(str(input_datapath))

                result_filename_template = f'{anony_method}_imwh{frame.shape[1]}x{frame.shape[0]}_{filename}'
                if output_dirpath is not None:  
                    output_filepath = rf"{output_dirpath}/{result_filename_template}_an.jpg"
                    print(output_filepath)
                    util.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(output_filepath)                

            if visualize or not issave:
                cv2.imshow(f"{AN_TEXT} (press q to exit)", frame)
                if cv2.waitKey() == ord('q'):
                    cv2.destroyAllWindows()

    elif input_format == "video":
        print("loading video footage...")

        def an_ad_offline_result_presenter(video_frames):
            """
            video_frames: BGR, [b, h, w, 3]
            ad_frame_list: RGB, [b, h, w, 3]
            """
            # AN inference
            an_success, frames = axisANObj.inference(video_frames)

            # Visualization           
            print("AN: {:0.03f} secs".format(axisANObj.an_proc_time))

            ad_score_fig, ad_frame_list = None, frames

            if frames is not None:
                f = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype("uint8")
                frames = list(map(f, frames))
                
                # AD inference
                if axisADObj is not None:
                    ad_success, ad_scores = axisADObj.get_ad_status_offline(frames)
                    print("AD: {:0.03f} secs".format(axisADObj.ad_proc_time))

                    ad_score_fig, ax = plt.subplots(figsize=(4, 2.5))
                    _  = ax.plot(ad_scores, label="VAD Score", color="blue")
                    _  = ax.plot([axisADObj.ad_thr]*len(ad_scores), label=f"VAD Threshold", color="gray", linestyle="--")
                    _  = ax.set_title(f"Anomaly Detection (dThr={axisADObj.ad_thr})", fontsize=10)
                    _  = plt.xlabel("Frames", fontsize=10)
                    _  = plt.ylabel("Value", fontsize=10)
                    _  = ax.legend(fontsize=8)

                    if ad_scores is not None:
                        ad_frame_list = []
                        for i, frame in enumerate(frames[:len(ad_scores)]):
                            frame = axisADObj.add_ad_status_to_frame(frame, ad_scores[i])
                            ad_frame_list.append(frame)

            return ad_score_fig, ad_frame_list
    
        timer.restart()
        video_frames, video_meta = util.videofile_loader(input_datapath, start_sec=0, clip_duration=None, fps=fps, target_resolution=input_imgsz[::-1])

        frame_size = video_frames[0].shape
        video_length = len(video_frames)
        print(f"videofile frame length to process: {video_length}, frame_size: {frame_size}")
        ad_score_fig, ad_frame_list = an_ad_offline_result_presenter(video_frames)
       
        if issave:
            input_datapath = Path(input_datapath)
            filename = ".".join(input_datapath.name.split(".")[:-1])
            print(str(input_datapath))
            result_filename_template = f'{ad_method}_{ad_model_src}_{anony_method}_imwh{frame_size[1]}x{frame_size[0]}_{filename}'

            if output_dirpath is not None:
                if ad_score_fig is not None:
                    util.save_figure(f'{result_filename_template}_ad', ad_score_fig, filepath=output_dirpath)

                if ad_frame_list is not None:
                    mpyVidObj = util.mpy.ImageSequenceClip(ad_frame_list, fps=video_meta["fps"])
                    mpyVidObj.write_videofile(rf'{output_dirpath}/{result_filename_template}_ad.mp4', fps=video_meta["fps"], codec="libx264")
                    mpyVidObj.close()
        
        if (visualize or not issave) and (ad_frame_list is not None):
            for frame in ad_frame_list:
                cv2.imshow(f"{AN_TEXT}-{app_mode.upper()} (press q to exit)", frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break
    else:
        raise f"Undefine input format! place choose from {cfg.input_data_source_formats}"               

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="LA3D: Lightweight Anonymization (AN) for VAD")
    print("#"*80)

    parser.add_argument('-a', '--app_mode', type=str, default="an",  choices=["an", "ad", "an-ad"], help="an: for AN , ad: for VAD and an-ad: integration of AN and AD.")

    parser.add_argument('-if', '--input_format', type=str, default="webcam", choices=["webcam", "image", "video"], help="app mode an is supported for all, while ad is for video files.")
    parser.add_argument('-id', '--input_datapath', type=str, default=None)
    parser.add_argument('-is', '--input_imgsz', nargs='+', default=[320, 240])
    parser.add_argument('-fps', '--fps', type=int, default=None)

    parser.add_argument('-anm', '--anony_method', type=str, default="mask", choices=cfg.anony_method_options_list)
    parser.add_argument('-odc', '--object_detection_classes', nargs='+', default=cfg.object_detection_classname_list)
    parser.add_argument('-ods', '--object_detection_imgsz', nargs='+', default=cfg.object_detection_imgsz)
    parser.add_argument('-odt', '--object_detection_thr', type=float, default=cfg.object_detection_thr)
    parser.add_argument('-anr', '--alpha_mask_scale', type=float, default=1.0)
    parser.add_argument('-anl', '--alpha_dim_scale', type=float, default=0.5)

    parser.add_argument('-adm', '--ad_method', type=str, default=cfg.ad_method_options_list[0])
    parser.add_argument('-ads', '--ad_model_src', type=str, choices=["ucf", "xd"], default=cfg.ad_model_src)
    parser.add_argument('-adt', '--ad_thr', type=float, default=cfg.ad_thr)

    parser.add_argument('-v', '--visualize', default=False, action="store_true")
    parser.add_argument('-s', '--issave', default=False, action="store_true")
    parser.add_argument('-od', '--output_dir', type=str, default=cfg.result_path)

    args = parser.parse_args()
    print(args)
    args = vars(args)
    main(**args)   

    print("#"*80)


# python main.py -a an -if webcam -anm mask -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if webcam -anm no-an -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if webcam -anm edge -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if webcam -anm blur -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if webcam -anm adaptive_blur -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if webcam -anm adaptive_full_blur -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if webcam -anm adaptive_max_blur -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if webcam -anm pixelization -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if webcam -anm adaptive_pixelization -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if webcam -anm adaptive_max_pixelization -odc person -ods 320 240 -odt 0.25 -v

# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm no-an -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm mask -odc person -ods 320 240 -odt 0.25 -v
# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm mask -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm edge -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm adaptive_blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm adaptive_full_blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm adaptive_max_blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm pixelization -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm adaptive_pixelization -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an -if image -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\VISPR\2017_17368641.jpg" -anm adaptive_max_pixelization -odc person -ods 320 240 -odt 0.25 -s

# python main.py -a an -if video -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm mask -odc person -ods 320 240 -odt 0.25 -v 
# python main.py -a an-ad -ads ucf -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm no-an -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads ucf -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm mask -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads ucf -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm edge -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads ucf -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads ucf -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm adaptive_blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads ucf -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm adaptive_full_blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads ucf -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm adaptive_max_blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads ucf -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm pixelization -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads ucf -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm adaptive_pixelization -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads ucf -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\UCF_Crime\Burglary033_x264.mp4" -anm adaptive_max_pixelization -odc person -ods 320 240 -odt 0.25 -s

# python main.py -a an -if video -id "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm mask -odc person -ods 320 240 -odt 0.25 -v 
# python main.py -a an-ad -ads xd -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm no-an -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads xd -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm mask -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads xd -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm edge -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads xd -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads xd -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm adaptive_blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads xd -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm adaptive_full_blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads xd -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm adaptive_max_blur -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads xd -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm pixelization -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads xd -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm adaptive_pixelization -odc person -ods 320 240 -odt 0.25 -s
# python main.py -a an-ad -ads xd -if video -id  "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\LA3D\data\XD_Violence\Fast.Five.2011__#00-32-56_00-33-26_label_B2-0-0.mp4" -anm adaptive_max_pixelization -odc person -ods 320 240 -odt 0.25 -s