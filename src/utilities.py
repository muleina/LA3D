# =============================================================================
# LA3D: Lightweight Anonymization (AN) and Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides utils functions for LA3D
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: October 2024
# =============================================================================
from threading import Thread
import queue 
# import glob
import cv2
import os, sys
# import torch
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import time
from pathlib import Path
from PIL import Image
from typing import Tuple

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)

global data_path
global figure_path
global src_path
global save_dir
global load_dir

def get_working_dirs(base_path):
    src_path = "{}//src//".format(base_path)
    data_path = "{}//data//".format(base_path)
    result_path = "{}//results//".format(base_path)
    return src_path, data_path, result_path

root_path = os.path.dirname(current_path)
src_path, data_path, result_path = get_working_dirs(root_path)
figure_path = result_path + "//figures//"
save_dir = load_dir = root_path + "//data//"
print(root_path, src_path, data_path)

class ProcTimer():
    def __init__(self):
        self.start_time = time.time()
        self.end_time = 0
        self.proc_time = 0
        self.time_status = True

    def restart(self):
        self.start_time = time.time()
        self.end_time = 0
        self.time_status = True

    def stop(self):
        self.time_status = False
        self.end_time = time.time()
        self.proc_time = self.end_time - self.start_time

    def get_proctime(self, time_format="s"):
        if self.time_status:
            self.end_time = time.time()
            self.proc_time = self.end_time - self.start_time

        if time_format == "s":
            return self.proc_time

    def display_proctime(self, time_format="s"):
        print("process time: {:0.5} seconds.".format(
            self.get_proctime(time_format=time_format)))

class VideoCameraThreadQueue:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src, timeout=0.0, fps=5, width=640, height=480, **kwargs):
        
        self.stopped = True

        try:
            self.stream = cv2.VideoCapture(src)
            # set settings
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_FPS, fps)

        except Exception as ex:
            print(f"Video capture init error: {ex}. Retying with CAP_DSHOW...")
            try:
                self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW) # to avoid error global cap_msmf.cpp:1759 CvCapture_MSMF::grabFrame videoio(MSMF): can't grab frame. Error: -1072873821
                # set settings
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.stream.set(cv2.CAP_PROP_FPS, fps)
            except Exception as ex:
                print(f"VideoCapture init error: {ex}")
                self.stream = None
                self.stop()

        self.start()
        # _ = self.stream.read()
        self.stopped = False

    def start(self):   
        try:
            self.q = queue.Queue()
            self.stream_thread = Thread(target=self._reader, daemon=True)
            self.stopped = False
            self.stream_thread.start()
        except Exception as ex:
            print(f"Thread init failed: {ex}")
            self.stopped = True
            self.stop()

        return self

    def _reader(self):
        while (not self.stopped) and self.stream.isOpened():
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                break
                # continue

            if not self.q.empty():
                try:
                    self.q.get_nowait()   
                except queue.Empty:
                    pass

            self.q.put(self.frame)

    def read(self):
        # return self.q.get()
    
        try:
            return self.q.get()
            # return self.q.get(block=True, timeout=0.1) 
        except queue.Empty:
            if self.stream is None:
                self.stop()
                return None
            
            if not self.stream_thread.is_alive() and (not self.stopped):
                print("stream thread is dead. restarting....")
                try:
                    self.stream_thread = Thread(target=self._reader, daemon=True)
                    self.stream_thread.start()
                    self.stopped = False
                except Exception as ex:
                    print(f"VideoCapture init error: {ex}")

            return None
    
    
    def stop(self):
        try:
            # safely close video stream
            # self.stream_thread.stop()
            self.stream = None
        except Exception as ex: 
            print(f"ERROR: VideoCameraThreadQueue stream stopping: {ex}")

        self.stopped = True
   
def videofile_loader(source_datapath, start_sec=0, clip_duration=None, fps=None, target_resolution=None, **kwargs):
    """
    Video file loader
    target_resolution: [h, w]
    frames: BGR: [h, w, 3]
    """

    print("videofile_loader...")

    timer = ProcTimer()
    timer.restart()

    mpyVidObj = mpy.VideoFileClip(source_datapath, target_resolution=target_resolution)
    mpyVidObj = mpyVidObj if fps is None else mpyVidObj.set_fps(fps)
    # video_length = mpyVidObj.reader.nframes # does not update after subclip
    mpyVidObj: mpy.VideoClip = mpyVidObj.subclip(start_sec, start_sec + mpyVidObj.duration if clip_duration is None else start_sec + clip_duration)
    
    video_duration = mpyVidObj.duration
    video_fps = round(mpyVidObj.fps)
    # frame_size = mpyVidObj.size
    # num_frames = round(video_duration*video_fps)
    
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in mpyVidObj.iter_frames(fps=None, with_times=False, logger=None, dtype="uint8")]
    num_frames = len(frames)
    frame_size = frames[0].shape
    
    timer.display_proctime()
    
    video_meta = {    
                    "num_frames": num_frames, 
                    "fps": video_fps, 
                    "duration_sec": video_duration, 
                    "frame_size": frame_size
                }
    print(f"loaded frame duration: {video_duration} secs, frame length: {num_frames}, fps: {video_fps}, resolution: {frame_size}")

    return frames, video_meta

def convertcolor_hexstr2tuple(color, color_format="rgb"):
    if isinstance(color, str):
        hex2rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        if color_format == "bgr":
            hex2rgb = hex2rgb[::-1]
        color = hex2rgb
    return color

def image_resize(x, target_resolution, mode="cv", ref="w"):
    if mode == "pil": # rgb
        ispil = not isinstance(x, np.ndarray)
        if not ispil:
            x = Image.fromarray(x)

        x_input_size = (x.width, x.height)
        w, h = target_resolution
        if ref == "w":
            alpha_size_scale = w/x_input_size[0]
            h = int(x_input_size[1]*alpha_size_scale)
        elif ref == "h":
            alpha_size_scale = h/x_input_size[1]
            w = int(x_input_size[0]*alpha_size_scale)
        else:
            pass

        anony_base_size = (w, h)
        x = x.resize(anony_base_size, Image.LANCZOS)

        if ispil: 
            return x
        return np.asarray(x)

    elif mode == "cv": #bgr
        x_input_size = x.shape[:-1][::-1]
        w, h = target_resolution
        if ref == "w":
            alpha_size_scale = w/x_input_size[0]
            h = int(x_input_size[1]*alpha_size_scale)
        elif ref == "h":
            alpha_size_scale = h/x_input_size[1]
            w = int(x_input_size[0]*alpha_size_scale)
        else:
            pass

        anony_base_size = (w, h)
        x = cv2.resize(x, anony_base_size)
        return x
        
    else:
        raise Exception(f"Undefined mode {mode}. choose from cv or pil.")
        
def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    print(ratio, new_size, delta_w, delta_h)
    print(top, bottom, left, right)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def add_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    delta_w = new_shape[0] - original_shape[0]
    delta_h = new_shape[1] - original_shape[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    print(delta_w, delta_h)
    print(top, bottom, left, right)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def remove_pad(image: np.array, 
                    new_shape: Tuple[int, int]
                    ) -> np.array:
    """Maintains resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    delta_w = original_shape[0] - new_shape[0] 
    delta_h = original_shape[1] - new_shape[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    print(original_shape, delta_w, delta_h)
    print(top, bottom, left, right)
    image = image[bottom:original_shape[1]-top, left:original_shape[0]-right]
    return image

def save_figure(filename, fig, filepath=None, isshow=False, issave=True, dpi=300, format=".jpg"):
    if not filepath:
        filepath = figure_path
    
    print("saving ", filepath)
    if issave:
        fig.savefig(Path(Path(filepath) , Path(f"{filename}{format}".format(filename, format))), dpi=dpi, bbox_inches='tight')
        if isshow:
            plt.show(fig)
        else:
            plt.close()

    if isshow:
        plt.show(fig)
