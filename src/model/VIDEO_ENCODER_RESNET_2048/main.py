from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from extract_features import run
# from extract_features_org import run
from resnet import i3_res50, i3_res50_nl
import os
from tqdm import tqdm
import cv2

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)

        # Check if camera opened successfully
        if (self.stream.isOpened() == False): 
            print("Error opening video stream or file")

        (self.grabbed, self.frame) = self.stream.read()
        
        self.stopped = False

    def start(self):    
        return self

    def get(self):
        if self.stream.isOpened():
            (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


def generate(**kwargs):
    datasetpath = kwargs.get("datasetpath", None) 
    outputpath = kwargs.get("outputpath", None) 
    pretrainedpath = kwargs.get("pretrainedpath", None) 
    frequency = kwargs.get("frequency", None) 
    batch_size = kwargs.get("batch_size", None) 
    sample_mode = kwargs.get("sample_mode", None) 
    fileloader = kwargs.get("fileloader", "cv2")
    
    rgb_list_filepath = kwargs.pop("rgb_list_filepath", None)
    if rgb_list_filepath is not None:
        rgb_list_file = list(open(rgb_list_filepath))
        f = lambda x: "_".join(x.strip('\n').split('\\')[-1].split('/')[-1].split('_')[:-1])
        rgb_list_file = list(map(f, rgb_list_file))
        print(f"rgb_list_file len: {len(rgb_list_file)}")
        print(rgb_list_file)
    else:
        rgb_list_file = None

    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = outputpath+ "/temp/"
    rootdir = Path(datasetpath)
    # videos = [str(f) for f in rootdir.glob('**/*.mp4')]
    videos = [str(f) for f in rootdir.glob('*/*.mp4')]
    print(videos)
    # return

    # setup the model
    if "_nl_" in pretrainedpath:
        i3d = i3_res50_nl(400, pretrainedpath)
    else:
        i3d = i3_res50(400, pretrainedpath)

    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    Path(outputpath).mkdir(parents=True, exist_ok=True)
    
    for video in tqdm(videos):
        videoname = ".".join(Path(video).name.split(".")[:-1])
        print(f"videoname: {videoname}...")
        if rgb_list_file is not None:
            if videoname not in rgb_list_file: 
                print(f"videoname: {videoname} is skipped. not in the seleted file list!")
                continue
        if Path(outputpath + "/" + videoname+".npy").exists():
            print(f"videoname: {videoname} is skipped. already processed!")
            continue

        startime = time.time()
        print("Generating for {0}".format(video))

        if fileloader == "cv2":

            video_getter = VideoGet(video).start()
            frames = []
            while video_getter.stream.isOpened():
                if video_getter.grabbed == True:
                    frame = video_getter.frame # HxWx3
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame.astype("uint8"))
                    video_getter.get()   
                else:
                    break
                
            video_fps = round(video_getter.stream.get(cv2.CAP_PROP_FPS))
            video_length = video_getter.stream.get(cv2.CAP_PROP_FRAME_COUNT)
            video_duration = video_length/video_fps
            frame_size = frames[0].shape
            
            # Destroy all windows while exiting
            # Closes all the frames
            video_getter.stream.release()
            cv2.destroyAllWindows()

            print(f"loaded frame duration: {video_duration} secs, frame length: {video_length}, fps: {video_fps}, resolution: {frame_size}")

            features = run(i3d, frequency, None, frames, batch_size, sample_mode)
            np.save(outputpath + "/" + videoname, features)
            print("Obtained features of size: ", features.shape)
            print("done in {0}.".format(time.time() - startime))

            # del frames

        else:
            Path(temppath).mkdir(parents=True, exist_ok=True)
            ffmpeg.input(video).output('{}%d.jpg'.format(temppath), start_number=0).global_args('-loglevel', 'quiet').run()
            print("Preprocessing done..")

            features = run(i3d, frequency, temppath, None, batch_size, sample_mode)
            np.save(outputpath + "/" + videoname, features)
            print("Obtained features of size: ", features.shape)
            shutil.rmtree(temppath)
            print("done in {0}.".format(time.time() - startime))

        # del features
        # gc.collect()

        # break

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, default="samplevideos/")
    parser.add_argument('--rgb_list_filepath', type=str, default=None)
    parser.add_argument('--outputpath', type=str, default="output")
    parser.add_argument('--fileloader', type=str, default="cv2")
    parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sample_mode', type=str, default="oversample")
    args = parser.parse_args()
    generate(**vars(args))    


# python main.py --datasetpath=samplevideos/ --outputpath=output_nl --pretrainedpath "pretrained\i3d_r50_nl_kinetics.pth"
# python main.py --fileloader cv2 --batch_size 20 --pretrainedpath "pretrained\i3d_r50_nl_kinetics.pth" --datasetpath "D:\UiA\Projects\AI4CITIZEN\Datasets\UCF\UCFCrime\Anomaly_Videos" --rgb_list_filepath "D:\UiA\Projects\AI4CITIZEN\Datasets\UCF\UCFCrime\UCF_Crimes-Train-Test-Split\Anomaly_Detection_splits\ucf-i3d-test.list" --outputpath "D:\UiA\Projects\AI4CITIZEN\Datasets\UCF\UCFCrime\UCF_Test_ten_i3d_ours"
# python main.py --fileloader pil --batch_size 20 --pretrainedpath "pretrained\i3d_r50_nl_kinetics.pth" --datasetpath "D:\UiA\Projects\AI4CITIZEN\Datasets\UCF\UCFCrime\Anomaly_Videos" --rgb_list_filepath "D:\UiA\Projects\AI4CITIZEN\Datasets\UCF\UCFCrime\UCF_Crimes-Train-Test-Split\Anomaly_Detection_splits\ucf-i3d-test.list" --outputpath "D:\UiA\Projects\AI4CITIZEN\Datasets\UCF\UCFCrime\UCF_Test_ten_i3d_ours"

# python main.py --fileloader pil --batch_size 20 --pretrainedpath "/home/mulugetawa/AI4CITIZENS/PACA/src/model/I3D_Feature_Extraction_resnet/pretrained/i3d_r50_nl_kinetics.pth" --datasetpath "/home/mulugetawa/AI4CITIZENS/Datasets/UCF/UCFCrime/Anomaly_Videos" --rgb_list_filepath "/home/mulugetawa/AI4CITIZENS/Datasets/UCF/UCFCrime/UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/ucf-i3d-test.list" --outputpath "/home/mulugetawa/AI4CITIZENS/Datasets/UCF/UCFCrime/UCF_Test_ten_i3d_ours"

# python main.py --fileloader pil --batch_size 20 --pretrainedpath "pretrained\i3d_r50_nl_kinetics.pth" --datasetpath "D:\UiA\Projects\AI4CITIZEN\Datasets\XD_Violence" --rgb_list_filepath "D:\UiA\Projects\AI4CITIZEN\Datasets\XD_Violence\xd-i3d-test.list" --outputpath "D:\UiA\Projects\AI4CITIZEN\Datasets\XD_Violence\XD_Test_ten_i3d_ours"
