# =============================================================================
# LA3D: Lightweight Anonymization (AN) and Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides interfacing to the PEL4VAD VAD model.
# It includes functions to load the model, perform predictions, and handle configurations.
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: May 2024
# =============================================================================
import os, sys
import torch
import numpy as np
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)

import configs
from .model import XModel

def load_model(filepath, datasource="xd", **kwargs):
    """Load the XModel from a specified file path.

    Args:
        filepath (str): Path to the model checkpoint file.
        datasource (str): Source of the data, used to build the configuration.
        **kwargs: Additional keyword arguments for model initialization.
    Returns:
        XModel: An instance of the XModel with loaded weights.
    """
    if os.path.isfile(filepath):
        print('loading pretrained checkpoint from {}.'.format(filepath))
        cfg = configs.build_config(datasource)
        model = XModel(cfg)
        weight_dict = torch.load(filepath, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print('{} size mismatch: load {} given {}. thus, weights are not loaded!'.format(
                        name, param.size(), model_dict[name].size()))
            else:
               print('{} not found in model dict.'.format(name))
    else:
        raise FileNotFoundError(f'Model file {filepath} not found.')
    return model

def predict(data, model, frequency: int = 16, smooth: str = None,  device: str = "cuda") -> np.ndarray:
    """Predict the anomaly scores for the input data using the XModel.

    Args:       
        data (torch.Tensor): Input data tensor of shape (batch_size, channels, height, width).
        model (XModel): The XModel instance.
        frequency (int): Frequency for repeating the predictions.
        smooth (str): Smoothing method to apply to the predictions ('fixed', 'slide', or None).
        device (str): Device to run the model on ('cuda' or 'cpu').
    Returns:
        np.ndarray: Anomaly scores for the input data, repeated according to the frequency.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = device if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    model.eval()
    data = data.permute((1, 0, 2)).float().to(device)
    pred = torch.zeros(0).to(device)
    with torch.no_grad():
        seq_len = [data.shape[1]]*data.shape[0]
        logits, _ = model(data, seq_len)
        logits = torch.mean(logits, 0)
        logits = logits.squeeze(dim=-1)
        seq = len(logits)
        # if smooth == 'fixed':
        #     logits = fixed_smooth(logits, cfg.kappa)
        # elif smooth == 'slide':
        #     logits = slide_smooth(logits, cfg.kappa)
        # else:
        #     pass
        pred = logits[:seq]
    pred = list(pred.cpu().detach().numpy())
    pred = np.repeat(np.array(pred), frequency)
    return pred
        