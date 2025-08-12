# =============================================================================
# LA3D: Lightweight Anonymization (AN) and Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides interfacing to the MGFN VAD model.
# It includes functions to load the model, perform predictions, and handle configurations.
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: May 2024
# =============================================================================
import os, sys
import option
import numpy as np
import torch
from tqdm import tqdm
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)

args=option.parse_args()

from config import *
from .models.mgfn import mgfn as Model

def load_model(filepath: str = 'mgfn_ucf.pkl', **kwargs):
    """Load the MGFN model from a specified file path.

    Args:
        filepath (str): Path to the model checkpoint file.
        **kwargs: Additional keyword arguments for model initialization. 
        Returns:    
        Model: An instance of the MGFN model with loaded weights."""
    if os.path.isfile(filepath):    
        model = Model(**kwargs)
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(filepath, map_location=torch.device('cpu')).items()})
        print(f'Loaded model from {filepath}')
    else:
        raise FileNotFoundError(f'Model file {filepath} not found.')
    return model

def predict(data, model, frequency: int = 16, device: str = "cuda"):
    """Predict the anomaly scores for the input data using the MGFN model."
    Args:
        data (torch.Tensor): Input data tensor of shape (batch_size, channels, height, width).
        model (Model): The MGFN model instance.   
        frequency (int): Frequency for repeating the predictions.
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
    data = data.to(device)
    pred = torch.zeros(0).to(device)
    with torch.no_grad():
        for i, inputs in tqdm(enumerate(data)):
            input = inputs.permute(0, 2, 1, 3)
            _, _, _, _, logits = model(input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
    pred = list(pred.cpu().detach().numpy())
    pred = np.repeat(np.array(pred), frequency)
    return pred
