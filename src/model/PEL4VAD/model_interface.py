from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import os, sys
import numpy as np
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)


# from config import *
# from .models.mgfn import mgfn as Model
# from .datasets.dataset import Dataset

sys.path.append(r"C:\Users\mulugetawa\OneDrive - Universitetet i Agder\UiA\Projects\AI4CITIZEN\CodeSources\PEL4VAD")
import configs
from .test import test_func
from .model import XModel
from .dataset import UCFDataset, XDataset, SHDataset
from torch.utils.data import DataLoader


def load_model(filepath, datasource="xd", **kwargs):
    if os.path.isfile(filepath):
        print('loading pretrained checkpoint from {}.'.format(filepath))
        cfg = configs.build_config(datasource)
        model = XModel(cfg)
        # print(model)
        weight_dict = torch.load(filepath)
        # print(weight_dict)
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
       print('Not found pretrained checkpoint file.')
    
    return model

def get_frames(videofilepath):
    return videofilepath

def predict(data, model, frequency=16, smooth=None, device="cuda"):
    model = model.to(device)
    model.eval()
    pred = torch.zeros(0)

    data = data.permute((1, 0, 2))
    with torch.no_grad():
        data = data.float().to(device)
        seq_len = [data.shape[1]]*data.shape[0]
        # print("seq_len: ", seq_len, data.shape)

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

        # print("pred.shape: ", pred.shape)

        pred = list(pred.cpu().detach().numpy())
    
        pred = np.repeat(np.array(pred), frequency)
        # print("pred.shape: ", pred.shape)

        return pred
        
def eval(dataloader, model, args):
    with torch.no_grad():
        pred = torch.zeros(0)
        featurelen =[]
        for i, inputs in tqdm(enumerate(dataloader)):
            sig = np.load()
            pred = torch.cat((pred, sig))

        gt = np.load(args.gt)
        # pred = list(pred.cpu().detach().numpy())
        # pred = np.repeat(np.array(pred), 16)
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        print('rec_auc : ' + str(rec_auc))
        return rec_auc, pr_auc


def test(dataloader, model, args, device):
    plt.clf()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        featurelen =[]
        for i, inputs in tqdm(enumerate(dataloader)):

            input = inputs[0].to(device)
            input = input.permute(0, 2, 1, 3)
            _, _, _, _, logits = model(input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            featurelen.append(len(sig))
            pred = torch.cat((pred, sig))

        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        print('rec_auc : ' + str(rec_auc))
        return rec_auc, pr_auc

if __name__ == '__main__':
    # args = option.parse_args()
    # config = Config(args)
    # device = torch.device("cuda")
    # model = Model()
    # test_filepath = DataLoader(Dataset(args, test_mode=True),
    #                           batch_size=1, shuffle=False,
    #                           num_workers=0, pin_memory=False)
    # data = get_frames(test_filepath)
    # model = model.to(device)
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('mgfn_ucf.pkl').items()})
    # auc = predict(data, model, args, device)
    pass
