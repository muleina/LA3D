from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import os, sys

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)

import option

args=option.parse_args()

from config import *
from .models.mgfn import mgfn as Model
from .datasets.dataset import Dataset

def load_model(filepath='mgfn_ucf.pkl', **kwargs):
    model = Model(**kwargs)
    # print("Model Expected")
    # print(model)
    # print("Model Loaded")
    # print(torch.load(filepath))
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(filepath).items()})

    # weight_dict = torch.load(filepath)
    # # print(weight_dict)
    # model_dict = model.state_dict()
    # for name, param in weight_dict.items():
    #     if 'module' in name:
    #         name = '.'.join(name.split('.')[1:])
    #     if name in model_dict:
    #         if param.size() == model_dict[name].size():
    #             model_dict[name].copy_(param)
    #         else:
    #             print('{} size mismatch: load {} given {}. thus, weights are not loaded!'.format(
    #                 name, param.size(), model_dict[name].size()))
    #     else:
    #         print('{} not found in model dict.'.format(name))


    return model

def get_frames(videofilepath):
    return videofilepath

def predict(data, model, frequency=16, device="cuda"):
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        pred = torch.zeros(0)
        featurelen =[]
        # print(data.shape)
        for i, inputs in tqdm(enumerate(data)):
            # print(inputs.shape)
            input = inputs.to(device)
            # print(input.shape)
            input = input.permute(0, 2, 1, 3)
            # bs, ncrops, t, c = input.size()
            print("bs, ncrops, t, c: ", input.shape)
            _, _, _, _, logits = model(input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            featurelen.append(len(sig))
            pred = torch.cat((pred, sig))
            print("pred.shape: ", pred.shape)

        print("featurelen: ", featurelen)

        pred = list(pred.cpu().detach().numpy())
        
        pred = np.repeat(np.array(pred), frequency)
        print("pred.shape: ", pred.shape)

        return pred
    
        # # from TedSTAD
        # ratio = float(len(list(gt))) / float(len(pred))
        # # In case size mismatch btwn predictions and gt.
        # if ratio == 1.0:
        #     final_pred = pred
        # else:
        #     print(f'Ground truth not exact shape: {ratio}')
        #     final_pred = np.zeros_like(gt, dtype='float32')
        #     for i in range(len(pred)):
        #         b = int(i * ratio + 0.5)
        #         e = int((i + 1) * ratio + 0.5)
        #         final_pred[b:e] = pred[i]
            
        # return final_pred
    
        
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
    args = option.parse_args()
    config = Config(args)
    device = torch.device("cuda")
    model = Model()
    test_filepath = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    data = get_frames(test_filepath)
    model = model.to(device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('mgfn_ucf.pkl').items()})
    auc = predict(data, model, args, device)
