import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from PIL import  Image

def _eval_pr(y_pred, y, num, cuda=True):
    """计算多个阈值下的 precision 和 recall"""
    if cuda:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall

def cal_fmeasure(precision, recall):
    """计算 F-measure，返回所有阈值中的最大值"""
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])
    return max_fmeasure

def Eval_FMeasure(self):
    # print('eval[F-Measure]:{} dataset with {} method.'.format(
    #     self.dataset, self.method))
    avg_fmeasure, img_num = 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in self.loader:

            if self.cuda:
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
            else:
                pred = trans(pred)
                gt = trans(gt)

            # 归一化 pred 到 [0, 1]
            if torch.min(gt) != torch.max(gt):
                pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
            else:
                pred = pred / torch.max(pred) if torch.max(pred) > 0 else pred

            # 归一化 gt 到 [0, 1]
            if torch.min(gt) != torch.max(gt):
                gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt) + 1e-20)
            else:
                gt = gt / torch.max(gt) if torch.max(gt) > 0 else gt

            # 计算 256 个阈值下的 precision 和 recall
            prec, recall = _eval_pr(pred, gt, 256, self.cuda)
            
            # 转换为 numpy 数组用于 cal_fmeasure
            prec_np = prec.cpu().numpy()
            recall_np = recall.cpu().numpy()
            
            # 计算 F-measure
            fmeasure = cal_fmeasure(prec_np, recall_np)

            if fmeasure == fmeasure:  # for Nan
                avg_fmeasure += fmeasure
                img_num += 1.0
        avg_fmeasure /= img_num
        return avg_fmeasure




if __name__ == "__main__":
    # get img file in a list
    label_path = '/home/xxxxxx'
    pre_path = '/home/xxxxx'

    img_list = os.listdir(pre_path)
    # print(img_list)
    # loader = []
    trans = transforms.Compose([transforms.ToTensor()])
    avg_fmeasure, img_num = 0.0, 0.0
    for i,name in enumerate(img_list):
        # if name.endswith('.png'):
            pred = cv2.imread(os.path.join(pre_path, name), cv2.IMREAD_GRAYSCALE)
        #     # print(predict)
            gt = cv2.imread(os.path.join(label_path, name), cv2.IMREAD_GRAYSCALE)
            pred = trans(pred).cuda()
            gt = trans(gt).cuda()

            if name == '2.png':
                print(pred)

            pred = pred.squeeze()
            gt = gt.squeeze()

            # 归一化 pred 到 [0, 1]
            if torch.min(gt) != torch.max(gt):
                pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
            else:
                pred = pred / torch.max(pred) if torch.max(pred) > 0 else pred

            # 归一化 gt 到 [0, 1]
            if torch.min(gt) != torch.max(gt):
                gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt) + 1e-20)
            else:
                gt = gt / torch.max(gt) if torch.max(gt) > 0 else gt

            # 计算 256 个阈值下的 precision 和 recall
            prec, recall = _eval_pr(pred, gt, 256, cuda=True)
            
            # 转换为 numpy 数组用于 cal_fmeasure
            prec_np = prec.cpu().numpy()
            recall_np = recall.cpu().numpy()
            
            # 计算 F-measure
            fmeasure = cal_fmeasure(prec_np, recall_np)

            if fmeasure == fmeasure:  # for Nan
                avg_fmeasure += fmeasure
                img_num += 1.0


    avg_fmeasure /= img_num
    print(avg_fmeasure)

