import os
import time

import numpy as np
import torch
# import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from PIL import  Image

def Eval_IOU(self):
    avg_iou, img_num = 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in self.loader:

            if self.cuda:
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
            else:
                pred = trans(pred)
                gt = trans(gt)

            pred = (pred >= 0.5)
            gt = (gt >= 0.5)

            iou = torch.sum((pred & gt)) / torch.sum((pred | gt))

            if iou == iou:  # for Nan
                avg_iou += iou
                img_num += 1.0
        avg_iou /= img_num
        return avg_iou.item()








if __name__ == "__main__":
    # get img file in a list
    label_path = '/root/xxxxx'
    pre_path = '/root/xxxxx'
    img_list = os.listdir(pre_path)
    trans = transforms.Compose([transforms.ToTensor()])
    avg_iou, img_num = 0.0, 0.0
    for i,name in enumerate(img_list):
            pred = cv2.imread(os.path.join(pre_path, name), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join(label_path, name), cv2.IMREAD_GRAYSCALE)
            pred = trans(pred)#.cuda()
            gt = trans(gt)#.cuda()

            pred = pred.squeeze()

            pred = (pred >= 0.5)
            gt = (gt >= 0.5)

            iou = torch.sum((pred & gt)) / torch.sum((pred | gt))

            if iou == iou:  # for Nan
                avg_iou += iou
                img_num += 1.0


    avg_iou /= img_num
    print(avg_iou.item())
