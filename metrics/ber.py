import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2



if __name__ == "__main__":

    label_path = '/root/xxxxx'
    pre_path = '/root/xxxxx'
    img_list = os.listdir(pre_path)
    # loader = []
    trans = transforms.Compose([transforms.ToTensor()])
    total_bers = np.zeros((2,), dtype=float)
    total_bers_count = np.zeros((2,), dtype=float)
    avg_ber, img_num = 0.0, 0.0
    for i, name in enumerate(img_list):
            pred = cv2.imread(os.path.join(pre_path, name), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join(label_path, name), cv2.IMREAD_GRAYSCALE)
            print("pred_shape", name, pred.shape)
            pred = trans(pred).cuda()
            gt = trans(gt).cuda()

            pred = (pred >= 0.5)
            gt = (gt >= 0.5)

            N_p = torch.sum(gt) + 1e-20
            N_n = torch.sum(torch.logical_not(gt)) + 1e-20  # should we add this？

            TP = torch.sum(pred & gt)
            TN = torch.sum(torch.logical_not(pred) & torch.logical_not(gt))

            ber = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

            if ber == ber:  # for Nan
                avg_ber += ber
                img_num += 1.0

    avg_ber /= img_num
    print(avg_ber.item() * 100)