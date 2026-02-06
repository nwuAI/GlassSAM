import os
import time

import numpy as np
import torch
from torchvision import transforms
import cv2


def Eval_mae(self):
    avg_mae, img_num = 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in self.loader:

            if self.cuda:
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
            else:
                pred = trans(pred)
                gt = trans(gt)

            pred = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
            gt = torch.where(gt >= 0.5, torch.ones_like(gt), torch.zeros_like(gt))


            mea = torch.abs(pred - gt).mean()
            if mea == mea:  # for Nan
                avg_mae += mea
                img_num += 1.0
        avg_mae /= img_num
        return avg_mae.item()



if __name__ == "__main__":
    label_path = '/home/zh/ZH_temp/dataset/glass_depth_dataset/test/masks'
    pre_path = '/home/zh/ZH_temp/SAM_2024/run/run_glass/2025-12-17-22-46-(mirror-sam-med2d-glass)/predict_bestpth_rgbd/iter8_predict'
    # pre_path = '/media/user/shuju/zh/CVPR2021_PDNet-main/run/2022-07-01-11-25(glassrgbt_merged-new_year_convnext_128_5)/predicts'
    # label_path = '/media/user/shuju/zh/RGBT-GLASS-MERGED/test/GT'
    img_list = os.listdir(pre_path)
    # loader = []
    trans = transforms.Compose([transforms.ToTensor()])
    total_bers = np.zeros((2,), dtype=float)
    total_bers_count = np.zeros((2,), dtype=float)
    avg_mae, img_num = 0.0, 0.0
    for i,name in enumerate(img_list):
        # if name.endswith('.png'):
            pred = cv2.imread(os.path.join(pre_path, name),cv2.IMREAD_GRAYSCALE)
            #print(predict)
            gt = cv2.imread(os.path.join(label_path, name),cv2.IMREAD_GRAYSCALE)
            pred = trans(pred).cuda()
            gt = trans(gt).cuda()
            if name == '2.png':
                print(pred, pred.shape)

            pred = pred.squeeze()

            pred = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
            gt = torch.where(gt >= 0.5, torch.ones_like(gt), torch.zeros_like(gt))

            mea = torch.abs(pred - gt).mean()

            if mea == mea:  # for Nan
                avg_mae += mea
                img_num += 1.0
    avg_mae /= img_num
    print(avg_mae.item())