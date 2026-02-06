import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from PIL import Image
from scipy import ndimage

def wFmeasure(FG, GT):
    """
    wFmeasure Compute the Weighted F-beta measure (as proposed in "How to Evaluate
    Foreground Maps?" [Margolin et. al - CVPR'14])

    Arguments:
        FG (np.ndarray): Binary/Non binary foreground map with values in the range [0 1].
        GT (np.ndarray): Binary ground truth. Type: bool
    Return:
        float : The Weighted F-beta score
    """
    FG = FG.detach().cpu().numpy() if isinstance(FG, torch.Tensor) else FG
    GT = GT.detach().cpu().numpy() if isinstance(GT, torch.Tensor) else GT
    
    # 确保 GT 是 bool 类型（用于索引）
    GT = (GT >= 0.5).astype(bool)
    
    if not GT.max():
        return 0

    E = np.abs(FG - GT)

    Dst, IDXT = ndimage.distance_transform_edt(1 - GT.astype(np.float64), return_indices=True)
    # Pixel dependency
    Et = E.copy()
    Et[np.logical_not(GT)] = Et[IDXT[0][np.logical_not(GT)], IDXT[1][np.logical_not(GT)]]  # To deal correctly with the edges of the foreground region
    EA = ndimage.gaussian_filter(Et, 5, mode='constant', truncate=0.5)
    MIN_E_EA = E.copy()
    MIN_E_EA[np.logical_and(GT, EA < E)] = EA[np.logical_and(GT, EA < E)]
    # Pixel importance
    B = np.ones(GT.shape)
    B[np.logical_not(GT)] = 2 - 1 * np.exp(np.log(1 - 0.5) / 5 * Dst[np.logical_not(GT)])
    Ew = MIN_E_EA * B

    TPw = GT.sum() - Ew[GT].sum()
    FPw = Ew[np.logical_not(GT)].sum()

    R = 1 - Ew[GT].mean()  # Weighted Recall
    P = TPw / (TPw + FPw + np.finfo(np.float64).eps)  # Weighted Precision

    Q = 2 * R * P / (R + P + np.finfo(np.float64).eps)  # Beta=1
    # Q = (1 + Beta ** 2) * R * P / (R + Beta * P + np.finfo(np.float64).eps)

    return Q

def Eval_FMeasure(self):
    # print('eval[F-Measure]:{} dataset with {} method.'.format(
    #     self.dataset, self.method))
    avg_fmeasure, img_num = 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in self.loader:

            if self.cuda:
                pred = trans(pred).cuda(1)
                gt = trans(gt).cuda(1)
            else:
                pred = trans(pred)
                gt = trans(gt)

            # 归一化 pred 到 [0, 1]
            if torch.min(pred) != torch.max(pred):
                pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
            else:
                pred = pred / torch.max(pred) if torch.max(pred) > 0 else pred

            # 将 gt 转换为二值化（0 或 1），并转换为 bool 类型
            gt = (gt >= 0.5).float()

            # 确保 pred 和 gt 是 2D 数组（如果是 3D，取第一个通道或 squeeze）
            if len(pred.shape) > 2:
                pred = pred.squeeze()
            if len(gt.shape) > 2:
                gt = gt.squeeze()

            # 计算加权 F-measure（wFmeasure 内部会将 GT 转换为 bool）
            fmeasure = wFmeasure(pred, gt)

            if fmeasure == fmeasure:  # for Nan
                avg_fmeasure += fmeasure
                img_num += 1.0
        avg_fmeasure /= img_num
        return avg_fmeasure


if __name__ == "__main__":
    label_path = '/home/xxxxx'
    pre_path = "/home/xxxxx"

    img_list = os.listdir(pre_path)
    # print(img_list)
    # loader = []
    trans = transforms.Compose([transforms.ToTensor()])
    avg_fmeasure, img_num = 0.0, 0.0
    for i,name in enumerate(img_list):
        # 跳过非图像文件
        if not (name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg')):
            continue
        
        pred_path = os.path.join(pre_path, name)
        
        # 检查预测文件是否存在
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction file not found: {pred_path}, skipping...")
            continue
        
        # 读取预测图像
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            print(f"Warning: Failed to read prediction image: {pred_path}, skipping...")
            continue
        
        # 尝试匹配 GT 文件名：先尝试相同文件名，如果不存在则尝试添加 _mask 后缀
        base_name = os.path.splitext(name)[0]  # 去掉扩展名
        ext = os.path.splitext(name)[1]  # 获取扩展名
        
        # 方式1：直接使用相同文件名
        gt_name = name
        gt_path = os.path.join(label_path, gt_name)
        
        # 方式2：如果方式1不存在，尝试添加 _mask 后缀
        if not os.path.exists(gt_path):
            gt_name = f"{base_name}_mask{ext}"
            gt_path = os.path.join(label_path, gt_name)
        
        # 检查 GT 文件是否存在
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found for {name}, tried: {name} and {gt_name}, skipping...")
            continue
        
        # 读取 GT 图像
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"Warning: Failed to read ground truth image: {gt_path}, skipping...")
            continue
        
        pred = trans(pred).cuda(1)
        gt = trans(gt).cuda(1)

        pred = pred.squeeze()
        gt = gt.squeeze()

        # 归一化 pred 到 [0, 1]
        if torch.min(pred) != torch.max(pred):
            pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
        else:
            pred = pred / torch.max(pred) if torch.max(pred) > 0 else pred

        # 将 gt 转换为二值化（0 或 1）
        gt = (gt >= 0.5).float()

        # 计算加权 F-measure（wFmeasure 内部会将 GT 转换为 bool）
        fmeasure = wFmeasure(pred, gt)

        if fmeasure == fmeasure:  # for Nan
            avg_fmeasure += fmeasure
            img_num += 1.0


    avg_fmeasure /= img_num
    print(avg_fmeasure)

