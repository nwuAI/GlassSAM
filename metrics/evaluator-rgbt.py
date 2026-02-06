import os
import time

import numpy as np
import torch
from torchvision import transforms
import cv2


class Eval_thread():
    def __init__(self, loader, method, output_dir, cuda):
        self.loader = loader
        self.method = method.replace("_results", '')
        self.dataset = "rgbt"
        self.cuda = cuda
        self.output_dir = output_dir
        self.logfile = os.path.join(output_dir, 'result.txt')

    def run(self):
        Res = {}

        ber = self.Eval_BER()
        Res['BER'] = ber

        iou = self.Eval_IOU()
        Res['IOU'] = iou

        mae = self.Eval_mae()
        Res['MAE'] = mae

        Fm, prec, recall = self.Eval_fmeasure()
        max_f = Fm.max().item()
        mean_f = Fm.mean().item()
        Res['MaxFm'] = max_f

        with open(self.logfile, 'a', encoding='UTF-8') as target:
            target.write(
                self.method + ",\t" + str(round(Res['IOU'] * 100, 2)) + ",\t" + str(round(Res['MAE'], 3)) + ",\t" + str(
                    round(Res['BER'], 3)) + ",\t" + str(
                    round(Res['MaxFm'], 3)) + '\n')

    def Eval_mae(self):
        # print('eval[MAE]:{} dataset with {} method.'.format(
        #     self.dataset, self.method))
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

                # pred = (pred >= 0.5)
                # gt = (gt >= 0.5)

                pred = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
                gt = torch.where(gt >= 0.5, torch.ones_like(gt), torch.zeros_like(gt))

                # pred = 1 - pred
                # gt = 1 - gt

                mea = torch.abs(pred - gt).mean()
                if mea == mea:  # for Nan
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num
            return avg_mae.item()

    def Eval_fmeasure(self):
        # print('eval[FMeasure]:{} dataset with {} method.'.format(
        #     self.dataset, self.method))
        beta2 = 0.3
        avg_f, avg_p, avg_r, img_num = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:

                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()

                    # pred = 1 - pred
                    # gt = 1 - gt
                    if torch.min(gt) != torch.max(gt):
                        pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                           torch.min(pred) + 1e-20)
                    else:
                        pred = pred / torch.max(pred)

                else:
                    pred = trans(pred)
                    gt = trans(gt)

                    # pred = 1 - pred
                    # gt = 1 - gt

                    if torch.min(gt) != torch.max(gt):
                        pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                           torch.min(pred) + 1e-20)
                    else:
                        pred = pred / torch.max(pred)

                prec, recall = self._eval_pr(pred, gt, 255)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0  # for Nan
                avg_f += f_score
                avg_p += prec
                avg_r += recall
                img_num += 1.0
            Fm = avg_f / img_num
            avg_p = avg_p / img_num
            avg_r = avg_r / img_num
            return Fm, avg_p, avg_r

    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() +
                                                                    1e-20)
        return prec, recall

    def Eval_IOU(self):
        # print('eval[IOU]:{} dataset with {} method.'.format(
        #     self.dataset, self.method))
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

                # pred = 1 - pred
                # gt = 1 - gt

                pred = (pred >= 0.5)
                gt = (gt >= 0.5)

                iou = torch.sum((pred & gt)) / torch.sum((pred | gt))

                if iou == iou:  # for Nan
                    avg_iou += iou
                    img_num += 1.0
            avg_iou /= img_num
            return avg_iou.item()

    def Eval_BER(self):
        # print('eval[BER]:{} dataset with {} method.'.format(
        #     self.dataset, self.method))
        avg_ber, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])

            self.total_bers = np.zeros((2,), dtype=np.float32)
            self.total_bers_count = np.zeros((2,), dtype=np.float32)

            for pred, gt in self.loader:

                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)

                # pred = 1 - pred
                # gt = 1 - gt

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
            return avg_ber.item() * 100

if __name__ == "__main__":

    import os
    import torch
    from torchvision import transforms
    import cv2
    from torch.utils.data import DataLoader, Dataset

    def load_images(mask_dir, gt_dir):
        mask_images = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]
        gt_images = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')]
        return mask_images, gt_images


    def create_loader(mask_images, gt_images):
        trans = transforms.Compose([transforms.ToTensor()])
        data = []
        for mask_img_path, gt_img_path in zip(mask_images, gt_images):
            mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)

            if mask_img is None:
                print(f"Error loading mask image: {mask_img_path}")
                continue
            if gt_img is None:
                print(f"Error loading GT image: {gt_img_path}")
                continue

            # mask_img = trans(mask_img)
            # gt_img = trans(gt_img)
            data.append((mask_img, gt_img))
        return data


    import os

    mask_dir = '/home/xxxxx'
    gt_dir = '/home/xxxxx'
    output_dir = '/home/xxxxxx'

    if not os.path.exists(mask_dir):
        print(f"Mask directory does not exist: {mask_dir}")
    if not os.path.exists(gt_dir):
        print(f"GT directory does not exist: {gt_dir}")
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")

    use_cuda = torch.cuda.is_available()

    mask_images, gt_images = load_images(mask_dir, gt_dir)
    data_loader = create_loader(mask_images, gt_images)

    evaluator = Eval_thread(data_loader, mask_dir, output_dir, cuda=True)
    evaluator.run()





