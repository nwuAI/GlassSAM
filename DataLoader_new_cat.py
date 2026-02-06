
import os
from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, train_transforms_aug, get_boxes_from_mask, init_point_sampling, train_transforms_irseg, train_transforms_glass, get_boxes_from_mask_glass, init_point_sampling_glass
import json
from PIL import Image, ImageFile
from torchvision import transforms
import random
# from utils import Resize

def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


from utils import test_transforms_glass_union

class TestingDataset_Glass_union(Dataset):
    def __init__(self, data_path, image_size=(256,256), mode='test', requires_name=True,
                 point_num=1, return_ori_mask=True, prompt_path=None,
                 pair_check=True, check_k=50):
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num

        dataset_rgb = json.load(open(os.path.join(data_path, f'label2image_{mode}.json'), "r"))
        dataset_aux = json.load(open(os.path.join(data_path, f'label2thermal_{mode}.json'), "r"))

        # 顺序一致：按插入顺序取
        self.label_paths = list(dataset_rgb.keys())
        self.image_paths = list(dataset_rgb.values())
        self.aux_label_paths = list(dataset_aux.keys())
        self.aux_paths = list(dataset_aux.values())

        if len(self.label_paths) != len(self.aux_paths):
            raise RuntimeError(f"VAL RGB/AUX 数量不一致：{len(self.label_paths)} vs {len(self.aux_paths)}")

        if pair_check:
            import random
            idxs = random.sample(range(len(self.label_paths)), k=min(check_k, len(self.label_paths)))
            for i in idxs:
                if os.path.normpath(self.label_paths[i]) != os.path.normpath(self.aux_label_paths[i]):
                    raise RuntimeError(
                        "检测到 VAL 下 label2image 与 label2thermal 在 index 配对下 label 不一致（顺序错位）！\n"
                        f"i={i}\nlabel2image_key={self.label_paths[i]}\nlabel2thermal_key={self.aux_label_paths[i]}"
                    )

        self.pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.pixel_std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    @staticmethod
    def _ensure_hwc3(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[:, :, None]
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        if x.shape[2] != 3:
            raise ValueError(f"aux 通道数异常：{x.shape}")
        return x.astype(np.float32)

    @staticmethod
    def _read_aux_auto(path: str) -> np.ndarray:
        arr = np.array(Image.open(path))
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        elif arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return TestingDataset_Glass_union._ensure_hwc3(arr)

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        image_input = {}

        rgb_path = self.image_paths[index]
        aux_path = self.aux_paths[index]
        mask_path = self.label_paths[index]  # label2image 的 key 就是 mask path

        # RGB
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"报错图片 {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - self.pixel_mean) / self.pixel_std

        # AUX
        aux = self._read_aux_auto(aux_path)

        # mask
        ori_np_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ori_np_mask is None:
            raise RuntimeError(f"报错mask {mask_path}")
        ori_np_mask = ori_np_mask.astype(np.float32)
        if ori_np_mask.max() > 1.0:
            ori_np_mask = ori_np_mask / 255.0
        ori_np_mask = (ori_np_mask > 0.5).astype(np.float32)

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        tfm = test_transforms_glass_union(self.image_size, h, w)
        aug = tfm(image=rgb, aux=aux, mask=ori_np_mask)
        image_t = aug['image']                     # [3,H,W]
        aux_t   = aug['aux']                       # [3,H,W]
        mask_t  = aug['mask'].to(torch.int64)      # [H,W]

        # prompts/boxes/points
        if self.prompt_path is None:
            boxes = get_boxes_from_mask_glass(mask_t, max_pixel=0)
            point_coords, point_labels = init_point_sampling_glass(mask_t, self.point_num)
        else:
            prompt_key = os.path.basename(mask_path)
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        image_input["image"] = image_t
        image_input["depth"] = aux_t
        image_input["label"] = mask_t.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = os.path.dirname(mask_path)

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask

        if self.requires_name:
            image_input["name"] = os.path.basename(mask_path)

        return image_input



import os, json
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import build_train_transforms_union

class TrainingDataset_Glass_union(Dataset):
    def __init__(self, data_dir, image_size=(256,256), mode='train',
                 requires_name=True, point_num=1, mask_num=5,
                 pair_check=True, check_k=50):
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num

        self.pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.pixel_std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        rgb2label = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        aux2label = json.load(open(os.path.join(data_dir, f'thermal2label_{mode}.json'), "r"))

        # 保留“顺序一致”的配对方式（按 json 中的插入顺序）
        self.image_paths = list(rgb2label.keys())
        self.label_paths = list(rgb2label.values())

        self.aux_paths = list(aux2label.keys())
        self.aux_label_paths = list(aux2label.values())

        if len(self.image_paths) != len(self.aux_paths):
            raise RuntimeError(f"RGB/AUX 数量不一致：{len(self.image_paths)} vs {len(self.aux_paths)}")

        if pair_check:
            import random
            idxs = random.sample(range(len(self.image_paths)), k=min(check_k, len(self.image_paths)))
            for i in idxs:
                if os.path.normpath(self.label_paths[i]) != os.path.normpath(self.aux_label_paths[i]):
                    raise RuntimeError(
                        "检测到 RGB 与 AUX 在 index 配对下 label 不一致（说明顺序已错位）！\n"
                        f"i={i}\nRGB={self.image_paths[i]}\nAUX={self.aux_paths[i]}\n"
                        f"RGB_label={self.label_paths[i]}\nAUX_label={self.aux_label_paths[i]}"
                    )

        self.geo_tf, self.color_tf, self.to_tensor_tf = build_train_transforms_union(self.image_size)

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _ensure_hwc3(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[:, :, None]
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        if x.shape[2] != 3:
            raise ValueError(f"aux 通道数异常：{x.shape}")
        return x.astype(np.float32)

    @staticmethod
    def _read_aux_auto(path: str) -> np.ndarray:
        """
        自动处理：
        - uint16（depth）：/65535
        - uint8（thermal / aolp / dolp）：/255
        然后转成 HWC3
        """
        arr = np.array(Image.open(path))
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        elif arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)

        return TrainingDataset_Glass_union._ensure_hwc3(arr)

    def __getitem__(self, index):
        image_input = {}

        rgb_path  = self.image_paths[index]
        aux_path  = self.aux_paths[index]
        mask_path = self.label_paths[index]

        # --- RGB: BGR->RGB + normalize ---
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"报错图片 {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - self.pixel_mean) / self.pixel_std  # HWC

        # --- AUX: 显式归一化（uint16/uint8）+ 3ch ---
        aux = self._read_aux_auto(aux_path)  # HWC3 float32

        # --- mask ---
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"报错mask {mask_path}")
        m = m.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0
        m = (m > 0.5).astype(np.float32)

        # --- augmentation：几何同步 + 颜色只给 RGB ---
        aug = self.geo_tf(image=rgb, aux=aux, mask=m)
        rgb_aug, aux_aug, mask_aug = aug['image'], aug['aux'], aug['mask']
        rgb_aug = self.color_tf(image=rgb_aug)['image']  # only RGB

        out = self.to_tensor_tf(image=rgb_aug, aux=aux_aug, mask=mask_aug)
        image_t = out['image']                       # [3,H,W]
        aux_t   = out['aux']                         # [3,H,W]
        mask_t  = out['mask'].to(torch.int64)        # [H,W]

        # --- boxes/points（恢复stack + 空mask兜底）---
        masks_list = []
        boxes_list = []
        point_coords_list = []
        point_labels_list = []

        masks_list.append(mask_t)

        boxes = get_boxes_from_mask_glass(mask_t)
        if (not torch.is_tensor(boxes)) or boxes.numel() == 0:
            boxes = torch.zeros((4,), dtype=torch.float32)
        else:
            boxes = boxes.reshape(-1)[:4].to(torch.float32)
        boxes_list.append(boxes)

        point_coords, point_labels = init_point_sampling_glass(mask_t, self.point_num)
        point_coords_list.append(point_coords)
        point_labels_list.append(point_labels)

        mask = torch.stack(masks_list, dim=0)  # [1,H,W]
        boxes = torch.stack(boxes_list, dim=0)  # [1,4]
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        # ---- 输出（把你训练 loop 需要的 key 都返回）----
        image_input["image"] = image_t.unsqueeze(0)  # [1,3,H,W]
        image_input["depth"] = aux_t.unsqueeze(0)  # [1,3,H,W]（语义是aux）
        image_input["label"] = mask.unsqueeze(1)  # [1,1,H,W]
        image_input["boxes"] = boxes  # [1,4]
        image_input["point_coords"] = point_coords  # [1,...]
        image_input["point_labels"] = point_labels  # [1,...]

        if self.requires_name:
            image_input["name"] = os.path.basename(rgb_path)

        return image_input


if __name__ == "__main__":
    train_dataset = TrainingDataset("/root/autodl-tmp/SAM_Med2D_main/dataset/", image_size=224, mode='train', requires_name=True, point_num=1, mask_num=1)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["depth"].shape, batched_image["label"].shape)

