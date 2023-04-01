from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
import torch.nn as nn
import numpy as np
import os
import torch
# Define the dataset
from torchvision import datasets
# from utils.config import *
import numpy as np  # linear algebra
import os
import torch
from torch.nn import functional as F
import copy
from PIL import Image
import albumentations as A

import warnings
warnings.filterwarnings("ignore")


def collate_fn(batch):
    return tuple(zip(*batch))


def get_albumentation(train):
    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform


class CTDetectionDataset(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        split="train",
        transform=None,
        target_transform=None,
        transforms=None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.split = split
        self.coco = COCO(os.path.join(
            root, split+"/images", "_annotations.coco.json"))
        self.ids = list(sorted(self.coco.imgs.keys()))
        print(self.ids)
        self.target_ids = [id for id in self.ids if (
            len(self._load_target(id)) > 0)]

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = np.load(os.path.join(self.root, self.split+"/images", path))
        return image

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = copy.deepcopy(self._load_target(id))

        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
            # print(transformed)
            image = transformed['image']
            boxes = transformed['bboxes']
        new_boxes = []
        for box in boxes:
            xmin = box[0]
            ymin = box[1]
            xmax = xmin + box[2]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {}
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor(
            [t["category_id"] for t in target], dtype=torch.int64)
        targ["image_id"] = torch.tensor([t["image_id"] for t in target])
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * \
            (boxes[:, 2] - boxes[:, 0])
        # Not defined .... Manually setted to 0
        # targ["iscrowd"] = torch.tensor([t["iscrowd"]  for t in target], dtype=torch.int64)
        targ["iscrowd"] = torch.tensor([1 for t in target], dtype=torch.int64)

        return image, targ

    def __len__(self) -> int:
        return len(self.ids)
