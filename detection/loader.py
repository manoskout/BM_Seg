import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import json
import torch
from typing import Tuple, Dict
# Define the dataset
from utils.config import *


# Data structure
# ouputs/images
# GM4A-109_000.npy
# outputs/metadata.json
# [PatientID] (e.g "GM4A-109")
# ["centroids"]
# [Slice index] (e.g "0")
# ["bbox"]
# List that contains multiple lists
# (e.g
# [
# [273, 161, 313, 201],
# [88, 159, 128, 199]
# ]
# )

# "centroids": {
#       "0": {
#         "centroid": [
#           [293, 181],
#           [108, 179]
#         ],
#         "bbox":
#       },

class BoneMarrowDataset(Dataset):
    def __init__(self, data_path: str, ann_path: str,) -> None:
        self.data_path = data_path
        self.ann_path = ann_path
        self.images_path = os.path.join(self.data_path, "images")
        self.ann = self.retrieve_metadata()
        self.image_files = [os.path.join(
            self.images_path, filename) for filename in os.listdir(self.images_path)]

    def retrieve_metadata(self) -> Dict:
        f = open(self.ann_path)
        return json.load(f)

    def fetch_img_info(self, filename: str) -> Tuple[str, int]:
        patient_id = filename.split("/")[-1].split("_")[0]
        slice_num = int(filename.split("/")[-1].split("_")[-1].split(".")[0])
        return patient_id, slice_num

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[index]
        patient_id, slice_num = self.fetch_img_info(img_path)
        image = np.load(img_path)
        targets = self.ann[patient_id]["centroids"][str(slice_num)]["bbox"]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
