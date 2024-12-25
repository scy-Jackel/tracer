import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


# create by cy 2019 0805
class MyDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.filenames = [f for f in os.listdir(data_path)]
        self.root_path = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        if index >= len(self.filenames):
            return None, None, None
        else:
            imgs = default_loader(os.path.join(self.root_path, self.filenames[index]))
            # img_np = np.array(imgs)
            img_tensor = self.transforms(imgs)
            label = np.zeros((1))
            return self.filenames[index], img_tensor, torch.from_numpy(label)

    def __len__(self):
        return len(self.filenames)
