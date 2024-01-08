import torch
from cv2 import imread
import os
from transform import xception_default_data_transforms

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root_dir = 'bases'

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'base_{idx}.png')
        return imread(img_name), torch.tensor([1,0])    # Real

class PoisonDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root_dir = 'poisons'

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'poison_{idx}.png')
        preprocess = xception_default_data_transforms['train']
        return preprocess(imread(img_name)), torch.tensor([1,0])    # Real