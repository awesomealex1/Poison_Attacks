import torch
from cv2 import imread
import os
from transform import xception_default_data_transforms
import shutil
import tqdm
import cv2
from PIL import Image as pil_image

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        os.makedirs('bases', exist_ok=True)
        self.root_dir = 'bases'

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'base_{idx}.png')
        base = imread(img_name)
        base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        preprocess = xception_default_data_transforms['test']
        base = preprocess(pil_image.fromarray(base))
        base = base.unsqueeze(0)
        return base[0], torch.tensor([1,0])    # Real

class PoisonDataset(torch.utils.data.Dataset):
    def __init__(self):
        os.makedirs('poisons', exist_ok=True)
        self.root_dir = 'poisons'

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'poison_{idx}.png')
        preprocess = xception_default_data_transforms['train']
        return preprocess(imread(img_name)), torch.tensor([1,0])    # Real

def fill_bases_directory(image_paths=None):
    base_directory = 'data/ff/original_sequences/youtube/c23/images'

    if not image_paths:
        image_paths = []
        with open('base_images.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_paths.append(line.strip())
        
    print('Filling bases directory')
    os.makedirs('bases', exist_ok=True)
    pb = tqdm.tqdm(total=len(image_paths))
    for i, image_path in enumerate(image_paths):
        shutil.copy(os.path.join(base_directory,image_path), f'bases/base_{i}.png')
        pb.update(1)
    pb.close()
    print('Done filling bases directory')