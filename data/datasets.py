import torch
from cv2 import imread
import os
from transform import xception_default_data_transforms
import shutil
import tqdm
import cv2
from PIL import Image as pil_image
import json

DATASET_PATHS = {
    'original_youtube': 'original_sequences/youtube/c23/images',
    'Deepfakes': 'manipulated_sequences/Deepfakes/c23/images',
    'Face2Face': 'manipulated_sequences/Face2Face/c23/images',
    'FaceSwap': 'manipulated_sequences/FaceSwap/c23/images',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures/c23/images',
    }

def get_data_labels_from_split(split_path):
    root_dir = 'data/ff'
    video_ids = []
    with open(split_path) as splits_list:
        splits = json.load(splits_list)
        for split in splits:
            target = split[0]
            source = split[1]
            video_ids.append(f'{target}_{source}')

    labels = []
    image_file_paths = []
    for name, path in DATASET_PATHS:
        for video_id in video_ids:
            if name == 'original_youtube':
                image_file_paths.append(os.path.join(root_dir, path, video_id[:3]))
                labels.append(0)
            else:
                image_file_paths.append(os.path.join(root_dir, path, video_id))
                labels.append(1)
    
    return image_file_paths, labels

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        train_split_path = 'data/ff/splits/train.json'
        self.image_file_paths, self.labels = get_data_labels_from_split(train_split_path)

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img = imread(self.image_file_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocess = xception_default_data_transforms['train']
        img = preprocess(pil_image.fromarray(img))
        img = img.unsqueeze(0)
        return img[0], self.labels[idx]    # Real
    
class ValDataset(torch.utils.data.Dataset):
    def __init__(self):
        val_split_path = 'data/ff/splits/val.json'
        self.image_file_paths, self.labels = get_data_labels_from_split(val_split_path)

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img = imread(self.image_file_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocess = xception_default_data_transforms['val']
        img = preprocess(pil_image.fromarray(img))
        img = img.unsqueeze(0)
        return img[0], self.labels[idx]    # Real
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        test_split_path = 'data/ff/splits/test.json'
        self.image_file_paths, self.labels = get_data_labels_from_split(test_split_path)

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img = imread(self.image_file_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocess = xception_default_data_transforms['test']
        img = preprocess(pil_image.fromarray(img))
        img = img.unsqueeze(0)
        return img[0], self.labels[idx]    # Real

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root_dir = 'data/bases'
        os.makedirs(self.root_dir, exist_ok=True)

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
        self.root_dir = 'data/poisons'
        os.makedirs(self.root_dir, exist_ok=True)

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'poison_{idx}.png')
        poison = imread(img_name)
        poison = cv2.cvtColor(poison, cv2.COLOR_BGR2RGB)
        preprocess = xception_default_data_transforms['test']
        poison = preprocess(pil_image.fromarray(poison))
        poison = poison.unsqueeze(0)
        return poison[0], 0    # Real

def fill_bases_directory(image_paths=None):
    base_class_directory = 'data/ff/original_sequences/youtube/c23/images'

    if not image_paths:
        image_paths = []
        with open('base_images.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_paths.append(line.strip())
        
    print('Filling bases directory')
    os.makedirs('data/bases', exist_ok=True)
    pb = tqdm.tqdm(total=len(image_paths))
    for i, image_path in enumerate(image_paths):
        shutil.copy(os.path.join(base_class_directory,image_path), f'data/bases/base_{i}.png')
        pb.update(1)
    pb.close()
    print('Done filling bases directory')