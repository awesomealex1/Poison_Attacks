import numpy as np
import torch
from cv2 import imread
import os
from transform import xception_default_data_transforms
import shutil
import tqdm
import cv2
from PIL import Image as pil_image
from PIL.Image import open as pil_open
import json
import dlib
from torchvision import transforms
import time

DATASET_PATHS = {
    'original_youtube': 'original_sequences/youtube/c23/images',
    'Deepfakes': 'manipulated_sequences/Deepfakes/c23/images',
    'Face2Face': 'manipulated_sequences/Face2Face/c23/images',
    'FaceSwap': 'manipulated_sequences/FaceSwap/c23/images',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures/c23/images',
    }

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, face=False, prepare=True):
        train_split_path = 'data/ff/splits/train.json'
        self.image_file_paths, self.labels = get_data_labels_from_split(train_split_path)
        self.face = face
        self.prepare = prepare

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img_name = self.image_file_paths[idx]
        if self.face:
            img = get_face(img_name)
            if self.prepare:
                return prepare_image(img, xception_default_data_transforms['train']), self.labels[idx]
            return img, self.labels[idx]
        a = time.time()
        img = pil_open(img_name)
        b = time.time()
        print((b-a)*32)
        if self.prepare:
            a = time.time()
            p = prepare_image(img, xception_default_data_transforms['train']), self.labels[idx]
            b = time.time()
            print("XXXXXX", (b-a)*32)
            return p
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.convert("RGB")
        #img = pil_image.fromarray(img)
        to_tensor = transforms.Compose([transforms.ToTensor()])
        img = to_tensor(img)
        return img, self.labels[idx]
    
class ValDataset(torch.utils.data.Dataset):
    def __init__(self, face=False, prepare=True):
        val_split_path = 'data/ff/splits/val.json'
        self.image_file_paths, self.labels = get_data_labels_from_split(val_split_path)
        self.face = face
        self.prepare = prepare

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img_name = self.image_file_paths[idx]
        if self.face:
            img = get_face(img_name)
            if self.prepare:
                return prepare_image(img, xception_default_data_transforms['val']), self.labels[idx]
            return img, self.labels[idx]
        img = imread(img_name)
        if self.prepare:
            return prepare_image(img, xception_default_data_transforms['val']), self.labels[idx]
        return img, self.labels[idx]
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, face=False, prepare=True):
        test_split_path = 'data/ff/splits/test.json'
        self.image_file_paths, self.labels = get_data_labels_from_split(test_split_path)
        self.face = face
        self.prepare = prepare

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img_name = self.image_file_paths[idx]
        if self.face:
            img = get_face(img_name)
            if self.prepare:
                return prepare_image(img, xception_default_data_transforms['test']), self.labels[idx]
            return img, self.labels[idx]
        img = imread(img_name)
        if self.prepare:
            return prepare_image(img, xception_default_data_transforms['test']), self.labels[idx]
        return img, self.labels[idx]

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, face=False, prepare=True):
        self.root_dir = 'data/bases'
        os.makedirs(self.root_dir, exist_ok=True)
        self.face = face
        self.prepare = prepare

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'base_{idx}.png')
        if self.face:
            img = get_face(img_name)
            if self.prepare:
                return prepare_image(img, xception_default_data_transforms['test']), 0
            return img, 0
        img = imread(img_name)
        if self.prepare:
            return prepare_image(img, xception_default_data_transforms['test']), 0    # Real
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = pil_image.fromarray(img)
        to_tensor = transforms.Compose([transforms.ToTensor()])
        img = to_tensor(img)
        return img, 0

class PoisonDataset(torch.utils.data.Dataset):
    def __init__(self, face=False, prepare=True):
        self.root_dir = 'data/poisons'
        os.makedirs(self.root_dir, exist_ok=True)
        self.face = face
        self.prepare = prepare

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'poison_{idx}.png')
        if self.face:
            img = get_face(img_name)
            if self.prepare:
                return prepare_image(img, xception_default_data_transforms['test']), 0
            return img, 0
        img = imread(img_name)
        if self.prepare:
            return prepare_image(img, xception_default_data_transforms['test']), 0    # Real
        return img, 0

def get_face(img_name):
    img = imread(img_name)
    face_detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    height, width = img.shape[:2]
    if len(faces):
        face = faces[0]
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = img[y:y+size, x:x+size]
        return cropped_face
    print('Could not find a face')
    return img

def fill_bases_directory(image_paths=None):
    base_class_directory = 'data/ff/original_sequences/youtube/c23/images'

    if not image_paths:
        image_paths = []
        with open('base_images.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_paths.append(line.strip())
        
    print('Filling bases directory')
    if os.path.isdir('data/bases'):
        shutil.rmtree('data/bases', ignore_errors=False, onerror=None)
    os.makedirs('data/bases')
    pb = tqdm.tqdm(total=len(image_paths))
    for i, image_path in enumerate(image_paths):
        shutil.copy(os.path.join(base_class_directory,image_path), f'data/bases/base_{i}.png')
        pb.update(1)
    pb.close()
    print('Done filling bases directory')

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def prepare_image(img, transform):
    if img is None:
        print(f'Could not read image')
        #img = imread('data/ff/original_sequences/youtube/c23/images/970/0049.png')
        img = pil_open('data/ff/original_sequences/youtube/c23/images/970/0049.png')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.convert("RGB")
    #img = transform(pil_image.fromarray(img))
    return img

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
    for name, path in DATASET_PATHS.items():
        for video_id in video_ids:
            if name == 'original_youtube':
                video_id = video_id[:3]
                label = 0
            else:
                label = 1
            for image_file in os.listdir(os.path.join(root_dir, path, video_id)):
                image_file_paths.append(os.path.join(root_dir, path, video_id, image_file))
                labels.append(label)
    
    return image_file_paths, labels

def get_random_fake():
    root_dir = 'data/ff'
    video_paths = []
    for name, path in DATASET_PATHS.items():
        if name != 'original_youtube':
            video_paths = os.listdir(os.path.join(root_dir, path))
    
    random_vid = video_paths[np.random.randint(len(video_paths))]
    images = os.listdir(os.path.join(root_dir, path, random_vid))
    random_image = images[np.random.randint(len(images))]
    image = imread(os.path.join(root_dir, path, random_vid, random_image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image.fromarray(image)
    to_tensor = transforms.Compose([transforms.ToTensor()])
    return torch.unsqueeze(to_tensor(image), 0)