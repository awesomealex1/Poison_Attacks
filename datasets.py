import numpy as np
import torch
from cv2 import imread
import os
from transform import xception_default_data_transforms, meso_transform
import shutil
import tqdm
import cv2
from PIL import Image as pil_image
from PIL.Image import open as pil_open
import json
import dlib
from torchvision import transforms

DATASET_PATHS = {
    'original_youtube': 'original_sequences/youtube/c23/images',
    'Deepfakes': 'manipulated_sequences/Deepfakes/c23/images',
    'Face2Face': 'manipulated_sequences/Face2Face/c23/images',
    'FaceSwap': 'manipulated_sequences/FaceSwap/c23/images',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures/c23/images',
    }

SPLIT_PATHS = {
    'train': 'data/ff/splits/train.json',
    'val': 'data/ff/splits/val.json',
    'test': 'data/ff/splits/test.json'
}

CUSTOM_PATH = {
    'base': 'data/bases',
    'poison': 'data/poisons'
}

class FFDataset(torch.utils.data.Dataset):
    def __init__(self, split, meso=False, face=False, prepare=True):
        split_path = SPLIT_PATHS[split]
        self.image_file_paths, self.labels = get_data_labels_from_split(split_path)
        self.face = face
        self.prepare = prepare
        self.face_detector = dlib.get_frontal_face_detector()
        self.meso = meso
        self.split = split

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img_name = self.image_file_paths[idx]
        to_tensor = transforms.Compose([transforms.ToTensor()])
        if self.face:
            img = get_face(img_name, self.face_detector)
            if self.prepare and not self.meso:
                return prepare_image(img, xception_default_data_transforms[self.split]), self.labels[idx]
            elif self.prepare and self.meso:
                return prepare_image(img, meso_transform), self.labels[idx]
            return to_tensor(img), self.labels[idx]
        img = pil_open(img_name)
        if self.prepare and not self.meso:
            return prepare_image(img, xception_default_data_transforms[self.split]), self.labels[idx]
        elif self.prepare and self.meso:
            return prepare_image(img, meso_transform), self.labels[idx]
        img = img.convert("RGB")
        return to_tensor(img), self.labels[idx]
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, type, face=False, prepare=True, network_name=None):
        self.root_dir = CUSTOM_PATH[type]
        if network_name != None:
            self.root_dir += network_name
        os.makedirs(self.root_dir, exist_ok=True)
        self.face = face
        self.prepare = prepare
        self.face_detector = dlib.get_frontal_face_detector()

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'{self.type}_{idx}.png')
        if self.face:
            img = get_face(img_name, self.face_detector)
            if self.prepare:
                return prepare_image(img, xception_default_data_transforms['test']), 0
            return img, 0
        img = pil_open(img_name)
        if self.prepare:
            return prepare_image(img, xception_default_data_transforms['test']), 0    # Real
        img = img.convert("RGB")
        to_tensor = transforms.Compose([transforms.ToTensor()])
        return to_tensor(img), 0

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, face=False, prepare=True, meso=False):
        train_split_path = 'data/ff/splits/train.json'
        self.image_file_paths, self.labels = get_data_labels_from_split(train_split_path)
        self.face = face
        self.prepare = prepare
        self.face_detector = dlib.get_frontal_face_detector()
        self.meso = meso

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img_name = self.image_file_paths[idx]
        if self.face:
            img = get_face(img_name, self.face_detector)
            if self.prepare and self.meso:
                return prepare_image(img, meso_transform), 0
            elif self.prepare:
                return prepare_image(img, xception_default_data_transforms['train']), 0
            return img, self.labels[idx]
        img = pil_open(img_name)
        if self.prepare and self.meso:
            return prepare_image(img, meso_transform), 0
        elif self.prepare:
            return prepare_image(img, xception_default_data_transforms['train']), 0
        img = img.convert("RGB")
        to_tensor = transforms.Compose([transforms.ToTensor()])
        return to_tensor(img), self.labels[idx]
    
class ValDataset(torch.utils.data.Dataset):
    def __init__(self, face=False, prepare=True):
        val_split_path = 'data/ff/splits/val.json'
        self.image_file_paths, self.labels = get_data_labels_from_split(val_split_path)
        self.face = face
        self.prepare = prepare
        self.face_detector = dlib.get_frontal_face_detector()

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img_name = self.image_file_paths[idx]
        if self.face:
            img = get_face(img_name, self.face_detector)
            if self.prepare:
                return prepare_image(img, xception_default_data_transforms['val']), self.labels[idx]
            return img, self.labels[idx]
        img = pil_open(img_name)
        if self.prepare:
            return prepare_image(img, xception_default_data_transforms['val']), self.labels[idx]
        img = img.convert("RGB")
        to_tensor = transforms.Compose([transforms.ToTensor()])
        return to_tensor(img), self.labels[idx]
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, face=False, prepare=True):
        test_split_path = 'data/ff/splits/test.json'
        self.image_file_paths, self.labels = get_data_labels_from_split(test_split_path)
        self.face = face
        self.prepare = prepare
        self.face_detector = dlib.get_frontal_face_detector()

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img_name = self.image_file_paths[idx]
        if self.face:
            img = get_face(img_name, self.face_detector)
            if self.prepare:
                return prepare_image(img, xception_default_data_transforms['test']), self.labels[idx]
            return img, self.labels[idx]
        img = pil_open(img_name)
        if self.prepare:
            return prepare_image(img, xception_default_data_transforms['test']), self.labels[idx]
        img = img.convert("RGB")
        to_tensor = transforms.Compose([transforms.ToTensor()])
        return to_tensor(img), self.labels[idx]
    
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, network_name, face=False, prepare=True):
        self.root_dir = f'data/bases/{network_name}'
        os.makedirs(self.root_dir, exist_ok=True)
        self.face = face
        self.prepare = prepare
        self.face_detector = dlib.get_frontal_face_detector()

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'base_{idx}.png')
        if self.face:
            img = get_face(img_name, self.face_detector)
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
    def __init__(self, network_name, face=False, prepare=True, meso=False):
        self.root_dir = f'data/poisons/{network_name}'
        os.makedirs(self.root_dir, exist_ok=True)
        self.face = face
        self.prepare = prepare
        self.face_detector = dlib.get_frontal_face_detector()
        self.meso = meso

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'poison_{idx}.png')
        if self.face:
            img = get_face(img_name, self.face_detector)
            if self.prepare and self.meso:
                return prepare_image(img, meso_transform), 0
            elif self.prepare:
                return prepare_image(img, xception_default_data_transforms['test']), 0
            return img, 0
        img = pil_open(img_name)
        if self.prepare and self.meso:
            return prepare_image(img, meso_transform), 0
        elif self.prepare:
            return prepare_image(img, xception_default_data_transforms['test']), 0
        img = img.convert("RGB")
        to_tensor = transforms.Compose([transforms.ToTensor()])
        return to_tensor(img), 0

def get_face(img_name, face_detector=dlib.get_frontal_face_detector()):
    img = pil_open(img_name)
    gray = np.asarray(img.convert('L'))
    faces = face_detector(gray, 1)
    height, width = gray.shape[:2]
    if len(faces):
        face = faces[0]
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = img.crop((x, y, x+size, y+size))
        #cropped_face = np.asarray(img)[y:y+size, x:x+size]
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
        img = pil_open('data/ff/original_sequences/youtube/c23/images/970/0049.png')
    img = img.convert("RGB")
    return transform(img)

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

def get_random_fake(face=False):
    test_split_path = 'data/ff/splits/test.json'
    image_file_paths, labels = get_data_labels_from_split(test_split_path)
    random_idx = np.random.randint(len(image_file_paths))
    while labels[random_idx] == 0:
        random_idx = np.random.randint(len(image_file_paths))
    
    random_image = image_file_paths[random_idx]
    if face:
        image = get_face(random_image)
    else:
        image = imread(random_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    to_tensor = transforms.Compose([transforms.ToTensor()])
    return torch.unsqueeze(to_tensor(image), 0)

def get_random_real(face=False):
    real_dir = 'data/ff/original_sequences/youtube/c23/images'
    video_paths = os.listdir(real_dir)
    
    random_vid = video_paths[np.random.randint(len(video_paths))]
    images = os.listdir(os.path.join(real_dir, random_vid))
    random_image = images[np.random.randint(len(images))]
    if face:
        image = get_face(os.path.join(real_dir, random_vid, random_image))
    else:
        image = imread(os.path.join(real_dir, random_vid, random_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    to_tensor = transforms.Compose([transforms.ToTensor()])
    return torch.unsqueeze(to_tensor(image), 0)