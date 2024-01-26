from cv2 import imread
import os
import cv2
from transform import xception_default_data_transforms
from PIL import Image as pil_image
from torchvision.utils import save_image
import torch

DATASET_PATHS = {
    'original_actors': 'ff/original_sequences/actors/c23',
    'original_youtube': 'data/ff/original_sequences/youtube/c23',
    'Deepfakes': 'data/ff/manipulated_sequences/Deepfakes/c23',
    'Face2Face': 'ff/manipulated_sequences/Face2Face/c23',
    'FaceSwap': 'ff/manipulated_sequences/FaceSwap/c23',
    'NeuralTextures': 'ff/manipulated_sequences/NeuralTextures/c23',
    'FaceShifter': 'ff/manipulated_sequences/FaceShifter/c23'
}

def get_one_fake_ff():
    path = os.path.join(DATASET_PATHS['Deepfakes'], 'images/100_077/0100.png')
    target = imread(path)
    preprocess = xception_default_data_transforms['test']
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    target = preprocess(pil_image.fromarray(target))
    target = target.unsqueeze(0)

    return target

def get_one_real_ff():
    path = os.path.join(DATASET_PATHS['original_youtube'], 'images/953/0100.png')
    return imread(path)

def get_targets_base_pair():
    path_target = os.path.join(DATASET_PATHS['Deepfakes'], 'images/100_100.png')
    path_base = path = os.path.join(DATASET_PATHS['original_youtube'], 'images/100_100.png')
    return (imread(path_target), imread(path_base))

def save_network(network, name):
    '''
    Saves network to network/weights directory.
    Args:
        network: Network to save
        name: Name to save as
    '''
    torch.save(network, f'network/weights/{name}.p')
    print(f'Saved network as {name}')

def save_poisons(poisons):
    '''
    Saves poisons to data/poisons directory.
    Args:
        poisons: List of images
    '''
    print('Saving poisons')
    os.makedirs('data/poisons', exist_ok=True)
    for i, poison in enumerate(poisons):
        save_image(poison[0], f'data/poisons/poison_{i}.png')
    print('Finished saving poisons')