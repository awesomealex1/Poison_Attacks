from cv2 import imread
import os

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
    path = os.path.join(DATASET_PATHS['Deepfakes'], 'images/953_974/0100.png')
    print(os.path.exists(path))
    print(path)
    return imread(path)

def get_one_real_ff():
    path = os.path.join(DATASET_PATHS['original_youtube'], 'images/953/0100.png')
    return imread(path)

def get_targets_base_pair():
    path_target = os.path.join(DATASET_PATHS['Deepfakes'], 'images/100_100.png')
    path_base = path = os.path.join(DATASET_PATHS['original_youtube'], 'images/100_100.png')
    return (imread(path_target), imread(path_base))