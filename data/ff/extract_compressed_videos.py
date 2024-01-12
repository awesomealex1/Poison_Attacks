"""
Extracts images from (compressed) videos, used for the FaceForensics++ dataset

Usage: see -h or https://github.com/ondyari/FaceForensics

Author: Andreas Roessler
Date: 25.01.2019
"""
import os
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import shutil


DATASET_PATHS = {
    'original_youtube': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}
COMPRESSION = ['c0', 'c23', 'c40']


def extract_frames(data_path, output_path, method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)
    print(output_path)
    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)),
                        image)
            frame_num += 1
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))

def find_corrupt(data_path, dataset, compression):
    corrupt_videos = []
    corrupt_images = []
    for dataset in DATASET_PATHS.keys():
        videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
        images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
        images = os.listdir(images_path)
        for image_folder in images:
            images_in_folder = os.listdir(join(images_path, image_folder))
            random_image = cv2.imread(join(images_path, image_folder, images_in_folder[0]))
            if random_image is None:
                corrupt_videos.append(join(videos_path, image_folder + '.mp4'))
                corrupt_images.append(join(images_path, image_folder))
    return zip(corrupt_videos, corrupt_images)

def find_missing(data_path, dataset, compression):
    missing_videos = []
    missing_images = []
    for dataset in DATASET_PATHS.keys():
        videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
        images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
        videos = os.listdir(videos_path)
        for video in videos:
            if not os.path.exists(join(images_path, video.split('.')[0])):
                missing_videos.append(join(videos_path, video))
                missing_images.append(join(images_path, video.split('.')[0]))
    return zip(missing_videos, missing_images)

def fix_corrupt(corrupt_paths):
    for video_path,image_path in corrupt_paths:
        print(image_path)
        if os.path.exists(image_path):
            shutil.rmtree(image_path)

    num_processes = 4
    with Pool(num_processes) as p:
        p.starmap(extract_frames, corrupt_paths)

def extract_method_videos(data_path, dataset, compression):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')

    videos = os.listdir(videos_path)
    data_paths = [join(videos_path, video) for video in videos]
    output_paths = [join(images_path, video.split('.')[0]) for video in videos]

    print("Starting frame extraction for:", dataset)

    num_processes = 4
    with Pool(num_processes) as p:
        p.starmap(extract_frames, zip(data_paths, output_paths))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c0')
    args = p.parse_args()

    corrupt = True # Use this if some videos havent been extracted properly and you need to repeat for single videos
    if corrupt:
        corrupt_paths = find_corrupt(**vars(args))
        missing_paths = find_missing(**vars(args))
        fix_corrupt(corrupt_paths)
        fix_corrupt(missing_paths)
    else:
        if args.dataset == 'all':
            for dataset in DATASET_PATHS.keys():
                args.dataset = dataset
                extract_method_videos(**vars(args))
        else:
            extract_method_videos(**vars(args))
