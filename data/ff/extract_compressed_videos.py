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


DATASET_PATHS = {
#    'original_actors': 'original_sequences/actors',
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


def extract_method_videos(data_path, dataset, compression):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')

    videos = os.listdir(videos_path)
    data_paths = [join(videos_path, video) for video in videos]
    output_paths = [join(images_path, video.split('.')[0]) for video in videos]

    with Pool(5) as p:
        p.map(lambda x, y: extract_frames(x,y), data_paths, output_paths)

    #for video in tqdm(os.listdir(videos_path)):
    #    image_folder = video.split('.')[0]
    #    extract_frames(join(videos_path, video),
    #                   join(images_path, image_folder))


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

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))
