import torch
from tqdm import tqdm
from datasets import BaseDataset, PoisonDataset, TrainDataset, TestDataset, get_random_fake, get_random_real, get_image
from network.models import get_xception_untrained
import argparse
from data_util import save_poisons
from train import train_full
from datetime import datetime
import os
from torchvision.utils import save_image
from torchvision import transforms
import psutil

def main(device, model_path, targets_path):
    '''
    Main function to run a poisoning attack on the Xception network.
    Args:
        device: cuda or cpu
        max_iters: Maximum number of iterations to create one poison
        beta_0: beta 0 from poison frogs paper
        lr: Learning rate for poison creation
        n_bases: Number of base images to create (when not having preselected bases)
        model_path: Path to model to use for attack
    Does not return anything but will create files with data and prints results.
    '''
    network = torch.load(model_path, map_location=device).to(device)
    for target in sorted(os.listdir(targets_path)):
        target_tensor = get_image(os.path.join(targets_path, target))
        print(target)
        print(f'Target prediction: {predict_image(network, target_tensor, device)}')

def predict_image(network, image, device):
	'''
	Predicts the label of an input image.
	Args:
		image: numpy image
		network: torch model with linear layer at the end
	Returns:
		prediction (1 = fake, 0 = real)
		output: Output of network
	'''
	image = transform(image)
	post_function = torch.nn.Softmax(dim = 1)
	image = image.to(device)
	output = network(image)
	output = post_function(output)
	_, prediction = torch.max(output, 1)    # argmax

	return int(prediction.item()), output  # If prediction is 1, then fake, else real

def transform(img):
	transform = transforms.Compose([
		transforms.Resize((299, 299)),
		transforms.Normalize([0.5]*3, [0.5]*3)
	])
	return transform(img)

if __name__ == "__main__":
	p = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument('--model_path', type=str, help='Path to model to use for attack', default='network/weights/xception_full_c23_trained_from_scratch_03_31_2024_15_54_00_frozen1.p')
	p.add_argument('--targets_path', type=str, help='Path to directory with targets', default='data/targets/finetuning_full/targets')
	args = p.parse_args()
	
	os.sched_setaffinity(0,set(range(48)))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	main(device, args.model_path, args.targets_path)