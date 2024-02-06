import torch
from network.models import get_xception_untrained
import argparse
from train import train_full
from datetime import datetime
from train import eval_network_test

def main(device):
	'''
	Main function to train xception full image.
	Args:
		device: cuda or cpu
	Does not return anything but will create files with data and prints results.
	'''
	print('Starting xception full training')

	network = get_xception_untrained()
	network = network.to(device)
	day_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
	network_name = f'xception_full_c23_trained_from_scratch_{day_time}'
	network = train_full(network, device, name=network_name)
	eval_network_test(network, device, name=network_name)

if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	main(device)