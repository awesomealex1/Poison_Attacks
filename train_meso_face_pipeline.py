import torch
from network.mesonet import MesoInception4
from train_meso import train_face
from datetime import datetime
from train_meso import eval_network_test
import os
x
def main(device):
	'''
	Main function to train meso on face images.
	Args:
		device: cuda or cpu
	Does not return anything but will create files with data and prints results.
	'''
	print('Starting meso face training')
	os.sched_setaffinity(0,set(range(48)))
	network = MesoInception4()
	network = network.to(device)
	day_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
	network_name = f'meso_face_c23_trained_from_scratch_{day_time}'
	network = train_face(network, device, name=network_name)
	eval_network_test(network, device, name=network_name)

if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	main(device)