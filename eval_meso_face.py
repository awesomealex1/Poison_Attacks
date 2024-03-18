import torch
from datetime import datetime
from train_meso import eval_network_test
import os

def main(device):
    '''
    Main function to train xception full image.
    Args:
        device: cuda or cpu
    Does not return anything but will create files with data and prints results.
    '''
    print('Starting xception full training')
    os.sched_setaffinity(0, set(range(48)))
    model_path = 'network/weights/meso_face_c23_trained_from_scratch_02_24_2024_16_44_095.p'
    network = torch.load(model_path, map_location=device).to(device)
    day_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    network_name = f'meso_face_c23_baseline_{day_time}'
    eval_network_test(network, device, name=network_name, face=True)

if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	main(device)