import torch
from tqdm import tqdm
from datasets import FFDataset, PoisonDataset, TrainDataset, TestDataset, get_random_fake
from network.mesonet import MesoInception4
import argparse
from data_util import save_poisons
from train_meso import train_face
from datetime import datetime
import os
from torchvision.utils import save_image
from torchvision import transforms
import psutil

def main(device, max_iters, beta_0, lr, min_base_score, n_bases, model_path, max_poison_distance):
	'''
	Main function to run a poisoning attack on the Meso network.
	Args:
		device: cuda or cpu
		max_iters: Maximum number of iterations to create one poison
		beta_0: beta 0 from poison frogs paper
		lr: Learning rate for poison creation
		n_bases: Number of base images to create (when not having preselected bases)
		model_path: Path to model to use for attack
	Does not return anything but will create files with data and prints results.
	'''
	print('Starting baseline poison attack v2 for meso face')
	network = torch.load(model_path, map_location=device).to(device)
	day_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
	network_name = f'meso_face_c23_baseline_attack_v2_{day_time}'
	# Preparing for poison attack
	beta = beta_0 * (16*8*8)**2/(256*256)**2    # base_instance_dim = 256*256 and feature_dim = 16*8*8
	feature_space = get_headless_network(network)

	target = get_random_fake(face=True)
	target = target.to(device)
	while predict_image(network, target, device)[1][0][1].item() <= 0.9:
		target = get_random_fake(face=True)
		target = target.to(device)
	os.makedirs(f'data/targets/{network_name}', exist_ok=True)
	save_image(target, f'data/targets/{network_name}/target.png')
	print(f'Original target prediction: {predict_image(network, target, device)}')

	poisons = feature_coll(feature_space, target, max_iters, beta, lr, network, device, network_name=network_name, n_bases=n_bases, max_poison_distance=max_poison_distance)

	save_poisons(poisons, network_name)
	del poisons
	if device.type == 'cuda':
		torch.cuda.empty_cache()
	poison_dataset = PoisonDataset(network_name=network_name)
	train_dataset = TrainDataset(face=True)
	merged_dataset = torch.utils.data.ConcatDataset([poison_dataset, train_dataset])
	del poison_dataset, train_dataset
	network_scratch = MesoInception4()
	network_scratch = network_scratch.to(device)
	network_scratch_name = f'meso_face_c23_baseline_attack_v2_scratch_{day_time}'
	poisoned_network = train_face(network_scratch, device, dataset=merged_dataset, name=network_scratch_name, target=target)
	print(f'Target prediction before retraining from scratch: {predict_image(network, target, device)}')
	print(f'Target prediction after retraining from scratch: {predict_image(poisoned_network, target, device)}')

def predict_image(network, image, device):
	image = transform(image)
	post_function = torch.nn.Softmax(dim = 1)
	image = image.to(device)
	print(image.shape)
	output = network(image)
	output = post_function(output)
	_, prediction = torch.max(output, 1)    # argmax

	return int(prediction.item()), output  # If prediction is 1, then fake, else real

def feature_coll(feature_space, target, max_iters, beta, lr, network, device, network_name, max_poison_distance=-1, n_bases=0):
	poisons = []
	base_dataset = FFDataset('test' , prepare=False, face=True)
	base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=1, shuffle=False)
	while len(poisons) < n_bases:
		base, label = next(iter(base_loader))
		base, label = base.to(device), label.to(device)
		if label.item() == 0:
			poison = single_poison(feature_space, target, base, max_iters, beta, lr, network, device, max_poison_distance=max_poison_distance)
			dist = torch.norm(feature_space(transform(poison)) - feature_space(transform(target)))
			if dist <= max_poison_distance:
				poisons.append(poison)
				print(f'Poison {len(poisons)}/{n_bases} created')
			else:
				print(f'Poison was too far from target in features space: {dist}')
			del base, label, poison
		else:
			del base, label
	del base_dataset, base_loader
	return poisons

def single_poison(feature_space, target, base, max_iters, beta, lr, network, device, decay_coef=0.9, M=20, max_poison_distance=-1):
	x = base
	prev_x = base
	prev_M_objectives = []
	pbar = tqdm(total=max_iters)
	for i in range(max_iters):
		x = forward_backward(feature_space, target, base, x, beta, lr)
		target2, x2, base2 = transform(target), transform(x), transform(base)
		target_space, x_space = feature_space(target2), feature_space(x2)

		if i % 100 == 0:
			print(f'Poison prediction: {predict_image(network, x, device)}')
			print(f'Poison-target feature space distance: {torch.norm(x_space - target_space)}')
			print(f'Poison-base distance: {torch.norm(x2 - base2)}')
		
		new_obj = torch.norm(x_space - target_space) + beta*torch.norm(x2 - base2)
		
		if i == max_iters-1 or i == 0:
			print(new_obj)
		
		if i % 1000 == 0 and i > 0:
			lr /= 2
		if max_poison_distance > 0 and i == 4000 and torch.norm(x_space - target_space) > max_poison_distance * 1.5:
			print(max_poison_distance)
			print(f'Poison was too far from target in features space: {torch.norm(x_space - target_space)}')
			del x2, target2, base2, x_space, target_space
			break
		del x2, target2, base2, x_space, target_space
		if device.type == 'cuda':
			torch.cuda.empty_cache()

		pbar.update(1)
	pbar.close()
	del prev_M_objectives
	del new_obj
	return x

def forward_backward(feature_space, target, base, x, beta, lr):
	x_hat = forward(feature_space, target, x, lr)
	new_x = backward(base, x_hat, beta, lr)
	return new_x

def forward(feature_space, target, x, lr):
	detached_x = x.detach()  # Detach x from the computation graph
	x = detached_x.clone().requires_grad_(True)  # Clone and set requires_grad
	print(target.shape, x.shape)
	print(transform(target).shape, transform(x).shape)
	print(feature_space)
	target_space, x_space = feature_space(transform(target)), feature_space(transform(x))
	distance = torch.norm(x_space - target_space)   # Frobenius norm
	feature_space.zero_grad()
	distance.backward()
	img_grad = x.grad.data

	# Update x based on the gradients
	x_hat = x - lr * img_grad
	return x_hat

def backward(base, x_hat, beta, lr):	
	return (x_hat + lr * beta * base) / (1 + beta * lr)

class Flatten(torch.nn.Module):
	'''Layer used for flattening the network.'''
	def forward(self, input):
		return input.view(input.size(0), -1)

def get_headless_network(network):
	'''Returns the network without the last layer.'''
	layer_cake = list(network.children())
	return torch.nn.Sequential(*(layer_cake[:-1]), Flatten())

def transform(img):
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.Normalize([0.5]*3, [0.5]*3)
	])
	return transform(img)

if __name__ == "__main__":
	p = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument('--beta', type=float, help='Beta 0 value for feature collision attack', default=0.1)
	p.add_argument('--max_iters', type=int, help='Maximum iterations for poison creation', default=2000)
	p.add_argument('--poison_lr', type=float, help='Learning rate for poison creation', default=0.0001)
	p.add_argument('--min_base_score', type=float, help='Minimum score for base to be classified as', default=0.9)
	p.add_argument('--n_bases', type=int, help='Number of base images to create', default=50)
	p.add_argument('--model_path', type=str, help='Path to model to use for attack', default='network/weights/meso_face_c23_trained_from_scratch_02_24_2024_16_44_095.p')
	p.add_argument('--max_poison_distance', type=float, help='Maximum distance between poison and target in feature space', default=-1)
	args = p.parse_args()
	
	os.sched_setaffinity(0,set(range(48)))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	main(device, args.max_iters, args.beta, args.poison_lr, args.min_base_score, args.n_bases, args.model_path, args.max_poison_distance)