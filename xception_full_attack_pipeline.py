import torch
from tqdm import tqdm
from datasets import BaseDataset, PoisonDataset, TrainDataset, TestDataset, ValDataset, fill_bases_directory, get_random_fake, get_random_real
from network.models import get_xception_full, get_xception_untrained, model_selection
import argparse
from data_util import save_network, save_poisons, get_one_fake_ff
from train import train_on_ff, train_full, train_transfer
from datetime import datetime
import os
from torchvision.utils import save_image
from torchvision import transforms

def main(device, max_iters, beta_0, lr, pretrained, preselected_bases, min_base_score, max_base_distance, n_bases, model_path, transfer, max_poison_distance):
	'''
	Main function to run a poisoning attack on the Xception network.
	Args:
		device: cuda or cpu
		max_iters: Maximum number of iterations to create one poison
		beta_0: beta 0 from poison frogs paper
		lr: Learning rate for poison creation
		pretrained: Whether to use FF++ provided pretrained network
		preselected_bases: Whether to use a txt file with base images
		max_base_distance: Maximum distance between base and target (when not having preselected bases)
		n_bases: Number of base images to create (when not having preselected bases)
		model_path: Path to model to use for attack
		transfer: Whether to use transfer learning in attack
	Does not return anything but will create files with data and prints results.
	'''
	print('Starting poison attack')

	if pretrained:
		network = get_xception_full(device)
	elif model_path:
		print(model_path)
		print(os.path.isfile(model_path))
		network = torch.load(model_path, map_location=device)
	else:
		network = get_xception_untrained()
	
	network = network.to(device)

	day_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
	network_name = f'xception_full_c23_trained_from_scratch_{day_time}'

	if not (pretrained or model_path):
		network = train_full(network, device, name=network_name, epochs=1)
	
	# Preparing for poison attack
	beta = beta_0 * 2048**2/(299*299)**2    # base_instance_dim = 299*299 and feature_dim = 2048
	feature_space = get_headless_network(network)
	headless_network = get_headless_network(network)
	target = get_random_fake()
	target = target.to(device)
	while predict_image(network, target, device, processed=False)[1][0][1].item() <= 0.9:
		target = get_random_fake()
		target = target.to(device)

	if not preselected_bases:
		bases = create_bases(min_base_score, max_base_distance, n_bases, headless_network, target, network, device)
	else:
		fill_bases_directory()

	os.makedirs(f'data/bases/{network_name}', exist_ok=True)
	for i in range(len(bases)):
		save_image(bases[i], f'data/bases/{network_name}/base_{i}.png')
	os.makedirs(f'data/{network_name}', exist_ok=True)
	save_image(target, f'data/{network_name}/target.png')

	print(f'Original target prediction: {predict_image(network, target, device, processed=False)}')
	poisons = feature_coll(feature_space, target, max_iters, beta, lr, network, device, max_poison_distance=max_poison_distance, network_name=network_name, n_bases=n_bases)
	save_poisons(poisons, network_name)

	poison_dataset = PoisonDataset(network_name=network_name)
	train_dataset = TrainDataset()
	merged_dataset = torch.utils.data.ConcatDataset([poison_dataset, train_dataset])

	# Poisoning network and eval
	if not transfer:
		untrained_network = get_xception_untrained()
		untrained_network.to(device)
		poisoned_network = train_full(untrained_network, device, dataset=merged_dataset, name=network_name, target=preprocess(target))
	else:
		poisoned_network = train_transfer(network, device, dataset=merged_dataset, name=network_name, target=preprocess(target))
	print(f'Target prediction after retraining from scratch: {predict_image(poisoned_network, target, device, processed=False)}')

def create_bases(min_base_score, max_base_distance, n_bases, feature_space, target, network, device):
	print('Creating bases')
	base_images = []
	train_dataset = TrainDataset(prepare=False)
	target_feature = feature_space(preprocess(target))
	data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
	pbar = tqdm(total=n_bases)
	for i, (image, label) in enumerate(data_loader, 0):
		image,label = image.to(device), label.to(device)
		if label.item() == 0:   # If real
			image_features = feature_space(preprocess(image))
			_, image_score = predict_image(network, preprocess(image), device)
			distance = torch.norm(image_features - target_feature)
			if image_score[0][0].item() >= min_base_score and distance <= max_base_distance:
				print(image_score)
				base_images.append(image)
				pbar.update(1)
		if len(base_images) == n_bases:
			break
	pbar.close()
	return base_images

def predict_image(network, image, device, processed=True):
	'''
	Predicts the label of an input image.
	Args:
		image: numpy image
		network: torch model with linear layer at the end
	Returns:
		prediction (1 = fake, 0 = real)
		output: Output of network
	'''
	if not processed:
		image = preprocess(image)
	post_function = torch.nn.Softmax(dim = 1)
	image = image.to(device)
	output = network(image)
	output = post_function(output)
	_, prediction = torch.max(output, 1)    # argmax

	return int(prediction.item()), output  # If prediction is 1, then fake, else real

def feature_coll(feature_space, target, max_iters, beta, lr, network, device, max_poison_distance=-1, network_name=None, n_bases=0):
	'''
	Performs feature collision attack on the target image.
	Args:
		feature_space: Feature space of network
		target: Target image
		max_iters: Maximum number of iterations to create one poison
		beta: Beta value for feature collision attack
		lr: Learning rate for poison creation
		network: Network to attack
	Returns:
		poisons: List of poisons
	'''
	poisons = []
	if max_poison_distance < 0:
		base_dataset = BaseDataset(prepare=False, network_name=network_name)
		base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=1, shuffle=False)
		for i, (base,label) in enumerate(base_loader, 1):
			base, label = base.to(device), label.to(device)
			poison = single_poison(feature_space, target, base, max_iters, beta, lr, network, device)
			poisons.append(poison)
			print(f'Poison {i}/{len(base_dataset)} created')
	else:
		i = 0
		while len(poisons) < n_bases:
			base, label = get_random_real(), torch.Tensor(0)
			base, label = base.to(device), label.to(device)
			poison = single_poison(feature_space, target, base, max_iters, beta, lr, network, device)
			dist = torch.norm(feature_space(preprocess(poison)) - feature_space(preprocess(target)))
			if dist <= max_poison_distance:
				poisons.append(poison)
				print(f'Poison {len(poisons)}/{n_bases} created')
			else:
				print(f'Poison was too far from target in features space: {dist}')
			i += 1
			if i % 10 == 0:
				max_poison_distance += 5
	return poisons

def single_poison(feature_space, target, base, max_iters, beta, lr, network, device, decay_coef=0.9, M=20):
	'''
	Creates a single poison.
	Args:
		feature_space: Feature space of network
		target: Target image
		base: Base image
		max_iters: Maximum number of iterations to create one poison
		beta: Beta value for feature collision attack
		lr: Learning rate for poison creation
		network: Network to attack
		decay_coef: Decay coefficient for learning rate
		M: Number of previous objectives to average over (used for learning rate decay)
	Returns:
		x: Poison image
	'''
	x = base
	prev_x = base
	prev_M_objectives = []
	pbar = tqdm(total=max_iters)
	for i in range(max_iters):
		x = forward_backward(feature_space, target, base, x, beta, lr)
		target2 = preprocess(target)
		x2 = preprocess(x)
		base2 = preprocess(base)
		target_space = feature_space(target2)
		x_space = feature_space(x2)
		if i == max_iters-1 or i == 0:
			print(f'Poison prediction: {predict_image(network, x, device, processed=False)}')
			print(f'Poison-target feature space distance: {torch.norm(x_space - target_space)}')
			print(f'Poison-base distance: {torch.norm(x2 - base2)}')

		new_obj = torch.norm(x_space - target_space) + beta*torch.norm(x2 - base2)
		avg_of_last_M = sum(prev_M_objectives)/float(min(M, i+1))
		
		if i == max_iters-1 or i == 0:
			print(new_obj)

		if new_obj >= avg_of_last_M and (i % M/2 == 0):
			lr *= decay_coef
			x = prev_x
		else:
			prev_x = x
		
		if i < M-1:
			prev_M_objectives.append(new_obj)
		else:
			#first remove the oldest obj then append the new obj
			del prev_M_objectives[0]
			prev_M_objectives.append(new_obj)

		pbar.update(1)
	pbar.close()
	return x

def forward_backward(feature_space, target, base, x, beta, lr):
	'''Performs forward and backward passes.'''
	x_hat = forward(feature_space, target, x, lr)
	new_x = backward(base, x_hat, beta, lr)
	return new_x

def forward(feature_space, target, x, lr):
	'''Performs forward pass.'''
	detached_x = x.detach()  # Detach x from the computation graph
	x = detached_x.clone().requires_grad_(True)  # Clone and set requires_grad
	target_space = feature_space(target)
	x_space = feature_space(x)
	distance = torch.norm(x_space - target_space)   # Frobenius norm

	feature_space.zero_grad()
	distance.backward()
	img_grad = x.grad.data

	# Update x based on the gradients
	x_hat = x - lr * img_grad
	return x_hat

def backward(base, x_hat, beta, lr):
	'''Performs backward pass.'''
	return (x_hat + lr * beta * base) / (1 + beta * lr)

class Flatten(torch.nn.Module):
	'''Layer used for flattening the network.'''
	def forward(self, input):
		return input.view(input.size(0), -1)

def get_headless_network(network):
	'''Returns the network without the last layer.'''
	layer_cake = list(network.model.children())
	return torch.nn.Sequential(*(layer_cake[:-1]), Flatten())

def preprocess(img):
	transform = transforms.Compose([
		transforms.Resize((299, 299)),
		transforms.Normalize([0.5]*3, [0.5]*3)
	])
	return transform(img)

if __name__ == "__main__":
	p = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument('--beta', type=float, help='Beta 0 value for feature collision attack', default=0.1)
	p.add_argument('--max_iters', type=int, help='Maximum iterations for poison creation', default=2000)
	p.add_argument('--poison_lr', type=float, help='Learning rate for poison creation', default=0.0001)
	p.add_argument('--pretrained', action='store_true', help='Whether to use FF++ provided pretrained network')
	p.add_argument('--preselected_bases', action='store_true', help='Whether to use a txt file with base images')
	p.add_argument('--max_base_distance', type=float, help='Maximum distance between base and target', default=9000)
	p.add_argument('--min_base_score', type=float, help='Minimum score for base to be classified as', default=0.9)
	p.add_argument('--n_bases', type=int, help='Number of base images to create', default=5)
	p.add_argument('--model_path', type=str, help='Path to model to use for attack', default=None)
	p.add_argument('--transfer', action='store_true', help='Whether to use transfer learning')
	p.add_argument('--max_poison_distance', type=float, help='Maximum distance between poison and target in feature space', default=50)
	args = p.parse_args()
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	main(device, args.max_iters, args.beta, args.poison_lr, args.pretrained, args.preselected_bases, args.min_base_score, args.max_base_distance, args.n_bases, args.model_path, args.transfer, args.max_poison_distance)