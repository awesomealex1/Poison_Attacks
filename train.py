from data_util import save_network
import torch
from tqdm import tqdm
from datasets import FFDataset
import torch.nn as nn
from experiment_util import save_training_epoch, save_validation_epoch, save_test, save_target_results
from torchvision import transforms
import os, psutil

def train_full(network, device, dataset=FFDataset('train'), name='xception_full_c23_trained_from_scratch', target=None, transfer=False):
    if transfer:
        network = train_on_ff(network, device, dataset, f'{name}_frozen', frozen=True, epochs=5, target=target)
    else:
        network = train_on_ff(network, device, dataset, f'{name}_frozen', frozen=True, epochs=3, target=target)
        network = train_on_ff(network, device, dataset, name, frozen=False, epochs=7, target=target)
    return network

def train_face(network, device, dataset=FFDataset('train', face=True), name='xception_face_c23_trained_from_scratch', target=None, transfer=False):
    if transfer:
        network = train_on_ff(network, device, dataset, f'{name}_frozen', frozen=True, epochs=3, target=target, face=True)
    else:
        network = train_on_ff(network, device, dataset, f'{name}_frozen', frozen=True, epochs=3, target=target, face=True)
        network = train_on_ff(network, device, dataset, name, frozen=False, epochs=7, target=target, face=True)
    return network

def train_transfer(network, device, dataset=FFDataset('train'), name='xception_full_transfer_c23', target=None):
    #network.apply(randomize_last_layer)
    network = train_on_ff(network, device, dataset, f'{name}_frozen', frozen=True, epochs=3, target=target)
    return network

def train_on_ff(network, device, dataset=FFDataset('train'), name='xception_full_c23_trained_from_scratch', frozen=False, epochs=8, lr=0.0002, batch_size=32, target=None, face=False, start_epoch=0):
    '''
    Trains the network.
    Args:
        network (torch.nn.Module): The network to train.
        device (str): The device to use for training (cuda or cpu).
        dataset (torch.utils.data.Dataset, optional): The dataset to train on. Defaults to TrainDataset().
        name (str, optional): The name of the trained network. Defaults to 'xception_full_c23_trained_from_scratch'.
        frozen (bool, optional): Whether to freeze all but the last layer of the network. Defaults to False.
        epochs (int, optional): The number of training epochs. Defaults to 3.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.0002.
        batch_size (int, optional): The batch size for training. Defaults to 32.
    Returns:
        torch.nn.Module: The trained network.
    '''
    print('Training on FF++')

    network = freeze_all_but_last_layer(network) if frozen else unfreeze_all(network)
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    weight = torch.tensor([4.0, 1.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    best_score = None
    best_network = None
    for epoch in range(start_epoch, epochs):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=False)
        total_loss = 0.0
        fake_correct = fake_incorrect = real_correct = real_incorrect = 0

        pb = tqdm(total=len(data_loader))
        for i, (image, label) in enumerate(data_loader, 0):
            optimizer.zero_grad()
            image,label = image.to(device), label.to(device)
            outputs = network(image)
            loss = criterion(outputs, label)
            for j, pred in enumerate(outputs, 0):
                real_score = pred[0].item()
                fake_score = pred[1].item()
                if real_score > fake_score and label[j].item() == 0:
                    real_correct += 1
                elif real_score < fake_score and label[j].item() == 0:
                    real_incorrect += 1
                elif real_score > fake_score and label[j].item() == 1:
                    fake_incorrect += 1
                elif real_score < fake_score and label[j].item() == 1:
                    fake_correct += 1
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pb.update(1)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        pb.close()
        total_loss /= len(data_loader)
        save_network(network, f'{name}{epoch}')
        save_training_epoch(name, epoch, total_loss, fake_correct, fake_incorrect, real_correct, real_incorrect)

        eval_network(network, device, name=f'{name}', target=target, fraction_to_eval=1, epoch=epoch, face=face)
        score = (fake_correct + real_correct)/(fake_correct + fake_incorrect + real_correct + real_incorrect)
        if best_score is None or score > best_score:
            best_score = score
            best_network = f'{name}{epoch}'
    
    print(f'Best network: {best_network}')
    return network

def randomize_last_layer(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

def freeze_all_but_last_layer(network):
    '''Freezes all but the last layer of the network.'''
    layer_cake = list(network.model.children())
    for layer in layer_cake[:-1]:
        for param in layer.parameters():
            param.requires_grad = False
    return network

def unfreeze_all(network):
    '''Unfreezes all layers of the network.'''
    layer_cake = list(network.model.children())
    for layer in layer_cake:
        for param in layer.parameters():
            param.requires_grad = True
    return network

def eval_network(network, device, batch_size=32, name='xception_full_c23_trained_from_scratch', target=None, fraction_to_eval=1, epoch=0, face=False):
    '''Evaluates the network performance on test set.'''
    print('Evaluating network')
    val_dataset = FFDataset('val', face=face)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=False)
    criterion = torch.nn.CrossEntropyLoss()

    fake_correct = fake_incorrect = real_correct = real_incorrect = 0

    network.eval()
    pb = tqdm(total=len(val_loader)*fraction_to_eval)
    total_loss = 0.0
    with torch.no_grad():
        for i, (image, label) in enumerate(val_loader, 0):
            image, label = image.to(device), label.to(device)
            prediction = network(image)
            loss = criterion(prediction, label)
            for j, pred in enumerate(prediction, 0):
                real_score = pred[0].item()
                fake_score = pred[1].item()
                if real_score > fake_score and label[j].item() == 0:
                    real_correct += 1
                elif real_score < fake_score and label[j].item() == 0:
                    real_incorrect += 1
                elif real_score > fake_score and label[j].item() == 1:
                    fake_incorrect += 1
                elif real_score < fake_score and label[j].item() == 1:
                    fake_correct += 1
            total_loss += loss.item()
            pb.update(1)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            if i > len(val_loader)*fraction_to_eval:
                break
        if target != None:
            print('Target scores:', network(preprocess(target.to(device))))
            print('Target prediction:', predict_image(network, target, device, processed=False))
            save_target_results(name, network(preprocess(target.to(device))), predict_image(network, target, device, processed=False))

    pb.close()
    total_loss /= len(val_loader)
    save_validation_epoch(name, epoch, total_loss, fake_correct, fake_incorrect, real_correct, real_incorrect)
    print('Finished evaluation:',fake_correct, fake_incorrect, real_correct, real_incorrect, total_loss)
    return fake_correct, fake_incorrect, real_correct, real_incorrect

def eval_network_test(network, device, batch_size=100, name='xception_full_c23_trained_from_scratch', target=None, fraction_to_eval=1, face=False):
    '''Evaluates the network performance on test set.'''
    print('Evaluating network')
    test_dataset = FFDataset('test', face=face)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()

    fake_correct = fake_incorrect = real_correct = real_incorrect = 0

    network.eval()
    pb = tqdm(total=len(test_loader)*fraction_to_eval)
    total_loss = 0.0
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader, 0):
            image, label = image.to(device), label.to(device)
            prediction = network(image)
            loss = criterion(prediction, label)
            for j, pred in enumerate(prediction, 0):
                real_score = pred[0].item()
                fake_score = pred[1].item()
                if real_score > fake_score and label[j].item() == 0:
                    real_correct += 1
                elif real_score < fake_score and label[j].item() == 0:
                    real_incorrect += 1
                elif real_score > fake_score and label[j].item() == 1:
                    fake_incorrect += 1
                elif real_score < fake_score and label[j].item() == 1:
                    fake_correct += 1
            total_loss += loss.item()
            pb.update(1)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            if i > len(test_loader)*fraction_to_eval:
                break
        if target != None:
            print('Target scores:', network(target.to(device)))
            print('Target prediction:', predict_image(network, target, device, processed=False))
    pb.close()
    total_loss /= len(test_loader)
    save_test(name, total_loss, fake_correct, fake_incorrect, real_correct, real_incorrect)
    print('Finished evaluation:',fake_correct, fake_incorrect, real_correct, real_incorrect, total_loss)
    return fake_correct, fake_incorrect, real_correct, real_incorrect

def preprocess(img):
	transform = transforms.Compose([
		transforms.Resize((299, 299)),
		transforms.Normalize([0.5]*3, [0.5]*3)
	])
	return transform(img)

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
