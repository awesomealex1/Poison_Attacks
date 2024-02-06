from data_util import save_network
import torch
from tqdm import tqdm
from datasets import TrainDataset, ValDataset, TestDataset
import torch.nn as nn
from experiment_util import save_training_epoch, save_validation_epoch
import os, platform, subprocess, re

def train_full(network, device, dataset=TrainDataset(), name='xception_full_c23_trained_from_scratch', target=None):
    network = train_on_ff(network, device, dataset, f'{name}_frozen', frozen=True, epochs=3, target=target)
    network = train_on_ff(network, device, dataset, name, frozen=False, epochs=7, target=target)
    return network

def train_transfer(network, device, dataset=TrainDataset(), name='xception_full_transfer_c23', target=None):
    network.apply(randomize_last_layer)
    network = train_on_ff(network, device, dataset, f'{name}_frozen', frozen=True, epochs=3, target=target)
    return network

def train_on_ff(network, device, dataset=TrainDataset(), name='xception_full_c23_trained_from_scratch', frozen=False, epochs=8, lr=0.0002, batch_size=32, target=None):
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
    if frozen:
        network = freeze_all_but_last_layer(network)
    else:
        network = unfreeze_all(network)
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    weight = torch.tensor([4.0, 1.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    best_score = None
    best_network = None
    total_loss = 0.0

    for epoch in range(epochs):
        pb = tqdm(total=len(data_loader))
        for i, (image, label) in enumerate(data_loader, 0):
            optimizer.zero_grad()
            image,label = image.to(device), label.to(device)
            outputs = network(image)
            loss = criterion(outputs, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pb.update(1)
        pb.close()
        total_loss /= len(data_loader)
        save_network(network, f'{name}{epoch}')
        fake_correct, fake_incorrect, real_correct, real_incorrect = eval_network(network, device, name=f'{name}{epoch}', target=target, fraction_to_eval=1)
        save_training_epoch(name, epoch, total_loss, fake_correct, fake_incorrect, real_correct, real_incorrect)
        score = (fake_correct + real_correct)/(fake_correct + fake_incorrect + real_correct + real_incorrect)
        if best_score is None or score > best_score:
            best_score = score
            best_network = f'{name}{epoch}'
    
    print(f'Best network: {best_network}')
    return network

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""

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

def eval_network(network, device, batch_size=100, name='xception_full_c23_trained_from_scratch', target=None, fraction_to_eval=1):
    '''Evaluates the network performance on test set.'''
    print('Evaluating network')
    print('Loading Test Set')
    val_dataset = ValDataset()
    val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()
    print('Finished loading Test Set')

    fake_correct = 0
    fake_incorrect = 0
    real_correct = 0
    real_incorrect = 0

    print('Starting evaluation')
    network.eval()
    pb = tqdm(total=len(val_dataset)*fraction_to_eval)
    total_loss = 0.0
    with torch.no_grad():
        for i, (image, label) in enumerate(val_dataset, 0):
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
            if i > len(val_dataset)*fraction_to_eval:
                break
        if target != None:
            print('Target prediction:', network(target.to(device)))
    pb.close()
    total_loss /= len(val_dataset)
    save_validation_epoch(name, 0, total_loss, fake_correct, fake_incorrect, real_correct, real_incorrect)
    print('Finished evaluation:',fake_correct, fake_incorrect, real_correct, real_incorrect, total_loss)
    return fake_correct, fake_incorrect, real_correct, real_incorrect
