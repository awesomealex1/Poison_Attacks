from data_util import save_network
import torch
from tqdm import tqdm
from datasets import TrainDataset, ValDataset, TestDataset

def train_full(network, device, dataset=TrainDataset(), name='xception_full_c23_trained_from_scratch'):
    network = train_on_ff(network, device, dataset, name, frozen=True)
    network = train_on_ff(network, device, dataset, name, frozen=False, epochs=7)
    return network

def train_on_ff(network, device, dataset=TrainDataset(), name='xception_full_c23_trained_from_scratch', frozen=False, epochs=3, lr=0.0002, batch_size=32):
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
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_score = None
    best_network = None

    for epoch in range(epochs):
        pb = tqdm(total=len(data_loader))
        for i, (image, label) in enumerate(data_loader, 0):
            optimizer.zero_grad()
            image,label = image.to(device), label.to(device)
            outputs = network(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            pb.update(1)
        pb.close()

        save_network(network, f'{name}{epoch}')
        fake_correct, fake_incorrect, real_correct, real_incorrect = eval_network(network, device, file_name=f'{name}{epoch}')
        score = (fake_correct + real_correct)/(fake_correct + fake_incorrect + real_correct + real_incorrect)
        if best_score is None or score > best_score:
            best_score = score
            best_network = f'{name}{epoch}'
    
    print(f'Best network: {best_network}')
    return network

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

def eval_network(network, device, batch_size=100, file_name='results.txt'):
    '''Evaluates the network performance on test set.'''
    print('Evaluating network')
    print('Loading Test Set')
    test_dataset = TestDataset()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    print('Finished loading Test Set')

    fake_correct = 0
    fake_incorrect = 0
    real_correct = 0
    real_incorrect = 0

    print('Starting evaluation')
    results_file = open(file_name, 'w')
    network.eval()
    pb = tqdm(total=len(test_loader))
    total_loss = 0.0
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader, 0):
            image, label = image.to(device), label.to(device)
            prediction = network(image)
            loss = criterion(prediction, label)
            for i, pred in enumerate(prediction, 0):
                real_score = pred[0].item()
                fake_score = pred[1].item()
                results_file.write(f'{real_score} {fake_score} {label[i].item()} \n')
                if real_score > fake_score and label[i].item() == 0:
                    real_correct += 1
                elif real_score < fake_score and label[i].item() == 0:
                    real_incorrect += 1
                elif real_score > fake_score and label[i].item() == 1:
                    fake_incorrect += 1
                elif real_score < fake_score and label[i].item() == 1:
                    fake_correct += 1
            total_loss += loss.item()
            pb.update(1)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    pb.close()
    results_file.write(f'{real_correct} {fake_correct} {real_incorrect} {fake_incorrect} {total_loss}')
    results_file.close()

    print('Finished evaluation:',fake_correct, fake_incorrect, real_correct, real_incorrect, total_loss)
    return fake_correct, fake_incorrect, real_correct, real_incorrect
