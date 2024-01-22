from data_util import save_network
import torch
from tqdm import tqdm
from datasets import TrainDataset, ValDataset, TestDataset
from xception_ff_feature_collision import eval_network

def train_full(network, device, dataset=TrainDataset(), name='xception_full_c23_trained_from_scratch'):
    network = train_on_ff(network, device, dataset, name, frozen=True)
    network = train_on_ff(network, device, dataset, name, frozen=False)
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
        eval_network(network, device, file_name=f'{name}{epoch}')
    return network

def freeze_all_but_last_layer(network):
    '''
    Freezes all but the last layer of the network.
    Args:
        network: Network to freeze
    Returns:
        network: Network with all but last layer frozen
    '''
    layer_cake = list(network.model.children())
    for layer in layer_cake[:-1]:
        for param in layer.parameters():
            param.requires_grad = False
    return network

def unfreeze_all(network):
    layer_cake = list(network.model.children())
    for layer in layer_cake:
        for param in layer.parameters():
            param.requires_grad = True
    return network