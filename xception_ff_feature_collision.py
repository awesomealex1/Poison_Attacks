from data import data_util
import torch
from tqdm import tqdm
from torchvision.utils import save_image
import os
from data.datasets import BaseDataset, PoisonDataset, TrainDataset, TestDataset, ValDataset, fill_bases_directory
import json
from network.models import model_selection
import argparse

def main(device, create_poison, retrain, create_bases, max_iters, beta_0, lr, evaluate, pretrained, retrain_scratch, preselected_bases, max_base_distance, n_bases):
    '''
    Main function to run a poisoning attack on the Xception network.
    Args:
        device: cuda or cpu
        create_poison: Whether to create poisons
        retrain: Whether to retrain the network with poisons
        create_bases: Whether to populate the data/bases directory
        max_iters: Maximum number of iterations to create one poison
        beta_0: beta 0 from poison frogs paper
        lr: Learning rate for poison creation
        evaluate: Whether to evaluate the network (takes a long time)
        pretrained: Whether to use FF++ provided pretrained network
        retrain_scratch: Whether to retrain from scratch
        preselected_bases: Whether to use a txt file with base images
        max_base_distance: Maximum distance between base and target (when not having preselected bases)
        n_bases: Number of base images to create (when not having preselected bases)
    Does not return anything but will create files with data and prints results.
    '''
    print('Starting poison attack')
    beta = beta_0 * 2048**2/(299*299)**2    # base_instance_dim = 299*299 and feature_dim = 2048

    if create_bases:
        fill_bases_directory()

    if pretrained:
        network = get_xception_full(device)
    else:
        network = get_xception_untrained()
    network.to(device)

    if not pretrained:
        network = train_on_ff(network, device)
        save_network(network, 'xception_full_c23_trained_from_scratch')

    feature_space, last_layer = get_feature_space(network)
    target = data_util.get_one_fake_ff()
    target = target.to(device)

    print(create_poison)

    if create_poison:
        if not preselected_bases:
            create_bases(max_base_distance, n_bases)
        poisons = feature_coll(feature_space, target, max_iters, beta, lr, network, device)
        save_poisons(poisons)
    
    if evaluate:
        eval_network(network, device)
    
    print(f'Original target prediction: {predict_image(network, target, device)}')

    if retrain_scratch:
        network = get_xception_untrained()
    
    if retrain_scratch:
        poisoned_network = retrain_with_poisons_scratch(network, device)
        print(f'Target prediction after retraining from scratch: {predict_image(poisoned_network, target, device)}')
    elif retrain:
        poisoned_network = retrain_with_poisons(network, device)
        print(f'Target prediction after retraining: {predict_image(poisoned_network, target, device)}')
    
    save_network(poisoned_network, 'xception_full_c23_poisoned')
    eval_network(network, device)

def save_poisons(poisons):
    '''
    Saves poisons to data/poisons directory.
    Args:
        poisons: List of images
    '''
    print('Saving poisons')
    os.makedirs('data/poisons', exist_ok=True)
    for i, poison in enumerate(poisons):
        save_image(poison[0], f'data/poisons/poison_{i}.png')
    print('Finished saving poisons')

def save_network(network, name):
    '''
    Saves network to network/weights directory.
    Args:
        network: Network to save
        name: Name to save as
    '''
    torch.save(network, f'network/weights/{name}.p')
    print(f'Saved network as {name}')

def create_bases(max_base_distance, n_bases):
    pass

def train_on_ff(network, device):
    '''
    Trains the network on FF++ dataset.
    Args:
        network: Network to train
    Returns:
        network: Trained network
    '''
    print('Training on FF++')
    network = freeze_all_but_last_layer(network)
    network.train()
    network = torch.nn.DataParallel(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    weight = torch.tensor([4, 1])
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    epochs = 1
    batch_size = 128
    train_dataset = TrainDataset()
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
    
    print('Finished training on FF++')
    return network

def retrain_with_poisons(network, device):
    '''
    Retrains the network with poisons. Not from scratch, but already trained on FF++.
    Args:
        network: Retrained network
    '''
    print('Retraining with poisons')
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    weight = torch.tensor([4, 1])
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    epochs = 1
    batch_size = 1
    poison_dataset = PoisonDataset()
    poison_loader = torch.utils.data.DataLoader(poison_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        pb = tqdm(total=len(poison_loader))
        for i, (image, label) in enumerate(poison_loader, 0):
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            outputs = network(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            pb.update(1)
        pb.close()
    
    print('Finished retraining with poisons')
    return network

def retrain_with_poisons_scratch(network, device):
    '''
    Retrains with poisons from scratch (not trained on FF++).
    Args:
        network: Retrained network
    '''
    print('Retraining with poisons from scratch')
    network = freeze_all_but_last_layer(network)
    network.train()
    network = torch.nn.DataParallel(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    weight = torch.tensor([4, 1])
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    epochs = 1
    batch_size = 128
    poison_dataset = PoisonDataset()
    train_dataset = TrainDataset()
    merged_dataset = torch.utils.data.ConcatDataset([poison_dataset, train_dataset])
    data_loader = torch.utils.data.DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)

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
    
    print('Finished retraining with poisons')
    return network

def eval_network(network, device, batch_size=100):
    '''
    Evaluates the network performance on test set.
    Args:
        network: Network to evaluate
        batch_size: Batch size for evaluation
    Returns:
        fake_correct: Number of fake images correctly classified
        fake_incorrect: Number of fake images incorrectly classified
        real_correct: Number of real images correctly classified
        real_incorrect: Number of real images incorrectly classified
    Also writes results to results.txt
    '''
    print('Evaluating network')
    print('Loading Test Set')
    test_dataset = TestDataset()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print('Finished loading Test Set')

    fake_correct = 0
    fake_incorrect = 0
    real_correct = 0
    real_incorrect = 0

    print('Starting evaluation')
    results_file = open('results.txt', 'w')
    network.eval()
    pb = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader, 0):
            image, label = image.to(device), label.to(device)
            prediction = network(image)
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
            pb.update(1)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    pb.close()
    results_file.write(f'{real_correct} {fake_correct} {real_incorrect} {fake_incorrect}')
    results_file.close()

    print('Finished evaluation:',fake_correct, fake_incorrect, real_correct, real_incorrect)
    return fake_correct, fake_incorrect, real_correct, real_incorrect

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
    post_function = torch.nn.Softmax(dim = 1)
    image = image.to(device)
    output = network(image)
    output = post_function(output)
    _, prediction = torch.max(output, 1)    # argmax

    return int(prediction.item()), output  # If prediction is 1, then fake, else real

def feature_coll(feature_space, target, max_iters, beta, lr, network, device):
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
    base_dataset = BaseDataset()
    base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=1, shuffle=False)
    for i, (base,label) in enumerate(base_loader, 1):
        base, label = base.to(device), label.to(device)
        poison = single_poison(feature_space, target, base, max_iters, beta, lr, network, device)
        poisons.append(poison)
        print(f'Poison {i}/{len(base_dataset)} created')
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
        print(f'Poison prediction: {predict_image(network, x, device)}')
        target_space = feature_space(target)
        x_space = feature_space(x)
        print(f'Poison-target distance: {torch.norm(x_space - target_space)}')

        new_obj = torch.norm(x_space - target_space) + beta*torch.norm(x - base)
        avg_of_last_M = sum(prev_M_objectives)/float(min(M, i+1))

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
    '''
    Performs forward and backward passes.
    Args:
        feature_space: Feature space of network
        target: Target image
        base: Base image
        x: Current poison image
        beta: Beta value for feature collision attack
        lr: Learning rate for poison creation
    Returns:
        new_x: New poison image
    '''
    x_hat = forward(feature_space, target, x, lr)
    new_x = backward(base, x_hat, beta, lr)
    return new_x

def forward(feature_space, target, x, lr):
    '''
    Performs forward pass.
    Args:
        feature_space: Feature space of network
        target: Target image
        x: Current poison image
        lr: Learning rate for poison creation
    Returns:
        x_hat: New poison image
    '''
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
    '''
    Performs backward pass.
    Args:
        base: Base image
        x_hat: New poison image
        beta: Beta value for feature collision attack
        lr: Learning rate for poison creation
    Returns:
        new_x: New poison image
    '''
    return (x_hat + lr * beta * base) / (1 + beta * lr)

def get_xception_full(device):
    '''
    Returns the pretrained full image xception network.
    Returns:
        model: Pretrained full image xception network
    '''
    model_path = 'network/weights/xception_full_c23.p'
    model = torch.load(model_path, map_location=device)
    return model

def get_xception_untrained():
    '''
    Returns an untrained xception network.
    Returns:
        network: Untrained xception network
    '''
    network = model_selection('xception', num_out_classes=2)[0]
    return network

class Flatten(torch.nn.Module):
    '''Layer used for flattening the network.'''
    def forward(self, input):
        return input.view(input.size(0), -1)

def get_feature_space(network):
    '''
    Returns the feature space of the network.
    Args:
        network: Network to get feature space of
    Returns:
        headless_network: Network without last layer
        last_layer: Last layer of network
    '''
    layer_cake = list(network.model.children())
    last_layer = layer_cake[-1]
    headless_network = torch.nn.Sequential(*(layer_cake[:-1]), Flatten())
    return headless_network, last_layer

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

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--create_poison', action='store_true', help='Whether attack should be performed to create poison samples')
    p.add_argument('--cpu', action='store_true', help='Whether to use cpu')
    p.add_argument('--retrain', action='store_true', help='Whether to retrain the network with poisons')
    p.add_argument('--evaluate', action='store_true', help='Whether to evaluate network')
    p.add_argument('--beta', type=float, help='Beta 0 value for feature collision attack', default=0.25)
    p.add_argument('--max_iters', type=int, help='Maximum iterations for poison creation', default=200)
    p.add_argument('--poison_lr', type=float, help='Learning rate for poison creation', default=0.001)
    p.add_argument('--create_bases', action='store_true', help='Whether to populate the data/bases directory')
    p.add_argument('--pretrained', action='store_true', help='Whether to use FF++ provided pretrained network')
    p.add_argument('--retrain_scratch', action='store_true', help='Whether to retrain from scratch')
    p.add_argument('--preselected_bases', action='store_true', help='Whether to use a txt file with base images')
    p.add_argument('--max_base_distance', type=float, help='Maximum distance between base and target', default=500)
    p.add_argument('--n_bases', type=int, help='Number of base images to create', default=5)
    args = p.parse_args()

    use_gpu = not args.cpu

    if use_gpu == True:
        if not torch.cuda.is_available():
            print('GPU not available, falling back to CPU')
            use_gpu = False
    
    device = torch.device('cuda' if use_gpu else 'cpu')

    main(device, args.create_poison, args.retrain, args.create_bases, args.max_iters, args.beta, args.poison_lr, args.evaluate, args.pretrained, args.retrain_scratch, args.preselected_bases, args.max_base_distance, args.n_bases)