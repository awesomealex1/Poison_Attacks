import torch
from tqdm import tqdm
from datasets import BaseDataset, PoisonDataset, TrainDataset, TestDataset, ValDataset, fill_bases_directory
from network.models import get_xception_full, get_xception_untrained, model_selection
import argparse
from data_util import save_network, save_poisons, get_one_fake_ff
from train import train_on_ff

def main(device, create_poison, retrain, create_bases, max_iters, beta_0, lr, evaluate, pretrained, retrain_scratch, preselected_bases, max_base_distance, n_bases, model_path):
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
    elif model_path:
        network = torch.load(model_path, map_location=device)
    else:
        network = get_xception_untrained()
    network.to(device)

    if not pretrained:
        network = train_on_ff(network, device)
        #network = train_on_ff_unfrozen(network, device)
        save_network(network, 'xception_full_c23_trained_from_scratch2_full_jan22')

    feature_space, last_layer = get_feature_space(network)
    target = get_one_fake_ff()
    target = target.to(device)

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

def create_bases(max_base_distance, min_base_score, n_bases, feature_space, target, network, device):
    base_images = []
    train_dataset = TrainDataset()
    target_feature = feature_space(target)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for i, (image, label) in enumerate(data_loader, 0):
        image,label = image.to(device), label.to(device)
        if label.item() == 0:   # If real
            image_features = feature_space(image)
            _, image_score = predict_image(network, image, device)

        outputs = network(image)

def retrain_with_poisons(network, device):
    '''Retrains the network with poisons. Not from scratch, but already trained on FF++.'''
    print('Retraining with poisons')
    poison_dataset = PoisonDataset()
    network = train_on_ff(network, device, dataset=poison_dataset, name='xception_full_c23_trained_from_scratch_poisoned', frozen=False, epochs=1)   
    
    print('Finished retraining with poisons')
    return network

def retrain_with_poisons_scratch(network, device):
    '''Retrains with poisons from scratch (not trained on FF++).'''
    print('Retraining with poisons from scratch')
    poison_dataset = PoisonDataset()
    train_dataset = TrainDataset()
    merged_dataset = torch.utils.data.ConcatDataset([poison_dataset, train_dataset])
    network = train_on_ff(network, device, dataset=merged_dataset, name='xception_full_c23_trained_from_scratch_poisoned', frozen=True)
    print('Finished retraining with poisons')
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

def get_feature_space(network):
    '''Returns the feature space of the network.'''
    layer_cake = list(network.model.children())
    last_layer = layer_cake[-1]
    headless_network = torch.nn.Sequential(*(layer_cake[:-1]), Flatten())
    return headless_network, last_layer

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
    p.add_argument('--min_base_score', type=float, help='Minimum score for base to be classified as', default=0.9)
    p.add_argument('--n_bases', type=int, help='Number of base images to create', default=5)
    p.add_argument('--model_path', type=str, help='Path to model to use for attack', default=None)
    args = p.parse_args()

    use_gpu = not args.cpu

    if use_gpu == True:
        if not torch.cuda.is_available():
            print('GPU not available, falling back to CPU')
            use_gpu = False
    
    device = torch.device('cuda' if use_gpu else 'cpu')

    main(device, args.create_poison, args.retrain, args.create_bases, args.max_iters, args.beta, args.poison_lr, args.evaluate, args.pretrained, args.retrain_scratch, args.preselected_bases, args.max_base_distance, args.n_bases, args.model_path)