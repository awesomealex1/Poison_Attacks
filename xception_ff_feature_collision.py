from data import data_util
import torch
from tqdm import tqdm
from torchvision.utils import save_image
import os
from data.datasets import BaseDataset, PoisonDataset, TrainDataset, TestDataset, ValDataset, fill_bases_directory
import json

def main():
    print('Starting poison attack')

    create_bases = False                    # Whether we need to populate the data/bases directory
    max_iters = 200                         # Maximum number of iterations to create one poison
    beta_0 = 0.25                           # beta 0 from poison frogs paper
    beta = beta_0 * 2048**2/(299*299)**2    # base_instance_dim = 299*299 and feature_dim = 2048
    lr = 0.001                              # Learning rate for poison creation

    if create_bases:
        fill_bases_directory()

    # Prepare network and data
    network = get_xception()
    feature_space, last_layer = get_feature_space(network)
    target = data_util.get_one_fake_ff()

    # Perform feature collison attack and create poisons
    #poisons = feature_coll(feature_space, target, max_iters, beta, lr, network)
    #save_poisons(poisons)
    print('Before:',predict_image(network, target))
    eval_network(network)                               # Evaluate network before retraining
    poisoned_network = retrain_with_poisons(network)    # Retrain network with poisons
    print('After:',predict_image(poisoned_network, target))
    save_network(poisoned_network, 'xception_face_detection_c23_poisoned')
    #eval_poisons(network, poisons)                      # Evaluate poisons (how they are classified by the network)
    eval_network(network)                               # Evaluate network after retraining

def save_poisons(poisons):
    print('Saving poisons')
    os.makedirs('data/poisons', exist_ok=True)
    for i, poison in enumerate(poisons):
        save_image(poison[0], f'data/poisons/poison_{i}.png')
    print('Finished saving poisons')

def save_network(network, name):
    torch.save(network, f'network/weights/{name}.p')
    print(f'Saved network as {name}')

def retrain_with_poisons(network):
    print('Retraining with poisons')

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 1
    batch_size = 1
    poison_dataset = PoisonDataset()
    poison_loader = torch.utils.data.DataLoader(poison_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        pb = tqdm(total=len(poison_loader))
        for i, (image, label) in enumerate(poison_loader, 0):
            optimizer.zero_grad()
            outputs = network(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            pb.update(1)
        pb.close()
    
    print('Finished retraining with poisons')
    return network

def eval_network(network, images_per_video=1, batch_size=100):
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
            prediction = network(image.cuda())
            for i, pred in enumerate(prediction, 0):
                real_score = pred[0].item()
                fake_score = pred[1].item()
                results_file.write(f'{real_score} {fake_score} {label[i].item()} \n')
            if i % 100000 == 0:
                print(prediction)
                print(label)
            pb.update(1)
            torch.cuda.empty_cache()
    pb.close()
    results_file.close()

    print('Finished evaluation:',fake_correct, fake_incorrect, real_correct, real_incorrect)
    return fake_correct, fake_incorrect, real_correct, real_incorrect

def predict_image(network, image):
    post_function = torch.nn.Softmax(dim = 1)
    output = network(image.cuda())
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    cpu = True
    if cpu:
        prediction = float(prediction.cpu().numpy())
    else:
        prediction = float(prediction.numpy())

    return int(prediction), output  # If prediction is 1, then fake, else real

def eval_poisons(network, poisons):
    pass

def feature_coll(feature_space, target, max_iters, beta, lr, network):
    poisons = []
    base_dataset = BaseDataset()
    base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=1, shuffle=False)
    for i, (base,label) in enumerate(base_loader, 1):
        poison = single_poison(feature_space, target, base, max_iters, beta, lr, network)
        poisons.append(poison)
        print(f'Poison {i}/{len(base_dataset)} created')
    return poisons

def single_poison(feature_space, target, base, max_iters, beta, lr, network, decay_coef=0.9, M=20):
    x = base
    prev_x = base
    prev_M_objectives = []
    pbar = tqdm(total=max_iters)
    for i in range(max_iters):
        x = forward_backward(feature_space, target, base, x, beta, lr)
        print(predict_image(network, x))
        target_space = feature_space(target)
        x_space = feature_space(x)
        print(torch.norm(x_space - target_space))

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
    x_hat = forward(feature_space, target, x, lr)
    new_x = backward(base, x_hat, beta, lr)
    return new_x

def forward(feature_space, target, x, lr):
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
    return (x_hat + lr * beta * base) / (1 + beta * lr)

def get_xception():
    model_path = 'network/weights/xception_full_c23.p'
    cpu = False
    if cpu:
        model = torch.load(model_path, map_location='cpu')
    else:
        model = torch.load(model_path)
        model.cuda()
    return model

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def get_feature_space(network):
    layer_cake = list(network.model.children())
    last_layer = layer_cake[-1]
    headless_network = torch.nn.Sequential(*(layer_cake[:-1]), Flatten())
    return headless_network, last_layer

if __name__ == "__main__":
    main()