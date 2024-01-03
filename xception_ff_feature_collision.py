from data import data_util
import torch
from transform import xception_default_data_transforms
from PIL import Image as pil_image
import cv2
from tqdm import tqdm

def main():
    print('Starting poison attack')
    n_poisons = 10       # Number of poisons to create
    max_iters = 100      # Maximum number of iterations to create one poison
    beta = 0.9           # Beta parameter for poison creation
    lr = 500.0*255       # Learning rate for poison creation

    network = get_xception()
    feature_space, last_layer = get_feature_space(network)
    target = data_util.get_one_fake_ff()
    base = data_util.get_one_real_ff()
    
    preprocess = xception_default_data_transforms['test']

    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    
    target = preprocess(pil_image.fromarray(target))
    base = preprocess(pil_image.fromarray(base))

    target = target.unsqueeze(0)
    base = base.unsqueeze(0)


    print(target)
    print(base)

    poisons = feature_coll(feature_space, target, base, n_poisons, max_iters, beta, lr)

    eval_network(network)

    retrain_with_poisons(network, poisons)

    eval_poisons(network, poisons)
    eval_network(network)

def retrain_with_poisons(network, poisons):
    pass

def eval_network(network):
    images = []
    labels = []

    fake_correct = 0
    fake_incorrect = 0
    real_correct = 0
    real_incorrect = 0

    for image, label in zip(images, labels):
        score = predict_image(image)
        if label == 'real' and score == 'real':
            real_correct += 1
        elif label == 'real' and score == 'fake':
            real_incorrect += 1
        elif label == 'fake' and score == 'fake':
            fake_correct += 1
        elif label == 'fake' and score == 'real':
            fake_incorrect += 1
        else:
            print("Evaluation mistake",label,score)
    
    return fake_correct, fake_incorrect, real_correct, real_incorrect

def predict_image(network, image):
    return network(image)

def eval_poisons(network, poisons):
    pass

def feature_coll(network, target, base, n_poisons, max_iters, beta, lr):
    poisons = []
    for i in range(n_poisons):
        poison = single_poison(network, target, base, max_iters, beta, lr)
        poisons.append(poison)

def single_poison(network, target, base, max_iters, beta, lr):
    x = base
    pbar = tqdm(total=max_iters)
    for i in range(max_iters):
        x = forward_backward(network, target, base, x, beta, lr)
        pbar.update(1)
    return x

def forward_backward(network, target, base, x, beta, lr):
    x_hat = forward(network, target, base, x, lr)
    new_x = backward(base, x_hat, beta, lr)
    return new_x

def forward(network, target, base, x, lr):
    target_space = network(target)
    x_space = network(x)
    distance = torch.norm(x_space - target_space)
    x_hat = x - lr * distance
    return x_hat

def backward(base, x_hat, beta, lr):
    return (x_hat + lr * beta * base) / (1 + beta * lr)

def get_xception():
    model_path = 'network/weights/xception_face_detection_c23.p'
    cpu = True
    if cpu:
        model = torch.load(model_path, map_location='cpu')
    else:
        model = torch.load(model_path)
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