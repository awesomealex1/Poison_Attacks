import torch

def feature_collision(network, target, base, iterations):
    poisons = []
    return poisons

def bypass_last_layer(network):
    """
    Hacky way of separating features and classification head for many networks.
    Patch this function if problems appear.
    """

    layer_cake = list(network.children())
    last_layer = layer_cake[-1]
    headless_network = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())  # this works most of the time all of the time :<
    return headless_network, last_layer