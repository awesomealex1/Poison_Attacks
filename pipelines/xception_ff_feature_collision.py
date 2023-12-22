from attacks import feature_collision, attack_util
from data import data_util
from networks.xception import get_xception

def __main__(self):
    network = get_xception()
    attack = feature_collision
    target = data_util.get_one_fake_ff()
    base = data_util.get_one_real_ff()

    poisons = attack_util.run_attack(network, target, base, attack)

    eval_network(network)

    retrain_with_poisons(network, poisons)

    eval_poisons(network, poisons)
    eval_network(network)

def retrain_with_poisons(network, poisons):
    pass

def eval_network(network):
    pass

def eval_poisons(network, poisons):
    pass