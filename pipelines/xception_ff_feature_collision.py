from attacks import feature_collision, attack_util
from data import data_util
from networks import xception

def __main__(self):
    network = xception
    attack = feature_collision
    target = data_util.get_one_fake_ff()
    base = data_util.get_one_real_ff()

    attack_util.run_attack(network, target, base, attack)
