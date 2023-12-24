import feature_collision

def run_attack(network, target, base, attack, iterations):
    assert network and target and base and attack, \
    "Didn't receive all parameters to perform attack"

    if attack == "FC":
        feature_collision.feature_collision(network, target, base, iterations)
