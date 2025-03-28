from hyperopt import hp

def generate_domain_space(prototype):
    """generates the domain space of a specific algorithm. (Legacy)

    Returns:
        dict: the domain space.
    """
    domain_space = {}
    for k, v in prototype.items():
        domain_space[k] = hp.choice(k, v)
    return domain_space
