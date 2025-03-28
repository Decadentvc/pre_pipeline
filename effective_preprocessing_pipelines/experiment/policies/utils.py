from .iterative import Iterative
from .split import Split
from .adaptive import Adaptive
from .joint import Joint

def initiate(name, config):
    """Initiates a policy.

    Args:
        name: name of the policy (i.e., iterative, split, adaptive, joint).
        config: extra config. 

    Returns:
        object: the instance of the class policy.
    """
    policies = {
        'iterative': Iterative,
        'split': Split,
        'adaptive': Adaptive,
        'joint': Joint
    }
    if name in policies:
        return policies[name](config)
    else:
        print('Invalid dataset. Possible choices: {}'.format(
            ', '.join(policies.keys())
        ))
        exit(1)  # TODO: Throw exception