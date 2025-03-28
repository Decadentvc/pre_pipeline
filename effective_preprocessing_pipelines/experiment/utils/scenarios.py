import yaml


def load(path):
    """Load a scenario from a specified path.

    Args:
        path: from where the scenario has to be loaded.

    Returns:
        the scenario. 
    """
    scenario = None
    with open(path, "r") as f:
        try:
            scenario = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            scenario = None
    if scenario is not None:
        scenario["file_name"] = path.split("/")[-1].split(".")[0]
    return scenario


def validate(scenario):
    """Check if a scenario is valid.

    Args:
        scenario: a loaded scenario.

    Returns:
        bool: whether a scenario is valid or not.
    """
    return True  #  TODO


def to_config(scenario, args):
    """Convert the scenario to a dictionary (config) and enrich it with several arguments.

    Args:
        scenario: scenario to convert.
        args: arguments to include in the output.

    Returns:
        dict: the converted scenario.
    """
    config = {
        "seed": scenario["control"]["seed"],
        "time": scenario["setup"]["runtime"],
        "algorithm": scenario["setup"]["algorithm"],
        "experiment": args.experiment,
        "mode": args.mode,
        "toy": args.toy_example,
    }
    if scenario["policy"] is not None:
        config.update(scenario["policy"])
    return config
