import argparse
from functools import reduce
import operator

from .scenarios import load


def parse_args():
    """Parse the arguments given via CLI.

    Returns:
        dict: arguments and their values.
    """

    parser = argparse.ArgumentParser(
        description="""
        Automated Machine Learning Workflow creation and configuration
        """
    )

    # I need to force argparse to do what was descriped in spec

    parser.add_argument(
        "-s",
        "--scenario",
        nargs="?",
        type=str,
        required=True,
        help="path to the scenario to execute",
    )

    parser.add_argument(
        "-p",
        "--pipeline",
        nargs="+",
        type=str,
        required=True,
        help="step of the pipeline to execute",
    )

    parser.add_argument(
        "-r",
        "--result_path",
        nargs="?",
        type=str,
        required=True,
        help="path where put the raw results",
    )

    parser.add_argument(
        "-exp",
        "--experiment",
        nargs="?",
        type=str,
        required=True,
        help="type of the experiments",
    )

    parser.add_argument(
        "-m",
        "--mode",
        nargs="?",
        type=str,
        required=False,
        help="algorithm or pipeline_algorithm",
    )

    parser.add_argument(
        "-np",
        "--num_pipelines",
        nargs="?",
        type=int,
        required=False,
        help="number of pipelines to split the budget",
    )

    parser.add_argument(
        "-toy",
        "--toy_example",
        nargs="?",
        type=bool,
        required=False,
        help="wether to use the toy example or not",
    )

    parser.add_argument(
        "-c",
        "--customize",
        nargs="+",
        help="Customize scenario by overwriting specific variables",
        required=False,
    )

    args = parser.parse_args()
    customs = args.customize
    parsed_customs = []

    if customs is not None:
        scenario = load(args.scenario)
        for field in customs:
            value = field.split("=")[-1]
            path = field.split("=")[0].split(".")
            c = scenario
            for p in path:
                if p not in c:
                    print("Could not find path: {}".format(".".join(path)))
                    exit(3)
                c = c[p]
            parsed_customs.append((path, value))
    else:
        parsed_customs = []

    args.customize = parsed_customs
    return args


def apply_scenario_customization(scenario, customs):
    """Enrich the scenario with supplementary fields.

    Args:
        scenario: scenario to customize.
        customs: supplementary key-values.
    """
    def get_from_dict(data, keys):
        return reduce(operator.getitem, keys, data)

    def set_in_dict(data, keys, value):
        get_from_dict(data, keys[:-1])[keys[-1]] = value

    for field in customs:
        value = field[1]
        try:
            value = int(field[1])
        except Exception as e:
            print(e)
            try:
                value = float(field[1])
            except:
                pass
        set_in_dict(scenario, field[0], value)
    return scenario
