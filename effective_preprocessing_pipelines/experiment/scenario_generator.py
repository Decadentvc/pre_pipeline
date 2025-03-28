from utils.common import *
from collections import OrderedDict
import os
import copy
import warnings

warnings.filterwarnings("ignore")


# Base scenario, containing the common elements
base = OrderedDict(
    [
        ("title", ""),
        (
            "setup",
            {
                "policy": "split",
            },
        ),
        ("control", {"seed": 42}),
    ]
)


def __write_scenario(path, scenario):
    """Util that writes a scneario in a given path.

    Args:
        path: where to write.
        scenario: what to write.
    """
    try:
        # print("   -> {}".format(path))
        with open(path, "w") as f:
            for k, v in scenario.items():
                if isinstance(v, str):
                    f.write("{}: {}\n".format(k, v))
                else:
                    f.write("{}:\n".format(k))
                    for i, j in v.items():
                        f.write("  {}: {}\n".format(i, j))
    except Exception as e:
        print(e)


def generate_scenarios(args):
    """Creates the scenarios according to the given parameters.

    Args:
        args: taken from utils.common.parse_args.
    """

    # Get scenario and result path
    if args.toy_example == True:
        scenario_path = create_directory(SCENARIO_PATH, "toy")
    elif args.cache == True:
        scenario_path = create_directory(SCENARIO_PATH, "paper")
    else:
        scenario_path = create_directory(SCENARIO_PATH, "paper_new")
    scenario_path = create_directory(scenario_path, args.experiment)

    # Get scenarios according to the experiment at hand
    datasets = get_filtered_datasets(args.experiment, args.toy_example)

    # Set scenario specification
    for dataset in datasets:
        for algorithm in algorithms:
            # Get the base scenario
            scenario = copy.deepcopy(base)
            scenario["title"] = "{} on dataset n. {} with Split policy".format(
                algorithm, dataset
            )
            # Set the runtime and step_pipeline according to wheter
            # it is the toy example or not
            if args.toy_example:
                runtime = 10
                step_pipeline = 5
            else:
                # Set the runtime and step_pipeline according to
                # wheter it is the preprocessing impact experiment or now
                if args.experiment == "pipeline_impact":
                    runtime = 130
                    step_pipeline = 50
                else:
                    runtime = 400
                    step_pipeline = 400
            scenario["setup"]["runtime"] = runtime
            scenario["setup"]["dataset"] = dataset
            scenario["setup"]["algorithm"] = algorithm
            scenario["policy"] = {"step_pipeline": step_pipeline}

            algorithm_acronym = "".join([c for c in algorithm if c.isupper()]).lower()

            # If it is evaluation
            if args.experiment == "custom_prototypes":
                experiment_steps = ["algorithm", "pipeline_algorithm"]
                # We set the scenario according to the mode (here experiment_step)
                for experiment_step in experiment_steps:
                    if experiment_step == "algorithm":
                        step_pipeline = 0
                    else:
                        if args.toy_example:
                            step_pipeline = 5
                        else:
                            step_pipeline = 75
                    scenario["policy"]["step_pipeline"] = step_pipeline
                    if experiment_step == "pipeline_algorithm":
                        if args.toy_example:
                            scenario["setup"]["runtime"] = 5
                        else:
                            scenario["setup"]["runtime"] = 200
                    path = create_directory(scenario_path, experiment_step)
                    path = os.path.join(
                        path, "{}_{}.yaml".format(algorithm_acronym, dataset)
                    )
                    __write_scenario(path, scenario)
            else:
                if args.experiment == "exhaustive_prototypes":
                    if args.toy_example:
                        scenario["policy"]["step_pipeline"] = 5
                    else:
                        scenario["policy"]["step_pipeline"] = 200
                path = os.path.join(
                    scenario_path, "{}_{}.yaml".format(algorithm_acronym, dataset)
                )
                __write_scenario(path, scenario)


if __name__ == "__main__":
    args = parse_args()
    generate_scenarios(args)
