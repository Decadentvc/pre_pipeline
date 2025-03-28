from sklearn.impute import SimpleImputer

from pipeline.PrototypeSingleton import PrototypeSingleton
from utils import scenarios, serializer, cli, datasets
from policies import initiate

import json


def load_dataset(id, args):
    """Load a dataset from its id.

    Args:
        id: id from OpenML.
        args: taken from utils.cli.parse_args.

    Returns:
        X, y are respectivelly the data items and the labels.
    """
    # Load the data from the csv
    X, y, categorical_indicator = datasets.load_from_csv(id)

    # Impute missing values if needed
    if args.experiment == "prototype_construction" or (
        args.experiment == "custom_prototypes" and args.mode == "algorithm"
    ):
        X = SimpleImputer(strategy="constant").fit_transform(X)
    print(f"dataset id: {id}")
    print(f"X:\n{X}")
    print(f"y:\n{y}")

    # Set all the dataset characteristics on a singleton
    PrototypeSingleton.getInstance().setPipeline(args.pipeline)
    num_features = [i for i, x in enumerate(categorical_indicator) if x == False]
    cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
    print(
        "numeriche: "
        + str(len(num_features))
        + " categoriche: "
        + str(len(cat_features))
    )
    PrototypeSingleton.getInstance().setFeatures(num_features, cat_features)
    PrototypeSingleton.getInstance().set_X_y(X, y)
    return X, y


def main(args):
    """Launch the optimization experiments according to the specified parameters.

    Args:
        args: taken from utils.cli.parse_args.
    """

    # Get scenario and config
    scenario = scenarios.load(args.scenario)
    scenario = cli.apply_scenario_customization(scenario, args.customize)
    config = scenarios.to_config(scenario, args)

    # Adapt the config according to the experiment
    if args.experiment == "custom_prototypes":
        if args.mode == "pipeline_algorithm":
            config["time"] /= args.num_pipelines
            config["step_pipeline"] /= args.num_pipelines
        else:
            if args.num_pipelines == 0:
                config["time"] = 10 if args.toy_example else 400
            else:
                config["time"] = 6 if args.toy_example else 240
                config["step_pipeline"] = 1 if args.toy_example else 40

    print("SCENARIO:\n {}".format(json.dumps(scenario, indent=4, sort_keys=True)))

    try:
        # Load the dataset
        X, y = load_dataset(scenario["setup"]["dataset"], args)
        # Run the optimization
        policy = initiate(scenario["setup"]["policy"], config)
        policy.run(X, y)
    except Exception as e:
        print(e)
        policy = None
    finally:
        # Save the result
        serializer.serialize_results(
            scenario=scenario,
            result_path=args.result_path,
            policy=policy,
            pipeline=args.pipeline,
        )


if __name__ == "__main__":
    args = cli.parse_args()
    main(args)
