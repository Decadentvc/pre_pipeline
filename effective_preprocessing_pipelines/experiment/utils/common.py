import argparse
import os
import openml
import json

# Common paths
RESOURCES_PATH = os.path.join("./", "resources")
SCENARIO_PATH = os.path.join(RESOURCES_PATH, "scenarios")
RAW_RESULT_PATH = os.path.join(RESOURCES_PATH, "raw_results")
ARTIFACTS_PATH = os.path.join(RESOURCES_PATH, "artifacts")
META_FEATURES_PATH = os.path.join(RESOURCES_PATH, "meta_features")
DATASETS_PATH = os.path.join(RESOURCES_PATH, "datasets")


algorithms = ['RandomForest', 'NaiveBayes', 'KNearestNeighbors']

# Dataset used to verify the impact of pre-processing
pipeline_impact_suite = [1461]

def parse_args():
    """Parse the arguments given via CLI.

    Returns:
        dict: arguments and their values.
    """
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=False, help="step of the pipeline to execute")
    parser.add_argument("-exp", "--experiment", nargs="?", type=str, required=False, help="type of the experiments")
    parser.add_argument("-mode", "--mode", nargs="?", type=str, required=False, help="algorithm or algorithm_pipeline")
    parser.add_argument("-toy", "--toy_example", action='store_true', default=False, help="wether it is a toy example or not")
    parser.add_argument("-cache", "--cache", action='store_true', default=False, help="wether to use the intermediate results or not")
    args = parser.parse_args()
    return args

def create_directory(result_path, directory):
    """Create a directory in the specified path.

    Args:
        result_path: where to create a directory.
        directory: name of the directory.

    Returns:
        os.path: the resulting path.
    """
    result_path = os.path.join(result_path, directory)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path

def get_filtered_datasets(experiment, toy):
    """Retrieve the dataset list for a certain experiment.

    Args:
        experiment: keyword of the experiment.
        toy: whether it is the toy example or not.

    Returns:
        list: list of OpenML ids.
    """
    if experiment == "pipeline_impact":
        return pipeline_impact_suite
    else:
        import pandas as pd

        with open(os.path.join(RESOURCES_PATH, "ext.json")) as f:
            data = json.load(f)

        df = pd.read_csv(os.path.join(META_FEATURES_PATH, "simple-meta-features.csv"))
        df = df.loc[df['did'].isin(list(dict.fromkeys(data["default"])))]
        df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
        df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
        df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
        if toy:
            df = df.loc[df['NumberOfInstances'] <= 2000]
            df = df.loc[df['NumberOfFeatures'] <= 10]
            df = df.sort_values(by=['NumberOfInstances', 'NumberOfFeatures'])
            df = df[:10]
        df = df['did']
        for dataset_id in data["extra"]:
            if not os.path.exists(os.path.join(DATASETS_PATH, f"{dataset_id}.csv")):
                dataset = openml.datasets.get_dataset(dataset_id)
                X, y, categorical_indicator, _ = dataset.get_data(
                    dataset_format="array", target=dataset.default_target_attribute
                )
                X, y = pd.DataFrame(X), pd.DataFrame(y)
                df_merged=pd.concat([X, y],axis=1)
                df_merged.columns = list(range(len(df_merged.columns)))
                df_merged.to_csv(os.path.join(DATASETS_PATH, f"{dataset_id}.csv"), index=False)
                with open(os.path.join(DATASETS_PATH, "categorical_indicators.json")) as f:
                    categorical_indicator_json = json.load(f)
                categorical_indicator_json[dataset_id] = categorical_indicator
                with open(os.path.join(DATASETS_PATH, "categorical_indicators.json"), "w") as outfile:
                    json.dump(categorical_indicator_json, outfile)
        return df.values.flatten().tolist() + data["extra"]