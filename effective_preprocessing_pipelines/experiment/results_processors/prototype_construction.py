import collections
import logging
import os
import json
import warnings
import subprocess
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, chi2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from yellowbrick.target import FeatureCorrelation
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from scipy import stats as s
from os import listdir
from os.path import isfile, join
from utils.common import *


def create_possible_categories(pipeline):
    """Creates all the categories of the optimization process outcome.

    Args:
        pipeline: pair of transformation.

    Returns:
        dict: common name in the key, specific name in the value.
    """
    first = pipeline[0][0].upper()
    second = pipeline[1][0].upper()
    first_or_second = first + "o" + second
    first_second = first + second
    second_first = second + first
    draw = first_second + "o" + second_first
    baseline = "baseline"
    inconsistent = "inconsistent"
    not_exec = "not_exec"
    not_exec_once = "not_exec_once"

    return {
        "first": first,
        "second": second,
        "first_or_second": first_or_second,
        "first_second": first_second,
        "second_first": second_first,
        "draw": draw,
        "baseline": baseline,
        "inconsistent": inconsistent,
        "not_exec": not_exec,
        "not_exec_once": not_exec_once,
    }


def merge_dict(list):
    """Merges dictionaries and keep values of common keys in list.

    Args:
        list: list of dictionaries to merge.

    Returns:
        dict: merged dictionary.
    """
    new_dict = {}
    for key, value in list[0].items():
        new_value = []
        for dict in list:
            new_value.append(dict[key])
        new_dict[key] = new_value
    return new_dict


def load_results(input_path, filtered_datasets):
    """Loads the results of the optimization process.

    Args:
        input_path: where to load the results.
        filtered_datasets: OpenML ids of the datasets.

    Returns:
        dict: the loaded results.
    """
    comparison = {}
    confs = [os.path.join(input_path, "conf1"), os.path.join(input_path, "conf2")]
    for path in confs:
        files = [f for f in listdir(path) if isfile(join(path, f))]
        results = [f[:-5] for f in files if f[-4:] == "json"]
        comparison[path] = {}
        for algorithm in algorithms:
            for dataset in filtered_datasets:
                acronym = "".join([a for a in algorithm if a.isupper()]).lower()
                acronym += "_" + str(dataset)
                if acronym in results:
                    try:
                        with open(os.path.join(path, acronym + ".json")) as json_file:
                            data = json.load(json_file)
                            accuracy = (
                                data["context"]["best_config"]["score"] // 0.0001 / 100
                            )
                            pipeline = (
                                str(data["context"]["best_config"]["pipeline"])
                                .replace(" ", "")
                                .replace(",", " ")
                            )
                            num_iterations = data["context"]["iteration"] + 1
                            best_iteration = (
                                data["context"]["best_config"]["iteration"] + 1
                            )
                            baseline_score = (
                                data["context"]["baseline_score"] // 0.0001 / 100
                            )
                    except:
                        accuracy = 0
                        pipeline = ""
                        num_iterations = 0
                        best_iteration = 0
                        baseline_score = 0
                else:
                    accuracy = 0
                    pipeline = ""
                    num_iterations = 0
                    best_iteration = 0
                    baseline_score = 0

                comparison[path][acronym] = {}
                comparison[path][acronym]["accuracy"] = accuracy
                comparison[path][acronym]["baseline_score"] = baseline_score
                comparison[path][acronym]["num_iterations"] = num_iterations
                comparison[path][acronym]["best_iteration"] = best_iteration
                comparison[path][acronym]["pipeline"] = pipeline

    return dict(
        collections.OrderedDict(
            sorted(merge_dict([comparison[confs[0]], comparison[confs[1]]]).items())
        )
    )


def save_simple_results(result_path, simple_results, filtered_datasets):
    """Saves the results of the optimization process.

    Args:
        result_path: where to save the results.
        simple_results: the resuts to save.
        filtered_datasets: OpenML ids of the datasets.
    """

    def values_to_string(values):
        return [str(value).replace(",", "") for value in values]

    for algorithm in algorithms:
        acronym = "".join([a for a in algorithm if a.isupper()]).lower()
        if os.path.exists("{}.csv".format(acronym)):
            os.remove("{}.csv".format(acronym))
        with open(os.path.join(result_path, "{}.csv".format(acronym)), "w") as out:
            first_element = simple_results[list(simple_results.keys())[0]]
            conf_keys = first_element["conf1"].keys()
            conf1_header = ",".join([a + "1" for a in conf_keys])
            conf2_header = ",".join([a + "2" for a in conf_keys])
            result_header = ",".join(first_element["result"].keys())
            header = ",".join([result_header, conf1_header, conf2_header])
            out.write("dataset,name,dimensions," + header + "\n")

    df = pd.read_csv(os.path.join(META_FEATURES_PATH, "simple-meta-features.csv"))
    df = df.loc[df["did"].isin(filtered_datasets)]

    for key, value in simple_results.items():
        acronym = key.split("_")[0]
        data_set = key.split("_")[1]
        name = df.loc[df["did"] == int(data_set)]["name"].values.tolist()[0]
        dimensions = " x ".join(
            [
                str(int(a))
                for a in df.loc[df["did"] == int(data_set)][
                    ["NumberOfInstances", "NumberOfFeatures"]
                ]
                .values.flatten()
                .tolist()
            ]
        )

        with open(os.path.join(result_path, "{}.csv".format(acronym)), "a") as out:
            results = ",".join(values_to_string(value["result"].values()))
            conf1 = ",".join(values_to_string(value["conf1"].values()))
            conf2 = ",".join(values_to_string(value["conf2"].values()))
            row = ",".join([data_set, name, dimensions, results, conf1, conf2])
            out.write(row + "\n")


def compose_pipeline(pipeline1, pipeline2, scheme):
    """Creates pipelines and aprameters to check the validity and compare the steps

    Args:
        pipeline1: mask for the presence of each transformation in the first pipeline.
        pipeline2: mask for the presence of each transformation in the second pipeline.
        scheme: pair of transformations at hand.
    """
    pipelines = {"pipeline1": [], "pipeline2": []}
    parameters = {"pipeline1": [], "pipeline2": []}
    for step in scheme:
        if pipeline1 != "":
            raw_pipeline1 = json.loads(
                pipeline1.replace("'", '"')
                .replace(" ", ",")
                .replace("True", "1")
                .replace("False", "0")
            )
            pipelines["pipeline1"].append(raw_pipeline1[step][0].split("_")[1])
            for param in raw_pipeline1[step][1]:
                parameters["pipeline1"].append(raw_pipeline1[step][1][param])
        if pipeline2 != "":
            raw_pipeline2 = json.loads(
                pipeline2.replace("'", '"')
                .replace(" ", ",")
                .replace("True", "1")
                .replace("False", "0")
            )
            pipelines["pipeline2"].append(raw_pipeline2[step][0].split("_")[1])
            for param in raw_pipeline2[step][1]:
                parameters["pipeline2"].append(raw_pipeline2[step][1][param])
    return pipelines, parameters


def have_same_steps(pipelines):
    """Checks if the pipelines have the same steps.

    Args:
        pipelines: pipelines to check.

    Returns:
        bool: wheter they have the same steps or not.
    """
    pipeline1_has_first = not pipelines["pipeline1"][0].__contains__("NoneType")
    pipeline1_has_second = not pipelines["pipeline1"][1].__contains__("NoneType")
    pipeline2_has_first = not pipelines["pipeline2"][0].__contains__("NoneType")
    pipeline2_has_second = not pipelines["pipeline2"][1].__contains__("NoneType")
    both_just_first = (
        pipeline1_has_first
        and not pipeline1_has_second
        and pipeline2_has_first
        and not pipeline2_has_second
    )
    both_just_second = (
        not pipeline1_has_first
        and pipeline1_has_second
        and not pipeline2_has_first
        and pipeline2_has_second
    )
    both_baseline = (
        not pipeline1_has_first
        and not pipeline1_has_second
        and not pipeline2_has_first
        and not pipeline2_has_second
    )
    return both_just_first or both_just_second or both_baseline


def check_validity(pipelines, result, acc1, acc2):
    """Checks the validity for a specific optimized pair of transformation on a specific dataset and ML algorithm.

    Args:
        pipelines: pair of transformations at hand.
        result: results of the optimization.
        acc1: accuracy of the first pipeline.
        acc2: accuracy of the second pipeline.

    Returns:
        _type_: _description_
    """
    if pipelines["pipeline1"] == [] and pipelines["pipeline2"] == []:
        validity, problem = False, "not_exec"
    elif pipelines["pipeline1"] == [] or pipelines["pipeline2"] == []:
        validity, problem = False, "not_exec_once"
    else:
        if pipelines["pipeline1"].__contains__("NoneType") and pipelines[
            "pipeline2"
        ].__contains__("NoneType"):
            validity = result == 0
        elif pipelines["pipeline1"].__contains__("NoneType") and not (
            pipelines["pipeline2"].__contains__("NoneType")
        ):
            validity = result == 0 or result == 2
        elif not (pipelines["pipeline1"].__contains__("NoneType")) and pipelines[
            "pipeline2"
        ].__contains__("NoneType"):
            validity = result == 0 or result == 1
        else:
            validity = True
        problem = "" if validity else "inconsistent"

    if not (validity) and pipelines["pipeline1"] != [] and pipelines["pipeline2"] != []:
        if have_same_steps(pipelines):
            validity, problem, result = True, "", 0

    return validity, problem, result


def compute_result(result, pipelines, categories, baseline_scores, scores):
    """Computes the validation in Table 3 for a specific optimized pair of transformation on a specific dataset and ML algorithm.

    Args:
        result: results of the optimization process.
        pipelines: pair of transformations at hand.
        categories: possible categories.
        baseline_scores: score of the baseline.
        scores: scores of the two different order.

    Returns:
        string: final category of the case.
    """
    if baseline_scores[0] != baseline_scores[1]:
        raise Exception("Baselines with different scores")

    # case a, b, c, e, i
    if result == 0 and (
        baseline_scores[0] == scores[0] or baseline_scores[1] == scores[1]
    ):
        return "baseline"
    # case d, o
    elif (
        pipelines["pipeline1"].count("NoneType") == 2
        or pipelines["pipeline2"].count("NoneType") == 2
    ):
        if (
            pipelines["pipeline1"].count("NoneType") == 2
            and pipelines["pipeline2"].count("NoneType") == 0
        ):
            if result == 2:
                return categories["second_first"]
            else:
                raise Exception(
                    "pipeline2 is not winning. "
                    + str(pipelines)
                    + " baseline_score "
                    + str(baseline_scores[0])
                    + " scores "
                    + str(scores)
                )
        elif (
            pipelines["pipeline1"].count("NoneType") == 0
            and pipelines["pipeline2"].count("NoneType") == 2
        ):
            if result == 1:
                return categories["first_second"]
            else:
                raise Exception(
                    "pipeline1 is not winning. "
                    + str(pipelines)
                    + " baseline_score "
                    + str(baseline_scores[0])
                    + " scores "
                    + str(scores)
                )
        else:
            raise Exception(
                "Baseline doesn't draw with a pipeline with just one operation. pipelines:"
                + str(pipelines)
                + " baseline_score "
                + str(baseline_scores[0])
                + " scores "
                + str(scores)
            )
    # case f, m, l, g
    elif (
        pipelines["pipeline1"].count("NoneType") == 1
        and pipelines["pipeline2"].count("NoneType") == 1
    ):
        # case f
        if (
            pipelines["pipeline1"][0] == "NoneType"
            and pipelines["pipeline2"][0] == "NoneType"
        ):
            if result == 0:
                return categories["second"]
            else:
                raise Exception(
                    "pipelines is not drawing. "
                    + str(pipelines)
                    + " baseline_score "
                    + str(baseline_scores[0])
                    + " scores "
                    + str(scores)
                )
        # case m
        elif (
            pipelines["pipeline1"][1] == "NoneType"
            and pipelines["pipeline2"][1] == "NoneType"
        ):
            if result == 0:
                return categories["first"]
            else:
                raise Exception(
                    "pipelines is not drawing. "
                    + str(pipelines)
                    + " baseline_score "
                    + str(baseline_scores[0])
                    + " scores "
                    + str(scores)
                    + " algorithm "
                )
        # case g, l
        elif (
            pipelines["pipeline1"][0] == "NoneType"
            and pipelines["pipeline2"][1] == "NoneType"
        ) or (
            pipelines["pipeline1"][1] == "NoneType"
            and pipelines["pipeline2"][0] == "NoneType"
        ):
            if result == 0:
                return categories["first_or_second"]
            else:
                raise Exception(
                    "pipelines is not drawing. "
                    + str(pipelines)
                    + " baseline_score "
                    + str(baseline_scores[0])
                    + " scores "
                    + str(scores)
                    + " algorithm "
                )
    # case h, n
    elif pipelines["pipeline1"].count("NoneType") == 1:
        # case h
        if pipelines["pipeline1"][0] == "NoneType":
            if result == 0:
                return categories["second"]
            elif result == 2:
                return categories["second_first"]
            else:
                raise Exception(
                    "pipeline2 is not winning. "
                    + str(pipelines)
                    + " baseline_score "
                    + str(baseline_scores[0])
                    + " scores "
                    + str(scores)
                    + " algorithm "
                )
        # case n
        elif pipelines["pipeline1"][1] == "NoneType":
            if result == 0:
                return categories["first"]
            elif result == 2:
                return categories["second_first"]
            else:
                raise Exception(
                    "pipeline2 is not winning. "
                    + str(pipelines)
                    + " baseline_score "
                    + str(baseline_scores[0])
                    + " scores "
                    + str(scores)
                    + " algorithm "
                )
    # case p, q
    elif pipelines["pipeline2"].count("NoneType") == 1:
        # case p
        if pipelines["pipeline2"][0] == "NoneType":
            if result == 0:
                return categories["second"]
            elif result == 1:
                return categories["first_second"]
            else:
                raise Exception(
                    "pipeline1 is not winning. "
                    + str(pipelines)
                    + " baseline_score "
                    + str(baseline_scores[0])
                    + " scores "
                    + str(scores)
                    + " algorithm "
                )
        # case q
        elif pipelines["pipeline2"][1] == "NoneType":
            if result == 0:
                return categories["first"]
            elif result == 1:
                return categories["first_second"]
            else:
                raise Exception(
                    "pipeline1 is not winning. "
                    + str(pipelines)
                    + " baseline_score "
                    + str(baseline_scores[0])
                    + " scores "
                    + str(scores)
                    + " algorithm "
                )
    # case r
    elif (
        pipelines["pipeline1"].count("NoneType") == 0
        and pipelines["pipeline2"].count("NoneType") == 0
    ):
        if result == 0:
            return categories["draw"]
        elif result == 1:
            return categories["first_second"]
        elif result == 2:
            return categories["second_first"]
    else:
        raise Exception(
            "This configuration matches nothing. "
            + str(pipelines)
            + " baseline_score "
            + str(baseline_scores[0])
            + " scores "
            + str(scores)
            + " algorithm "
        )


def instantiate_results(
    grouped_by_dataset_result,
    grouped_by_algorithm_results,
    dataset,
    acronym,
    categories,
):
    """Creates the data structure to host the grouped results.

    Args:
        grouped_by_dataset_result: dict that will contain the results grouped by datasets.
        grouped_by_algorithm_results: dict that will contain the result grouped by algorithm.
        dataset: OpenML id of the dataset.
        acronym: acronym of the optimized algorithm.
        categories: possible categories.
    """
    if not (grouped_by_dataset_result.__contains__(dataset)):
        grouped_by_dataset_result[dataset] = {}

    if not (grouped_by_algorithm_results.__contains__(acronym)):
        grouped_by_algorithm_results[acronym] = {}
        for _, category in categories.items():
            grouped_by_algorithm_results[acronym][category] = 0


def get_winner(accuracy1, accuracy2):
    """Determines the winner bwteen two configurations, given the accuracies.

    Args:
        accuracy1: accuracy of the first configuration.
        accuracy2: accuracy of the second configuration.

    Returns:
        int: 1 if the first is the freater, 0 if it is a draw, 2 otherwise.
    """
    if accuracy1 > accuracy2:
        return 1
    elif accuracy1 == accuracy2:
        return 0
    elif accuracy1 < accuracy2:
        return 2
    else:
        raise ValueError("A very bad thing happened.")


def rich_simple_results(simple_results, pipeline_scheme, categories):
    """Enriches the simple results with the compose pipelines.

    Args:
        simple_results: raw results of the optimization process.
        pipeline_scheme: pair of the transformations at hand.
        categories: possible categories.

    Returns:
        dict: enriched results.
    """
    for key, value in simple_results.items():
        first_configuration = value[0]
        second_configuration = value[1]
        pipelines, parameters = compose_pipeline(
            first_configuration["pipeline"],
            second_configuration["pipeline"],
            pipeline_scheme,
        )

        try:
            winner = get_winner(
                first_configuration["accuracy"], second_configuration["accuracy"]
            )
        except Exception as e:
            print(str(e))

        validity, label, winner = check_validity(
            pipelines,
            winner,
            first_configuration["accuracy"],
            second_configuration["accuracy"],
        )

        if validity:
            try:
                baseline_scores = [
                    first_configuration["baseline_score"],
                    second_configuration["baseline_score"],
                ]
                accuracies = [
                    first_configuration["accuracy"],
                    second_configuration["accuracy"],
                ]
                label = compute_result(
                    winner, pipelines, categories, baseline_scores, accuracies
                )
            except Exception as e:
                print(str(e))
                label = categories["inconsistent"]

        first_configuration["pipeline"] = pipelines["pipeline1"]
        second_configuration["pipeline"] = pipelines["pipeline2"]

        first_configuration["parameters"] = parameters["pipeline1"]
        second_configuration["parameters"] = parameters["pipeline2"]

        value.append({"winner": winner, "validity": validity, "label": label})
        simple_results[key] = {"conf1": value[0], "conf2": value[1], "result": value[2]}
    return simple_results


def aggregate_results(simple_results, categories):
    """Aggregates the results by dataset and by algorithm.

    Args:
        simple_results: the raw results of the optimization process.
        categories: possible categories.

    Returns:
        dict: aggregated results by dataset.
        dict: aggregated results by algorithm.
    """
    grouped_by_dataset_result = {}
    grouped_by_algorithm_results = {}

    for key, value in simple_results.items():
        acronym = key.split("_")[0]
        data_set = key.split("_")[1]
        instantiate_results(
            grouped_by_dataset_result,
            grouped_by_algorithm_results,
            data_set,
            acronym,
            categories,
        )

        grouped_by_dataset_result[data_set][acronym] = value["result"]["label"]
        grouped_by_algorithm_results[acronym][value["result"]["label"]] += 1

    return grouped_by_algorithm_results, grouped_by_dataset_result


def compute_summary(grouped_by_algorithm_results, categories):
    """Creates summary from the aggregated results by algorithm.

    Args:
        grouped_by_algorithm_results: aggregated results by algorithm.
        categories: possible categories.

    Returns:
        dict: summary.
    """
    summary = {}
    for _, category in categories.items():
        summary[category] = sum(
            x[category] for x in grouped_by_algorithm_results.values()
        )
    return summary


def save_grouped_by_algorithm_results(
    result_path, grouped_by_algorithm_results, summary, no_algorithms=False
):
    """Saves the results aggregated by algorithm.

    Args:
        result_path: where to save the results.
        grouped_by_algorithm_results: results to save.
        summary: summary of the results.
    """
    with open(os.path.join(result_path, "summary.csv"), "w") as out:
        out.write("," + ",".join(summary.keys()) + "\n")
        if not (no_algorithms):
            for key, value in grouped_by_algorithm_results.items():
                row = key
                for k, v in value.items():
                    row += "," + str(v)
                row += "\n"
                out.write(row)
        row = "summary"
        for key, value in summary.items():
            row += "," + str(value)
        out.write(row)


def save_details_grouped_by_dataset_result(
    result_path, details_grouped_by_dataset_result
):
    """Saves the results aggregated by dataset.

    Args:
        result_path: where to save the results.
        details_grouped_by_dataset_result: results to save.
    """
    for element in algorithms:
        header = False
        acronym = "".join([a for a in element if a.isupper()]).lower()

        with open(os.path.join(result_path, "{}.csv".format(acronym)), "w") as out:

            for dataset, detail_results in details_grouped_by_dataset_result.items():
                if not (header):
                    out.write("," + ",".join(detail_results[acronym].keys()) + "\n")
                    header = True
                results = ",".join(
                    list(
                        str(elem)
                        .replace(",", "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        for elem in detail_results[acronym].values()
                    )
                )
                out.write(dataset + "," + results + "\n")


def extract_results(input_path, filtered_data_sets, pipeline, categories):
    """Extracts and aggregate the optimization results.

    Args:
        input_path: where to load the results.
        filtered_data_sets: OpenML ids of the datasets.
        pipeline: pair of transformations at hand.
        categories: possible categories.

    Returns:
        dict: raw results of the optimization process.
        dict: results aggregated by algorithm.
        dict: results aggregated by dataset.
        dict: summary.
    """
    # load and format the results
    simple_results = load_results(input_path, filtered_data_sets)
    simple_results = rich_simple_results(simple_results, pipeline, categories)

    # summarize the results
    grouped_by_algorithm_results, grouped_by_data_set_result = aggregate_results(
        simple_results, categories
    )
    summary = compute_summary(grouped_by_algorithm_results, categories)

    return (
        simple_results,
        grouped_by_algorithm_results,
        grouped_by_data_set_result,
        summary,
    )


def compute_summary_from_data_set_results(dataset_results, categories):
    """Computes summary for 10x4cv

    Args:
        dataset_results: raw results.
        categories: possible categories.

    Returns:
        dict: summary.
    """
    summary = {
        algorithm: {category: 0 for _, category in categories.items()}
        for algorithm in ["knn", "nb", "rf"]
    }
    for _, results in dataset_results.items():
        for algorithm, category in results.items():
            summary[algorithm][category] += 1
    return summary


def extract_results_10x4cv(
    input_path, filtered_data_sets, pipeline, categories, folds, repeat
):
    """Loads the results from the 10x4cv.

    Args:
        input_path: where to load the results.
        filtered_data_sets: OpenML ids of the datasets.
        pipeline: pair of transformations.
        categories: possible categories.
        folds: number of folds.
        repeat: number of times to repeat the fold-cv

    Returns:
        dict: summary of the 10x4cv.
    """
    from sklearn.model_selection import RepeatedKFold

    # load and format the results
    simple_results = load_results(input_path, filtered_data_sets)
    simple_results = rich_simple_results(simple_results, pipeline, categories)

    # summarize the results
    _, grouped_by_data_set_result = aggregate_results(simple_results, categories)

    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeat, random_state=1)

    summaries = []
    datasets = list(grouped_by_data_set_result.keys())
    for train_index, test_index in rkf.split(datasets):
        train_dict = {
            datasets[your_key]: grouped_by_data_set_result[datasets[your_key]]
            for your_key in train_index
        }
        test_dict = {
            datasets[your_key]: grouped_by_data_set_result[datasets[your_key]]
            for your_key in test_index
        }
        train_per_algorithm = compute_summary_from_data_set_results(
            train_dict, categories
        )
        train_summary = compute_summary(train_per_algorithm, categories)
        train_per_algorithm["summary"] = train_summary
        test_per_algorithm = compute_summary_from_data_set_results(
            test_dict, categories
        )
        test_summary = compute_summary(test_per_algorithm, categories)
        test_per_algorithm["summary"] = test_summary
        summaries.append({"train": train_per_algorithm, "test": test_per_algorithm})

    return summaries


def save_results(
    result_path,
    filtered_data_sets,
    simple_results,
    grouped_by_algorithm_results,
    summary,
):
    """Saves both raw and aggregated results.

    Args:
        result_path: where to save the results.
        filtered_data_sets: OpenML ids of the datasets.
        simple_results: raw results of the optimization process.
        grouped_by_algorithm_results: aggregated results by algorithm.
        summary: summary of the results.
    """
    save_simple_results(result_path, simple_results, filtered_data_sets)
    save_grouped_by_algorithm_results(
        result_path, grouped_by_algorithm_results, summary
    )


def save_results_10x4cv(result_path, summaries):
    """Saves the results of the 10x4cv.

    Args:
        result_path: where to save the results.
        summaries: summary to save.
    """
    for batch in range(len(summaries)):
        for set_, results in summaries[batch].items():
            with open(
                os.path.join(result_path, str(batch + 1) + set_ + ".csv"), "w"
            ) as out:
                out.write("," + ",".join(results["summary"].keys()) + "\n")
                for key, value in results.items():
                    row = key
                    for k, v in value.items():
                        row += "," + str(v)
                    row += "\n"
                    out.write(row)


def save_summary(summary, result_path):
    """Saves the summary of the results.

    Args:
        summary: summary to save.
        result_path: where to save the summary.
    """
    if os.path.exists("summary.csv"):
        os.remove("summary.csv")
    with open(os.path.join(result_path, "summary.csv"), "w") as out:
        keys = summary[list(summary.keys())[0]].keys()
        header = ",".join(keys)
        out.write("," + header + "\n")
        for algorithm, results in summary.items():
            result_string = ",".join([str(elem) for elem in results.values()])
            out.write(algorithm + "," + result_string + "\n")


def max_frequency(x):
    """Returns the element with max frequency and its frequency in a list.

    Args:
        x: list.

    Returns:
        int: elem with max frequency.
        counter: frequency.
    """
    counter = 0
    num = x[0]

    for i in x:
        curr_frequency = x.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num, counter


def create_num_equal_elements_matrix(grouped_by_dataset_result):
    """Creates a matrix counting the invalid results.

    Args:
        grouped_by_dataset_result: aggregated results byalgorithm

    Returns:
        dict: invalid results
    """
    num_equal_elements_matrix = np.zeros((len(algorithms), len(algorithms)))

    for _, value in grouped_by_dataset_result.items():
        list_values = []
        for _, label in value.items():
            if (
                label != "inconsistent"
                and label != "not_exec"
                and label != "not_exec_once"
                and label != "no_majority"
            ):
                list_values.append(label)
        if list_values:
            _, freq = max_frequency(list_values)
            num_equal_elements_matrix[len(list_values) - 1][freq - 1] += 1

    return num_equal_elements_matrix


def save_num_equal_elements_matrix(result_path, num_equal_elements_matrix):
    """Saves the matrix with invalid results.

    Args:
        result_path: where to save the results-
        num_equal_elements_matrix: the matrix to save.
    """
    with open(os.path.join(result_path, "num_equal_elements_matrix.csv"), "w") as out:
        out.write(
            "length,"
            + ",".join(str(i) for i in range(1, len(algorithms) + 1))
            + ",tot\n"
        )
        for i in range(0, np.size(num_equal_elements_matrix, 0)):
            row = str(i + 1)
            sum = 0
            for j in range(0, np.size(num_equal_elements_matrix, 1)):
                value = int(num_equal_elements_matrix[i][j])
                sum += value
                row += "," + str(value)
            row += "," + str(sum) + "\n"
            out.write(row)


def get_results(grouped_by_dataset_result):
    """Gets the results aggregated by dataset in a pandas.DataFrame.

    Args:
        grouped_by_dataset_result: the aggregated results by dataset.

    Returns:
        pandas.DataFrame: dataframe of the aggregated results by dataset.
    """
    data = []
    for dataset, value in grouped_by_dataset_result.items():
        for algorithm, result in value.items():
            if (
                result != "inconsistent"
                and result != "not_exec"
                and result != "no_majority"
            ):
                data.append([int(dataset), algorithm, result])

    df = pd.DataFrame(data)
    df.columns = ["dataset", "algorithm", "class"]
    return df


def encode_data(data):
    """Encodes with ordinal encoding.

    Args:
        data: data to encode.

    Returns:
        pandas.DataFrame: encoded data.
    """
    numeric_features = data.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns
    categorical_features = data.select_dtypes(include=["object"]).columns
    reorder_features = list(numeric_features) + list(categorical_features)
    encoded = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("a", FunctionTransformer())]), numeric_features),
            ("cat", Pipeline(steps=[("b", OrdinalEncoder())]), categorical_features),
        ]
    ).fit_transform(data)
    encoded = pd.DataFrame(encoded, columns=reorder_features)

    return encoded


def join_result_with_extended_meta_features(filtered_datasets, data):
    """Enriches the results of the optimization process with the complete OpenML meta-fratures in resources/meta_features.

    Args:
        filtered_datasets: OpenML ids of the datasets.
        data: results of the optimization process.

    Returns:
        pandas.DataFrame: dataframe with meta-features.
    """
    meta = pd.read_csv(os.path.join(META_FEATURES_PATH, "extended-meta-features.csv"))
    meta = meta.loc[meta["id"].isin(filtered_datasets)]
    meta = meta.drop(columns=["name", "runs"])

    join = pd.merge(data, meta, left_on="dataset", right_on="id")
    join = join.drop(columns=["id"])

    return join


def join_result_with_extracted_meta_features(data, impute):
    """Enriches the results of the optimization process with the extractes meta-fratures in resources/meta_features.

    Args:
        data: results of the optimization process.
        impute: wheter to impute the missing meta-features or not.

    Returns:
        pandas.DataFrame: dataframe with meta-features.
    """
    meta = pd.read_csv(
        os.path.join(
            META_FEATURES_PATH,
            ("imputed-mean-" if impute else "") + "extracted-meta-features.csv",
        ),
        index_col=False,
    )

    join = pd.merge(meta, data, left_on="id", right_on="dataset")
    join = join.drop(columns=["id"])

    return join


def join_result_with_simple_meta_features(filtered_datasets, data):
    """Enriches the results of the optimization process with the basic meta-fratures in resources/meta_features.

    Args:
        filtered_datasets: OpenML ids of the datasets.
        data: results of the optimization process.

    Returns:
        pandas.DataFrame: dataframe with meta-features.
    """
    meta = pd.read_csv(os.path.join(META_FEATURES_PATH, "simple-meta-features.csv"))
    meta = meta.loc[meta["did"].isin(filtered_datasets)]
    meta = meta.drop(columns=["version", "status", "format", "uploader", "row", "name"])
    meta = meta.astype(int)

    join = pd.merge(data, meta, left_on="dataset", right_on="did")
    join = join.drop(columns=["did"])

    return join


def modify_class(data, categories, option):
    """Given the optimization results, determines if an order can decided.

    Args:
        data: results of the optimization process.
        categories: distinct labels in the optimization process.
        option: how to aggregate the results.

    Returns:
        pandas.DataFrame: the results with the labels changed.
    """
    for key, value in categories.items():
        if option == "group_all":
            if key == "first_second" or key == "second_first" or key == "not_exec_once":
                data = data.replace(value, "order")
            else:
                data = data.replace(value, "no_order")
        if option == "group_no_order":
            if (
                key != "first_second"
                and key != "second_first"
                and key != "not_exec_once"
            ):
                data = data.replace(value, "no_order")
            if key == "not_exec_once":
                data = data.drop(data.index[data["class"] == value].tolist())

    return data


def create_correlation_matrix(data):
    """Creates a correlation matrix with several frequency tests (e.g., kendall, pearson, spearman).

    Args:
        data: frequencies of the optimization process.

    Returns:
        numpy.array: array of indecies
    """
    encoded = encode_data(data)

    kendall = encoded.corr(method="kendall")["class"].to_frame()
    pearson = encoded.corr(method="pearson")["class"].to_frame()
    spearman = encoded.corr(method="spearman")["class"].to_frame()
    kendall.columns = ["kendall"]
    pearson.columns = ["pearson"]
    spearman.columns = ["spearman"]

    correlation_matrix = pd.concat([kendall, pearson, spearman], axis=1, sort=False)

    X, y = encoded.drop(columns=["class"]), encoded["class"]
    visualizer = FeatureCorrelation(
        method="mutual_info-classification", labels=X.columns
    )
    visualizer.fit(X, y)

    correlation_matrix = correlation_matrix.drop("class", axis=0)
    correlation_matrix["mutual_info-classification"] = visualizer.scores_.tolist()

    return correlation_matrix


def save_data_frame(result_path, data_frame, index):
    """Saves a pandas.DataFrame.

    Args:
        result_path: where to save the data frame.
        data_frame: data frame to save.
        index: wheter to include the index or not.
    """
    data_frame.to_csv(result_path, index=index)


def save_correlation_matrix(result_path, name, correlation_matrix, group_no_order):
    """Saves correlation matrix of the frequencies tests.

    Args:
        result_path: where to save the set.
        name: file name to save.
        correlation_matrix: matrix of correlations from the frequencies tests.
        group_no_order: whether aggregate the result by algorithm or not.
    """
    save_data_frame(
        os.path.join(
            result_path, name + ("_grouped" if group_no_order else "") + ".csv"
        ),
        correlation_matrix,
        index=True,
    )


def save_train_meta_learner(result_path, name, train_meta_learner, group_no_order):
    """Saves the train set for the meta learner.

    Args:
        result_path: where to save the set.
        name: file name to save.
        train_meta_learner: set to save.
        group_no_order: whether aggregate the result by algorithm or not.
    """
    save_data_frame(
        os.path.join(
            result_path, name + ("_grouped" if group_no_order else "") + ".csv"
        ),
        train_meta_learner,
        index=False,
    )


def chi2test(observed, distribution, prob=0.95):
    """Performs the chi square test.

    Args:
        observed: the observed frequencies.
        distribution: frequencies of the distribution to compare with.
        prob (optional): probability of the chi square test. Defaults to 0.95.

    Returns:
        _type_: _description_
    """
    table = [observed, distribution]
    stat, p, dof, expected = chi2_contingency(table)
    critical = chi2.ppf(prob, dof)
    statistic_test = abs(stat) >= critical
    alpha = 1.0 - prob
    p_value = p <= alpha
    return (
        critical // 0.0001 / 10000,
        stat // 0.0001 / 10000,
        statistic_test,
        alpha // 0.0001 / 10000,
        p // 0.0001 / 10000,
        p_value,
    )


def chi2tests(grouped_by_algorithm_results, summary, categories, uniform):
    """Performs the chi square tests for the prototype construction.

    Args:
        grouped_by_algorithm_results: frequencies of the optimization process.
        summary: summary of the optimization process.
        categories: labels of the pair of transofrmations.
        uniform: wheter to compare with a uniform distribution or not.
    """

    def run_chi2test(observed, uniform, formatted_input):
        tot = sum(observed)
        observed.sort(reverse=True)

        if uniform:
            length = len(observed)
            uniform_frequency = tot / length
            distribution = [uniform_frequency] * length
        else:
            distribution = [tot * 0.9, tot * 0.1]

        critical, stat, statistic_test, alpha, p, p_value = chi2test(
            observed, distribution
        )

        formatted_output = {
            "critical": critical,
            "stat": stat,
            "statistic_test": statistic_test,
            "alpha": alpha,
            "p": p,
            "p_value_test": p_value,
        }

        formatted_input.update(formatted_output)
        return formatted_input

    grouped_by_algorithm_results["summary"] = summary
    test = {}
    order_test = {}
    not_order_test = {}
    for algorithm, values in grouped_by_algorithm_results.items():

        total = sum(a for a in values.values())
        not_valid = sum(
            [
                values[categories["inconsistent"]],
                values[categories["not_exec"]],
                values[categories["not_exec_once"]],
            ]
        )
        valid = total - not_valid

        order = sum(
            [values[categories["first_second"]], values[categories["second_first"]]]
        )
        not_order = valid - order

        formatted_input = {"valid": valid, "order": order, "not_order": not_order}

        test[algorithm] = run_chi2test([order, not_order], uniform, formatted_input)

        first_second = values[categories["first_second"]]
        second_first = values[categories["second_first"]]

        formatted_input = {
            "order": order,
            categories["first_second"]: first_second,
            categories["second_first"]: second_first,
        }

        order_test[algorithm] = run_chi2test(
            [first_second, second_first], uniform, formatted_input
        )

        if uniform:
            first = values[categories["first"]]
            second = values[categories["second"]]
            first_or_second = values[categories["first_or_second"]]
            draw = values[categories["draw"]]
            baseline = values[categories["baseline"]]

            formatted_input = {
                "not_order": not_order,
                categories["first"]: first,
                categories["second"]: second,
                categories["first_or_second"]: first_or_second,
                categories["draw"]: draw,
                categories["baseline"]: baseline,
            }

            not_order_test[algorithm] = run_chi2test(
                [first_second, second_first], uniform, formatted_input
            )

    return {"test": test, "order_test": order_test, "not_order_test": not_order_test}


def save_chi2tests(result_path, tests):
    """Saves the chi square test results.

    Args:
        result_path: where to save the results.
        tests: results of the test.
    """

    def saver(collection, name):
        with open(name, "w") as out:
            header = False
            for algorithm, values in collection.items():
                if not header:
                    out.write("," + ",".join(values.keys()) + "\n")
                    header = True
                row = algorithm
                for _, value in values.items():
                    row += "," + str(value)
                row += "\n"
                out.write(row)

    for key, value in tests.items():
        if value:
            saver(value, os.path.join(result_path, key + ".csv"))


def experiments_summarizer(pipeline, toy, cache):
    """Summarizes the frequencies obtained in the optimization process.

    Args:
        pipeline: pair of transformations at hand.
        toy: whether it is the toy example or not.
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)
    if toy:
        path = os.path.join(RAW_RESULT_PATH, "toy")
    elif cache:
        path = os.path.join(RAW_RESULT_PATH, "paper")
    else:
        path = os.path.join(RAW_RESULT_PATH, "paper_new")
    input_path = os.path.join(path, "prototype_construction", "_".join(pipeline))
    result_path = create_directory(input_path, "summary")
    categories = create_possible_categories(pipeline)
    filtered_data_sets = get_filtered_datasets(
        experiment="prototype_construction", toy=toy
    )

    (
        simple_results,
        grouped_by_algorithm_results,
        grouped_by_data_set_result,
        summary,
    ) = extract_results(input_path, filtered_data_sets, pipeline, categories)

    save_results(
        create_directory(result_path, "algorithms_summary"),
        filtered_data_sets,
        simple_results,
        grouped_by_algorithm_results,
        summary,
    )

    # compute the chi square test
    for uniform in [True, False]:
        temp_result_path = create_directory(result_path, "chi2tests")
        tests = chi2tests(grouped_by_algorithm_results, summary, categories, uniform)
        if uniform:
            temp_result_path = create_directory(temp_result_path, "uniform")
        else:
            temp_result_path = create_directory(temp_result_path, "binary")
        save_chi2tests(temp_result_path, tests)

    # compute the matrix with the number of equal result per data set
    num_equal_elements_matrix = create_num_equal_elements_matrix(
        grouped_by_data_set_result
    )
    save_num_equal_elements_matrix(
        create_directory(result_path, "correlations"), num_equal_elements_matrix
    )
    data = get_results(grouped_by_data_set_result)

    # create the correlation matrices
    for group_no_order in [True, False]:
        join = join_result_with_simple_meta_features(filtered_data_sets, data)
        if group_no_order:
            join = modify_class(join, categories, "group_no_order")
        correlation_matrix = create_correlation_matrix(join)
        save_correlation_matrix(
            create_directory(result_path, "correlations"),
            "correlation_matrix",
            correlation_matrix,
            group_no_order,
        )


def experiments_summarizer_10x4cv(pipeline, toy, cache):
    """Summarizes the frequencies of the 10x4cv analysis.

    Args:
        pipeline: pair of transformations at hand.
        toy: whether it is the toy example or not.
    """

    # configure environment
    if toy:
        path = os.path.join(RAW_RESULT_PATH, "toy")
    elif cache:
        path = os.path.join(RAW_RESULT_PATH, "paper")
    else:
        path = os.path.join(RAW_RESULT_PATH, "paper_new")
    input_path = os.path.join(path, "prototype_construction", "_".join(pipeline))
    result_path = create_directory(input_path, "summary")
    result_path = create_directory(result_path, "10x4cv")
    categories = create_possible_categories(pipeline)
    filtered_data_sets = get_filtered_datasets(
        experiment="prototype_construction", toy=toy
    )
    folds = 3 if toy else 4
    repeat = 10

    summaries = extract_results_10x4cv(
        input_path, filtered_data_sets, pipeline, categories, folds, repeat
    )

    save_results_10x4cv(create_directory(result_path, "raw"), summaries)

    prob = 0.95
    a = (
        0.1
        if categories["first_second"] == "FN"
        else (0.9 if categories["first_second"] == "DF" else 0.5)
    )
    b = round(1 - a, 2)
    results = pd.DataFrame()
    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    for batch in range(len(summaries)):
        total_train = (
            summaries[batch]["train"]["summary"][categories["first_second"]]
            + summaries[batch]["train"]["summary"][categories["second_first"]]
        )
        total_test = (
            summaries[batch]["test"]["summary"][categories["first_second"]]
            + summaries[batch]["test"]["summary"][categories["second_first"]]
        )
        train_frequencies = [
            summaries[batch]["train"]["summary"][categories["first_second"]],
            summaries[batch]["train"]["summary"][categories["second_first"]],
        ]
        test_frequencies = [
            summaries[batch]["test"]["summary"][categories["first_second"]],
            summaries[batch]["test"]["summary"][categories["second_first"]],
        ]
        try:
            critical, stat, statistic_test, alpha, p, p_value = chi2test(
                train_frequencies, test_frequencies, prob
            )
            (
                critical_train,
                stat_train,
                statistic_test_train,
                alpha_train,
                p_train,
                p_value_train,
            ) = chi2test(train_frequencies, [total_train * a, total_train * b], prob)
            (
                critical_test,
                stat_test,
                statistic_test_test,
                alpha_test,
                p_test,
                p_value_test,
            ) = chi2test(test_frequencies, [total_test * a, total_test * b], prob)
            results = results.append(
                {
                    "critical": critical,
                    "stat": stat,
                    "statistic_test": statistic_test,
                    "alpha": alpha,
                    "p": p,
                    "p_value": p_value,
                },
                ignore_index=True,
            )
            results_train = results_train.append(
                {
                    "critical": critical_train,
                    "stat": stat_train,
                    "statistic_test": statistic_test_train,
                    "alpha": alpha_train,
                    "p": p_train,
                    "p_value": p_value_train,
                },
                ignore_index=True,
            )
            results_test = results_test.append(
                {
                    "critical": critical_test,
                    "stat": stat_test,
                    "statistic_test": statistic_test_test,
                    "alpha": alpha_test,
                    "p": p_test,
                    "p_value": p_value_test,
                },
                ignore_index=True,
            )
        except:
            pass
    results.to_csv(os.path.join(result_path, "summary.csv"), index=False)
    results_train.to_csv(os.path.join(result_path, "summary_train.csv"), index=False)
    results_test.to_csv(os.path.join(result_path, "summary_test.csv"), index=False)

    results = pd.DataFrame()
    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    for repetition in range(repeat):
        batch_results = pd.DataFrame()
        batch_train_results = pd.DataFrame()
        batch_test_results = pd.DataFrame()
        for fold in range(folds):
            batch = repetition * folds + fold
            total_train = (
                summaries[batch]["train"]["summary"][categories["first_second"]]
                + summaries[batch]["train"]["summary"][categories["second_first"]]
            )
            total_test = (
                summaries[batch]["test"]["summary"][categories["first_second"]]
                + summaries[batch]["test"]["summary"][categories["second_first"]]
            )
            train_frequencies = [
                summaries[batch]["train"]["summary"][categories["first_second"]],
                summaries[batch]["train"]["summary"][categories["second_first"]],
            ]
            test_frequencies = [
                summaries[batch]["test"]["summary"][categories["first_second"]],
                summaries[batch]["test"]["summary"][categories["second_first"]],
            ]
            try:
                critical, stat, statistic_test, alpha, p, p_value = chi2test(
                    train_frequencies, test_frequencies, prob
                )
                (
                    critical_train,
                    stat_train,
                    statistic_test_train,
                    alpha_train,
                    p_train,
                    p_value_train,
                ) = chi2test(
                    train_frequencies, [total_train * a, total_train * b], prob
                )
                (
                    critical_test,
                    stat_test,
                    statistic_test_test,
                    alpha_test,
                    p_test,
                    p_value_test,
                ) = chi2test(test_frequencies, [total_test * a, total_test * b], prob)
                batch_results = batch_results.append(
                    {
                        "critical": critical,
                        "stat": stat,
                        "statistic_test": statistic_test,
                        "alpha": alpha,
                        "p": p,
                        "p_value": p_value,
                    },
                    ignore_index=True,
                )
                batch_train_results = batch_train_results.append(
                    {
                        "critical": critical_train,
                        "stat": stat_train,
                        "statistic_test": statistic_test_train,
                        "alpha": alpha_train,
                        "p": p_train,
                        "p_value": p_value_train,
                    },
                    ignore_index=True,
                )
                batch_test_results = batch_test_results.append(
                    {
                        "critical": critical_test,
                        "stat": stat_test,
                        "statistic_test": statistic_test_test,
                        "alpha": alpha_test,
                        "p": p_test,
                        "p_value": p_value_test,
                    },
                    ignore_index=True,
                )
            except:
                pass
        batch_results = batch_results.mean().round(3)
        batch_train_results = batch_train_results.mean().round(3)
        batch_test_results = batch_test_results.mean().round(3)
        results = results.append(batch_results, ignore_index=True)
        results_train = results_train.append(batch_train_results, ignore_index=True)
        results_test = results_test.append(batch_test_results, ignore_index=True)
    results.to_csv(os.path.join(result_path, "summary_with_mean_.csv"), index=False)
    results_train.to_csv(
        os.path.join(result_path, "summary_train_with_mean_.csv"), index=False
    )
    results_test.to_csv(
        os.path.join(result_path, "summary_test_with_mean_.csv"), index=False
    )


def graph_maker(toy, cache):
    """Plots the frequencies obtained in the optimization process.

    Args:
        toy: whether it is the toy example or not.
    """
    logging.getLogger("matplotlib.font_manager").disabled = True
    if toy:
        input_path = os.path.join(RAW_RESULT_PATH, "toy")
        result_path = os.path.join(ARTIFACTS_PATH, "toy")
    elif cache:
        input_path = os.path.join(RAW_RESULT_PATH, "paper")
        result_path = os.path.join(ARTIFACTS_PATH, "paper")
    else:
        input_path = os.path.join(RAW_RESULT_PATH, "paper_new")
        result_path = os.path.join(ARTIFACTS_PATH, "paper_new")
    input_path = os.path.join(input_path, "prototype_construction")

    data = {}
    data[0] = {
        "title": r"$T_1$ = Feat. Eng., $T_2$ = Normalize",
        "data": pd.read_csv(
            os.path.join(
                input_path,
                "features_normalize",
                "summary",
                "algorithms_summary",
                "summary.csv",
            )
        ).reindex([1, 0, 2, 3]),
    }
    data[1] = {
        "title": r"$T_1$ = Discretize, $T_2$ = Feat. Eng.",
        "data": pd.read_csv(
            os.path.join(
                input_path,
                "discretize_features",
                "summary",
                "algorithms_summary",
                "summary.csv",
            )
        ).reindex([1, 0, 2, 3]),
    }
    data[2] = {
        "title": r"$T_1$ = Feat. Eng., $T_2$ = Rebalance",
        "data": pd.read_csv(
            os.path.join(
                input_path,
                "features_rebalance",
                "summary",
                "algorithms_summary",
                "summary.csv",
            )
        ).reindex([1, 0, 2, 3]),
    }
    data[3] = {
        "title": r"$T_1$ = Discretize, $T_2$ = Rebalance",
        "data": pd.read_csv(
            os.path.join(
                input_path,
                "discretize_rebalance",
                "summary",
                "algorithms_summary",
                "summary.csv",
            )
        ).reindex([1, 0, 2, 3]),
    }
    labels = [r"$T_1$", r"$T_2$", r"$T_1 \to T_2$", r"$T_2 \to T_1$", "Baseline"]
    colors = [
        "gold",
        "mediumspringgreen",
        "royalblue",
        "sienna",
        "mediumpurple",
        "salmon",
    ]
    patterns = ["/", "\\", "o", "-", "x", ".", "O", "+", "*", "|"]

    SMALL_SIZE = 8
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    fig, axs = plt.subplots(2, 2)
    n_groups = 3

    for i in range(0, 2):
        for j in range(0, 2):
            data[i * 2 + j]["data"].columns = list(
                range(0, len(data[i * 2 + j]["data"].columns))
            )
            data[i * 2 + j]["data"] = data[i * 2 + j]["data"].drop(columns=[3, 6])
            index = np.arange(n_groups)
            bar_width = 0.4

            for k in range(1, 6):
                axs[i, j].bar(
                    (index * bar_width * 8) + (bar_width * (k - 1)),
                    data[i * 2 + j]["data"].iloc[:-1, k],
                    bar_width,
                    label=labels[k - 1],
                    color=colors[k - 1],
                    hatch=patterns[k - 1],
                )

            axs[i, j].set(ylabel="Number of wins")
            axs[i, j].set(xlabel="Algorithms")
            axs[i, j].set_title(data[i * 2 + j]["title"])
            axs[i, j].set_ylim([0, 40])
            plt.setp(
                axs,
                xticks=(index * bar_width * 8) + 0.8,
                xticklabels=["NB", "KNN", "RF"],
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=8,
        bbox_to_anchor=(0.5, 1.0),
    )
    text = fig.text(-0.2, 1.05, "", transform=axs[1, 1].transAxes)
    fig.set_size_inches(20, 10, forward=True)
    fig.tight_layout(h_pad=3.0, w_pad=4.0)
    fig.savefig(
        os.path.join(result_path, "Figure4.pdf"),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )


def graph_maker_10x4cv(toy, cache):
    """Plots the 10x4cv chart.

    Args:
        toy: whether it is the toy example or not.
    """
    logging.getLogger("matplotlib.font_manager").disabled = True
    cv_file_name = "summary_with_mean_.csv"
    if toy:
        prototype_construction_path = os.path.join(RAW_RESULT_PATH, "toy")
        plot_path = os.path.join(ARTIFACTS_PATH, "toy")
    elif cache:
        prototype_construction_path = os.path.join(RAW_RESULT_PATH, "paper")
        plot_path = os.path.join(ARTIFACTS_PATH, "paper")
    else:
        prototype_construction_path = os.path.join(RAW_RESULT_PATH, "paper_new")
        plot_path = os.path.join(ARTIFACTS_PATH, "paper_new")
    prototype_construction_path = os.path.join(
        prototype_construction_path, "prototype_construction"
    )

    fn_path = os.path.join(prototype_construction_path, "features_normalize")
    fn_cv_path = os.path.join(fn_path, "summary", "10x4cv")
    try:
        fn_df = pd.read_csv(os.path.join(fn_cv_path, cv_file_name))
        fn_df["fn"] = fn_df["p"]
    except:
        pass

    df_path = os.path.join(prototype_construction_path, "discretize_features")
    df_cv_path = os.path.join(df_path, "summary", "10x4cv")
    try:
        df_df = pd.read_csv(os.path.join(df_cv_path, cv_file_name))
        df_df["df"] = df_df["p"]
    except:
        pass

    if "fn_df" in locals() and "df_df" in locals():
        df = pd.concat([fn_df["fn"], df_df["df"]], axis=1)
        ticks_array = [r"$F \rightarrow N$", r"$D \rightarrow F$"]
    elif "fn_df" in locals():
        df = fn_df["fn"]
        ticks_array = [r"$F \rightarrow N$"]
    elif "df_df" in locals():
        df = df_df["df"]
        ticks_array = [r"$D \rightarrow F$"]

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots()
    ax.boxplot(df, widths=0.3)
    ax.axhline(y=0.05, color="grey", linestyle="--")
    ax.set_xticklabels(ticks_array)
    ax.set_ylabel("Means of the p-values")
    ax.set_yticks([0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.set_size_inches(12, 6, forward=True)
    fig.savefig(os.path.join(plot_path, "Figure5.pdf"))


def run_p_binom_test(toy, cache):
    """Performs the biomial test to understand if the frequencies obtain in the experiments are significant.

    Args:
        toy: whether it is the toy example or not.
    """
    experiment = "toy" if toy else ("paper" if cache else "paper_new")
    subprocess.call(
        f"Rscript experiment/results_processors/p_binom_test.R {experiment}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def prototype_construction(toy_example, cache):
    """Performs the prototype construction analysis.

    Args:
        toy_example: wheter it is the toy example or not.
    """
    print("PC02. Perform significance test over the results\n")
    experiments_summarizer(pipeline=["features", "rebalance"], toy=toy_example, cache=cache)
    experiments_summarizer(pipeline=["discretize", "features"], toy=toy_example, cache=cache)
    experiments_summarizer(pipeline=["features", "normalize"], toy=toy_example, cache=cache)
    experiments_summarizer(pipeline=["discretize", "rebalance"], toy=toy_example, cache=cache)

    print("PC03. Validate results with 10x4 CV\n")
    experiments_summarizer_10x4cv(pipeline=["features", "rebalance"], toy=toy_example, cache=cache)
    experiments_summarizer_10x4cv(pipeline=["discretize", "features"], toy=toy_example, cache=cache)
    experiments_summarizer_10x4cv(pipeline=["features", "normalize"], toy=toy_example, cache=cache)
    experiments_summarizer_10x4cv(pipeline=["discretize", "rebalance"], toy=toy_example, cache=cache)

    print(
        "PC04. Select winning order for each pair (using results from PC02 and PC03)\n"
    )
    print("PC05. Combine ordered pairs of transformations\n")
    graph_maker(toy_example, cache)
    graph_maker_10x4cv(toy_example, cache)
    run_p_binom_test(toy_example, cache)
