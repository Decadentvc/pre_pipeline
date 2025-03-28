import os
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import gridspec
from os import listdir
from os.path import isfile, join
from utils.common import *


def load_results_pipelines(input_path, filtered_data_sets):
    """Loads the results about the exhaustive prototypes optimization.

    Args:
        input_path: where to load the results.
        filtered_data_sets: OpenML ids of the datasets.

    Returns:
        dict: for each pair of ML algorithm and dataset, it reports the accuracy of each prototype.
    """
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == "json"]

    for acronym in filtered_data_sets:
        algorithm = acronym.split("_")[0]
        data_set = acronym.split("_")[1]

        if acronym in results:
            if not (algorithm in results_map.keys()):
                results_map[algorithm] = {}
            results_map[algorithm][data_set] = []
            with open(os.path.join(input_path, acronym + ".json")) as json_file:
                data = json.load(json_file)
                for i in range(0, 24):
                    index = data["pipelines"][i]["index"]
                    accuracy = data["pipelines"][i]["accuracy"]
                    results_map[algorithm][data_set].append(
                        {"index": index, "accuracy": accuracy}
                    )

    return results_map


def load_results_auto(input_path, filtered_data_sets):
    """Loads the results about the custom prototypes optimization.

    Args:
        input_path: where to load the results.
        filtered_data_sets: OpenML ids of the datasets.

    Returns:
        dict: for each pair of ML algorithm and dataset, it reports the accuracy of each prototype.
    """
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == "json"]

    for acronym in filtered_data_sets:
        algorithm = acronym.split("_")[0]
        data_set = acronym.split("_")[1]

        if not (algorithm in results_map.keys()):
            results_map[algorithm] = {}
        if acronym in results:
            with open(os.path.join(input_path, acronym + ".json")) as json_file:
                data = json.load(json_file)
                try:
                    accuracy = data["context"]["best_config"]["score"] // 0.0001 / 100
                except:
                    accuracy = 0

                try:
                    baseline = data["context"]["baseline_score"] // 0.0001 / 100
                except:
                    baseline = 0
        else:
            accuracy = 0
            baseline = 0

        results_map[algorithm][data_set] = (accuracy, baseline)

    return results_map


def declare_winners(results_map):
    """Declares the best prototype among the optimized ones.

    Args:
        results_map: for each pair of ML algorithm and dataset, it reports the accuracy of each prototype.

    Returns:
        dict: for each pair of ML algorithm and dataset, it reports the best prototype.
    """
    winners_map = {}
    for algorithm, value in results_map.items():
        winners_map[algorithm] = {}

        for dataset, pipelines in value.items():
            index_max_accuracy = -1
            for i in range(0, 24):
                if index_max_accuracy == -1:
                    if results_map[algorithm][dataset][i]["accuracy"] != 0:
                        index_max_accuracy = i
                else:
                    if (
                        results_map[algorithm][dataset][index_max_accuracy]["accuracy"]
                        < results_map[algorithm][dataset][i]["accuracy"]
                    ):
                        index_max_accuracy = i
            winners_map[algorithm][dataset] = index_max_accuracy

    return winners_map


def get_winners_accuracy(results_map):
    """Enriches the winners_map with the information of accuracy.

    Args:
        results_map: for each pair of ML algorithm and dataset, it reports the best prototype.

    Returns:
        dict: for each pair of ML algorithm and dataset, it reports the accuracy of the best prototype.
    """
    accuracy_map = {}
    for algorithm, value in results_map.items():
        accuracy_map[algorithm] = {}

        for dataset, pipelines in value.items():
            index_max_accuracy = -1
            for i in range(0, 24):
                if index_max_accuracy == -1:
                    if results_map[algorithm][dataset][i]["accuracy"] != 0:
                        index_max_accuracy = i
                else:
                    if (
                        results_map[algorithm][dataset][index_max_accuracy]["accuracy"]
                        < results_map[algorithm][dataset][i]["accuracy"]
                    ):
                        index_max_accuracy = i
            accuracy_map[algorithm][dataset] = results_map[algorithm][dataset][
                index_max_accuracy
            ]["accuracy"]

    return accuracy_map


def summarize_winners(winners_map):
    """Pivots the winners_map.

    Args:
        winners_map: for each pair of ML algorithm and dataset, it reports the accuracy of the best prototype.

    Returns:
        dict: for each protoype counts the winners.
    """
    summary_map = {}
    for algorithm, value in winners_map.items():
        summary_map[algorithm] = {}
        for i in range(-1, 24):
            summary_map[algorithm][i] = 0

        for _, winner in value.items():
            summary_map[algorithm][winner] += 1

    return summary_map


def save_summary(summary_map, results_path, plots_path, plot):
    """Saves and plot the summary about the exhaustive prototypes optimization.

    Args:
        summary_map: for each protoype counts the winners.
        results_path: where to save the results.
        plots_path: where to save the plot.
        plot: wheter to plot or not.
    """
    if os.path.exists("summary.csv"):
        os.remove("summary.csv")
    total = {}
    algorithm_map = {"nb": "NB", "knn": "KNN", "rf": "RF"}
    win = {}
    pipelines = []

    for algorithm, value in summary_map.items():
        pipelines = []
        with open(os.path.join(results_path, "{}.csv".format(algorithm)), "w") as out:
            out.write("pipeline,winners\n")
            winners_temp = []
            for pipeline, winner in value.items():
                out.write(str(pipeline) + "," + str(winner) + "\n")
                pipelines.append(str(pipeline))
                winners_temp.append(int(winner))
                total[str(pipeline)] = (
                    winner
                    if not (str(pipeline) in total.keys())
                    else total[str(pipeline)] + winner
                )
            win[algorithm] = winners_temp[1:]
            pipelines = pipelines[1:]

    if plot:
        winners = {
            "pipelines": pipelines,
            "nb": [e / 168 * 100 for e in win["nb"]],
            "knn": [e / 168 * 100 for e in win["knn"]],
            "rf": [e / 168 * 100 for e in win["rf"]],
        }
        winners["total"] = [
            winners["nb"][j] + winners["knn"][j] + winners["rf"][j]
            for j in range(len(winners["knn"]))
        ]
        winners = pd.DataFrame.from_dict(winners)
        # winners = winn ers.sort_values(by=['total'], ascending=False)

        SMALL_SIZE = 14
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 18

        plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

        plt.bar(
            [str(int(a) + 1) for a in winners["pipelines"]],
            winners["nb"],
            label=algorithm_map["nb"],
            color="lightcoral",
        )
        plt.bar(
            [str(int(a) + 1) for a in winners["pipelines"]],
            winners["knn"],
            bottom=winners["nb"],
            label=algorithm_map["knn"],
            color="darkturquoise",
        )
        plt.bar(
            [str(int(a) + 1) for a in winners["pipelines"]],
            winners["rf"],
            bottom=winners["nb"] + winners["knn"],
            label=algorithm_map["rf"],
            color="violet",
        )

        plt.xlabel("Prototype ID", labelpad=20.0)
        plt.ylabel(
            "Percentage of cases for which a prototype\nachieved the best performance",
            labelpad=20.0,
        )
        plt.yticks(
            ticks=np.linspace(0, 20, 11),
            labels=["{}%".format(int(x)) for x in np.linspace(0, 20, 11)],
        )
        # plt.title('Comparison of the goodness of the prototypes')
        plt.legend()
        # plt.subplots_adjust(left=5, bottom=5, right=5, top=5, wspace=0, hspace=0)
        fig = plt.gcf()
        fig.set_size_inches(15, 6)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_path, "Figure12.pdf"))

        plt.clf()


def save_comparison(results_pipelines, results_auto, result_path, plot_path, plot):
    """Saves and plot the comparison between the custom and exhaustive prototypes optimization.

    Args:
        results_pipelines: results about the exhaustive prototypes optimization.
        results_auto: results about the custom prototypes optimization.
        result_path: where to save the comparison.
        plot_path: where to save the plot.
        plot: whether to plot or not.
    """
    if os.path.exists("comparison.csv"):
        os.remove("comparison.csv")
    plot_results = {}

    for algorithm, value in results_pipelines.items():
        plot_results[algorithm] = {}
        with open(os.path.join(result_path, "{}.csv".format(algorithm)), "w") as out:
            out.write("dataset,exhaustive,pseudo-exhaustive,baseline,score\n")
            for dataset, accuracy in value.items():
                score = (
                    0
                    if (accuracy - results_auto[algorithm][dataset][1]) == 0
                    else (
                        results_auto[algorithm][dataset][0]
                        - results_auto[algorithm][dataset][1]
                    )
                    / (accuracy - results_auto[algorithm][dataset][1])
                )
                out.write(
                    str(dataset)
                    + ","
                    + str(accuracy)
                    + ","
                    + str(results_auto[algorithm][dataset][0])
                    + ","
                    + str(results_auto[algorithm][dataset][1])
                    + ","
                    + str(score)
                    + "\n"
                )
                plot_results[algorithm][dataset] = score

    if plot:
        SMALL_SIZE = 14
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 18

        plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=SMALL_SIZE)  # fontsize of the figure title

        plt.axhline(y=1.0, color="#aaaaaa", linestyle="--")

        plt.boxplot(
            [
                [value for value in plot_results["nb"].values() if value != 0],
                [value for value in plot_results["knn"].values() if value != 0],
                [value for value in plot_results["rf"].values() if value != 0],
            ]
        )

        # plt.xlabel('Algorithms', labelpad=15.0)
        plt.xticks([1, 2, 3], ["NB", "KNN", "RF"])
        plt.ylabel("Normalized distance", labelpad=15.0)
        plt.xlabel("Algorithms", labelpad=15.0)
        if "paper" in plot_path:
            plt.yticks(np.linspace(0, 1.2, 7))
            plt.ylim(0.0, 1.25)
        # plt.title('Evaluation of the prototype building through the proposed precedence')
        # plt.tight_layout()
        # plt.subplots_adjust(left=5, bottom=5, right=5, top=5, wspace=0, hspace=0)
        plt.tight_layout(pad=4)
        fig = plt.gcf()
        fig.set_size_inches(15, 6)
        fig.savefig(os.path.join(plot_path, "Figure13.pdf"))

        plt.clf()


def load_custom_vs_ml_algorithm(
    input_path, filtered_data_sets, algorithm_comparison=False
):
    """Loads the results of the comparison between the custom prototypes optimization and the solely ML algorithm optimization.

    Args:
        input_path: from where to load the results.
        filtered_data_sets: list of the OpenML ids of the datasets.
        algorithm_comparison (optional): wheter to aggregate by algorithm or not. Defaults to False.

    Returns:
        _type_: _description_
    """
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == "json"]
    for acronym in filtered_data_sets:
        if acronym in results:
            with open(os.path.join(input_path, acronym + ".json")) as json_file:
                data = json.load(json_file)
                try:
                    accuracy = data["context"]["best_config"]["score"] // 0.0001 / 100
                    if not algorithm_comparison:
                        pipeline = (
                            str(data["context"]["best_config"]["pipeline"])
                            .replace(" ", "")
                            .replace(",", " ")
                        )
                        prototype = str(data["pipeline"]).replace(" ", "")
                        discretize = (
                            "not_in_prototype"
                            if not ("discretize" in prototype)
                            else (
                                "not_in_pipeline"
                                if data["context"]["best_config"]["pipeline"][
                                    "discretize"
                                ][0]
                                == "discretize_NoneType"
                                else "in_pipeline"
                            )
                        )
                    num_iterations = data["context"]["iteration"] + 1
                    best_iteration = data["context"]["best_config"]["iteration"] + 1
                except:
                    accuracy = 0
                    num_iterations = 0
                    best_iteration = 0

                try:
                    baseline_score = data["context"]["baseline_score"] // 0.0001 / 100
                except:
                    baseline_score = 0

        else:
            accuracy = 0
            pipeline = ""
            num_iterations = 0
            best_iteration = 0
            baseline_score = 0

        results_map[acronym] = {}
        results_map[acronym]["accuracy"] = accuracy
        results_map[acronym]["baseline_score"] = baseline_score
        results_map[acronym]["num_iterations"] = num_iterations
        results_map[acronym]["best_iteration"] = best_iteration
        if not algorithm_comparison:
            results_map[acronym]["pipeline"] = pipeline
            results_map[acronym]["prototype"] = prototype
            results_map[acronym]["discretize"] = discretize

    return results_map


def merge_custom_vs_ml_algorithm(
    auto_results, other_results, other_label, filtered_data_sets
):
    """Merges the results of the custom prototypes optimization and the solely ML algorithm optimization.

    Args:
        auto_results: results of the custom prototypes optimization.
        other_results: results of the exhaustive prototypes optimization.
        other_label: labels of the exhaustive protoypes optimization
        filtered_data_sets: list of the OpenML ids of the datasets.

    Returns:
        dict, dict: compariso of the optimization, summary of the comparison.
    """
    auto_label = "pipeline_algorithm"
    comparison = {}
    summary = {}
    for algorithm in algorithms:
        acronym = "".join([a for a in algorithm if a.isupper()]).lower()
        summary[acronym] = {auto_label: 0, other_label: 0, "draw": 0}
        comparison[acronym] = {}

    for key in filtered_data_sets:
        acronym = key.split("_")[0]
        data_set = key.split("_")[1]

        baseline_score = auto_results[key]["baseline_score"]
        if auto_results[key]["baseline_score"] != other_results[key]["baseline_score"]:
            baseline_score = (
                auto_results[key]["baseline_score"]
                if auto_results[key]["baseline_score"]
                > other_results[key]["baseline_score"]
                else other_results[key]["baseline_score"]
            )

        comparison[acronym][data_set] = {
            auto_label: auto_results[key]["accuracy"],
            other_label: other_results[key]["accuracy"],
            "baseline": baseline_score,
        }

        max_element = max(
            [
                comparison[acronym][data_set][auto_label],
                comparison[acronym][data_set][other_label],
                comparison[acronym][data_set]["baseline"],
            ]
        )
        min_element = min(
            [
                comparison[acronym][data_set][auto_label],
                comparison[acronym][data_set][other_label],
                comparison[acronym][data_set]["baseline"],
            ]
        )

        if max_element != min_element:
            other_score = (comparison[acronym][data_set][other_label] - min_element) / (
                max_element - min_element
            )
            auto_score = (comparison[acronym][data_set][auto_label] - min_element) / (
                max_element - min_element
            )
        else:
            other_score = 0
            auto_score = 0

        comparison[acronym][data_set]["a_score"] = other_score
        comparison[acronym][data_set]["pa_score"] = auto_score

        if max_element != min_element:
            comparison[acronym][data_set]["a_percentage"] = other_score / (
                other_score + auto_score
            )
            comparison[acronym][data_set]["pa_percentage"] = auto_score / (
                other_score + auto_score
            )
        else:
            comparison[acronym][data_set]["a_percentage"] = 0.5
            comparison[acronym][data_set]["pa_percentage"] = 0.5

        winner, loser = (
            (other_label, auto_label)
            if comparison[acronym][data_set][other_label]
            > comparison[acronym][data_set][auto_label]
            else (
                (auto_label, other_label)
                if comparison[acronym][data_set][other_label]
                < comparison[acronym][data_set][auto_label]
                else ("draw", "draw")
            )
        )
        summary[acronym][winner] += 1

        if winner == "draw":
            winner, loser = auto_label, other_label

    new_summary = {auto_label: 0, other_label: 0, "draw": 0}
    for algorithm, results in summary.items():
        for category, result in summary[algorithm].items():
            new_summary[category] += summary[algorithm][category]

    summary["summary"] = new_summary

    return comparison, summary


def save_custom_vs_ml_algorithm_comparison(comparison, result_path):
    """Saves comparison between the custom prototypes optimization and the solely ML algorithm optimization.

    Args:
        comparison: results of the comparison.
        result_path: where to save the results.
    """

    def values_to_string(values):
        return [str(value).replace(",", "") for value in values]

    for algorithm in algorithms:
        acronym = "".join([a for a in algorithm if a.isupper()]).lower()
        if os.path.exists("{}.csv".format(acronym)):
            os.remove("{}.csv".format(acronym))
        with open(os.path.join(result_path, "{}.csv".format(acronym)), "w") as out:
            keys = comparison[acronym][list(comparison[acronym].keys())[0]].keys()
            header = ",".join(keys)
            out.write("dataset," + header + "\n")
            for dataset, results in comparison[acronym].items():
                result_string = ",".join(values_to_string(results.values()))
                out.write(dataset + "," + result_string + "\n")


def save_custom_vs_ml_algorithm_summary(summary, result_path):
    """Saves the summary of the comparison between the custom prototypes optimization and the solely ML algorithm optimization.

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


def plot_custom_vs_ml_algorithm_comparison(comparison, result_path):
    """Plots the comparison between the custom prototypes optimization and the solely ML algorithm optimization.

    Args:
        comparison: the comparison to plot.
        result_path: where to plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    gs = gridspec.GridSpec(4, 4)
    fig = plt.figure()

    SMALL_SIZE = 8
    MEDIUM_SIZE = 17
    BIGGER_SIZE = 22

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    for i in range(0, 3):
        algorithm = algorithms[i]
        acronym = "".join([a for a in algorithm if a.isupper()]).lower()

        keys = []
        a_percentages = []
        pa_percentages = []
        for key, value in comparison[acronym].items():
            # keys.append(openml.datasets.get_dataset(key).name)
            keys.append(key)
            a_percentages.append(
                comparison[acronym][key]["a_percentage"] // 0.0001 / 100
            )
            pa_percentages.append(
                comparison[acronym][key]["pa_percentage"] // 0.0001 / 100
            )

        data = {
            "dataset": keys,
            "a_percentages": a_percentages,
            "pa_percentages": pa_percentages,
        }
        df = pd.DataFrame.from_dict(data)
        df = df.sort_values(by=["a_percentages", "pa_percentages"])

        if i == 0:
            plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        if i == 1:
            plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
        if i == 2:
            plt.subplot2grid((4, 4), (2, 1), colspan=2, rowspan=2)

        plt.bar(
            df["dataset"].tolist(),
            df["pa_percentages"].tolist(),
            label="Pre-processing and hyper-parameter optimization",
            color=(1.0, 0.5, 0.15, 1.0),
        )
        plt.bar(
            df["dataset"].tolist(),
            df["a_percentages"].tolist(),
            bottom=df["pa_percentages"].tolist(),
            label="Hyper-parameter optimization",
            color=(0.15, 0.5, 0.7, 1.0),
        )

        plt.axhline(y=50, color="#aaaaaa", linestyle="--")

        plt.xlabel("Data sets")
        plt.ylabel("Normalized improvement\npercentage")
        plt.yticks(
            ticks=np.linspace(0, 100, 11),
            labels=["{}%".format(x) for x in np.linspace(0, 100, 11)],
        )
        plt.xticks(ticks=[])
        plt.title("Approaches comparison for {}".format(algorithm))

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.98),
    )
    text = fig.text(-0.2, 20.15, "")
    fig.set_size_inches(20, 10, forward=True)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(
        os.path.join(result_path, "Figure14.pdf"),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )

    plt.clf()


def exhaustive_prototypes(toy, cache, plot):
    """Performs the exhaustive prototypes optimization analysis.

    Args:
        toy: wheter it is the toy example or not.
        plot: wheter to plot or not.
    """
    if plot:
        print("EE08. Check for the existence of a universal pipeline prototype\n")

        if not toy:
            print(
                "\tWarning: Given the huge amount of data to check, depending on your laptop, this operation might take several minutes (up to 20mins)"
            )
            print(
                "\tWe do not provide the status bar because it depends on the memory usage, do not cancel the execution\n"
            )

    # configure environment
    if toy:
        results_path = os.path.join(RAW_RESULT_PATH, "toy")
        plots_path = os.path.join(ARTIFACTS_PATH, "toy")
    elif cache == True:
        results_path = os.path.join(RAW_RESULT_PATH, "paper")
        plots_path = os.path.join(ARTIFACTS_PATH, "paper")
    else:
        results_path = os.path.join(RAW_RESULT_PATH, "paper_new")
        plots_path = os.path.join(ARTIFACTS_PATH, "paper_new")
    results_path = os.path.join(results_path, "exhaustive_prototypes")

    filtered_data_sets = [
        "_".join(i)
        for i in list(
            itertools.product(
                ["knn", "nb", "rf"],
                [
                    str(integer)
                    for integer in get_filtered_datasets("experimental_evaluation", toy)
                ],
            )
        )
    ]
    # print(filtered_data_sets)

    results = load_results_pipelines(results_path, filtered_data_sets)
    # print(results)

    winners = declare_winners(results)
    # print(winners)

    summary = summarize_winners(winners)
    # print(summary)

    results_path = create_directory(results_path, "summary")
    save_summary(summary, results_path, plots_path, plot)


def custom_vs_exhaustive(toy, cache, plot):
    """Performs the comparison between the custom and exhaustive prototypes optimization.

    Args:
        toy: wheter it is the toy example or not.
        plot: wheter to plot or not.
    """
    if plot:
        print("EE06. Compare and plot the results from EE04 and EE05\n")

        if not toy:
            print(
                "\tWarning: Given the huge amount of data to check, depending on your laptop, this operation might take several minutes (up to 20mins)"
            )
            print(
                "\tWe do not provide the status bar because it depends on the memory usage, do not cancel the execution\n"
            )

    # configure environment
    if toy:
        results_path = os.path.join(RAW_RESULT_PATH, "toy")
        plots_path = os.path.join(ARTIFACTS_PATH, "toy")
    elif cache:
        results_path = os.path.join(RAW_RESULT_PATH, "paper")
        plots_path = os.path.join(ARTIFACTS_PATH, "paper")
    else:
        results_path = os.path.join(RAW_RESULT_PATH, "paper_new")
        plots_path = os.path.join(ARTIFACTS_PATH, "paper_new")

    custom_prototypes_results_path = os.path.join(results_path, "custom_prototypes")
    custom_prototypes_pipeline_algorithm_results_path = os.path.join(
        custom_prototypes_results_path, "pipeline_algorithm"
    )
    exhaustive_prototypes_results_path = os.path.join(
        results_path, "exhaustive_prototypes"
    )
    new_results_path = create_directory(custom_prototypes_results_path, "summary")
    new_results_path = create_directory(new_results_path, "custom_vs_exhaustive")

    filtered_data_sets = [
        "_".join(i)
        for i in list(
            itertools.product(
                ["knn", "nb", "rf"],
                [
                    str(integer)
                    for integer in get_filtered_datasets("experimental_evaluation", toy)
                ],
            )
        )
    ]

    results_pipelines = load_results_pipelines(
        exhaustive_prototypes_results_path, filtered_data_sets
    )
    results_pipelines = get_winners_accuracy(results_pipelines)
    results_auto = load_results_auto(
        custom_prototypes_pipeline_algorithm_results_path, filtered_data_sets
    )

    save_comparison(results_pipelines, results_auto, new_results_path, plots_path, plot)


def custom_vs_ml_algorithm(toy, cache, plot):
    """Performs the comparison between the custom prototypes and the ML algorithm optimization.

    Args:
        toy: wheter it is the toy example or not.
        plot: wheter to plot or not.
    """
    if plot:
        print("EE03. Compare and plot the results from EE01 and EE02\n")
    else:
        print(
            "EA04. Perform exploratory analysis: prototypes versus physical pipeline\n"
        )

    if not toy:
        print(
            "\tWarning: Given the huge amount of data to check, depending on your laptop, this operation might take several minutes (up to 20mins)"
        )
        print(
            "\tWe do not provide the status bar because it depends on the memory usage, do not cancel the execution\n"
        )

    # configure environment
    if toy:
        results_path = os.path.join(RAW_RESULT_PATH, "toy")
        plots_path = os.path.join(ARTIFACTS_PATH, "toy")
    elif cache:
        results_path = os.path.join(RAW_RESULT_PATH, "paper")
        plots_path = os.path.join(ARTIFACTS_PATH, "paper")
    else:
        results_path = os.path.join(RAW_RESULT_PATH, "paper_new")
        plots_path = os.path.join(ARTIFACTS_PATH, "paper_new")
    results_path = os.path.join(results_path, "custom_prototypes")
    input_auto = os.path.join(results_path, "pipeline_algorithm")
    input_algorithm = os.path.join(results_path, "algorithm")
    results_path = create_directory(results_path, "summary")
    results_path = create_directory(results_path, "custom_vs_ml_algorithm")

    filtered_data_sets = [
        "_".join(i)
        for i in list(
            itertools.product(
                ["knn", "nb", "rf"],
                [
                    str(integer)
                    for integer in get_filtered_datasets("experimental_evaluation", toy)
                ],
            )
        )
    ]

    auto_results = load_custom_vs_ml_algorithm(
        input_auto, filtered_data_sets, algorithm_comparison=True
    )
    algorithm_results = load_custom_vs_ml_algorithm(
        input_algorithm, filtered_data_sets, algorithm_comparison=True
    )

    comparison, summary = merge_custom_vs_ml_algorithm(
        auto_results, algorithm_results, "algorithm", filtered_data_sets
    )
    # print(comparison)
    save_custom_vs_ml_algorithm_comparison(comparison, results_path)
    save_custom_vs_ml_algorithm_summary(summary, results_path)
    if plot:
        plot_custom_vs_ml_algorithm_comparison(comparison, plots_path)
