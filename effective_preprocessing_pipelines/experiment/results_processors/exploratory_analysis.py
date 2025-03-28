from __future__ import print_function
import copy
from os import listdir
from os.path import isfile, join

import os
import json
import itertools
import subprocess

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .experimental_evaluation import load_results_pipelines, load_results_auto
from utils.common import *


def diff(first, second):
    """Performs the differences between two lists.

    Args:
        first: first list.
        second: second list.

    Returns:
        list: difference of the two lists.
    """
    second = set(second)
    return [item for item in first if item not in second]


def transformation_analysis(custom_prototypes_results_path, new_results_path, plots_path):
    """Performs the analysis about transformations.

    Args:
        custom_prototypes_results_path: where to load the data.
        new_results_path: where to save the outcome.
        plots_path: where to plot.
    """
    # configure environment
    input_auto = os.path.join(custom_prototypes_results_path, "pipeline_algorithm")
    discretize_count, normalize_count = {}, {}
    results_map = pd.DataFrame()
    for algorithm in ["nb", "knn", "rf"]:
        discretize_count[algorithm], normalize_count[algorithm] = 0, 0
        df = pd.read_csv(os.path.join(custom_prototypes_results_path,
                         "summary", "custom_vs_ml_algorithm", algorithm + ".csv"))
        ids = list(df["dataset"])

        files = [f for f in listdir(input_auto) if isfile(join(input_auto, f))]
        results = [f[:-5] for f in files if f[-4:] == 'json']

        for dataset in ids:
            acronym = algorithm + "_" + str(dataset)
            if acronym in results:
                with open(os.path.join(input_auto, acronym + '.json')) as json_file:
                    data = json.load(json_file)
                    try:
                        pipeline = data['context']['best_config']['pipeline']
                    except:
                        pass
                    for transformation in ["encode", "features", "impute", "normalize", "discretize", "rebalance"]:
                        try:
                            results_map = results_map.append(pd.DataFrame({
                                "algorithm": [algorithm],
                                "dataset": [dataset],
                                "transformation": [pipeline[transformation][0].split("_")[0]],
                                "operator": [pipeline[transformation][0].split("_")[1]]
                            }), ignore_index=True)
                        except:
                            pass

    results_map.to_csv(os.path.join(
        new_results_path, "transformations.csv"), index=False)

    result = results_map.groupby(
        ['algorithm', 'transformation', 'operator']).count()
    result_sum = result.groupby(['algorithm', 'transformation']).sum()
    for algorithm in result.index.get_level_values('algorithm').unique():
        for transformation in result.index.get_level_values('transformation').unique():
            for operator in result.index.get_level_values('operator').unique():
                try:
                    result.loc[algorithm, transformation,
                               operator] /= result_sum.loc[algorithm, transformation]
                except:
                    pass
    result = result.reset_index()
    result = result[result['operator'] != 'NoneType']
    result = result.set_index(['transformation', 'operator', 'algorithm'])

    labels = ["NB", "KNN", "RF"]
    colors = [
        'mediumpurple', 
        'xkcd:dark grass green', 
        'xkcd:kermit green', 
        'xkcd:lime green', 
        'xkcd:light pea green', 
        'xkcd:dark coral', 
        'xkcd:salmon',
        'xkcd:sun yellow', 
        'xkcd:straw', 
        'xkcd:aqua green', 
        'xkcd:light aquamarine', 
        'xkcd:pumpkin orange', 
        'xkcd:apricot', 
        'xkcd:light peach']
    x = np.arange(len(labels))  # the label locations
    width = 0.125  # the width of the bars
    MEDIUM_SIZE = 15

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots()
    column = -1
    i = 0
    cumulative = False
    last_transformation, last_operator = '', ''
    transformation_map = {
        'encode': 0,
        'normalize': 1,
        'discretize': 5,
        'impute': 7,
        'rebalance': 9,
        'features': 11
    }
    for transformation, i in transformation_map.items():
        for operator in result.index.get_level_values('operator').unique():
            try:
                result.loc[transformation, operator]
                flag = True
            except:
                flag = False
            if flag:
                curr_bar = []
                for algo in ["nb", "knn", "rf"]:
                    try:
                        c = result.loc[transformation, operator, algo].to_numpy().flatten()[
                            0]
                    except:
                        c = 0
                    curr_bar.append(c)
                if transformation != last_transformation or last_transformation == '':
                    column += 1
                    ax.bar((x * width * 8) + (width * (column - 1)) - 0.2, curr_bar, width,
                           color=colors[i], label=transformation[0].upper() + "  " + (" " if transformation[0].upper() == "I" else "") + operator)
                else:
                    if not(cumulative):
                        last_bar = []
                        for algo in ["nb", "knn", "rf"]:
                            try:
                                c = result.loc[last_transformation, last_operator, algo].to_numpy().flatten()[
                                    0]
                            except:
                                c = 0
                            last_bar.append(c)
                    ax.bar((x * width * 8) + (width * (column - 1)) - 0.2, curr_bar, width, bottom=last_bar,
                           color=colors[i], label=transformation[0].upper() + "  " + (" " if transformation[0].upper() == "I" else "") + operator)
                if transformation == last_transformation:
                    cumulative = True
                    last_bar = [curr_bar[0] + last_bar[0], curr_bar[1] +
                                last_bar[1], curr_bar[2] + last_bar[2]]
                else:
                    cumulative = False
                last_transformation, last_operator = transformation, operator
                i += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Usage')
    ax.set_xlabel('Algorithms')
    ax.set_yticks(np.linspace(0, 1, 11))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3, bbox_to_anchor=(0.55, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.set_size_inches(10, 5)
    fig.savefig(os.path.join(plots_path, "Figure9.pdf"), bbox_extra_artists=(lgd, text), bbox_inches='tight')


def physical_pipelines_analysis(custom_prototypes_results_path, new_results_path, plots_path):
    """Performs the analysis about the physical pipelines.

    Args:
        custom_prototypes_results_path: where to load the data.
        new_results_path: where to save the outcome.
        plots_path: where to plot.
    """
    # configure environment
    input_auto = os.path.join(custom_prototypes_results_path, "pipeline_algorithm")
    results_map = pd.DataFrame()
    for algorithm in ["knn", "nb", "rf"]:

        df = pd.read_csv(os.path.join(custom_prototypes_results_path,
                         "summary", "custom_vs_ml_algorithm", algorithm + ".csv"))
        ids = list(df["dataset"])

        files = [f for f in listdir(input_auto) if isfile(join(input_auto, f))]
        results = [f[:-5] for f in files if f[-4:] == 'json']

        for dataset in ids:
            acronym = algorithm + "_" + str(dataset)
            if acronym in results:
                with open(os.path.join(input_auto, acronym + '.json')) as json_file:
                    data = json.load(json_file)
                    try:
                        accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                        pipeline = data['context']['best_config']['pipeline']
                        pipeline_conf = data['pipeline']
                        pipeline_conf = ''.join([a[0] for a in pipeline_conf]).upper()

                        encode_flag = "None" in pipeline["encode"][0]
                        features_flag = "None" in pipeline["features"][0]
                        impute_flag = "None" in pipeline["impute"][0]
                        try:
                            normalize_flag = "None" in pipeline["normalize"][0]
                        except:
                            normalize_flag = True
                        try:
                            discretize_flag = "None" in pipeline["discretize"][0]
                        except:
                            discretize_flag = True
                        rebalance_flag = "None" in pipeline["rebalance"][0]
                        if encode_flag or not(encode_flag):
                            pipeline_conf = pipeline_conf.replace("E", "")
                        if features_flag:
                            pipeline_conf = pipeline_conf.replace("F", "")
                        if impute_flag or not(impute_flag):
                            pipeline_conf = pipeline_conf.replace("I", "")
                        if normalize_flag:
                            pipeline_conf = pipeline_conf.replace("N", "")
                        if discretize_flag:
                            pipeline_conf = pipeline_conf.replace("D", "")
                        if rebalance_flag:
                            pipeline_conf = pipeline_conf.replace("R", "")
                    except:
                        pass

                    results_map = results_map.append(pd.DataFrame(
                        {"algorithm": [algorithm], "pipeline": [pipeline_conf]}), ignore_index=True)

    results_map.to_csv(os.path.join(new_results_path, "physical_pipelines.csv"), index=False)
    results_map = results_map.pivot_table(index='algorithm', columns='pipeline', aggfunc=len, fill_value=0)
    results_map["sum"] = results_map.sum(axis=1)
    results_map = results_map.div(results_map["sum"], axis=0)
    results_map = results_map.drop(['sum'], axis=1)
    results_map = results_map.reset_index()
    results_map = results_map.set_index(["algorithm"])
    results_map = results_map.reindex(["nb", "knn", "rf"])
    results_map = results_map.reset_index()
    results_map.to_csv(os.path.join(new_results_path, "physical_pipelines_pivoted.csv"), index=False)
    results_map = results_map.rename(columns={
        'DF': r'$D \to F$',
        'RF': r'$R \to F$',
        'D': r'$D$',
        'N': r'$N$',
        'R': r'$R$',
        'DFR': r'$D \to F \to R$',
        'DR': r'$D \to R$',
        'DRF': r'$D \to R \to F$',
        'NFR': r'$N \to F \to R$',
        'NRF': r'$N \to R \to F$',
        'RD': r'$R \to D$',
        'DF': r'$D \to F$',
        'DR': r'$D \to R$',
        'FR': r'$F \to R$',
        'NF': r'$N \to F$',
        'NR': r'$N \to R$',
        'RDF': r'$R \to D \to F$',
    })
    # print(results_map)
    labels = [x.upper() for x in results_map["algorithm"]]
    patterns = ['/', '\\', '-\\', '-', '+', 'x', 'o',
                '//', '\\\\', 'O.', '--', '++', 'xx', 'OO', '\\|']
    colors = ['skyblue', 'orange', 'lightgreen', 'tomato', 'mediumorchid', 'xkcd:medium brown', 'xkcd:pale pink', 'xkcd:greyish',
              'xkcd:aquamarine', 'xkcd:dodger blue', 'xkcd:sun yellow', 'xkcd:flat green', 'xkcd:red orange', 'xkcd:lighter purple', 'xkcd:silver']
    x = np.arange(len(labels))  # the label locations
    width = 0.125  # the width of the bars
    MEDIUM_SIZE = 12
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots()
    rects = {}
    columns = list(results_map.columns)[2:]
    for column in range(len(columns)):
        ax.bar((x * width * 20) + (width * (column)),
               results_map[columns[column]], width, label=columns[column], color=colors[column], hatch=patterns[column])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Usage')
    ax.set_xlabel('Algorithms')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_xticks(x*2.5+0.88)
    ax.set_xticklabels(labels)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, bbox_to_anchor=(0.55, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.set_size_inches(10, 5)
    fig.savefig(os.path.join(plots_path, "Figure8.pdf"), bbox_extra_artists=(lgd, text), bbox_inches='tight')


def prototypes_impact_analysis(exhaustive_prototypes_results_path, custom_prototypes_results_path, plots_path, toy):
    """Performs the analysis about the impact of the prototypes.

    Args:
        exhaustive_prototypes_results_path: where to load the data about the exhaustive prototypes optimization.
        custom_prototypes_results_path: where to load the data about the custom prototypes optimization.
        plots_path: where to plot.
        toy: whether it is the toy example or not
    """
    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(
        ["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets("exploratory_analysis", toy)]))]

    results_pipelines = load_results_pipelines(exhaustive_prototypes_results_path, filtered_data_sets)

    custom_prototypes_pipeline_algorithm_results_path = os.path.join(
        custom_prototypes_results_path, "pipeline_algorithm")
    results_auto = load_results_auto(custom_prototypes_pipeline_algorithm_results_path, filtered_data_sets)
    
    for algorithm in results_auto.keys():
        for dataset in results_auto[algorithm].keys():
            if results_auto[algorithm][dataset][1] == 0:
                evaluation_3 = pd.read_csv(os.path.join(
                    custom_prototypes_results_path, "summary", "custom_vs_ml_algorithm", algorithm + ".csv"))
                evaluation_3 = evaluation_3.set_index(['dataset'])
                results_auto[algorithm][dataset] = (
                    results_auto[algorithm][dataset][0], evaluation_3.loc[int(dataset)]["baseline"])
                    
    impacts = pd.DataFrame()
    for algorithm in results_auto.keys():
        for dataset in results_auto[algorithm].keys():
            try:
                for elem in results_pipelines[algorithm][dataset]:
                    impacts = impacts.append({'algorithm': algorithm, 'dataset': dataset, 'index': int(
                        elem['index']) + 1, 'impact': elem['accuracy']-results_auto[algorithm][dataset][1]}, ignore_index=True)
            except:
                pass
            
    knn = impacts[impacts["algorithm"] == "knn"]
    nb = impacts[impacts["algorithm"] == "nb"]
    rf = impacts[impacts["algorithm"] == "rf"]
    knn = knn.pivot(index='dataset', columns='index', values='impact')
    nb = nb.pivot(index='dataset', columns='index', values='impact')
    rf = rf.pivot(index='dataset', columns='index', values='impact')
    data = [nb, knn, rf]
    fig, ax = plt.subplots(1, 3, sharey=True, constrained_layout=True)
    for i in range(len(data)):
        ax[i].boxplot(data[i], showfliers=False)
        ax[i].set_title("NB" if i == 0 else ("KNN" if i == 1 else "RF"))
        ax[i].set_xlabel('Prototypes ID')
        ax[i].set_xticklabels(range(1, 25), fontsize=8)
    fig.text(0.0, 0.5, 'Impact over the baseline',
             va='center', rotation='vertical')
    fig.set_size_inches(12, 3)
    fig.savefig(os.path.join(plots_path, "Figure7.pdf"))


def get_paths(toy, cache):
    """Gets all the paths for the exploratory analysis.

    Args:
        toy: whether it is the toy example or not.

    Returns:
        string: path to the exhaustive prototypes optimization results.
        string: path to the custom prototypes optimization results.
        string: where to plot.
        string: where to save the outcome.
    """
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
    exhaustive_prototypes_results_path = os.path.join(results_path, "exhaustive_prototypes")
    new_results_path = create_directory(results_path, "exploratory_analysis")
    return exhaustive_prototypes_results_path, custom_prototypes_results_path, plots_path, new_results_path


def meta_learning_input_preparation(results_path, custom_prototypes_results_path):
    """Prepares the input for the meta-learning process.

    Args:
        results_path: where to save the outcome.
        custom_prototypes_results_path: path to the custom prototypes optimization results.
    """
    custom_prototypes_pipeline_algorithm_results_path = os.path.join(
        custom_prototypes_results_path, "pipeline_algorithm")
    custom_vs_ml_algorithm_summary_results_path = os.path.join(
        custom_prototypes_results_path, "summary", "custom_vs_ml_algorithm")

    # Meta-features Loading
    all_classification = pd.read_csv(os.path.join(META_FEATURES_PATH, "extended_meta_features_all_classification.csv"))
    openml_cc_18 = pd.read_csv(os.path.join(META_FEATURES_PATH, "extended_meta_features_openml_cc_18.csv"))
    study_1 = pd.read_csv(os.path.join(META_FEATURES_PATH, "extended_meta_features_study_1.csv"))
    all_classification = all_classification[all_classification['ID'].isin(diff(all_classification['ID'], openml_cc_18['ID']))]
    meta_features = pd.concat([openml_cc_18, all_classification, study_1], ignore_index=True, sort=True)
    algorithm_acronyms = ["".join([c for c in algorithm if c.isupper()]).lower() for algorithm in algorithms]

    baseline_results = {}
    # Results Loading
    for algorithm in algorithm_acronyms:
        baseline_results[algorithm] = pd.read_csv(os.path.join(
            custom_vs_ml_algorithm_summary_results_path, algorithm + '.csv'))
        baseline_results[algorithm].rename(
            columns={"dataset": "ID"}, inplace=True)

    results_map = {}
    for algorithm in algorithm_acronyms:
        results_map[algorithm] = pd.DataFrame()
        df = pd.read_csv(os.path.join(
            custom_vs_ml_algorithm_summary_results_path, algorithm + ".csv"))
        ids = list(df["dataset"])

        files = [f for f in listdir(custom_prototypes_pipeline_algorithm_results_path) if isfile(
            join(custom_prototypes_pipeline_algorithm_results_path, f))]
        results = [f[:-5] for f in files if f[-4:] == 'json']

        for dataset in ids:
            acronym = algorithm + "_" + str(dataset)
            if acronym in results:
                with open(os.path.join(custom_prototypes_pipeline_algorithm_results_path, acronym + '.json')) as json_file:
                    data = json.load(json_file)

                    try:
                        pipeline = data['context']['best_config']['pipeline']
                    except:
                        pass

                    encode_flag = pipeline["encode"][0].split("_", 1)[1]
                    features_flag = pipeline["features"][0].split("_", 1)[1]
                    impute_flag = pipeline["impute"][0].split("_", 1)[1]
                    try:
                        normalize_flag = pipeline["normalize"][0].split("_", 1)[1]
                    except:
                        normalize_flag = "NoneType"
                    try:
                        discretize_flag = pipeline["discretize"][0].split("_", 1)[1]
                    except:
                        discretize_flag = "NoneType"
                    rebalance_flag = pipeline["rebalance"][0].split("_", 1)[1]

                    results_map[algorithm] = results_map[algorithm].append(pd.DataFrame({
                        "ID": [dataset],
                        "baseline": [baseline_results[algorithm].loc[baseline_results[algorithm]["ID"] == dataset, "baseline"].iloc[0]],
                        "encode": [encode_flag],
                        "features": [features_flag],
                        "impute": [impute_flag],
                        "normalize": [normalize_flag],
                        "discretize": [discretize_flag],
                        "rebalance": [rebalance_flag]
                    }), ignore_index=True)

    for algorithm in algorithm_acronyms:
        results_map[algorithm] = pd.merge(
            results_map[algorithm], meta_features, on="ID")

    data = copy.deepcopy(results_map)

    # Data Preparation
    manual_fs_data = data.copy()

    for algorithm in algorithm_acronyms:
        data[algorithm]["algorithm"] = algorithm
        manual_fs_data[algorithm]["algorithm"] = algorithm
    union = pd.concat([data["knn"], data["nb"], data["rf"]], ignore_index=True)
    manual_fs_union = pd.concat(
        [manual_fs_data["knn"], manual_fs_data["nb"], manual_fs_data["rf"]], ignore_index=True)

    # Data Saving
    manual_fs_union.to_csv(os.path.join(
        results_path, 'meta_learning_input' + '.csv'), index=False)


def run_meta_learning(toy, cache):
    """Performs meta-learning.

    Args:
        toy: whether it is the toy example or not.
    """
    
    experiment = "toy" if toy else ("paper" if cache else "paper_new")
    subprocess.call(
        f"Rscript experiment/results_processors/meta_learner.R {experiment}", 
        shell=True, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL)


def exploratory_analysis(toy_example, cache):
    """Performs the whole exploratory analysis.

    Args:
        toy: whether it is the toy example or not.
    """
    exhaustive_prototypes_results_path, custom_prototypes_results_path, plots_path, new_results_path = get_paths(toy_example, cache)

    print("EA05. Plot results\n")
    if not toy_example:
        print("\tWarning: Given the huge amount of data to check, depending on your laptop, this operation might take several minutes (up to 20mins)")
        print("\tWe do not provide the status bar because it depends on the memory usage, do not cancel the execution\n")
    prototypes_impact_analysis(exhaustive_prototypes_results_path, custom_prototypes_results_path, plots_path, toy_example)
    transformation_analysis(custom_prototypes_results_path, new_results_path, plots_path)
    physical_pipelines_analysis(custom_prototypes_results_path, new_results_path, plots_path)
    print("EA06. Extract meta-features from datasets\n")
    meta_learning_input_preparation(new_results_path, custom_prototypes_results_path)
    print("EA07. Perform meta-learning\n")
    run_meta_learning(toy_example, cache)
    print("EA08. Plot and check the significance of the results\n")