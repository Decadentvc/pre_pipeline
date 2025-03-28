import logging
from statistics import mean

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.common import *

from os import listdir
from os.path import isfile, join


def load_results(input_path, filtered_datasets):
    """Loads the results for the experiment about data pre-processing impact.

    Args:
        input_path: where to load the results.
        filtered_datasets: OpenML ids of the datasets.

    Returns:
        dict: the loaded results.
    """
    result = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == 'json']
    for algorithm in algorithms:
        for dataset in filtered_datasets:
            acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
            acronym += '_' + str(dataset)
            if acronym in results:
                try:
                    with open(os.path.join(input_path, acronym + '.json')) as json_file:
                        data = json.load(json_file)
                        best_accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                        best_config = data['context']['best_config']['iteration']
                        num_iterations = data['context']['iteration'] + 1
                        baseline_score = data['context']['baseline_score'] // 0.0001 / 100
                        history = data['context']['history']

                        result[acronym] = {}
                        result[acronym]['best_accuracy'] = best_accuracy
                        result[acronym]['best_config'] = best_config
                        result[acronym]['num_iterations'] = num_iterations
                        result[acronym]['baseline_score'] = baseline_score
                        result[acronym]['history'] = history
                except:
                    pass

    return result


def find_pipeline_iterations(history):
    """Finds the number of iterations in which data pre-processing was addressed.

    Args:
        history: the whole history of iterations.

    Returns:
        int: iteration number.
    """
    for iteration in history:
        if iteration['step'] == 'algorithm':
            pass
        else:
            return iteration['iteration']


def perform_algorithm_pipeline_analysis(results, toy):
    """Summarizes the results of the optimization process and performs the analysis about the data pre-processing impact.

    Args:
        results: results of the optimization process.
        toy : whether it is the toy example or not.

    Returns:
        dict: the outcome to plot.
    """
    pipelines_iterations, algorithm_iterations = [], []

    for result in results:
        results[result]['algorithm_iterations'] = find_pipeline_iterations(
            results[result]['history'])
        results[result]['pipeline_iterations'] = results[result]['num_iterations'] - results[result]['algorithm_iterations']
        algorithm_iterations.append(results[result]['algorithm_iterations'])
        pipelines_iterations.append(results[result]['pipeline_iterations'])

    scores = {}
    half_iteration = 5 if toy else 50
    for result in results:
        try:
            acronym = result.split('_')[0]
            if not(acronym in scores):
                scores[acronym] = {}
            scores[acronym][result] = []
            scores[acronym][result].append((0, results[result]['baseline_score']))
            max_score = results[result]['baseline_score']
            for i in range(1, half_iteration+1):
                if i <= results[result]['algorithm_iterations']:
                    scores[acronym][result].append(
                        (i, results[result]['history'][i - 1]['max_history_score'] // 0.0001 / 100))
                    max_score = results[result]['history'][i - 1]['max_history_score'] // 0.0001 / 100
                else:
                    scores[acronym][result].append((i, max_score))
            for i in range(1, half_iteration+1):
                scores[acronym][result].append((half_iteration + i, results[result]['history']
                                            [results[result]['algorithm_iterations'] + i - 1]['max_history_score'] // 0.0001 / 100))
        except:
            pass
    return perform_analysis(results, scores, toy)


def perform_analysis(results, scores, toy):
    """Performs the analysis about the data pre-processing impact.

    Args:
        results: results of the optimization process.
        scores: max scores of the data pre-processing and ML algorithm phases.
        toy : whether it is the toy example or not.

    Returns:
        dict: the outcome to plot.
    """
    scores_to_kpi = {}
    outcome = {}

    max_iteration = 10 if toy else 100
    for result in results:
        try:
            for i in range(0, max_iteration+1):
                acronym = result.split('_')[0]
                if not (acronym in scores_to_kpi):
                    scores_to_kpi[acronym] = []
                    outcome[acronym] = []
                try:
                    scores_to_kpi[acronym][i].append(
                        scores[acronym][result][i][1] / results[result]['baseline_score'])
                except:
                    scores_to_kpi[acronym].append([])
                
                try:
                    scores_to_kpi[acronym][i].append(
                        scores[acronym][result][i][1] / results[result]['baseline_score'])
                except:
                    scores_to_kpi[acronym][i].append(max(scores_to_kpi[acronym])[0])
        except:
           pass

    for i in range(1, max_iteration+1):
        for key in outcome.keys():
            outcome[key].append(mean(scores_to_kpi[key][i]))

    return outcome


def save_analysis(analysis, result_path, toy):
    """Saves the outcome of the data pre-processing impact analysis.

    Args:
        analysis: outcome of the analysis.
        result_path: where to save the outcome.
        toy : whether it is the toy example or not.
    """

    max_iteration = 10 if toy else 100

    x = np.linspace(0, max_iteration, num=max_iteration)

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    try:
        plt.plot(x, analysis['nb'], label='NB', linewidth=2.5, color='lightcoral')
    except:
        pass
    
    try:
        plt.plot(x, analysis['knn'], label='KNN', linewidth=2.5, color='darkturquoise')
    except:
        pass

    try:
        plt.plot(x, analysis['rf'], label='RF', linewidth=2.5, color='violet')
    except:
        pass
    plt.xlabel('Configurations visited')
    plt.ylabel('Ratio of predictive accuracy change')
    plt.legend()
    plt.xlim(0, max_iteration)
    plt.ylim(0.8, 1.5)
    plt.axvline(x=max_iteration/2, color='#aaaaaa', linestyle='--')
    plt.grid(False)
    plt.tick_params(axis='both', which='both', length=5, color='#aaaaaa')
    if not toy:
        plt.xticks(np.linspace(0, max_iteration, int(max_iteration/10 + max_iteration/100)))
    fig = plt.gcf()
    fig.set_size_inches(12, 6, forward=True)
    fig.savefig(os.path.join(result_path, 'Figure2.pdf'))


def pipeline_impact(toy, cache):
    """Performs the analysis about the data pre-processing impact.

    Args:
        toy : whether it is the toy example or not.
    """
    logging.getLogger('matplotlib.font_manager').disabled = True
    if toy:
        path = os.path.join(RAW_RESULT_PATH, "toy")
        result_path = create_directory(ARTIFACTS_PATH, 'toy')
    elif cache:
        path = os.path.join(RAW_RESULT_PATH, "paper")
        result_path = create_directory(ARTIFACTS_PATH, 'paper')
    else:
        path = os.path.join(RAW_RESULT_PATH, "paper_new")
        result_path = create_directory(ARTIFACTS_PATH, "paper_new")
    input_path = os.path.join(path, "pipeline_impact")
    filtered_data_sets = get_filtered_datasets(experiment='pipeline_impact', toy=toy)

    pipeline_algorithm_results = load_results(input_path, filtered_data_sets)

    pipeline_algorithm_analysis = perform_algorithm_pipeline_analysis(pipeline_algorithm_results, toy)

    save_analysis(pipeline_algorithm_analysis, result_path, toy)
