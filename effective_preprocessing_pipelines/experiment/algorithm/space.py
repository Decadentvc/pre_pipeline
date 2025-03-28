from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from .utils import generate_domain_space

algorithms = {
    'RandomForest': RandomForestClassifier,
    'SVM': SVC,
    'DecisionTree': DecisionTreeClassifier,
    'NaiveBayes': GaussianNB,
    'KNearestNeighbors': KNeighborsClassifier,
    'NeuralNet': MLPClassifier
}

# 4800
grid_random_forest = {
    "n_estimators": [10, 25, 50, 75, 100],#, 150, 200],
    "max_depth": [1, 2, 3, 4, None],
    "max_features": [1, 2, 3, None],
    "min_samples_split": [2, 3, 5],
    #"min_weight_fraction_leaf": [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    'max_leaf_nodes': [2, 3, 5, None],
    #'min_impurity_decrease': [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    "bootstrap": [True, False],
    #"oob_score": [True, False]
    "criterion": ["gini", "entropy"],
    #"class_weight": [None, "balanced"]
}

# 768
grid_svm = {
    "C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5.],
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "degree": [2, 3, 4, 5, 10, 20],
    "gamma": ['auto', 'scale'],
    "shrinking": [True, False]
}

# 4800
grid_decision_tree = {
    "max_depth": [1, 2, 3, 4, None],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "max_features": [1, 2, 3, None],
    'max_leaf_nodes': [2, 3, 5, None],
    "splitter": ['best', 'random'],
    "criterion": ["gini", "entropy"],
    #"class_weight": [None, "balanced"]
}

# 1944
grid_naive_bayes = {
    "priors": [None],
    "var_smoothing": [1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1.0],
}

# 1944
grid_knn = {
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19],
    "weights": ['uniform', 'distance'],
    "metric": ['minkowski', 'euclidean', 'manhattan'],
}

# 1944
grid_neural_network = {
    "hidden_layer_sizes": [(10,), (100,), (50,), 
    (10,) * 2, (100,) * 2, (50,) * 2,
    (10,) * 5, (100,) * 5, (50,) * 5,
    (10,) * 10, (100,) * 10, (50,) * 10,
    (10,) * 25, (100,) * 25, (50,) * 25,
    (10,) * 50, (100,) * 50, (50,) * 50,
    ],
    "activation": ['logistic', 'tanh', 'relu'],
    "solver": ['lbfgs', 'sgd', 'adam'],
    "alpha": [0.0001, 0.001, 0.01, 0.00001],
    "learning_rate": ['constant', 'invscaling', 'adaptive'],
}

parameter_grid = {
    'RandomForest': grid_random_forest,
    'SVM': grid_svm,
    'DecisionTree': grid_decision_tree,
    'NaiveBayes': grid_naive_bayes,
    'KNearestNeighbors': grid_knn,
    'NeuralNet': grid_neural_network
}


def get_domain_space(algorithm_name):
    """generates the domain space of a specific algorithm.

    Args:
        algorithm_name: algorithm name

    Returns:
        dict: the domain space.
    """
    if algorithm_name in parameter_grid.keys():
        return generate_domain_space(parameter_grid.get(algorithm_name))
    else:
        print('Invalid algorithm. Possible choices: {}'.format(
            ','.join(algorithms.keys())
        ))
        exit(1)
