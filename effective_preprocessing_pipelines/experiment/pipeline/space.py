"""Search space of each operator.
"""

def params_SimpleImputer():
    return {
        'strategy': ['most_frequent', 'constant']
    }

def params_IterativeImputer():
    return {
        'initial_strategy': ['most_frequent', 'constant'],
        'imputation_order': ['ascending', 'descending', 'roman', 'arabic', 'random']
    }

def params_OneHotEncoder():
    return {
    }

def params_OrdinalEncoder():
    return {
    }

def params_NearMiss():
    return {
        'n_neighbors': [1,2,3]
    }

def params_CondensedNearestNeighbour():
    return {
        'n_neighbors': [1,2,3]
    }

def params_SMOTE():
    return {
        'k_neighbors': [5,6,7]
    }

def params_StandardScaler():
    return {
        'with_mean': [True, False],
        'with_std': [True, False]
    }

def params_RobustScaler():
    return {
        'quantile_range':[(25.0, 75.0),(10.0, 90.0), (5.0, 95.0)],
        'with_centering': [True, False],
        'with_scaling': [True, False]
    }

def params_KBinsDiscretizer():
    return {
        'n_bins':[3, 5, 7],
        'encode': ['onehot', 'onehot-dense', 'ordinal'],
        'strategy': ['uniform', 'quantile', 'kmeans']
    }

def params_Binarizer():
    return {
        'threshold':[0.0, 0.5, 2.0, 5.0]
    }

def params_PCA():
    return {
        'n_components':[1, 2, 3, 4],
    }

def params_TruncatedSVD():
    return {
        'n_components':[1, 2, 3, 4],
    }

def params_SelectKBest():
    return {
        'k':[1, 2, 3, 4],
    }

def params_FeatureUnion():
    return {
        'pca__n_components':[1, 2, 3, 4],
        'selectkbest__k':[1, 2, 3, 4]
    }
