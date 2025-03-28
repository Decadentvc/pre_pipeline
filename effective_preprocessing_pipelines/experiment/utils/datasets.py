from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_covtype
from .common import *


def load(name):
    """Load a datset given the name.

    Args:
        name: name of the dataset.

    Returns:
        numpy.array: data itemsof the dataset.
        numpy.array: target of the dataset.
    """
    loader = {
        'breast': breast_cancer,
        'iris': iris,
        'wine': wine,
        'digits': digits,
        'covtype': covtype,
    }
    if name in loader:
        return loader[name]()
    else:
        print('Invalid dataset. Possible choices: {}'.format(
            ', '.join(loader.keys())
        ))
        exit(1)  # TODO: Throw exception

def breast_cancer():
    """Loade the breast cancer dataset.

    Returns:
        numpy.array: data items (data.data) of the dataset.
        numpy.array: target (data.target) of the dataset.
    """
    data = load_breast_cancer()
    return data.data, data.target

def iris():
    """Loade the iris dataset.

    Returns:
        numpy.array: data items (data.data) of the dataset.
        numpy.array: target (data.target) of the dataset.
    """
    data = load_iris()
    return data.data, data.target

def wine():
    """Loade the wine dataset.

    Returns:
        numpy.array: data items (data.data) of the dataset.
        numpy.array: target (data.target) of the dataset.
    """
    data = load_wine()
    return data.data, data.target

def digits():
    """Loade the digits dataset.

    Returns:
        numpy.array: data items (data.data) of the dataset.
        numpy.array: target (data.target) of the dataset.
    """
    digits = load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target

def covtype():
    """Loade the covtype dataset.

    Returns:
        numpy.array: data items (data.data) of the dataset.
        numpy.array: target (data.target) of the dataset.
    """
    data = fetch_covtype
    return data.data, data.target

def load_from_openml(id):
    """Load a dataset given its id on OpenML.

    Args:
        id: id of the dataset.

    Returns:
        numpy.array: data items (X) of the dataset.
        numpy.array: target (y) of the dataset.
        list: mask that indicates categorical features.
    """
    import openml
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )
    return X, y, categorical_indicator

def load_from_csv(id, input_path = DATASETS_PATH):
    """Load a dataset given its id on OpenML from resources/datasets.

    Args:
        id: id of the dataset.

    Returns:
        numpy.array: data items (X) of the dataset.
        numpy.array: target (y) of the dataset.
        list: mask that indicates categorical features.
    """
    import pandas as pd
    import json
    df = pd.read_csv(os.path.join(input_path, f"{id}.csv"))
    with open(os.path.join(input_path, "categorical_indicators.json")) as f:
        categorical_indicators = json.load(f)
    categorical_indicator = categorical_indicators[str(id)]   
    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    return X, y, categorical_indicator
