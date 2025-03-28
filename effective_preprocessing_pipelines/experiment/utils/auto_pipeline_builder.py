import numpy as np
import openml
import os
import pandas as pd

from enum import Enum


class UndefinedOrders(Enum):
    """Transformation pairs with an undefined order.

    Args:
        Enum: name of the transformation pair.
    """
    features_rebalance = 1
    discretize_rebalance = 2


class DefinedOrders(Enum):
    """Transformation pairs with a defined order.

    Args:
        Enum: name of the transformation pair.
    """
    first_second = 1
    second_first = 2


def load_metafeatures(id):
    """Load the meta-features of a specific dataset from resources/metafeatures.

    Args:
        id: OpenML id of the dataset.

    Returns:
        pandas.DataFrame: meta-features of the dataset.
    """
    df = pd.read_csv("meta_features/extracted-meta-features.csv")
    df = df.loc[df["id"] == id]
    df = df.drop(["id"], axis=1)
    df = df.fillna("na")
    return df


def build_pipeline(features_rebalance_order, discretize_rebalance_order):
    """Build a pipeline given two pre-defined orders of transformations.

    Args:
        features_rebalance_order: order of the features-rebalance pair.
        discretize_rebalance_order: order of the discretize-rebalance pair.

    Returns:
        list: the built pipeline.
    """
    pipelines = []
    if features_rebalance_order == DefinedOrders.first_second:
        pipelines.append("impute encode normalize features rebalance")
    else:
        pipelines.append("impute encode normalize rebalance features")

    pipeline = "impute encode "

    if (
        features_rebalance_order == DefinedOrders.first_second
        and discretize_rebalance_order == DefinedOrders.second_first
    ):
        pipelines.append(pipeline + "rebalance discretize features")
        pipelines.append(pipeline + "discretize features rebalance")
    else:
        if (
            features_rebalance_order == DefinedOrders.first_second
            and discretize_rebalance_order == DefinedOrders.first_second
        ):
            pipeline += "discretize features rebalance"
        elif (
            features_rebalance_order == DefinedOrders.second_first
            and discretize_rebalance_order == DefinedOrders.first_second
        ):
            pipeline += "discretize rebalance features"
        elif (
            features_rebalance_order == DefinedOrders.second_first
            and discretize_rebalance_order == DefinedOrders.second_first
        ):
            pipeline += "rebalance discretize features"
        pipelines.append(pipeline)

    return pipelines


def framework_table_pipelines():
    """Build all the possible pipeline in scikit-learn.

    Returns:
        list of list: list of the pipelines.
    """
    from itertools import permutations

    all_perm = permutations(
        ["impute", "encode", "normalize", "discretize", "features", "rebalance"]
    )
    valid_perm = []
    pipelines = []

    for i in list(all_perm):
        if (
            i.index("encode") < i.index("normalize")
            and i.index("discretize") > i.index("encode")
            and i.index("impute") < i.index("encode")
            and i.index("impute") < i.index("discretize")
            and i.index("rebalance") > i.index("encode")
            and i.index("rebalance") > i.index("impute")
            and i.index("features") > i.index("encode")
            and i.index("features") > i.index("impute")
        ):
            valid_perm.append(i)

    for i in list(valid_perm):
        pipelines.append(" ".join(i))

    return pipelines


def pseudo_exhaustive_pipelines():
    """Build the customize pipelines.

    Returns:
        list of list: list of the pipelines.
    """
    pipelines = []

    pipelines.append("impute encode normalize features rebalance")
    pipelines.append("impute encode normalize rebalance features")

    pipeline = "impute encode "
    pipelines.append(pipeline + "rebalance discretize features")
    pipelines.append(pipeline + "discretize rebalance features")
    pipelines.append(pipeline + "discretize features rebalance")

    return pipelines
