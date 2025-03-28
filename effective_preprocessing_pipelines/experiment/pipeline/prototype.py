from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer, KBinsDiscretizer, \
    Binarizer, OneHotEncoder, OrdinalEncoder, FunctionTransformer

from imblearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion

from .PrototypeSingleton import PrototypeSingleton


def get_baseline():
    """Gets the signature of the baseline.
    """
    baseline = {}
    for k in PrototypeSingleton.getInstance().getPrototype().keys():
        baseline[k] = ('{}_NoneType'.format(k), {})
    return baseline

def pipeline_conf_to_full_pipeline(args, algorithm, seed, algo_config):
    """Gets the pipeline configuration.

    Args:
        args: config of the prototype.
        algorithm: legacy.
        seed: seed for reproducibility.
        algo_config: config of the algorithm.

    Returns:
        imblearn.Pipeline: instantiated pipeline.
    """
    if args == {}:
        args = get_baseline()
    op_to_class = {'pca': PCA, 'selectkbest': SelectKBest}
    operators = []
    for part in PrototypeSingleton.getInstance().getParts():
        item = args[part]
        #print(item, PrototypeSingleton.getInstance().getFeatures())
        if 'NoneType' in item[0]:
            continue
        else:
            params =  {k.split('__', 1)[-1]:v for k,v in item[1].items()}
            transformation_param = item[0].split('_',1)[0]
            operator_param = item[0].split('_', 1)[-1]
            if transformation_param == 'features' and operator_param == 'FeatureUnion':
                fparams = {'pca':{}, 'selectkbest':{}}
                for p,v in params.items():
                    op = p.split('__')[0]
                    pa = p.split('__')[1]
                    if op not in fparams:
                        fparams[op] = {}
                    fparams[op][pa] = v
                oparams = []
                for p,v in fparams.items():
                    oparams.append((p, op_to_class[p](**v)))
                operator = FeatureUnion(oparams)
            else:
                if 'random_state' in globals()[operator_param]().get_params():
                    operator = globals()[operator_param](random_state=seed, **params)
                else:
                    operator = globals()[operator_param](**params)
                if transformation_param == 'encode':
                    numerical_features, categorical_features = PrototypeSingleton.getInstance().getFeatures()
                    operator = ColumnTransformer(
                        transformers=[
                            ('num', Pipeline(steps=[('identity_numerical', FunctionTransformer())]),
                            numerical_features),
                            ('cat', Pipeline(steps=[('encoding', operator)]),
                            categorical_features)])
                    if operator_param == 'OneHotEncoder':
                        PrototypeSingleton.getInstance().applyOneHotEncoding()
                    else:
                        PrototypeSingleton.getInstance().applyColumnTransformer()
                elif transformation_param == 'normalize':
                    numerical_features, categorical_features = PrototypeSingleton.getInstance().getFeatures()
                    operator = ColumnTransformer(
                        transformers=[
                            ('num', Pipeline(steps=[('normalizing', operator)]),
                            numerical_features),
                            ('cat', Pipeline(steps=[('identity_categorical', FunctionTransformer())]),
                            categorical_features)])
                    PrototypeSingleton.getInstance().applyColumnTransformer()
                elif transformation_param == 'discretize':
                    numerical_features, categorical_features = PrototypeSingleton.getInstance().getFeatures()
                    operator = ColumnTransformer(
                        transformers=[
                            ('num', Pipeline(steps=[('discretizing', operator)]),
                            numerical_features),
                            ('cat', Pipeline(steps=[('identity', FunctionTransformer())]),
                            categorical_features)])
                    PrototypeSingleton.getInstance().applyDiscretization()
                operators.append((part, operator))

    PrototypeSingleton.getInstance().resetFeatures()
    if 'random_state' in algorithm().get_params():
        clf = algorithm(random_state=seed, **algo_config)
    else:
        clf = algorithm(**algo_config)
    return Pipeline(operators + [("classifier", clf)]), operators