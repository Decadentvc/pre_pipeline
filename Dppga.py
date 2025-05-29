import numpy as np
import optuna
from tqdm import tqdm
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    PowerTransformer, KBinsDiscretizer, Binarizer,
    OneHotEncoder
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from functools import partial
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

class ResampleWrapper(BaseEstimator):
    def __init__(self, resampler):
        self.resampler = resampler
        
    def fit(self, X, y):
        return self
    
    def fit_resample(self, X, y):
        return self.resampler.fit_resample(X, y)

class TransformWrapper(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer
        
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.transformer.transform(X)

class PrototypeSingleton:
    _instance = None
    POOL = {
        "impute": [None, SimpleImputer(), IterativeImputer()],
        "encode": [None, OneHotEncoder(sparse_output=False, handle_unknown='ignore')],
        "rebalance": [None, NearMiss(version=1), SMOTE()],
        "normalize": [
            None, StandardScaler(),
            PowerTransformer(), MinMaxScaler(), RobustScaler()
        ],
        "discretize": [None, KBinsDiscretizer(), Binarizer()],
        "features": [
            None, PCA(), SelectKBest(),
            FeatureUnion([("pca", PCA()), ("selectkbest", SelectKBest())])
        ],
    }
    
    PARAM_GRIDS = {
        "impute": {
            SimpleImputer: {"strategy": ["most_frequent"]},
            IterativeImputer: {"initial_strategy": ['most_frequent', 'constant'], "imputation_order": ['ascending', 'descending', 'roman', 'arabic', 'random']}
        },
        "encode":{OneHotEncoder:{}},
        "rebalance":{
            NearMiss:{"n_neighbors": (1, 3)},
            SMOTE:{"k_neighbors": (5, 7)}
        },
        "normalize": {
            StandardScaler: {"with_mean": [True, False], "with_std": [True, False]},
            RobustScaler: {"quantile_range":[(25.0, 75.0),(10.0, 90.0), (5.0, 95.0)],"with_centering": [True, False], "with_scaling": [True, False]}
        },
        "discretize":{
            KBinsDiscretizer:{"n_bins":[3, 5, 7],"encode": ['onehot', 'onehot-dense', 'ordinal'],"strategy": ['uniform', 'quantile', 'kmeans']},
            Binarizer:{"threshold":[0.0, 0.5, 2.0, 5.0]}
        },
        "features": {
            PCA: {"n_components":[1, 2, 3, 4]},
            SelectKBest: {"k": [1, 2, 3, 4]},
            FeatureUnion: {
                "pca__n_components": [1, 2, 3, 4],
                "selectkbest__k": [1, 2, 3, 4]
            }
        }
    }

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

class BayesianPipelineOptimizer:
    def __init__(self, steps_order, model, n_trials=50, cv=3):
        self.steps_order = steps_order
        self.model = model
        self.n_trials = n_trials
        self.cv = cv
        self.study = None
        
        self.step_types = {
            'impute': 'transformer',
            'encode': 'transformer',
            'normalize': 'transformer',
            'discretize': 'transformer',
            'rebalance': 'resampler',
            'features': 'transformer'
        }

    def _build_dynamic_pipeline(self, trial, X):
        singleton = PrototypeSingleton.get_instance()
        num_steps = []
        cat_steps = []
        main_steps = []
        param_mapping = {}

        for step_type in self.steps_order:
            op_pool = singleton.POOL[step_type]
            op_idx = trial.suggest_categorical(f"{step_type}_op", list(range(len(op_pool))))
            processor = op_pool[op_idx]
            
            if processor is None:
                continue
                
            op_class = processor.__class__
            param_grid = singleton.PARAM_GRIDS[step_type].get(op_class, {})
            params = {}
            
            for param, values in param_grid.items():
                if isinstance(values, list):
                    params[param] = trial.suggest_categorical(
                        f"{step_type}_{op_class.__name__}_{param}", 
                        values
                    )
                elif isinstance(values, tuple):
                    if all(isinstance(v, int) for v in values):
                        params[param] = trial.suggest_int(
                            f"{step_type}_{op_class.__name__}_{param}",
                            min(values),
                            max(values)
                        )
                    else:
                        params[param] = trial.suggest_float(
                            f"{step_type}_{op_class.__name__}_{param}",
                            min(values),
                            max(values)
                        )
            
            cloned_op = clone(processor)
            cloned_op.set_params(**params)
            
            if self.step_types[step_type] == 'resampler':
                wrapped_op = ResampleWrapper(cloned_op)
                main_steps.append((f"{step_type}_{op_class.__name__}", wrapped_op))
            else:
                wrapped_op = TransformWrapper(cloned_op)
                if step_type in ['impute', 'normalize', 'discretize']:
                    num_steps.append((f"{step_type}_{op_class.__name__}", wrapped_op))
                elif step_type == 'encode':
                    cat_steps.append((f"{step_type}_{op_class.__name__}", wrapped_op))
                else:
                    main_steps.append((f"{step_type}_{op_class.__name__}", wrapped_op))
            
            param_mapping[(step_type, op_class)] = params

        transformers = []
        if num_steps:
            transformers.append(('num', ImbPipeline(num_steps), list(range(X.shape[1]))))
        if cat_steps:
            transformers.append(('cat', ImbPipeline(cat_steps), list(range(X.shape[1]))))
        
        pipeline_steps = []
        if transformers:
            pipeline_steps.append(('preprocessing', ColumnTransformer(transformers, remainder='drop')))
        
        resample_steps = [s for s in main_steps if isinstance(s[1], ResampleWrapper)]
        other_steps = [s for s in main_steps if not isinstance(s[1], ResampleWrapper)]
        
        pipeline_steps += other_steps
        pipeline_steps += resample_steps
        pipeline_steps.append(('classifier', self.model))
        
        return ImbPipeline(pipeline_steps), param_mapping

    def _objective(self, trial, X, y):
        try:
            singleton = PrototypeSingleton.get_instance()
            singleton.num_features = list(range(X.shape[1]))
            singleton.cat_features = []
            
            pipeline, _ = self._build_dynamic_pipeline(trial, X)
            
            scores = cross_val_score(
                pipeline, X, y, 
                cv=self.cv, 
                scoring='accuracy',
                error_score='raise'
            )
            return np.mean(scores)
        except Exception as e:
            print(f"\n⚠️ Trial {trial.number} failed: {str(e)}")
            return float('-inf')

    def optimize(self, X, y):
        self.study = optuna.create_study(
            direction='maximize', 
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,
                multivariate=True,
                group=True
            )
        )
        objective_with_data = partial(self._objective, X=X, y=y)
        
        with tqdm(total=self.n_trials, desc="Optimizing Pipeline") as pbar:
            def update_pbar(study, trial):
                pbar.update(1)
                current_best = study.best_value if study.best_value != float('-inf') else 0.0
                pbar.set_postfix({"Best Acc": f"{current_best:.4f}"})
            
            self.study.optimize(objective_with_data, n_trials=self.n_trials, callbacks=[update_pbar])
        
        return self.study.best_params, self.study.best_value

    def format_final_result(self):
        if not self.study or self.study.best_value == float('-inf'):
            return "No valid configuration found"
            
        best_params = self.study.best_params
        config = {}
        
        for param in best_params:
            parts = param.split('_')
            step_type = parts[0]
            op_class_name = parts[1]
            
            if step_type not in config:
                config[step_type] = {'operator': None, 'params': {}}
            
            param_name = '_'.join(parts[2:])
            param_value = best_params[param]
            
            op_pool = PrototypeSingleton.POOL[step_type]
            for op in op_pool:
                if op is not None and op.__class__.__name__ == op_class_name:
                    config[step_type]['operator'] = op.__class__
                    break
                    
            config[step_type]['params'][param_name] = param_value
        
        output = []
        for step in self.steps_order:
            if step not in config or config[step]['operator'] is None:
                output.append(f"{step}: None")
                continue
                
            op_class = config[step]['operator']
            param_str = ", ".join(
                f"{k}={v}" for k, v in config[step]['params'].items() 
                if k in PrototypeSingleton.PARAM_GRIDS[step].get(op_class, {})
            )
            output.append(f"{step}: {op_class.__name__}({param_str})")
        
        return "\n".join(output)

if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_breast_cancer, fetch_openml
    from sklearn.datasets import make_moons, make_circles
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import pandas as pd

    # 新增本地数据集加载函数
    def load_local_data(file_path='./local_data.csv', 
                       target_column='target',
                       feature_columns=None):
        """
        从本地CSV文件加载数据集
        参数：
            file_path: 数据文件路径（默认当前目录的local_data.csv）
            target_column: 目标变量列名（默认'target'）
            feature_columns: 特征列列表（None表示自动选择除目标列外的所有列）
        返回：
            X, y 格式与sklearn数据集一致
        """
        try:
            df = pd.read_csv(file_path)
            
            # 自动检测特征列（如果未指定）
            if feature_columns is None:
                feature_columns = [col for col in df.columns if col != target_column]
                
            # 处理可能的缺失值（简单用中位数填充）
            # df[feature_columns] = df[feature_columns].fillna(df[feature_columns].median())
            
            return df[feature_columns].values, df[target_column].values
        except Exception as e:
            print(f"加载本地数据失败: {str(e)}")
            return None, None

    # 泰坦尼克数据集加载（原有功能保留）
    def load_titanic():
        raw = fetch_openml('titanic', version=1)
        X = raw.data[['pclass', 'age', 'sibsp', 'fare']].fillna(0).values.astype(float)
        y = (raw.target == '1').astype(int).values
        return X, y

    # 数据集配置（新增本地数据集选项）
    DATASET_CHOICE = 8  # 修改这个值切换数据集

    datasets = {
        # 内置数据集
        1: ("Breast Cancer", load_breast_cancer()),
        2: ("Titanic", load_titanic()),
        3: ("Synthetic", make_classification(
            n_samples=1000, n_features=10, 
            n_informative=5, n_classes=3)),
        4: ("Moons", make_moons(n_samples=1000, noise=0.3)),
        5: ("Circles", make_circles(n_samples=1000, noise=0.2, factor=0.5)),
        6: ("Complex1", make_classification(
            n_samples=1000, n_features=20, 
            n_informative=8, n_redundant=5,
            n_clusters_per_class=2, n_classes=5)),
        7: ("Complex2", make_classification(
            n_samples=1000, n_features=25,
            n_informative=10, n_repeated=2,
            n_classes=3, flip_y=0.3)),
        # 新增本地数据集选项
        8: ("Local Dataset", load_local_data(
            file_path='dataset_temp/abcsds_pokemon_Pokemon.csv',  # 实际路径
            target_column='label'))
    }

    # 数据加载逻辑（兼容本地和内置数据集）
    dataset_name, dataset = datasets[DATASET_CHOICE]
    
    # 处理不同数据源类型
    if isinstance(dataset, tuple):  # 处理函数返回的元组
        X, y = dataset
    elif hasattr(dataset, 'data'):  # 处理sklearn数据集对象
        X, y = dataset.data, dataset.target
    else:  # 处理其他可能的数据格式
        raise ValueError("不支持的数据集格式")

    # 有效管道原型
    steps_order_candidates = [
        ['impute', 'encode', 'normalize', 'rebalance', 'features'],
        # ['impute', 'encode', 'normalize', 'features', 'rebalance'],
        # ['impute','encode', 'rebalance', 'discretize', 'features'],
        # ['impute','encode', 'discretize', 'features', 'rebalance'],
        # ['impute','encode', 'discretize', 'rebalance', 'features']
    ]

    best_overall = {
        'accuracy': -np.inf,
        'config': None,
        'steps_order': None
    }

    for i, steps_order in enumerate(steps_order_candidates, 1):
        print(f"\n{'='*40}\nOptimizing Pipeline Config {i}: {steps_order}\n{'='*40}")
        
        optimizer = BayesianPipelineOptimizer(
            steps_order=steps_order,
            model=RandomForestClassifier(n_estimators=50, random_state=42),
            n_trials=30,
            cv=3
        )
        
        best_params, best_score = optimizer.optimize(X, y)
        
        if best_score > best_overall['accuracy']:
            best_overall['accuracy'] = best_score
            best_overall['config'] = optimizer.format_final_result()
            best_overall['steps_order'] = steps_order

    # 基准测试
    baseline = RandomForestClassifier(n_estimators=50, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_score = accuracy_score(y_test, baseline.predict(X_test))

    # 最终输出
    print(f"\n{'='*40}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print("="*40)
    print(f"Baseline Accuracy: {baseline_score:.4f}")
    
    if best_overall['accuracy'] != -np.inf:
        print(f"\n Best Optimized Accuracy: {best_overall['accuracy']:.4f}")
        print(f" From Effective pipeline prototype: {best_overall['steps_order']}")
        print("\n Best Configuration:")
        print(best_overall['config'])
    else:
        print("\n No valid configuration found in any steps_order")