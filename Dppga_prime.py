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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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
    def __init__(self, steps_order, model="RF", n_trials=50, cv=3):
        self.steps_order = steps_order
        self.model_type = model
        self.n_trials = n_trials
        self.cv = cv
        self.study = None
        
        self.model_config = {
            "RF": RandomForestClassifier(n_estimators=50, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "NB": GaussianNB()
        }

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

        model_instance = self.model_config.get(self.model_type, 
                                             self.model_config["RF"])    

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
        pipeline_steps.append(('classifier', model_instance))
        
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
        
        model_name = self.model_type
        if model_name in self.model_config:
            model_class = type(self.model_config[model_name]).__name__
            output.append(f"\nmodel: {model_class} (fixed parameters)")

        return "\n".join(output)

if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_breast_cancer, fetch_openml
    from sklearn.datasets import make_moons, make_circles
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import pandas as pd

    # 新增本地数据集加载函数
    def load_local_data(file_path='./local_data.csv', 
                    target_column='label',
                    feature_columns=None):
        """
        从本地CSV文件加载数据集并自动处理字符串特征
        
        参数：
            file_path: 数据文件路径
            target_column: 目标变量列名
            feature_columns: 特征列列表（None表示自动选择除目标列外的所有列）
            
        返回：
            X, y 格式与sklearn数据集一致
        """
        try:
            # 加载数据
            df = pd.read_csv(file_path)
            
            # 检查目标列是否存在
            if target_column not in df.columns:
                raise ValueError(f"目标列 '{target_column}' 不存在于数据集中")
            
            # 自动检测特征列
            if feature_columns is None:
                feature_columns = [col for col in df.columns if col != target_column]
            elif target_column in feature_columns:
                feature_columns.remove(target_column)
            
            # 复制原始数据以避免修改原始DataFrame
            df_processed = df.copy()
            
            # 处理字符串特征：将非数值特征转换为数值
            label_encoders = {}
            for col in feature_columns:
                if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
                    # 使用LabelEncoder转换字符串特征
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
            
            # 处理目标变量中的字符串
            if df[target_column].dtype == 'object' or isinstance(df[target_column].dtype, pd.CategoricalDtype):
                le_target = LabelEncoder()
                df_processed[target_column] = le_target.fit_transform(df[target_column].astype(str))
                print(f"目标列已编码为: {le_target.classes_}")
            
            # 填充缺失值（只处理特征列）
            for col in feature_columns:
                # 数值列用中位数填充
                if np.issubdtype(df_processed[col].dtype, np.number):
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                # 分类列用众数填充
                else:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            
            # 转换为numpy数组
            X = df_processed[feature_columns].values.astype(np.float32)
            y = df_processed[target_column].values
            
            # 验证数据维度
            print(f"加载数据集成功: {file_path}")
            print(f"特征形状: {X.shape}, 目标形状: {y.shape}")
            print(f"特征类型: {type(X[0,0])}, 目标类型: {type(y[0])}")
            print(f"缺失值统计 - 特征: {np.isnan(X).sum()}, 目标: {np.isnan(y).sum()}")
            
            return X, y
        except Exception as e:
            print(f"加载本地数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    # 泰坦尼克数据集加载
    def load_titanic():
        raw = fetch_openml('titanic', version=1)
        X = raw.data[['pclass', 'age', 'sibsp', 'fare']].fillna(0).values.astype(float)
        y = (raw.target == '1').astype(int).values
        return X, y

    # 数据集配置
    DATASET_CHOICE = 1  # 修改这个值切换数据集

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
        # 本地数据集选项
        8: ("Local Dataset", None)
    }

    # 数据加载逻辑
    if DATASET_CHOICE == 8:  # 本地数据集
        print("\n正在加载本地数据集...")
        X, y = load_local_data(
            file_path='Haipipe/data/dataset/primaryobjects_voicegender/voice.csv',
            target_column='label'
        )
        
        # 检查数据有效性
        if X is None or y is None:
            print("加载本地数据集失败，请检查路径和格式")
            exit()
        
        # 强制转换为float类型
        try:
            X = X.astype(np.float32)
            y = y.astype(np.float32).astype(int)  # 目标需要是整数
        except Exception as e:
            print(f"转换数据类型失败: {str(e)}")
            exit()
            
        dataset_name = "Local Dataset"
        # 添加额外的调试信息
        print("数据格式检查:")
        print(f"X类型: {type(X)}, y类型: {type(y)}")
        print(f"X形状: {X.shape}, y形状: {y.shape}")
        print(f"X示例: {X[:3]}, y示例: {y[:3]}")
        print(f"NaN值统计 - X: {np.isnan(X).sum()}, y: {np.isnan(y).sum()}")
        print(f"唯一值计数 (y): {np.unique(y, return_counts=True)}")
    else:
        # 原有内置数据集的加载逻辑保持不变
        dataset_name, dataset = datasets[DATASET_CHOICE]
        if isinstance(dataset, tuple):
            X, y = dataset
        elif hasattr(dataset, 'data'):
            X, y = dataset.data, dataset.target

    # 有效管道原型
    steps_order_candidates = [
        # ['impute', 'encode', 'normalize', 'rebalance', 'features'],
        # ['impute', 'encode', 'normalize', 'features', 'rebalance'],
        # ['impute','encode', 'rebalance', 'discretize', 'features'],
        # ['impute','encode', 'discretize', 'features', 'rebalance'],
        # ['impute','encode', 'discretize', 'rebalance', 'features'],
        ['impute']
    ]

    model_choices = ["RF"] #["RF", "SVM", "KNN", "NB"] 
    best_overall = {
        'accuracy': -np.inf,
        'config': None,
        'steps_order': None,
        'model_type': None
    }

    for model_choice in model_choices:  # 遍历模型类型
        for i, steps_order in enumerate(steps_order_candidates, 1):
            print(f"\n{'='*40}")
            print(f"Optimizing Pipeline Config {i} with {model_choice}:")
            print(f"Steps: {steps_order}\n{'='*40}")
            
            optimizer = BayesianPipelineOptimizer(
                steps_order=steps_order,
                model=model_choice,  # 传入模型类型字符串
                n_trials=30,
                cv=3
            )
            
            best_params, best_score = optimizer.optimize(X, y)
            
            if best_score > best_overall['accuracy']:
                best_overall['accuracy'] = best_score
                best_overall['config'] = optimizer.format_final_result()
                best_overall['steps_order'] = steps_order
                best_overall['model_type'] = model_choice

    # 基准测试
    baseline_scores = []
    
    # RF基准
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(X_train, y_train)
    rf_score = accuracy_score(y_test, rf.predict(X_test))
    baseline_scores.append(("RandomForest", rf_score))
    
    # SVM基准
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_score = accuracy_score(y_test, svm.predict(X_test))
    baseline_scores.append(("SVM", svm_score))

    # KNN基准
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_score = accuracy_score(y_test, knn.predict(X_test))
    baseline_scores.append(("KNN", knn_score))
    
    # Naive Bayes基准
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_score = accuracy_score(y_test, nb.predict(X_test))
    baseline_scores.append(("NaiveBayes", nb_score))
    # 最终输出
    print(f"\n{'='*40}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print("="*40)
    
    # 输出所有基准模型成绩
    for model_name, score in baseline_scores:
        print(f"{model_name} Baseline Accuracy: {score:.4f}")
    
    if best_overall['accuracy'] != -np.inf:
        print(f"\nBest Optimized Accuracy: {best_overall['accuracy']:.4f}")
        print(f"Model Type: {best_overall['model_type']}")
        print(f"Effective pipeline prototype: {best_overall['steps_order']}")
        print("\nBest Configuration:")
        print(best_overall['config'])
    else:
        print("\nNo valid configuration found in any steps_order")
