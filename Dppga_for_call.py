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
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
    def __init__(self, steps_order, model, n_trials=50, cv=3):
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
        
        return "\n".join(output)

class PipelineOptimizer:
    def __init__(self, file_path, target_column, steps_order_candidates, 
                 n_trials=50, cv=3,  model_choices=["RF", "SVM", "KNN", "NB"]):
        """
        初始化管道优化器
        
        参数:
            file_path: 数据文件路径
            target_column: 目标变量列名
            steps_order_candidates: 管道步骤顺序候选列表
            n_trials: Optuna试验次数(默认50)
            cv: 交叉验证折数(默认3)
            model: 使用的分类模型(默认RandomForestClassifier)
        """
        self.file_path = file_path
        self.target_column = target_column
        self.steps_order_candidates = steps_order_candidates
        self.n_trials = n_trials
        self.cv = cv
        self.model_choices = model_choices
        
        # 模型配置
        self.model_config = {
            "RF": RandomForestClassifier(n_estimators=50, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "NB": GaussianNB()
        }
        
        self.X, self.y = self._load_data()
        self.baseline_scores = {}
        self.best_overall = None
        self.per_steps_order_best = {}
        
    def _load_data(self):
        """加载并预处理数据"""
        try:
            df = pd.read_csv(self.file_path)
            
            if self.target_column not in df.columns:
                raise ValueError(f"目标列 '{self.target_column}' 不存在于数据集中")
            
            feature_columns = [col for col in df.columns if col != self.target_column]
            
            df_processed = df.copy()
            label_encoders = {}
            
            for col in feature_columns:
                if df_processed[col].dtype == 'object' or isinstance(df_processed[col].dtype, pd.CategoricalDtype):
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    label_encoders[col] = le
            
            if df_processed[self.target_column].dtype == 'object' or isinstance(df_processed[self.target_column].dtype, pd.CategoricalDtype):
                le_target = LabelEncoder()
                df_processed[self.target_column] = le_target.fit_transform(df_processed[self.target_column].astype(str))
            
            # 填充缺失值
            for col in feature_columns:
                if np.issubdtype(df_processed[col].dtype, np.number):
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            
            X = df_processed[feature_columns].values.astype(np.float32)
            y = df_processed[self.target_column].values.astype(int)
            
            return X, y
            
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            raise e
    
    def _compute_baselines(self):
        """计算基准模型性能"""
        for model_name in self.model_choices:
            if model_name in self.model_config:
                model = clone(self.model_config[model_name])
                scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring='accuracy')
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                self.baseline_scores[model_name] = {
                    'mean_accuracy': mean_score,
                    'std_accuracy': std_score,
                    'scores': scores
                }
    
    def optimize(self):
        """执行管道优化"""
        # 计算基准性能
        self._compute_baselines()
        
        self.best_overall = {
            'accuracy': -np.inf,
            'config': None,
            'steps_order': None,
            'model_type': None
        }
        
        # 初始化每个步骤顺序的存储
        for steps_order in self.steps_order_candidates:
            key = tuple(steps_order)  # 使用元组作为键，因为列表不可哈希
            self.per_steps_order_best[key] = {
                'accuracy': -np.inf,
                'model_type': None,
                'config': None
            }
        
        for model_name in self.model_choices:
            if model_name not in self.model_config:
                print(f"⚠️ 未知模型类型: {model_name}, 跳过")
                continue
                
            for steps_order in self.steps_order_candidates:
                key = tuple(steps_order)
                
                print(f"\n{'='*40}\n优化步骤顺序: {'->'.join(steps_order)} | 使用模型: {model_name}\n{'='*40}")
                
                optimizer = BayesianPipelineOptimizer(
                    steps_order=steps_order,
                    model=model_name,
                    n_trials=self.n_trials,
                    cv=self.cv
                )
                
                _, best_score = optimizer.optimize(self.X, self.y)
                
                # 更新全局最佳结果
                if best_score > self.best_overall['accuracy']:
                    self.best_overall['accuracy'] = best_score
                    self.best_overall['config'] = optimizer.format_final_result()
                    self.best_overall['steps_order'] = steps_order
                    self.best_overall['model_type'] = model_name
                
                # 更新当前步骤顺序的最佳结果
                if best_score > self.per_steps_order_best[key]['accuracy']:
                    self.per_steps_order_best[key]['accuracy'] = best_score
                    self.per_steps_order_best[key]['model_type'] = model_name
                    self.per_steps_order_best[key]['config'] = optimizer.format_final_result()


    
    def get_results(self):
        """获取优化结果"""
        if not self.best_overall:
            raise RuntimeError("必须先调用 optimize() 方法获取结果")
        
        # 创建基准分数的易读格式
        baseline_summary = {}
        for model_name, scores in self.baseline_scores.items():
            baseline_summary[model_name] = f"{scores['mean_accuracy']} "

        per_steps_best = {}
        for steps, data in self.per_steps_order_best.items():
            steps_str = "->".join(steps)
            per_steps_best[steps_str] = {
                "accuracy": data['accuracy'],
                "model_type": data['model_type'],
                "configuration": data['config']
            }
        
        return {
            "dataset_info": {
                "file_path": self.file_path,
                "target_column": self.target_column,
                "num_samples": self.X.shape[0],
                "num_features": self.X.shape[1],
                "num_classes": len(np.unique(self.y))
            },
            "baseline_performance": baseline_summary,
            "optimization_results": {
                "accuracy": self.best_overall['accuracy'],
                "model_type": self.best_overall['model_type'],
                "best_steps_order": self.best_overall['steps_order'],
                "configuration": self.best_overall['config']
            },
            "per_steps_order_best": per_steps_best  # 每个步骤顺序的最佳结果
        }