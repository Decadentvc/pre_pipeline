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


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from copy import deepcopy

class ModelManager:
    """
    管理可用模型、提供外部接口选择模型和设置参数（不设置 random_state）。
    Keys: "DT","RF","GBDT","LR","KNN","SVM"
    """
    def __init__(self):
        # 默认模型实例（可替换默认参数）
        self._models = {
            "DT": DecisionTreeClassifier(),                     # Decision Tree
            "RF": RandomForestClassifier(n_estimators=50, random_state=42),  # Random Forest
            "GBDT": GradientBoostingClassifier(random_state=42),            # GBDT
            "LR": LogisticRegression(max_iter=1000),            # Logistic Regression
            "KNN": KNeighborsClassifier(n_neighbors=5),         # KNN
            "SVM": SVC(probability=True, random_state=42)       # SVM
        }
        # 记录哪些模型被启用（默认全部可用）
        self._enabled = {k: True for k in self._models.keys()}

    def enable(self, key):
        if key in self._models:
            self._enabled[key] = True
        else:
            raise KeyError(f"Unknown model key: {key}")

    def disable(self, key):
        if key in self._models:
            self._enabled[key] = False
        else:
            raise KeyError(f"Unknown model key: {key}")

    def set_model_params(self, key, **params):
        """
        为指定模型设置参数。会忽略名为 'random_state' 的参数（不允许外部修改该参数）。
        支持设置模型绝大多数 scikit-learn 参数（若参数名错误会抛异常）。
        """
        if key not in self._models:
            raise KeyError(f"Unknown model key: {key}")
        params = {k: v for k, v in params.items() if k != "random_state"}
        # 克隆实例以避免修改默认实例（更安全）
        est = clone(self._models[key])
        try:
            est.set_params(**params)
        except Exception as e:
            raise ValueError(f"Failed to set params for {key}: {e}")
        self._models[key] = est

    def get_model_config(self):
        """
        返回用于注入到优化器的 model_config 字典，
        仅包含被 enable 的模型映射：key -> estimator instance
        """
        return {k: deepcopy(v) for k, v in self._models.items() if self._enabled.get(k, False)}

    def available_keys(self):
        return [k for k, v in self._enabled.items() if v]

class EnhancedPipelineRunner:
    """
    一个增强运行器：使用 ModelManager 的 model_config，
    并直接使用 BayesianPipelineOptimizer来进行搜索。
    """
    def __init__(self, file_path, target_column, steps_order_candidates,
                 model_manager: ModelManager, n_trials=50, cv=3):
        self.file_path = file_path
        self.target_column = target_column
        self.steps_order_candidates = steps_order_candidates
        self.n_trials = n_trials
        self.cv = cv
        self.model_manager = model_manager

        # 利用原 PipelineOptimizer 的 _load_data 实现（不修改其类）
        # 仅实例化以获取 X,y（并不会调用其 optimize）
        tmp = PipelineOptimizer(file_path=file_path, target_column=target_column,
                                steps_order_candidates=[steps_order_candidates], n_trials=1, cv=cv,
                                model_choices=[])  # model_choices 传空以跳过不必要部分
        self.X = tmp.X
        self.y = tmp.y

        # 结果记录
        self.best_overall = {'accuracy': -np.inf, 'config': None, 'steps_order': None, 'model_type': None}
        self.per_steps_order_best = {}
        self.per_model_best = {}  # 键: 模型名称, 值: {'accuracy': float, 'steps_order': list, 'config': str}

    def _compute_baselines(self):
        """
        基准性能：对 ModelManager 当前的模型集合做 cross_val_score（accuracy）。
        """
        baseline_scores = {}
        model_config = self.model_manager.get_model_config()
        for key, est in model_config.items():
            try:
                scores = cross_val_score(clone(est), self.X, self.y, cv=self.cv, scoring='accuracy')
                baseline_scores[key] = {
                    'mean_accuracy': float(np.mean(scores)),
                    'std_accuracy': float(np.std(scores)),
                    'scores': scores
                }
            except Exception as e:
                baseline_scores[key] = {'error': str(e)}
        return baseline_scores

    def optimize(self):
        """
        主优化流程：遍历 ModelManager 中的模型键与 steps_order_candidates，
        为每对 (model_key, steps_order) 实例化一个 BayesianPipelineOptimizer，
        将 optimizer.model_config 覆盖为 ModelManager 的配置（实例级覆盖），
        然后运行 optimizer.optimize(...)。
        """
        model_config = self.model_manager.get_model_config()
        model_keys = list(model_config.keys())
        
        # 初始化每个模型的最佳结果存储
        self.per_model_best = {}
        for model_key in model_keys:
            self.per_model_best[model_key] = {
                'accuracy': -np.inf,
                'steps_order': None,
                'config': None
            }
        
        # 初始化 per_steps storage
        for steps_order in self.steps_order_candidates:
            key = tuple(steps_order)
            self.per_steps_order_best[key] = {'accuracy': -np.inf, 'model_type': None, 'config': None}
        
        # 初始化全局最佳结果
        self.best_overall = {
            'accuracy': -np.inf,
            'model_type': None,
            'steps_order': None,
            'config': None
        }
        
        # 遍历模型与步骤组合
        for model_key in model_keys:
            for steps_order in self.steps_order_candidates:
                print(f"\n--- Optimize: Model={model_key} | Steps={'->'.join(steps_order)} ---")
                optimizer = BayesianPipelineOptimizer(steps_order=steps_order, model=model_key,
                                                    n_trials=self.n_trials, cv=self.cv)
                # 覆盖实例的 model_config，使其使用我们 ModelManager 提供的模型实例
                optimizer.model_config = model_config
                
                # 优化（使用原有 optimize 接口）
                try:
                    best_params, best_score = optimizer.optimize(self.X, self.y)
                except Exception as e:
                    print(f"Optimization failed for {model_key} with steps {steps_order}: {e}")
                    best_params, best_score = None, float('-inf')
                
                # 更新当前模型的最佳结果
                if best_score is not None and best_score > self.per_model_best[model_key]['accuracy']:
                    self.per_model_best[model_key]['accuracy'] = best_score
                    self.per_model_best[model_key]['steps_order'] = steps_order
                    self.per_model_best[model_key]['config'] = optimizer.format_final_result()
                
                # 更新全局最佳结果
                if best_score is not None and best_score > self.best_overall['accuracy']:
                    self.best_overall['accuracy'] = best_score
                    self.best_overall['model_type'] = model_key
                    self.best_overall['steps_order'] = steps_order
                    self.best_overall['config'] = optimizer.format_final_result()
                
                # 更新当前步骤顺序的最佳结果
                key = tuple(steps_order)
                if best_score is not None and best_score > self.per_steps_order_best[key]['accuracy']:
                    self.per_steps_order_best[key]['accuracy'] = best_score
                    self.per_steps_order_best[key]['model_type'] = model_key
                    self.per_steps_order_best[key]['config'] = optimizer.format_final_result()
        
        return {
            "baseline_performance": self._compute_baselines(),
            "optimization_results": {
                "accuracy": self.best_overall['accuracy'],
                "model_type": self.best_overall['model_type'],
                "best_steps_order": self.best_overall['steps_order'],
                "configuration": self.best_overall['config']
            },
            "per_steps_order_best": self.per_steps_order_best,
            "per_model_best": self.per_model_best  # 新增：每个模型的最佳结果
        }

# === 示例外部调用：将演示如何使用 ModelManager 与 EnhancedPipelineRunner ===
if __name__ == "__main__":
    # 简单示例（请替换为真实数据路径与列名）
    FILE_PATH = "your_dataset.csv"   # <-- 替换为真实 CSV 路径
    TARGET_COL = "label"             # <-- 替换为真实目标列名

    # 定义候选步骤顺序（示例）
    steps_order_candidates = [
        ["impute", "encode", "normalize", "features", "rebalance"],
        ["impute", "normalize", "features"]
    ]

    # 创建 ModelManager 并自定义模型参数
    mm = ModelManager()
    # 设置 KNN 参数示例
    mm.set_model_params("KNN", n_neighbors=3)
    # 设置 Logistic Regression 参数示例
    mm.set_model_params("LR", penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
    # （可选）禁用某一模型，例如禁用 GBDT
    # mm.disable("GBDT")

    # 创建并运行增强运行器
    runner = EnhancedPipelineRunner(
        file_path=FILE_PATH,
        target_column=TARGET_COL,
        steps_order_candidates=steps_order_candidates,
        model_manager=mm,
        n_trials=30,   # Optuna 迭代次数示例
        cv=3
    )

    results = runner.optimize()

    print("\n=== Baseline performance ===")
    print(results["baseline_performance"])
    print("\n=== Best overall ===")
    print(results["optimization_results"])
    print("\n=== Per steps-order best ===")
    for k, v in results["per_steps_order_best"].items():
        print(k, "=>", v)
