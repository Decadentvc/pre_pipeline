import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    PowerTransformer, KBinsDiscretizer, Binarizer,
    OneHotEncoder
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class PrototypeSingleton:
    """管理预处理组件池和参数空间的单例类"""
    _instance = None
    POOL = {
        "impute": [None, SimpleImputer, IterativeImputer],
        "encode": [None, OneHotEncoder],
        "rebalance": [None, NearMiss, SMOTE],
        "normalize": [
            None, StandardScaler,
            PowerTransformer, MinMaxScaler, RobustScaler
        ],
        "discretize": [None, KBinsDiscretizer, Binarizer],
        "features": [
            None, PCA, SelectKBest,
            partial(FeatureUnion, transformer_list=[("pca", PCA()), ("selectkbest", SelectKBest())])
        ],
    }
    
    PARAM_GRIDS = {
        "impute": {
            SimpleImputer: {"strategy": ["constant", "most_frequent"]},
            IterativeImputer: {"initial_strategy": ['most_frequent', 'constant'], "imputation_order": ['ascending', 'descending', 'roman', 'arabic', 'random']}
        },
        "encode":{
            OneHotEncoder:{}
        },
        "rebalance":{
            NearMiss: {"n_neighbors": [1,2,3]},
            SMOTE: {"k_neighbors": [5,6,7]}
        },
        "normalize": {
            StandardScaler: {"with_mean": [True, False], "with_std": [True, False]},
            RobustScaler: {"quantile_range":[(25.0, 75.0),(10.0, 90.0), (5.0, 95.0)],"with_centering": [True, False], "with_scaling": [True, False]},
            PowerTransformer: {"method": ['yeo-johnson', 'box-cox'], "standardize": [True, False]},
            MinMaxScaler: {}
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

class PipelineOptimizer:
    """支持操作符选择和参数联合优化的优化器"""
    
    def __init__(self, steps_order, model):
        self.steps_order = steps_order
        self.model = model
        self.best_score = 0
        self.best_params = None
        self.process_map = {
            'impute': {'num', 'cat'},
            'encode': {'cat'},
            'normalize': {'num'},
            'discretize': {'num'},
            'rebalance': {'global'},
            'features': {'global'}
        }

    def build_dynamic_pipeline(self):
        """构建动态流水线结构"""
        return ImbPipeline([
            ('preprocessing', ColumnTransformer([
                ('num', Pipeline([
                    ('impute', None),
                    ('normalize', None),
                    ('discretize', None),
                ]), self.num_features),
                ('cat', Pipeline([
                    ('impute', None),
                    ('encode', None),
                ]), self.cat_features),
            ])),
            ('rebalance', None),
            ('features', None),
            ('classifier', self.model),
        ])

    def generate_param_grid(self):
        """生成合并操作符选择和参数优化的参数网格"""
        singleton = PrototypeSingleton.get_instance()
        param_grid = {}
        
        # 处理数值和类别特征步骤
        for step_name in self.steps_order:
            if step_name not in ['rebalance', 'features']:  # 全局步骤后续处理
                self._add_step_params(param_grid, step_name)

        # 处理全局步骤
        for step_name in ['rebalance', 'features']:
            if step_name in self.steps_order:
                self._add_step_params(param_grid, step_name, is_global=True)
                
        return param_grid

    def _add_step_params(self, param_grid, step_name, is_global=False):
        """添加步骤参数到参数网格"""
        singleton = PrototypeSingleton.get_instance()
        processors = singleton.POOL.get(step_name, [])
        
        # 获取步骤路径前缀
        prefix = self._get_param_prefix(step_name, is_global)
        if not prefix:
            return

        # 添加操作符选择
        param_grid[f'{prefix}'] = [
            None if p is None else p() if callable(p) else p 
            for p in processors
        ]

        # 添加参数配置
        for processor in processors:
            if processor is None:
                continue
                
            processor_cls = processor() if callable(processor) else processor.__class__
            if isinstance(processor, partial):
                processor_cls = processor.func
                
            params = singleton.PARAM_GRIDS[step_name].get(processor_cls, {})
            for param, values in params.items():
                param_grid[f'{prefix}__{param}'] = values

    def _get_param_prefix(self, step_name, is_global=False):
        """获取参数前缀"""
        targets = self.process_map.get(step_name, set())
        if 'num' in targets:
            return f'preprocessing__num__{step_name}'
        elif 'cat' in targets:
            return f'preprocessing__cat__{step_name}'
        elif is_global or 'global' in targets:
            return step_name
        return None

    def optimize(self, X, y):
        """执行联合优化"""
        self.num_features = list(range(X.shape[1]))
        self.cat_features = []
        
        # 构建动态流水线
        pipeline = self.build_dynamic_pipeline()
        
        # 生成参数网格
        param_grid = self.generate_param_grid()
        
        # 执行网格搜索
        search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, error_score='raise')
        search.fit(X, y)
        
        # 记录最佳结果
        self.best_score = search.best_score_
        self.best_params = search.best_params_
        return search.best_estimator_, self.best_params, self.best_score

# 测试案例
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    optimizer = PipelineOptimizer(
        steps_order=['impute', 'encode', 'normalize', 'features', 'rebalance'],
        model=RandomForestClassifier(n_estimators=50, random_state=42)
    )
    
    best_pipe, best_params, best_score = optimizer.optimize(X, y)
    
    # 基准测试
    baseline = RandomForestClassifier(n_estimators=50, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_score = accuracy_score(y_test, baseline.predict(X_test))
    
    print("\nFinal Results:")
    print(f"Baseline Accuracy: {baseline_score:.4f}")
    print(f"Optimized Accuracy: {best_score:.4f}")
    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"{k}: {v}")