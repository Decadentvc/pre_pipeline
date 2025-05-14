import numpy as np
import itertools
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
from functools import partial

class PrototypeSingleton:
    """管理预处理组件池和参数空间的单例类"""
    _instance = None
    POOL = {
        "impute": [None, SimpleImputer(), IterativeImputer()],
        "encode": [None, OneHotEncoder()],
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
            SimpleImputer: {"strategy": ["constant", "most_frequent"]},
            IterativeImputer: {"initial_strategy": ['most_frequent', 'constant'], "imputation_order": ['ascending', 'descending', 'roman', 'arabic', 'random']}
        },
        "encode":{
            OneHotEncoder:{ }
        },
        "reblance":{
            NearMiss:{"n_neighbors": [1,2,3]},
            SMOTE:{"k_neighbors": [5,6,7]}
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

class PipelineOptimizer:
    """支持全组合遍历与参数联合优化的优化器"""
    
    def __init__(self, steps_order, model):
        """
        :param steps_order: 步骤顺序
        :param model: 最终分类器
        """
        self.steps_order = steps_order
        self.model = model
        self.best_score = 0
        self.best_combo = None
        self.best_params = None
        self.best_step_mapping = None
        
        # 步骤处理路由配置
        self.process_map = {
            'impute': {'num', 'cat'},
            'encode': {'cat'},
            'normalize': {'num'},
            'discretize': {'num'},
            'rebalance': {'global'},
            'features': {'global'}
        }

    def _parse_optimized_params(self, current_params, step_mapping):
        """解析优化后的参数到步骤名称和处理器"""
        params_by_step = {}
        if not current_params:
            return params_by_step
        
        for param_key, param_value in current_params.items():
            # 遍历步骤映射寻找匹配的前缀
            for (step_name, step_cls), prefix in step_mapping.items():
                if param_key.startswith(f"{prefix}__"):
                    param_name = param_key[len(prefix)+2:]
                    key = (step_name, step_cls)
                    if key not in params_by_step:
                        params_by_step[key] = {}
                    params_by_step[key][param_name] = param_value
                    break  # 参数键唯一匹配
        
        # 转换为可读字符串
        optimized_desc = {}
        for (step_name, step_cls), params in params_by_step.items():
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            optimized_desc[(step_name, step_cls)] = f"{step_cls.__name__}({param_str})"
        return optimized_desc

    def _build_pipeline(self, combo):
        """构建流水线并返回步骤映射（包含完整参数前缀）"""
        singleton = PrototypeSingleton.get_instance()
        step_mapping = {}
        num_steps = []
        cat_steps = []
        global_steps = []

        for step_name, processor in zip(self.steps_order, combo):
            if processor is None:
                continue
                
            step_cls = processor.__class__
            step_id = f"{step_name}_{step_cls.__name__}".lower()
            
            # 确定处理路径和参数前缀
            targets = self.process_map.get(step_name, set())
            if 'num' in targets:
                num_steps.append((step_id, processor))
                full_prefix = f"preprocessing__num__{step_id}"
            elif 'cat' in targets:
                cat_steps.append((step_id, processor))
                full_prefix = f"preprocessing__cat__{step_id}"
            elif 'global' in targets:
                global_steps.append((step_id, processor))
                full_prefix = step_id
            
            step_mapping[(step_name, step_cls)] = full_prefix

        # 构建完整流水线
        transformers = []
        if num_steps:
            transformers.append(('num', Pipeline(num_steps), singleton.num_features))
        if cat_steps:
            transformers.append(('cat', Pipeline(cat_steps), singleton.cat_features))
        
        pipeline_steps = []
        if transformers:
            pipeline_steps.append(('preprocessing', ColumnTransformer(transformers, remainder='drop')))
        
        for step_id, processor in global_steps:
            pipeline_steps.append((step_id, processor))
        
        pipeline_steps.append(('classifier', self.model))
        return Pipeline(pipeline_steps), step_mapping

    def _generate_param_grid(self, combo, step_mapping):
        """生成参数网格（基于完整前缀）"""
        param_grid = {}
        singleton = PrototypeSingleton.get_instance()
        
        for step_name, processor in zip(self.steps_order, combo):
            if processor is None:
                continue
                
            step_cls = processor.__class__
            class_params = singleton.PARAM_GRIDS.get(step_name, {}).get(step_cls, {})
            prefix = step_mapping.get((step_name, step_cls), "")
            
            for param, values in class_params.items():
                param_path = f"{prefix}__{param}"
                param_grid[param_path] = values
                
        return param_grid

    def optimize(self, X, y):
        singleton = PrototypeSingleton.get_instance()
        singleton.num_features = list(range(X.shape[1]))
        singleton.cat_features = []
        
        pool = PrototypeSingleton.POOL
        all_combos = list(itertools.product(*[pool[step] for step in self.steps_order]))
        
        for combo_idx, combo in enumerate(all_combos):
            print(f"\nProcessing combination {combo_idx+1}/{len(all_combos)}")
            print("Initial configuration:")
            print("\n".join(self._format_combo(combo)))
            
            try:
                pipe, step_mapping = self._build_pipeline(combo)
                param_grid = self._generate_param_grid(combo, step_mapping)
                
                if not param_grid:
                    score = self._evaluate_pipeline(pipe, X, y)
                    current_params = None
                else:
                    search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy')
                    search.fit(X, y)
                    score = search.best_score_
                    current_params = search.best_params_
                
                # 显示优化后的参数
                print(f"\n🔍 Optimized score: {score:.4f}")
                if current_params:
                    optimized_desc = self._parse_optimized_params(current_params, step_mapping)
                    print("Optimized parameters:")
                    for step in self.steps_order:
                        processor = combo[self.steps_order.index(step)]
                        if processor is None:
                            continue
                        step_cls = processor.__class__
                        key = (step, step_cls)
                        desc = optimized_desc.get(key, 
                            f"{step_cls.__name__}(default parameters)")
                        print(f"  {step}: {desc}")
                else:
                    print("No parameter optimization needed")

                # 更新最佳记录
                if score > self.best_score:
                    self.best_score = score
                    self.best_combo = combo
                    self.best_params = current_params
                    self.best_step_mapping = step_mapping
                    print("\n🏆 New best configuration found!")
                    
            except Exception as e:
                print(f"⚠️ Optimization failed: {str(e)}")
                continue
        
        return self.best_combo, self.best_params, self.best_score

    def _evaluate_pipeline(self, pipe, X, y):
        """评估无参数优化的流水线"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)
        return accuracy_score(y_test, pipe.predict(X_test))

    def _format_combo(self, combination):
        """仅显示初始配置"""
        config = []
        for step, processor in zip(self.steps_order, combination):
            if processor is None:
                config.append(f"{step}: None")
                continue
            params = processor.get_params()
            param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k in self._get_relevant_params(step, processor))
            config.append(f"{step}: {processor.__class__.__name__}({param_str})")
        return config
    
    def _get_relevant_params(self, step_name, processor):
        """获取该步骤需要显示的参数"""
        singleton = PrototypeSingleton.get_instance()
        param_defs = singleton.PARAM_GRIDS.get(step_name, {}).get(processor.__class__, {})
        return param_defs.keys()

    def format_final_result(self):
        """格式化最终结果输出，基于步骤映射解析参数路径"""
        lines = []
        for step, processor in zip(self.steps_order, self.best_combo):
            if processor is None:
                lines.append(f"{step}: None")
                continue
            
            step_cls = processor.__class__
            mapping_key = (step, step_cls)
            if not self.best_step_mapping or mapping_key not in self.best_step_mapping:
                lines.append(f"{step}: {step_cls.__name__}()")
                continue
            
            parent_path, step_id = self.best_step_mapping[mapping_key]
            param_prefix = f"{parent_path}__{step_id}" if parent_path else step_id
            params = {}
            
            if self.best_params:
                for param_key, param_value in self.best_params.items():
                    if param_key.startswith(f"{param_prefix}__"):
                        param_name = param_key.split("__")[-1]
                        params[param_name] = param_value
            
            param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
            lines.append(f"{step}: {step_cls.__name__}({param_str})")
        
        return "\n".join(lines)

# 测试案例
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 初始化优化器
    optimizer = PipelineOptimizer(
        steps_order=['impute', 'encode', 'normalize', 'features', 'rebalance'],
        # steps_order=['impute',  'normalize', 'features'],
        model=RandomForestClassifier(n_estimators=50, random_state=42)
    )
    
    # 执行全组合优化（实际使用时建议限制组合数量）
    best_combo, best_params, best_score = optimizer.optimize(X, y)
    
    # 基准测试
    baseline = RandomForestClassifier(n_estimators=50, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_score = accuracy_score(y_test, baseline.predict(X_test))
    
    # 打印结果
    print("\nFinal Results:")
    print(f"Baseline Accuracy: {baseline_score:.4f}")
    print(f"Optimized Accuracy: {best_score:.4f}")
    print("Best Configuration:")
    print(optimizer.format_final_result())