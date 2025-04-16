import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    PowerTransformer, KBinsDiscretizer, Binarizer,
    OneHotEncoder, OrdinalEncoder, FunctionTransformer
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import itertools

class PrototypeSingleton:
    """Singleton管理预处理流程和特征状态"""
    _instance = None
    POOL = {
        "impute": [None, SimpleImputer(), IterativeImputer()],
        "encode": [None, OneHotEncoder()],
        "rebalance": [None, NearMiss(), SMOTE()],
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
    
    def __init__(self):
        self.parts = []
        self.X = None
        self.y = None
        self.num_features = []
        self.cat_features = []
        self.current_num = []
        self.current_cat = []

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    def configure(self, steps, X, y, num_features, cat_features):
        self.parts = steps
        self.X = X
        self.y = y
        self.num_features = num_features
        self.cat_features = cat_features
        self.reset_features()

    def reset_features(self):
        self.current_num = self.num_features.copy()
        self.current_cat = self.cat_features.copy()

    def apply_column_transformer(self):
        self.current_num = list(range(len(self.current_num)))
        self.current_cat = list(range(len(self.current_num), 
                                    len(self.current_num)+len(self.current_cat)))

    def apply_onehot_encoding(self):
        if len(self.cat_features) > 0:
            new_cat_dims = sum([len(np.unique(self.X[:,i])) 
                              for i in self.cat_features])
            self.current_cat = list(range(len(self.num_features), 
                                        len(self.num_features)+new_cat_dims))

class PipelineOptimizer:
    """修正后的Pipeline优化器，遍历所有预处理组合"""
    
    def __init__(self, steps, model):
        """
        steps: 预处理步骤列表，如 ['impute','encode', 'normalize', 'features']
        model: 最终的分类器或回归器
        """
        self.steps = steps
        self.model = model
        self.step_options = self._get_step_options()
        self.all_combinations = self._generate_combinations()
    
    def _get_step_options(self):
        """从PrototypeSingleton.POOL中获取每个步骤的候选处理器"""
        singleton = PrototypeSingleton.get_instance()
        return {step: singleton.POOL[step] for step in self.steps}
    
    def _generate_combinations(self):
        """生成所有可能的步骤组合（笛卡尔积）"""
        options = [self.step_options[step] for step in self.steps]
        return list(itertools.product(*options))
    
    def _build_pipeline(self, combination, num_features, cat_features):
        """根据组合构建预处理Pipeline"""
        # 定义步骤到处理流程的映射
        processing_flows = {
            'impute': ['num', 'cat'],
            'encode': ['cat'],
            'normalize': ['num'],
            'features': ['global']
        }
        
        numerical_steps = []
        categorical_steps = []
        global_steps = []
        
        # 解析组合中的每个步骤
        for step_name, processor in zip(self.steps, combination):
            if processor is None:
                continue
            flows = processing_flows.get(step_name, [])
            for flow in flows:
                if flow == 'num':
                    numerical_steps.append((f"{step_name}", processor))
                elif flow == 'cat':
                    categorical_steps.append((f"{step_name}", processor))
                elif flow == 'global':
                    global_steps.append((f"{step_name}", processor))
        
        # 构建数值和类别Pipeline
        num_pipeline = Pipeline(numerical_steps) if numerical_steps else 'passthrough'
        cat_pipeline = Pipeline(categorical_steps) if categorical_steps else 'passthrough'
        
        # 构建ColumnTransformer
        transformers = []
        if len(num_features) > 0:
            transformers.append(('num', num_pipeline, num_features))
        if len(cat_features) > 0:
            transformers.append(('cat', cat_pipeline, cat_features))
        
        pipeline_steps = []
        if transformers:
            pipeline_steps.append(('preprocessor', ColumnTransformer(transformers)))
        
        # 添加全局处理步骤
        for step_name, processor in global_steps:
            pipeline_steps.append((step_name, processor))
        
        # 添加模型
        pipeline_steps.append(('classifier', self.model))
        
        return Pipeline(pipeline_steps)
    
    def optimize(self, X, y, test_size=0.2, n_iter=None):
        """执行优化并评估所有组合"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        singleton = PrototypeSingleton.get_instance()
        num_features = singleton.num_features
        cat_features = singleton.cat_features
        
        best_score = 0
        best_pipeline = None
        
        for i, combination in enumerate(self.all_combinations):
            if n_iter is not None and i >= n_iter:
                break
            print(f"Evaluating combination {i+1}/{len(self.all_combinations)}: {combination}")
            
            try:
                pipeline = self._build_pipeline(combination, num_features, cat_features)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                print(f"Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_pipeline = pipeline
                    print(f"New best score: {best_score:.4f}")
            except Exception as e:
                print(f"Error with combination {combination}: {str(e)}")
                continue
        
        if best_pipeline is None:
            raise RuntimeError("No valid pipeline found.")
        
        return best_pipeline, best_score

def main():
    """测试优化器"""
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 配置PrototypeSingleton
    singleton = PrototypeSingleton.get_instance()
    num_features = list(range(X.shape[1]))  # 所有特征均为数值型
    cat_features = []
    singleton.configure(steps=[], X=X, y=y, num_features=num_features, cat_features=cat_features)
    
    # 创建模型
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # 创建优化器
    optimizer = PipelineOptimizer(
        steps=['impute','encode', 'normalize', 'features'],
        model=model
    )
    
    # 执行优化
    best_pipe, best_score = optimizer.optimize(X, y)
    
    # 基准测试
    baseline = RandomForestClassifier(n_estimators=50, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_score = accuracy_score(y_test, baseline.predict(X_test))
    
    print(f"\n优化结果:")
    print(f"- 优化后的准确率: {best_score:.4f}")
    print(f"- 基准准确率: {baseline_score:.4f}")
    print("最优pipeline步骤:")
    for name, step in best_pipe.steps:
        print(f"  {name}: {step.__class__.__name__ if hasattr(step, '__class__') else step}")

if __name__ == "__main__":
    main()