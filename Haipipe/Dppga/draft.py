from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    PowerTransformer, KBinsDiscretizer, Binarizer,
    OneHotEncoder
)
from sklearn.model_selection import GridSearchCV

# 隐式存在的数据集和模型
X, y = load_iris(return_X_y=True)
pipeline = Pipeline([
    ('imputer', SimpleImputer()),     # 预处理步骤
    ('scaler', StandardScaler()),     # 预处理步骤
    ('clf', LogisticRegression())     # 必须包含最终模型！
])

# 参数网格显式关联预处理参数
param_grid = {
    'imputer__strategy': ['mean', 'median'],
    'scaler__with_mean': [True, False],
    'clf__C': [0.1, 1, 10]           # 模型参数也可同时优化
}

# 通过交叉验证评估效果
searcher = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
searcher.fit(X, y)  # 这里传入数据集X,y

print("最佳参数:", searcher.best_params_)
print("验证集得分:", searcher.best_score_)