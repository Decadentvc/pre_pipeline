from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from imblearn.pipeline import Pipeline as ImbPipeline 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, KBinsDiscretizer, Binarizer
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
import sys
import numpy as np

# 假设已有数据集和ML算法（以分类任务为例）
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 预定义的转换操作符库（修正后）
OPERATOR_LIBRARY = {
    # 编码
    "Encoding(E)": {
        "One Hot": {
            "constructor": OneHotEncoder,
            "params": {
                "handle_unknown": hp.choice("E_handle_unknown", ["ignore"]),
                "drop": hp.choice("E_drop", [None, "first"])
            }
        },
        "Ordinal（兼容版）": { 
            "constructor": OrdinalEncoder,
            "params": {
                # 版本检测参数
                "encoded_missing_value": hp.choice("E_missing_value", [-1, -999]) if hasattr(OrdinalEncoder, 'handle_unknown') else {}
            }
        }
    },
    
    # 归一化
    "Normalization(N)": {
        "Standard Scaler": {
            "constructor": StandardScaler,
            "params": {
                "with_mean": hp.choice("N_std_mean", [True, False]),
                "with_std": hp.choice("N_std_std", [True, False])
            }
        },
        "Power Transform": {
            "constructor": PowerTransformer,
            "params": {
                "method": hp.choice("N_power_method", ["yeo-johnson", "box-cox"]),
                "standardize": hp.choice("N_power_std", [True, False])
            }
        },
        "MinMax Scaler": {
            "constructor": MinMaxScaler,
            "params": {
                "feature_range": hp.choice("N_minmax_range", [(0,1), (-1,1)])
            }
        },
        "Robust Scaler": {
            "constructor": RobustScaler,
            "params": {
                "quantile_range": hp.choice("N_robust_quantile", [(25.0,75.0), (5.0,95.0)])
            }
        }
    },
    
    # 离散化
    "Discretization(D)": {
        "KBins": {
            "constructor": KBinsDiscretizer,
            "params": {
                "n_bins": hp.quniform("D_kbins", 3, 10, 1),
                "encode": hp.choice("D_encode", ["onehot", "ordinal"]),
                "strategy": hp.choice("D_strategy", ["uniform", "quantile", "kmeans"])
            }
        },
        "Binarization": {
            "constructor": Binarizer,
            "params": {
                "threshold": hp.uniform("D_threshold", 0.3, 0.7)
            }
        }
    },
    
    # 插补
    "Imputation(I)": {
        "Univariate": {
            "constructor": SimpleImputer,
            "params": {
                "strategy": hp.choice("I_uni_strategy", ["mean", "median", "most_frequent"])
            }
        },
        "Multivariate": {
            "constructor": IterativeImputer,
            "params": {
                "n_nearest_features": hp.quniform("I_multi_n", 2, 10, 1),
                "initial_strategy": hp.choice("I_multi_init", ["mean", "median"])
            }
        }
    },
    
    # 重新平衡
    "Rebalancing(R)": {
        "Near Miss": {
            "constructor": NearMiss,
            "params": {
                "version": hp.choice("R_nearmiss_ver", [1, 2, 3]),
                "n_neighbors": hp.quniform("R_nearmiss_n", 3, 7, 1)
            }
        },
        "SMOTE": {
            "constructor": SMOTE,
            "params": {
                "k_neighbors": hp.quniform("R_smote_k", 3, 7, 1),
                "sampling_strategy": hp.choice("R_smote_strategy", ["minority", "not majority"])
            }
        }
    },
    
    # 特征工程
    "Feat.Eng.(F)": {
        "PCA": {
            "constructor": PCA,
            "params": {
                "n_components": hp.uniform("F_pca_n", 0.7, 0.95),
                "svd_solver": hp.choice("F_pca_solver", ["auto", "full"])
            }
        },
        "Select K Best": {
            "constructor": SelectKBest,
            "params": {
                "k": hp.quniform("F_kbest", 5, 30, 5)
            }
        },
    }
}

def build_pipeline(proto_steps, best_params):
    steps = []
    for step in proto_steps:
        step_config = best_params[step]
        op_name = step_config['operator']
        op_params = step_config['params']

        # 辅助函数：转换整数参数
        def to_int(value):
            return int(value) if isinstance(value, float) else value

        # 统一处理整数参数转换
        if step == "I":
            if op_name == "Univariate":
                transformer = SimpleImputer(strategy=op_params.get("strategy", "mean"))
            elif op_name == "Multivariate":
                transformer = IterativeImputer(
                    initial_strategy=op_params.get("initial_strategy", "mean"),
                    n_nearest_features=to_int(op_params.get("n_nearest_features", 5))
                )

        elif step == "E":
            if op_name == "One Hot":
                # 保持原有逻辑
                valid_params = {
                    "handle_unknown": op_params.get("handle_unknown", "ignore"),
                    "drop": op_params.get("drop", None)
                }
                transformer = OneHotEncoder(**valid_params)
            else:
                # 新版兼容处理
                if hasattr(OrdinalEncoder, 'handle_unknown'):  # 新版本特性检测
                    valid_params = {
                        "handle_unknown": "use_encoded_value",
                        "unknown_value": op_params.get("encoded_missing_value", -1)
                    }
                else:  # 旧版本回退方案
                    valid_params = {
                        "handle_unknown": "ignore"  # 虽然旧版本不支持，但后续添加安全机制
                    }
                # 创建带安全机制的编码器
                transformer = SafeOrdinalEncoder(**valid_params)

        elif step == "N":
            if op_name == "Standard Scaler":
                valid_params = {
                    "with_mean": op_params.get("with_mean", True),
                    "with_std": op_params.get("with_std", True)
                }
                transformer = StandardScaler(**valid_params)
            elif op_name == "Power Transform":
                valid_params = {
                    "method": op_params.get("method", "yeo-johnson"),
                    "standardize": op_params.get("standardize", True)
                }
                transformer = PowerTransformer(**valid_params)
            elif op_name == "MinMax Scaler":
                valid_params = {"feature_range": op_params.get("feature_range", (0, 1))}
                transformer = MinMaxScaler(**valid_params)
            elif op_name == "Robust Scaler":
                valid_params = {"quantile_range": op_params.get("quantile_range", (25.0, 75.0))}
                transformer = RobustScaler(**valid_params)

        elif step == "D":
            if op_name == "KBins":
                valid_params = {
                    "n_bins": _ensure_int(op_params.get("n_bins", 5)),
                    "encode": op_params.get("encode", "ordinal"),
                    "strategy": op_params.get("strategy", "quantile")
                }
                transformer = KBinsDiscretizer(**valid_params)
            elif op_name == "Binarization":
                valid_params = {"threshold": op_params.get("threshold", 0.5)}
                transformer = Binarizer(**valid_params)

        elif step == "R":  
            if op_name == "Near Miss":
                transformer = NearMiss(
                    version=to_int(op_params.get("version", 3)),
                    n_neighbors=to_int(op_params.get("n_neighbors", 3))
                )
            else:
                transformer = SMOTE(
                    k_neighbors=to_int(op_params.get("k_neighbors", 5)),
                    sampling_strategy=op_params.get("sampling_strategy", "auto")
                )

        elif step == "F":
            if op_name == "PCA":
                transformer = PCA(
                    n_components=min(op_params.get("n_components", 0.9), 0.95),
                    svd_solver=op_params.get("svd_solver", "auto")
                )
            else:
                transformer = SelectKBest(k=to_int(op_params.get("k", 10)))

        steps.append((f"{step}_{op_name}", transformer))
    
    return ImbPipeline(steps) 

def optimization_objective(params, proto_steps, X, y):
    try:
        # 深拷贝参数防止污染
        safe_params = {k: dict(v) for k, v in params.items()}
        
        # 动态参数修正
        for step in proto_steps:
            if step == "F" and safe_params[step]['operator'] == "Select K Best":
                safe_params[step]['params']['k'] = min(
                    int(safe_params[step]['params']['k']), 
                    X.shape[1]
                )
            if step == "R" and safe_params[step]['operator'] == "SMOTE":
                min_class = min(np.bincount(y))
                safe_params[step]['params']['k_neighbors'] = min(
                    int(safe_params[step]['params']['k_neighbors']),
                    min_class - 1
                )
        
        pipeline = build_pipeline(proto_steps, safe_params)
        full_pipeline = clone(pipeline).set_params(
            classifier=RandomForestClassifier(n_estimators=100)
        )
        
        score = cross_val_score(full_pipeline, X, y, cv=3, scoring='accuracy').mean()
        return {'loss': 1 - score, 'status': STATUS_OK}
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return {'loss': 1.0, 'status': STATUS_OK}

from sklearn.base import BaseEstimator, TransformerMixin

class SafeOrdinalEncoder(BaseEstimator, TransformerMixin):
    """带未知值处理的OrdinalEncoder兼容版"""
    def __init__(self, handle_unknown='ignore', unknown_value=-1):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoder = OrdinalEncoder() if not hasattr(OrdinalEncoder, 'handle_unknown') else \
                      OrdinalEncoder(handle_unknown=handle_unknown, unknown_value=unknown_value)
        self.categories_ = None

    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        self.categories_ = self.encoder.categories_
        return self

    def transform(self, X):
        try:
            return self.encoder.transform(X)
        except ValueError as e:
            if "Found unknown categories" in str(e) and not hasattr(OrdinalEncoder, 'handle_unknown'):
                # 旧版本兼容处理：将未知值替换为特殊编码
                X_trans = self.encoder.transform(X)
                mask = np.isnan(X_trans)
                X_trans[mask] = self.unknown_value
                return X_trans.astype(int)
            raise

def optimize_pipeline_prototype(proto_steps, X, y, max_evals=50):
    """主优化函数"""
    space = {}
    for step in proto_steps:
        step_type = {
            'I': "Imputation(I)",
            'E': "Encoding(E)",
            'N': "Normalization(N)",
            'D': "Discretization(D)",
            'R': "Rebalancing(R)",
            'F': "Feat.Eng.(F)"
        }[step]
        
        operators = []
        for op in OPERATOR_LIBRARY[step_type]:
            # 版本兼容性过滤
            if step == 'E' and "Ordinal（新版）" in op:
                if not hasattr(OrdinalEncoder, 'handle_unknown'):
                    continue
            operators.append(op)
        
        space[step] = {
            'operator': hp.choice(f"{step}_operator", operators),
            'params': hp.pchoice(
                f"{step}_op_params",
                [(1.0 / len(operators), OPERATOR_LIBRARY[step_type][op]['params']) 
                 for op in operators]
            )
        }
    
    trials = Trials()
    try:
        best = fmin(
            fn=lambda params: optimization_objective(params, proto_steps, X, y),
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            catch_eval_exceptions=True  # 关键参数：捕获评估异常
        )
    except AllTrialsFailed:
        raise RuntimeError("所有试验失败，请检查：1.数据质量 2.参数空间定义 3.预处理逻辑")
    
    # 解码最佳参数并后处理
    best_params = space_eval(space, best)
    for step in proto_steps:
        # 转换离散参数
        for param, val in best_params[step]['params'].items():
            if 'quniform' in param:
                best_params[step]['params'][param] = int(val)
    
    return {
        "best_score": trials.best_trial['result']['loss'],  
        "best_params": best_params,
        "best_pipeline": build_pipeline(proto_steps, best_params)
    }

# 示例用法
if __name__ == "__main__":

    prototype = ['I', 'E', 'N', 'R', 'F']
    
    # 加载数据集
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # 0: 恶性, 1: 良性
    
    # 划分训练集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 运行优化
    result = optimize_pipeline_prototype(
        proto_steps=prototype,
        X=X_train,
        y=y_train,
        max_evals=30
    )
    
    print(f"Best Accuracy: {result['best_score']:.4f}")
    print("Best Pipeline Steps:")
    for name, step in result['best_pipeline'].steps:
        print(f"{name}: {step}")