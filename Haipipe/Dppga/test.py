from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from imblearn.pipeline import Pipeline 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, KBinsDiscretizer, Binarizer
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
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
        "Ordinal": {
            "constructor": OrdinalEncoder,
            "params": {}  # 移除无效参数
        }
    },
    
    # 归一化（保持不变）
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
    
    # 离散化（保持不变）
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
    
    # 插补（保持不变）
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
    
    # 重新平衡（保持不变）
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
    
    # 特征工程（保持不变）
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
        "PCA + Select K Best": {
            "constructor": lambda**kwargs: Pipeline([
                ('pca', PCA(n_components=kwargs['pca_n'])),
                ('select', SelectKBest(k=kwargs['select_k']))
            ]),
            "params": {
                "pca_n": hp.uniform("F_comb_pca", 0.5, 0.9),
                "select_k": hp.quniform("F_comb_k", 5, 20, 5)
            }
        }
    }
}

def build_pipeline(proto_steps, best_params):
    steps = []
    for step in proto_steps:
        step_config = best_params[step]
        op_name = step_config['operator']
        op_params = step_config['params']
        
        if step == "I":
            if op_name == "Univariate":
                valid_params = {k: v for k, v in op_params.items() if k == "strategy"}
                transformer = SimpleImputer(**valid_params)
            elif op_name == "Multivariate":
                valid_params = {k: v for k, v in op_params.items() if k in ["initial_strategy", "order"]}
                transformer = IterativeImputer(**valid_params)

        elif step == "E":
            if op_name == "One Hot":
                valid_params = {k: v for k, v in op_params.items() if k in ["handle_unknown", "drop"]}
                transformer = OneHotEncoder(**valid_params)
            elif op_name == "Ordinal":
                valid_params = {}  # 旧版本无参数
                transformer = OrdinalEncoder(**valid_params)

        elif step == "N":
            if op_name == "Standard Scaler":
                valid_params = {k: v for k, v in op_params.items() if k in ["with_mean", "with_std"]}
                transformer = StandardScaler(**valid_params)
            elif op_name == "Power Transform":
                valid_params = {k: v for k, v in op_params.items() if k in ["method", "standardize"]}
                transformer = PowerTransformer(**valid_params)
            elif op_name == "MinMax Scaler":
                valid_params = {k: v for k, v in op_params.items() if k == "feature_range"}
                transformer = MinMaxScaler(**valid_params)
            elif op_name == "Robust Scaler":
                valid_params = {k: v for k, v in op_params.items() if k == "quantile_range"}
                transformer = RobustScaler(**valid_params)

        elif step == "D":
            if op_name == "KBins":
                valid_params = {
                    "n_bins": int(op_params.get("n_bins", 5)),
                    "encode": op_params.get("encode", "ordinal"),
                    "strategy": op_params.get("strategy", "quantile")
                }
                transformer = KBinsDiscretizer(**valid_params)
            elif op_name == "Binarization":
                valid_params = {k: v for k, v in op_params.items() if k == "threshold"}
                transformer = Binarizer(**valid_params)

        elif step == "R":
            if op_name == "Near Miss":
                valid_params = {
                    "version": int(op_params.get("version", 3)),
                    "n_neighbors": int(op_params.get("n_neighbors", 3))
                }
                transformer = NearMiss(**valid_params)
            elif op_name == "SMOTE":
                valid_params = {
                    "k_neighbors": int(op_params.get("k_neighbors", 5)),
                    "sampling_strategy": op_params.get("sampling_strategy", "auto")
                }
                transformer = SMOTE(**valid_params)

        elif step == "F":
            if op_name == "PCA":
                valid_params = {
                    "n_components": min(op_params.get("n_components", 0.9), 0.95),
                    "svd_solver": op_params.get("svd_solver", "auto")
                }
                transformer = PCA(**valid_params)
            elif op_name == "Select K Best":
                valid_params = {"k": int(op_params.get("k", 10))}
                transformer = SelectKBest(**valid_params)
            elif op_name == "PCA + Select K Best":
                valid_params = {
                    "pca_n": min(op_params.get("pca_n", 0.8), 0.9),
                    "select_k": int(op_params.get("select_k", 10))
                }
                transformer = Pipeline([
                    ('pca', PCA(n_components=valid_params["pca_n"])),
                    ('select', SelectKBest(k=valid_params["select_k"]))
                ])

        steps.append((f"{step}_{op_name}", transformer))
    
    return Pipeline(steps)  

def optimization_objective(params, proto_steps, X, y):
    """优化目标函数：评估管道性能"""
    try:
        # 1. 构建完整管道
        pipeline = build_pipeline(proto_steps, params)
        
        # 2. 添加分类器（可配置）
        full_pipeline = Pipeline([
            ('preprocessing', pipeline),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])
        
        # 3. 交叉验证评估
        score = cross_val_score(full_pipeline, X, y, 
                               cv=3, scoring='f1_weighted').mean()
        return {'loss': -score, 'status': STATUS_OK}
    
    except Exception as e:
        print(f"Invalid configuration: {e}")
        return {'loss': 0, 'status': STATUS_OK}

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
        
        # 获取该步骤的所有操作符
        operators = list(OPERATOR_LIBRARY[step_type].keys())
        num_ops = len(operators)
        
        # 确保每个操作符被选中的概率相等且总和为1.0
        space[step] = {
            'operator': hp.choice(f"{step}_operator", operators),
            'params': hp.pchoice(
                f"{step}_op_params",
                [(1.0 / num_ops, OPERATOR_LIBRARY[step_type][op]['params']) 
                 for op in operators]
            )
        }
    
    # 运行优化
    trials = Trials()
    best = fmin(
        fn=lambda params: optimization_objective(params, proto_steps, X, y),
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    # 解码最佳参数并后处理
    best_params = space_eval(space, best)
    for step in proto_steps:
        # 转换离散参数
        for param, val in best_params[step]['params'].items():
            if 'quniform' in param:
                best_params[step]['params'][param] = int(val)
    
    return {
        "best_score": -trials.best_trial['result']['loss'],
        "best_params": best_params,
        "best_pipeline": build_pipeline(proto_steps, best_params)
    }

# 示例用法
if __name__ == "__main__":
    # 假设输入原型为 ['I', 'E', 'N', 'R', 'F']
    prototype = ['I', 'E', 'N', 'R', 'F']
    
    # 加载数据集（示例）
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # 加载并预处理数据
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    X = X.drop(['body', 'cabin', 'boat', 'home.dest'], axis=1)
    X = X.select_dtypes(include=['number'])
    y = LabelEncoder().fit_transform(y)
    
    # 划分训练集
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 运行优化
    result = optimize_pipeline_prototype(
        proto_steps=prototype,
        X=X_train,
        y=y_train,
        max_evals=30
    )
    
    print(f"Best F1 Score: {result['best_score']:.4f}")
    print("Best Pipeline Steps:")
    for name, step in result['best_pipeline'].steps:
        print(f"{name}: {step}")