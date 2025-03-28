# 模块2：基于SMBO的管道实例化优化（核心代码）
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.space_eval import space_eval
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 预定义的转换操作符库
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
            "params": {
                "handle_unknown": hp.choice("E_ord_handle", ["use_encoded_value"]),
                "unknown_value": hp.uniform("E_unknown", -1, 1)
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
        "PCA + Select K Best": {
            "constructor": lambda​**kwargs: Pipeline([
                ('pca', PCA(n_components=kwargs['pca_n'])),
                ('select', SelectKBest(k=kwargs['select_k']))
            ]),
            "params": {
                "pca_n": [1, 2, 3, 4],  # PCA 主成分数
                "select_k": [1, 2, 3, 4]  # 选择 Top-K 特征
            }
        }
    }
}

def build_pipeline(proto_steps, params):
    """根据原型步骤和优化参数构建具体管道"""
    pipeline_steps = []
    
    for step in proto_steps:
        step_type = {
            'I': "Imputation(I)",
            'E': "Encoding(E)",
            'N': "Normalization(N)",
            'D': "Discretization(D)",
            'R': "Rebalancing(R)",
            'F': "Feat.Eng.(F)"
        }[step]
        
        # 获取该步骤的配置
        step_config = params[step]
        operator_name = step_config['operator']
        operator_config = OPERATOR_LIBRARY[step_type][operator_name]
        
        # 实例化转换器
        constructor = operator_config["constructor"]
        operator_params = {k: v for k, v in step_config['params'].items() 
                          if k in operator_config["params"]}
        
        # 处理特殊参数类型
        if operator_name == "PCA + Select K Best":
            # 处理管道嵌套参数
            transformer = constructor(**operator_params)
        else:
            # 常规实例化
            transformer = constructor(**operator_params)
        
        pipeline_steps.append((f"{step}_{operator_name}", transformer))
    
    return Pipeline(pipeline_steps)

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
    # 构建参数空间
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
        
        space[step] = {
            'operator': hp.choice(f"{step}_operator", 
                                list(OPERATOR_LIBRARY[step_type].keys())),
            'params': OPERATOR_LIBRARY[step_type][
                hp.pchoice(f"{step}_op", 
                          [(1.0/len(OPERATOR_LIBRARY[step_type]), k) 
                           for k in OPERATOR_LIBRARY[step_type]])]["params"]
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
    
    # 解码最佳参数
    best_params = space_eval(space, best)
    
    # 后处理参数类型
    for step in proto_steps:
        op_name = best_params[step]['operator']
        step_type = {
            'I': "Imputation(I)",
            'E': "Encoding(E)",
            'N': "Normalization(N)",
            'D': "Discretization(D)",
            'R': "Rebalancing(R)",
            'F': "Feat.Eng.(F)"
        }[step]
        
        # 转换离散参数
        for param, val in best_params[step]['params'].items():
            if "quniform" in param:
                best_params[step]['params'][param] = int(val)
    
    return {
        "best_score": -trials.best_trial['result']['loss'],
        "best_params": best_params,
        "best_pipeline": build_pipeline(proto_steps, best_params)
    }

# 示例用法
if __name__ == "__main__":
    # 假设输入原型为 ['I', 'E', 'N', 'F']
    prototype = ['I', 'E', 'N', 'R','F']
    
    # 加载数据集（示例）
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    
    # 运行优化
    result = optimize_pipeline_prototype(
        proto_steps=prototype,
        X=X,
        y=y,
        max_evals=30
    )
    
    print(f"Best F1 Score: {result['best_score']:.4f}")
    print("Best Pipeline Steps:")
    for name, step in result['best_pipeline'].steps:
        print(f"{name}: {step}")