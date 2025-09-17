from prototype_evaluate import ModelManager, EnhancedPipelineRunner
import numpy as np

# 定义步骤顺序候选列表
steps_order_candidates = [
    ['impute'],  
    # ['impute', 'encode', 'normalize', 'features', 'rebalance']  
]

# 创建模型管理器并自定义模型参数
mm = ModelManager()

# KNN 参数设置
mm.set_model_params("KNN",
    n_neighbors=5,          # n_neighbors: 邻居数量，可选值：整数(通常3-15)
    weights='uniform',       # weights: 邻居权重，可选值：'uniform'(等权重), 'distance'(距离加权)
    algorithm='auto',        # algorithm: 最近邻算法，可选值：'auto','ball_tree','kd_tree','brute'
    leaf_size=30,            # leaf_size: 叶节点大小，可选值：整数(影响构建和查询效率)
    p=2,                     # p: 闵可夫斯基距离参数，可选值：1(曼哈顿距离),2(欧氏距离)
    metric='minkowski'       # metric: 距离度量，可选值：'minkowski','euclidean','manhattan','chebyshev'
)

# Logistic Regression 参数设置
mm.set_model_params("LR",
    penalty='l2',            # penalty: 正则化类型，可选值：'l1','l2','elasticnet','none'
    C=1.0,                   # C: 正则化强度，可选值：浮点数(值越小正则化越强)
    fit_intercept=True,      # fit_intercept: 是否拟合截距项，可选值：布尔值
    solver='lbfgs',          # solver: 优化算法，可选值：'newton-cg','lbfgs','liblinear','sag','saga'
    max_iter=100,            # max_iter: 最大迭代次数，可选值：整数
    multi_class='auto',      # multi_class: 多分类策略，可选值：'ovr'(一对多), 'multinomial'(多项式)
    class_weight=None        # class_weight: 类别权重，可选值：None, 'balanced', 字典形式权重
)

# Random Forest 参数设置
mm.set_model_params("RF",
    n_estimators=100,        # n_estimators: 树的数量，可选值：整数
    criterion='gini',        # criterion: 分裂标准，可选值：'gini','entropy'
    max_depth=None,          # max_depth: 树的最大深度，可选值：整数或None
    min_samples_split=2,     # min_samples_split: 分裂所需最小样本数，可选值：整数或浮点数
    min_samples_leaf=1,      # min_samples_leaf: 叶节点最小样本数，可选值：整数或浮点数
    max_features='sqrt',     # max_features: 考虑的最大特征数，可选值：'auto','sqrt','log2',整数或浮点数
    bootstrap=True,          # bootstrap: 是否使用bootstrap采样，可选值：布尔值
    oob_score=False,         # oob_score: 是否使用袋外样本评估，可选值：布尔值
    class_weight=None        # class_weight: 类别权重，可选值：None, 'balanced', 'balanced_subsample'
)

# SVM 参数设置
mm.set_model_params("SVM",
    C=1.0,                   # C: 正则化参数，可选值：浮点数
    kernel='rbf',            # kernel: 核函数类型，可选值：'linear','poly','rbf','sigmoid','precomputed'
    degree=3,                # degree: 多项式核的阶数，可选值：整数
    gamma='scale',           # gamma: 核函数系数，可选值：'scale','auto',浮点数
    coef0=0.0,               # coef0: 核函数中的独立项，可选值：浮点数
    shrinking=True,          # shrinking: 是否使用收缩启发式，可选值：布尔值
    class_weight=None        # class_weight: 类别权重，可选值：None, 'balanced'
)

# Decision Tree 参数设置
mm.set_model_params("DT",
    criterion='gini',        # criterion: 分裂标准，可选值：'gini','entropy'
    splitter='best',         # splitter: 分裂策略，可选值：'best','random'
    max_depth=None,          # max_depth: 树的最大深度，可选值：整数或None
    min_samples_split=2,     # min_samples_split: 分裂所需最小样本数，可选值：整数或浮点数
    min_samples_leaf=1,      # min_samples_leaf: 叶节点最小样本数，可选值：整数或浮点数
    max_features=None,       # max_features: 考虑的最大特征数，可选值：'auto','sqrt','log2',整数或浮点数
    class_weight=None        # class_weight: 类别权重，可选值：None, 'balanced'
)

# GBDT 参数设置
mm.set_model_params("GBDT",
    loss='log_loss',         # loss: 损失函数，可选值：'log_loss','exponential'
    learning_rate=0.1,       # learning_rate: 学习率，可选值：浮点数(通常0.01-0.2)
    n_estimators=100,        # n_estimators: 树的数量，可选值：整数
    subsample=1.0,           # subsample: 样本采样比例，可选值：浮点数(0.0-1.0)
    criterion='friedman_mse',# criterion: 分裂标准，可选值：'friedman_mse','squared_error','mse','mae'
    min_samples_split=2,     # min_samples_split: 分裂所需最小样本数，可选值：整数或浮点数
    min_samples_leaf=1,      # min_samples_leaf: 叶节点最小样本数，可选值：整数或浮点数
    max_depth=3,             # max_depth: 树的最大深度，可选值：整数
    max_features=None,       # max_features: 考虑的最大特征数，可选值：'auto','sqrt','log2',整数或浮点数
    validation_fraction=0.1, # validation_fraction: 早停验证集比例，可选值：浮点数(0.0-1.0)
    n_iter_no_change=None,   # n_iter_no_change: 早停迭代次数，可选值：整数或None
    tol=0.0001               # tol: 早停容忍度，可选值：浮点数
)

# （可选）禁用某个模型
# mm.disable("NB")  # 禁用Naive Bayes


# 只启用并配置我们想要评估的模型（例如RF）
# for model_key in mm.available_keys():
#     mm.disable(model_key)
# mm.enable("RF")
# mm.set_model_params("RF", n_estimators=150, max_depth=8, min_samples_split=5)


# 创建并运行增强运行器
print("Starting enhanced pipeline optimization...")
runner = EnhancedPipelineRunner(
    file_path='Haipipe/data/dataset/primaryobjects_voicegender/voice.csv',
    target_column='label',
    steps_order_candidates=steps_order_candidates,
    model_manager=mm,
    n_trials=5,   # Optuna迭代次数
    cv=3
)

results = runner.optimize()
print("Optimization completed!")

# 打印结果摘要
print("\n" + "="*50)
print("="*5 + " DATASET INFORMATION " + "="*5)
print("="*50)
print(f"File path:          {runner.file_path}")
print(f"Target column:      {runner.target_column}")
print(f"Number of samples:  {runner.X.shape[0]}")
print(f"Number of features: {runner.X.shape[1]}")
print(f"Number of classes:  {len(np.unique(runner.y))}")

print("\n" + "="*50)
print("="*5 + " BASELINE PERFORMANCE " + "="*5)
print("="*50)
for model_name, score_info in results["baseline_performance"].items():
    if 'error' in score_info:
        print(f"{model_name:<5}: Error - {score_info['error']}")
    else:
        print(f"{model_name:<5}: Mean Accuracy = {score_info['mean_accuracy']:.4f} ± {score_info['std_accuracy']:.4f}")

# 检查优化结果是否存在
if results["optimization_results"]["accuracy"] != -np.inf:
    print("\n" + "="*50)
    print("="*5 + " OPTIMIZATION RESULTS " + "="*5)
    print("="*50)
    print(f"Best accuracy:       {results['optimization_results']['accuracy']:.4f}")
    print(f"Best model type:     {results['optimization_results']['model_type']}")
    print(f"Optimal steps order: {'->'.join(results['optimization_results']['best_steps_order'])}")
    
    print("\n" + "="*50)
    print("="*5 + " BEST CONFIGURATION " + "="*5)
    print("="*50)
    print(results['optimization_results']['configuration'])
else:
    print("\n⚠️ WARNING: No valid optimized configuration found")

# 打印每个模型的最佳结果
print("\n" + "="*50)
print("="*5 + " BEST RESULTS PER MODEL " + "="*5)
print("="*50)

for model_name, result in results["per_model_best"].items():
    print(f"\nModel: {model_name}")
    print(f"Best accuracy:    {result['accuracy']:.4f}")
    print(f"Steps order:      {'->'.join(result['steps_order'])}")
    
    if result['config']:
        print("\nBest configuration for this model:")
        print(result['config'])
    else:
        print("No valid configuration found for this model")

# 打印整体最佳结果
print("\n" + "="*50)
print("="*5 + " OVERALL BEST RESULT " + "="*5)
print("="*50)

if results["optimization_results"]["accuracy"] != -np.inf:
    best_result = results["optimization_results"]
    print(f"Best model:       {best_result['model_type']}")
    print(f"Best accuracy:    {best_result['accuracy']:.4f}")
    print(f"Steps order:      {'->'.join(best_result['best_steps_order'])}")
    
    print("\nBest configuration overall:")
    print(best_result['configuration'])
else:
    print("⚠️ WARNING: No valid optimized configuration found")