from prototype_evaluate import ModelManager, EnhancedPipelineRunner
import numpy as np
import os
import glob
import csv
from tqdm import tqdm

# 定义获取模型参数的函数
def get_model_params(model_type):
    if model_type == "KNN":
        # 为KNN定义5组参数，基于示例保留关键参数
        return [
            {'n_neighbors': 3, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30, 'p': 2, 'metric': 'minkowski'},
            {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'ball_tree', 'leaf_size': 40, 'p': 1, 'metric': 'manhattan'},
            {'n_neighbors': 7, 'weights': 'uniform', 'algorithm': 'kd_tree', 'leaf_size': 20, 'p': 2, 'metric': 'euclidean'},
            {'n_neighbors': 10, 'weights': 'distance', 'algorithm': 'brute', 'leaf_size': 50, 'p': 2, 'metric': 'chebyshev'},
            {'n_neighbors': 15, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30, 'p': 1, 'metric': 'minkowski'}
        ]
    elif model_type == "LR":
        return [
            {'penalty': 'l1', 'C': 0.01, 'solver': 'liblinear', 'multi_class': 'ovr'},
            {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'multi_class': 'multinomial'},
            {'penalty': 'elasticnet', 'C': 1.0, 'solver': 'saga', 'multi_class': 'ovr'},
            {'penalty': None, 'C': 10.0, 'solver': 'newton-cg', 'multi_class': 'multinomial'},
            {'penalty': 'l2', 'C': 100.0, 'solver': 'sag', 'multi_class': 'ovr'}
        ]
    elif model_type == "RF":
        return [
            {'n_estimators': 50, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2'},
            {'n_estimators': 200, 'criterion': 'gini', 'max_depth': None, 'max_features': 0.3},
            {'n_estimators': 150, 'criterion': 'entropy', 'max_depth': 15, 'max_features': 0.5},
            {'n_estimators': 100, 'criterion': 'gini', 'max_depth': 20, 'max_features': None}
        ]
    elif model_type == "SVM":
        return [
            {'C': 0.1, 'kernel': 'linear', 'gamma': 'scale'},
            {'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto'},
            {'C': 10.0, 'kernel': 'poly', 'degree': 2, 'gamma': 0.1},
            {'C': 0.5, 'kernel': 'sigmoid', 'gamma': 'scale'},
            {'C': 100.0, 'kernel': 'poly', 'degree': 3, 'gamma': 'auto'}
        ]
    elif model_type == "DT":
        return [
            {'criterion': 'gini', 'max_depth': 3, 'max_features': 'sqrt'},
            {'criterion': 'entropy', 'max_depth': 7, 'max_features': 'log2'},
            {'criterion': 'gini', 'max_depth': None, 'max_features': 0.5},
            {'criterion': 'entropy', 'max_depth': 12, 'max_features': None},
            {'criterion': 'gini', 'max_depth': 5, 'max_features': 0.7}
        ]
    elif model_type == "GBDT":
        return [
            {'learning_rate': 0.01, 'n_estimators': 50, 'max_depth': 3, 'max_features': 'sqrt'},
            {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 5, 'max_features': 'log2'},
            {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 7, 'max_features': 0.3},
            {'learning_rate': 0.2, 'n_estimators': 150, 'max_depth': None, 'max_features': 0.5},
            {'learning_rate': 0.15, 'n_estimators': 100, 'max_depth': 4, 'max_features': None}
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# 定义步骤顺序候选列表
steps_order_candidates = [
    ['impute'],
    ['impute', 'encode', 'normalize', 'features', 'rebalance']
]

# 模型列表
model_types = ["KNN", "LR", "RF", "SVM", "DT", "GBDT"]

# 数据集目录
datasets_dir = 'datasets/dataset_csv_std_duplicate_removal'
dataset_files = sorted(glob.glob(os.path.join(datasets_dir, '*.csv')))
assert len(dataset_files) == 165, f"Expected 165 datasets, found {len(dataset_files)}"

# 输出文件
output_file = 'evaluation_results.csv'

# 总组合数
num_models = len(model_types)
num_param_groups = 5
num_datasets = len(dataset_files)
total_combinations = num_models * num_param_groups * num_datasets  # 4950

# 初始化CSV文件
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['序号', '模型种类', '模型参数组合编号', '模型具体参数设置', '数据集编号', '数据集名', 'steps_order_candidates的准确率排名'])

# 进度条
progress_bar = tqdm(total=total_combinations, desc="Processing combinations", position=0, leave=True)

# 序号
seq = 1

# 遍历所有组合
for model_type in model_types:
    param_groups = get_model_params(model_type)
    for param_id, param_dict in enumerate(param_groups, start=1):
        for dataset_id, dataset_file in enumerate(dataset_files, start=1):
            dataset_name = os.path.basename(dataset_file)
            
            # 为每个steps_order_candidate运行优化，收集准确率
            order_accuracies = []
            for order in steps_order_candidates:
                mm = ModelManager()
                
                # 禁用所有模型，只启用当前模型
                for key in mm.available_keys():
                    mm.disable(key)
                mm.enable(model_type)
                
                # 设置模型参数
                mm.set_model_params(model_type, **param_dict)
                
                # 创建运行器（假设所有数据集的目标列为'label'）
                runner = EnhancedPipelineRunner(
                    file_path=dataset_file,
                    target_column='label',
                    steps_order_candidates=[order],  # 单个order以获取其性能
                    model_manager=mm,
                    n_trials=5,
                    cv=3
                )
                
                # 运行优化
                results = runner.optimize()
                
                # 获取准确率
                acc = results["optimization_results"]["accuracy"]
                if acc == -np.inf:
                    acc = 0.0  # 或处理为无效
                order_str = '->'.join(order)
                order_accuracies.append((order_str, acc))
            
            # 对准确率排序（降序）
            order_accuracies.sort(key=lambda x: x[1], reverse=True)
            
            # 构建排名字符串
            ranking_str = '; '.join([f"{order}: {acc:.4f}" for order, acc in order_accuracies])
            
            # 写入CSV
            with open(output_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    seq,
                    model_type,
                    param_id,
                    str(param_dict),
                    dataset_id,
                    dataset_name,
                    ranking_str
                ])
            
            # 更新进度
            progress_bar.update(1)
            seq += 1

# 关闭进度条
progress_bar.close()

print("Evaluation completed! Results saved to", output_file)