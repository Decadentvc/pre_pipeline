from prototype_evaluate import ModelManager, EnhancedPipelineRunner
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import json
from datetime import datetime

def get_model_parameter_sets(model_type):
    """根据模型类型返回5组不同的参数设置"""
    if model_type == "LR":
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
    
    elif model_type == "KNN":
        return [
            {'n_neighbors': 3, 'weights': 'uniform', 'algorithm': 'auto', 'p': 2},
            {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'kd_tree', 'p': 1},
            {'n_neighbors': 7, 'weights': 'uniform', 'algorithm': 'ball_tree', 'p': 2},
            {'n_neighbors': 10, 'weights': 'distance', 'algorithm': 'auto', 'p': 1},
            {'n_neighbors': 15, 'weights': 'uniform', 'algorithm': 'brute', 'p': 2}
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_dataset_files(dataset_dir):
    """获取数据集目录中的所有CSV文件"""
    pattern = os.path.join(dataset_dir, "*.csv")
    dataset_files = glob.glob(pattern)
    return sorted(dataset_files)

def determine_target_column(file_path):
    """自动确定目标列（假设最后一列或名为'label'的列）"""
    try:
        # 读取前几行来检查列名
        df_sample = pd.read_csv(file_path, nrows=5)
        if 'label' in df_sample.columns:
            return 'label'
        else:
            # 返回最后一列
            return df_sample.columns[-1]
    except Exception as e:
        print(f"Error determining target column for {file_path}: {e}")
        return 'label'  # 默认值

def run_single_evaluation(dataset_path, model_type, param_set, param_index, 
                         steps_order_candidates, n_trials=3, cv=2):
    """运行单个[模型, 数据集]组合的评估"""
    try:
        # 确定目标列
        target_column = determine_target_column(dataset_path)
        
        # 创建模型管理器并设置参数
        mm = ModelManager()
        mm.set_model_params(model_type, **param_set)
        
        # 禁用其他模型，只启用当前模型
        for model_key in mm.available_keys():
            if model_key != model_type:
                mm.disable(model_key)
        
        # 创建并运行增强运行器
        runner = EnhancedPipelineRunner(
            file_path=dataset_path,
            target_column=target_column,
            steps_order_candidates=steps_order_candidates,
            model_manager=mm,
            n_trials=n_trials,
            cv=cv
        )
        
        results = runner.optimize()
        
        # 提取性能指标
        accuracy = results["optimization_results"]["accuracy"]
        if accuracy == -np.inf:  # 处理失败情况
            accuracy = 0.0
            
        return {
            'accuracy': accuracy,
            'model_type': model_type,
            'param_index': param_index,
            'params': param_set,
            'dataset_path': dataset_path,
            'dataset_name': os.path.basename(dataset_path),
            'steps_order': results["optimization_results"]["best_steps_order"],
            'success': True
        }
        
    except Exception as e:
        print(f"Error evaluating {model_type} on {os.path.basename(dataset_path)}: {e}")
        return {
            'accuracy': 0.0,
            'model_type': model_type,
            'param_index': param_index,
            'params': param_set,
            'dataset_path': dataset_path,
            'dataset_name': os.path.basename(dataset_path),
            'steps_order': [],
            'success': False,
            'error': str(e)
        }

def main():
    # 配置参数
    dataset_dir = "datasets/dataset_csv_std_duplicate_removal"
    output_file = "evaluation_results_comprehensive.csv"
    steps_order_candidates = [['impute']]
    
    # 获取所有数据集文件
    dataset_files = get_dataset_files(dataset_dir)
    if not dataset_files:
        print(f"No CSV files found in {dataset_dir}")
        return
    
    print(f"Found {len(dataset_files)} datasets")
    
    # 定义要评估的模型
    model_types = ["LR", "RF", "SVM", "DT", "GBDT", "KNN"]
    
    # 准备所有要运行的组合
    all_combinations = []
    combination_id = 1
    
    for dataset_idx, dataset_path in enumerate(dataset_files, 1):
        for model_type in model_types:
            param_sets = get_model_parameter_sets(model_type)
            for param_idx, param_set in enumerate(param_sets, 1):
                all_combinations.append({
                    'combination_id': combination_id,
                    'dataset_idx': dataset_idx,
                    'dataset_path': dataset_path,
                    'model_type': model_type,
                    'param_idx': param_idx,
                    'param_set': param_set
                })
                combination_id += 1
    
    total_combinations = len(all_combinations)
    print(f"Total combinations to evaluate: {total_combinations}")
    
    # 存储所有结果
    all_results = []
    
    # 创建进度条
    with tqdm(total=total_combinations, desc="Overall Progress", 
              position=0, leave=True, ncols=100) as pbar:
        
        # 按数据集分组处理，便于计算排名
        dataset_groups = {}
        for combo in all_combinations:
            dataset_name = os.path.basename(combo['dataset_path'])
            if dataset_name not in dataset_groups:
                dataset_groups[dataset_name] = []
            dataset_groups[dataset_name].append(combo)
        
        # 对每个数据集进行处理
        for dataset_name, combinations in dataset_groups.items():
            dataset_results = []
            
            # 评估该数据集的所有组合
            for combo in combinations:
                result = run_single_evaluation(
                    dataset_path=combo['dataset_path'],
                    model_type=combo['model_type'],
                    param_set=combo['param_set'],
                    param_index=combo['param_idx'],
                    steps_order_candidates=steps_order_candidates,
                    n_trials=3,  # 减少试验次数以加快速度
                    cv=2         # 减少交叉验证折数
                )
                
                result.update({
                    'combination_id': combo['combination_id'],
                    'dataset_idx': combo['dataset_idx']
                })
                
                dataset_results.append(result)
                pbar.update(1)
                pbar.set_postfix({
                    'Dataset': f"{combo['dataset_idx']}/165",
                    'Model': combo['model_type'],
                    'Params': combo['param_idx']
                })
            
            # 计算当前数据集内的排名
            accuracies = [r['accuracy'] for r in dataset_results]
            # 创建排名（从高到低，准确率越高排名越靠前）
            sorted_indices = np.argsort(accuracies)[::-1]  # 降序排列
            ranks = np.zeros_like(sorted_indices)
            for rank, idx in enumerate(sorted_indices, 1):
                ranks[idx] = rank
            
            # 分配排名
            for i, result in enumerate(dataset_results):
                result['rank'] = int(ranks[i])
                all_results.append(result)
    
    # 保存结果到CSV文件
    save_results_to_csv(all_results, output_file)
    
    # 打印摘要统计信息
    print_summary(all_results, total_combinations)

def save_results_to_csv(results, output_file):
    """将结果保存到CSV文件"""
    # 准备数据
    data = []
    for result in results:
        row = {
            'combination_id': result['combination_id'],
            'model_type': result['model_type'],
            'param_index': result['param_index'],
            'params': json.dumps(result['params']),  # 将参数字典转为JSON字符串
            'dataset_index': result['dataset_idx'],
            'dataset_name': result['dataset_name'],
            'accuracy': result['accuracy'],
            'rank': result['rank'],
            'success': result['success'],
            'steps_order': '->'.join(result['steps_order']) if result['steps_order'] else 'N/A'
        }
        if 'error' in result:
            row['error'] = result['error']
        data.append(row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # 同时保存一个简化的版本
    simplified_file = output_file.replace('.csv', '_simplified.csv')
    simplified_cols = ['combination_id', 'model_type', 'param_index', 'dataset_name', 
                      'accuracy', 'rank', 'success']
    df[simplified_cols].to_csv(simplified_file, index=False)
    print(f"Simplified results saved to: {simplified_file}")

def print_summary(results, total_combinations):
    """打印评估摘要"""
    successful_runs = sum(1 for r in results if r['success'])
    failed_runs = total_combinations - successful_runs
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total combinations evaluated: {total_combinations}")
    print(f"Successful runs: {successful_runs} ({successful_runs/total_combinations*100:.1f}%)")
    print(f"Failed runs: {failed_runs} ({failed_runs/total_combinations*100:.1f}%)")
    
    # 按模型类型统计
    print("\nPerformance by model type:")
    model_stats = {}
    for result in results:
        if result['success']:
            model_type = result['model_type']
            if model_type not in model_stats:
                model_stats[model_type] = []
            model_stats[model_type].append(result['accuracy'])
    
    for model_type, accuracies in model_stats.items():
        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"  {model_type}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(accuracies)})")
    
    # 最佳性能组合
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['accuracy'])
        print(f"\nBest performance:")
        print(f"  Accuracy: {best_result['accuracy']:.4f}")
        print(f"  Model: {best_result['model_type']} (Param set #{best_result['param_index']})")
        print(f"  Dataset: {best_result['dataset_name']}")
        print(f"  Rank: #{best_result['rank']} in its dataset")

if __name__ == "__main__":
    # 创建输出目录（如果不存在）
    os.makedirs("results", exist_ok=True)
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    print("Starting comprehensive model-dataset evaluation...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        main()
        print(f"\nEvaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()