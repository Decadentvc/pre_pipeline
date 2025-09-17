from Dppga_for_call import PipelineOptimizer
import numpy as np

# 定义步骤顺序候选列表
steps_order_candidates = [
    ['impute'],  
    # ['impute', 'encode', 'normalize', 'features', 'rebalance']  
]

# 创建优化器实例
optimizer = PipelineOptimizer(
    file_path='Haipipe/data/dataset/primaryobjects_voicegender/voice.csv',
    target_column='label',
    steps_order_candidates=steps_order_candidates,
    n_trials=10,
    cv=3,
    model_choices=["RF"]  # 所有可用模型
)

# 执行优化
print("Starting pipeline optimization...")
optimizer.optimize()
print("Optimization completed!")

# 获取结果
results = optimizer.get_results()

# 打印结果摘要
print("\n" + "="*50)
print("="*5 + " DATASET INFORMATION " + "="*5)
print("="*50)
print(f"File path:          {results['dataset_info']['file_path']}")
print(f"Target column:      {results['dataset_info']['target_column']}")

print("\n" + "="*50)
print("="*5 + " BASELINE PERFORMANCE " + "="*5)
print("="*50)
for model_name, score_summary in results['baseline_performance'].items():
    print(f"{model_name:<5}: {score_summary}")

# 检查优化结果是否存在
if results['optimization_results']['accuracy'] != -np.inf:
    print("\n" + "="*50)
    print("="*5 + " OPTIMIZATION RESULTS " + "="*5)
    print("="*50)
    print(f"Best accuracy:       {results['optimization_results']['accuracy']:.4f}")
    print(f"Best model type:     {results['optimization_results']['model_type']}")
    print(f"Optimal steps order: {results['optimization_results']['best_steps_order']}")
    
    print("\n" + "="*50)
    print("="*5 + " BEST CONFIGURATION " + "="*5)
    print("="*50)
    print(results['optimization_results']['configuration'])
else:
    print("\n⚠️ WARNING: No valid optimized configuration found")

# 打印每个步骤顺序的最佳结果
print("\n" + "="*50)
print("="*5 + " BEST RESULTS PER STEPS ORDER " + "="*5)
print("="*50)

for steps_str, step_data in results['per_steps_order_best'].items():
    print(f"\nSteps: {steps_str}")
    print(f"Best accuracy:    {step_data['accuracy']}")
    print(f"Best model:       {step_data['model_type']}")
    
    if step_data['configuration']:
        print("\nBest configuration for these steps:")
        print(step_data['configuration'])
    else:
        print("No valid configuration found for these steps")