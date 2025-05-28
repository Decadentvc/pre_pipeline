from Dppga_for_call import PipelineOptimizer

optimizer = PipelineOptimizer(
    file_path='Haipipe/data/dataset/primaryobjects_voicegender/voice.csv',
    target_column='label',
    steps_order_candidates=[
        ['impute', 'encode', 'normalize', 'rebalance', 'features'],
        ['impute', 'encode', 'normalize', 'features', 'rebalance']
    ],
    n_trials=50
)

optimizer.optimize()
results = optimizer.get_results()

print(f"基准准确率: {results['Baseline Accuracy']}")
print(f"最佳优化准确率: {results['Best Optimized Accuracy']}")
print(f"最优步骤顺序: {results['Effective Pipeline Prototype']}")