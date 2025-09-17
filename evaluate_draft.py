import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from Dppga_for_call import PipelineOptimizer  

class FeatureExtractor:
    @staticmethod
    def _global_features(df):
        return {
            'num_rows': len(df),
            'num_cols': df.shape[1],
            'total_missing': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'density': df.count().sum() / (df.size + 1e-9)
        }

    @staticmethod
    def _dtype_features(df):
        dtypes = df.dtypes.astype(str).value_counts().to_dict()
        total_cols = len(df.columns)
        return {
            'numeric_ratio': dtypes.get('float64', 0) / total_cols,
            'category_ratio': dtypes.get('category', 0) / total_cols,
            'object_ratio': dtypes.get('object', 0) / total_cols,
            'int_ratio': dtypes.get('int64', 0) / total_cols
        }

    @staticmethod
    def _column_aggregations(df):
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            return {}
            
        mean_of_means = numeric_df.mean().mean() if not numeric_df.empty else 0
        std_of_stds = numeric_df.std().std() if not numeric_df.empty else 0
        
        # 计算偏度时跳过常数列
        skew_values = numeric_df.apply(lambda x: stats.skew(x) if x.nunique() > 1 else 0)
        kurtosis_values = numeric_df.apply(lambda x: stats.kurtosis(x) if x.nunique() > 1 else 0)
        
        # 离群值检测
        outlier_mask = numeric_df.apply(
            lambda x: (np.abs(x - x.mean()) > 3*x.std()) if x.std() > 0 else pd.Series(False, index=x.index)
        )
        outlier_ratio = outlier_mask.mean().mean()
        
        return {
            'mean_of_means': mean_of_means,
            'std_of_stds': std_of_stds,
            'skewness_avg': skew_values.mean(),
            'kurtosis_avg': kurtosis_values.mean(),
            'outlier_ratio': outlier_ratio
        }

    @staticmethod
    def _entropy_features(df, num_bins=10):
        entropies = []
        for col in df.columns:
            col_data = df[col].dropna()
            unique_count = col_data.nunique()
            if len(col_data) < 2 or unique_count == 1:
                continue
                
            try:
                if pd.api.types.is_numeric_dtype(col_data):
                    n = len(col_data)
                    bins = max(min(num_bins, int(np.log2(n)) + 1) if n > 0 else 1, 1)
                    p_data = col_data.value_counts(bins=bins, normalize=True)
                else:
                    p_data = col_data.value_counts(normalize=True)
                entropies.append(stats.entropy(p_data))
            except Exception:
                continue
        
        return {
            'avg_entropy': np.mean(entropies) if entropies else 0,
            'max_entropy': np.max(entropies) if entropies else 0,
            'entropy_variance': np.var(entropies) if entropies else 0,
            'computed_entropy_cols': len(entropies)
        }

    @staticmethod
    def _structural_features(df):
        time_cols = [col for col in df.columns if ('date' in col.lower()) or (df[col].dtype == 'datetime64[ns]')]
        
        high_card = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if not pd.isna(unique_ratio) and unique_ratio > 0.9:
                high_card.append(col)
                
        return {
            'has_time_series': int(len(time_cols) > 0),
            'high_cardinality_ratio': len(high_card)/len(df.columns)
        }

    @staticmethod
    def _model_features(df, n_samples=1000):
        sample_size = min(n_samples, len(df))
        if sample_size < 2:
            return {}
            
        try:
            sample_df = df.sample(sample_size, random_state=42).select_dtypes(include=np.number)
            if sample_df.empty:
                return {}
                
            filled_df = sample_df.fillna(sample_df.mean())
            if filled_df.isnull().any().any():
                filled_df = filled_df.fillna(0)
                
            valid_cols = filled_df.columns[filled_df.nunique() > 1]
            if len(valid_cols) < 1:
                return {}
                
            model = IsolationForest(random_state=42)
            model.fit(filled_df[valid_cols])
            scores = model.decision_function(filled_df[valid_cols])
            
            return {
                'anomaly_score': np.mean(scores)
            }
        except Exception:
            return {}

    @staticmethod
    def extract_features(file_path):
        try:
            encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, nrows=10000, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue
                    
            if df is None or df.empty:
                return None
                
            df.columns = [f"col_{i}" if isinstance(name, int) or name == '' else str(name).strip() for i, name in enumerate(df.columns)]
            
            features = {}
            features.update(FeatureExtractor._global_features(df))
            features.update(FeatureExtractor._dtype_features(df))
            features.update(FeatureExtractor._column_aggregations(df))
            features.update(FeatureExtractor._entropy_features(df))
            features.update(FeatureExtractor._structural_features(df))
            features.update(FeatureExtractor._model_features(df))
            
            return features
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

def analyze_datasets(directory='dataset_temp', output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    candidates = [['discretize', 'features'], ['features', 'discretize']]
    candidate_str = [f"{candidates[0][0]}->{candidates[0][1]}", 
                    f"{candidates[1][0]}->{candidates[1][1]}"]
    model_choices = ["RF", "KNN", "NB"]
    
    processed_count = 0
    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue
            
        file_path = os.path.join(directory, filename)
        print(f"\nProcessing dataset: {filename}")
        
        features = FeatureExtractor.extract_features(file_path)
        if not features:
            print(f"⛔ Skipped {filename} due to feature extraction failure")
            continue
            
        print(f"✅ Extracted {len(features)} features for {filename}")
        dataset_results = {'dataset': filename, **features}
        
        for model_name in model_choices:
            try:
                print(f"🔧 Running pipeline optimization for {filename} with {model_name}...")
                optimizer = PipelineOptimizer(
                    file_path=file_path,
                    target_column='label',
                    steps_order_candidates=candidates,
                    n_trials=30,
                    cv=3,
                    model_choices=[model_name]
                )
                optimizer.optimize()
                eval_result = optimizer.get_results()
                print(f"🏆 Optimization completed for {filename} with {model_name}")
                
                # 获取基准性能
                baseline_score = float(eval_result['baseline_performance'][model_name].strip())
                
                # 获取每种步骤顺序的最佳性能
                step1_acc = eval_result['per_steps_order_best'][candidate_str[0]]['accuracy']
                step2_acc = eval_result['per_steps_order_best'][candidate_str[1]]['accuracy']
                
                # 获取全局最佳优化性能
                optimized_score = eval_result['optimization_results']['accuracy']
                best_step_order = eval_result['optimization_results']['best_steps_order']
                
                # 确定策略
                if step1_acc == step2_acc and step1_acc > baseline_score:
                    strategy = 'Draw'  # 两种顺序效果相同
                elif step1_acc > step2_acc:
                    strategy = 'discretize_first'
                elif step1_acc < step2_acc:
                    strategy = 'features_first'
                elif step1_acc <= baseline_score and step2_acc <= baseline_score:
                    strategy = 'baseline'  # 优化结果不如基线

                
                improvement = optimized_score - baseline_score
                print(f"  Baseline: {baseline_score:.4f}, " 
                      f"Order1 {candidate_str[0]}: {step1_acc:.4f}, "
                      f"Order2 {candidate_str[1]}: {step2_acc:.4f}, "
                      f"Optimized: {optimized_score:.4f}, "
                      f"Strategy: {strategy}")
                
                # 记录详细结果
                dataset_results.update({
                    f'baseline_{model_name}': baseline_score,
                    f'step1_acc_{model_name}': step1_acc,
                    f'step2_acc_{model_name}': step2_acc,
                    f'optimized_{model_name}': optimized_score,
                    f'improvement_{model_name}': improvement,
                    f'strategy_{model_name}': strategy,
                    f'best_step_order_{model_name}': best_step_order
                })
                
            except Exception as e:
                print(f"❌ Evaluation failed for {filename} with {model_name}: {str(e)}")
                # 添加失败标记
                dataset_results.update({
                    f'baseline_{model_name}': np.nan,
                    f'step1_acc_{model_name}': np.nan,
                    f'step2_acc_{model_name}': np.nan,
                    f'optimized_{model_name}': np.nan,
                    f'improvement_{model_name}': np.nan,
                    f'strategy_{model_name}': 'failed',
                    f'best_step_order_{model_name}': 'N/A'
                })
                continue
        
        results.append(dataset_results)
        processed_count += 1
        print(f"✔️ Completed {processed_count} datasets")
    
    print(f"\nTotal datasets processed: {processed_count}")
    
    if not results:
        print("No valid datasets processed")
        return
    
    df = pd.DataFrame(results)
    
    # 保存结果
    dataset_info_path = os.path.join(output_dir, 'dataset_strategy_results.csv')
    df.to_csv(dataset_info_path, index=False)
    print(f"\n💾 Saved dataset information to: {dataset_info_path}")
    
    # 准备PCA数据
    non_feature_cols = ['dataset'] + [col for col in df.columns if any(x in col for x in ['_acc_', 'baseline', 'optimized', 'improvement', 'strategy', 'config'])]
    features = [col for col in df.columns if col not in non_feature_cols]
    
    if not features:
        print("No features available for PCA")
        return
    
    # 修复：确保所有特征都是数值类型
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 修复：检查并移除非数值列
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if not non_numeric_cols.empty:
        print(f"⚠️ Found non-numeric columns: {list(non_numeric_cols)}. Removing them for PCA.")
        X = X.drop(columns=non_numeric_cols)
    
    # 修复：确保所有值都是标量
    for col in X.columns:
        if any(isinstance(x, (list, tuple, dict)) for x in X[col]):
            print(f"⚠️ Column '{col}' contains non-scalar values. Converting to numeric.")
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 移除方差为0的列
    nonzero_var_cols = X.columns[X.var() > 0]
    if len(nonzero_var_cols) == 0:
        print("No features with non-zero variance for PCA")
        return
    X = X[nonzero_var_cols]
    
    if X.shape[0] > 1:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # PCA降维
    n_components = min(2, X_scaled.shape[1], max(1, X_scaled.shape[0]-1))
    if n_components <= 0:
        print("Not enough data points for PCA")
        return
    
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1] if n_components > 1 else np.zeros(len(coords))
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('3al_50datasets_fd', 
                fontsize=20, y=1.05)
    
    color_map = {
        'baseline': '#FF6B6B',     # 红色 - 基准策略
        'discretize_first': "#190AEB",  # 蓝色 - discretize_first策略
        'features_first': "#28EA38",   # 绿色 - features_first策略
        'Draw': 'black',           # 黑色 - 两种策略效果相同
        'failed': 'gray'           # 灰色 - 失败情况
    }
    
    for i, model_name in enumerate(model_choices):
        ax = axes[i]
        strategy_col = f'strategy_{model_name}'
        
        # 为每种策略绘制点
        for strategy, color in color_map.items():
            mask = (df[strategy_col] == strategy) & (strategy != 'failed')  # 排除失败的点
            if mask.any():
                ax.scatter(
                    df.loc[mask, 'x'], 
                    df.loc[mask, 'y'], 
                    c=color, 
                    label=strategy, 
                    alpha=0.8, 
                    edgecolors='w',
                    s=120
                )
        
        # 添加点标签（带改进值）
        for idx, row in df.iterrows():
            strategy = row[strategy_col]
            if strategy != 'failed':
                improvement = row.get(f'improvement_{model_name}', 0)
                label = f"{row['dataset']}\nΔ={improvement:.3f}" if not np.isnan(improvement) else row['dataset']
                
                ax.annotate(
                    label, 
                    (row['x'], row['y']), 
                    xytext=(7, 7), 
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                )
        
        # 设置图表信息
        xlabel = f"PC1 ({explained_variance[0]*100:.1f}%)" if i == 0 else ""
        ylabel = f"PC2 ({explained_variance[1]*100:.1f}%)" if i == 1 else ""
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{model_name} Algorithm', fontsize=16, pad=12)
        ax.legend(title='Optimal Strategy', title_fontsize='13', loc='best')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)
    
    # 保存和显示
    visualization_path = os.path.join(output_dir, 'dataset_strategy_visualization.png')
    plt.tight_layout()
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ Saved visualization to: {visualization_path}")
    plt.show()
    
    # 策略分析报告
    strategy_report = []
    for model_name in model_choices:
        if f'strategy_{model_name}' in df.columns:
            model_df = df[df[f'strategy_{model_name}'] != 'failed'].copy()
            
            if not model_df.empty:
                # 统计策略分布
                strategy_counts = model_df[f'strategy_{model_name}'].value_counts().to_dict()
                
                # 计算平均改进
                avg_improvement = model_df[f'improvement_{model_name}'].mean()
                
                # 准备数据行
                report_row = {
                    'algorithm': model_name,
                    'datasets_evaluated': len(model_df),
                    'baseline_wins': strategy_counts.get('baseline', 0),
                    'discretize_first_wins': strategy_counts.get('discretize_first', 0),
                    'features_first_wins': strategy_counts.get('features_first', 0),
                    'draws': strategy_counts.get('Draw', 0),
                    'avg_improvement': avg_improvement,
                    'step1_better_count': (model_df[f'step1_acc_{model_name}'] > model_df[f'step2_acc_{model_name}']).sum(),
                    'step2_better_count': (model_df[f'step2_acc_{model_name}'] > model_df[f'step1_acc_{model_name}']).sum(),
                    'equal_performance': (model_df[f'step1_acc_{model_name}'] == model_df[f'step2_acc_{model_name}']).sum()
                }
                strategy_report.append(report_row)
    
    if strategy_report:
        report_df = pd.DataFrame(strategy_report)
        report_path = os.path.join(output_dir, 'strategy_analysis_report.csv')
        report_df.to_csv(report_path, index=False)
        print(f"\n📊 Saved strategy analysis report to: {report_path}")
        
        # 打印简要报告
        print("\n=== Strategy Analysis Summary ===")
        print(report_df)
    
    return df

if __name__ == "__main__":
    results_df = analyze_datasets(
        directory='datasets/dataset_std_dr_demo50',
        output_dir='analysis_results/3al_fd_50'
    )
    
    if results_df is not None:
        print("\nAnalysis completed successfully!")
    else:
        print("\nNo valid results were generated")