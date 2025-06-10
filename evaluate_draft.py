import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest

# 假设Dppga_for_call模块已正确实现
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
        # 修复: 使用正确的 astype() 而不是 ast()
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
            
        # 更安全的统计计算
        mean_of_means = numeric_df.mean().mean() if not numeric_df.empty else 0
        std_of_stds = numeric_df.std().std() if not numeric_df.empty else 0
        
        # 计算偏度时跳过常数列
        skew_values = numeric_df.apply(lambda x: stats.skew(x) if x.nunique() > 1 else 0)
        kurtosis_values = numeric_df.apply(lambda x: stats.kurtosis(x) if x.nunique() > 1 else 0)
        
        # 更安全的离群值检测
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
                    # 避免零除错误并确保合理的分箱数
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
                
            # 填充缺失值并计算异常得分
            filled_df = sample_df.fillna(sample_df.mean())
            if filled_df.isnull().any().any():
                filled_df = filled_df.fillna(0)
                
            # 跳过低方差列
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
            # 尝试不同编码来读取文件
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
                
            # 重命名列以解决无效列名问题
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
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化结果收集器
    results = []
    candidates = [['rebalance', 'features'], ['features', 'rebalance']]
    
    # 遍历数据集目录
    processed_count = 0
    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue
            
        file_path = os.path.join(directory, filename)
        print(f"\nProcessing dataset: {filename}")
        
        # 特征提取
        features = FeatureExtractor.extract_features(file_path)
        if not features:
            print(f"⛔ Skipped {filename} due to feature extraction failure")
            continue
            
        print(f"✅ Extracted {len(features)} features for {filename}")
        
        # 管道优化评估
        try:
            print(f"🔧 Running pipeline optimization for {filename}...")
            optimizer = PipelineOptimizer(
                file_path=file_path,
                target_column='label',
                steps_order_candidates=candidates,
                n_trials=30
            )
            optimizer.optimize()
            eval_result = optimizer.get_results()
            print(f"🏆 Optimization completed for {filename}")
            
            # 确定获胜策略
            baseline = eval_result['Baseline Accuracy']
            optimized = eval_result['Best Optimized Accuracy']
            pipeline = eval_result['Effective Pipeline Prototype']
            
            if baseline > optimized:
                strategy = 'baseline'
            else:
                strategy = 'rebalance_first' if pipeline == candidates[0] else 'features_first'
            
            improvement = optimized - baseline
            print(f"  Baseline: {baseline:.4f}, Optimized: {optimized:.4f}, Improvement: {improvement:.4f}")
            print(f"  Winning strategy: {strategy}")
            
            # 记录结果
            results.append({
                'dataset': filename,
                **features,
                'strategy': strategy,
                'baseline_accuracy': baseline,
                'optimized_accuracy': optimized,
                'accuracy_improvement': improvement,
                'pipeline_config': eval_result['Best Configuration']
            })
            
            processed_count += 1
        except Exception as e:
            print(f"❌ Evaluation failed for {filename}: {str(e)}")
            continue
    
    print(f"\nTotal datasets processed: {processed_count}/{len(os.listdir(directory))} csv files")
    
    # 转换为DataFrame并处理数据
    if not results:
        print("No valid datasets processed")
        return
    
    df = pd.DataFrame(results)
    
    # 保存数据集信息到CSV
    dataset_info_path = os.path.join(output_dir, 'dataset_strategy_results.csv')
    df.to_csv(dataset_info_path, index=False)
    print(f"\n💾 Saved dataset information to: {dataset_info_path}")
    
    # 为PCA准备数据
    non_feature_cols = ['dataset', 'strategy', 'baseline_accuracy', 
                       'optimized_accuracy', 'pipeline_config', 'accuracy_improvement']
    features = [col for col in df.columns if col not in non_feature_cols]
    
    if not features:
        print("No features available for PCA")
        return
    
    X = df[features].fillna(0)
    
    # 移除方差为0的列
    nonzero_var_cols = X.columns[X.var() > 0]
    if len(nonzero_var_cols) == 0:
        print("No features with non-zero variance for PCA")
        return
    
    X = X[nonzero_var_cols]
    
    # 检查是否有足够的数据点进行标准化
    if X.shape[0] > 1:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # PCA降维
    n_components = min(2, X_scaled.shape[1], max(1, X_scaled.shape[0]-1))
    if n_components > 0:
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(X_scaled)
        
        # 记录解释方差
        explained_variance = pca.explained_variance_ratio_
        
        df['x'] = coords[:, 0]
        if n_components > 1:
            df['y'] = coords[:, 1]
        else:
            df['y'] = np.zeros(len(coords))
    else:
        print("Not enough data points for PCA")
        return
    
    # 可视化
    plt.figure(figsize=(14, 10))
    color_map = {
        'baseline': '#FF6B6B',
        'rebalance_first': "#190AEB",
        'features_first': "#28EA38"
    }
    
    for strategy, color in color_map.items():
        mask = df['strategy'] == strategy
        if mask.any():
            plt.scatter(
                df.loc[mask, 'x'], 
                df.loc[mask, 'y'], 
                c=color, 
                label=strategy, 
                alpha=0.8, 
                edgecolors='w',
                s=100
            )
    
    # 添加点标签（数据集名称）
    for idx, row in df.iterrows():
        plt.annotate(
            row['dataset'], 
            (row['x'], row['y']), 
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=9,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2)
        )
    
    # 设置图表信息
    xlabel = f"PC1 ({explained_variance[0]*100:.1f}%)"
    if n_components > 1:
        ylabel = f"PC2 ({explained_variance[1]*100:.1f}%)"
    else:
        ylabel = "No second component"
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.title('Dataset Characteristics vs. Optimal Resampling Strategy', fontsize=16)
    plt.legend(title='Optimal Strategy', title_fontsize='13')
    plt.grid(alpha=0.3)
    
    # 保存可视化图像
    visualization_path = os.path.join(output_dir, 'dataset_strategy_visualization.png')
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ Saved visualization to: {visualization_path}")
    
    # 显示图表
    plt.tight_layout()
    plt.show()
    
    # 额外分析 - 按策略分组统计
    if 'accuracy_improvement' in df.columns:
        strategy_summary = df.groupby('strategy')['accuracy_improvement'].agg(['mean', 'count'])
        strategy_summary['mean'] = strategy_summary['mean'].apply(lambda x: f"{x:.4f}")
        strategy_summary_path = os.path.join(output_dir, 'strategy_summary.csv')
        strategy_summary.to_csv(strategy_summary_path)
        print(f"\n🧾 Strategy summary saved to: {strategy_summary_path}")
    
    # 返回数据框以便进一步分析
    return df

if __name__ == "__main__":
    results_df = analyze_datasets(
        directory='dataset_temp',
        output_dir='analysis_results'
    )
    
    if results_df is not None:
        print("\nAnalysis completed successfully!")
        
        # 打印简要结果
        if 'strategy' in results_df.columns:
            print("\nStrategy Distribution:")
            print(results_df['strategy'].value_counts())
            
        if 'accuracy_improvement' in results_df.columns:
            avg_improvement = results_df['accuracy_improvement'].mean()
            print(f"\nAverage Accuracy Improvement: {avg_improvement:.4f}")
    else:
        print("\nNo valid results were generated")