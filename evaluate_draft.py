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
        dtypes = df.dtypes.astype(str).value_counts().to_dict()
        return {
            'numeric_ratio': dtypes.get('float64', 0) / len(df.columns),
            'category_ratio': dtypes.get('category', 0) / len(df.columns),
            'object_ratio': dtypes.get('object', 0) / len(df.columns)
        }

    @staticmethod
    def _column_aggregations(df):
        numeric_df = df.select_dtypes(include=np.number)
        return {} if numeric_df.empty else {
            'mean_of_means': numeric_df.mean().mean(),
            'std_of_stds': numeric_df.std().std(),
            'skewness_avg': numeric_df.apply(stats.skew).mean(),
            'kurtosis_avg': numeric_df.apply(stats.kurtosis).mean(),
            'outlier_ratio': ((np.abs(numeric_df - numeric_df.mean()) > 3*numeric_df.std()).mean().mean())
        }

    @staticmethod
    def _entropy_features(df, num_bins=10):
        entropies = []
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) < 2 or col_data.nunique() == 1:
                continue
            if pd.api.types.is_numeric_dtype(col_data):
                try:
                    n = len(col_data)
                    bins = min(num_bins, int(np.log2(n) + 1)) if n > 0 else 1
                    p_data = col_data.value_counts(bins=bins, normalize=True)
                except:
                    continue
            else:
                p_data = col_data.value_counts(normalize=True)
            entropies.append(stats.entropy(p_data))
        return {
            'avg_entropy': np.mean(entropies) if entropies else 0,
            'max_entropy': np.max(entropies) if entropies else 0,
            'entropy_variance': np.var(entropies) if entropies else 0
        }

    @staticmethod
    def _structural_features(df):
        time_cols = [col for col in df.columns if ('date' in col.lower()) or (df[col].dtype == 'datetime64[ns]')]
        high_card = [col for col in df.columns if (df[col].nunique() / len(df)) > 0.9]
        return {
            'has_time_series': int(len(time_cols) > 0),
            'high_cardinality_ratio': len(high_card)/len(df.columns)
        }

    @staticmethod
    def _model_features(df, n_samples=1000):
        sample_df = df.sample(min(n_samples, len(df))).select_dtypes(include=np.number)
        return {} if sample_df.empty else {
            'anomaly_score': IsolationForest().fit(sample_df.fillna(0)).decision_function(sample_df.fillna(0)).mean()
        }

    @staticmethod
    def extract_features(file_path):
        try:
            df = pd.read_csv(file_path, encoding_errors='ignore', nrows=10000)
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

def analyze_datasets(directory='dataset_temp'):
    # 初始化结果收集器
    results = []
    candidates = [['rebalance', 'features'], ['features', 'rebalance']]
    
    # 遍历数据集目录
    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue
            
        file_path = os.path.join(directory, filename)
        
        # 特征提取
        features = FeatureExtractor.extract_features(file_path)
        if not features:
            print(f"Skipped {filename} due to feature extraction failure")
            continue
            
        # 管道优化评估
        try:
            optimizer = PipelineOptimizer(
                file_path=file_path,
                target_column='label',
                steps_order_candidates=candidates,
                n_trials=30
            )
            optimizer.optimize()
            eval_result = optimizer.get_results()
        except Exception as e:
            print(f"Evaluation failed for {filename}: {str(e)}")
            continue
        
        # 确定获胜策略
        baseline = eval_result['Baseline Accuracy']
        optimized = eval_result['Best Optimized Accuracy']
        pipeline = eval_result['Effective Pipeline Prototype']
        
        if baseline > optimized:
            strategy = 'baseline'
        else:
            strategy = 'rebalance_first' if pipeline == candidates[0] else 'features_first'
        
        # 记录结果
        results.append({
            'dataset': filename,
            **features,
            'strategy': strategy
        })
    
    # 转换为DataFrame并处理数据
    df = pd.DataFrame(results).set_index('dataset')
    if df.empty:
        print("No valid datasets processed")
        return
    
    # 数据预处理
    features = df.columns[df.columns != 'strategy']
    X = df[features].fillna(0)
    X = StandardScaler().fit_transform(X)
    
    # PCA降维
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
    
    # 可视化
    plt.figure(figsize=(12, 8))
    color_map = {
        'baseline': '#FF6B6B',
        'rebalance_first': '#4ECDC4',
        'features_first': '#556270'
    }
    
    for strategy, color in color_map.items():
        mask = df['strategy'] == strategy
        plt.scatter(df[mask]['x'], df[mask]['y'], c=color, label=strategy, alpha=0.7, edgecolors='w')
    
    plt.title('2D Visualization of Dataset Characteristics with Winning Strategy')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Winning Strategy')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_datasets()