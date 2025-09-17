import os
import shutil
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings

# 忽略特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============== 特征提取器类 ==============
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
            'numeric_ratio': dtypes.get('float64', 0) / (total_cols or 1),
            'category_ratio': dtypes.get('category', 0) / (total_cols or 1),
            'object_ratio': dtypes.get('object', 0) / (total_cols or 1),
            'int_ratio': dtypes.get('int64', 0) / (total_cols or 1)
        }

    @staticmethod
    def _column_aggregations(df):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}
            
        mean_of_means = numeric_df.mean().mean() if not numeric_df.empty else 0
        std_of_stds = numeric_df.std().std() if not numeric_df.empty else 0
        
        # 计算偏度时跳过常数列
        skew_values = numeric_df.apply(lambda x: stats.skew(x.dropna()) if x.nunique() > 1 and len(x.dropna()) > 1 else 0)
        kurtosis_values = numeric_df.apply(lambda x: stats.kurtosis(x.dropna()) if x.nunique() > 1 and len(x.dropna()) > 1 else 0)
        
        # 离群值检测
        outlier_mask = numeric_df.apply(
            lambda x: (np.abs(x - x.mean()) > 3*x.std()) 
            if x.std() > 0 and not x.empty else pd.Series(False, index=x.index)
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
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if not pd.isna(unique_ratio) and unique_ratio > 0.9:
                high_card.append(col)
                
        return {
            'has_time_series': int(len(time_cols) > 0),
            'high_cardinality_ratio': len(high_card)/max(len(df.columns), 1)
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
            # 尝试多种编码方式
            encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    # 仅读取前10,000行以加速处理
                    df = pd.read_csv(file_path, nrows=10000, encoding=encoding)
                    if not df.empty:
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    continue
                    
            if df is None or df.empty:
                print(f"无法读取文件或文件为空: {file_path}")
                return None
                
            # 处理列名：处理缺失或数字列名情况
            df.columns = [
                f"col_{i}" if isinstance(name, int) or str(name).strip() == '' 
                else str(name).strip() 
                for i, name in enumerate(df.columns)
            ]
            
            # 提取各类特征
            features = {}
            features.update(FeatureExtractor._global_features(df))
            features.update(FeatureExtractor._dtype_features(df))
            features.update(FeatureExtractor._column_aggregations(df))
            features.update(FeatureExtractor._entropy_features(df))
            features.update(FeatureExtractor._structural_features(df))
            features.update(FeatureExtractor._model_features(df))
            
            features['file_path'] = file_path  # 保存文件路径
            return features
        except Exception as e:
            print(f"处理文件时出错 {file_path}: {str(e)}")
            return None

# ============== 主程序 ==============
def main():
    # 1. 设置路径
    input_dir = 'datasets/dataset_csv_std_duplicate_removal'
    output_dir = 'datasets/dataset_std_dr_demo50'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 2. 收集所有CSV文件
    all_files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith('.csv')
    ]
    
    if not all_files:
        print(f"在 {input_dir} 中未找到CSV文件")
        return
    
    print(f"找到 {len(all_files)} 个CSV文件，开始特征提取...")
    
    # 3. 提取所有文件的特征
    features_list = []
    skipped_files = 0
    
    for i, file_path in enumerate(all_files):
        print(f"处理文件中 ({i+1}/{len(all_files)}): {os.path.basename(file_path)}", end='\r')
        features = FeatureExtractor.extract_features(file_path)
        if features:
            features_list.append(features)
        else:
            skipped_files += 1
    
    print(f"\n特征提取完成! 成功处理: {len(features_list)} 文件, 跳过: {skipped_files} 文件")
    
    # 检查是否有足够的文件
    if len(features_list) < 50:
        print(f"只有 {len(features_list)} 个文件可供选择，少于50个")
        return
    
    # 4. 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    file_paths = features_df['file_path']  # 保存文件路径
    feature_columns = features_df.drop(columns='file_path').columns.tolist()
    features_df = features_df.drop(columns='file_path')
    
    # 5. 处理缺失值
    print("处理缺失值...")
    features_df.fillna({
        'mean_of_means': 0,
        'std_of_stds': 0,
        'skewness_avg': 0,
        'kurtosis_avg': 0,
        'outlier_ratio': 0,
        'anomaly_score': 0,
        'avg_entropy': 0,
        'max_entropy': 0,
        'entropy_variance': 0,
        'computed_entropy_cols': 0
    }, inplace=True)
    
    # 6. 特征标准化
    print("标准化特征...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    # 7. PCA降维（保留95%的方差）
    print("应用PCA降维...")
    pca = PCA(n_components=0.95, random_state=42)
    pca_features = pca.fit_transform(scaled_features)
    print(f"原始维度: {scaled_features.shape[1]}，降维后维度: {pca_features.shape[1]}")
    
    # 8. KMeans聚类
    print("聚类选择代表性样本...")
    n_clusters = 50
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_features)
    
    # 9. 选样策略：每个簇选取一个代表
    selected_indices = []
    
    # 选择每个簇中离中心最远的点（特征最独特的样本）
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_points = pca_features[cluster_mask]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        
        # 计算距离并选择最远的点
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        farthest_idx = np.argmax(distances)
        
        # 获取原始索引
        cluster_indices = np.where(cluster_mask)[0]
        selected_index = cluster_indices[farthest_idx]
        selected_indices.append(selected_index)
    
    # 10. 添加极端特征样本以确保多样性
    extreme_features = ['total_missing', 'duplicate_rows', 'outlier_ratio', 'high_cardinality_ratio']
    
    # 选择每种特征的最大值
    for feat in extreme_features:
        if feat in features_df.columns:
            selected_indices.append(features_df[feat].idxmax())
    
    # 选择每种特征的最小值
    for feat in extreme_features:
        if feat in features_df.columns:
            selected_indices.append(features_df[feat].idxmin())
    
    # 添加一个非常小的数据集
    small_idx = features_df['num_rows'].idxmin()
    selected_indices.append(small_idx)
    
    # 添加一个非常大的数据集
    large_idx = features_df['num_rows'].idxmax()
    selected_indices.append(large_idx)
    
    # 去重并确保不超过50个
    selected_indices = list(set(selected_indices))
    if len(selected_indices) > 50:
        selected_indices = selected_indices[:50]
    
    # 11. 复制选定的文件
    selected_files = [file_paths.iloc[idx] for idx in selected_indices]
    
    print("\n选定的数据集:")
    for i, file_path in enumerate(selected_files):
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_dir, filename)
        shutil.copy2(file_path, dest_path)
        print(f"{i+1}. {filename}")
    
    print(f"\n已成功复制 {len(selected_files)} 个数据集到 {output_dir}")

if __name__ == "__main__":
    main()