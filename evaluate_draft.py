import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import json
from Dppga_for_call import PipelineOptimizer
from scipy.stats import chi2_contingency

CONFIG = {
    "data_dir": "dataset_csv",
    "result_dir": "cluster_results",
    "min_cluster_size": 5,
    "tsne_perplexity": 30,
    "feature_export": "features.csv",
    "cluster_export": "clusters.csv",
    "visualization": "cluster_visual.png"
}

class DataSetCluster:
    def __init__(self):
        self.features = []
        self.file_paths = []
        os.makedirs(CONFIG['result_dir'], exist_ok=True)

    def _load_datasets(self):
        """加载所有CSV文件路径"""
        for root, _, files in os.walk(CONFIG['data_dir']):
            for file in files:
                if file.lower().endswith('.csv'):
                    self.file_paths.append(os.path.join(root, file))

    @staticmethod
    def _global_features(df):
        """全局统计特征"""
        return {
            'num_rows': len(df),
            'num_cols': df.shape[1],
            'total_missing': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'density': df.count().sum() / (df.size + 1e-9)
        }

    @staticmethod
    def _dtype_features(df):
        """数据类型分布"""
        dtypes = df.dtypes.astype(str).value_counts().to_dict()
        return {
            'numeric_ratio': dtypes.get('float64', 0) / len(df.columns),
            'category_ratio': dtypes.get('category', 0) / len(df.columns),
            'object_ratio': dtypes.get('object', 0) / len(df.columns)
        }

    @staticmethod
    def _column_aggregations(df):
        """列聚合统计"""
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            return {}
            
        return {
            'mean_of_means': numeric_df.mean().mean(),
            'std_of_stds': numeric_df.std().std(),
            'skewness_avg': numeric_df.apply(stats.skew).mean(),
            'kurtosis_avg': numeric_df.apply(stats.kurtosis).mean(),
            'outlier_ratio': ((np.abs(numeric_df - numeric_df.mean()) > 3*numeric_df.std()).mean().mean())
        }

    @staticmethod
    def _entropy_features(df, num_bins=10):
        """信息熵特征计算"""
        entropies = []
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            # 跳过全空或单一值列
            if len(col_data) < 2 or col_data.nunique() == 1:
                continue
                
            # 数值型数据分箱计算
            if pd.api.types.is_numeric_dtype(col_data):
                try:
                    # 自动确定最佳分箱数（Sturges准则）
                    n = len(col_data)
                    bins = min(num_bins, int(np.log2(n) + 1)) if n > 0 else 1
                    p_data = col_data.value_counts(bins=bins, normalize=True)
                except Exception as e:
                    print(f"数值列分箱失败 [{col}]: {str(e)}")
                    continue
            # 非数值型直接计数
            else:
                p_data = col_data.value_counts(normalize=True)
                
            # 计算熵值
            entropy = stats.entropy(p_data)
            entropies.append(entropy)
        
        return {
            'avg_entropy': np.mean(entropies) if entropies else 0,
            'max_entropy': np.max(entropies) if entropies else 0,
            'entropy_variance': np.var(entropies) if entropies else 0
        }

    @staticmethod
    def _structural_features(df):
        """结构模式特征"""
        time_cols = [col for col in df.columns 
                    if ('date' in col.lower()) or 
                       (df[col].dtype == 'datetime64[ns]')]
        high_card = [col for col in df.columns 
                    if (df[col].nunique() / len(df)) > 0.9]
        return {
            'has_time_series': int(len(time_cols) > 0),
            'high_cardinality_ratio': len(high_card)/len(df.columns)
        }

    @staticmethod
    def _model_features(df, n_samples=1000):
        """模型代理特征"""
        sample_df = df.sample(min(n_samples, len(df))).select_dtypes(include=np.number)
        if sample_df.shape[1] == 0:
            return {}
            
        clf = IsolationForest().fit(sample_df.fillna(0))
        return {
            'anomaly_score': clf.decision_function(sample_df.fillna(0)).mean()
        }

    def _extract_features(self, file_path):
        """综合特征提取"""
        try:
            df = pd.read_csv(file_path, encoding_errors='ignore', nrows=10000)
            features = {}
            
            # 分层特征提取
            features.update(self._global_features(df))
            features.update(self._dtype_features(df))
            features.update(self._column_aggregations(df))
            features.update(self._entropy_features(df))
            features.update(self._structural_features(df))
            features.update(self._model_features(df))
            
            return features
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def _preprocess_features(self):
        """特征预处理"""
        feature_df = pd.DataFrame(self.features)
        
        # 处理缺失值
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                feature_df[col] = feature_df[col].fillna('missing')
            else:
                feature_df[col] = feature_df[col].fillna(feature_df[col].mean())
                
        # 标准化处理
        numeric_cols = feature_df.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        feature_df[numeric_cols] = scaler.fit_transform(feature_df[numeric_cols])
        
        return feature_df

    def _cluster_analysis(self, feature_df):
        """聚类分析"""
        # 降维可视化
        tsne = TSNE(n_components=2, perplexity=CONFIG['tsne_perplexity'])
        embeddings = tsne.fit_transform(feature_df)
        
        # HDBSCAN聚类
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=CONFIG['min_cluster_size'],
            gen_min_span_tree=True
        )
        clusters = clusterer.fit_predict(feature_df)
        
        return embeddings, clusters

    def _visualize(self, embeddings, clusters):
        """可视化结果"""
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=embeddings[:, 0], y=embeddings[:, 1],
            hue=clusters, palette="tab20", legend="full",
            s=100, alpha=0.8, edgecolor='none'
        )
        plt.title("Dataset Cluster Visualization")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.savefig(os.path.join(CONFIG['result_dir'], CONFIG['visualization']))
        plt.close()

    def _save_results(self, feature_df, clusters):
        """保存结果"""
        # 添加集群标签到特征矩阵
        feature_df = feature_df.copy()
        feature_df['cluster'] = clusters
        
        # 创建结果文件
        result_df = pd.DataFrame({
            'file_path': self.file_paths,
            'cluster': clusters
        })
        
        # 确保目录存在
        os.makedirs(CONFIG['result_dir'], exist_ok=True)
        
        # 保存文件
        result_df.to_csv(
            os.path.join(CONFIG['result_dir'], CONFIG['cluster_export']), 
            index=False
        )
        feature_df.to_csv(
            os.path.join(CONFIG['result_dir'], CONFIG['feature_export']),
            index=False
        )
        
        # 生成集群报告
        if feature_df['cluster'].nunique() > 1:
            cluster_report = feature_df.groupby('cluster').mean().T.to_dict()
        else:
            cluster_report = {"single_cluster": feature_df.mean().to_dict()}
        
        with open(os.path.join(CONFIG['result_dir'], 'cluster_report.json'), 'w') as f:
            json.dump(cluster_report, f, indent=2)

    def run(self):
        """主流程"""
        print("Loading datasets...")
        self._load_datasets()
        
        print(f"Processing {len(self.file_paths)} datasets...")
        for path in tqdm(self.file_paths):
            features = self._extract_features(path)
            if features:
                self.features.append(features)
        
        print("Preprocessing features...")
        feature_df = self._preprocess_features()
        
        print("Running cluster analysis...")
        embeddings, clusters = self._cluster_analysis(feature_df)
        
        print("Generating visualization...")
        self._visualize(embeddings, clusters)
        
        print("Saving results...")
        self._save_results(feature_df, clusters)
        
        print(f"Results saved to {CONFIG['result_dir']} directory")

class PipelineEvaluator:
    def __init__(self, data_dir="dataset_csv"):
        self.data_dir = data_dir
        self.results = []
        self.cluster_map = None
    
    def _cluster_datasets(self):
        """执行数据集聚类"""
        cluster = DataSetCluster()
        cluster.run()
        
        # 读取聚类结果
        cluster_df = pd.read_csv(os.path.join(CONFIG['result_dir'], 'clusters.csv'))
        self.cluster_map = dict(zip(cluster_df['file_path'], cluster_df['cluster']))
    
    def _detect_target(self, df):
        """自动检测目标列（示例逻辑）"""
        # 可根据实际需求优化此逻辑
        for col in df.columns:
            if df[col].nunique() == 2 and df[col].dtype == 'int64':
                return col
        return df.columns[-1]  # 默认最后一列为目标
    
    def _evaluate_dataset(self, file_path):
        """评估单个数据集"""
        try:
            df = pd.read_csv(file_path)
            target = self._detect_target(df)
            
            # 创建优化器
            optimizer = PipelineOptimizer(
                file_path=file_path,
                target_column=target,
                steps_order_candidates=[
                    ['rebalance', 'features'],
                    ['features', 'rebalance']
                ],
                n_trials=50
            )
            optimizer.optimize()
            res = optimizer.get_results()
            
            # 记录结果
            return {
                'dataset': os.path.basename(file_path),
                'cluster': self.cluster_map.get(file_path, -1),
                'baseline': res['baseline_accuracy'],
                'best_order': res['best_order'],
                'optimized_acc': res['best_optimized_accuracy']
            }
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def run_evaluation(self):
        """主运行流程"""
        # Step 1: 数据集聚类
        self._cluster_datasets()
        
        # Step 2: 遍历数据集评估
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith('.csv'):
                    file_path = os.path.join(root, file)
                    result = self._evaluate_dataset(file_path)
                    if result:
                        self.results.append(result)
        
        # Step 3: 保存结果
        self.results_df = pd.DataFrame(self.results)
        self.results_df.to_csv('evaluation_results.csv', index=False)
    
    def analyze_results(self):
        """分析与聚类的关联性"""
        # 统计每个簇的最佳顺序分布
        contingency_table = pd.crosstab(
            self.results_df['cluster'],
            self.results_df['best_order']
        )
        
        # 卡方检验
        chi2, p, dof, _ = chi2_contingency(contingency_table)
        
        # 可视化
        plt.figure(figsize=(12, 6))
        
        # 热力图
        plt.subplot(121)
        sns.heatmap(
            contingency_table.div(contingency_table.sum(1), axis=0),
            annot=True, cmap='YlGnBu'
        )
        plt.title("Best Order Distribution per Cluster")
        
        # 提升幅度对比
        plt.subplot(122)
        self.results_df['improvement'] = self.results_df['optimized_acc'] - self.results_df['baseline']
        sns.boxplot(
            x='cluster', y='improvement', 
            hue='best_order',
            data=self.results_df
        )
        plt.title("Accuracy Improvement by Cluster")
        
        plt.tight_layout()
        plt.savefig('cluster_analysis.png')
        
        return {
            'contingency_table': contingency_table,
            'chi2_test': {'chi2': chi2, 'p_value': p, 'dof': dof}
        }

if __name__ == "__main__":
    evaluator = PipelineEvaluator()
    evaluator.run_evaluation()
    analysis_results = evaluator.analyze_results()
    
    print("\n=== 关键统计结果 ===")
    print("1. 最佳操作分布:")
    print(analysis_results['contingency_table'])
    
    print("\n2. 卡方检验结果:")
    print(f"P值: {analysis_results['chi2_test']['p_value']:.4f}")
    if analysis_results['chi2_test']['p_value'] < 0.05:
        print("-> 聚类结果与最佳操作顺序显著相关")
    else:
        print("-> 未发现显著相关性")