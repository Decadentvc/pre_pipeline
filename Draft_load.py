if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from sklearn.model_selection import train_test_split
    import pandas as pd
def load_local_data(file_path='./local_data.csv', 
                        target_column='target',
                        feature_columns=None):
            """
            从本地CSV文件加载数据集
            参数：
                file_path: 数据文件路径（默认当前目录的local_data.csv）
                target_column: 目标变量列名（默认'target'）
                feature_columns: 特征列列表（None表示自动选择除目标列外的所有列）
            返回：
                X, y 格式与sklearn数据集一致
            """
            try:
                df = pd.read_csv(file_path)
                
                # 自动检测特征列（如果未指定）
                if feature_columns is None:
                    feature_columns = [col for col in df.columns if col != target_column]
                    
                # 处理可能的缺失值（简单用中位数填充）
                # df[feature_columns] = df[feature_columns].fillna(df[feature_columns].median())
                
                return df[feature_columns].values, df[target_column].values
            except Exception as e:
                print(f"加载本地数据失败: {str(e)}")
                return None, None
        
dataset_name, dataset = ("Local Dataset", load_local_data(
            file_path='dataset_temp/abcsds_pokemon_Pokemon.csv',  # 实际路径
            target_column='label'))


X, y = dataset
baseline = RandomForestClassifier(n_estimators=50, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
baseline.fit(X_train, y_train)
baseline_score = accuracy_score(y_test, baseline.predict(X_test))
print(f"Baseline Accuracy: {baseline_score:.4f}")