import os
import pandas as pd
import numpy as np
import re
import logging
from collections import defaultdict

# 配置日志记录
logging.basicConfig(
    filename='rename_label.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def detect_label_column(df):
    """增强版标签列检测函数"""
    # 预处理：统一列名格式
    df.columns = df.columns.str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
    
    # 特征评分系统
    scores = defaultdict(float)
    label_keywords = {
        'target', 'label', 'class', 'response', 
        'outcome', 'y', 'result', 'diagnosis'
    }
    
    for col in df.columns:
        col_clean = re.sub(r'_+', '_', col).strip('_')
        
        # 规则1：列名关键词匹配
        if any(kw in col_clean for kw in label_keywords):
            scores[col] += 3.0
            
        # 规则2：唯一值比例（分类问题）
        unique_ratio = df[col].nunique() / len(df)
        if 0.005 < unique_ratio < 0.5:
            scores[col] += 2.0
        elif df[col].dtype == 'object' and unique_ratio == 1.0:
            scores[col] -= 1.0  # 排除全唯一字符列
            
        # 规则3：低缺失率
        missing_rate = df[col].isnull().mean()
        if missing_rate < 0.05:
            scores[col] += 1.5
        else:
            scores[col] -= 1.0
            
        # 规则4：位置得分（首/末列）
        if col == df.columns[0] or col == df.columns[-1]:
            scores[col] += 1.0
            
        # 规则5：数据类型异常（唯一非数值列）
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if col not in numeric_cols and len(numeric_cols) == df.shape[1]-1:
            scores[col] += 2.0
            
    # 处理无评分项
    if not scores:
        # 最终试探：与所有其他列相关性最低的数值列
        try:
            corr_matrix = df.corr().abs()
            avg_corr = corr_matrix.mean(axis=1)
            return avg_corr.idxmin()
        except:
            return df.columns[-1]  # 默认返回最后一列
    
    # 获取最高分候选
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    top_score = sorted_scores[0][1]
    
    # 处理平局情况
    candidates = [k for k,v in scores.items() if v == top_score]
    if len(candidates) > 1:
        # 优先选择出现更早的列
        return df.columns[df.columns.isin(candidates)][0]
    
    return sorted_scores[0][0]

def process_dataset(file_path):
    """处理单个数据集文件"""
    try:
        # 尝试多种编码读取
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
            
        original_cols = set(df.columns)
        
        # 检测标签列
        label_col = detect_label_column(df)
        if label_col not in df.columns:
            logging.error(f"标签列检测失败: {file_path}")
            return False
            
        # 如果已经是label则跳过
        if label_col == 'label':
            logging.info(f"无需修改: {file_path}")
            return True
            
        # 处理列名冲突
        if 'label' in df.columns:
            # 保留原始标签列数据，重命名旧label列
            df['original_label_backup'] = df['label']
            logging.warning(f"检测到冲突列，已备份原始label列: {file_path}")
            
        # 执行重命名
        df.rename(columns={label_col: 'label'}, inplace=True)
        
        # 移除可能的重复列（当标签列被误检测时）
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 保存文件（先写临时文件再替换）
        temp_path = file_path + '.tmp'
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, file_path)
        
        logging.info(f"成功更名 {label_col} -> label : {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"处理失败 {file_path}: {str(e)}", exc_info=True)
        return False

def main():
    dataset_dir = 'dataset_temp'
    processed = 0
    failed = 0
    
    for filename in os.listdir(dataset_dir):
        if not filename.endswith('.csv'):
            continue
            
        file_path = os.path.join(dataset_dir, filename)
        if process_dataset(file_path):
            processed += 1
        else:
            failed += 1
            
    print(f"处理完成！成功: {processed}, 失败: {failed}")
    print("详见日志文件: rename_label.log")

if __name__ == "__main__":
    main()