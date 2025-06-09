import os
import json
import pandas as pd
from collections import defaultdict

# 路径配置
DATASET_DIR = "dataset"
JSON_PATH = "Haipipe/haipipe/core/datasetinfo.json"
OUTPUT_DIR = "dataset_csv_std"

def process_datasets():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 读取数据集信息文件
    try:
        with open(JSON_PATH, 'r') as f:
            dataset_info = json.load(f)
    except Exception as e:
        print(f"无法读取JSON文件: {e}")
        return
    
    # 按子目录名分组数据集信息
    dir_to_info = defaultdict(list)
    for info_id, info in dataset_info.items():
        dir_name = info["dataset"]
        dir_to_info[dir_name].append(info)
    
    # 遍历数据集目录
    processed_count = 0
    errors = []
    
    for dir_name in os.listdir(DATASET_DIR):
        dir_path = os.path.join(DATASET_DIR, dir_name)
        if not os.path.isdir(dir_path):
            continue
        
        # 获取子目录对应的数据集信息
        info_list = dir_to_info.get(dir_name)
        if not info_list:
            errors.append(f"{dir_name}: 未在JSON中找到数据集信息")
            continue
            
        # 检查索引一致性
        unique_indexes = set()
        for info in info_list:
            unique_indexes.add(tuple(sorted(info["index"])))
            
        if len(unique_indexes) > 1:
            errors.append(f"{dir_name}: 存在冲突的标签索引 - {list(unique_indexes)}")
            continue
            
        # 使用第一个一致的信息
        info = info_list[0]
        label_idxs = info["index"]
        
        # 查找CSV文件
        csv_files = [f for f in os.listdir(dir_path) if f.lower().endswith(".csv")]
        if not csv_files:
            errors.append(f"{dir_name}: 未找到CSV文件")
            continue
        if len(csv_files) > 1:
            errors.append(f"{dir_name}: 找到多个CSV文件 ({', '.join(csv_files)})")
            continue
            
        csv_file = csv_files[0]
        csv_path = os.path.join(dir_path, csv_file)
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            
            # 备份原有label列
            if "label" in df.columns:
                df = df.rename(columns={"label": "label_ori"})
            
            # 处理标签索引
            if not label_idxs:
                errors.append(f"{dir_name}: 未指定标签索引")
                continue
                
            if len(label_idxs) > 1:
                errors.append(f"{dir_name}: 不支持多标签索引 ({label_idxs})")
                continue
                
            label_idx = label_idxs[0]
            if label_idx < 0 or label_idx >= len(df.columns):
                errors.append(f"{dir_name}/{csv_file}: 无效标签索引 {label_idx} (数据集有 {len(df.columns)} 列)")
                continue
                
            # 获取标签列名并重命名
            label_col = df.columns[label_idx]
            if label_col != "label":  # 只有当标签列不是"label"时才重命名
                df = df.rename(columns={label_col: "label"})
            
            # 保存处理后的CSV
            new_name = f"{dir_name}__{csv_file.replace('.csv', '')}.csv"
            output_path = os.path.join(OUTPUT_DIR, new_name)
            df.to_csv(output_path, index=False)
            
            processed_count += 1
            print(f"处理成功: {dir_name} -> {new_name} (标签列: '{label_col}' -> 'label')")
        
        except Exception as e:
            errors.append(f"{dir_name}/{csv_file}: 处理错误 - {str(e)}")
    
    # 输出汇总报告
    print("\n===== 处理完成 =====")
    print(f"成功处理: {processed_count} 个数据集")
    if errors:
        print(f"\n错误总数: {len(errors)}")
        for error in errors:
            print(f"  {error}")

if __name__ == "__main__":
    process_datasets()