import pandas as pd
import os
import shutil
import time
import numpy as np
from collections import defaultdict

def normalize_df(df):
    """规范化DataFrame：列排序，行排序"""
    # 按列名排序列
    sorted_cols = sorted(df.columns)
    df = df[sorted_cols]
    
    # 按所有列值排序行
    # 处理混合类型：将所有列转为字符串以确保安全比较
    for col in df.columns:
        if not all(isinstance(x, (int, float)) for x in df[col].dropna()):
            df[col] = df[col].astype(str)
    
    df = df.sort_values(by=sorted_cols, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    return df

def is_subset(df_small, df_large):
    """检查df_small是否是df_large的子集（处理混合数据类型）"""
    if len(df_small) > len(df_large):
        return False
        
    # 处理混合数据类型：确保列类型一致
    combined = pd.concat([df_small, df_large], keys=['small', 'large'])
    for col in combined.columns:
        if combined[col].dtype == object:
            # 对象列使用字符串比较
            combined[col] = combined[col].astype(str)
        elif pd.api.types.is_numeric_dtype(combined[col]):
            # 数值列转换为浮点数
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
    
    df_small = combined.xs('small')
    df_large = combined.xs('large')
    
    # 如果行数相同，直接比较整个DataFrame
    if len(df_small) == len(df_large):
        try:
            return df_small.equals(df_large)
        except:
            # 回退到合并方法
            return pd.merge(
                df_small, df_large, 
                how='left', indicator=True
            )['_merge'].eq('both').all()
    
    # 检查所有行是否在大数据集中存在
    merged = pd.merge(
        df_small, df_large, 
        how='left', indicator=True
    )
    return merged['_merge'].eq('both').all()

def process_datasets(input_dir, output_dir):
    """主处理函数：识别唯一数据集并保存到输出目录"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(output_dir, "duplicate_removal_log.txt")
    with open(log_file, 'w') as log:
        log.write(f"Dataset Deduplication Log - {time.ctime()}\n")
        log.write("="*80 + "\n")
        log.write(f"Input Directory: {input_dir}\n")
        log.write(f"Output Directory: {output_dir}\n\n")
        
        # 阶段1：获取所有CSV文件
        all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        log.write(f"[Stage 1] Found {len(all_files)} CSV files in input directory\n\n")
        
        if not all_files:
            log.write("No CSV files found. Exiting.\n")
            return
        
        # 阶段2：按列结构分组
        groups = defaultdict(list)
        read_errors = []
        df_dict = {}
        loaded_datasets_count = 0
        skipped_files = []
        
        for file in all_files:
            file_path = os.path.join(input_dir, file)
            try:
                # 读取CSV文件，处理混合数据类型
                df = pd.read_csv(file_path, dtype=str, on_bad_lines='skip')
                
                # 跳过空文件
                if df.empty:
                    skipped_files.append(file)
                    continue
                
                # 规范化DataFrame
                normalized = normalize_df(df)
                loaded_datasets_count += 1
                
                # 创建列签名（排序的列名元组）
                col_signature = tuple(sorted(df.columns))
                
                # 存储DataFrame和元数据
                groups[col_signature].append(file)
                df_dict[file] = {
                    'original': df,
                    'normalized': normalized,
                    'row_count': len(df),
                    'col_count': len(df.columns)
                }
                
            except Exception as e:
                read_errors.append(f"Error reading {file}: {str(e)}")
        
        # 记录分组统计
        log.write(f"[Stage 2] Grouped datasets by column structure:\n")
        log.write(f"- Total datasets loaded: {loaded_datasets_count}\n")
        log.write(f"- Total groups created: {len(groups)}\n")
        log.write(f"- Files skipped: {len(skipped_files)} (empty files)\n\n")
        
        # 记录跳过的文件
        if skipped_files:
            log.write("Skipped empty files:\n")
            for file in skipped_files:
                log.write(f"- {file}\n")
            log.write("\n")
        
        # 记录读取错误
        if read_errors:
            log.write("\n" + "!"*80 + "\n")
            log.write("FILE READ ERRORS:\n")
            log.write("\n".join(read_errors) + "\n")
            log.write("!"*80 + "\n\n")
        
        # 阶段3：处理每个列结构组
        kept_files = []
        removed_files = []
        duplicate_info = []
        subset_info = []
        group_processing_stats = []

        for signature, files in groups.items():
            group_name = ", ".join(signature)
            log.write(f"\nProcessing column group: {group_name} (size: {len(files)} datasets)\n")
            
            # 按行数降序排序
            sorted_files = sorted(
                files, 
                key=lambda f: (df_dict[f]['row_count'], f),
                reverse=True
            )
            
            # 确定唯一数据集
            unique_files = []
            group_removed = []
            
            for candidate in sorted_files:
                candidate_data = df_dict[candidate]
                is_duplicate = False
                reason = ""
                
                # 检查是否被保留数据集包含
                for kept in unique_files:
                    kept_data = df_dict[kept]
                    
                    try:
                        # 检查是否完全相等（行数相同更快比较）
                        if candidate_data['row_count'] == kept_data['row_count']:
                            if candidate_data['normalized'].equals(kept_data['normalized']):
                                is_duplicate = True
                                reason = f"Duplicate of {kept}"
                                break
                        
                        # 检查子集关系
                        if is_subset(candidate_data['normalized'], kept_data['normalized']):
                            is_duplicate = True
                            reason = f"Subset of {kept} ({candidate_data['row_count']} vs {kept_data['row_count']} rows)"
                            break
                    
                    except Exception as e:
                        # 处理比较过程中的错误
                        error_msg = f"Error comparing {candidate} with {kept}: {str(e)}"
                        log.write(error_msg + "\n")
                        is_duplicate = True
                        reason = f"Comparison error: {str(e)}"
                
                # 如果唯一则保留
                if not is_duplicate:
                    unique_files.append(candidate)
                else:
                    group_removed.append(candidate)
                    removed_files.append(candidate)
                    if "Duplicate" in reason:
                        duplicate_info.append(f"{candidate} --> {reason}")
                    else:
                        subset_info.append(f"{candidate} --> {reason}")
            
            # 记录本组处理结果
            kept_files.extend(unique_files)
            log.write(f"- Kept {len(unique_files)} datasets\n")
            log.write(f"- Removed {len(group_removed)} duplicates/subsets\n")
            group_processing_stats.append(f"{group_name}: kept {len(unique_files)}/removed {len(group_removed)}")
        
        # 阶段4：保存结果
        # 复制保留的文件到输出目录
        for file in kept_files:
            src = os.path.join(input_dir, file)
            dst = os.path.join(output_dir, file)
            shutil.copy2(src, dst)
        
        # 记录结果统计
        log.write("\n" + "="*80 + "\n")
        log.write("FINAL PROCESSING RESULTS\n")
        log.write("="*80 + "\n\n")
        
        log.write("PROCESSING STAGES SUMMARY:\n")
        log.write(f"1. Input datasets: {len(all_files)}\n")
        log.write(f"2. Skipped datasets: {len(skipped_files)} (empty files)\n")
        log.write(f"3. Datasets processed: {loaded_datasets_count}\n")
        log.write(f"4. Final kept datasets: {len(kept_files)}\n\n")
        
        log.write("GROUP PROCESSING DETAILS:\n")
        for stat in group_processing_stats:
            log.write(f"- {stat}\n")
        log.write("\n")
        
        log.write(f"Total datasets removed: {len(removed_files)}\n")
        log.write(f"- Duplicates: {len(duplicate_info)}\n")
        log.write(f"- Subsets: {len(subset_info)}\n\n")
        
        # 记录详细情况
        if duplicate_info:
            log.write("\nDUPLICATE DATASETS:\n")
            log.write("-------------------\n")
            for line in duplicate_info:
                log.write(line + "\n")
        
        if subset_info:
            log.write("\nSUBSET DATASETS:\n")
            log.write("----------------\n")
            for line in subset_info:
                log.write(line + "\n")
        
        # 列出保留的文件
        log.write("\nKEPT DATASETS:\n")
        log.write("--------------\n")
        kept_files.sort()
        for file in kept_files:
            meta = df_dict[file]
            log.write(f"{file} ({meta['row_count']} rows, {meta['col_count']} columns)\n")
        
        log.write("\n" + "="*80 + "\n")
        log.write("PROCESS COMPLETED SUCCESSFULLY\n")
        log.write("="*80 + "\n")

# 主执行
if __name__ == "__main__":
    # 配置路径
    input_dir = "datasets/dataset_csv_std"
    output_dir = "datasets/dataset_csv_std_duplicate_removal"
    
    print("Starting dataset deduplication process...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    # 处理数据集
    process_datasets(input_dir, output_dir)
    
    duration = time.time() - start_time
    print(f"Processing complete! Time taken: {duration:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Detailed log available at: {os.path.join(output_dir, 'duplicate_removal_log.txt')}")