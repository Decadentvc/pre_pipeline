import os
import json
import glob

# 配置路径
notebooks_dir = "./data/notebook"
datasets_base = "./data/dataset"
datasetinfo_path = "./haipipe/core/datasetinfo.json"
output_file = "notebook_records.json"

def main():
    # 加载数据集信息
    with open(datasetinfo_path) as f:
        datasetinfo = json.load(f)
    
    records = []
    print('start')
    # 遍历notebook目录
    for nb_file in os.listdir(notebooks_dir):
        if not nb_file.endswith(".ipynb"):
            continue
        
        # 提取纯净的文件名（不带扩展名）
        nb_name = os.path.splitext(nb_file)[0]
        
        # 查找对应的数据集信息
        if nb_name not in datasetinfo:
            print(f"Warning: No dataset info found for {nb_file}")
            continue
        
        info = datasetinfo[nb_name]
        dataset_dir = os.path.join(datasets_base, info["dataset"])
        
        # 验证数据集目录是否存在
        if not os.path.exists(dataset_dir):
            print(f"Error: Dataset directory not found for {nb_file}")
            continue
            
        # 查找数据集目录中的CSV文件
        csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
        
        # 处理可能的异常情况
        if not csv_files:
            print(f"Error: No CSV files found in {dataset_dir}")
            continue
        elif len(csv_files) > 1:
            print(f"Warning: Multiple CSV files found in {dataset_dir}. Using first one.")
        
        # 构建记录
        record = {
            "notebook_path": os.path.join(notebooks_dir, nb_file),
            "dataset_path": csv_files[0],  # 使用第一个找到的CSV文件
            "index": info["index"][0] if isinstance(info["index"], list) else info["index"]
        }
        records.append(record)
    
    # 保存结果
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)
    
    print(f"Successfully processed {len(records)} notebooks")

if __name__ == "__main__":
    main()