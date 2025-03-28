#!/bin/bash
#chmod +x run_pipe.sh  

# 配置参数
INPUT_JSON="notebook_records.json"
LOG_DIR="execution_logs"
MAX_RUNS=100

# 检查依赖项
if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required. Install with: sudo apt-get install jq"
    exit 1
fi

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 主执行函数
run_pipeline() {
    local count=0
    
    # 读取并处理前10条记录
    jq -c ".[0:${MAX_RUNS}][]" "${INPUT_JSON}" | while read -r record; do
        ((count++))
        
        # 解析参数
        local notebook=$(jq -r '.notebook_path' <<< "${record}")
        local dataset=$(jq -r '.dataset_path' <<< "${record}")
        local index=$(jq -r '.index' <<< "${record}")
        
        # 生成带时间戳的日志文件
        local timestamp=$(date +"%Y%m%d_%H%M%S")
        local log_file="${LOG_DIR}/run_${count}_${timestamp}.log"
        
        # 打印执行信息
        echo " Processing ${count}/${MAX_RUNS}: ${notebook##*/}"
        echo "   Dataset: ${dataset##*/}"
        echo "   Label index: ${index}"
        
        # 执行Python程序并记录日志
        {
            echo "=== Execution started at $(date) ==="
            time python3 - <<END
import sys
# 添加Haipipe模块所在路径（根据实际情况修改）
#sys.path.append("/home/hitlt24/Haipipe")
from example_c import quick_start  # 显式导入函数

print(f"Debug - Notebook path: {r'${notebook}'}")
print(f"Debug - Dataset path: {r'${dataset}'}")
print(f"Debug - index: {r'${index}'}")

quick_start(
    r"${notebook}",  # 使用原始字符串防止转义
    r"${dataset}", 
    r"${index}",
    "LogisticRegression",
    hai_program_save_path="hai_program_${count}.py"
)
END
            echo "=== Execution finished at $(date) ==="
        } > "${log_file}" 2>&1
        
        # 检查执行结果
        if [ $? -eq 0 ]; then
            echo "Success: Log saved to ${log_file}"
        else
            echo "Failed: Check ${log_file} for details"
        fi
        
        echo "─────────────────────────────────────"
    done
}

# 启动主流程
if [ -f "${INPUT_JSON}" ]; then
    run_pipeline
else
    echo "ERROR: Input file ${INPUT_JSON} not found!"
    exit 1
fi