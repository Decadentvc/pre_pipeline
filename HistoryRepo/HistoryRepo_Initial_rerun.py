#!/usr/bin/env python3
import csv
import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime

# 导入你系统中的 ModelManager 和 EnhancedPipelineRunner
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prototype_evaluate_duration_limit import ModelManager, EnhancedPipelineRunner


INPUT_FILE = "evaluation_results.csv"
OUTPUT_FILE = "evaluation_rerun_results.csv"
DATASETS_DIR = "../datasets/dataset_csv_std_duplicate_removal"
TARGET_COLUMN = "label"

N_TRIALS = 10
CV = 3


# ----------------- 工具函数 -----------------
def is_invalid_accuracy(value):
    """判断 best_accuracy 是否无效"""
    if value is None or value == "":
        return True
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return True
        return False
    except Exception:
        return True


def load_param_sets():
    """
    因为原脚本中 param_sets 由 get_param_sets(model_type) 生成，
    此处为了复用，直接复制 param_sets 定义。
    """
    from HistoryRepo_Initial import get_param_sets  # 如果你另存，需要改路径
    return get_param_sets



# ----------------- 主逻辑：筛选异常条目 -----------------
def find_abnormal_rows():
    rows = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            best_acc = row["best_accuracy"]
            status = row["status"]
            error_msg = row["error_message"]

            abnormal = False

            if is_invalid_accuracy(best_acc):
                abnormal = True

            if status != "SUCCESS":
                abnormal = True

            if error_msg and error_msg.strip() != "":
                abnormal = True

            if abnormal:
                rows.append(row)

    print(f"找到异常条目 {len(rows)} 个。")
    return rows



# ----------------- 重跑单条任务 -----------------
def rerun_single_entry(row, param_sets):
    dataset_name = row["dataset_name"]
    ds_path = os.path.join(DATASETS_DIR, dataset_name)

    model_type = row["model_type"]
    param_set_id = int(row["param_set_id"])
    params = param_sets(model_type)[param_set_id - 1]

    # candidate 为一个字符串形式的 List，需要解析
    best_candidate = row["best_candidate"]
    if best_candidate:
        try:
            candidate = json.loads(best_candidate)
        except:
            candidate = None
    else:
        candidate = None

    if candidate is None:
        print(f"条目 index={row['index']} 缺少 best_candidate，跳过。")
        return None

    # 解除 timeout
    runner = EnhancedPipelineRunner(
        file_path=ds_path,
        target_column=TARGET_COLUMN,
        steps_order_candidates=[candidate],
        model_manager=ModelManager(),
        n_trials=N_TRIALS,
        cv=CV,
        trial_timeout=None  # 🚀 关键：解除超时限制
    )

    # 设置模型参数
    try:
        runner.model_manager.set_model_params(model_type, **params)
    except:
        runner.model_manager.set_model_params(model_type, params)

    # 运行优化
    print(f"  → 正在重跑 index={row['index']}  dataset={dataset_name}  model={model_type}")
    results = runner.optimize()

    # 提取结果
    from main_script import extract_accuracy_from_results
    acc = extract_accuracy_from_results(results, model_key=model_type)

    return acc, results



# ----------------- 主流程 -----------------
def main():
    abnormal_rows = find_abnormal_rows()
    param_sets = load_param_sets()

    header = None
    # 读取原文件头
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    # 输出文件
    out_f = open(OUTPUT_FILE, "w", newline="", encoding="utf-8")
    writer = csv.writer(out_f)
    writer.writerow(header)

    for row in abnormal_rows:
        acc, results = rerun_single_entry(row, param_sets)

        # 更新 row 中的信息
        row["best_accuracy"] = acc if acc is not None else ""
        row["status"] = "RERUN_SUCCESS" if acc is not None else "RERUN_FAILED"
        row["error_message"] = ""

        # 保存
        writer.writerow([row[h] for h in header])
        out_f.flush()

    out_f.close()
    print("\n重跑完毕，结果保存在:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
