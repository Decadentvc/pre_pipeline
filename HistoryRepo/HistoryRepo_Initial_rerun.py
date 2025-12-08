#!/usr/bin/env python3
import os
import csv
import json
import glob
import time
import traceback
from datetime import datetime
from pathlib import Path
import argparse
import psutil
import threading
import io
import contextlib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prototype_evaluate_duration_limit import ModelManager, EnhancedPipelineRunner


# ---------------- 配置默认值 -----------------
DEFAULT_DATASETS_DIR = "../datasets/dataset_csv_std_duplicate_removal"
RERUN_CSV = "rerun_results.csv"
LOGS_DIR = "rerun_logs"
TARGET_COLUMN = "label"
N_TRIALS = 10
CV = 3

# 每个候选 pipeline 顺序
STEPS_ORDER_CANDIDATES = [
    (1, ['impute', 'encode', 'normalize', 'features', 'discretize', 'rebalance']),
    (2, ['impute', 'encode', 'normalize', 'features', 'rebalance', 'discretize']),
    (3, ['impute', 'encode', 'normalize', 'rebalance', 'discretize', 'features']),
    (4, ['impute', 'encode', 'normalize', 'rebalance', 'features', 'discretize']),
    (5, ['impute', 'encode', 'discretize', 'features', 'normalize', 'rebalance']),
    (6, ['impute', 'encode', 'discretize', 'rebalance', 'features', 'normalize']),
    (7, ['impute', 'encode', 'features', 'normalize', 'rebalance', 'discretize']),
    (8, ['impute', 'encode', 'rebalance', 'discretize', 'features', 'normalize']),
]


# ---------------- 模型参数集合 -----------------
def get_param_sets(model_type: str):
    if model_type == "SVM":
        return [
            {'C': 0.1, 'kernel': 'linear', 'gamma': 'scale', 'max_iter': 500},
            {'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto', 'max_iter': 500},
            {'C': 10.0, 'kernel': 'poly', 'degree': 2, 'gamma': 0.1, 'max_iter': 500},
            {'C': 0.5, 'kernel': 'sigmoid', 'gamma': 'scale', 'max_iter': 500},
            {'C': 100.0, 'kernel': 'poly', 'degree': 3, 'gamma': 'auto', 'max_iter': 500}
        ]
    elif model_type == "LR":
        return [
            {'penalty': 'l1', 'C': 0.01, 'solver': 'liblinear', 'multi_class': 'ovr', 'max_iter': 300},
            {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'multi_class': 'multinomial', 'max_iter': 300},
            {'penalty': 'elasticnet', 'C': 1.0, 'solver': 'saga', 'multi_class': 'ovr', 'l1_ratio': 0.5, 'max_iter': 100},
            {'penalty': None, 'C': 10.0, 'solver': 'newton-cg', 'multi_class': 'multinomial', 'max_iter': 100},
            {'penalty': 'l2', 'C': 100.0, 'solver': 'sag', 'multi_class': 'ovr', 'max_iter': 100}
        ]
    elif model_type == "RF":
        return [
            {'n_estimators': 50, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2'},
            {'n_estimators': 200, 'criterion': 'gini', 'max_depth': None, 'max_features': 0.3},
            {'n_estimators': 150, 'criterion': 'entropy', 'max_depth': 15, 'max_features': 0.5},
            {'n_estimators': 100, 'criterion': 'gini', 'max_depth': 20, 'max_features': None}
        ]
    elif model_type == "DT":
        return [
            {'criterion': 'gini', 'max_depth': 3, 'max_features': 'sqrt'},
            {'criterion': 'entropy', 'max_depth': 7, 'max_features': 'log2'},
            {'criterion': 'gini', 'max_depth': None, 'max_features': 0.5},
            {'criterion': 'entropy', 'max_depth': 12, 'max_features': None},
            {'criterion': 'gini', 'max_depth': 5, 'max_features': 0.7}
        ]
    elif model_type == "GBDT":
        return [
            {'learning_rate': 0.01, 'n_estimators': 50, 'max_depth': 3, 'max_features': 'sqrt'},
            {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 5, 'max_features': 'log2'},
            {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 7, 'max_features': 0.3},
            {'learning_rate': 0.2, 'n_estimators': 150, 'max_depth': None, 'max_features': 0.5},
            {'learning_rate': 0.15, 'n_estimators': 100, 'max_depth': 4, 'max_features': None}
        ]
    elif model_type == "KNN":
        return [
            {'n_neighbors': 3, 'weights': 'uniform', 'algorithm': 'auto', 'p': 2, 'metric': 'minkowski'},
            {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'kd_tree', 'p': 2, 'metric': 'euclidean'},
            {'n_neighbors': 7, 'weights': 'uniform', 'algorithm': 'ball_tree', 'p': 1, 'metric': 'manhattan'},
            {'n_neighbors': 9, 'weights': 'distance', 'algorithm': 'brute', 'p': 2, 'metric': 'minkowski'},
            {'n_neighbors': 11, 'weights': 'uniform', 'algorithm': 'auto', 'p': 2, 'metric': 'chebyshev'}
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------- 提取异常条目 -----------------
def is_entry_abnormal(row):
    if row["status"] != "SUCCESS":
        return True
    if row.get("error_message"):
        return True
    return False


# ---------------- 静默运行 candidate -----------------
def run_runner_silent(runner, log_path=None):
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    results = None
    exc = None
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            results = runner.optimize()
    except Exception as e:
        exc = e
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()
    stdout_buf.close()
    stderr_buf.close()
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- LOG at {datetime.utcnow().isoformat()} ---\n")
            if out: f.write("STDOUT:\n" + out + "\n")
            if err: f.write("STDERR:\n" + err + "\n")
    if exc:
        exc.captured_stdout = out
        exc.captured_stderr = err
        raise exc
    return results


# ---------------- 安全运行 candidate（无超时） -----------------
def run_candidate(runner, model_type, log_path):
    try:
        results = run_runner_silent(runner, log_path)
        acc = None
        if isinstance(results, dict):
            # 尝试从各种位置提取 accuracy
            if "per_model_best" in results and model_type in results["per_model_best"]:
                r = results["per_model_best"][model_type]
                if isinstance(r, dict) and "accuracy" in r:
                    acc = float(r["accuracy"])
            elif "optimization_results" in results and isinstance(results["optimization_results"], dict):
                if "accuracy" in results["optimization_results"]:
                    acc = float(results["optimization_results"]["accuracy"])
            elif "baseline_performance" in results and model_type in results["baseline_performance"]:
                bp = results["baseline_performance"][model_type]
                if isinstance(bp, dict) and "mean_accuracy" in bp:
                    acc = float(bp["mean_accuracy"])
        return acc, "SUCCESS", results
    except Exception as e:
        return None, f"{type(e).__name__}", None


# ---------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--rerun_csv", type=str, default=RERUN_CSV)
    parser.add_argument("--logs_dir", type=str, default=LOGS_DIR)
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--cv", type=int, default=CV)
    parser.add_argument("--target_column", type=str, default=TARGET_COLUMN)
    parser.add_argument("--max_memory_mb", type=int, default=4096)
    parser.add_argument("--max_cpu_seconds", type=int, default=None)
    args = parser.parse_args()

    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    # 读取异常条目
    abnormal_entries = []
    orig_csv = "evaluation_results.csv"
    if not os.path.exists(orig_csv):
        print(f"ERROR: {orig_csv} not found")
        return

    with open(orig_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_entry_abnormal(row):
                abnormal_entries.append(row)

    if not abnormal_entries:
        print("No abnormal entries found. Nothing to rerun.")
        return

    # 已完成的重跑条目
    completed = set()
    if os.path.exists(args.rerun_csv):
        with open(args.rerun_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(int(row["index"]))

    first_run = not os.path.exists(args.rerun_csv) or os.path.getsize(args.rerun_csv) == 0
    csv_file = open(args.rerun_csv, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if first_run:
        csv_writer.writerow([
            "index","timestamp_utc","dataset_id","dataset_name",
            "model_type","param_set_id","param_json",
            "candidate_accuracies_json","candidate_ranking_json",
            "best_accuracy","best_candidate","best_candidate_id","accuracy_ranking",
            "status","error_message"
        ])
        csv_file.flush()

    mm = ModelManager()

    for entry in abnormal_entries:
        idx = int(entry["index"])
        if idx in completed:
            continue
        
        print(f"Rerunning entry index {idx} ...")
        
        ds_name = entry["dataset_name"]
        dataset_id = entry["dataset_id"]
        model_type = entry["model_type"]
        param_set_id = int(entry["param_set_id"])
        params = get_param_sets(model_type)[param_set_id-1]

        candidate_accs = {}
        ranking = []
        best_candidate = None
        best_acc = None
        best_cand_id = None
        status = "SUCCESS"
        error_msg = ""

        try:
            # 禁用其他模型，只启用当前
            if hasattr(mm, "available_keys") and hasattr(mm, "disable"):
                for k in mm.available_keys():
                    mm.disable(k)
            if hasattr(mm, "enable"):
                mm.enable(model_type)
            try:
                mm.set_model_params(model_type, **params)
            except TypeError:
                mm.set_model_params(model_type, params)

            for cand_id, steps in STEPS_ORDER_CANDIDATES:
                ds_path_full = os.path.join(args.datasets_dir, ds_name) if args.datasets_dir else os.path.join(".", ds_name)
                runner = EnhancedPipelineRunner(
                    file_path=ds_path_full, target_column=args.target_column,
                    steps_order_candidates=[steps], model_manager=mm,
                    n_trials=args.n_trials, cv=args.cv, trial_timeout=None
                )
                log_fname = f"rerun_idx{idx}_model{model_type}_p{param_set_id}_cand{cand_id}.log"
                log_path = os.path.join(args.logs_dir, log_fname)
                acc, stat, _ = run_candidate(runner, model_type, log_path)
                candidate_accs[str(cand_id)] = acc
                if stat != "SUCCESS":
                    status = stat

            # 选最优
            valid = [(cid, a) for cid, a in candidate_accs.items() if a is not None]
            valid.sort(key=lambda x: -x[1])
            if valid:
                best_cand_id, best_acc = valid[0]
                for cid, steps in STEPS_ORDER_CANDIDATES:
                    if str(cid) == str(best_cand_id):
                        best_candidate = steps
                        break

            # 生成排名
            acc_dict = {}
            for cid, acc in candidate_accs.items():
                if acc is not None:
                    acc_dict.setdefault(acc, []).append(cid)
            sorted_acc = sorted(acc_dict.items(), key=lambda x: -x[0])
            ranking_parts = []
            for acc, cids in sorted_acc:
                if len(cids) == 1:
                    ranking_parts.append(str(cids[0]))
                else:
                    ranking_parts.append("=".join(sorted(map(str, cids))))
            ranking_str = ">".join(ranking_parts)
            ranking = ranking_parts

        except Exception as e:
            status = "ERROR"
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        csv_writer.writerow([
            idx, datetime.utcnow().isoformat(), dataset_id, ds_name,
            model_type, param_set_id, json.dumps(params, ensure_ascii=False),
            json.dumps(candidate_accs, ensure_ascii=False),
            json.dumps(ranking, ensure_ascii=False),
            best_acc if best_acc is not None else "",
            json.dumps(best_candidate, ensure_ascii=False) if best_candidate else "",
            best_cand_id if best_cand_id is not None else "",
            ranking_str, status, error_msg
        ])
        csv_file.flush()
        print(f"Rerun idx {idx} completed: status={status}")

    csv_file.close()
    print("All rerun tasks completed.")


if __name__ == "__main__":
    main()
