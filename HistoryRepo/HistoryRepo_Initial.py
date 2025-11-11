#!/usr/bin/env python3

import os
import glob
import json
import csv
import time
import traceback
import io
import sys
import shutil
import contextlib
from datetime import datetime
from pathlib import Path

# 确保可以导入上一级目录的 prototype_evaluate
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prototype_evaluate_duration_limit import ModelManager, EnhancedPipelineRunner

# ----------------- 可配置项 -----------------
DATASETS_DIR = "../datasets/dataset_csv_std_duplicate_removal"
OUTPUT_CSV = "grid_evaluation_results.csv"
LOGS_DIR = "runner_logs"
TARGET_COLUMN = "label"
N_TRIALS = 10
CV = 3
TRIAL_TIMEOUT = 2

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
# ------------------------------------------------------

# ---------- 模型参数集合 ----------
def get_param_sets(model_type: str):
    if model_type == "LR":
        return [
            {'penalty': 'l1', 'C': 0.01, 'solver': 'liblinear', 'multi_class': 'ovr'},
            {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'multi_class': 'multinomial'},
            {'penalty': 'elasticnet', 'C': 1.0, 'solver': 'saga', 'multi_class': 'ovr', 'l1_ratio': 0.5, 'max_iter': 500},
            {'penalty': None, 'C': 10.0, 'solver': 'newton-cg', 'multi_class': 'multinomial'},
            {'penalty': 'l2', 'C': 100.0, 'solver': 'sag', 'multi_class': 'ovr'}
        ]
    elif model_type == "RF":
        return [
            {'n_estimators': 50, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2'},
            {'n_estimators': 200, 'criterion': 'gini', 'max_depth': None, 'max_features': 0.3},
            {'n_estimators': 150, 'criterion': 'entropy', 'max_depth': 15, 'max_features': 0.5},
            {'n_estimators': 100, 'criterion': 'gini', 'max_depth': 20, 'max_features': None}
        ]
    elif model_type == "SVM":
        return [
            {'C': 0.1, 'kernel': 'linear', 'gamma': 'scale'},
            {'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto'},
            {'C': 10.0, 'kernel': 'poly', 'degree': 2, 'gamma': 0.1},
            {'C': 0.5, 'kernel': 'sigmoid', 'gamma': 'scale'},
            {'C': 100.0, 'kernel': 'poly', 'degree': 3, 'gamma': 'auto'}
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

# ---------- 提取准确率 ----------
def extract_accuracy_from_results(results, model_key=None):
    try:
        if isinstance(results, dict):
            if "per_model_best" in results and model_key in results["per_model_best"]:
                r = results["per_model_best"][model_key]
                if isinstance(r, dict) and 'accuracy' in r:
                    return float(r['accuracy'])
            if "optimization_results" in results and isinstance(results["optimization_results"], dict):
                if 'accuracy' in results["optimization_results"]:
                    return float(results["optimization_results"]['accuracy'])
            if "baseline_performance" in results and model_key in results["baseline_performance"]:
                bp = results["baseline_performance"][model_key]
                if isinstance(bp, dict) and 'mean_accuracy' in bp:
                    return float(bp['mean_accuracy'])
    except Exception:
        pass
    return None

# ---------- 静默运行 ----------
def run_runner_silent(runner, save_logs=True, log_path=None):
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

    if save_logs and log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n--- LOG at {datetime.utcnow().isoformat()} ---\n")
            if out:
                lf.write("STDOUT:\n" + out + "\n")
            if err:
                lf.write("STDERR:\n" + err + "\n")

    if exc:
        exc.captured_stdout = out
        exc.captured_stderr = err
        raise exc

    return results, out, err

# ---------- 主流程 ----------
def main():
    models = ["KNN", "LR", "RF", "SVM", "DT", "GBDT"]
    dataset_paths = sorted(glob.glob(os.path.join(DATASETS_DIR, "*.csv")))
    if not dataset_paths:
        print(f"ERROR: no datasets found under {DATASETS_DIR}. 请检查路径。")
        return

    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    n_datasets = len(dataset_paths)
    n_models = len(models)
    n_param_sets = 5
    n_candidates = len(STEPS_ORDER_CANDIDATES)
    total_tasks = n_datasets * n_models * n_param_sets

    completed = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["dataset_name"], row["model_type"], int(row["param_set_id"])))

    first_run = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
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
    task_counter = 0
    start_time = time.time()

    try:
        for ds_idx, ds_path in enumerate(dataset_paths, start=1):
            ds_name = os.path.basename(ds_path)
            for model_type in models:
                param_sets = get_param_sets(model_type)
                for p_idx, params in enumerate(param_sets, start=1):
                    task_counter += 1

                    if (ds_name, model_type, p_idx) in completed:
                        continue

                    candidate_accs = {}
                    ranking = []
                    best_candidate = None
                    best_acc = None
                    best_cand_id = None
                    status = "SUCCESS"
                    error_msg = ""

                    try:
                        for k in mm.available_keys():
                            mm.disable(k)
                        mm.enable(model_type)
                        mm.set_model_params(model_type, **params)

                        for cand_id, cand in STEPS_ORDER_CANDIDATES:
                            runner = EnhancedPipelineRunner(
                                file_path=ds_path, target_column=TARGET_COLUMN,
                                steps_order_candidates=[cand], model_manager=mm,
                                n_trials=N_TRIALS, cv=CV, trial_timeout=TRIAL_TIMEOUT
                            )
                            log_fname = f"{task_counter:05d}_ds{ds_idx}_model{model_type}_p{p_idx}_cand{cand_id}.log"
                            log_path = os.path.join(LOGS_DIR, log_fname)
                            try:
                                results, _, _ = run_runner_silent(runner, True, log_path)
                                acc = extract_accuracy_from_results(results, model_type)
                            except Exception:
                                acc = None
                            candidate_accs[str(cand_id)] = acc

                        valid = [(cid, a) for cid, a in candidate_accs.items() if a is not None]
                        valid.sort(key=lambda x: -x[1])
                        if valid:
                            best_cand_id, best_acc = valid[0]
                            best_candidate = [c for cid, c in STEPS_ORDER_CANDIDATES if str(cid) == str(best_cand_id)][0]

                        # 计算准确率排名
                        acc_dict = {}
                        for cid, acc in candidate_accs.items():
                            acc_dict.setdefault(acc, []).append(cid)
                        if None in acc_dict:
                            del acc_dict[None]
                        sorted_acc = sorted(acc_dict.items(), key=lambda x: -x[0])
                        ranking_parts = []
                        for acc, cids in sorted_acc:
                            if len(cids) == 1:
                                ranking_parts.append(str(cids[0]))
                            else:
                                ranking_parts.append("=".join(sorted(map(str, cids))))
                        ranking_str = ">".join(ranking_parts)

                    except Exception as e:
                        status = "ERROR"
                        error_msg = f"{type(e).__name__}: {str(e)}"

                    csv_writer.writerow([
                        task_counter, datetime.utcnow().isoformat(), ds_idx, ds_name,
                        model_type, p_idx, json.dumps(params, ensure_ascii=False),
                        json.dumps(candidate_accs, ensure_ascii=False),
                        json.dumps(ranking, ensure_ascii=False),
                        best_acc if best_acc is not None else "",
                        json.dumps(best_candidate, ensure_ascii=False) if best_candidate else "",
                        best_cand_id if best_cand_id is not None else "",
                        ranking_str, status, error_msg
                    ])
                    csv_file.flush()

        print("\nAll tasks completed. Results written to:", OUTPUT_CSV)

    finally:
        csv_file.close()


if __name__ == "__main__":
    main()
