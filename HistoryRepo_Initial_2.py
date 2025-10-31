#!/usr/bin/env python3
"""
full_grid_evaluate.py

遍历 datasets/dataset_csv_std_duplicate_removal 目录下的所有 CSV 数据集，
对 6 种模型（KNN, LR, RF, SVM, DT, GBDT）每种模型的 5 组参数、以及
预设的 steps_order_candidates 列表中的每一个候选顺序 完整遍历并评估。

每个 [模型, 参数组, 数据集] 组合：对 steps_order_candidates 中的每个候选顺序
各运行一次 EnhancedPipelineRunner.optimize()（在子进程输出被静默捕获的
情况下），提取准确率并记录排名。结果逐行写入 CSV（追加模式），并把 runner
的 stdout/stderr 日志写入 logs 目录下的单独文件。

用法: 在包含 prototype_evaluate (ModelManager, EnhancedPipelineRunner) 的环境中运行：
    python full_grid_evaluate.py

脚本顶部有一些可配置的常量：数据集目录、输出 CSV、候选步骤顺序、n_trials、cv 等。
"""

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

# 你原始示例里用到的两个类
from prototype_evaluate import ModelManager, EnhancedPipelineRunner

# ----------------- 可配置项（按需修改） -----------------
DATASETS_DIR = "datasets/dataset_csv_std_duplicate_removal"  # 包含所有 CSV 的目录
OUTPUT_CSV = "grid_evaluation_results.csv"
LOGS_DIR = "runner_logs"
TARGET_COLUMN = "label"  # 默认 target 列名
N_TRIALS = 3  # Optuna 迭代次数（可调整）
CV = 3        # 交叉验证折数（可调整）

# steps_order_candidates: 在这里定义你要比较的候选步骤顺序列表
STEPS_ORDER_CANDIDATES = [
    ['impute', 'encode', 'normalize', 'features', 'rebalance'],
    ['impute', 'normalize', 'encode', 'features', 'rebalance'],
    # 你可以在这里添加更多候选顺序
]
# ------------------------------------------------------

# ---------- 模型参数集合（每个模型 5 组参数） ----------
def get_param_sets(model_type: str):
    if model_type == "LR":
        return [
            {'penalty': 'l1', 'C': 0.01, 'solver': 'liblinear', 'multi_class': 'ovr'},
            {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'multi_class': 'multinomial'},
            {'penalty': 'elasticnet', 'C': 1.0, 'solver': 'saga', 'multi_class': 'ovr'},
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

# ---------- 提取准确率的工具（兼容多种返回结构） ----------
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

# ---------- 静默运行 runner.optimize()，捕获 stdout/stderr 并可写入日志 ----------

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
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n--- LOG at {datetime.utcnow().isoformat()} ---\n")
                if out:
                    lf.write("STDOUT:\n")
                    lf.write(out + "\n")
                if err:
                    lf.write("STDERR:\n")
                    lf.write(err + "\n")
        except Exception:
            pass

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

    first_run = not os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if first_run:
        header = [
            "index",
            "timestamp_utc",
            "dataset_id",
            "dataset_name",
            "model_type",
            "param_set_id",
            "param_json",
            "candidate_accuracies_json",
            "candidate_ranking_json",
            "best_accuracy",
            "best_candidate",
            "status",
            "error_message"
        ]
        csv_writer.writerow(header)
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

                    # 预先打印一次总体进度（单行）
                    elapsed = int(time.time() - start_time)
                    elapsed_s = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    terminal_width = shutil.get_terminal_size((120, 20)).columns
                    base_progress = (f"Progress {task_counter}/{total_tasks} | ds {ds_idx}/{n_datasets} ({ds_name}) "
                                     f"| model {model_type} | param {p_idx}/{n_param_sets} | elapsed {elapsed_s}")
                    print("\r" + base_progress.ljust(terminal_width), end="", flush=True)

                    candidate_accs = {}
                    status = "SUCCESS"
                    error_msg = ""
                    best_acc = None
                    best_candidate = None

                    try:
                        # 禁用/启用模型（若 ModelManager 支持）
                        try:
                            if hasattr(mm, "available_keys") and hasattr(mm, "disable"):
                                for k in mm.available_keys():
                                    try:
                                        mm.disable(k)
                                    except Exception:
                                        pass
                            if hasattr(mm, "enable"):
                                mm.enable(model_type)
                        except Exception:
                            pass

                        # 设置模型参数
                        try:
                            mm.set_model_params(model_type, **params)
                        except TypeError:
                            try:
                                mm.set_model_params(model_type, params)
                            except Exception:
                                raise

                        # 遍历每个 candidate，并静默运行
                        for cand_idx, cand in enumerate(STEPS_ORDER_CANDIDATES, start=1):
                            # 更新并显示当前 candidate 进度
                            cand_progress = (f"Progress {task_counter}/{total_tasks} | ds {ds_idx}/{n_datasets} ({ds_name}) "
                                             f"| model {model_type} | param {p_idx}/{n_param_sets} | candidate {cand_idx}/{n_candidates}")
                            print("\r" + cand_progress.ljust(terminal_width), end="", flush=True)

                            runner = EnhancedPipelineRunner(
                                file_path=ds_path,
                                target_column=TARGET_COLUMN,
                                steps_order_candidates=[cand],
                                model_manager=mm,
                                n_trials=N_TRIALS,
                                cv=CV
                            )

                            log_fname = f"{task_counter:05d}_ds{ds_idx}_model{model_type}_p{p_idx}_cand{cand_idx}.log"
                            log_path = os.path.join(LOGS_DIR, log_fname)

                            try:
                                results, captured_out, captured_err = run_runner_silent(runner, save_logs=True, log_path=log_path)
                                acc = extract_accuracy_from_results(results, model_type)
                                if acc is None and hasattr(runner, "last_result"):
                                    acc = extract_accuracy_from_results(runner.last_result, model_type)
                            except Exception as e:
                                status = "ERROR"
                                # 收集并截断日志，写入 error_msg
                                captured_out = getattr(e, 'captured_stdout', '')
                                captured_err = getattr(e, 'captured_stderr', '')
                                short_out = captured_out[:2000] if captured_out else ''
                                short_err = captured_err[:2000] if captured_err else ''
                                error_msg = f"{type(e).__name__}: {str(e)}\nCaptured STDOUT (truncated):\n{short_out}\nCaptured STDERR (truncated):\n{short_err}\n"
                                acc = None

                            candidate_accs[json.dumps(cand, ensure_ascii=False)] = float(acc) if acc is not None else None

                            # 更新单行进度显示（包含已完成 candidate 数量与当前 best）
                            non_null_vals = [v for v in candidate_accs.values() if v is not None]
                            best_so_far = max(non_null_vals) if non_null_vals else None
                            summary = f" done_cands={len(candidate_accs)}/{n_candidates}"
                            if best_so_far is not None:
                                summary += f" | best_sofar={best_so_far:.4f}"
                            print("\r" + (cand_progress + summary).ljust(terminal_width), end="", flush=True)

                        # 所有 candidate 完成 -> 计算排名
                        sortable = [(cand, a) for cand, a in candidate_accs.items() if a is not None]
                        # 排序：按 accuracy 降序
                        sortable.sort(key=lambda x: (-x[1], x[0]))
                        ranking = [cand for cand, _ in sortable]
                        if sortable:
                            best_candidate = ranking[0]
                            best_acc = sortable[0][1]
                        else:
                            best_candidate = None
                            best_acc = None

                    except KeyboardInterrupt:
                        status = "INTERRUPTED"
                        error_msg = "KeyboardInterrupt by user"
                        print("\nKeyboardInterrupt received. Exiting gracefully and flushing results.")
                        raise
                    except Exception as e:
                        status = "ERROR"
                        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                        best_candidate = None
                        best_acc = None

                    # 写 CSV 行
                    csv_writer.writerow([
                        task_counter,
                        datetime.utcnow().isoformat(),
                        ds_idx,
                        ds_name,
                        model_type,
                        p_idx,
                        json.dumps(params, ensure_ascii=False),
                        json.dumps(candidate_accs, ensure_ascii=False),
                        json.dumps(ranking, ensure_ascii=False),
                        best_acc if best_acc is not None else "",
                        best_candidate if best_candidate is not None else "",
                        status,
                        error_msg
                    ])
                    csv_file.flush()

        print("\nAll tasks completed. Results written to:", OUTPUT_CSV)

    except KeyboardInterrupt:
        print("\nRun interrupted by user. Partial results saved to:", OUTPUT_CSV)
    finally:
        csv_file.close()


if __name__ == "__main__":
    main()
